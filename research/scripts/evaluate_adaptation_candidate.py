"""Evaluate one predeclared adaptation candidate without opening the final test."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from laya.adapt_calibration import fit_calibration, metric_report, probability_records  # noqa: E402
from laya.adapt_data import fingerprint, read_dataset, target_index  # noqa: E402
from laya.adapt_eval import collect_partition, verify_prediction_dataset  # noqa: E402
from laya.adapt_model import checkpoint_files  # noqa: E402
from laya.adapt_train import TrainingConfig, _atomic_save  # noqa: E402
from laya.risk import select_risk_policy  # noqa: E402


def validate_candidate(protocol, candidate_id, dataset_dir, model_dir):
    questions, rows, manifest = read_dataset(dataset_dir)
    if fingerprint(manifest) != protocol["dataset_manifest_sha256"]:
        raise ValueError("dataset differs from the predeclared experiment")
    qid = protocol["question_id"]
    if (set(questions) != {qid} or questions[qid]["type"] != "choice"
            or len(questions[qid]["criteria"]) != protocol["n_options"]):
        raise ValueError("candidate evaluation requires the protocol's fixed choice question")
    candidates = [candidate for candidate in protocol["candidates"] if candidate["id"] == candidate_id]
    if len(candidates) != 1:
        raise ValueError("choose exactly one predeclared candidate")
    candidate, files = candidates[0], checkpoint_files(model_dir)
    if candidate["epochs"] == 0:
        if files != protocol["base_model_files"]:
            raise ValueError("baseline checkpoint differs from the pinned model files")
    else:
        run_dir = Path(model_dir).parent
        identity = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        receipt = json.loads((Path(model_dir) / "adaptation_manifest.json").read_text(encoding="utf-8"))
        expected = {**asdict(TrainingConfig()), **protocol["training_common"], **protocol["rendering"],
                    "epochs": candidate["epochs"], "train_encoder": candidate["train_encoder"]}
        if (identity["settings"] != expected or identity["source_files"] != protocol["base_model_files"]
                or identity["dataset_manifest_sha256"] != fingerprint(manifest)
                or result["status"] != "complete" or result["run_sha256"] != fingerprint(identity)
                or receipt["provenance"] != result or receipt["files"] != files):
            raise ValueError("adapted checkpoint does not match its declared training candidate")
    settings = protocol["policy"]
    if (settings["max_model_candidates"] < len(protocol["candidates"])
            or not math.isclose(settings["alpha_per_candidate"] * settings["max_model_candidates"],
                                settings["family_alpha"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("protocol does not allocate its error probability over all model candidates")
    if (protocol["calibration"]["partition"] != "calibration" or settings["partition"] != "policy"
            or protocol["calibration"]["objective"] != "hard_label_nll"
            or protocol["calibration"]["buckets"] is not False
            or protocol["calibration"]["temperature_range"] != [.5, 5.0]
            or settings["score"] != "max_calibrated_probability"):
        raise ValueError("unsupported calibration or policy protocol")
    return questions, rows, manifest


def evaluate(protocol, candidate_id, dataset_dir, model_dir, output_dir, *, resume=False, log=print):
    """Freeze candidate identity, fit on calibration, and select on policy only."""
    questions, rows, manifest = validate_candidate(protocol, candidate_id, dataset_dir, model_dir)
    output_dir = Path(output_dir)
    identity = {"schema_version": 1, "candidate": candidate_id, "protocol_sha256": fingerprint(protocol),
                "model_files": checkpoint_files(model_dir)}
    if output_dir.exists():
        if not resume:
            raise FileExistsError("candidate output exists; resume explicitly")
        recorded = json.loads((output_dir / "candidate.json").read_text(encoding="utf-8"))
        if recorded != identity:
            raise ValueError("candidate evaluation differs from its frozen identity")
    else:
        if resume:
            raise FileNotFoundError("candidate evaluation does not exist")
        output_dir.mkdir(parents=True)
        _atomic_save(identity, output_dir / "candidate.json")
    predictions = {}
    for role in ("calibration", "policy"):
        folder = output_dir / role
        predictions[role] = collect_partition(dataset_dir, model_dir, folder, split=role,
                                              **protocol["rendering"], resume=resume and folder.exists(), log=log)
        verify_prediction_dataset(predictions[role], dataset_dir)
        if role == "calibration":
            calibration = fit_calibration(predictions[role], questions)
            _atomic_save(calibration, output_dir / "calibration.json")
    qid = protocol["question_id"]
    settings = protocol["policy"]
    samples = probability_records(predictions["policy"], questions, calibration, question_id=qid)
    selection = select_risk_policy(samples, required_languages=protocol["required_languages"],
                                   thresholds=settings["thresholds"], max_error=settings["max_error"],
                                   min_coverage=settings["min_coverage"], alpha=settings["alpha_per_candidate"])
    counts = [1] * protocol["n_options"]
    for row in rows:
        if manifest["assignments"][row["id"]] == "train":
            counts[target_index(questions[qid], row["targets"][qid])] += 1
    baseline = copy.deepcopy(predictions["policy"])
    baseline["binding"]["model_files"] = {"training_class_counts": fingerprint(counts)}
    for row in baseline["records"]:
        row["logits"] = [math.log(count) for count in counts]
    baseline["records_sha256"] = fingerprint(baseline["records"])
    result = {"schema_version": 1, "stage": "candidate_policy_evaluation", **identity,
              "binding": predictions["policy"]["binding"], "question_id": qid,
              "calibration_sha256": fingerprint(calibration), "selection": selection,
              "metrics": {"raw": metric_report(predictions["policy"], questions),
                          "calibrated": metric_report(predictions["policy"], questions, calibration),
                          "training_prior": metric_report(baseline, questions)},
              "prediction_files": {role: fingerprint(value) for role, value in predictions.items()},
              "final_test_evaluated": False}
    _atomic_save(result, output_dir / "report.json")
    log(json.dumps({"candidate": candidate_id, "selection_passed": selection["passed"],
                    "report": str(output_dir / "report.json")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "candidate", "dataset", "model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    evaluate(protocol, args.candidate, args.dataset, args.model, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
