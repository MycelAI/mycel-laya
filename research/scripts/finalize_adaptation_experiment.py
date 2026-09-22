"""Freeze a candidate selection, then evaluate and publish that exact policy once."""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from laya.adapt_bundle import (
    _verify_evidence_runtime,
    export_bundle,
    load_bundle,
)
from laya.adapt_calibration import fit_calibration
from laya.adapt_data import fingerprint, read_dataset
from laya.adapt_eval import collect_partition, verify_prediction_dataset
from laya.adapt_model import checkpoint_files, file_sha256
from laya.adapt_policy import evaluate_bound_policy, select_bound_policy
from laya.adapt_train import _atomic_save, _run_lock
from research.scripts.evaluate_adaptation_candidate import (
    validate_candidate,
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selection_files(directory):
    files = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("frozen selection must contain local files")
        if path.is_file() and path != Path(directory) / "selection.json":
            files[path.relative_to(directory).as_posix()] = file_sha256(path)
    return files


def freeze_selection(protocol, candidate_dirs, dataset_dir, destination):
    """Select among a completed prefix of the predeclared candidate sequence.

    This phase never reads final-test predictions or loads a model. The resulting
    receipt is required by the separate final-evaluation phase.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("selection is already frozen; use a new experiment, not an overwritten decision")
    questions, _, manifest = read_dataset(dataset_dir)
    if fingerprint(manifest) != protocol["dataset_manifest_sha256"]:
        raise ValueError("dataset differs from the predeclared protocol")
    qid = protocol["question_id"]
    if (set(questions) != {qid} or questions[qid]["type"] != "choice"
            or len(questions[qid]["criteria"]) != protocol["n_options"]):
        raise ValueError("candidate reports require the protocol's fixed choice question")
    order = [candidate["id"] for candidate in protocol["candidates"]]
    if len(set(order)) != len(order):
        raise ValueError("protocol candidate IDs must be unique")
    settings, records = protocol["policy"], {}
    if (settings["max_model_candidates"] < len(order)
            or not math.isclose(settings["alpha_per_candidate"] * settings["max_model_candidates"],
                                settings["family_alpha"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("candidate comparison exceeds the declared family error budget")
    for directory in candidate_dirs:
        directory = Path(directory)
        report, calibration = _read(directory / "report.json"), _read(directory / "calibration.json")
        candidate = report.get("candidate")
        if (candidate not in order or candidate in records or report.get("stage") != "candidate_policy_evaluation"
                or report.get("protocol_sha256") != fingerprint(protocol)
                or report.get("question_id") != qid
                or report.get("final_test_evaluated") is not False):
            raise ValueError("candidate report does not belong to this untouched-test experiment")
        predictions = {role: _read(directory / role / "predictions.json") for role in ("calibration", "policy")}
        for role, artifact in predictions.items():
            if (artifact.get("split") != role or report["prediction_files"][role] != fingerprint(artifact)
                    or report["model_files"] != artifact["binding"]["model_files"]
                    or report.get("binding") != artifact["binding"]):
                raise ValueError("candidate report does not match its prediction evidence")
            if artifact["binding"]["rendering"] != protocol["rendering"]:
                raise ValueError("candidate evidence uses different rendering settings from the declared protocol")
            questions = verify_prediction_dataset(artifact, dataset_dir)
        fitted = fit_calibration(predictions["calibration"], questions)
        if fingerprint(fitted) != fingerprint(calibration) or report["calibration_sha256"] != fingerprint(calibration):
            raise ValueError("candidate calibration differs from its recorded fit")
        policy = select_bound_policy(predictions["policy"], calibration, dataset_dir,
                                     question_id=protocol["question_id"], required_languages=protocol["required_languages"],
                                     thresholds=settings["thresholds"], max_error=settings["max_error"],
                                     min_coverage=settings["min_coverage"], alpha=settings["alpha_per_candidate"],
                                     declaration={"protocol_sha256": fingerprint(protocol), "candidate": candidate})
        if fingerprint(policy["selection"]) != fingerprint(report["selection"]):
            raise ValueError("candidate selection does not reproduce under the declared protocol")
        records[candidate] = {"report": report, "calibration": calibration, "policy": policy, "predictions": predictions}
    if not records or set(records) != set(order[:len(records)]):
        raise ValueError("evaluate a completed prefix of the candidate order; do not omit earlier results")
    eligible = [candidate for candidate in order if candidate in records and records[candidate]["policy"]["selection"]["passed"]]
    if not eligible:
        raise ValueError("no candidate meets the declared gate; final-test selection is prohibited")

    def rank(candidate):
        slices = records[candidate]["policy"]["selection"]["by_language"].values()
        minimum = min(item["selected"]["accepted"] / item["selected"]["samples"] for item in slices)
        return minimum, -order.index(candidate)

    selected = max(eligible, key=rank)
    chosen = records[selected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".laya-selection-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "selection"
        stage.mkdir()
        payloads = {"protocol.json": protocol, "policy.json": chosen["policy"], "calibration.json": chosen["calibration"],
                    "calibration-predictions.json": chosen["predictions"]["calibration"],
                    "policy-predictions.json": chosen["predictions"]["policy"],
                    "candidate-reports.json": {candidate: records[candidate]["report"] for candidate in order if candidate in records}}
        for name, value in payloads.items():
            _atomic_save(value, stage / name)
        selection = {"schema_version": 1, "stage": "frozen_candidate_selection", "candidate": selected,
                     "protocol_sha256": fingerprint(protocol), "dataset_manifest_sha256": fingerprint(manifest),
                     "policy_sha256": fingerprint(chosen["policy"]), "files": _selection_files(stage),
                     "ranking": {candidate: list(rank(candidate)) for candidate in eligible},
                     "final_test_evaluated": False}
        _atomic_save(selection, stage / "selection.json")
        checksum = file_sha256(stage / "selection.json")
        stage.rename(destination)
    return {"directory": str(destination.resolve()), "selection_sha256": checksum, "candidate": selected}


def _load_selection(directory, checksum):
    directory = Path(directory)
    path = directory / "selection.json"
    if path.is_symlink() or file_sha256(path) != checksum:
        raise ValueError("frozen selection differs from the trusted receipt")
    selection = _read(path)
    if (type(selection.get("schema_version")) is not int or selection["schema_version"] != 1
            or selection.get("stage") != "frozen_candidate_selection" or selection.get("final_test_evaluated") is not False
            or _selection_files(directory) != selection.get("files")):
        raise ValueError("frozen selection artifacts are missing or changed")
    payload = {name: _read(directory / filename) for name, filename in (
        ("protocol", "protocol.json"), ("policy", "policy.json"), ("calibration", "calibration.json"),
        ("calibration_predictions", "calibration-predictions.json"), ("policy_predictions", "policy-predictions.json"))}
    if (selection["protocol_sha256"] != fingerprint(payload["protocol"])
            or selection["dataset_manifest_sha256"] != payload["protocol"]["dataset_manifest_sha256"]
            or selection["policy_sha256"] != fingerprint(payload["policy"])
            or payload["policy"].get("declaration") != {
                "protocol_sha256": selection["protocol_sha256"], "candidate": selection["candidate"]}):
        raise ValueError("frozen selection has inconsistent protocol or policy links")
    return selection, payload


def finalize(selection_dir, expected_selection_sha256, dataset_dir, model_dir, run_dir, bundle_dir, *, resume=False, log=print):
    """Run only the frozen candidate, resumably, and preserve a failed final gate."""
    selection, payload = _load_selection(selection_dir, expected_selection_sha256)
    protocol, policy, calibration = payload["protocol"], payload["policy"], payload["calibration"]
    validate_candidate(protocol, selection["candidate"], dataset_dir, model_dir)
    if checkpoint_files(model_dir) != policy["binding"]["model_files"]:
        raise ValueError("final checkpoint differs from the selected policy")
    for role in ("calibration_predictions", "policy_predictions"):
        _verify_evidence_runtime(payload[role])
    run_dir, bundle_dir = Path(run_dir), Path(bundle_dir)
    identity = {"schema_version": 1, "selection_sha256": expected_selection_sha256,
                "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
                "model_files": checkpoint_files(model_dir), "bundle_directory": str(bundle_dir.resolve())}
    if run_dir.exists():
        if not resume:
            raise FileExistsError("final run exists; resume the same frozen decision explicitly")
        if _read(run_dir / "run.json") != identity:
            raise ValueError("cannot replace a selected model or policy after final evaluation has begun")
    else:
        if resume:
            raise FileNotFoundError("no final run to resume")
        if bundle_dir.exists():
            raise FileExistsError("bundle destination exists before final evaluation")
        run_dir.mkdir(parents=True)
        _atomic_save(identity, run_dir / "run.json")
    with _run_lock(run_dir):
        if (run_dir / "result.json").is_file():
            result = _read(run_dir / "result.json")
            evaluation = _read(run_dir / "final-report.json")
            receipt = _read(run_dir / "bundle-receipt.json")
            expected_status = "qualified" if evaluation["evaluation"]["passed"] else "failed_gate"
            expected_mode = "qualified" if expected_status == "qualified" else "review_only"
            if (result.get("run_sha256") != fingerprint(identity)
                    or result.get("final_report_sha256") != fingerprint(evaluation)
                    or result.get("status") != expected_status or result.get("candidate") != selection["candidate"]
                    or result.get("bundle") != receipt or receipt.get("mode") != expected_mode
                    or receipt.get("directory") != str(bundle_dir.resolve())
                    or fingerprint(_read(bundle_dir / "evaluation.json")) != fingerprint(evaluation)):
                raise ValueError("completed final-run receipt or report changed")
            load_bundle(bundle_dir, expected_manifest_sha256=receipt["manifest_sha256"])
            return result
        test_dir = run_dir / "test"
        predictions = collect_partition(dataset_dir, model_dir, test_dir, split="test", **protocol["rendering"],
                                        resume=resume and test_dir.exists(), log=log)
        evaluation = evaluate_bound_policy(predictions, policy, calibration, dataset_dir)
        _atomic_save(evaluation, run_dir / "final-report.json")
        passed = evaluation["evaluation"]["passed"]
        receipt_path = run_dir / "bundle-receipt.json"
        if bundle_dir.exists():
            if not receipt_path.is_file():
                raise ValueError("existing bundle has no recorded publication receipt; do not adopt it silently")
            receipt = _read(receipt_path)
            load_bundle(bundle_dir, expected_manifest_sha256=receipt["manifest_sha256"])
            if (fingerprint(_read(bundle_dir / "evaluation.json")) != fingerprint(evaluation)
                    or receipt.get("directory") != str(bundle_dir.resolve())
                    or receipt.get("mode") != ("qualified" if passed else "review_only")):
                raise ValueError("existing bundle belongs to another final evaluation")
        else:
            receipt = export_bundle(bundle_dir, dataset_dir=dataset_dir, model_dir=model_dir,
                                     calibration_predictions=payload["calibration_predictions"],
                                     policy_predictions=payload["policy_predictions"], calibration=calibration,
                                     policy=policy, test_predictions=predictions, review_only=not passed,
                                     receipt_path=receipt_path)
        result = {"schema_version": 1, "status": "qualified" if passed else "failed_gate",
                  "candidate": selection["candidate"], "run_sha256": fingerprint(identity),
                  "final_report_sha256": fingerprint(evaluation), "bundle": receipt}
        _atomic_save(result, run_dir / "result.json")
        log(json.dumps(result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    for name in ("protocol", "dataset", "output"):
        select.add_argument("--" + name, required=True)
    select.add_argument("--candidate-report", action="append", required=True,
                        help="completed candidate output directory; repeat in protocol order")
    final = commands.add_parser("finalize")
    for name in ("selection", "expected_selection_sha256", "dataset", "model", "run", "bundle"):
        final.add_argument("--" + name.replace("_", "-"), required=True)
    final.add_argument("--resume", action="store_true")
    final.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(freeze_selection(_read(args.protocol), args.candidate_report, args.dataset, args.output), indent=2))
    else:
        if args.threads is not None:
            if args.threads < 1:
                parser.error("--threads must be positive")
            torch.set_num_threads(args.threads)
        finalize(args.selection, args.expected_selection_sha256, args.dataset, args.model, args.run, args.bundle,
                 resume=args.resume)


if __name__ == "__main__":
    main()
