"""Verified local deployment bundles with explicit automate/review outcomes."""
import argparse
import copy
import hashlib
import json
import math
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tokenizers
import torch
import transformers

from .adapt_calibration import (
    calibrated_probabilities,
    fit_calibration,
    validate_calibration,
)
from .adapt_data import _json_value, fingerprint, read_dataset, validate_questions
from .adapt_eval import verify_prediction_dataset
from .adapt_model import (
    checkpoint_files,
    file_sha256,
    forward_batch,
    load_local_checkpoint,
    prepare_input,
)
from .adapt_policy import (
    evaluate_bound_policy,
    select_bound_policy,
    validate_bound_policy,
)
from .adapt_train import _atomic_save
from .agent import _fix_tokenizer_config
from .risk import binomial_upper_bound

_MODEL_CODE = ("adapt_model.py", "common.py", "agent.py")
_BUNDLE_CODE = (*_MODEL_CODE, "adapt_data.py", "adapt_calibration.py", "adapt_policy.py", "risk.py", "adapt_bundle.py")


def _runtime_contract():
    # Git's CRLF checkout conversion must not prevent Windows/Linux portability.
    code = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in _BUNDLE_CODE}
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(), "implementation": code}


def _verify_evidence_runtime(predictions):
    collection = predictions.get("collection", {})
    runtime, contract = collection.get("runtime", {}), _runtime_contract()
    if any(runtime.get(name) != contract[name] for name in
           ("python", "torch", "numpy", "transformers", "tokenizers", "device", "threads")):
        raise ValueError("prediction evidence needs the recorded CPU runtime used by this exporter")
    for name in _MODEL_CODE:
        contents = Path(__file__).with_name(name).read_bytes()
        lf = contents.replace(b"\r\n", b"\n")
        equivalents = {hashlib.sha256(value).hexdigest() for value in (contents, lf, lf.replace(b"\n", b"\r\n"))}
        if collection.get("implementation", {}).get(name) not in equivalents:
            raise ValueError("prediction renderer/model implementation differs from this exporter")


def _inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("deployment bundles must contain local files, not symbolic links")
        if path.is_file() and path != Path(directory) / "bundle.json":
            result[path.relative_to(directory).as_posix()] = file_sha256(path)
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _calibrated_config(source, calibration):
    config = copy.deepcopy(source)
    config.update(calibration["binding"]["rendering"])
    config.update(temperature=copy.deepcopy(calibration["temperature"]),
                  temperature_by_options=copy.deepcopy(calibration["temperature_by_options"]))
    config["calibration"] = {"artifact_sha256": fingerprint(calibration), "source_split": "calibration"}
    if isinstance(config.get("adaptation"), dict):
        config["adaptation"]["calibration_status"] = "fitted"
    return config


def _check_summary(summary, protocol, tail):
    samples, accepted, errors = (summary.get(name) for name in ("samples", "accepted", "errors"))
    if (any(type(value) is not int for value in (samples, accepted, errors))
            or not 0 <= errors <= accepted <= samples or accepted == 0):
        raise ValueError("qualified report has invalid evidence counts")
    upper = binomial_upper_bound(errors, accepted, tail)
    lower = 1 - binomial_upper_bound(samples - accepted, samples, tail)
    if (summary.get("qualified") is not True or upper > protocol["max_error"] or lower < protocol["min_coverage"]
            or not math.isclose(summary.get("error_upper", -1), upper, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(summary.get("coverage_lower", -1), lower, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("qualified report does not satisfy its uncertainty bounds")


def _check_components(questions, calibration, policy, evaluation, mode):
    validate_questions(questions)
    validate_calibration(calibration)
    validate_bound_policy(policy, calibration, questions, require_passed=mode == "qualified")
    if mode not in ("qualified", "review_only"):
        raise ValueError("unsupported deployment mode")
    if evaluation is not None:
        if (not isinstance(evaluation, dict) or type(evaluation.get("schema_version")) is not int
                or evaluation["schema_version"] != 1 or evaluation.get("stage") != "bound_final_test"
                or evaluation.get("policy_sha256") != fingerprint(policy)
                or evaluation.get("calibration_sha256") != fingerprint(calibration)
                or fingerprint(evaluation.get("binding")) != fingerprint(policy["binding"])):
            raise ValueError("final evaluation is not bound to this policy and calibration")
        report = evaluation.get("evaluation", {})
        if (report.get("selection_sha256") != fingerprint(policy["selection"])
                or report.get("stage") != "final_test"
                or fingerprint(report.get("protocol")) != fingerprint(policy["selection"].get("protocol"))):
            raise ValueError("final evaluation does not match the selected decision rule")
    if mode == "qualified" and (evaluation is None or evaluation.get("evaluation", {}).get("passed") is not True):
        raise ValueError("automatic deployment requires a passing independent final evaluation")
    if mode == "qualified":
        selection, report = policy["selection"], evaluation["evaluation"]
        protocol = selection["protocol"]
        languages = protocol["required_languages"]
        if (not languages or set(selection["by_language"]) != set(languages)
                or set(report["by_language"]) != set(languages) or not selection["threshold_grid"]
                or report.get("n_options") != selection["n_options"]):
            raise ValueError("qualified policy is missing declared language or option evidence")
        for language in languages:
            selected, tested = selection["by_language"][language]["selected"], report["by_language"][language]
            threshold = selected.get("threshold")
            if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold) or not 0 <= threshold <= 1
                    or threshold not in selection["threshold_grid"] or tested.get("threshold") != threshold):
                raise ValueError("final evaluation threshold differs from the selected policy")
            _check_summary(selected, protocol, protocol["alpha"] / (2 * len(languages) * len(selection["threshold_grid"])))
            _check_summary(tested, protocol, protocol["alpha"] / (2 * len(languages)))


def export_bundle(destination, *, dataset_dir, model_dir, calibration_predictions, policy_predictions,
                  calibration, policy, test_predictions=None, review_only=False, receipt_path=None):
    """Reverify evidence and publish a new calibrated SDK model plus policy bundle.

    The calibration fit and selection are recomputed from their frozen prediction
    artifacts; final bounds are verified from the already locked test predictions.
    This is deterministic verification, not a new trial or a model-selection step.
    Review-only publication does not authorize automation and needs no test access.
    """
    if type(review_only) is not bool:
        raise ValueError("review_only must be boolean")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("bundle destination already exists")
    if receipt_path is not None and Path(receipt_path).resolve().is_relative_to(destination.resolve()):
        raise ValueError("keep the trusted publication receipt outside the bundle")
    questions, _, _ = read_dataset(dataset_dir)
    verify_prediction_dataset(calibration_predictions, dataset_dir)
    verify_prediction_dataset(policy_predictions, dataset_dir)
    for evidence in (calibration_predictions, policy_predictions):
        _verify_evidence_runtime(evidence)
    fit = calibration.get("fit", {})
    fitted = fit_calibration(calibration_predictions, questions, fit_buckets=fit.get("fit_buckets"),
                             min_per_type=fit.get("min_per_type"), min_per_bucket=fit.get("min_per_bucket"))
    if fingerprint(fitted) != fingerprint(calibration):
        raise ValueError("calibration does not reproduce from its frozen evidence")
    validate_bound_policy(policy, calibration, questions)
    selected = policy["selection"]
    settings = selected["protocol"]
    reproduced = select_bound_policy(policy_predictions, calibration, dataset_dir,
                                      question_id=policy["question_id"], thresholds=selected["threshold_grid"],
                                      declaration=policy.get("declaration"), **settings)
    if fingerprint(reproduced) != fingerprint(policy):
        raise ValueError("policy does not reproduce from its frozen selection evidence")
    files = checkpoint_files(model_dir)
    if files != calibration["binding"]["model_files"]:
        raise ValueError("checkpoint files differ from the calibrated model")
    evaluation = None
    if test_predictions is not None:
        _verify_evidence_runtime(test_predictions)
        evaluation = evaluate_bound_policy(test_predictions, policy, calibration, dataset_dir)
    mode = "review_only" if review_only else "qualified"
    _check_components(questions, calibration, policy, evaluation, mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".laya-bundle-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        model = stage / "model"
        for name in files:
            output = model / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(model_dir) / name, output)
        if checkpoint_files(model) != files:
            raise ValueError("source model changed while copying deployment assets")
        for name in ("rl_agent_config.json", "tokenizer/tokenizer_config.json"):
            source = stage / "source-config" / name
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model / name, source)
        _atomic_save(_calibrated_config(_read_json(model / "rl_agent_config.json"), calibration),
                      model / "rl_agent_config.json")
        _fix_tokenizer_config(str(model))
        for name, value in (("questions.json", questions), ("calibration.json", calibration),
                            ("policy.json", policy), ("evaluation.json", evaluation),
                            ("runtime.json", _runtime_contract())):
            _atomic_save(value, stage / name)
        provenance = Path(model_dir) / "adaptation_manifest.json"
        if provenance.is_file():
            training = _read_json(provenance)
            if training.get("files") != files:
                raise ValueError("source adaptation receipt does not match the model files")
            _atomic_save(training, stage / "source_adaptation.json")
        manifest = {"schema_version": 1, "stage": "selective_bundle", "mode": mode,
                    "files": _inventory(stage), "source_binding": copy.deepcopy(calibration["binding"]),
                    "question_id": policy["question_id"], "policy_sha256": fingerprint(policy),
                    "calibration_sha256": fingerprint(calibration)}
        _atomic_save(manifest, stage / "bundle.json")
        checksum = file_sha256(stage / "bundle.json")
        receipt = {"directory": str(destination.resolve()), "manifest_sha256": checksum, "mode": mode}
        if receipt_path is not None:
            receipt_path = Path(receipt_path)
            if receipt_path.exists():
                if _read_json(receipt_path) != receipt:
                    raise ValueError("publication receipt already names a different bundle")
            else:
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_save(receipt, receipt_path)
        stage.rename(destination)
    return receipt


def _has_content(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    return value is not None


class SelectiveBundle:
    """Use load_bundle to construct a verified instance; no external action is taken."""

    def __init__(self, checkpoint, questions, calibration, policy, mode, checksum):
        self._checkpoint = checkpoint
        self._questions = copy.deepcopy(questions)
        self._calibration = copy.deepcopy(calibration)
        self._policy = copy.deepcopy(policy)
        self._mode = mode
        self.manifest_sha256 = checksum

    @property
    def questions(self):
        return copy.deepcopy(self._questions)

    def predict(self, state, *, language):
        """Return a typed decision or an explicit review result, using unrounded scores.

        Language is trusted application metadata, not language detection. Passing
        these structural checks does not establish semantic domain membership.
        """
        qid = self._policy["question_id"]
        result = {"status": "review", "reason": None, "question_id": qid, "value": None,
                  "manifest_sha256": self.manifest_sha256}
        if self._mode != "qualified":
            reason = "review_only_bundle" if self._policy["selection"]["passed"] else "policy_not_qualified"
            return {**result, "reason": reason}
        languages = self._policy["selection"]["protocol"]["required_languages"]
        if not isinstance(language, str) or language not in languages:
            return {**result, "reason": "unsupported_language"}
        if not isinstance(state, (str, dict, list)):
            return {**result, "reason": "empty_or_invalid_state"}
        try:
            # Reject non-JSON objects and non-finite values before model rendering.
            _json_value(state, "state")
            json.dumps(state, allow_nan=False)
            if not _has_content(state):
                return {**result, "reason": "empty_or_invalid_state"}
            item = prepare_input(self._checkpoint, state, self._questions[qid])
        except (TypeError, ValueError, RecursionError):
            return {**result, "reason": "unsupported_input"}
        try:
            with torch.inference_mode():
                logits, _ = forward_batch(self._checkpoint, [item])
            if not torch.isfinite(logits).all():
                return {**result, "reason": "non_finite_prediction"}
            probabilities = calibrated_probabilities(logits[0, :len(item["markers"])].cpu().tolist(),
                                                      item["qtype"], self._calibration)
        except (ValueError, RuntimeError):
            return {**result, "reason": "inference_unavailable"}
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        question = self._questions[qid]
        value = (list(question["criteria"])[index] if question["type"] == "choice"
                 else bool(index) if question["type"] == "noul" else index)
        threshold = self._policy["selection"]["by_language"][language]["selected"]["threshold"]
        suggestion = {"type": question["type"], "value": value, "probabilities": probabilities,
                      "max_probability": probabilities[index], "threshold": threshold}
        if question["type"] == "score":
            suggestion["expected_score"] = sum(i * probability for i, probability in enumerate(probabilities))
        if probabilities[index] < threshold:
            return {**result, "reason": "below_threshold", "suggestion": suggestion}
        return {**result, "status": "automate", "reason": "qualified_policy", "value": value, "prediction": suggestion}


def load_bundle(directory, *, expected_manifest_sha256, device="cpu"):
    """Verify a separately trusted manifest hash before constructing the CPU model.

    Checksums do not authenticate an attacker who can replace the trusted hash.
    Preserve the returned export receipt through a trusted release channel.
    """
    if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)):
        raise ValueError("supply the trusted lowercase SHA-256 from the bundle export receipt")
    if device != "cpu":
        raise ValueError("these deployment bundles currently validate CPU fp32 inference only")
    directory = Path(directory)
    path = directory / "bundle.json"
    if path.is_symlink() or file_sha256(path) != expected_manifest_sha256:
        raise ValueError("bundle manifest differs from the trusted hash")
    manifest = _read_json(path)
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("stage") != "selective_bundle" or _inventory(directory) != manifest.get("files")):
        raise ValueError("bundle files are missing, added or changed")
    questions, calibration, policy, evaluation, runtime = [
        _read_json(directory / name) for name in
        ("questions.json", "calibration.json", "policy.json", "evaluation.json", "runtime.json")]
    _check_components(questions, calibration, policy, evaluation, manifest.get("mode"))
    if (fingerprint(manifest.get("source_binding")) != fingerprint(calibration["binding"])
            or manifest.get("calibration_sha256") != fingerprint(calibration)
            or manifest.get("policy_sha256") != fingerprint(policy)
            or manifest.get("question_id") != policy["question_id"]):
        raise ValueError("bundle manifest has inconsistent evidence links")
    if runtime != _runtime_contract():
        raise ValueError("bundle runtime or inference implementation differs; verify a new release before deployment")
    config = _read_json(directory / "model/rl_agent_config.json")
    changed_configs = {"rl_agent_config.json", "tokenizer/tokenizer_config.json"}
    if any(manifest["files"].get("model/" + name) != checksum for name, checksum in
           calibration["binding"]["model_files"].items() if name not in changed_configs):
        raise ValueError("bundle model assets differ from the calibrated source checkpoint")
    if any(manifest["files"].get("source-config/" + name) != calibration["binding"]["model_files"].get(name)
           for name in changed_configs):
        raise ValueError("bundle source configurations differ from the calibrated checkpoint")
    source_config = _read_json(directory / "source-config/rl_agent_config.json")
    if config != _calibrated_config(source_config, calibration):
        raise ValueError("SDK model configuration differs from the evaluated calibration/rendering")
    with tempfile.TemporaryDirectory(prefix="laya-bundle-tokenizer-") as temporary:
        source = Path(temporary) / "tokenizer/tokenizer_config.json"
        source.parent.mkdir()
        shutil.copyfile(directory / "source-config/tokenizer/tokenizer_config.json", source)
        _fix_tokenizer_config(temporary)
        if _read_json(source) != _read_json(directory / "model/tokenizer/tokenizer_config.json"):
            raise ValueError("tokenizer configuration differs from the calibrated source")
    checkpoint = load_local_checkpoint(directory / "model", device=device) if manifest["mode"] == "qualified" else None
    if _inventory(directory) != manifest["files"]:
        raise ValueError("bundle files changed during loading")
    return SelectiveBundle(checkpoint, questions, calibration, policy, manifest["mode"], expected_manifest_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model", "calibration_predictions", "policy_predictions", "calibration", "policy", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--test-predictions")
    parser.add_argument("--receipt")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    receipt = export_bundle(args.output, dataset_dir=args.dataset, model_dir=args.model,
                             calibration_predictions=_read_json(args.calibration_predictions),
                             policy_predictions=_read_json(args.policy_predictions), calibration=_read_json(args.calibration),
                             policy=_read_json(args.policy),
                             test_predictions=_read_json(args.test_predictions) if args.test_predictions else None,
                             review_only=args.review_only, receipt_path=args.receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
