"""Bounded temperature fitting and typed metrics for frozen logit artifacts."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from .adapt_data import fingerprint, validate_questions
from .adapt_eval import validate_predictions, verify_prediction_dataset
from .adapt_train import _atomic_save
from .common import QTYPES, TEMP_MAX, TEMP_MIN, temp_bucket


def _log_probabilities(logits, temperature):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = values - values.max()
    result = values - np.log(np.exp(values).sum())
    if not np.isfinite(result).all():
        raise ValueError("logits exceed the numerical range supported by calibration")
    return result


def fit_temperature(records):
    """Minimize hard-label NLL over the SDK's temperature range.

    NLL is convex in inverse temperature. Bisection on its monotone derivative
    handles interior and boundary optima without fitting outside runtime limits.
    Callers supply validated records; no inherited configuration is merged.
    """
    if not records:
        raise ValueError("temperature fitting needs labelled logits")
    count, width = len(records), max(len(row["logits"]) for row in records)
    values, valid = np.zeros((count, width)), np.zeros((count, width), dtype=bool)
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    for index, row in enumerate(records):
        vector = np.asarray(row["logits"], dtype=np.float64)
        # Removing a row constant preserves the optimum and improves stability.
        vector = vector - vector.max()
        values[index, :len(vector)] = vector
        valid[index, :len(vector)] = True
    if not np.isfinite(values).all():
        raise ValueError("logits exceed the numerical range supported by calibration")
    if np.all(values == 0):
        return 1.0
    gold = values[np.arange(count), labels]

    def derivative(inverse):
        scaled = np.where(valid, inverse * values, -np.inf)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return float(np.mean((probabilities * values).sum(axis=1) - gold))

    low, high = 1 / TEMP_MAX, 1 / TEMP_MIN
    if derivative(low) >= 0:
        return float(TEMP_MAX)
    if derivative(high) <= 0:
        return float(TEMP_MIN)
    for _ in range(60):
        midpoint = (low + high) / 2
        if derivative(midpoint) > 0:
            high = midpoint
        else:
            low = midpoint
    return float(1 / ((low + high) / 2))


def validate_calibration(calibration, *, binding=None):
    if (not isinstance(calibration, dict) or type(calibration.get("schema_version")) is not int
            or calibration["schema_version"] != 1 or calibration.get("stage") != "calibration"
            or calibration.get("source_split") != "calibration"):
        raise ValueError("expected a calibration-partition temperature artifact")
    if not isinstance(calibration.get("binding"), dict):
        raise ValueError("calibration has no checkpoint/data/schema binding")
    if binding is not None and fingerprint(calibration["binding"]) != fingerprint(binding):
        raise ValueError("calibration model, data, rendering or question schema differs")
    temperatures, buckets = calibration.get("temperature"), calibration.get("temperature_by_options")
    if not isinstance(temperatures, list) or len(temperatures) != 3 or not isinstance(buckets, dict):
        raise ValueError("calibration needs three temperatures and an explicit bucket map")
    keys = {kind + ":" + size for kind in QTYPES for size in ("2", "3-5", "6-10", "11+")}
    if set(buckets) - keys or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        or not TEMP_MIN <= value <= TEMP_MAX for value in temperatures + list(buckets.values())
    ):
        raise ValueError("calibration temperatures must satisfy the SDK bounds and known buckets")
    for field in ("group_hashes", "id_hashes"):
        hashes = calibration.get(field)
        if not isinstance(hashes, list) or not hashes or any(
            not isinstance(value, str) or len(value) != 64 for value in hashes
        ):
            raise ValueError("calibration is missing its sample identities")


def calibrated_probabilities(logits, qtype, calibration):
    """Full precision probabilities using intentional bucket-before-type precedence."""
    validate_calibration(calibration)
    if type(qtype) is not int or qtype not in QTYPES.values():
        raise ValueError("unknown question type")
    temperature = calibration["temperature_by_options"].get(temp_bucket(qtype, len(logits)),
                                                              calibration["temperature"][qtype])
    return np.exp(_log_probabilities(logits, temperature)).tolist()


def metric_report(predictions, questions, calibration=None, *, bins=15):
    """Per-question/language hard-label metrics; fitting-set metrics are descriptive.

    Brier is the sum over classes per example. ECE uses equal-width bins with
    confidence 0 and 1 included. Ordinal MAE uses the expected level; normalized
    ranked probability score uses cumulative probabilities over K-1 boundaries.
    """
    rows = validate_predictions(predictions, questions)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    if calibration is not None:
        validate_calibration(calibration, binding=predictions["binding"])

    def summarize(subset, kind):
        probabilities, nll, correct, maes, rps = [], [], [], [], []
        for row in subset:
            temperature = 1.0 if calibration is None else calibration["temperature_by_options"].get(
                temp_bucket(row["qtype"], len(row["logits"])), calibration["temperature"][row["qtype"]])
            logp = _log_probabilities(row["logits"], temperature)
            p = np.exp(logp)
            probabilities.append(p)
            nll.append(-logp[row["label"]])
            correct.append(int(np.argmax(p) == row["label"]))
            if kind == "score":
                maes.append(abs(float(np.arange(len(p)) @ p) - row["label"]))
                gold_cdf = (np.arange(len(p) - 1) >= row["label"]).astype(float)
                rps.append(float(np.mean((np.cumsum(p)[:-1] - gold_cdf) ** 2)) if len(p) > 1 else 0.0)
        p = np.asarray(probabilities)
        targets = np.asarray([row["label"] for row in subset])
        gold = np.eye(p.shape[1])[targets]
        confidence = p.max(axis=1)
        correctness = np.asarray(correct)
        assignments = np.minimum((confidence * bins).astype(int), bins - 1)
        ece = sum(float(np.mean(assignments == index)) * abs(float(confidence[assignments == index].mean())
                    - float(correctness[assignments == index].mean()))
                  for index in range(bins) if np.any(assignments == index))
        result = {"samples": len(subset), "accuracy": float(correctness.mean()), "nll": float(np.mean(nll)),
                  "brier": float(np.mean(np.sum((p - gold) ** 2, axis=1))), "ece": ece, "ece_bins": bins}
        if kind == "score":
            result.update(ordinal_mae=float(np.mean(maes)), ranked_probability_score=float(np.mean(rps)))
        return result

    result = {}
    for qid, question in questions.items():
        subset = [row for row in rows if row["question_id"] == qid]
        if subset:
            result[qid] = {"all": summarize(subset, question["type"]), "by_language": {
                lang: summarize([row for row in subset if row["language"] == lang], question["type"])
                for lang in sorted({row["language"] for row in subset})}}
    return {"split": predictions["split"], "calibrated": calibration is not None,
            "records_sha256": predictions["records_sha256"], "questions": result}


def fit_calibration(predictions, questions, *, fit_buckets=False, min_per_type=20, min_per_bucket=2000):
    """Fit only calibration records, replacing all inherited temperature settings.

    Per-type and bucket keys match the SDK and PR #19's temperature-map format.
    This opt-in wrapper additionally binds the exact model, data and ordered schema.
    """
    questions = validate_questions(questions)
    rows = validate_predictions(predictions, questions)
    if predictions["split"] != "calibration":
        raise ValueError("fit temperatures only on the calibration partition")
    if type(fit_buckets) is not bool or any(type(value) is not int or value < 1
                                          for value in (min_per_type, min_per_bucket)):
        raise ValueError("fit_buckets must be boolean and minimum counts must be positive integers")
    temperatures, buckets, counts = [1.0, 1.0, 1.0], {}, {}
    for kind, qtype in QTYPES.items():
        subset = [row for row in rows if row["qtype"] == qtype]
        if subset:
            if len(subset) < min_per_type:
                raise ValueError("insufficient calibration records for %s" % kind)
            temperatures[qtype] = fit_temperature(subset)
        counts[kind] = len(subset)
    if fit_buckets:
        for key in sorted({temp_bucket(row["qtype"], len(row["logits"])) for row in rows}):
            subset = [row for row in rows if temp_bucket(row["qtype"], len(row["logits"])) == key]
            counts[key] = len(subset)
            if len(subset) >= min_per_bucket:
                buckets[key] = fit_temperature(subset)
    calibration = {"schema_version": 1, "stage": "calibration", "source_split": "calibration",
                   "binding": copy.deepcopy(predictions["binding"]), "temperature": temperatures,
                   "temperature_by_options": buckets, "counts": counts,
                   "source_records_sha256": predictions["records_sha256"],
                   "group_hashes": sorted({fingerprint(row["group_id"]) for row in rows}),
                   "id_hashes": sorted({fingerprint(row["id"]) for row in rows}),
                   "fit": {"objective": "hard_label_nll", "temperature_range": [TEMP_MIN, TEMP_MAX],
                           "fit_buckets": fit_buckets, "min_per_type": min_per_type, "min_per_bucket": min_per_bucket}}
    calibration["fit_metrics"] = {"raw": metric_report(predictions, questions),
                                  "calibrated": metric_report(predictions, questions, calibration)}
    return calibration


def probability_records(predictions, questions, calibration, *, question_id):
    """Validated risk-policy inputs, retaining independent family identities."""
    rows = validate_predictions(predictions, questions)
    validate_calibration(calibration, binding=predictions["binding"])
    if question_id not in questions:
        raise ValueError("unknown policy question")
    if predictions["split"] != "calibration":
        previous_groups, previous_ids = set(calibration["group_hashes"]), set(calibration["id_hashes"])
        if any(fingerprint(row["group_id"]) in previous_groups
               or fingerprint(row["id"]) in previous_ids for row in rows):
            raise ValueError("evaluation reuses calibration examples or related groups")
    return [{"id": row["id"], "group_id": row["group_id"], "language": row["language"],
             "target": row["label"], "probabilities": calibrated_probabilities(row["logits"], row["qtype"], calibration)}
            for row in rows if row["question_id"] == question_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-buckets", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("calibration output already exists")
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    questions = verify_prediction_dataset(predictions, args.dataset)
    result = fit_calibration(predictions, questions, fit_buckets=args.fit_buckets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(result, destination)
    print(json.dumps({"temperature": result["temperature"], "temperature_by_options": result["temperature_by_options"]}))


if __name__ == "__main__":
    main()
