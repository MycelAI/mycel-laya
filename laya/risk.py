"""Offline selection and independent testing of categorical decision thresholds.

These opt-in utilities do not change Agent confidence or authorize deployment.
Inputs must be independent representatives from a fixed, audited population.
"""
import hashlib
import json
import math


DEFAULT_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99)


def _number(value, name, *, low=0.0, high=1.0):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError("%s must be finite and between %s and %s" % (name, low, high))
    return float(value)


def binomial_upper_bound(events: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound, by log-CDF inversion.

    No observations gives 1, not a zero-error claim. Assumes independent Bernoulli
    trials. Alpha is the tail probability for this one comparison, not a family.
    """
    if type(events) is not int or type(trials) is not int or not 0 <= events <= trials:
        raise ValueError("events and trials must be integers with 0 <= events <= trials")
    alpha = _number(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if events == trials:
        return 1.0
    if events == 0:
        return -math.expm1(math.log(alpha) / trials)
    coefficients = [math.lgamma(trials + 1) - math.lgamma(j + 1) - math.lgamma(trials - j + 1)
                    for j in range(events + 1)]
    low, high = events / trials, 1.0
    for _ in range(60):
        p = (low + high) / 2
        if p == low or p == high:
            break
        log_p, log_q = math.log(p), math.log1p(-p)
        terms = [coefficient + j * log_p + (trials - j) * log_q
                 for j, coefficient in enumerate(coefficients)]
        largest = max(terms)
        log_cdf = largest + math.log(math.fsum(math.exp(term - largest) for term in terms))
        if log_cdf > math.log(alpha):
            low = p
        else:
            high = p
    return high


def _hash(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _protocol(languages, max_error, min_coverage, alpha):
    if (not isinstance(languages, (list, tuple)) or not languages
            or any(not isinstance(lang, str) or not lang.strip() for lang in languages)
            or len(set(languages)) != len(languages)):
        raise ValueError("required_languages must be a nonempty list of unique language tags")
    max_error = _number(max_error, "max_error")
    min_coverage = _number(min_coverage, "min_coverage")
    alpha = _number(alpha, "alpha")
    if max_error >= 1 or min_coverage <= 0 or not 0 < alpha < 1:
        raise ValueError("require max_error < 1, min_coverage > 0 and 0 < alpha < 1")
    return {"required_languages": sorted(languages), "max_error": max_error,
            "min_coverage": min_coverage, "alpha": alpha}


def _samples(samples, languages, n_options=None):
    """Validate one fixed question's predictions; repeated groups cannot inflate n."""
    rows, ids, groups = [], set(), set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each prediction sample must be a dictionary")
        for field in ("id", "group_id", "language"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError("prediction %s must be a nonempty string" % field)
        key, group, lang = sample["id"], sample["group_id"], sample["language"]
        if lang not in languages:
            raise ValueError("prediction language %r was not declared in the protocol" % lang)
        if key in ids or (group, lang) in groups:
            raise ValueError("predictions must have unique ids and one independent group per language")
        ids.add(key)
        groups.add((group, lang))
        probabilities = sample.get("probabilities")
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) < 2:
            raise ValueError("predictions need a probability vector with at least two options")
        probabilities = [_number(value, "probability") for value in probabilities]
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        if n_options is None:
            n_options = len(probabilities)
        if len(probabilities) != n_options:
            raise ValueError("all predictions must have the same fixed option schema")
        target = sample.get("target")
        if type(target) is not int or not 0 <= target < n_options:
            raise ValueError("target must be an integer option index")
        prediction = max(range(n_options), key=probabilities.__getitem__)
        rows.append({"id": key, "group_id": group, "language": lang, "probabilities": probabilities,
                     "target": target, "score": probabilities[prediction], "error": int(prediction != target)})
    return sorted(rows, key=lambda row: row["id"]), n_options


def _summary(rows, threshold, protocol, tail):
    selected = [] if threshold is None else [row for row in rows if row["score"] >= threshold]
    n, accepted = len(rows), len(selected)
    errors = sum(row["error"] for row in selected)
    upper = binomial_upper_bound(errors, accepted, tail)
    # A lower bound on successes is one minus an upper bound on failures.
    lower = 1 - binomial_upper_bound(n - accepted, n, tail)
    qualified = upper <= protocol["max_error"] and lower >= protocol["min_coverage"]
    return {"threshold": threshold, "samples": n, "accepted": accepted, "errors": errors,
            "coverage": accepted / n if n else 0.0, "error_rate": errors / accepted if accepted else None,
            "error_upper": upper, "coverage_lower": lower, "qualified": qualified}


def select_risk_policy(samples, *, required_languages, thresholds=DEFAULT_THRESHOLDS,
                       max_error=0.05, min_coverage=0.5, alpha=0.05):
    """Select from a threshold grid fixed before observing selection-set outcomes.

    Scores are max calibrated class probabilities, not the SDK's entropy-based
    confidence. Bonferroni covers both bounds, every threshold and every declared
    language. Missing evidence or insufficient coverage cannot qualify.
    """
    protocol = _protocol(required_languages, max_error, min_coverage, alpha)
    if not isinstance(thresholds, (list, tuple)) or not thresholds:
        raise ValueError("thresholds must be a nonempty fixed list")
    grid = [_number(value, "threshold") for value in thresholds]
    if len(set(grid)) != len(grid):
        raise ValueError("threshold grid must not contain duplicates")
    grid.sort()
    rows, n_options = _samples(samples, protocol["required_languages"])
    tail = protocol["alpha"] / (2 * len(required_languages) * len(grid))
    by_language = {}
    for lang in protocol["required_languages"]:
        subset = [row for row in rows if row["language"] == lang]
        candidates = [_summary(subset, threshold, protocol, tail) for threshold in grid]
        eligible = [candidate for candidate in candidates if candidate["qualified"]]
        chosen = max(eligible, key=lambda row: (row["accepted"], row["threshold"])) if eligible else None
        by_language[lang] = {"selected": chosen, "candidates": candidates}
    return {"schema_version": 1, "stage": "selection", "protocol": protocol, "threshold_grid": grid,
            "n_options": n_options, "tail_probability": tail, "by_language": by_language,
            "samples_sha256": _hash(rows), "group_hashes": sorted({_hash(row["group_id"]) for row in rows}),
            "id_hashes": sorted({_hash(row["id"]) for row in rows}),
            "passed": all(row["selected"] is not None for row in by_language.values())}


def evaluate_risk_policy(samples, selection):
    """Test a previously selected policy once on fresh independent groups.

    The report concerns the predeclared population and sampling scheme, not future
    distribution shifts. Keep this report bound to model, schema and calibration
    artifacts externally; these functions do not load or authenticate a bundle.
    """
    if (not isinstance(selection, dict) or type(selection.get("schema_version")) is not int
            or selection["schema_version"] != 1 or selection.get("stage") != "selection"
            or selection.get("passed") is not True):
        raise ValueError("independent testing requires a qualifying selection report")
    settings = selection.get("protocol")
    if not isinstance(settings, dict):
        raise ValueError("selection report has no protocol")
    try:
        protocol = _protocol(settings["required_languages"], settings["max_error"],
                             settings["min_coverage"], settings["alpha"])
    except KeyError as exc:
        raise ValueError("selection protocol is incomplete") from exc
    n_options = selection.get("n_options")
    if type(n_options) is not int or n_options < 2:
        raise ValueError("selection report needs a fixed option count")
    by_language = selection.get("by_language")
    if not isinstance(by_language, dict) or set(by_language) != set(protocol["required_languages"]):
        raise ValueError("selection language slices do not match its protocol")
    thresholds = {}
    for lang, entry in by_language.items():
        chosen = entry.get("selected") if isinstance(entry, dict) else None
        if not isinstance(chosen, dict) or chosen.get("qualified") is not True:
            raise ValueError("each required language must have a qualifying selected threshold")
        thresholds[lang] = _number(chosen.get("threshold"), "selected threshold")
    previous = {}
    for field in ("group_hashes", "id_hashes"):
        hashes = selection.get(field)
        if (not isinstance(hashes, list) or not hashes or any(not isinstance(value, str) or len(value) != 64
                                                             for value in hashes)):
            raise ValueError("selection report is missing its sample identities")
        previous[field] = set(hashes)
    rows, _ = _samples(samples, protocol["required_languages"], n_options)
    if (any(_hash(row["group_id"]) in previous["group_hashes"] for row in rows)
            or any(_hash(row["id"]) in previous["id_hashes"] for row in rows)):
        raise ValueError("final test reuses selection-set examples or related groups")
    tail = protocol["alpha"] / (2 * len(protocol["required_languages"]))
    results = {lang: _summary([row for row in rows if row["language"] == lang], threshold, protocol, tail)
               for lang, threshold in thresholds.items()}
    return {"schema_version": 1, "stage": "final_test", "protocol": protocol, "n_options": n_options,
            "selection_sha256": _hash(selection), "samples_sha256": _hash(rows),
            "tail_probability": tail, "by_language": results,
            "passed": all(result["qualified"] for result in results.values())}
