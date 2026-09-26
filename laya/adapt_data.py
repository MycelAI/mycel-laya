"""Validated labelled examples and reproducible group splits for Laya adaptation.

Opt-in dataset utilities. No model loading, downloads or training happen here.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Any, Dict, Iterable, Mapping


SCHEMA_VERSION = 1
SPLITS = ("train", "calibration", "policy", "test")
DEFAULT_FRACTIONS = {"train": 0.6, "calibration": 0.15, "policy": 0.1, "test": 0.15}
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")


def canonical_json(value: Any) -> str:
    """Stable JSON for metadata fingerprints; refuse non-finite numbers."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a nonempty string" % field)
    return value


def _json_value(value: Any, field: str) -> None:
    # Reject non-string keys rather than allowing JSON to silently coerce them.
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("%s must have string object keys" % field)
            _json_value(item, field)
    elif isinstance(value, list):
        for item in value:
            _json_value(item, field)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("%s must contain only JSON values" % field)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s must not contain NaN or infinity" % field)


def validate_questions(questions: Mapping) -> Dict[str, Dict]:
    """Copy and validate the existing SDK's typed question schema for training."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty dictionary")
    _json_value(questions, "questions")
    for qid, question in questions.items():
        _text(qid, "question id")
        if not isinstance(question, dict):
            raise ValueError("question %r must be a dictionary" % qid)
        kind = question.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("question %r has an unsupported type" % qid)
        if "instructions" not in question:
            raise ValueError("question %r is missing instructions" % qid)
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, (dict, list)) or not criteria:
                raise ValueError("choice %r must have nonempty dict/list criteria" % qid)
            labels = list(criteria)
            for label in labels:
                _text(label, "choice label")
            if len(set(labels)) != len(labels):
                raise ValueError("choice %r has duplicate labels" % qid)
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("score %r must have a nonempty list of levels" % qid)
        elif criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise ValueError("noul %r criteria must use false/true string keys" % qid)
    return copy.deepcopy(questions)


def question_fingerprint(questions: Mapping) -> str:
    """Preserve all object ordering: structured instructions also render as text."""
    checked = validate_questions(questions)
    return fingerprint(json.dumps(checked, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def _records_fingerprint(rows: list) -> str:
    # Metadata key order has no effect, but state object order affects SDK text.
    return fingerprint([{**row, "state": json.dumps(row["state"], ensure_ascii=False,
                                                   allow_nan=False, separators=(",", ":"))} for row in rows])


def target_index(question: Mapping, target: Any) -> int:
    """Convert a hard gold label to the SDK's option index without coercion."""
    kind = question["type"]
    if kind == "choice":
        labels = list(question["criteria"])
        if not isinstance(target, str) or target not in labels:
            raise ValueError("choice target must be one of the question's labels")
        return labels.index(target)
    if kind == "noul":
        if not isinstance(target, bool):
            raise ValueError("noul target must be a JSON boolean")
        return int(target)
    if kind == "score":
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(question["criteria"]):
            raise ValueError("score target must be an integer index of a rubric level")
        return target
    raise ValueError("unsupported question type")


def validate_records(records: Iterable[Mapping], questions: Mapping) -> list:
    """Validate hard-labelled examples and provenance, preserving caller objects.

    Required fields: id, group_id, language, state, targets, source. Source records
    dataset/revision/license/record_id. Related translations, conversations and
    augmentations must share a globally unique group_id. Source identities and
    normalized exact duplicates are also linked automatically. Missing task labels
    are allowed, but unknown labels are not. A language tag is a source assertion,
    not a verified language detection result.
    """
    checked = validate_questions(questions)
    result, seen = [], set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each example must be a dictionary")
        _json_value(row, "example")
        key = _text(row.get("id"), "example id")
        if key in seen:
            raise ValueError("duplicate example id %r" % key)
        seen.add(key)
        _text(row.get("group_id"), "group_id for %r" % key)
        language = row.get("language")
        if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
            raise ValueError("example %r needs a lowercase language tag, e.g. en or de" % key)
        if not isinstance(row.get("state"), (str, dict, list)):
            raise ValueError("example %r state must be text, a JSON object or a list" % key)
        targets = row.get("targets")
        if not isinstance(targets, dict) or not targets or set(targets) - set(checked):
            raise ValueError("example %r needs targets for known question ids" % key)
        for qid, target in targets.items():
            try:
                target_index(checked[qid], target)
            except ValueError as exc:
                raise ValueError("example %r, question %r: %s" % (key, qid, exc)) from exc
        source = row.get("source")
        if not isinstance(source, dict):
            raise ValueError("example %r needs source provenance" % key)
        for field in ("dataset", "revision", "license", "record_id"):
            _text(source.get(field), "source.%s for %r" % (field, key))
        result.append(copy.deepcopy(row))
    if not result:
        raise ValueError("dataset must contain at least one labelled example")
    return sorted(result, key=lambda row: row["id"])


def _normalized_state(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    if isinstance(value, list):
        return [_normalized_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized_state(item) for key, item in value.items()}
    return value


def make_split_manifest(records: Iterable[Mapping], questions: Mapping, *, seed: str = "laya-v1",
                        fractions: Mapping = None, fixed_splits: Mapping = None) -> Dict:
    """Hash related groups into four partitions, independent of input row order.

    Groups connected by normalized duplicate states or source identities are merged
    transitively. Contradictory labels for duplicate states are rejected. Row counts
    include duplicates and must not be used as independent statistical sample sizes.
    Fractions apply to unreserved groups. fixed_splits optionally reserves record
    IDs for named partitions and propagates that reservation to their whole family.
    Conflicting reservations fail. Small splits can be empty.
    """
    rows = validate_records(records, questions)
    _text(seed, "split seed")
    fractions = dict(DEFAULT_FRACTIONS if fractions is None else fractions)
    if set(fractions) != set(SPLITS) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in fractions.values()
    ) or not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("split fractions must be nonnegative train/calibration/policy/test values summing to one")
    if fixed_splits is not None:
        if (not isinstance(fixed_splits, dict) or set(fixed_splits) - {row["id"] for row in rows}
                or any(name not in SPLITS for name in fixed_splits.values())):
            raise ValueError("fixed_splits must map known example ids to named partitions")
    parent = {row["group_id"]: row["group_id"] for row in rows}

    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def merge(a, b):
        first, second = sorted((root(a), root(b)))
        parent[second] = first

    duplicate_states, source_groups, duplicate_count = {}, {}, 0
    for row in rows:
        # Different revisions of the same source record remain related even if its
        # text has changed. Adapters must retain the original identity for variants.
        source = row["source"]
        identity = (source["dataset"], source["record_id"])
        if identity in source_groups:
            merge(source_groups[identity], row["group_id"])
        else:
            source_groups[identity] = row["group_id"]
        state_hash = fingerprint(_normalized_state(row["state"]))
        if state_hash in duplicate_states:
            first = duplicate_states[state_hash]
            shared = set(first["targets"]) & set(row["targets"])
            if any(first["targets"][qid] != row["targets"][qid] for qid in shared):
                raise ValueError("contradictory duplicate labels in examples %r and %r" % (first["id"], row["id"]))
            first["targets"].update(row["targets"])
            merge(first["group_id"], row["group_id"])
            duplicate_count += 1
        else:
            # A private copy accumulates partial labels without changing dataset rows.
            duplicate_states[state_hash] = copy.deepcopy(row)
    reserved = {}
    for row in rows:
        if fixed_splits is not None and row["id"] in fixed_splits:
            group, role = root(row["group_id"]), fixed_splits[row["id"]]
            if group in reserved and reserved[group] != role:
                raise ValueError("related examples have conflicting fixed split reservations")
            reserved[group] = role
    assignments, groups = {}, {}
    counts = {name: 0 for name in SPLITS}
    languages = {name: {} for name in SPLITS}
    group_sets = {name: set() for name in SPLITS}
    language_group_sets = {name: {} for name in SPLITS}
    for row in rows:
        group = root(row["group_id"])
        value = int(fingerprint([seed, group]), 16) / 2 ** 256
        # Rounding a 256-bit hash to float can produce 1.0; never fall back to a
        # zero-probability partition, including one reserved for official tests.
        bound = 0.0
        chosen = next(name for name in reversed(SPLITS) if fractions[name] > 0)
        for name in SPLITS:
            bound += fractions[name]
            if value < bound:
                chosen = name
                break
        chosen = reserved.get(group, chosen)
        assignments[row["id"]] = chosen
        groups[row["id"]] = group
        counts[chosen] += 1
        group_sets[chosen].add(group)
        lang = row["language"]
        languages[chosen][lang] = languages[chosen].get(lang, 0) + 1
        language_group_sets[chosen].setdefault(lang, set()).add(group)
    manifest = {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "fractions": {name: fractions[name] for name in SPLITS},
        "questions_sha256": question_fingerprint(questions), "records_sha256": _records_fingerprint(rows),
        "assignments": assignments, "groups": groups, "counts": counts, "languages": languages,
        "group_counts": {name: len(groups) for name, groups in group_sets.items()},
        "language_group_counts": {name: {lang: len(groups) for lang, groups in languages.items()}
                                  for name, languages in language_group_sets.items()},
        "normalized_duplicate_rows": duplicate_count, "group_count": len(set(groups.values())),
    }
    if fixed_splits is not None:
        manifest["fixed_splits"] = dict(sorted(fixed_splits.items()))
    return manifest


def verify_split_manifest(records: Iterable[Mapping], questions: Mapping, manifest: Mapping) -> None:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("unsupported split manifest schema")
    expected = make_split_manifest(records, questions, seed=manifest.get("seed"), fractions=manifest.get("fractions"),
                                   fixed_splits=manifest.get("fixed_splits"))
    # JSON comparison also rejects bools masquerading as integer counts in Python.
    if canonical_json(manifest) != canonical_json(expected):
        raise ValueError("split manifest does not match the data, questions or recorded split configuration")


def select_independent_records(records: Iterable[Mapping], questions: Mapping, manifest: Mapping, *,
                               split: str, question_id: str) -> list:
    """Select one labelled example per connected group and language for a task.

    Selection uses IDs and the frozen split seed, never target values or model
    scores. Apply before evaluation; row-level error bounds on all family members
    would incorrectly treat dependent examples as independent evidence.
    """
    rows = validate_records(records, questions)
    verify_split_manifest(rows, questions, manifest)
    if split not in SPLITS or question_id not in questions:
        raise ValueError("select a known split and question id")
    selected = {}
    for row in rows:
        if manifest["assignments"][row["id"]] != split or question_id not in row["targets"]:
            continue
        key = (manifest["groups"][row["id"]], row["language"])
        rank = (fingerprint([manifest["seed"], question_id, row["id"]]), row["id"])
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, row)
    return sorted((row for _, row in selected.values()), key=lambda row: row["id"])


def write_dataset(destination, records: Iterable[Mapping], questions: Mapping, *, seed="laya-v1",
                  fractions=None, fixed_splits=None) -> Dict:
    """Publish a new dataset directory atomically; never overwrite an existing one."""
    rows = validate_records(records, questions)
    checked = validate_questions(questions)
    manifest = make_split_manifest(rows, checked, seed=seed, fractions=fractions, fixed_splits=fixed_splits)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("dataset destination already exists: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".laya-dataset-", dir=destination.parent) as temp:
        staged = Path(temp) / "dataset"
        staged.mkdir()
        # Preserve question/choice order; the fingerprint explicitly includes it.
        (staged / "questions.json").write_text(json.dumps(checked, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
        (staged / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                                               separators=(",", ":")) + "\n" for row in rows),
                                              encoding="utf-8", newline="\n")
        (staged / "splits.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
        staged.rename(destination)
    return manifest


def read_dataset(directory):
    """Read and verify a prepared dataset, returning questions, rows and manifest."""
    directory = Path(directory)
    questions = json.loads((directory / "questions.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    verify_split_manifest(rows, questions, manifest)
    return validate_questions(questions), validate_records(rows, questions), manifest
