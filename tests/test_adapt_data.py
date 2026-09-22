"""Offline adaptation data contract tests; no model weights or training.

Run: python tests/test_adapt_data.py
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.adapt_data import (  # noqa: E402
    DEFAULT_FRACTIONS, SPLITS, make_split_manifest, question_fingerprint,
    read_dataset, select_independent_records, target_index, validate_questions, validate_records,
    verify_split_manifest, write_dataset,
)


def questions():
    return {
        "queue": {"type": "choice", "instructions": "Route the ticket",
                  "criteria": {"billing": "payments", "technical": "bugs"}},
        "urgency": {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "high"]},
        "incident": {"type": "noul", "instructions": "Does this report an incident?"},
    }


def record(key, **changes):
    row = {"id": key, "group_id": "fixture:" + key, "language": "en", "state": "Ticket " + key,
           "targets": {"queue": "billing", "urgency": 0, "incident": False},
           "source": {"dataset": "local-synthetic", "revision": "v1", "license": "CC0-1.0", "record_id": key}}
    row.update(changes)
    return row


class ValidationTests(unittest.TestCase):
    def test_typed_targets_and_order(self):
        q = questions()
        for qid, gold, expected in [("queue", "technical", 1), ("urgency", 1, 1),
                                    ("incident", False, 0), ("incident", True, 1)]:
            self.assertEqual(target_index(q[qid], gold), expected)
        self.assertEqual(validate_records([record("b"), record("a", targets={"queue": "billing"})], q)[0]["id"], "a")
        q["queue"]["criteria"] = ["technical", "billing"]
        self.assertEqual(target_index(q["queue"], "billing"), 1)

    def test_invalid_targets_are_not_coerced(self):
        for qid, gold in [("queue", 0), ("queue", "unknown"), ("urgency", True), ("urgency", 0.0),
                          ("urgency", -1), ("urgency", 2), ("incident", 1), ("incident", "false")]:
            with self.subTest(qid=qid, gold=gold), self.assertRaises(ValueError):
                validate_records([record("a", targets={qid: gold})], questions())

    def test_malformed_records(self):
        cases = [{"id": " "}, {"group_id": None}, {"language": "EN"}, {"language": ""},
                 {"state": None}, {"state": True}, {"state": {1: "bad key"}},
                 {"targets": {}}, {"targets": {"missing": "x"}}, {"source": {}},
                 {"metadata": float("nan")}, {"metadata": float("inf")}, {"metadata": object()}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_records([record("a", **changes)], questions())
        for rows in ([], [None], [record("a"), record("a")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_records(rows, questions())
        for field in ("dataset", "revision", "license", "record_id"):
            row = record("a")
            del row["source"][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source." + field):
                validate_records([row], questions())

    def test_malformed_questions(self):
        cases = [{}, {"q": None}, {"q": {"type": "other", "instructions": "x"}},
                 {"q": {"type": "noul"}}, {"q": {"type": "choice", "instructions": "x", "criteria": []}},
                 {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}},
                 {"q": {"type": "choice", "instructions": "x", "criteria": [1]}},
                 {"q": {"type": "score", "instructions": "x", "criteria": {"low": 0}}},
                 {"q": {"type": "score", "instructions": "x", "criteria": []}},
                 {"q": {"type": "noul", "instructions": "x", "criteria": {"True": "yes"}}}]
        for q in cases:
            with self.subTest(q=q), self.assertRaises(ValueError):
                validate_questions(q)

    def test_fingerprints_preserve_rendered_order(self):
        q = questions()
        original = question_fingerprint(q)
        q["queue"]["criteria"] = dict(reversed(list(q["queue"]["criteria"].items())))
        self.assertNotEqual(original, question_fingerprint(q))
        q = questions()
        q["urgency"]["criteria"].reverse()
        self.assertNotEqual(original, question_fingerprint(q))
        self.assertNotEqual(original, question_fingerprint(dict(reversed(list(questions().items())))))
        q["queue"]["instructions"] = {"first": "x", "second": "y"}
        structured = question_fingerprint(q)
        q["queue"]["instructions"] = {"second": "y", "first": "x"}
        self.assertNotEqual(structured, question_fingerprint(q))

    def test_callers_objects_are_not_mutated(self):
        rows = [record("b"), record("a")]
        q, before = questions(), copy.deepcopy(rows)
        make_split_manifest(rows, q)
        checked = validate_records(rows, q)
        checked[0]["targets"]["queue"] = "technical"
        checked_q = validate_questions(q)
        checked_q["queue"]["criteria"]["extra"] = "x"
        self.assertEqual(rows, before)
        self.assertEqual(q, questions())


class SplitTests(unittest.TestCase):
    def test_reproducible_group_assignments(self):
        rows = [record(str(i), group_id="family:%d" % (i // 2), language="en" if i % 2 else "de") for i in range(400)]
        manifest = make_split_manifest(rows, questions())
        self.assertEqual(manifest, make_split_manifest(reversed(rows), questions()))
        self.assertNotEqual(manifest["assignments"], make_split_manifest(rows, questions(), seed="another")["assignments"])
        self.assertTrue(all(manifest["counts"].values()))
        for i in range(0, 400, 2):
            self.assertEqual(manifest["assignments"][str(i)], manifest["assignments"][str(i + 1)])
        self.assertEqual(sum(manifest["counts"].values()), 400)
        self.assertEqual(sum(manifest["group_counts"].values()), 200)
        for split in SPLITS:
            self.assertEqual(manifest["language_group_counts"][split]["de"], manifest["group_counts"][split])
            self.assertEqual(manifest["language_group_counts"][split]["en"], manifest["group_counts"][split])

    def test_transitive_duplicates_and_families_stay_together(self):
        rows = [record("a", state={"body": "café\n charge"}),
                record("b", state={"body": "cafe\u0301  charge"}, group_id="shared"),
                record("c", state={"body": "other"}, group_id="shared"),
                record("d", state={"body": "other"})]
        manifest = make_split_manifest(rows, questions())
        self.assertEqual(len(set(manifest["assignments"].values())), 1)
        self.assertEqual(manifest["group_count"], 1)
        self.assertEqual(manifest["normalized_duplicate_rows"], 2)
        self.assertEqual(sum(manifest["counts"].values()), 4)

    def test_duplicate_source_identity_links_revisions(self):
        a, b = record("a"), record("b")
        b["source"].update(record_id="a", revision="v2")
        manifest = make_split_manifest([a, b], questions())
        self.assertEqual(manifest["groups"]["a"], manifest["groups"]["b"])

    def test_different_datasets_do_not_link_by_record_id_alone(self):
        a, b = record("a"), record("b")
        b["source"].update(record_id="a", dataset="another-source")
        self.assertEqual(make_split_manifest([a, b], questions())["group_count"], 2)

    def test_duplicate_label_conflicts_including_partial_labels(self):
        rows = [record("a", state="same", targets={"queue": "billing"}),
                record("b", state="same", targets={"incident": False}),
                record("c", state="same", targets={"incident": True})]
        before = copy.deepcopy(rows)
        with self.assertRaisesRegex(ValueError, "contradictory duplicate"):
            make_split_manifest(rows, questions())
        self.assertEqual(rows, before)
        rows.pop()
        self.assertEqual(make_split_manifest(rows, questions())["normalized_duplicate_rows"], 1)

    def test_small_splits_are_reported_without_duplicating_rows(self):
        manifest = make_split_manifest([record("a")], questions())
        self.assertEqual(sorted(manifest["counts"].values()), [0, 0, 0, 1])

    def test_independent_representatives_use_neither_gold_values_nor_row_order(self):
        rows = [record(str(i), group_id="one-family", language="en" if i % 2 else "de") for i in range(20)]
        q = questions()
        manifest = make_split_manifest(rows, q)
        split = manifest["assignments"]["0"]
        chosen = select_independent_records(rows, q, manifest, split=split, question_id="queue")
        self.assertEqual(len(chosen), 2)
        self.assertEqual({row["language"] for row in chosen}, {"en", "de"})
        self.assertEqual(chosen, select_independent_records(reversed(rows), q, manifest, split=split, question_id="queue"))
        for row in rows:
            row["targets"]["queue"] = "technical"
        changed = select_independent_records(rows, q, make_split_manifest(rows, q), split=split, question_id="queue")
        self.assertEqual([row["id"] for row in chosen], [row["id"] for row in changed])
        chosen[0]["source"]["license"] = "changed"
        self.assertTrue(all(row["source"]["license"] == "CC0-1.0" for row in rows))

    def test_missing_labels_empty_partitions_and_invalid_selection(self):
        rows = [record("a", group_id="family", targets={"incident": False}),
                record("b", group_id="family", targets={"queue": "billing"})]
        q, manifest = questions(), make_split_manifest(rows, questions())
        split = manifest["assignments"]["a"]
        self.assertEqual([row["id"] for row in select_independent_records(rows, q, manifest, split=split,
                                                                         question_id="queue")], ["b"])
        empty = next(name for name in SPLITS if name != split)
        self.assertEqual(select_independent_records(rows, q, manifest, split=empty, question_id="queue"), [])
        self.assertEqual(select_independent_records(rows, q, manifest, split=split, question_id="urgency"), [])
        for selected_split, qid in [("unknown", "queue"), (split, "unknown")]:
            with self.assertRaises(ValueError):
                select_independent_records(rows, q, manifest, split=selected_split, question_id=qid)

    def test_invalid_split_parameters(self):
        for update in ({"train": 0}, {"train": -0.1}, {"train": True}, {"train": float("nan")},
                       {"train": 0.7}, {"other": 0.1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                make_split_manifest([record("a")], questions(), fractions={**DEFAULT_FRACTIONS, **update})
        with self.assertRaises(ValueError):
            make_split_manifest([record("a")], questions(), seed="")

    def test_manifest_rejects_tampering(self):
        rows, q = [record("a", state={"subject": "invoice", "body": "charged twice"})], questions()
        manifest = make_split_manifest(rows, q)
        verify_split_manifest(rows, q, manifest)
        variants = []
        for field, value in [("schema_version", True), ("schema_version", 2), ("records_sha256", "0" * 64),
                             ("group_count", True), ("normalized_duplicate_rows", False), ("seed", None)]:
            altered = copy.deepcopy(manifest)
            altered[field] = value
            variants.append(altered)
        altered = copy.deepcopy(manifest)
        altered["assignments"]["a"] = "unexpected"
        variants.append(altered)
        for altered in variants:
            with self.subTest(manifest=altered), self.assertRaises(ValueError):
                verify_split_manifest(rows, q, altered)
        rows[0]["state"] = dict(reversed(list(rows[0]["state"].items())))
        with self.assertRaises(ValueError):
            verify_split_manifest(rows, q, manifest)


class PersistenceTests(unittest.TestCase):
    def test_roundtrip_preserves_unicode_and_rendering_order(self):
        rows, q = [record("a", state={"z": "café", "a": "Rechnung"})], questions()
        q["queue"]["instructions"] = {"z": "first", "a": "second"}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "nested" / "data"
            manifest = write_dataset(directory, rows, q)
            loaded_q, loaded_rows, loaded_manifest = read_dataset(directory)
            self.assertEqual(loaded_q, q)
            self.assertEqual(loaded_rows, rows)
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(list(loaded_q["queue"]["criteria"]), ["billing", "technical"])
            self.assertEqual(list(loaded_q["queue"]["instructions"]), ["z", "a"])
            self.assertEqual(list(loaded_rows[0]["state"]), ["z", "a"])
            self.assertNotIn(b"\r\n", (directory / "records.jsonl").read_bytes())
            with self.assertRaises(FileExistsError):
                write_dataset(directory, rows, q)
            self.assertEqual(read_dataset(directory), (loaded_q, loaded_rows, loaded_manifest))

    def test_corrupted_disk_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "data"
            write_dataset(directory, [record("a")], questions())
            altered = record("a", targets={"queue": "technical"})
            (directory / "records.jsonl").write_text(json.dumps(altered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_dataset(directory)

    def test_failed_write_does_not_publish_partial_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "data"
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    write_dataset(directory, [record("a")], questions())
            self.assertFalse(directory.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
