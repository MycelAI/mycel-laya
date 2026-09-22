"""Synthetic source-adapter tests. No public corpus or model download."""
import importlib.util
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("arbanking_adapter", ROOT / "research/scripts/prepare_arbanking77.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

from laya.adapt_data import make_split_manifest  # noqa: E402


def row(qid, **changes):
    value = {"QID": qid, "QuestionID_MSA1": "Tr" + qid, "Intent_en": "card arrival",
             "Question_en": "Where is my card " + qid, "Question_MSA1": "أين بطاقتي " + qid}
    value.update(changes)
    return value


class AdapterTests(unittest.TestCase):
    def test_official_test_family_is_reserved_in_both_languages(self):
        q, rows, fixed, audit = adapter.convert_rows([row("1"), row("2", QuestionID_MSA1="Te2")], ["card arrival"])
        manifest = make_split_manifest(rows, q, fractions=adapter.FRACTIONS, fixed_splits=fixed)
        self.assertEqual(audit["paired_families"], 2)
        self.assertEqual(manifest["assignments"]["arbanking77:2:en"], "test")
        self.assertEqual(manifest["assignments"]["arbanking77:2:ar"], "test")
        self.assertEqual(rows[2]["source"]["record_id"], rows[3]["source"]["record_id"])

    def test_malformed_missing_and_wrong_script_remove_whole_families(self):
        bad_csv = row("2")
        bad_csv[None] = ["shifted value"]
        _, rows, _, audit = adapter.convert_rows([row("1"), bad_csv, row("3", Question_MSA1="NULL"),
                                                 row("4", Question_MSA1="Where is my card")], ["card arrival"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(audit["excluded_families"]), {"2", "3", "4"})

    def test_conflicting_duplicate_removes_both_translation_families(self):
        rows = [row("1"), row("2", Intent_en="refund", Question_en="Where is my card 1")]
        _, records, _, audit = adapter.convert_rows(rows, ["card arrival", "refund"])
        self.assertEqual(records, [])
        self.assertEqual(audit["exclusion_counts"], {"contradictory_normalized_duplicate": 2})

    def test_untraceable_identity_and_unknown_labels_fail(self):
        for rows in ([row("1"), row("1")], [row("bad")], [row("1", Intent_en="unknown")]):
            with self.assertRaises(ValueError):
                adapter.convert_rows(rows, ["card arrival"])

    def test_prepare_verifies_sources_and_publishes_audit_with_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "data").mkdir(parents=True)
            (source / "LICENSE").write_text("Synthetic test license", encoding="utf-8")
            (source / "README.md").write_text("Synthetic fixture", encoding="utf-8")
            labels = ["card arrival"] + ["fixture-%d" % i for i in range(76)]
            (source / "data/Banking77_intents.csv").write_text("label_en\n" + "\n".join(labels), encoding="utf-8")
            with (source / "data/Banking77_full_corpus.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row("1")))
                writer.writeheader()
                writer.writerow(row("1"))
            expected = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in adapter.EXPECTED}
            with self.assertRaisesRegex(ValueError, "pinned"):
                adapter.prepare(source, root / "bad")
            with patch.object(adapter, "EXPECTED", expected):
                audit = adapter.prepare(source, root / "prepared")
                self.assertEqual(audit["paired_families"], 1)
                self.assertEqual(json.loads((root / "prepared/source_audit.json").read_text()), audit)
                self.assertEqual((root / "prepared/SOURCE_LICENSE.txt").read_bytes(), (source / "LICENSE").read_bytes())
                original = Path.write_text

                def fail_audit(path, *args, **kwargs):
                    if path.name == "source_audit.json":
                        raise OSError("disk full")
                    return original(path, *args, **kwargs)

                with patch.object(Path, "write_text", fail_audit):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        adapter.prepare(source, root / "incomplete")
                self.assertFalse((root / "incomplete").exists())


if __name__ == "__main__":
    unittest.main()
