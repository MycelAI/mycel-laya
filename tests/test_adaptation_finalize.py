"""Synthetic selection/finalization lifecycle tests; no model-quality claims."""
import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import test_adapt_bundle as fixtures  # noqa: E402
import torch  # noqa: E402

from laya.adapt_bundle import load_bundle  # noqa: E402
from laya.adapt_data import fingerprint, read_dataset  # noqa: E402
from laya.adapt_model import checkpoint_files  # noqa: E402
from laya.adapt_policy import select_bound_policy  # noqa: E402
from research.scripts import finalize_adaptation_experiment as recipe  # noqa: E402


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BundleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        _, _, manifest = read_dataset(f.data.directory)
        self.protocol = json.loads((ROOT / "research/arbanking77_protocol.json").read_text())
        self.protocol.update(dataset_manifest_sha256=fingerprint(manifest), question_id="queue", n_options=2,
                             base_model_files=checkpoint_files(f.model.model_dir),
                             rendering={"max_len": 64, "head_max_len": 48})
        self.protocol["policy"]["thresholds"] = [.9]
        self.selection_dir = f.model.root / "selection"
        self.run_dir = f.model.root / "final-run"
        self.candidates = [self.candidate("base")]

    def candidate(self, candidate_id, predictions=None):
        f = self.fixture
        settings = self.protocol["policy"]
        predictions = predictions or f.data.artifacts["policy"]
        policy = select_bound_policy(predictions, f.calibration, f.data.directory,
                                     question_id="queue", required_languages=["en", "ar"],
                                     thresholds=settings["thresholds"], max_error=settings["max_error"],
                                     min_coverage=settings["min_coverage"], alpha=settings["alpha_per_candidate"])
        artifacts = {"calibration": f.data.artifacts["calibration"], "policy": predictions}
        report = {"schema_version": 1, "stage": "candidate_policy_evaluation", "candidate": candidate_id,
                  "protocol_sha256": fingerprint(self.protocol), "model_files": checkpoint_files(f.model.model_dir),
                  "binding": predictions["binding"], "question_id": "queue",
                  "calibration_sha256": fingerprint(f.calibration), "selection": policy["selection"],
                  "prediction_files": {role: fingerprint(value) for role, value in artifacts.items()},
                  "final_test_evaluated": False}
        directory = f.model.root / candidate_id
        directory.mkdir(exist_ok=True)
        for role, artifact in artifacts.items():
            (directory / role).mkdir(exist_ok=True)
            recipe._atomic_save(artifact, directory / role / "predictions.json")
        recipe._atomic_save(f.calibration, directory / "calibration.json")
        recipe._atomic_save(report, directory / "report.json")
        return directory

    def freeze(self):
        return recipe.freeze_selection(self.protocol, self.candidates, self.fixture.data.directory, self.selection_dir)

    def finalize(self, receipt, **kwargs):
        f = self.fixture
        return recipe.finalize(self.selection_dir, receipt["selection_sha256"], f.data.directory,
                                f.model.model_dir, self.run_dir, f.output, log=lambda value: None, **kwargs)

    def test_selection_precedes_test_and_completed_resume_does_not_infer(self):
        f = self.fixture
        with patch.object(recipe, "collect_partition", side_effect=AssertionError("no test during selection")):
            receipt = self.freeze()
        self.assertEqual(receipt["candidate"], "base")

        def collect(*args, **kwargs):
            self.assertEqual(kwargs["split"], "test")
            identity = recipe._read(self.run_dir / "run.json")
            self.assertEqual(identity["selection_sha256"], receipt["selection_sha256"])
            return f.data.artifacts["test"]

        with patch.object(recipe, "collect_partition", side_effect=collect) as collector:
            result = self.finalize(receipt)
            self.assertEqual(collector.call_count, 1)
        self.assertEqual(result["status"], "qualified")
        self.assertEqual(result["bundle"], recipe._read(self.run_dir / "bundle-receipt.json"))
        with patch.object(recipe, "collect_partition", side_effect=AssertionError("must reuse completed result")):
            self.assertEqual(self.finalize(receipt, resume=True), result)
        with self.assertRaises(FileExistsError):
            self.finalize(receipt)

    def test_failed_final_gate_produces_review_only_and_preserves_failure(self):
        f = self.fixture
        receipt = self.freeze()
        bad = copy.deepcopy(f.data.artifacts["test"])
        for row in bad["records"]:
            row["logits"] = [0.0, 2.0]
        bad["records_sha256"] = fingerprint(bad["records"])
        with patch.object(recipe, "collect_partition", return_value=bad):
            result = self.finalize(receipt)
        self.assertEqual(result["status"], "failed_gate")
        self.assertEqual(result["candidate"], "base")
        bundle = load_bundle(f.output, expected_manifest_sha256=result["bundle"]["manifest_sha256"])
        self.assertEqual(bundle.predict("charged", language="en")["status"], "review")
        with patch.object(recipe, "collect_partition", side_effect=AssertionError("must not retry failed test")):
            self.assertEqual(self.finalize(receipt, resume=True), result)

    def test_no_qualified_candidate_never_freezes_or_opens_test(self):
        bad = copy.deepcopy(self.fixture.data.artifacts["policy"])
        for row in bad["records"]:
            row["logits"] = [0.0, 2.0]
        bad["records_sha256"] = fingerprint(bad["records"])
        self.candidate("base", bad)
        with (patch.object(recipe, "collect_partition", side_effect=AssertionError("no test")),
              self.assertRaisesRegex(ValueError, "no candidate")):
            self.freeze()
        self.assertFalse(self.selection_dir.exists())

    def test_candidate_order_and_all_results_are_preserved(self):
        self.candidates.append(self.candidate("head-1epoch"))
        receipt = self.freeze()
        self.assertEqual(receipt["candidate"], "base")  # Equal coverage uses the declared order.
        self.assertEqual(list(recipe._read(self.selection_dir / "candidate-reports.json")), ["base", "head-1epoch"])
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_coverage_ranking_prefers_greatest_worst_language_coverage(self):
        worse = copy.deepcopy(self.fixture.data.artifacts["policy"])
        for row in worse["records"][:80]:
            row["logits"] = [0.0, 0.0]
        worse["records_sha256"] = fingerprint(worse["records"])
        self.candidate("base", worse)
        self.candidates.append(self.candidate("head-1epoch"))
        self.assertEqual(self.freeze()["candidate"], "head-1epoch")

    def test_missing_earlier_candidate_and_duplicate_reports_are_rejected(self):
        self.candidates = [self.candidate("head-1epoch")]
        with self.assertRaisesRegex(ValueError, "completed prefix"):
            self.freeze()
        self.candidates = [self.fixture.model.root / "base"] * 2
        with self.assertRaisesRegex(ValueError, "untouched-test experiment"):
            self.freeze()

    def test_report_protocol_role_and_rendering_mismatch_are_rejected(self):
        path = self.candidates[0] / "report.json"
        original = recipe._read(path)
        for field, value in (("protocol_sha256", "a" * 64), ("final_test_evaluated", True), ("question_id", "other")):
            with self.subTest(field=field):
                recipe._atomic_save({**original, field: value}, path)
                with self.assertRaises(ValueError):
                    self.freeze()
        recipe._atomic_save(original, path)
        self.protocol["rendering"]["max_len"] = 128
        self.candidate("base")  # Internally consistent evidence, but from the wrong rendering recipe.
        with self.assertRaisesRegex(ValueError, "rendering"):
            self.freeze()

    def test_changed_selection_or_checkpoint_fails_before_test(self):
        receipt = self.freeze()
        with patch.object(recipe, "collect_partition", side_effect=AssertionError("must not open test")):
            with self.assertRaisesRegex(ValueError, "trusted receipt"):
                self.finalize({**receipt, "selection_sha256": "a" * 64})
            path = self.fixture.model.model_dir / "rl_agent_config.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "pinned"):
                self.finalize(receipt)
        self.assertFalse(self.run_dir.exists())

    def test_publish_then_interrupted_result_write_recovers_same_bundle(self):
        receipt = self.freeze()
        save = recipe._atomic_save

        def interrupt_result(value, path):
            if path == self.run_dir / "result.json":
                raise OSError("interrupted result write")
            save(value, path)

        with patch.object(recipe, "collect_partition", return_value=self.fixture.data.artifacts["test"]):
            with (patch.object(recipe, "_atomic_save", side_effect=interrupt_result),
                  self.assertRaisesRegex(OSError, "interrupted result write")):
                self.finalize(receipt)
            published = recipe._read(self.run_dir / "bundle-receipt.json")
            with patch.object(recipe, "export_bundle", side_effect=AssertionError("must not republish")):
                recovered = self.finalize(receipt, resume=True)
        self.assertEqual(recovered["bundle"], published)

    def test_changed_completed_status_or_run_identity_is_rejected(self):
        receipt = self.freeze()
        with patch.object(recipe, "collect_partition", return_value=self.fixture.data.artifacts["test"]):
            self.finalize(receipt)
        path = self.run_dir / "result.json"
        original = recipe._read(path)
        recipe._atomic_save({**original, "status": "failed_gate"}, path)
        with self.assertRaisesRegex(ValueError, "completed final-run"):
            self.finalize(receipt, resume=True)
        recipe._atomic_save(original, path)
        identity_path = self.run_dir / "run.json"
        identity = recipe._read(identity_path)
        identity["selection_sha256"] = "b" * 64
        recipe._atomic_save(identity, identity_path)
        with self.assertRaisesRegex(ValueError, "after final evaluation"):
            self.finalize(receipt, resume=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
