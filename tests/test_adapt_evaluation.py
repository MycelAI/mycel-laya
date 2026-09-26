"""Offline independent logit collection and calibration regression tests."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from laya.adapt_calibration import (  # noqa: E402
    calibrated_probabilities, fit_calibration, metric_report, probability_records, validate_calibration,
)
from laya.adapt_data import fingerprint, question_fingerprint, write_dataset  # noqa: E402
from laya.adapt_eval import collect_partition, validate_predictions, verify_prediction_dataset  # noqa: E402
from laya.adapt_model import checkpoint_files  # noqa: E402
import test_adapt_train as fixtures  # noqa: E402


QUESTIONS = {"queue": {"type": "choice", "instructions": "Route", "criteria": ["a", "b"]}}


def artifact(rows, questions=QUESTIONS, split="calibration"):
    return {"schema_version": 1, "stage": "logits", "split": split,
            "binding": {"dataset_manifest_sha256": "d" * 64, "questions_sha256": question_fingerprint(questions),
                        "model_files": {"model.safetensors": "a" * 64}, "rendering": {"max_len": 64, "head_max_len": 48}},
            "records": rows, "records_sha256": fingerprint(rows)}


def binary_rows(n=100, correct=80, logits=(2.0, 0.0)):
    return [{"id": "row:%d" % i, "group_id": "group:%d" % i, "language": "en", "question_id": "queue",
             "qtype": 0, "label": int(i >= correct), "logits": list(logits)} for i in range(n)]


class CalibrationTests(unittest.TestCase):
    def test_known_interior_optimum_and_no_inherited_buckets(self):
        data = artifact(binary_rows())
        data["binding"]["model_files"]["rl_agent_config.json"] = "f" * 64
        before = copy.deepcopy(data)
        fitted = fit_calibration(data, QUESTIONS)
        self.assertAlmostEqual(fitted["temperature"][0], 2 / math.log(4), places=10)
        self.assertEqual(fitted["temperature"][1:], [1, 1])
        self.assertEqual(fitted["temperature_by_options"], {})
        self.assertLess(fitted["fit_metrics"]["calibrated"]["questions"]["queue"]["all"]["nll"],
                        fitted["fit_metrics"]["raw"]["questions"]["queue"]["all"]["nll"])
        self.assertEqual(data, before)

    def test_boundary_optima_and_uniform_logits(self):
        for correct, logits, expected in ((100, (2, 0), .5), (0, (2, 0), 5), (50, (0, 0), 1)):
            with self.subTest(correct=correct, logits=logits):
                fitted = fit_calibration(artifact(binary_rows(correct=correct, logits=logits)), QUESTIONS)
                self.assertEqual(fitted["temperature"][0], expected)

    def test_intentional_bucket_precedence_and_per_type_fallback(self):
        fitted = fit_calibration(artifact(binary_rows()), QUESTIONS, fit_buckets=True, min_per_bucket=50)
        self.assertIn("choice:2", fitted["temperature_by_options"])
        fitted["temperature"] = [4.0, 3.0, 2.0]
        fitted["temperature_by_options"] = {"choice:2": .5}
        two = calibrated_probabilities([2, 0], 0, fitted)
        three = calibrated_probabilities([2, 0, 0], 0, fitted)
        score = calibrated_probabilities([2, 0], 1, fitted)
        self.assertAlmostEqual(two[0], 1 / (1 + math.exp(-4)))
        self.assertAlmostEqual(three[0], math.exp(.5) / (math.exp(.5) + 2))
        self.assertAlmostEqual(score[0], 1 / (1 + math.exp(-2 / 3)))

    def test_refuse_wrong_partition_insufficient_evidence_and_changed_binding(self):
        data = artifact(binary_rows())
        for split in ("policy", "test"):
            with self.assertRaisesRegex(ValueError, "calibration partition"):
                fit_calibration({**data, "split": split}, QUESTIONS)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            fit_calibration(artifact(binary_rows(2, 1)), QUESTIONS)
        fitted = fit_calibration(data, QUESTIONS)
        for field, value in (("rendering", {"max_len": 128, "head_max_len": 48}),
                             ("model_files", {"model.safetensors": "b" * 64}),
                             ("dataset_manifest_sha256", "e" * 64)):
            changed = copy.deepcopy(data)
            changed["binding"][field] = value
            with self.assertRaisesRegex(ValueError, "differs"):
                metric_report(changed, QUESTIONS, fitted)

    def test_corrupt_logits_and_dependent_samples_are_rejected(self):
        rows = binary_rows()
        for changes in ({"qtype": True}, {"label": False}, {"label": 9}, {"logits": [1]},
                        {"logits": [True, 0]}, {"question_id": "missing"}):
            changed = copy.deepcopy(rows)
            changed[0].update(changes)
            with self.assertRaises(ValueError):
                validate_predictions(artifact(changed), QUESTIONS)
        changed = copy.deepcopy(rows)
        changed[1]["group_id"] = changed[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_predictions(artifact(changed), QUESTIONS)
        data = artifact(rows)
        data["records"][0]["logits"][0] = 9
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_predictions(data, QUESTIONS)

    def test_calibration_invalid_values_and_reuse_fail(self):
        data = artifact(binary_rows())
        fitted = fit_calibration(data, QUESTIONS)
        for value in (False, None, float("nan"), float("inf"), .1, 10):
            invalid = copy.deepcopy(fitted)
            invalid["temperature"][0] = value
            with self.assertRaisesRegex(ValueError, "bounds"):
                validate_calibration(invalid)
        with self.assertRaisesRegex(ValueError, "reuses"):
            probability_records({**data, "split": "policy"}, QUESTIONS, fitted, question_id="queue")
        fresh = binary_rows()
        for row in fresh:
            row["id"] = "new:" + row["id"]
            row["group_id"] = "new:" + row["group_id"]
        samples = probability_records(artifact(fresh, split="policy"), QUESTIONS, fitted, question_id="queue")
        self.assertEqual(len(samples), 100)
        self.assertAlmostEqual(samples[0]["probabilities"][0], .8)

    def test_analytic_ordinal_metrics_and_confidence_one_boundary(self):
        questions = {"severity": {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]}}
        row = {"id": "x", "group_id": "x", "language": "en", "question_id": "severity", "qtype": 1,
               "label": 1, "logits": np.log([.2, .3, .5]).tolist()}
        metrics = metric_report(artifact([row], questions), questions)["questions"]["severity"]["all"]
        self.assertAlmostEqual(metrics["nll"], -math.log(.3))
        self.assertAlmostEqual(metrics["brier"], .78)
        self.assertAlmostEqual(metrics["ordinal_mae"], .3)
        self.assertAlmostEqual(metrics["ranked_probability_score"], .145)
        certain = metric_report(artifact(binary_rows(2, 2, (1000, 0))), QUESTIONS)["questions"]["queue"]["all"]
        self.assertEqual(certain["accuracy"], 1)
        self.assertEqual(certain["ece"], 0)
        self.assertEqual(certain["nll"], 0)
        wrong = metric_report(artifact(binary_rows(2, 0, (1000, 0))), QUESTIONS)["questions"]["queue"]["all"]
        self.assertEqual(wrong["ece"], 1)
        self.assertEqual(wrong["nll"], 1000)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdaptTrainTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def collect(self, name, **kwargs):
        f = self.fixture
        return collect_partition(f.dataset, f.model_dir, f.root / name, split="calibration", batch_size=2,
                                 chunk_size=2, log=lambda value: None, **kwargs)

    def test_resume_matches_uninterrupted_logits_and_exact_partition(self):
        f = self.fixture
        original = checkpoint_files(f.model_dir)
        full = self.collect("full")
        stopped = self.collect("resumed", max_chunks=1)
        self.assertEqual(stopped["status"], "stopped")
        self.assertFalse((f.root / "resumed/predictions.json").exists())
        resumed = self.collect("resumed", resume=True)
        self.assertEqual(full["records"], resumed["records"])
        self.assertEqual(checkpoint_files(f.model_dir), original)
        self.assertEqual({row["id"] for row in resumed["records"]},
                         {key for key, role in f.manifest["assignments"].items() if role == "calibration"})
        verify_prediction_dataset(resumed, f.dataset)
        with patch("laya.adapt_eval.collect_logits", side_effect=AssertionError("must not repeat inference")):
            self.assertEqual(self.collect("resumed", resume=True)["records"], full["records"])

    def test_renaming_partition_or_changing_gold_cannot_relabel_evidence(self):
        f = self.fixture
        result = self.collect("collected")
        with self.assertRaisesRegex(ValueError, "declared partition"):
            verify_prediction_dataset({**result, "split": "test"}, f.dataset)
        changed = copy.deepcopy(result)
        changed["records"][0]["label"] = 1 - changed["records"][0]["label"]
        changed["records_sha256"] = fingerprint(changed["records"])
        with self.assertRaisesRegex(ValueError, "declared partition"):
            verify_prediction_dataset(changed, f.dataset)

    def test_collection_retains_canonical_connected_family(self):
        f = self.fixture
        rows = copy.deepcopy(f.rows)
        alias = copy.deepcopy(rows[0])
        alias.update(id="alias", group_id="unlinked-name", language="ar", state="other text")
        rows.append(alias)
        dataset = f.root / "grouped"
        manifest = write_dataset(dataset, rows, f.questions, fixed_splits={rows[0]["id"]: "calibration"})
        result = collect_partition(dataset, f.model_dir, f.root / "collected", split="calibration", log=lambda value: None)
        observed = [row for row in result["records"] if row["id"] == "alias"]
        self.assertEqual(len(observed), 3)
        self.assertTrue(all(row["group_id"] == manifest["groups"][rows[0]["id"]] for row in observed))

    def test_resume_rejects_tampered_chunk_and_changed_source(self):
        f = self.fixture
        self.collect("partial", max_chunks=1)
        path = f.root / "partial/chunks/00000000.json"
        chunk = json.loads(path.read_text())
        chunk["records"][0]["logits"][0] += 1
        path.write_text(json.dumps(chunk))
        with self.assertRaisesRegex(ValueError, "frozen inputs"):
            self.collect("partial", resume=True)
        config = f.model_dir / "rl_agent_config.json"
        config.write_text(config.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.collect("partial", resume=True)

    def test_failed_forward_keeps_completed_chunks_resumable(self):
        f = self.fixture
        self.collect("partial", max_chunks=1)
        original = (f.root / "partial/chunks/00000000.json").read_bytes()
        with patch("laya.adapt_eval.collect_logits", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.collect("partial", resume=True)
        self.assertEqual((f.root / "partial/chunks/00000000.json").read_bytes(), original)
        self.assertFalse((f.root / "partial/predictions.json").exists())
        self.assertEqual(self.collect("partial", resume=True)["stage"], "logits")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
