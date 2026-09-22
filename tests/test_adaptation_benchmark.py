"""Offline deployment-measurement contracts using a tiny synthetic bundle."""
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

from research.scripts import benchmark_adaptation_bundle as benchmark  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BundleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.receipt = self.fixture.publish()
        self.output = self.fixture.model.root / "measurement.json"
        self.probe = {"schema_version": 1, "stage": "deployment_probe", "warmup": 1, "repeats": 3,
                      "cases": [{"id": "message", "language": "en", "state": "charged error", "expect": "prediction"},
                                {"id": "empty", "language": "en", "state": "", "expect": "review",
                                 "reason": "empty_or_invalid_state"},
                                {"id": "language", "language": "de", "state": "charged error", "expect": "review",
                                 "reason": "unsupported_language"}]}

    def measure(self, **kwargs):
        return benchmark.measure(self.fixture.output, self.receipt["manifest_sha256"], self.probe,
                                 self.output, context="synthetic unit fixture", **kwargs)

    def test_actual_model_timing_and_memory_are_separate_from_guard_paths(self):
        result = self.measure()
        self.assertGreater(result["verified_load_ms"], 0)
        self.assertEqual(result, json.loads(self.output.read_text()))
        for stage in result["memory"].values():
            self.assertGreater(stage["resident_bytes"], 0)
            self.assertGreater(stage["peak_resident_bytes"], 0)
        for case in result["cases"]:
            self.assertEqual(len(case["samples_ms"]), 3)
            self.assertEqual(len(case["outcomes"]), 3)
            self.assertGreaterEqual(case["p95_ms"], case["p50_ms"])
            self.assertNotIn("state", case)
            self.assertTrue(case["stable_outcomes"])
        self.assertEqual([case["expect"] for case in result["cases"]], ["prediction", "review", "review"])
        self.assertTrue(benchmark.compare(result, result)["passed"])

    def test_different_hosts_may_differ_in_latency_but_must_preserve_decisions(self):
        left = self.measure()
        right = copy.deepcopy(left)
        right["host"]["system"] = "another OS"
        right["verified_load_ms"] *= 2
        for result in right["cases"][0]["outcomes"]:
            detail = result.get("prediction", result.get("suggestion"))
            detail["probabilities"][0] += 1e-8
            detail["probabilities"][1] -= 1e-8
        comparison = benchmark.compare(left, right)
        self.assertTrue(comparison["passed"])
        self.assertAlmostEqual(comparison["max_probability_difference"], 1e-8)
        right["cases"][0]["outcomes"][-1]["reason"] = "inference_unavailable"
        self.assertEqual(benchmark.compare(left, right)["mismatched_cases"], ["message"])

    def test_large_probability_drift_and_invalid_vectors_are_detected(self):
        left = self.measure()
        right = copy.deepcopy(left)
        result = right["cases"][0]["outcomes"][-1]
        detail = result.get("prediction", result.get("suggestion"))
        detail["probabilities"][0] += 1e-4
        detail["probabilities"][1] -= 1e-4
        self.assertFalse(benchmark.compare(left, right)["passed"])
        for value in (float("nan"), -1, True):
            detail["probabilities"][0] = value
            with self.assertRaisesRegex(ValueError, "probability vectors"):
                benchmark.compare(left, right)

    def test_incompatible_inputs_and_missing_repetitions_cannot_compare(self):
        left = self.measure()
        for field in ("bundle_manifest_sha256", "probe_sha256", "implementation_sha256", "runtime"):
            right = copy.deepcopy(left)
            right[field] = {"other": True} if field == "runtime" else "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "different"):
                benchmark.compare(left, right)
        right = copy.deepcopy(left)
        right["cases"][0]["outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "every repeated outcome"):
            benchmark.compare(left, right)
        right = copy.deepcopy(left)
        right["cases"][0]["case_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            benchmark.compare(left, right)

    def test_invalid_probe_and_guard_only_workload_fail_before_loading(self):
        with patch.object(benchmark, "load_bundle", side_effect=AssertionError("must not load")):
            for field, value in (("warmup", -1), ("repeats", 0), ("repeats", True)):
                original = self.probe[field]
                self.probe[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.measure()
                self.probe[field] = original
            self.probe["cases"] = self.probe["cases"][1:]
            with self.assertRaisesRegex(ValueError, "guard-only"):
                self.measure()
        self.assertFalse(self.output.exists())

    def test_outputs_are_external_and_never_overwrite_measurements(self):
        self.measure()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.measure()
        self.assertEqual(self.output.read_bytes(), before)
        self.output = self.fixture.output / "benchmark.json"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.measure()

    def test_review_only_bundle_is_not_reported_as_neural_performance(self):
        self.fixture.output = self.fixture.model.root / "review-only"
        self.receipt = self.fixture.publish(review_only=True, test_predictions=None)
        with self.assertRaisesRegex(ValueError, "review-only"):
            self.measure()
        self.assertFalse(self.output.exists())

    def test_inference_errors_or_wrong_expectations_cannot_look_like_fast_serving(self):
        with (patch("laya.adapt_bundle.forward_batch", side_effect=RuntimeError("inference failed")),
              self.assertRaisesRegex(ValueError, "inference_unavailable")):
            self.measure()
        self.probe["cases"][0]["state"] = ""
        with self.assertRaisesRegex(ValueError, "unexpected outcome"):
            self.measure()
        self.assertFalse(self.output.exists())

    def test_runtime_change_during_measurement_is_not_published(self):
        original = benchmark._runtime_contract()
        with (patch.object(benchmark, "_runtime_contract", side_effect=[original, {**original, "threads": 999}]),
              self.assertRaisesRegex(ValueError, "changed during measurement")):
            self.measure()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
