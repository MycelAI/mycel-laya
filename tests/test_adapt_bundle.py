"""Synthetic bundle integrity, SDK compatibility and selective runtime tests."""
import copy
import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_adapt_policy as policy_fixtures
import test_adapt_train as model_fixtures
import torch

from laya import load
from laya.adapt_bundle import (
    _runtime_contract,
    export_bundle,
    load_bundle,
)
from laya.adapt_calibration import fit_calibration
from laya.adapt_data import (
    fingerprint,
    question_fingerprint,
    read_dataset,
    write_dataset,
)
from laya.adapt_model import checkpoint_files, file_sha256
from laya.adapt_policy import select_bound_policy


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.data = policy_fixtures.BoundPolicyTests()
        self.data.setUp()
        self.addCleanup(self.data.doCleanups)
        self.model = model_fixtures.AdaptTrainTests()
        self.model.setUp()
        self.addCleanup(self.model.doCleanups)
        contract = _runtime_contract()
        files = checkpoint_files(self.model.model_dir)
        for artifact in self.data.artifacts.values():
            artifact["binding"]["model_files"] = files
            # Controlled synthetic evidence, not a model-quality experiment.
            artifact["collection"] = {"runtime": {key: value for key, value in contract.items() if key != "implementation"},
                                      "implementation": contract["implementation"]}
        self.calibration = fit_calibration(self.data.artifacts["calibration"], self.data.questions)
        self.policy = self.select()
        self.output = self.model.root / "bundle"

    def select(self):
        return select_bound_policy(self.data.artifacts["policy"], self.calibration, self.data.directory,
                                   question_id="queue", required_languages=["en", "ar"], thresholds=[.9])

    def publish(self, **changes):
        arguments = {"dataset_dir": self.data.directory, "model_dir": self.model.model_dir,
                     "calibration_predictions": self.data.artifacts["calibration"],
                     "policy_predictions": self.data.artifacts["policy"], "calibration": self.calibration,
                     "policy": self.policy, "test_predictions": self.data.artifacts["test"]}
        arguments.update(changes)
        return export_bundle(self.output, **arguments)

    def load(self, receipt):
        return load_bundle(self.output, expected_manifest_sha256=receipt["manifest_sha256"])

    def update_manifest(self, relative):
        path = self.output / "bundle.json"
        manifest = json.loads(path.read_text())
        manifest["files"][relative] = file_sha256(self.output / relative)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return {"manifest_sha256": file_sha256(path)}

    def test_roundtrip_matches_sdk_and_preserves_source(self):
        original = checkpoint_files(self.model.model_dir)
        receipt = self.publish()
        bundle = self.load(receipt)
        actual = bundle.predict("charged error", language="en")
        prediction = actual.get("prediction", actual.get("suggestion"))
        self.assertIsNotNone(prediction)
        agent = load(str(self.output / "model"), device="cpu")
        ordinary = agent.predict("charged error", self.data.questions)["answers"]["queue"]
        for left, right in zip(prediction["probabilities"], ordinary["probabilities"].values()):
            self.assertAlmostEqual(left, right, delta=5.1e-5)
        self.assertEqual(checkpoint_files(self.model.model_dir), original)
        config = json.loads((self.output / "model/rl_agent_config.json").read_text())
        self.assertEqual(config["temperature"], self.calibration["temperature"])
        self.assertEqual(config["temperature_by_options"], {})
        self.assertFalse(list(self.output.rglob("records.jsonl")))
        self.load(receipt)  # Ordinary SDK loading did not change the verified files.
        questions = bundle.questions
        questions["queue"]["criteria"].reverse()
        self.assertEqual(bundle.questions, self.data.questions)

    def test_unrounded_threshold_equality_and_review_value(self):
        bundle = self.load(self.publish())
        with patch("laya.adapt_bundle.calibrated_probabilities", return_value=[.9, .1]):
            result = bundle.predict("charged", language="en")
        self.assertEqual(result["status"], "automate")
        self.assertEqual(result["value"], "billing")
        below = math.nextafter(.9, 0)
        with patch("laya.adapt_bundle.calibrated_probabilities", return_value=[below, 1 - below]):
            result = bundle.predict("charged", language="ar")
        self.assertEqual(result["reason"], "below_threshold")
        self.assertIsNone(result["value"])
        self.assertEqual(result["suggestion"]["value"], "billing")

    def test_unsupported_inputs_review_without_a_forward_pass(self):
        bundle = self.load(self.publish())
        cycle = []
        cycle.append(cycle)
        with patch("laya.adapt_bundle.forward_batch", side_effect=AssertionError("must not infer")):
            self.assertEqual(bundle.predict("charged", language="de")["reason"], "unsupported_language")
            for state in ("", " ", {}, [], {"message": ""}, None, 42, {"x": float("nan")},
                          {1: "not a JSON string key"}, cycle, "charged " * 1000):
                with self.subTest(state_type=type(state).__name__):
                    result = bundle.predict(state, language="en")
                    self.assertEqual(result["status"], "review")
                    self.assertIsNone(result["value"])

    def test_inference_failure_and_nonfinite_logits_review(self):
        bundle = self.load(self.publish())
        with patch("laya.adapt_bundle.forward_batch", side_effect=RuntimeError("device error")):
            self.assertEqual(bundle.predict("charged", language="en")["reason"], "inference_unavailable")
        with patch("laya.adapt_bundle.forward_batch", return_value=(torch.tensor([[float("nan"), 0]]), {})):
            self.assertEqual(bundle.predict("charged", language="en")["reason"], "non_finite_prediction")

    def test_files_missing_changed_or_added_fail_before_model_loading(self):
        receipt = self.publish()
        candidates = ["model/model.safetensors", "model/tokenizer/tokenizer.json", "model/rl_agent_config.json",
                      "calibration.json", "policy.json", "evaluation.json", "questions.json"]
        with patch("laya.adapt_bundle.load_local_checkpoint", side_effect=AssertionError("must not load")):
            for relative in candidates:
                path = self.output / relative
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.subTest(path=relative), self.assertRaisesRegex(ValueError, "changed"):
                    self.load(receipt)
                path.write_bytes(original)
            extra = self.output / "model/tokenizer/unexpected.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "added"):
                self.load(receipt)
            extra.unlink()
            path = self.output / "questions.json"
            path.rename(self.output / "questions.missing")
            with self.assertRaisesRegex(ValueError, "missing"):
                self.load(receipt)

    def test_config_and_schema_cannot_be_rebound_to_old_evidence(self):
        self.publish()
        path = self.output / "model/rl_agent_config.json"
        config = json.loads(path.read_text())
        config["temperature_by_options"] = {"choice:2": 4}
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "SDK model configuration"):
            self.load(self.update_manifest("model/rl_agent_config.json"))
        path = self.output / "questions.json"
        questions = json.loads(path.read_text())
        questions["queue"]["criteria"].reverse()
        path.write_text(json.dumps(questions))
        with self.assertRaisesRegex(ValueError, "schema differs"):
            self.load(self.update_manifest("questions.json"))

    def test_changed_weights_cannot_reuse_calibration_with_a_new_manifest(self):
        self.publish()
        path = self.output / "model/model.safetensors"
        path.write_bytes(path.read_bytes() + b" ")
        with (patch("laya.adapt_bundle.load_local_checkpoint", side_effect=AssertionError("must not load")),
              self.assertRaisesRegex(ValueError, "calibrated source")):
            self.load(self.update_manifest("model/model.safetensors"))

    def test_unrelated_model_and_tokenizer_config_changes_cannot_reuse_evidence(self):
        self.publish()
        for name, field in (("rl_agent_config.json", "context_scale"), ("tokenizer/tokenizer_config.json", "do_lower_case")):
            relative = "model/" + name
            path = self.output / relative
            original = path.read_bytes()
            config = json.loads(original)
            config[field] = "unmeasured change"
            path.write_text(json.dumps(config))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "configuration differs"):
                self.load(self.update_manifest(relative))
            path.write_bytes(original)
            self.update_manifest(relative)

    def test_publication_receipt_survives_interruption_before_directory_rename(self):
        receipt_path = self.model.root / "release.json"
        with (patch("laya.adapt_bundle.Path.rename", side_effect=OSError("interrupted publication")),
              self.assertRaisesRegex(OSError, "interrupted publication")):
            self.publish(receipt_path=receipt_path)
        self.assertFalse(self.output.exists())
        recorded = json.loads(receipt_path.read_text())
        self.assertEqual(self.publish(receipt_path=receipt_path), recorded)
        self.load(recorded)

    def test_receipt_must_be_external_and_cannot_replace_another_release(self):
        with self.assertRaisesRegex(ValueError, "outside the bundle"):
            self.publish(receipt_path=self.output / "receipt.json")
        path = self.model.root / "other-receipt.json"
        path.write_text('{"existing": true}')
        with self.assertRaisesRegex(ValueError, "different bundle"):
            self.publish(receipt_path=path)
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(path.read_text()), {"existing": True})

    def test_trusted_manifest_and_runtime_are_required(self):
        receipt = self.publish()
        for checksum in (None, "", "a" * 64):
            with self.assertRaises(ValueError):
                load_bundle(self.output, expected_manifest_sha256=checksum)
        with self.assertRaisesRegex(ValueError, "CPU"):
            load_bundle(self.output, expected_manifest_sha256=receipt["manifest_sha256"], device="cuda")
        original = _runtime_contract()
        with (patch("laya.adapt_bundle._runtime_contract", return_value={**original, "torch": "different"}),
              self.assertRaisesRegex(ValueError, "runtime")):
            self.load(receipt)

    def test_missing_or_failed_final_test_cannot_publish_automatic_bundle(self):
        with self.assertRaisesRegex(ValueError, "independent final"):
            self.publish(test_predictions=None)
        self.assertFalse(self.output.exists())
        bad = copy.deepcopy(self.data.artifacts["test"])
        for row in bad["records"]:
            row["logits"] = [0.0, 2.0]
        bad["records_sha256"] = fingerprint(bad["records"])
        with self.assertRaisesRegex(ValueError, "independent final"):
            self.publish(test_predictions=bad)
        self.assertFalse(self.output.exists())

    def test_failed_policy_can_publish_only_review_without_test_or_model_load(self):
        for row in self.data.artifacts["policy"]["records"]:
            row["logits"] = [0.0, 2.0]
        self.data.artifacts["policy"]["records_sha256"] = fingerprint(self.data.artifacts["policy"]["records"])
        self.policy = self.select()
        self.assertFalse(self.policy["selection"]["passed"])
        receipt = self.publish(test_predictions=None, review_only=True)
        with patch("laya.adapt_bundle.load_local_checkpoint", side_effect=AssertionError("must not load")):
            bundle = self.load(receipt)
        self.assertEqual(bundle.predict("charged", language="en")["reason"], "policy_not_qualified")

    def test_changed_checkpoint_and_calibration_are_rejected_at_export(self):
        altered = copy.deepcopy(self.calibration)
        altered["temperature_by_options"] = {"choice:2": 4}
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            self.publish(calibration=altered)
        path = self.model.model_dir / "rl_agent_config.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checkpoint files|Checkpoint files"):
            self.publish()

    def test_new_bucket_fit_is_persisted_and_atomic_failure_leaves_no_bundle(self):
        self.calibration = fit_calibration(self.data.artifacts["calibration"], self.data.questions,
                                           fit_buckets=True, min_per_bucket=20)
        self.policy = self.select()
        with (patch("laya.adapt_bundle.shutil.copyfile", side_effect=OSError("disk full")),
              self.assertRaisesRegex(OSError, "disk full")):
            self.publish()
        self.assertFalse(self.output.exists())
        receipt = self.publish()
        self.load(receipt)
        config = json.loads((self.output / "model/rl_agent_config.json").read_text())
        self.assertEqual(config["temperature_by_options"], {"choice:2": self.calibration["temperature_by_options"]["choice:2"]})
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_false_qualified_summary_is_rejected_even_with_updated_file_hash(self):
        self.publish()
        path = self.output / "evaluation.json"
        report = json.loads(path.read_text())
        report["evaluation"]["by_language"]["en"]["errors"] = 200
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "uncertainty bounds"):
            self.load(self.update_manifest("evaluation.json"))

    def test_score_uses_selected_level_and_noul_uses_boolean(self):
        for kind, qtype, criteria, probabilities, expected in (
            ("score", 1, ["low", "medium", "high"], [.01, .02, .97], 2),
            ("noul", 2, None, [.05, .95], True),
        ):
            with self.subTest(kind=kind):
                _, rows, old_manifest = read_dataset(self.data.directory)
                questions = {"decision": {"type": kind, "instructions": "Rate"}}
                if criteria is not None:
                    questions["decision"]["criteria"] = criteria
                for row in rows:
                    row["targets"] = {"decision": False if kind == "noul" else 0}
                directory = self.model.root / ("data-" + kind)
                manifest = write_dataset(directory, rows, questions, fixed_splits=old_manifest["assignments"])
                self.data.directory, self.data.questions = directory, questions
                for artifact in self.data.artifacts.values():
                    artifact["binding"].update(dataset_manifest_sha256=fingerprint(manifest),
                                               questions_sha256=question_fingerprint(questions))
                    for row in artifact["records"]:
                        row.update(question_id="decision", qtype=qtype, logits=[2.0] + [0.0] * (len(probabilities) - 1))
                    artifact["records_sha256"] = fingerprint(artifact["records"])
                self.calibration = fit_calibration(self.data.artifacts["calibration"], questions)
                self.policy = select_bound_policy(self.data.artifacts["policy"], self.calibration, directory,
                                                  question_id="decision", required_languages=["en", "ar"], thresholds=[.9])
                self.output = self.model.root / ("bundle-" + kind)
                bundle = self.load(self.publish())
                with patch("laya.adapt_bundle.calibrated_probabilities", return_value=probabilities):
                    result = bundle.predict("charged", language="en")
                self.assertEqual(result["status"], "automate")
                self.assertIs(type(result["value"]), type(expected))
                self.assertEqual(result["value"], expected)
                if kind == "score":
                    self.assertAlmostEqual(result["prediction"]["expected_score"], 1.96)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
