"""Synthetic policy artifact links; no pretrained inference or quality claims."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.adapt_calibration import fit_calibration  # noqa: E402
from laya.adapt_data import fingerprint, question_fingerprint, write_dataset  # noqa: E402
from laya.adapt_policy import evaluate_bound_policy, select_bound_policy, validate_bound_policy  # noqa: E402


class BoundPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "dataset"
        self.questions = {"queue": {"type": "choice", "instructions": "Route", "criteria": ["billing", "technical"]}}
        rows, fixed = [], {}
        for role, count in (("train", 2), ("calibration", 30), ("policy", 200), ("test", 200)):
            for i in range(count):
                for lang in ("en", "ar"):
                    group, key = "%s:%d" % (role, i), "%s:%d:%s" % (role, i, lang)
                    rows.append({"id": key, "group_id": group, "language": lang, "state": key,
                                 "targets": {"queue": "billing"},
                                 "source": {"dataset": "synthetic", "revision": "v1", "license": "CC0-1.0", "record_id": group}})
                    fixed[key] = role
        manifest = write_dataset(self.directory, rows, self.questions, fixed_splits=fixed)
        self.artifacts = {}
        for role in ("calibration", "policy", "test"):
            records = [{"id": row["id"], "group_id": manifest["groups"][row["id"]], "language": row["language"],
                        "question_id": "queue", "qtype": 0, "label": 0, "logits": [2.0, 0.0]}
                       for row in rows if manifest["assignments"][row["id"]] == role]
            self.artifacts[role] = {"schema_version": 1, "stage": "logits", "split": role,
                "binding": {"dataset_manifest_sha256": fingerprint(manifest),
                            "questions_sha256": question_fingerprint(self.questions),
                            "model_files": {"model.safetensors": "a" * 64},
                            "rendering": {"max_len": 64, "head_max_len": 48}},
                "records": records, "records_sha256": fingerprint(records)}
        self.calibration = fit_calibration(self.artifacts["calibration"], self.questions)

    def select(self, predictions=None, **kwargs):
        return select_bound_policy(predictions or self.artifacts["policy"], self.calibration, self.directory,
                                   question_id="queue", required_languages=["en", "ar"], thresholds=[0, .9], **kwargs)

    def test_bound_selection_and_fresh_final_test(self):
        policy = self.select(declaration={"protocol_sha256": "b" * 64})
        self.assertTrue(policy["selection"]["passed"])
        report = evaluate_bound_policy(self.artifacts["test"], policy, self.calibration, self.directory)
        self.assertTrue(report["evaluation"]["passed"])
        self.assertEqual(report["policy_sha256"], fingerprint(policy))
        self.assertEqual(report["evaluation"]["selection_sha256"], fingerprint(policy["selection"]))
        for result in report["evaluation"]["by_language"].values():
            self.assertEqual(result["samples"], 200)
            self.assertLessEqual(result["error_upper"], .05)
            self.assertGreaterEqual(result["coverage_lower"], .5)

    def test_roles_and_changed_links_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "policy partition"):
            self.select(self.artifacts["test"])
        policy = self.select()
        with self.assertRaisesRegex(ValueError, "test partition"):
            evaluate_bound_policy(self.artifacts["policy"], policy, self.calibration, self.directory)
        altered = copy.deepcopy(self.calibration)
        altered["temperature"][0] = 2
        with self.assertRaisesRegex(ValueError, "calibration"):
            evaluate_bound_policy(self.artifacts["test"], policy, altered, self.directory)
        altered = copy.deepcopy(self.artifacts["test"])
        altered["binding"]["model_files"]["model.safetensors"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            evaluate_bound_policy(altered, policy, self.calibration, self.directory)

    def test_failed_selection_cannot_trigger_final_evaluation(self):
        bad = copy.deepcopy(self.artifacts["policy"])
        for row in bad["records"]:
            row["logits"] = [0.0, 2.0]
        bad["records_sha256"] = fingerprint(bad["records"])
        policy = self.select(bad)
        self.assertFalse(policy["selection"]["passed"])
        with self.assertRaisesRegex(ValueError, "did not qualify"):
            evaluate_bound_policy(self.artifacts["test"], policy, self.calibration, self.directory)

    def test_correct_option_count_and_calibration_binding_required(self):
        policy = self.select()
        policy["selection"]["n_options"] = 3
        with self.assertRaisesRegex(ValueError, "option schema"):
            validate_bound_policy(policy, self.calibration, self.questions)
        policy = self.select()
        policy["binding"]["rendering"]["head_max_len"] = 32
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_bound_policy(policy, self.calibration, self.questions)
        policy["binding"] = None
        with self.assertRaisesRegex(ValueError, "binding"):
            validate_bound_policy(policy, self.calibration, self.questions)


if __name__ == "__main__":
    unittest.main()
