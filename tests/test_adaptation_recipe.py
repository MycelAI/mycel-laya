"""Synthetic end-to-end candidate recipe tests; final test must remain unused."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from laya.adapt_data import fingerprint, write_dataset  # noqa: E402
from laya.adapt_model import checkpoint_files  # noqa: E402
import test_adapt_train as fixtures  # noqa: E402

spec = importlib.util.spec_from_file_location("adaptation_recipe", ROOT / "research/scripts/evaluate_adaptation_candidate.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdaptTrainTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        rows, fixed = [], {}
        for i in range(100):
            row = copy.deepcopy(f.rows[i % len(f.rows)])
            row.update(id="r%d" % i, group_id="g%d" % i, state="charged error %d" % i)
            row["source"]["record_id"] = str(i)
            row["targets"] = {"queue": row["targets"]["queue"]}
            rows.append(row)
            fixed[row["id"]] = "train" if i < 30 else "calibration" if i < 60 else "policy" if i < 90 else "test"
        self.dataset = f.root / "recipe-data"
        self.questions = {"queue": f.questions["queue"]}
        manifest = write_dataset(self.dataset, rows, self.questions, fixed_splits=fixed)
        self.protocol = json.loads((ROOT / "research/arbanking77_protocol.json").read_text())
        self.protocol.update(dataset_manifest_sha256=fingerprint(manifest), question_id="queue", n_options=2,
                             required_languages=["en"], base_model_files=checkpoint_files(f.model_dir),
                             rendering={"max_len": 64, "head_max_len": 48},
                             candidates=[{"id": "base", "epochs": 0, "train_encoder": False}])

    def test_full_recipe_uses_only_calibration_and_policy_and_resumes(self):
        f = self.fixture
        with patch.object(recipe, "collect_partition", wraps=recipe.collect_partition) as collect:
            result = recipe.evaluate(self.protocol, "base", self.dataset, f.model_dir, f.root / "evaluation",
                                     log=lambda value: None)
        self.assertEqual([call.kwargs["split"] for call in collect.call_args_list], ["calibration", "policy"])
        self.assertFalse(result["final_test_evaluated"])
        self.assertFalse(result["selection"]["passed"])
        self.assertEqual(result["metrics"]["training_prior"]["questions"]["queue"]["all"]["accuracy"], .5)
        self.assertEqual(result["metrics"]["raw"]["questions"]["queue"]["all"]["samples"], 30)
        again = recipe.evaluate(self.protocol, "base", self.dataset, f.model_dir, f.root / "evaluation", resume=True,
                                log=lambda value: None)
        self.assertEqual(result, again)

    def test_wrong_candidate_model_and_error_allocation_are_rejected(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, "predeclared candidate"):
            recipe.validate_candidate(self.protocol, "undeclared", self.dataset, f.model_dir)
        wrong = copy.deepcopy(self.protocol)
        wrong["base_model_files"]["model.safetensors"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "pinned"):
            recipe.validate_candidate(wrong, "base", self.dataset, f.model_dir)
        wrong = copy.deepcopy(self.protocol)
        wrong["policy"]["alpha_per_candidate"] = .05
        with self.assertRaisesRegex(ValueError, "allocate"):
            recipe.validate_candidate(wrong, "base", self.dataset, f.model_dir)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
