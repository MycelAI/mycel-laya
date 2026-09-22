"""Train, interrupt, resume and reload tiny local checkpoints without downloads."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from laya import load  # noqa: E402
from laya.adapt_data import write_dataset  # noqa: E402
from laya.adapt_model import checkpoint_files, collect_logits, load_local_checkpoint, prepare_input, prepare_items  # noqa: E402
from laya.adapt_train import TrainingConfig, _atomic_save, _run_lock, train  # noqa: E402
from laya.common import DecisionModel  # noqa: E402


class AdaptTrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "base"
        self.model_dir.mkdir()
        torch.manual_seed(123)
        cfg = BertConfig(vocab_size=24, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=1, intermediate_size=32, hidden_dropout_prob=.2)
        cfg.save_pretrained(self.model_dir / "encoder")
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "billing", "technical", "low", "high",
                      "true", "false", "charged", "error", "question", "choice", "score", "noul"]
        backend = Tokenizer(WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                            cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
        tokenizer.save_pretrained(self.model_dir / "tokenizer")
        model = DecisionModel(BertModel(cfg), head_layers=1)
        save_file(model.state_dict(), str(self.model_dir / "model.safetensors"))
        (self.model_dir / "rl_agent_config.json").write_text(json.dumps({
            "encoder": "unused/local", "head_layers": 1, "act_costs": {"act": 0},
            "max_len": 64, "head_max_len": 48, "temperature": [2, 3, 4],
            "temperature_by_options": {"choice:2": .5, "noul:2": 4},
        }), encoding="utf-8")
        self.questions = {
            "queue": {"type": "choice", "instructions": "Route", "criteria": ["billing", "technical"]},
            "urgency": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
            "incident": {"type": "noul", "instructions": "Is this an error?"},
        }
        self.rows = [{"id": "row:%d" % i, "group_id": "family:%d" % i, "language": "en",
                      "state": ("charged " if i % 2 else "error ") + str(i),
                      "targets": {"queue": "billing" if i % 2 else "technical", "urgency": i % 2,
                                  "incident": bool(i % 2)},
                      "source": {"dataset": "local-synthetic", "revision": "v1", "license": "CC0-1.0",
                                 "record_id": str(i)}} for i in range(20)]
        self.dataset = self.root / "dataset"
        self.manifest = write_dataset(self.dataset, self.rows, self.questions)
        self.settings = TrainingConfig(epochs=2, batch_size=2, grad_accum=2, learning_rate=.001,
                                       max_len=64, head_max_len=48, checkpoint_every=3)

    def run_training(self, name, **kwargs):
        return train(self.dataset, self.model_dir, self.root / name, config=kwargs.pop("config", self.settings),
                     log=lambda message: None, **kwargs)

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_interrupted_resume_matches_all_training_state_and_export(self):
        baseline_files = checkpoint_files(self.model_dir)
        full = self.run_training("full")
        stopped = self.run_training("resumed", max_updates=3)
        self.assertEqual(stopped["status"], "stopped")
        self.assertFalse((self.root / "resumed" / "export").exists())
        continued = self.run_training("resumed", resume=True)
        self.assertEqual(full, continued)
        first = torch.load(self.root / "full" / "checkpoint.pt", weights_only=True)
        second = torch.load(self.root / "resumed" / "checkpoint.pt", weights_only=True)
        self.assert_state_equal(first, second)
        self.assertEqual(checkpoint_files(self.root / "full" / "export"),
                         checkpoint_files(self.root / "resumed" / "export"))
        self.assertEqual(checkpoint_files(self.model_dir), baseline_files)
        self.assertEqual(self.run_training("resumed", resume=True), full)
        self.assertEqual(full["train_items"], 3 * self.manifest["counts"]["train"])

    def test_changed_weights_clear_calibration_and_reload_through_sdk(self):
        self.run_training("run", config=replace(self.settings, epochs=1))
        output = self.root / "run" / "export"
        cfg = json.loads((output / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["temperature"], [1, 1, 1])
        self.assertEqual(cfg["temperature_by_options"], {})
        self.assertEqual(cfg["adaptation"]["calibration_status"], "unfitted")
        initial = load_file(str(self.model_dir / "model.safetensors"))
        final = load_file(str(output / "model.safetensors"))
        changed = {key for key in initial if not torch.equal(initial[key], final[key])}
        self.assertTrue(any(key.startswith("scorer.") for key in changed))
        self.assertFalse(any(key.startswith(("encoder.", "act_head.")) for key in changed))
        local = load_local_checkpoint(output)
        items = prepare_items(local, self.rows[:1], self.questions)
        records = collect_logits(local, items, batch_size=2)
        agent = load(str(output), device="cpu")
        answers = agent.predict(self.rows[0]["state"], self.questions)["answers"]
        for record in records:
            probabilities = torch.softmax(torch.tensor(record["logits"]), -1).tolist()
            qid = record["question_id"]
            if qid == "incident":
                self.assertAlmostEqual(answers[qid]["noul"], probabilities[1], delta=5.1e-5)
            else:
                for actual, expected in zip(answers[qid]["probabilities"].values(), probabilities):
                    self.assertAlmostEqual(actual, expected, delta=5.1e-5)

    def test_training_consumes_only_training_partition(self):
        with patch("laya.adapt_train.prepare_items", wraps=prepare_items) as prepare:
            self.run_training("run", max_updates=1)
        used = prepare.call_args.args[1]
        self.assertEqual({row["id"] for row in used},
                         {key for key, split in self.manifest["assignments"].items() if split == "train"})

    def test_rendering_rejects_truncated_state_options_and_collisions(self):
        checkpoint = load_local_checkpoint(self.model_dir)
        with self.assertRaisesRegex(ValueError, "state is truncated"):
            prepare_input(checkpoint, "charged " * 100, self.questions["queue"])
        with self.assertRaisesRegex(ValueError, "instructions are truncated"):
            prepare_input(checkpoint, "charged", {**self.questions["queue"], "instructions": "question " * 100})
        for criteria in (["missing-a", "missing-b"], ["billing " * 50, "technical"]):
            question = {**self.questions["queue"], "criteria": criteria}
            with self.assertRaisesRegex(ValueError, "option text"):
                prepare_input(checkpoint, "charged", question)

    def test_full_encoder_training_is_supported(self):
        self.run_training("all", config=replace(self.settings, train_encoder=True), max_updates=1)
        state = torch.load(self.root / "all" / "checkpoint.pt", weights_only=True)
        original = load_file(str(self.model_dir / "model.safetensors"))
        self.assertTrue(any(not torch.equal(value, state["model"][key]) for key, value in original.items()
                            if key.startswith("encoder.")))

    def test_unequal_microbatches_match_single_batch_gradient(self):
        cfg_path = self.model_dir / "rl_agent_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["head_layers"] = 0  # No trainable dropout, so batch partitioning is comparable.
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        weights_path = self.model_dir / "model.safetensors"
        save_file({key: value for key, value in load_file(str(weights_path)).items() if not key.startswith("head.")},
                  str(self.model_dir / "new.safetensors"))
        os.replace(self.model_dir / "new.safetensors", weights_path)
        count = self.manifest["counts"]["train"] * 3
        single = replace(self.settings, epochs=1, batch_size=count + 1, grad_accum=1, max_grad_norm=1000000)
        micro = replace(single, batch_size=1, grad_accum=count + 1)
        self.run_training("single", config=single)
        self.run_training("micro", config=micro)
        first = torch.load(self.root / "single" / "checkpoint.pt", weights_only=True)
        second = torch.load(self.root / "micro" / "checkpoint.pt", weights_only=True)
        self.assertEqual(first["progress"]["updates"], 1)
        for key, value in first["optimizer"]["state"].items():
            torch.testing.assert_close(value["exp_avg"], second["optimizer"]["state"][key]["exp_avg"],
                                       atol=1e-7, rtol=1e-4)

    def test_resume_rejects_changed_settings_data_and_assets(self):
        self.run_training("run", max_updates=1)
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run_training("run", resume=True, config=replace(self.settings, grad_accum=3))
        cfg_path = self.model_dir / "rl_agent_config.json"
        cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.run_training("run", resume=True)
        with self.assertRaises(FileExistsError):
            self.run_training("run")
        with self.assertRaises(FileNotFoundError):
            self.run_training("missing", resume=True)

    def test_resume_rejects_changed_dataset_and_tokenization(self):
        self.run_training("run", max_updates=1)

        def changed_items(*args):
            items = prepare_items(*args)
            items[0]["ids"][-1] = 1
            return items

        with patch("laya.adapt_train.prepare_items", side_effect=changed_items):
            with self.assertRaisesRegex(ValueError, "tokenized"):
                self.run_training("run", resume=True)
        changed = self.root / "changed-data"
        write_dataset(changed, self.rows, self.questions, seed="different-split")
        with self.assertRaisesRegex(ValueError, "differs"):
            train(changed, self.model_dir, self.root / "run", config=self.settings, resume=True)

    def test_resume_rejects_changed_implementation(self):
        self.run_training("run", max_updates=1)
        with patch("laya.adapt_train.file_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "implementation"):
                self.run_training("run", resume=True)

    def test_resume_rejects_inconsistent_cursor_and_scheduler(self):
        self.run_training("run", max_updates=1)
        path = self.root / "run" / "checkpoint.pt"
        state = torch.load(path, weights_only=True)
        state["progress"]["cursor"] += 1
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "optimizer boundary"):
            self.run_training("run", resume=True)

    def test_failed_atomic_save_preserves_last_checkpoint(self):
        path = self.root / "state.pt"
        _atomic_save({"old": 1}, path, tensor=True)
        original = path.read_bytes()
        with patch("laya.adapt_train.torch.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                _atomic_save({"new": 2}, path, tensor=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.root.glob(".checkpoint-*")))

    def test_process_death_releases_run_lock(self):
        script = ("from laya.adapt_train import _run_lock; import sys; "
                  "lock = _run_lock(sys.argv[1]); lock.__enter__(); print('locked', flush=True); input()")
        process = subprocess.Popen([sys.executable, "-c", script, str(self.root)], cwd=ROOT,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(RuntimeError, "holds"):
                with _run_lock(self.root):
                    pass
        finally:
            process.terminate()
            process.communicate(timeout=20)
        with _run_lock(self.root):
            pass


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
