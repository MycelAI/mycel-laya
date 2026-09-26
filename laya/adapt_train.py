"""Resumable single-process, fp32 supervised adaptation of local Laya checkpoints."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import torch
import tokenizers
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

from .adapt_data import fingerprint, read_dataset
from .adapt_model import checkpoint_files, export_uncalibrated, file_sha256, forward_batch, load_local_checkpoint, prepare_items


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    train_encoder: bool = False
    seed: int = 42
    checkpoint_every: int = 50
    max_len: int = 512
    head_max_len: int = 192

    def validate(self):
        for name in ("epochs", "batch_size", "grad_accum", "checkpoint_every", "max_len", "head_max_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("learning_rate", "encoder_learning_rate", "max_grad_norm", "weight_decay"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                    or value < 0 or (name != "weight_decay" and value == 0)):
                raise ValueError("%s has an invalid training value" % name)
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.train_encoder) is not bool or self.head_max_len > self.max_len:
            raise ValueError("train_encoder must be boolean and head_max_len must not exceed max_len")


@contextmanager
def _run_lock(directory):
    # Kernel locks release on process death; a stale file is never proof of a live job.
    with (Path(directory) / ".run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another process holds the training run lock") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_save(value, destination):
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def _encode_checkpoint(value, tensors):
    """Store tensor values separately from a strictly tagged JSON structure."""
    if isinstance(value, torch.Tensor):
        name = "tensor_%08d" % len(tensors)
        tensors[name] = value.detach().cpu().contiguous()
        return {"type": "tensor", "name": name}
    if isinstance(value, np.generic):
        return _encode_checkpoint(value.item(), tensors)
    if isinstance(value, dict):
        return {"type": "dict", "items": [[_encode_checkpoint(key, tensors), _encode_checkpoint(item, tensors)]
                                          for key, item in value.items()]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode_checkpoint(item, tensors) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode_checkpoint(item, tensors) for item in value]}
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    raise ValueError("unsupported training state value: %s" % type(value).__name__)


def _decode_checkpoint(value, tensors, used, *, depth=0):
    if depth > 64 or type(value) is not dict or type(value.get("type")) is not str:
        raise ValueError("invalid checkpoint structure")
    kind = value["type"]
    if kind == "tensor":
        if set(value) != {"type", "name"} or type(value["name"]) is not str or value["name"] in used:
            raise ValueError("invalid checkpoint tensor reference")
        used.add(value["name"])
        if value["name"] not in tensors:
            raise ValueError("missing checkpoint tensor")
        return tensors[value["name"]]
    if kind in ("dict", "list", "tuple"):
        if set(value) != {"type", "items"} or type(value["items"]) is not list:
            raise ValueError("invalid checkpoint container")
        if kind == "dict":
            result = {}
            for pair in value["items"]:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError("invalid checkpoint mapping")
                key = _decode_checkpoint(pair[0], tensors, used, depth=depth + 1)
                if key in result:
                    raise ValueError("duplicate checkpoint mapping key")
                result[key] = _decode_checkpoint(pair[1], tensors, used, depth=depth + 1)
            return result
        items = [_decode_checkpoint(item, tensors, used, depth=depth + 1) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "null":
        if set(value) != {"type"}:
            raise ValueError("invalid checkpoint null")
        return None
    if set(value) != {"type", "value"}:
        raise ValueError("invalid checkpoint scalar")
    scalar = value["value"]
    if (kind == "bool" and type(scalar) is bool) or (kind == "int" and type(scalar) is int) or (
            kind == "float" and type(scalar) is float and math.isfinite(scalar)) or (
            kind == "str" and type(scalar) is str):
        return scalar
    raise ValueError("invalid checkpoint scalar")


def _save_checkpoint(state, destination):
    destination = Path(destination)
    tensors = {}
    structure = _encode_checkpoint(state, tensors)
    if not tensors:
        raise ValueError("checkpoint has no tensors")
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".safetensors", dir=destination.parent)
    os.close(fd)
    try:
        save_file(tensors, name, metadata={"format_version": "2",
                                            "structure": json.dumps(structure, allow_nan=False, separators=(",", ":"))})
        with open(name, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def _load_checkpoint(path):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format_version") != "2" or type(metadata.get("structure")) is not str:
            raise ValueError("unsupported training checkpoint format")
        if len(metadata["structure"]) > 4_000_000:
            raise ValueError("checkpoint structure is too large")
        structure = json.loads(metadata["structure"])
        tensors = {name: checkpoint.get_tensor(name).clone() for name in checkpoint.keys()}
    used = set()
    state = _decode_checkpoint(structure, tensors, used)
    if used != set(tensors) or type(state) is not dict:
        raise ValueError("checkpoint tensor inventory differs")
    return state


def _rng_state(device):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def _validate_progress(progress, count, effective_batch, config, scheduler):
    for field in ("epoch", "cursor", "updates", "examples_seen"):
        if type(progress.get(field)) is not int or progress[field] < 0:
            raise ValueError("invalid training progress: %s" % field)
    epoch, cursor = progress["epoch"], progress["cursor"]
    expected_updates = epoch * math.ceil(count / effective_batch) + cursor // effective_batch
    if (epoch > config.epochs or cursor >= count or cursor % effective_batch
            or (epoch == config.epochs and cursor) or progress["updates"] != expected_updates
            or progress["examples_seen"] != epoch * count + cursor
            or scheduler.last_epoch != expected_updates
            or not isinstance(progress.get("loss_sum"), (int, float))
            or not math.isfinite(progress["loss_sum"]) or progress["loss_sum"] < 0):
        raise ValueError("training progress and scheduler do not describe an optimizer boundary")


def train(dataset_dir, model_dir, run_dir, *, config=None, device="cpu", resume=False, max_updates=None, log=print):
    """Fit only the training partition, checkpointing at optimizer boundaries.

    max_updates limits additional updates in this invocation; it does not change
    the planned schedule. Resume requires the same data, assets, settings and
    recorded runtime. Interrupted writes preserve the previous checkpoint.
    """
    config = config or TrainingConfig()
    config.validate()
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer")
    run_dir = Path(run_dir)
    if resume and not (run_dir / "checkpoint.safetensors").is_file():
        raise FileNotFoundError("resume requires an existing checkpoint.safetensors")
    if not resume and run_dir.exists():
        raise FileExistsError("run directory already exists; use resume or choose a new directory")
    questions, rows, manifest = read_dataset(dataset_dir)
    train_rows = [row for row in rows if manifest["assignments"][row["id"]] == "train"]
    if not train_rows:
        raise ValueError("training partition has no labelled records")
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
               "tokenizers": tokenizers.__version__,
               "transformers": transformers.__version__, "device": str(torch.device(device)),
               "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled()}
    implementation = {name: file_sha256(Path(__file__).with_name(name)) for name in
                      ("adapt_train.py", "adapt_model.py", "adapt_data.py", "common.py", "agent.py")}
    identity = {"schema_version": 1, "dataset_manifest_sha256": fingerprint(manifest),
                "source_files": checkpoint_files(model_dir), "settings": asdict(config), "runtime": runtime,
                "implementation": implementation}
    if resume:
        recorded = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if fingerprint({key: value for key, value in recorded.items() if key != "items_sha256"}) != fingerprint(identity):
            raise ValueError("resume data, checkpoint, settings, implementation or runtime differs from the frozen run")
    else:
        run_dir.mkdir(parents=True)
    with _run_lock(run_dir):
        checkpoint = load_local_checkpoint(model_dir, device=device)
        checkpoint.config.update(max_len=config.max_len, head_max_len=config.head_max_len)
        items = prepare_items(checkpoint, train_rows, questions)
        identity["items_sha256"] = fingerprint(items)
        if resume and fingerprint(recorded) != fingerprint(identity):
            raise ValueError("tokenized training items differ from the frozen run")
        model = checkpoint.model
        encoder, head = [], []
        for name, parameter in model.named_parameters():
            enabled = not name.startswith("act_head.") and (config.train_encoder or not name.startswith("encoder."))
            parameter.requires_grad_(enabled)
            if enabled:
                (encoder if name.startswith("encoder.") else head).append(parameter)
        groups = [{"params": head, "lr": config.learning_rate}]
        if encoder:
            groups.append({"params": encoder, "lr": config.encoder_learning_rate})
        optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
        effective_batch = config.batch_size * config.grad_accum
        updates_per_epoch = math.ceil(len(items) / effective_batch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates_per_epoch * config.epochs)
        progress = {"epoch": 0, "cursor": 0, "updates": 0, "examples_seen": 0, "loss_sum": 0.0}
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if checkpoint.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        if resume:
            state = _load_checkpoint(run_dir / "checkpoint.safetensors")
            if (type(state.get("format_version")) is not int or state["format_version"] != 2
                    or fingerprint(state["identity"]) != fingerprint(identity)):
                raise ValueError("training checkpoint does not match the frozen run identity")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            progress = state["progress"]
            _validate_progress(progress, len(items), effective_batch, config, scheduler)
            _restore_rng(state["rng"], checkpoint.device)
            del state
        else:
            _atomic_save(identity, run_dir / "run.json")

        def save():
            _save_checkpoint({"format_version": 2, "identity": identity, "model": model.state_dict(),
                              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                              "rng": _rng_state(checkpoint.device), "progress": progress},
                             run_dir / "checkpoint.safetensors")

        if not resume:
            save()
        invocation_updates = 0
        while progress["epoch"] < config.epochs:
            order = list(range(len(items)))
            random.Random(config.seed + progress["epoch"]).shuffle(order)
            model.train()
            if not config.train_encoder:
                model.encoder.eval()
            while progress["cursor"] < len(items):
                indices = order[progress["cursor"]:progress["cursor"] + effective_batch]
                optimizer.zero_grad(set_to_none=True)
                update_loss = 0.0
                for start in range(0, len(indices), config.batch_size):
                    chunk = [items[index] for index in indices[start:start + config.batch_size]]
                    logits, batch = forward_batch(checkpoint, chunk)
                    # Sum / actual update size handles unequal and final microbatches.
                    loss = torch.nn.functional.cross_entropy(logits, batch["label"].to(checkpoint.device), reduction="sum")
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss; resume from the previous checkpoint")
                    (loss / len(indices)).backward()
                    update_loss += float(loss.detach().cpu())
                torch.nn.utils.clip_grad_norm_(encoder + head, config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                progress["cursor"] += len(indices)
                progress["updates"] += 1
                progress["examples_seen"] += len(indices)
                progress["loss_sum"] += update_loss
                invocation_updates += 1
                finished_epoch = progress["cursor"] == len(items)
                if finished_epoch:
                    progress["epoch"] += 1
                    progress["cursor"] = 0
                stopping = max_updates is not None and invocation_updates >= max_updates
                if progress["updates"] % config.checkpoint_every == 0 or finished_epoch or stopping:
                    save()
                    log(json.dumps({"updates": progress["updates"], "epoch": progress["epoch"],
                                    "loss": update_loss / len(indices)}, allow_nan=False))
                if stopping or finished_epoch:
                    break
            if max_updates is not None and invocation_updates >= max_updates:
                break
        complete = progress["epoch"] == config.epochs
        result = {"status": "complete" if complete else "stopped", "progress": progress,
                  "train_items": len(items), "run_sha256": fingerprint(identity)}
        if complete:
            model.eval()
            destination = run_dir / "export"
            if destination.exists():
                receipt = json.loads((destination / "adaptation_manifest.json").read_text(encoding="utf-8"))
                if receipt["files"] != checkpoint_files(destination) or receipt["provenance"] != result:
                    raise ValueError("existing export does not match this completed training run")
            else:
                export_uncalibrated(checkpoint, destination, result)
        _atomic_save(result, run_dir / "result.json")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model", "run"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--train-encoder", action="store_true")
    defaults = TrainingConfig()
    for name in ("epochs", "batch_size", "grad_accum", "seed", "checkpoint_every", "max_len", "head_max_len"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "encoder_learning_rate", "weight_decay", "max_grad_norm"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    arguments = parser.parse_args()
    settings = TrainingConfig(**{name: getattr(arguments, name) for name in asdict(defaults)})
    print(json.dumps(train(arguments.dataset, arguments.model, arguments.run, config=settings,
                           device=arguments.device, resume=arguments.resume, max_updates=arguments.max_updates), indent=2))


if __name__ == "__main__":
    main()
