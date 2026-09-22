"""Local checkpoint I/O and unrounded model records for domain adaptation."""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .adapt_data import target_index, validate_questions, validate_records
from .agent import Agent, _fix_tokenizer_config, _verify_compatibility
from .common import QTYPES, build_model, build_sequence, collate_items, render_options, serialize_state


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(directory):
    """Hash exactly the local files consumed by the SDK; no Hub fallback."""
    directory = Path(directory)
    required = [directory / "model.safetensors", directory / "rl_agent_config.json",
                directory / "encoder" / "config.json", directory / "tokenizer" / "tokenizer.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError("local checkpoint is missing %s" % path)
    files = set(required)
    for name in ("encoder", "tokenizer"):
        files.update(path for path in (directory / name).rglob("*") if path.is_file())
    return {path.relative_to(directory).as_posix(): file_sha256(path) for path in sorted(files)}


@dataclass
class LocalCheckpoint:
    model: torch.nn.Module
    tokenizer: object
    config: dict
    device: torch.device


def load_local_checkpoint(directory, device="cpu"):
    """Load local assets without rewriting tokenizer files in the source checkpoint."""
    from transformers import AutoTokenizer

    directory = Path(directory)
    # Fail before constructors have a chance to fall back to remote model assets.
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError("local checkpoint is missing %s" % (directory / name))
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("the adaptation runner currently supports CPU and CUDA devices")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select CPU explicitly")
    cfg = json.loads((directory / "rl_agent_config.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="laya-tokenizer-") as temporary:
        shutil.copytree(directory / "tokenizer", Path(temporary) / "tokenizer")
        _fix_tokenizer_config(temporary)
        tokenizer = AutoTokenizer.from_pretrained(Path(temporary) / "tokenizer", local_files_only=True)
    if any(value is None for value in (tokenizer.pad_token_id, tokenizer.mask_token_id,
                                       tokenizer.cls_token_id, tokenizer.sep_token_id)):
        raise ValueError("checkpoint tokenizer needs pad, mask, cls and sep tokens")
    model = build_model(cfg, encoder_dir=str(directory / "encoder"))
    weights = load_file(str(directory / "model.safetensors"))
    _verify_compatibility(model, cfg, weights, str(directory))
    model.load_state_dict(weights, strict=True)
    model.encoder.config.reference_compile = False
    model.to(device=target, dtype=torch.float32).eval()
    return LocalCheckpoint(model, tokenizer, cfg, target)


def prepare_input(checkpoint, state, question):
    """Render one question, refusing lost or indistinguishable option text.

    The opt-in adaptation path also refuses truncated state. An application can
    send such inputs for review instead of applying a policy to unseen truncation.
    """
    cfg, tokenizer = checkpoint.config, checkpoint.tokenizer
    internal = Agent._to_internal(question)
    expected = [[tokenizer.mask_token_id] + tokenizer(
        " " + option.replace(tokenizer.mask_token, " "), add_special_tokens=False)["input_ids"]
        for option in render_options(internal)]
    ids, markers = build_sequence(tokenizer, state, internal, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    if len(markers) != len(expected):
        raise ValueError("question loses options at the configured token budget")
    end = markers[-1] + len(expected[-1])
    boundaries = markers[1:] + [end]
    actual = [ids[start:stop] for start, stop in zip(markers, boundaries)]
    if (actual != expected or len(set(map(tuple, actual))) != len(actual)
            or end >= len(ids) or ids[end] != tokenizer.sep_token_id):
        raise ValueError("question option text is truncated or indistinguishable after tokenization")
    state_ids = tokenizer(serialize_state(state).replace(tokenizer.mask_token, " "),
                          add_special_tokens=False)["input_ids"]
    if ids[end + 1:-1] != state_ids:
        raise ValueError("state is truncated at the configured token budget")
    return {"ids": ids, "markers": markers, "qtype": QTYPES[question["type"]]}


def prepare_items(checkpoint, records, questions):
    """Tokenize hard-labelled records using exactly the SDK's question rendering."""
    questions = validate_questions(questions)
    rows = validate_records(records, questions)
    items = []
    for row in rows:
        for qid, question in questions.items():
            if qid not in row["targets"]:
                continue
            rendered = prepare_input(checkpoint, row["state"], question)
            label = target_index(question, row["targets"][qid])
            target = [float(i == label) for i in range(len(rendered["markers"]))]
            items.append({"id": row["id"], "group_id": row["group_id"], "language": row["language"],
                          "question_id": qid, **rendered, "label": label, "target": target})
    return items


def forward_batch(checkpoint, items):
    batch = collate_items([items], checkpoint.tokenizer.pad_token_id)
    if batch is None:
        raise ValueError("cannot run an empty model batch")
    logits, _ = checkpoint.model(*(batch[name].to(checkpoint.device) for name in (
        "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    return logits, batch


def collect_logits(checkpoint, items, batch_size=8):
    """Return raw, finite logits without applying inherited calibration or rounding."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    was_training = checkpoint.model.training
    checkpoint.model.eval()
    result = []
    try:
        with torch.inference_mode():
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                logits, _ = forward_batch(checkpoint, chunk)
                if not torch.isfinite(logits).all():
                    raise ValueError("model produced non-finite logits")
                for item, values in zip(chunk, logits.float().cpu().tolist()):
                    result.append({key: item[key] for key in (
                        "id", "group_id", "language", "question_id", "qtype", "label")})
                    result[-1]["logits"] = values[:len(item["markers"])]
    finally:
        checkpoint.model.train(was_training)
    return result


def export_uncalibrated(checkpoint, destination, provenance):
    """Export changed weights with no inherited temperature/bucket calibration."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("export destination already exists: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".laya-export-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "checkpoint"
        stage.mkdir()
        weights = {key: value.detach().cpu().contiguous() for key, value in checkpoint.model.state_dict().items()}
        weights["temperature"] = torch.ones_like(weights["temperature"])
        save_file(weights, str(stage / "model.safetensors"))
        checkpoint.model.encoder.config.save_pretrained(stage / "encoder")
        checkpoint.tokenizer.save_pretrained(stage / "tokenizer")
        _fix_tokenizer_config(str(stage))
        cfg = copy.deepcopy(checkpoint.config)
        cfg.update(temperature=[1.0, 1.0, 1.0], temperature_by_options={}, fine_tuned=True)
        cfg["adaptation"] = {"calibration_status": "unfitted", "objective": "hard_label_cross_entropy"}
        (stage / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt = {"schema_version": 1, "files": checkpoint_files(stage), "provenance": provenance,
                   "calibration_status": "unfitted"}
        (stage / "adaptation_manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                                                       encoding="utf-8")
        stage.rename(destination)
    return receipt
