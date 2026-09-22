"""Resumable raw-logit collection on a frozen, independent dataset partition."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
import tokenizers
import transformers

from .adapt_data import fingerprint, question_fingerprint, read_dataset, select_independent_records, validate_questions
from .adapt_model import checkpoint_files, collect_logits, file_sha256, load_local_checkpoint, prepare_items
from .adapt_train import _atomic_save, _run_lock
from .common import QTYPES


def validate_predictions(artifact, questions):
    """Check a raw-logit artifact against its ordered question schema.

    Hashes detect inconsistency, not a malicious replacement of all artifacts.
    Consumers must retain a trusted dataset/model binding separately.
    """
    questions = validate_questions(questions)
    if (not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int
            or artifact["schema_version"] != 1 or artifact.get("stage") != "logits"
            or artifact.get("split") not in ("calibration", "policy", "test")):
        raise ValueError("expected a versioned evaluation-partition logit artifact")
    binding = artifact.get("binding")
    if not isinstance(binding, dict) or binding.get("questions_sha256") != question_fingerprint(questions):
        raise ValueError("prediction question schema differs from its binding")
    if (not isinstance(binding.get("dataset_manifest_sha256"), str)
            or len(binding["dataset_manifest_sha256"]) != 64
            or not isinstance(binding.get("model_files"), dict) or not binding["model_files"]
            or not isinstance(binding.get("rendering"), dict)):
        raise ValueError("prediction artifact is missing its dataset/model/rendering binding")
    rows = artifact.get("records")
    if not isinstance(rows, list) or not rows or artifact.get("records_sha256") != fingerprint(rows):
        raise ValueError("prediction records are missing or do not match their fingerprint")
    seen, groups = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each logit record must be a dictionary")
        for field in ("id", "group_id", "language", "question_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("logit records need nonempty %s" % field)
        qid = row["question_id"]
        if qid not in questions:
            raise ValueError("logit record names an unknown question")
        key, group = (row["id"], qid), (row["group_id"], row["language"], qid)
        if key in seen or group in groups:
            raise ValueError("logits need one independent record per question, group and language")
        seen.add(key)
        groups.add(group)
        question = questions[qid]
        count = 2 if question["type"] == "noul" else len(question["criteria"])
        if type(row.get("qtype")) is not int or row["qtype"] != QTYPES[question["type"]]:
            raise ValueError("logit question type does not match the schema")
        if type(row.get("label")) is not int or not 0 <= row["label"] < count:
            raise ValueError("logit label is not a valid option index")
        logits = row.get("logits")
        if not isinstance(logits, list) or len(logits) != count or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in logits
        ):
            raise ValueError("logits must be finite numbers matching the exact option count")
    return copy.deepcopy(rows)


def _metadata(item):
    return {key: item[key] for key in ("id", "group_id", "language", "question_id", "qtype", "label")}


def verify_prediction_dataset(artifact, dataset_dir):
    """Check exact partition membership, targets and connected families on reload."""
    from .adapt_data import target_index

    questions, rows, manifest = read_dataset(dataset_dir)
    records = validate_predictions(artifact, questions)
    if artifact["binding"]["dataset_manifest_sha256"] != fingerprint(manifest):
        raise ValueError("prediction dataset manifest differs from the prepared dataset")
    expected = []
    for qid, question in questions.items():
        for row in select_independent_records(rows, questions, manifest, split=artifact["split"], question_id=qid):
            expected.append({"id": row["id"], "group_id": manifest["groups"][row["id"]],
                             "language": row["language"], "question_id": qid, "qtype": QTYPES[question["type"]],
                             "label": target_index(question, row["targets"][qid])})
    def key(row):
        return row["question_id"], row["id"]

    if sorted(expected, key=key) != sorted((_metadata(row) for row in records), key=key):
        raise ValueError("prediction records differ from independent representatives of the declared partition")
    return questions


def collect_partition(dataset_dir, model_dir, output_dir, *, split, device="cpu", batch_size=1,
                      max_len=None, head_max_len=None, chunk_size=64, resume=False, max_chunks=None, log=print):
    """Collect raw logits, with atomic chunks and exact input/environment resume.

    Only the explicitly requested partition is forwarded through the model. Test
    collection is an explicit caller action; this API cannot prevent prior peeking
    outside the experiment. Do it only after freezing the selected model/policy.
    """
    if split not in ("calibration", "policy", "test"):
        raise ValueError("select calibration, policy or test explicitly")
    for name, value in (("batch_size", batch_size), ("chunk_size", chunk_size)):
        if type(value) is not int or value < 1:
            raise ValueError("%s must be a positive integer" % name)
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    output_dir = Path(output_dir)
    if resume and not (output_dir / "collection.json").is_file():
        raise FileNotFoundError("resume requires collection.json")
    if not resume and output_dir.exists():
        raise FileExistsError("collection directory exists; use resume or a new directory")
    questions, rows, manifest = read_dataset(dataset_dir)
    selected = []
    for qid in questions:
        for row in select_independent_records(rows, questions, manifest, split=split, question_id=qid):
            # Connected duplicates may have different original group IDs.
            row["group_id"] = manifest["groups"][row["id"]]
            row["targets"] = {qid: row["targets"][qid]}
            selected.append((qid, row))
    if not selected:
        raise ValueError("requested partition has no independent labelled examples")
    files = checkpoint_files(model_dir)
    checkpoint = load_local_checkpoint(model_dir, device=device)
    for name, value in (("max_len", max_len), ("head_max_len", head_max_len)):
        if value is not None:
            if type(value) is not int or value < 1:
                raise ValueError("%s must be a positive integer" % name)
            checkpoint.config[name] = value
    rendering = {"max_len": checkpoint.config.get("max_len", 512),
                 "head_max_len": checkpoint.config.get("head_max_len", 192)}
    if rendering["head_max_len"] > rendering["max_len"]:
        raise ValueError("head_max_len must not exceed max_len")
    items = []
    for qid, row in selected:
        items.extend(prepare_items(checkpoint, [row], {qid: questions[qid]}))
    items.sort(key=lambda item: (item["question_id"], item["id"]))
    binding = {"dataset_manifest_sha256": fingerprint(manifest), "questions_sha256": question_fingerprint(questions),
               "model_files": files, "rendering": rendering}
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
               "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
               "device": str(checkpoint.device), "threads": torch.get_num_threads()}
    implementation = {name: file_sha256(Path(__file__).with_name(name)) for name in
                      ("adapt_eval.py", "adapt_model.py", "adapt_data.py", "common.py", "agent.py")}
    identity = {"schema_version": 1, "binding": binding, "split": split, "items_sha256": fingerprint(items),
                "batch_size": batch_size, "chunk_size": chunk_size, "runtime": runtime, "implementation": implementation}
    if resume:
        recorded = json.loads((output_dir / "collection.json").read_text(encoding="utf-8"))
        if fingerprint(recorded) != fingerprint(identity):
            raise ValueError("collection data, model, rendering, implementation or runtime differs")
    else:
        output_dir.mkdir(parents=True)
    with _run_lock(output_dir):
        chunks_dir = output_dir / "chunks"
        if not resume:
            chunks_dir.mkdir()
            _atomic_save(identity, output_dir / "collection.json")
        records, seconds = [], 0.0
        paths = sorted(chunks_dir.glob("*.json"))
        for number, path in enumerate(paths):
            if path.name != "%08d.json" % number or len(records) >= len(items):
                raise ValueError("collection chunks have a gap or unexpected extra data")
            chunk = json.loads(path.read_text(encoding="utf-8"))
            expected = items[len(records):len(records) + chunk_size]
            values = chunk.get("records", [])
            if (chunk.get("collection_sha256") != fingerprint(identity)
                    or chunk.get("records_sha256") != fingerprint(values)
                    or [_metadata(row) for row in values] != [_metadata(item) for item in expected]
                    or not isinstance(chunk.get("forward_seconds"), (int, float))
                    or not math.isfinite(chunk["forward_seconds"]) or chunk["forward_seconds"] < 0):
                raise ValueError("collection chunk does not match its frozen inputs")
            records.extend(values)
            seconds += chunk["forward_seconds"]
        invocation_chunks = 0
        for start in range(len(records), len(items), chunk_size):
            began = time.perf_counter()
            values = collect_logits(checkpoint, items[start:start + chunk_size], batch_size=batch_size)
            elapsed = time.perf_counter() - began
            chunk = {"collection_sha256": fingerprint(identity), "records": values,
                     "records_sha256": fingerprint(values), "forward_seconds": elapsed}
            _atomic_save(chunk, chunks_dir / ("%08d.json" % (start // chunk_size)))
            records.extend(values)
            seconds += elapsed
            invocation_chunks += 1
            log(json.dumps({"split": split, "records": len(records), "total": len(items), "forward_seconds": seconds}))
            if max_chunks is not None and invocation_chunks >= max_chunks:
                break
        if len(records) < len(items):
            return {"status": "stopped", "records": len(records), "total": len(items)}
        if checkpoint_files(model_dir) != files:
            raise ValueError("source checkpoint changed during collection")
        artifact = {"schema_version": 1, "stage": "logits", "split": split, "binding": binding,
                    "records": records, "records_sha256": fingerprint(records), "collection": identity,
                    "measurement": {"forward_seconds": seconds, "examples": len(records), "runtime": runtime,
                                    "scope": "model collection only; excludes checkpoint loading and input preparation"}}
        validate_predictions(artifact, questions)
        _atomic_save(artifact, output_dir / "predictions.json")
        return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("calibration", "policy", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = collect_partition(args.dataset, args.model, args.output, split=args.split, device=args.device,
                               batch_size=args.batch_size, chunk_size=args.chunk_size, max_len=args.max_len,
                               head_max_len=args.head_max_len, max_chunks=args.max_chunks, resume=args.resume)
    print(json.dumps({"status": result.get("status", "complete"), "output": args.output}))


if __name__ == "__main__":
    main()
