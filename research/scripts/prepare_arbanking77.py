"""Prepare paired English/MSA banking-support intent data from a pinned local source.

Download the four pinned source files separately; this adapter performs no network
access. It quarantines malformed/conflicting families and preserves official tests.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from laya.adapt_data import fingerprint, write_dataset  # noqa: E402
from laya.adapt_model import file_sha256  # noqa: E402


REVISION = "2e3a5639e254bc28828ed0af6d3059d64d3b97fc"
SOURCE = "https://github.com/SinaLab/ArBanking77"
EXPECTED = {
    "LICENSE": "2c9f00c84d0681f06ce5ccb0dfbccea531e3850f0f14a99561feb316f4a0fdff",
    "README.md": "7a9023e1c172a07d073ccf1c25d1a4e4e186f8e6c891a68562cd216ae78d3b7c",
    "data/Banking77_intents.csv": "aaba4ba1e0f8471e4594c1e2c6122d3b97f864bda7b587b360e6a41f44269c93",
    "data/Banking77_full_corpus.csv": "38863ae4093840f2418d76613d0c2134295fc78f1116361fa9ff00680b1309c1",
}
FRACTIONS = {"train": .7, "calibration": .15, "policy": .15, "test": 0}
SOURCE_ID = re.compile(r"(Tr|Te|D)[0-9]+\Z")


def _script_fraction(text, prefix):
    letters = [character for character in text if character.isalpha()]
    return sum(unicodedata.name(character, "").startswith(prefix) for character in letters) / max(1, len(letters))


def convert_rows(rows, labels):
    """Return paired examples and an audit, without inspecting model predictions."""
    if len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("source intent names must be unique, nonempty strings")
    questions = {"intent": {"type": "choice", "instructions": "Which banking support intent does the message express?",
                             "criteria": {label: None for label in labels}}}
    examples, fixed, excluded, seen_ids = [], {}, {}, set()
    for row in rows:
        qid = row.get("QID")
        if not isinstance(qid, str) or not qid.isdigit() or qid in seen_ids:
            raise ValueError("source QIDs must be unique numeric identifiers")
        seen_ids.add(qid)
        source_id = row.get("QuestionID_MSA1", "")
        if None in row or not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
            excluded[qid] = "malformed_csv_or_source_id"
            continue
        label = row.get("Intent_en")
        if label not in labels:
            raise ValueError("source row names an unknown intent")
        english, arabic = row.get("Question_en"), row.get("Question_MSA1")
        if any(not isinstance(text, str) or not text.strip() or text.strip() == "NULL" for text in (english, arabic)):
            excluded[qid] = "missing_paired_text"
            continue
        # This is a documented script sanity check, not independent language verification.
        if _script_fraction(english, "LATIN") < .8 or _script_fraction(arabic, "ARABIC") < .5:
            excluded[qid] = "script_sanity_check"
            continue
        for lang, text in (("en", english), ("ar", arabic)):
            key = "arbanking77:%s:%s" % (qid, lang)
            examples.append({"id": key, "group_id": "arbanking77:" + qid, "language": lang,
                             "state": {"message": text}, "targets": {"intent": label},
                             "source": {"dataset": SOURCE, "revision": REVISION, "license": "CC-BY-SA-4.0",
                                        "record_id": qid, "variant": "English" if lang == "en" else "MSA1",
                                        "source_split_id": source_id}})
            if source_id.startswith("Te"):
                fixed[key] = "test"
    texts = collections.defaultdict(list)
    for example in examples:
        normalized = " ".join(unicodedata.normalize("NFC", example["state"]["message"]).split())
        texts[fingerprint(normalized)].append(example)
    conflicting = set()
    for duplicates in texts.values():
        if len({example["targets"]["intent"] for example in duplicates}) > 1:
            conflicting.update(example["source"]["record_id"] for example in duplicates)
    for qid in conflicting:
        excluded[qid] = "contradictory_normalized_duplicate"
    examples = [example for example in examples if example["source"]["record_id"] not in conflicting]
    fixed = {example["id"]: fixed[example["id"]] for example in examples if example["id"] in fixed}
    audit = {"source_rows": len(seen_ids), "paired_families": len(examples) // 2,
             "excluded_families": dict(sorted(excluded.items())), "exclusion_counts": dict(collections.Counter(excluded.values())),
             "languages": ["en", "ar"], "label_count": len(labels),
             "source_assertion": "English queries and manually localized Modern Standard Arabic, first variant only",
             "language_check": "source tags plus >=80% Latin letters in English and >=50% Arabic letters in MSA",
             "gold_tasks": ["intent"], "ordinal_or_boolean_labels_invented": False}
    return questions, examples, fixed, audit


def prepare(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    hashes = {name: file_sha256(source_dir / name) for name in EXPECTED}
    if hashes != EXPECTED:
        raise ValueError("source files do not match the pinned ArBanking77 revision")
    with (source_dir / "data/Banking77_intents.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = [row["label_en"] for row in csv.DictReader(stream)]
    if len(labels) != 77:
        raise ValueError("expected the source's complete 77-intent label space")
    with (source_dir / "data/Banking77_full_corpus.csv").open(encoding="utf-8-sig", newline="") as stream:
        questions, examples, fixed, audit = convert_rows(csv.DictReader(stream), labels)
    if output_dir.exists():
        raise FileExistsError("prepared dataset already exists: %s" % output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arbanking-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "dataset"
        manifest = write_dataset(stage, examples, questions, seed="arbanking77-family-v1",
                                  fractions=FRACTIONS, fixed_splits=fixed)
        audit.update(source_revision=REVISION, files_sha256=hashes, dataset_manifest_sha256=fingerprint(manifest),
                     split_rows=manifest["counts"], split_groups=manifest["group_counts"],
                     normalized_duplicate_rows=manifest["normalized_duplicate_rows"])
        (stage / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (stage / "SOURCE_LICENSE.txt").write_bytes((source_dir / "LICENSE").read_bytes())
        stage.rename(output_dir)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "excluded_families"}, indent=2))
