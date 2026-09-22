# Labelled datasets for domain adaptation

`laya.adapt_data` prepares local labelled examples with a reproducible split
manifest. It does not download a checkpoint, train a model, fit calibration or
authorize automated decisions. The existing prediction API is unchanged.

Use the same question definitions as `Agent.predict`. Gold labels are strict:

| Question type | Gold target |
| --- | --- |
| `choice` | An exact criterion label, not its integer position |
| `score` | A zero-based integer rubric index; booleans and floats are rejected |
| `noul` | A JSON boolean, `true` or `false` |

An example can label only a subset of the questions. It must have at least one
label, and cannot name an unknown question. Structured JSON instructions and
criterion descriptions are supported. Question order, option order and structured
text field order are included in fingerprints because they can affect model input.

```python
from laya.adapt_data import read_dataset, write_dataset

questions = {
    "queue": {
        "type": "choice",
        "instructions": "Which queue should handle this ticket?",
        "criteria": {"billing": "payments and invoices", "technical": "bugs and outages"},
    }
}
examples = [{
    "id": "example:1",
    "group_id": "example:conversation:1",
    "language": "en",
    "state": {"subject": "Invoice", "body": "I was charged twice."},
    "targets": {"queue": "billing"},
    "source": {
        "dataset": "local-example",
        "revision": "v1",
        "license": "CC0-1.0",
        "record_id": "1",
    },
}]

# Choose a new output directory; existing datasets are never updated in place.
manifest = write_dataset("prepared-tickets", examples, questions, seed="tickets-v1")
questions, examples, manifest = read_dataset("prepared-tickets")
print(manifest["counts"])  # One example is insufficient for four usable partitions.
```

Replace the illustrative provenance with the source's actual dataset identifier,
immutable revision (or content hash), license and original record ID. Recording a
license does not verify permissions. Do not put gold labels, answer drafts or
other information unavailable at inference time in `state`. Language tags must be
lowercase (for example `en`, `de` or `pt-br`); they record the source assertion and
are not independently verified language detection.

## Grouping and independence

Assign a globally unique `group_id` to each conversation or related example
family. Translations, paraphrases and augmentations of the same example must
share this group. The utility additionally joins groups connected by either:

- The same source dataset and original record ID, including across revisions.
- Identical state after Unicode NFC and whitespace normalization, ignoring JSON
  object key order for duplicate detection only. Case and list order are preserved.

Connections are transitive. Contradictory labels for duplicate states fail
validation. Matching finds normalized exact duplicates; it does not discover
unmarked translations or semantic near-duplicates. Source adapters remain
responsible for preserving those relationships and auditing their data.

Each connected group is hashed with the seed into one of four roles: `train`
(60%), `calibration` (15%), `policy` (10%) and `test` (15%). Fractions can be
overridden with four nonnegative values summing to one. Assignment is independent of
input row order and uses no gold labels. Fractions are expected proportions, not
guaranteed split sizes or stratification. Small or skewed datasets can leave a
partition, task or language without enough evidence; inspect the counts before
training. Adding a connecting record can merge groups and change assignments,
so freeze the entire prepared dataset before running an experiment.

To preserve a source's official test set, pass `fixed_splits={"example:42": "test"}`
to `write_dataset` or `make_split_manifest`. A reserved example places its entire
connected family in that role; contradictory reservations are rejected. The
manifest records and verifies reservations. Fractions then apply only to the
remaining groups. Setting the test fraction to zero reserves that role exclusively
for the fixed test families. Missing roles still provide no usable evidence.

The manifest retains **all rows**, including duplicates, and reports both row
counts and connected-group counts by split and language. Neither retaining rows
nor placing families in separate splits makes repeated observations independent.
`select_independent_records(examples, questions, manifest, split="policy",
question_id="queue")` selects one labelled representative per connected group and
language. It uses the frozen seed, question ID and example ID, never target values
or model predictions. Apply it before evaluating a task. Missing task labels are
excluded; a partition without that label returns no evidence. Other questions can
select different representatives. These estimates describe this sampling scheme,
not the frequency-weighted population of all raw rows. Independence still depends
on the adapter correctly identifying families. Never use duplicate row counts as
binomial sample sizes.

Use training data for fitting weights, calibration data for temperatures, policy
data for selecting review thresholds, and test data only for a locked final
evaluation. This module records those roles; downstream runners must enforce
access to the appropriate partition. A manifest is not an access-control system.

## Persistence and verification

The prepared directory contains `questions.json`, `records.jsonl` and
`splits.json`. Rows are sorted by ID. UTF-8 and LF line endings give portable
files, while JSON state and question ordering survive a round trip. Publication
stages files beside the destination and renames the complete directory; a failed
write does not publish a partial dataset. Use a unique destination per publisher;
concurrent publication to the same path is not supported.

`read_dataset` revalidates labels, source fields, grouping, assignments, counts and
fingerprints. Data or schema changes that make the manifest inconsistent are
rejected. This is an integrity check, not authentication: an actor who can replace
all files can recompute a valid manifest. Experiment and deployment artifacts
should bind the hash of the frozen manifest from a separately trusted record.

Run the synthetic CPU-only contract tests with:

```shell
python tests/test_adapt_data.py
```

These checks establish data handling behavior. They do not measure prediction
accuracy, calibration quality or an achievable automation rate.
