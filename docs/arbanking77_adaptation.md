# English and Arabic support-intent adaptation experiment

This demonstration keeps all 77 banking-support intents and pairs English with
the first Modern Standard Arabic variant from
[SinaLab/ArBanking77](https://github.com/SinaLab/ArBanking77/tree/2e3a5639e254bc28828ed0af6d3059d64d3b97fc).
It measures intent routing only. There are no invented urgency or boolean labels,
and results must not be generalized to Arabic dialects or other support domains.

## Reproduce the data preparation

Obtain these files from source commit `2e3a5639e254bc28828ed0af6d3059d64d3b97fc`,
preserving their paths: `LICENSE`, `README.md`, `data/Banking77_intents.csv` and
`data/Banking77_full_corpus.csv`. The adapter checks their exact SHA-256 hashes
before reading them and performs no network access.

```shell
python research/scripts/prepare_arbanking77.py --source local-arbanking77 --output prepared-arbanking77
```

The repository's [corpus license](https://github.com/SinaLab/ArBanking77/blob/2e3a5639e254bc28828ed0af6d3059d64d3b97fc/LICENSE)
is **CC BY-SA 4.0**, copyright Sina Lab, Birzeit University (2023). It is distinct
from Laya's software license. The adapter preserves it as `SOURCE_LICENSE.txt`.
Retain this attribution and the source's terms when sharing derived data. The
English queries originate in [PolyAI's BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
Source research: Jarrar, Birim, Khalilia, Erden and Ghanem,
[ArBanking77 (2023)](http://www.jarrar.info/publications/JBKEG23.pdf), and Malaysha
et al., [AraFinNLP (2024)](https://www.jarrar.info/publications/MEEKJNBB24.pdf).

The pinned CSV has eight malformed rows with shifted columns. Whole source
families are quarantined: 855, 930, 2736, 6707, 6712, 8271, 9466 and 9483. The
adapter also rejects missing paired text, inconsistent source IDs, contradictory
duplicate labels and failed script checks. Script checks and source tags are
language assertions, not an independent native-speaker audit.

The resulting audit records 26,150 examples across 13,075 source families and 77
intents. Normalized duplicate links merge some families. Official `Te` families,
including related duplicates, remain reserved for final testing. Remaining groups
use seed `arbanking77-family-v1` and 70/15/15% training/calibration/policy splits.

| Partition | Rows | Connected groups |
| --- | ---: | ---: |
| Training | 13,988 | 6,991 |
| Calibration | 3,034 | 1,517 |
| Policy selection | 2,958 | 1,478 |
| Final test | 6,170 | 3,077 |

The frozen manifest fingerprint is
`d0035dba6987ebf5c69a223013e4ed57fe8bf44d64716dc24e1bc53ca62b58e0`.
Independent evaluation selects one record per connected group and language,
using identifiers rather than gold labels or predictions.

## Predeclared experiment

[The protocol](../research/arbanking77_protocol.json) fixes the source, model
revision, languages, option order through the dataset fingerprint, token budgets,
training candidates, calibration objective and threshold grid before policy
selection or final-test inference. The target is at least 50% coverage and at most
5% routing error, with simultaneous confidence bounds for both required languages.
The selection procedure allocates the 5% family error probability across four
possible model candidates, language slices, thresholds and the two bounds. Unused
candidate slots do not increase the available error probability.

Run the commands from the repository root. `local-multilingual` must contain the
multilingual checkpoint at the protocol's pinned model revision, with the files
listed in `base_model_files`. Preserve the same Python/library versions for
collection, finalization and deployment verification.

Keep candidate statistical analysis and finalization on one platform. Matching
Python/library versions can still produce different last-bit probabilities and
binomial bounds on Windows and Linux, so a copied report can fail exact replay
even when its source-file hashes match. Preserve original reports when moving
saved logits; any destination-platform re-analysis needs its own linked receipt
and verification of probabilities, individual threshold decisions, counts and
gate outcomes. The coordinator's exact artifact checks remain in force. Serving
portability is checked later using the identical exported bundle.

Use two CPU threads consistently, including during training and candidate
evaluation, so their runtime records agree with the finalization and measurement
commands below. Set these variables in the shell before starting Python.

PowerShell:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
```

POSIX shell:

```sh
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

The candidate recipe verifies the pinned base files (or the complete training-run
lineage for an adapted export), collects only calibration and policy partitions,
fits temperatures, computes independent policy metrics, and tests the fixed grid:

```shell
python research/scripts/evaluate_adaptation_candidate.py --protocol research/arbanking77_protocol.json --candidate base --dataset prepared-arbanking77 --model local-multilingual --output runs/base-evaluation
```

For a trained candidate, use its declared ID and the runner's `export/` directory.
`--resume` verifies and reuses saved inference chunks. The resulting report binds
the protocol, checkpoint, calibration and prediction artifacts. It includes a
training-only add-one-smoothed class-frequency baseline. The recipe never opens
the final test for inference and never publishes a deployment automatically.

Training candidates start independently from the pinned multilingual checkpoint.
Later candidates need not run if an earlier one qualifies. Neither repeated
final-test attempts nor replacing a failed model using final-test results is part
of this protocol. A failure to qualify is a measured limitation, not permission
to lower the gate after seeing outcomes. Base-model pretraining exposure to this
public corpus is unknown; the split protects adaptation-stage independence only.

### Train and evaluate adapted candidates

These single-line commands work in PowerShell and POSIX shells and spell out the
frozen training settings. Use a distinct run directory for each candidate. Start
with the head-only candidate:

```shell
python -m laya.adapt_train --dataset prepared-arbanking77 --model local-multilingual --run runs/head-1epoch --device cpu --epochs 1 --batch-size 1 --grad-accum 32 --learning-rate 0.0001 --encoder-learning-rate 0.00002 --weight-decay 0.01 --max-grad-norm 1 --seed 42 --checkpoint-every 25 --max-len 1024 --head-max-len 768
```

After it reports `complete` and publishes `export/`, collect its calibration and
policy evidence:

```shell
python research/scripts/evaluate_adaptation_candidate.py --protocol research/arbanking77_protocol.json --candidate head-1epoch --dataset prepared-arbanking77 --model runs/head-1epoch/export --output runs/head-evaluation
```

If no completed candidate qualifies, the next declared candidate trains the
encoder for one epoch:

```shell
python -m laya.adapt_train --dataset prepared-arbanking77 --model local-multilingual --run runs/encoder-1epoch --device cpu --epochs 1 --batch-size 1 --grad-accum 32 --learning-rate 0.0001 --encoder-learning-rate 0.00002 --weight-decay 0.01 --max-grad-norm 1 --seed 42 --checkpoint-every 25 --max-len 1024 --head-max-len 768 --train-encoder
```

Evaluate its completed export using `--candidate encoder-1epoch`,
`--model runs/encoder-1epoch/export` and a new `--output runs/encoder-evaluation`.
If that still leaves no qualifying candidate, the final declared candidate is:

```shell
python -m laya.adapt_train --dataset prepared-arbanking77 --model local-multilingual --run runs/encoder-3epoch --device cpu --epochs 3 --batch-size 1 --grad-accum 32 --learning-rate 0.0001 --encoder-learning-rate 0.00002 --weight-decay 0.01 --max-grad-norm 1 --seed 42 --checkpoint-every 25 --max-len 1024 --head-max-len 768 --train-encoder
```

Evaluate that export with `--candidate encoder-3epoch`, its own `export/` path and
a new output directory. All three training commands start from `local-multilingual`.
To resume interrupted training, repeat the same command with `--resume`, preserving
the run directory, source snapshot and recorded environment. See the
[training recovery contract](adaptation_training.md#recovery-contract). Resume an
interrupted candidate evaluation by adding `--resume` to its original evaluation
command. Training completion alone does not qualify a candidate; the independent
policy report must pass before selection and final testing.

The original CPU qualification run is pinned to source commit
`ef79fc5137f85f7e03e78b5aebabd1a0bc5cccd0`. Its run identity and resume
path remain on that source snapshot. The subsequent safetensors training-checkpoint
change applies to new runs; a new checkout must not be substituted into the
frozen run or its evidence.

## Freeze selection and finalize

The coordinator accepts a completed prefix of the candidate sequence: for
example, base alone, or base followed by head-1epoch. Supply every completed
candidate report, including failures. It rechecks dataset membership, rendering,
calibration and policy selection, then chooses the qualified model with the
highest minimum accepted fraction across required languages. Ties follow the
predeclared order. It refuses to freeze a selection if no candidate qualifies.
This command performs no model inference and reads no final-test predictions:

```shell
python research/scripts/finalize_adaptation_experiment.py select --protocol research/arbanking77_protocol.json --dataset prepared-arbanking77 --candidate-report runs/base-evaluation --output runs/frozen-selection
```

Add `--candidate-report runs/head-evaluation` when that candidate has also been
completed. Preserve the printed `selection_sha256` through a trusted channel.
Then provide that checksum and the selected model directory to the separate
final phase, using the same CPU runtime and thread count as candidate evaluation:

```shell
python research/scripts/finalize_adaptation_experiment.py finalize --selection runs/frozen-selection --expected-selection-sha256 TRUSTED_SELECTION_SHA256 --dataset prepared-arbanking77 --model local-multilingual --run runs/final --bundle deploy/arbanking77-v1 --threads 2
```

Replace the checksum placeholder with the selection receipt's value. For an
adapted candidate, replace the model path with that training run's `export/`.
The coordinator validates its lineage and writes `run.json` before collecting
test predictions. Use `--resume` after interruption: the same selection, model,
data and output location are required. Completed inference chunks are reused;
a completed final run returns its verified result without another forward pass.
An interruption after publishing the bundle recovers it using the durable receipt.

`result.json` records `qualified` only if the fixed policy passes its independent
final error and coverage bounds. Otherwise it records `failed_gate` and publishes
a [review-only bundle](adaptation_bundle.md); it does not try another candidate or
adjust a threshold. A review-only result does not meet the automation target.
`final-report.json` retains the measured failure as well as calibrated/raw metrics.
The bundle and its separately trusted receipt expose explicit `automate`/`review`
outcomes through `laya.adapt_bundle.load_bundle`.

The coordinator enforces identity within a run. It cannot prevent someone from
manually starting a new experiment after inspecting final results; doing so would
invalidate this protocol's held-out claim. Keep the frozen selection, candidate
reports and final-run records together as the audit trail. Selection records
include predictions and sample metadata and are not the deployable bundle.

## Deployment measurements

After a selected bundle passes its fixed final gate, use the
[deployment measurement recipe](adaptation_measurement.md) on Windows and Linux.
The same versioned unlabelled probe measures complete prediction calls, guard
paths and process memory, then compares outcomes for the identical bundle. These
measurements supplement the held-out quality report; they do not permit selecting
a replacement model or threshold after seeing final results.

## Execution evidence and limits

The pinned base checkpoint completed calibration and policy evaluation on
2026-09-22. The [aggregate result](../research/results/arbanking77_base_policy_20260922.json)
records the protocol, source and model fingerprints, evidence-file checksums,
metrics and every tested threshold. An independent replay reproduced the
temperature fit, metrics and selection from the saved prediction artifacts.
There were 3,034 calibration examples and 1,478 independent policy representatives
per language. The fitted choice temperature was 2.22636, with no bucket overrides.

| Policy slice | Intent accuracy, raw and calibrated | NLL, raw / calibrated | ECE (15 bins), raw / calibrated |
| --- | ---: | ---: | ---: |
| English | 47.56% | 3.361 / 2.345 | 0.324 / 0.085 |
| Modern Standard Arabic | 20.91% | 4.764 / 3.476 | 0.413 / 0.068 |

Temperature scaling improved these probability metrics without changing argmax
accuracy. **Neither language had a qualifying threshold.** At confidence 0.5,
English accepted 41.20% of examples with 31.69% observed error; Arabic accepted
12.25% with 58.01% observed error. The simultaneous bounds also failed, as did
every other threshold in the fixed grid. The coordinator correctly refused to
freeze this candidate. No final-test inference was performed and no deployment
was qualified. These are policy-partition results for the unadapted baseline;
the declared training candidates still require their own independent evaluation.

A [Windows/Linux replay audit](../research/results/arbanking77_base_cross_platform_replay_20260922.json)
used the same saved baseline logits and calibration, with no new inference.
The maximum absolute difference across 227,612 policy probabilities was
approximately `1.11e-16`; every one of the 35,472 per-example threshold comparisons,
aggregate count and gate outcome was unchanged. Seven binomial bounds also had
a maximum difference of approximately `1.11e-16`. These last-bit
differences changed the derived probability checksum and prevented direct exact
replay of the Windows report on Linux. A separate Linux report links the original
report and both derived checksums, preserves the original calibration and metrics,
and passes the unchanged replay checks. Final selection still correctly refuses
the failed baseline. This is evidence about numerical replay, not model accuracy.

The head-only candidate completed its one training epoch on 2026-09-22. Its
[training audit](../research/results/arbanking77_head_training_20260922.json)
records 438 optimizer updates over all 13,988 training examples, the completed
checkpoint and export hashes, and preservation of the pinned base files. The
export's configuration and temperature tensor both contain `[1, 1, 1]`, with
an empty bucket-override map and `unfitted` calibration status. This verifies
training completion and export consistency. It provides no held-out accuracy or
automation result; the subsequent policy evaluation below measures that candidate.

The [audited head-only policy result](../research/results/arbanking77_head_policy_20260923.json)
completed on 2026-09-23 with 3,034 calibration examples and 1,478 independent
policy representatives per language. The fitted choice temperature was 0.987064,
with no bucket overrides. Every threshold in the fixed grid failed to qualify
in both languages.

| Policy slice | Intent accuracy, raw and calibrated | NLL, raw / calibrated | ECE (15 bins), raw / calibrated |
| --- | ---: | ---: | ---: |
| English | 47.83% | 2.203 / 2.202 | 0.040 / 0.033 |
| Modern Standard Arabic | 21.38% | 3.353 / 3.356 | 0.047 / 0.051 |

At confidence 0.5, English accepted 38.77% of examples with 27.75% observed error;
Arabic accepted 10.01% with 45.95% observed error. The simultaneous bounds also
failed. Finalization recorded `head_not_qualified`, left the final test unopened
and preserved the encoder-training fallback. Deployment measurement was skipped.
The unchanged production coordinator independently replayed the completed
base-plus-head prefix and refused to freeze a selection, creating no final artifacts.
This head-only run does not establish useful automation. Its small accuracy
differences from the baseline have not been tested for statistical significance.
Temperature fitting changed probabilities, not argmax accuracy, and slightly
worsened Arabic policy NLL and ECE: fitting on calibration data does not guarantee
improvement in every held-out metric or language.

To shorten collection, Linux collected calibration and Windows collected policy
predictions from the identical trained export. Partition ownership was declared
before policy completion. Model/data/rendering bindings, library versions,
implementation hashes, batch size and chunk size matched. After all chunks and
dataset membership were verified, the unchanged candidate recipe resumed on Linux
with zero new forward calls and preserved the canonical prediction artifacts.
All temperature fitting, metrics and threshold selection ran on Linux; a separate
audit reproduced them and independently checked every threshold's accepted/error
counts. The aggregate result retains the source-receipt hashes. This verifies
collection provenance and numerical replay, not same-input Windows/Linux serving
parity or real-bundle performance.

The [one-epoch encoder training audit](../research/results/arbanking77_encoder1_training_20260923.json)
records a completed CPU run on 2026-09-23: 438 optimizer updates over all
13,988 training examples. The independent audit verified the checkpoint and
export hashes, preserved base files, finite exported tensors and neutral,
unfitted temperatures. This establishes training and export integrity only.
Calibration and independent policy evaluation are still required before this
candidate can be selected; the final test remains sealed.

A CPU pilot used four training-only examples from two source families, the actual
322M multilingual checkpoint, and the complete 77-option schema. All option text
remained distinct and untruncated at `max_len=1024`, `head_max_len=768`. On the
tested Linux host with two Torch threads, four warm forward passes took 8.46 s;
the process's peak resident memory was 4,887 MiB. Interrupted/resumed training
produced byte-identical exported checkpoint files to an uninterrupted run, and
source assets were unchanged. These timings include this host's conditions and
are not deployment throughput estimates.

That pilot establishes execution and recovery only. It does not establish
accuracy, calibration improvement, or an achievable automation rate. The separate
baseline results above do not establish useful adapted-model automation. Full
adaptation and independent evaluation are still required. Synthetic CPU tests
can be run separately without obtaining the corpus or pretrained weights:

```shell
python tests/test_arbanking_adapter.py
python tests/test_adapt_train.py
python tests/test_adaptation_finalize.py
```

The finalization tests include a complete run through real local-fixture forward
passes, calibration, selection, final evaluation, publication and prediction.
That fixture deliberately has a trivial scoring rule and synthetic labels; its
passing gate verifies that the stages connect, not that a trained model is useful.
