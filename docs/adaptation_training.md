# Resumable local adaptation

`python -m laya.adapt_train` fits a local Laya checkpoint on the `train` partition
of a [prepared labelled dataset](adaptation_data.md). It uses supervised hard-label
cross-entropy for choice, score and noul questions. It does not implement the
notebook's RLCD objective, train the action head, fit temperatures, or establish a
safe automation rate. Existing SDK calls and checkpoints are unchanged.

Supply a complete local checkpoint: `model.safetensors`, `rl_agent_config.json`,
`encoder/config.json` and the `tokenizer` directory. The runner performs no Hub
downloads and does not rewrite the source checkpoint's tokenizer. A CPU run needs
no pretrained downloads when using the tiny synthetic test fixtures below.

```shell
python -m laya.adapt_train --dataset prepared-tickets --model local-laya \
  --run runs/tickets-v1 --epochs 3 --batch-size 4 --grad-accum 4

# Resume the same run after interruption, with the identical arguments:
python -m laya.adapt_train --dataset prepared-tickets --model local-laya \
  --run runs/tickets-v1 --epochs 3 --batch-size 4 --grad-accum 4 --resume
```

These are POSIX shell continuations; use a single line or PowerShell backticks on
Windows. Default execution is CPU, fp32, one process, with a frozen encoder and
trainable decision head. `--train-encoder` enables encoder updates with a separate
learning rate. `--device cuda` requires available CUDA; it never silently falls
back. Mixed precision and distributed training are not supported by this runner.

`--max-len` and `--head-max-len` must fit the task. The opt-in path rejects missing,
truncated or token-identical options, and rejects truncated state. Preserving all
markers alone is insufficient: shortened option descriptions can change the
meaning of a task. The ordinary SDK's existing truncation behavior is unchanged.

## Recovery contract

The run directory contains `run.json`, `checkpoint.pt`, `.run.lock` and
`result.json`. Only a completed run publishes `export/`. Existing directories
require `--resume`; a new run never overwrites another run.

`run.json` binds the dataset manifest, source file hashes, tokenized training
items, settings, implementation files, Python/library versions, device and thread
configuration. A changed identity is rejected on resume. This intentionally does
not promise exact resume across library upgrades, code edits, operating systems
or hardware. Preserve the recorded environment and source snapshot.

Checkpoints contain weights, optimizer and scheduler state, Python/NumPy/Torch
random states, epoch permutation position and counters. Saves occur only at
optimizer boundaries; each update averages over its actual examples, including
uneven final microbatches. An interrupted update is replayed from the previous
saved boundary. `--checkpoint-every` controls this interval; epoch ends and a
requested `--max-updates` stop also save. `--max-updates` limits additional updates
in that invocation without changing the planned learning-rate schedule.

Writes stage a file beside its destination, flush it and atomically replace the
previous checkpoint. A kernel lock prevents simultaneous writers and releases on
process death. These are local-filesystem guarantees, not a distributed locking
or arbitrary network-filesystem durability protocol. Only load training state
from trusted runs; loading uses PyTorch's `weights_only=True` mode.

## Export and calibration

Changed weights invalidate inherited calibration. Every new training export sets
per-type temperatures to `[1, 1, 1]`, clears `temperature_by_options`, resets the
temperature tensor and records `calibration_status: unfitted`. It never silently
reuses a bucket override from the base model. Existing checkpoints retain their
intentional bucket precedence when loaded normally.

The export is an ordinary local SDK checkpoint, reloadable with
`laya.load("runs/tickets-v1/export", device="cpu")`. Its manifest records file hashes
and completed-run provenance. Fit fresh calibration on the separate calibration
partition before evaluating a review policy. Successful training or export is
configuration evidence, not an accuracy or calibration improvement claim.

## Verification

```shell
python tests/test_adapt_data.py
python tests/test_adapt_train.py
```

The tests construct a tiny local BERT and tokenizer. They check exact equality of
uninterrupted and resumed model, optimizer, scheduler, RNG and cursor state;
uneven accumulation; encoder freezing/updating; source preservation; stale
calibration clearing; SDK reload probabilities; identity rejection; failed writes;
and lock release after terminating a fixture subprocess. No weights are downloaded.
