# Independent evaluation and bound calibration

`laya.adapt_eval` collects **raw, unrounded logits** from one explicit partition
of a [prepared dataset](adaptation_data.md). It uses the local checkpoint helpers
and the SDK's sequence/model implementation. Existing checkpoint temperatures do
not affect these records. The prediction API is unchanged.

```shell
python -m laya.adapt_eval --dataset prepared-tickets --model runs/tickets-v1/export \
  --split calibration --output runs/tickets-v1/calibration-logits
python -m laya.adapt_calibration --dataset prepared-tickets \
  --predictions runs/tickets-v1/calibration-logits/predictions.json \
  --output runs/tickets-v1/calibration.json
python -m laya.adapt_eval --dataset prepared-tickets --model runs/tickets-v1/export \
  --split policy --output runs/tickets-v1/policy-logits
```

Use one line or PowerShell continuation syntax on Windows. Specify token budgets
with `--max-len` and `--head-max-len` when needed, and use identical rendering for
calibration, policy selection, final evaluation and serving. `--device` defaults
to CPU. Inputs that lose instruction, option or state text are rejected.

## Resumable evidence collection

Every task selects one labelled representative per connected family and language,
using the frozen split's IDs and seed, never model scores or gold-label values.
The canonical connected-family ID survives into the prediction record, even when
different original source IDs were merged as duplicates. This prevents related
examples from appearing independent when passed to a statistical policy test.

The output directory contains a frozen `collection.json`, atomically written
`chunks/`, and a final `predictions.json` only after completion. `--resume` verifies
model assets, dataset, schema, rendering, tokenized items, library/code versions,
thread settings and all saved chunks. A failed forward pass loses only the current
unsaved chunk. `--chunk-size` defaults to 64 examples; `--batch-size` controls
forward-pass batching separately. `--max-chunks` can stop an invocation without
changing the complete collection plan. One kernel lock prevents concurrent writers.

The final artifact binds source file hashes, the ordered question schema and the
dataset manifest. It records runtime and forward-collection elapsed time; this
timing excludes loading and input preparation and must not be presented as full
request latency. The collector never modifies the source model or dataset.

`verify_prediction_dataset(artifact, dataset_dir)` rechecks exact partition
membership, canonical families, independent selection and gold targets. A changed
partition tag or a changed target cannot become valid evidence by recomputing just
the artifact's record hash. Hashes provide integrity, not authentication against
someone replacing every input and trusted reference.

## Fitting and metrics

The calibration CLI verifies the artifact against the prepared dataset and fits
only the calibration partition. Hard-label NLL is convex in inverse temperature;
the fitter bisects its derivative inside the SDK's `[0.5, 5.0]` range. Missing types
remain neutral. A present type with fewer than 20 examples fails explicitly.

The output uses the SDK's `temperature` and `temperature_by_options` keys, also
used by [PR #19](https://github.com/NandhaKishorM/laya/pull/19). The opt-in adaptation
artifact adds exact source/model/schema bindings and sample identities. It does
not depend on installing that unmerged PR. Per-type fitting starts with an empty
bucket map, so inherited overrides cannot silently survive. `--fit-buckets` fits
only new buckets with at least 2,000 calibration records; valid buckets take
precedence over per-type fallback. These count minima are checks, not guarantees
of calibration quality.

`metric_report(predictions, questions, calibration)` reports, separately for each
task and language:

- Accuracy, NLL computed from log probabilities, and Brier summed over classes.
- ECE over 15 equal-width bins, including the confidence-one endpoint.
- For ordinal score tasks, expected-level MAE and ranked probability score
  normalized over the `K - 1` ordered boundaries.

Fitting-set metrics are marked with their partition. They are descriptive fit
diagnostics, not independent improvement evidence. Temperature fitting preserves
argmax labels, so it cannot by itself improve classification accuracy. Report
held-out scores separately and retain all predeclared languages.

The lower-level fitting and metric functions validate the artifact and schema;
call `verify_prediction_dataset` when loading external prediction artifacts.
`probability_records` additionally checks calibration binding and rejects reuse
of calibration IDs or related families on policy/test partitions. Its full
precision probabilities can feed [the risk-policy utilities](risk_policy.md);
the score is maximum class probability, not SDK entropy confidence.

For persisted policies, use `laya.adapt_policy.select_bound_policy(policy_logits,
calibration, dataset_dir, question_id="queue", required_languages=["en", "ar"],
alpha=0.0125)`. It checks the actual dataset partition and binds the exact model,
rendering, ordered schema, calibration and policy evidence. Choose alpha according
to the predeclared number of model candidates; the illustrated value allocates a
5% family probability across four models. An optional `declaration` records the
trusted protocol identifier. A failed selection remains a failed selection.

Persist the selected policy before collecting final-test predictions. Then call
`evaluate_bound_policy(test_logits, policy, calibration, dataset_dir)` to recompute
the independent test bounds and metrics. It refuses failed selections, wrong
partitions and changed calibration/model/schema links. The bound report carries
the policy fingerprint so a deployment exporter can verify which decision rule
was tested. These helpers do not authenticate arbitrary replacement of every
artifact or prevent final-test reuse outside the experiment coordinator.

Final-test collection is explicit (`--split test`). Run it only once a model,
calibration and threshold policy are locked. These files do not prevent someone
from inspecting the dataset outside the experiment, and this stage alone does
not publish or authorize a deployment bundle.

```shell
python tests/test_adapt_evaluation.py
python tests/test_adapt_policy.py
```

Tests use synthetic logit distributions and tiny local checkpoint fixtures. They
establish the collection, fitting and metric contracts without downloading weights
or measuring real-task accuracy.
