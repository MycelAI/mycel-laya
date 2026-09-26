# Verified selective-decision bundles

`laya.adapt_bundle` exports an ordinary calibrated SDK model together with a
verified decision policy. Its opt-in runtime returns either `automate` or
`review`. It does not route tickets or take other external actions itself, and
does not change `Agent.predict`, `Agent.system_one` or existing checkpoints.

## Publish from locked evidence

Prepare data, train if needed, collect independent logits, and fit calibration as
described in [training](adaptation_training.md) and
[evaluation](adaptation_evaluation.md). Select a bound policy on policy data and
persist it before collecting the final test. The model, schema, calibration,
threshold grid and experiment's model-candidate comparison budget must already be
fixed. A policy that fails selection cannot authorize final-test model selection.

```python
import json
from pathlib import Path
from laya.adapt_bundle import export_bundle

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

receipt = export_bundle(
    "deploy/tickets-v1",
    dataset_dir="prepared-tickets",
    model_dir="runs/tickets-v1/export",
    calibration_predictions=read("runs/tickets-v1/calibration-logits/predictions.json"),
    policy_predictions=read("runs/tickets-v1/policy-logits/predictions.json"),
    calibration=read("runs/tickets-v1/calibration.json"),
    policy=read("runs/tickets-v1/bound-policy.json"),
    test_predictions=read("runs/tickets-v1/test-logits/predictions.json"),
    receipt_path="release-receipt.json",
)
```

The exporter verifies dataset membership and family identities, reproduces the
temperature fit and policy selection, verifies final-test bounds, and checks the
source checkpoint and rendering implementation. Rechecking the same locked
predictions is deterministic integrity verification; it supplies no additional
statistical trial or opportunity to pick a different model.

Publication stages a new directory and renames it only after validation. Existing
destinations are refused. The original model and source data are unchanged.
An optional external `receipt_path` is written before the final rename so an
interrupted caller can recover the exact publication. An existing receipt must
match; it is never replaced with a different release. If only the receipt exists,
repeat the same export to finish publication before trying to load the bundle.
`model/` contains the standard SDK files with freshly fitted per-type temperatures
and **only the newly fitted bucket map**. Intentional new buckets retain precedence.
Inherited overrides cannot survive by merging an old configuration. Raw ticket
text and prediction records are not packaged; calibration/policy metadata retains
hashed sample identities, which are not an anonymization guarantee.

The manifest hashes every model, tokenizer, configuration, question, calibration,
policy, final-report and runtime file. Original model/tokenizer configurations
are retained under `source-config/` so loading can verify that only the declared
calibration, rendering and tokenizer compatibility changes were made. Unchanged
weights, encoder configuration and tokenizer assets must match the evaluated
source hashes. Keep `manifest_sha256` from the export
receipt in a separately trusted release record. Reading a checksum from the same
untrusted archive does not authenticate that archive.

`python -m laya.adapt_bundle --help` exposes the same export operation for file-based
workflows. Use `--receipt release-receipt.json` to save its JSON receipt outside the
bundle: adding files to a published bundle invalidates its inventory.

## Load and decide

Use the recorded Python/library versions and CPU thread setting. The first bundle
runtime supports CPU fp32; GPU numerical behavior is not asserted by CPU evidence.
Renderer and inference source fingerprints are checked, normalizing only Git's
Windows/Linux line endings. Other implementation changes need a verified release.

```python
import json
from pathlib import Path
import torch
from laya.adapt_bundle import load_bundle

receipt = json.loads(Path("release-receipt.json").read_text(encoding="utf-8"))
torch.set_num_threads(2)  # Use the setting recorded during this example's evaluation.
bundle = load_bundle(receipt["directory"], expected_manifest_sha256=receipt["manifest_sha256"])
result = bundle.predict({"message": "I was charged twice."}, language="en")
if result["status"] == "automate":
    queue = result["value"]
else:
    review_reason = result["reason"]
```

The question schema is fixed by the bundle. Runtime policy decisions use unrounded
maximum calibrated class probability with the same `>= threshold` comparison used
in evaluation. They do not use the SDK's rounded probabilities, entropy confidence
or action-head output.

| Outcome | Meaning |
| --- | --- |
| `automate / qualified_policy` | The input's declared language has a qualified rule and its score meets that rule |
| `review / below_threshold` | No automatic value; a separate suggestion may help human review |
| `review / unsupported_language` | The language was not included in the policy |
| `review / empty_or_invalid_state` | Missing content or an unsupported top-level input type |
| `review / unsupported_input` | Invalid JSON values or text lost at the configured token budget |
| `review / non_finite_prediction` | Model logits are unusable |
| `review / inference_unavailable` | Inference failed; no automatic value is returned |
| `review / policy_not_qualified` or `review_only_bundle` | This bundle cannot authorize automation |

For `choice`, the value is the exact label. For `noul`, it is a boolean. For
`score`, it is the **argmax rubric level**, since the policy's loss is exact-level
classification error; the expected score is separate diagnostic information.
An exact-level error bound is not an ordinal MAE guarantee.

Language is supplied by trusted application metadata, not detected by this
wrapper. Structural validity and high probability do not establish semantic
membership in the evaluated domain. In particular, a closed-set banking-intent
policy is not an out-of-domain detector or a guarantee under distribution shift.

## Review-only publication and SDK compatibility

Use `review_only=True` and omit test predictions to publish a bundle that always
returns review, including when no threshold meets the selection gate. It does not
load the neural model in the selective wrapper. Deferring every example is useful
as a fallback but **does not satisfy the automation coverage goal**. A qualified
bundle requires a passing independent final evaluation; missing or failed evidence
cannot be turned into an automatic mode merely by setting a flag.

The calibrated `model/` is also loadable by `laya.load(path, device="cpu")`. Those
ordinary SDK calls retain their existing response format and do not enforce this
bundle's review policy. Applications seeking selective decisions must use the
verified wrapper and check its `status` before acting.

```shell
python tests/test_adapt_bundle.py
```

Tests use synthetic evidence and tiny local weights. They check checksum and
internal-link failures, lost evidence, uncertainty bounds, runtime differences,
SDK probability agreement, temperature replacement, bucket persistence, atomic
publication, typed values, threshold equality and explicit review paths. They do
not establish real-model accuracy or successful deployment coverage.
