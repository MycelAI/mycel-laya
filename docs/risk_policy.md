# Selecting and testing review thresholds

`laya.risk` selects categorical decision thresholds and evaluates them on a
separate final test. It is an opt-in, offline utility. It leaves `Agent.predict`
unchanged and does not itself authorize deployment or load a model.

The score used here is the largest class probability. The SDK's `confidence`
field is based on normalized entropy; do not substitute it for the probability
vector. High scores alone are insufficient evidence of a low error rate.

## Input contract

Supply independent labelled predictions for one fixed categorical question, with
one record per connected example family and language:

```json
{
  "id": "ticket:42:en",
  "group_id": "conversation:42",
  "language": "en",
  "probabilities": [0.91, 0.09],
  "target": 0
}
```

Use the full, unrounded probability vector in the fixed option order and an
integer gold option index. Vectors need at least two finite, nonnegative values
and must sum to one within `1e-6` numerical tolerance. Ties choose the first
option, matching argmax. Repeated IDs or repeated groups within a language fail
validation. Groups can appear in several languages, but must remain entirely
within one partition. Group IDs and example IDs must be globally unique across
data sources; keep translation, conversation and augmentation families together.

Select representatives before inspecting predictions. Correct grouping and a
representative sample are prerequisites; declaring different IDs cannot make
dependent examples independent. If using the adaptation dataset contract, use
its connected-group IDs rather than the raw, unmerged source family IDs.

## Selection and independent evaluation

Freeze the model, question and option definitions, calibration, required
languages, error budget, coverage target and threshold grid before examining
selection outcomes. The defaults are a 5% error budget, at least 50% coverage and
a total failure probability `alpha=0.05`. The grid must be fixed independently
of the selection-set outcomes; do not tune it after seeing which trials pass.

```python
from laya.risk import evaluate_risk_policy, select_risk_policy

# selection_samples and final_samples are independently prepared predictions.
selection = select_risk_policy(
    selection_samples,
    required_languages=["en", "de"],
    thresholds=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    max_error=0.05,
    min_coverage=0.5,
    alpha=0.05,
)
if selection["passed"]:
    final_report = evaluate_risk_policy(final_samples, selection)
    print(final_report["passed"])
```

An example is selected when its largest probability is at least the threshold.
For each language and candidate, the report contains:

| Field | Meaning |
| --- | --- |
| `samples`, `accepted`, `errors` | Independent representatives, selected decisions and incorrect selected decisions |
| `coverage`, `error_rate` | Observed selected fraction and selected error fraction; error is `null` if none are selected |
| `error_upper` | One-sided exact binomial upper confidence bound on selected error |
| `coverage_lower` | One-sided exact binomial lower confidence bound on coverage |
| `qualified` | Both bounds meet their respective predeclared budgets |

Bounds use the [Clopper-Pearson method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html),
implemented by binomial CDF inversion without adding SciPy as a dependency.
Selection allocates `alpha / (2 * languages * thresholds)` to each bound, so
Bonferroni accounts for choosing among multiple candidates and reporting multiple
languages and metrics. Among qualifying thresholds it chooses the most accepted
examples, breaking ties in favor of the higher threshold. Every required language
must qualify; pooling successful and unsuccessful language slices cannot pass.

An independent final test allocates `alpha / (2 * languages)` per bound because
each language's threshold is already fixed. Reusing a selection-set ID or related
group is rejected, including reuse across languages. Empty test slices, small
samples, observed errors inconsistent with the budget and insufficient coverage
fail. Deferring everything cannot pass. Never retune a failed policy against the
same final test, repeatedly test candidates until one passes, or omit a failed
language from the protocol; those actions invalidate the declared procedure.

## Scope of the evidence

The simultaneous bounds concern independent draws from the evaluated population,
conditional on the frozen upstream model and calibration. They are not guarantees
under future distribution shift, incorrect labels, undetected dependence or a
different sampling scheme. The separate selection and final-test reports each
use their own stated family error probability. Report which experiment a claim
refers to; these are not a combined sequential confidence guarantee.

The selection report hashes sample identities so the final-test function can
reject reuse. It also records a content hash of the supplied predictions. These
are consistency aids, not authentication: callers must bind the report to the
exact checkpoint, tokenizer, schema, calibration, data manifest and frozen
protocol in a trusted deployment bundle. Matching vector lengths does not verify
label ordering or the model that produced them. This utility cannot verify how
samples were obtained, whether a final set was inspected elsewhere, or whether
the supplied report was replaced.

Run `python tests/test_risk.py` for offline regression checks. These use synthetic
predictions, published interval references, exact small-sample coverage
enumeration and high-precision tail calculations. They establish statistical
implementation behavior; they do not measure Laya's accuracy, calibration quality
or achievable automation rate on support tickets.
