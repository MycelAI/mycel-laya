# Measure a fixed selective deployment

Run the measurement recipe after choosing the model, locking its policy and
finishing its independent final evaluation. Use a **qualified** bundle and the
same bundle files and trusted manifest checksum on both hosts. The recipe refuses
review-only bundles: their fast review response does not measure neural serving.

The [versioned English/MSA probe](../research/arbanking77_deployment_probe.json)
contains six manually written text/JSON cases, including paraphrases and noisy
formatting, plus four structural review cases. It has no gold labels. Predictions
on these examples describe observed behavior, not accuracy, semantic robustness,
an OOD guarantee or coverage on a representative distribution. Do not use these
results to choose another model, refit temperatures or change the final policy.

## Run on each host

Use a fresh Python process, the bundle's recorded library versions and CPU thread
count. Preserve existing reports and choose a new output filename for each run.
The context argument records relevant conditions such as other running jobs.
For example, when the selected bundle was evaluated with two CPU threads:

```shell
python research/scripts/benchmark_adaptation_bundle.py run --bundle deploy/arbanking77-v1 --expected-manifest-sha256 TRUSTED_BUNDLE_SHA256 --probe research/arbanking77_deployment_probe.json --output measurements/windows.json --threads 2 --context "Windows CPU; describe concurrent work here"
```

Repeat on Linux using the same bundle and probe, recording `measurements/linux.json`
and the actual Linux host conditions. Replace the checksum placeholder with the
value from the separately preserved publication receipt. Neither command downloads
weights or opens the labelled final-test dataset.

The report retains each measured call and outcome, p50/p95 per case, implementation
and probe fingerprints, runtime/host metadata, and these distinct measurements:

| Measurement | Scope |
| --- | --- |
| `verified_load_ms` | Manifest/file checks, evidence checks and model construction; Python/import startup excluded |
| Prediction-case latency | Complete serial `predict`: validation, rendering, tokenization, forward pass, calibration and policy/output construction |
| Review-case latency | Structural guards, reported separately from neural inference |
| Resident memory | Current process RSS on Linux or working set on Windows |
| Peak resident memory | Process-lifetime RSS/working-set peak, including imports and loading |

The operating system's disk cache is not cleared. A verified-load measurement is
therefore not a controlled cold-disk benchmark. Memory is the whole process,
not the model's incremental footprint. Ten serial repetitions give exploratory
latency samples; they do not establish a production tail-latency SLO, throughput
under concurrency or GPU performance. Keep the recorded context with the numbers.

An inference failure, unexpected structural result or missing probability output
fails the measurement instead of being counted as a fast prediction. Raw input
states are represented by case fingerprints in reports; the separate probe file
is needed to reproduce them. Outcome values and probabilities remain in the report.

## Compare outcomes across hosts

```shell
python research/scripts/benchmark_adaptation_bundle.py compare --left measurements/windows.json --right measurements/linux.json --output measurements/comparison.json
```

Comparison requires matching bundle, probe, implementation and inference-runtime
identities. Every repeated outcome is checked against the reference: status,
reason, question, typed decision, suggestion value and threshold must agree;
probabilities use an absolute tolerance of `1e-6`. Timing and host metadata may
differ. Differences are retained in the report and produce a nonzero CLI exit
status. This proves agreement only on these probes, including observed stability
across their repetitions. In particular, it cannot rule out a threshold crossing
for an untested input whose probability lies close to the decision boundary.

```shell
python tests/test_adaptation_benchmark.py
```

Tests use tiny local weights and synthetic qualification evidence. They verify
measurement and comparison behavior, without establishing real-model performance
or accuracy.
