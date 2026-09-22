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

Training candidates start independently from the pinned multilingual checkpoint.
Later candidates need not run if an earlier one qualifies. Neither repeated
final-test attempts nor replacing a failed model using final-test results is part
of this protocol. A failure to qualify is a measured limitation, not permission
to lower the gate after seeing outcomes. Base-model pretraining exposure to this
public corpus is unknown; the split protects adaptation-stage independence only.

## Execution evidence and limits

A CPU pilot used four training-only examples from two source families, the actual
322M multilingual checkpoint, and the complete 77-option schema. All option text
remained distinct and untruncated at `max_len=1024`, `head_max_len=768`. On the
tested Linux host with two Torch threads, four warm forward passes took 8.46 s;
the process's peak resident memory was 4,887 MiB. Interrupted/resumed training
produced byte-identical exported checkpoint files to an uninterrupted run, and
source assets were unchanged. These timings include this host's conditions and
are not deployment throughput estimates.

That pilot establishes execution and recovery only. It does not establish
accuracy, calibration improvement, or an achievable automation rate. The full
adaptation and independent evaluation are still required. Synthetic CPU tests
can be run separately without obtaining the corpus or pretrained weights:

```shell
python tests/test_arbanking_adapter.py
python tests/test_adapt_train.py
```
