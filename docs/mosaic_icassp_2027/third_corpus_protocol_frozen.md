# Frozen prospective protocol: SOMOS v2

Status: frozen primary protocol. Any edit to this file invalidates its recorded
SHA-256 and requires a new protocol version. At freeze time, no SOMOS audio,
item-level MOS file, raw score file, or MOS value had been downloaded, opened,
or analyzed in this project.

## Scientific question

Does target-label calibration improve a prospectively fixed public bank of
speech-quality predictors on a TTS naturalness corpus whose official test split
contains unseen systems, listeners, and texts?

The primary estimand is utterance-level naturalness prediction on the official
SOMOS clean test split. This is a domain and system generalization test, not an
enhancement-quality replication.

## Dataset repository and immutable release

- Dataset: SOMOS, the Samsung Open MOS Dataset for the Evaluation of Neural
  Text-to-Speech Synthesis.
- Publisher: Zenodo.
- Record and version: v2, `10.5281/zenodo.7378801`, published 2022-11-30.
- Archive: `somos.zip`, 4.0 GB, published MD5
  `bdfde4cae256549dfab05d713136e4af`.
- Release partitions: `training_files/split1/clean/`.
- Official split files:
  `train_mos_list.txt`, `valid_mos_list.txt`, and `test_mos_list.txt`, paired
  with `TRAINSET`, `VALIDSET`, and `TESTSET`.
- Release design: 70/15/15 train, validation, and test, with unseen systems,
  listeners, and texts in the official split.

Only version v2 and the exact archive hash above are eligible. A changed archive
or another split requires a new protocol.

## Exact target column and item schema

The exact target column is `mos` from each clean split's `*_mos_list.txt` file.
The corresponding item identifier is `utt_id`. The target range is 1 through 5.
Group identifier: `source_group`, derived from the utterance ID by removing its
final underscore plus three-digit system token.
One normalized row contains:

```text
sample_id      utt_id from the clean split file
source_group   utt_id stem with the final underscore plus three-digit system removed
system_id      final three-digit token in the utt_id stem
split          train, valid, or test from the official clean split
mos            official clean mean naturalness score
predictor_*    frozen public-bank scores
```

The extraction rule must match `^(?P<source_group>.+)_(?P<system_id>\d{3})\.wav$`.
Every ID must match exactly once. System `000` denotes natural LJ Speech and is
retained.

## Exact predictor bank

The primary bank is fixed before labels at 27 outputs from ten public runners.
No locally trained D3 or U3-HuBERT output is included.

1. DNSMOS P.835 ONNX, overall and signal outputs. Local artifact SHA-256:
   `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`.
2. DNSMOS P.808 ONNX, one output. Local artifact SHA-256:
   `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91`.
3. TorchAudio SQUIM Objective, PESQ, STOI, and SI-SDR outputs, using
   `torchaudio==2.11.0` and `SQUIM_OBJECTIVE`.
4. NISQA MOS, repository `gabrielmittag/NISQA` at commit
   `fe84f0f252abec382b24367d5b22498a7ce34dbb` and its `weights/nisqa.tar`.
5. Distill-MOS, repository `microsoft/Distill-MOS` at commit
   `98c0a156b5dabf2b5a8fe9cee92145cdc2a2dcdb` and
   `ConvTransformerSQAModel`.
6. SCOREQ natural and synthetic non-reference outputs, repository
   `alessandroragano/scoreq` at commit
   `0cb0b168d0f7ec1419475d1e7b7ea699d8cd599e`.
7. UTMOSv2 pretrained fold-0 MOS, repository `sarulab-speech/UTMOSv2` at
   commit `cc2700db57bb83ee13dc31ebe1b868c254e15d09`.
8. SIGMOS overall, signal, noise, coloration, discontinuity, loudness, and
   reverberation outputs, repository `microsoft/SIG-Challenge` at commit
   `bf4525153b6ed998f19d9e79ff1fd00f55dec42b`.
9. Audiobox Aesthetics PQ, CU, CE, and PC outputs, code repository
   `facebookresearch/audiobox-aesthetics` at commit
   `2618e9d451b456e9328b39495b5e6234678aa550` and model repository
   `facebook/audiobox-aesthetics` at revision
   `9b1dd8e5df9af7216e836a98974fe3b82c56ded6`.
10. Uni-VERSA MOS, SCOREQ, UTMOS, NISQA MOS, and DNSMOS overall outputs,
    model repository `vvwangvv/universa-ext_wavlm-base_5metric` at revision
    `1fe08f4897655bf91e9b893030af872fa2a91694`.

All model weights downloaded by a runner must be hashed before its first score
is accepted. The environment lock, runner source hash, weight hashes, audio
preprocessing, row count, and output schema must accompany every score shard.
The intended preprocessing is mono audio with each runner's documented sample
rate; any resampling implementation and version must be recorded.

## Complete-case and failure rules

- The primary matrix uses one shared complete-case mask across all 27 outputs
  and every compared method.
- The primary result is valid only if every output is finite and nonconstant on
  the training split and at least 95 percent of each official split survives
  the shared mask.
- A runner implementation may be corrected without opening MOS values. Every
  failed shard and correction remains logged.
- If the 27-output primary bank is infeasible, it is reported as infeasible and
  is not silently redefined.
- A fixed secondary six-output bank may still be reported: Distill-MOS,
  DNSMOS overall, NISQA MOS, SCOREQ natural, UTMOSv2, and SIGMOS overall. It
  cannot replace the primary outcome.
- No row may be dropped separately for one method. No imputation is permitted.

## Methods and hyperparameters

The exact six controls are:

1. Best single output, selected on validation SRCC.
2. Equal-weight average of percentile features.
3. Standardized raw-score ridge.
4. Percentile-feature ridge.
5. Percentile-feature non-negative least squares.
6. Standardized sparse percentile-feature lasso.

Ridge alpha is selected from `{0.1, 1, 10, 100, 1000}`. Lasso alpha is selected
from 16 logarithmically spaced values from `1e-4` through `1e-1`. Selection uses
only the official validation split. Ties choose the stronger regularization,
then the lexicographically first method or output ID where applicable.

Raw-feature standardization is fitted on training values only. Percentiles use
an empirical CDF fitted on training values and applied unchanged to validation
and test values. Test-batch score distributions are never used to define rank
features.

## Label budgets

The fixed training-label budgets are 200, 500, 1,000, 2,500, and the full
official training split. Ten acquisition orders use seeds 0 through 9. For each
seed, smaller budgets are prefixes of one uniform permutation of training rows.
Validation and test rows never enter a budget fit. The same budget rows are used
for all six methods within a seed.

The full-data fit is the primary analysis. Budgets are secondary and report the
paired budget-minus-full gap for each acquisition seed.

## Primary comparison and metrics

- Primary metric: utterance-level Spearman correlation on the official clean
  test split.
- Primary comparison: standardized raw ridge minus equal ranks for the
  27-output bank.
- Primary uncertainty: 10,000-draw paired cluster bootstrap, resampling
  `source_group` and recomputing both test SRCC values and their difference.
- Secondary metrics: rank-ridge comparison, best-single comparison, system-level
  SRCC after averaging predictions and MOS by `system_id`, MAE, and Pearson
  correlation.
- Budget summary: mean and sample SD across the ten fixed acquisition orders,
  plus paired budget-minus-full gaps.

No null-hypothesis p-value or equivalence claim is permitted. Report point
estimates and percentile intervals. A confidence interval crossing zero is
described as inconclusive, not as evidence of no difference.

## Frozen exclusions and claim boundary

- Use only the official `split1/clean` release. Raw and full listener-score files
  are excluded from the primary experiment.
- Natural system `000` is retained. No system, sentence, or MOS-range filtering
  is permitted beyond the shared finite complete-case rule.
- Compactness selection is not part of the prospective primary analysis.
- Predictor weights, methods, feature transforms, hyperparameter grids, budgets,
  seeds, grouping, and metrics cannot change after target retrieval.
- Any later ablation, alternate split, alternate grouping, or changed predictor
  bank is labeled post hoc and stored separately.

## Reproduction and retrieval record

Before retrieval, run `scripts/freeze_validation_protocol.py` on this file and
archive both the protocol and its sidecar in an append-only remote or
institutional timestamp service. Retrieval must record the Zenodo endpoint,
DOI, archive MD5, local SHA-256, byte size, time, and extraction inventory.

The analysis must retain fitted hyperparameters, test predictions, acquisition
orders, complete-case IDs, bootstrap seed, all score-shard provenance, and a
canonical JSON report. A failed primary analysis is retained as a result.
