# Kaggle Mel-on Rank Experiments (M1/M2) — Design

- **Date:** 2026-06-09
- **Status:** Approved (conversational)
- **Author:** Vrishab + Claude

## Context / Problem

Experiment A1 (rank-aware ACR loss, **mel branch OFF**, final layer) scored **CodaBench
ACR UTT-SRCC 0.534 / CCR 0.314** — a regression versus the **0.574 / 0.369** baseline.
A1 changed two things from the baseline at once (removed the mel-CNN branch **and** added
rank loss), so the cause is confounded. The mel branch was a *speed* sacrifice for free
Colab (to dodge per-epoch audio decode), never a quality decision; the baseline had mel-on
and scored higher. The learnable spectrogram CNN likely captures enhancement
artifacts the frozen ASR encoder misses — the wrong thing to drop for a quality task.

Separately, A1's offline `pretrain_dev` SRCC was **0.6715** yet CodaBench came in *below*
baseline → in-distribution rank does **not** reliably predict CodaBench (domain shift).
**CodaBench is the metric of record; `pretrain_dev` is only a weak guardrail.**

## Goals

1. Restore the mel branch and run two experiments on Kaggle, **sharing one encoder cache**:
   - **M1 = mel-on + rank** (`acr_rank_alpha=1.0`) → `submission_mel_rank.zip`
   - **M2 = mel-on + no-rank** (`acr_rank_alpha=0.0`, clean control) → `submission_mel_norank.zip`
2. **Cleanly attribute the rank effect** with mel present: M1 vs M2 under an identical pipeline
   (the 0.574 baseline came from a different/original pipeline, so it is not a clean control).
3. Bake in today's hard-won Kaggle lessons so the run can't hang or mis-package.

## Non-Goals

- **Waveform/mel caching** — would cut per-epoch decode but needs new code, dataset changes,
  and ~4.5 GB extra disk that would breach the ~57.6 GB Kaggle cap (encoder cache ~45 GB +
  raw audio ~10 GB ≈ 55 GB already). Deferred (YAGNI).
- Layer selection (A2), Approach B data (NISQA/de-weight BVCC) — later.
- No changes to `src/model.py`, `src/losses.py`, `src/train.py` (mel-on + cached encoder
  feats + rank already work; only configs + a notebook are new).

## Approach

**Config-only parallel decode.** Set `use_mel_branch: true` (so `train.py` sets
`load_waveform=True`) and `num_workers: 4`, so the DataLoader decodes audio across Kaggle's
~4 cores in parallel each epoch (~13 min/epoch). Zero new code, zero extra disk, reliable.
Both runs reuse the **same final-layer encoder cache** (the Whisper branch is identical to A1).

## Design

### Components

**`configs/exp_mel_rank_kaggle.yaml`** and **`configs/exp_mel_norank_kaggle.yaml`** — clones of
`exp_a1_kaggle.yaml` with:
- `model.use_mel_branch: true`, `model.encoder_layer: -1`
- `training.acr_rank_alpha: 1.0` (rank) / `0.0` (norank)
- `training.epochs: 8` (A1 overfit after ~3–5; `best.pt` keeps the peak)
- `training.checkpoint_every_n_epochs: 2` (intermediate checkpoints, in case the
  `pretrain_dev`-selected `best.pt` overfits relative to CodaBench)
- `training.num_workers: 4`
- `cache_dir: /kaggle/temp/encoder_cache` (shared with both)
- `checkpoint_dir: /kaggle/working/checkpoints/exp_mel_rank` and `.../exp_mel_norank`
- **no** `drive_checkpoint_dir`

**`notebooks/make_kaggle_mel_notebook.py` → `notebooks/kaggle_mel.ipynb`** — run-all generator
mirroring `make_kaggle_notebook.py`.

### Data flow (notebook cells)

1. Probe `df -h` + `nvidia-smi` (no assumptions).
2. Clone/refresh repo → `/kaggle/working/voicemos-2026`; `pip install`; define fail-fast `sh()`.
3. HF token via `kaggle_secrets.UserSecretsClient`.
4. Download BVCC + TMHINT → `/kaggle/temp/datasets`.
5. Build manifests (assert `source` column) + prepare dev set.
6. Cache final-layer encoder features → `/kaggle/temp/encoder_cache` (skip-guarded;
   `--num_workers 4`). Shared by M1 and M2.
7. **Set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`** (model is cached by now) so all
   subsequent `sh()` train/predict subprocesses inherit it and can't network-hang.
8. Train M1 (`exp_mel_rank_kaggle.yaml`) → predict → `/kaggle/working/submission_mel_rank.zip`.
9. Train M2 (`exp_mel_norank_kaggle.yaml`) → predict → `/kaggle/working/submission_mel_norank.zip`.
10. Print both offline `pretrain_dev` SRCCs (guardrail) + submission paths.

### Lessons encoded (from today)

- **Offline mode before train/predict** prevents the `from_pretrained` network hang.
- **Predict is parallel** (`predict_dev` `num_workers=4`, shipped `0ce0321`) and zips as
  **`predictions.csv`** (shipped `0cd8db4`) — so submissions score instead of returning NA.
- `train.py` fail-loud guards (smoke check, per-epoch NaN diagnostic, always-`last.pt`,
  `best.pt` fallback; `8adc50e`) carry over unchanged.

## Error Handling

Reuse existing `train.py` safety net. Notebook `sh()` is fail-fast with unbuffered output.
Manifest `source`-column assert stays. Offline mode + parallel predict + correct arcname
are the three packaging/run fixes from today, now defaults.

## Testing

- **Config-shape checks (no GPU):** both configs load; `use_mel_branch is True`;
  `acr_rank_alpha` is 1.0 / 0.0 respectively; `drive_checkpoint_dir` absent;
  `cache_dir == /kaggle/temp/encoder_cache`.
- **Notebook-content asserts:** generated `kaggle_mel.ipynb` parses; contains `kaggle_secrets`,
  the `df`/`nvidia-smi` probe, `HF_HUB_OFFLINE`, both config filenames, both submission zip
  names; no `google.colab`.
- **Regression:** existing 46 tests stay green (no `src/` changes).
- **Manual (Kaggle):** cell-1 probe shows a live GPU + `/kaggle/temp` headroom; `[smoke]`
  line non-NaN; both `submission_*.zip` produced; upload both to CodaBench.

## Risks & Fallbacks

- **Disk cap (~57.6 GB):** encoder cache (~45 GB) + raw audio (~10 GB) ≈ 55 GB; no waveform
  cache keeps us under. If caching reports disk-full, fall back to a trimmed train manifest.
- **Per-epoch decode (~13 min):** parallelized at `num_workers=4`; `epochs: 8` bounds it.
- **`best.pt` may not transfer** (pretrain_dev ≠ CodaBench): `checkpoint_every_n_epochs: 2`
  keeps intermediate checkpoints to submit if the peak overfits.
- **DataLoader workers crash (shm):** fall back to `num_workers: 2` (still parallel).
