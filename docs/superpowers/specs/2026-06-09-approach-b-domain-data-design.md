# Approach B v1 — Domain-Closer Data (NISQA + TMHINT) Design

- **Date:** 2026-06-09
- **Status:** Approved (full autonomy granted)
- **Author:** Vrishab + Claude

## Context / Problem

Approach A is characterized and tapped out at ~baseline. Best CodaBench results:
baseline **0.574 ACR / 0.369 CCR**; mel-on no-rank (M2) **0.564 / 0.293**; mel-on rank
(M1) **0.557 / 0.336**; mel-off A1 **0.534 / 0.314**. Findings: **mel branch essential,
rank loss not worth it on ACR, none of A's knobs clear baseline.** The gap is a
**domain-transfer** problem, not model capacity — training data (BVCC English TTS-synthesis
MOS + TMHINT Mandarin quality MOS) is far from the test domain (multilingual speech-
**enhancement** quality). Also confirmed: `pretrain_dev` SRCC does NOT predict CodaBench.

## Goal

Train on domain-closer data and beat 0.574. In ONE Kaggle session on a SHARED encoder
cache, produce two submissions and take the winner:
- **B1** = NISQA + TMHINT, raw MOS
- **B2** = NISQA + TMHINT, per-source z-normalized MOS

NISQA (English degraded speech: codecs, packet loss, noise, clipping, bandpass) is a strong
match for enhancement-quality; TMHINT (Mandarin processed-speech quality) keeps multilingual
coverage. BVCC (TTS naturalness) is dropped as the prime transfer-killer.

## Non-Goals

- BVCC retention, AudioMOS, layer selection (A2) — out.
- Rank loss — dropped (no-rank won on ACR).
- predict_saved.py changes — B uses the notebook + predict_dev path.

## Verified facts

- NISQA Corpus: Zenodo `https://zenodo.org/record/4728081/files/NISQA_Corpus.zip`, free for
  research, ~14k samples; CSV `NISQA_corpus_file.csv` has columns **`db`** (split name),
  **`filepath_deg`** (degraded wav path, relative to corpus root), **`mos`** (target). Use
  the `NISQA_TRAIN*` / `NISQA_VAL*` splits (skip license-restricted TEST sets).
- The cache is keyed by MD5 of the audio path. B1 and B2 manifests share identical `path`
  values (only the `acr` column differs), so a single cache serves both.
- SRCC is invariant under any strictly monotonic transform → a linear min→max rescale of
  predictions preserves rank exactly. Required for B2 because the z-norm model predicts
  ~N(0,1) values and the current hard clamp to [1,5] would clip every below-mean prediction
  to 1 and destroy rank info.

## Design

### 1. `data/download.py` — `download_nisqa(output_dir)`
Download `NISQA_Corpus.zip` from Zenodo to `<output>/nisqa/`, extract (yields
`nisqa/NISQA_Corpus/`). Skip-guard if already extracted. Add `nisqa` to `--datasets` choices.

### 2. `data/build_manifests.py` — `parse_nisqa(nisqa_dir)`
Locate the corpus CSV by glob (`**/NISQA_corpus*.csv`) to be robust to the exact dir/file
name; corpus root = the CSV's directory. Read it, **assert** `db`/`filepath_deg`/`mos`
present (fail-loud). Keep rows whose `db` starts with `NISQA_TRAIN` or `NISQA_VAL`. Per row:
`path = join(corpus_root, filepath_deg)` (skip if missing, like BVCC/TMHINT), `acr=mos`,
`ccr=nan`, `language="en"`, `system=db`, `source="nisqa"`.

### 3. `data/build_manifests.py` — composition + normalization flags
- `--datasets {bvcc,tmhint,audiomos,nisqa}+` (default all): only parse/include the requested
  sources, so composition is explicit and reproducible regardless of what's on disk.
- `--normalize {none,per_source_z}` (default none): when `per_source_z`, standardize `acr`
  to mean 0 / std 1 **within each source**. Stats computed on the TRAIN union per source and
  applied to both train and dev (guard std==0 → leave unchanged). Helper
  `normalize_per_source(train_rows, dev_rows)`.

Build twice in the notebook: raw → `data/manifests`, z-norm → `data/manifests_znorm`.

### 4. `scripts/predict_dev.py` — `--score-mode {clamp,rescale}`
- `clamp` (default, current behavior): ACR `clamp[1,5]`, CCR `clamp[-3,3]` (B1).
- `rescale`: linear min→max map of the prediction set to [1,5] (ACR) and [-3,3] (CCR),
  monotonic → SRCC identical, keeps z-score predictions in range (B2).
- Pure helper `rescale_to_range(values, lo, hi)` (unit-tested; max==min → all `lo`).
Refactor ACR/CCR row building to gather raw predictions, then apply the chosen mode.

### 5. Configs (both: mel-on, no-rank, final layer, early_stop_patience 3, epochs 10, nw 4)
- `configs/exp_b1_kaggle.yaml`: manifests `data/manifests`, cache
  `/kaggle/temp/encoder_cache_b`, ckpt `/kaggle/working/checkpoints/exp_b1`, acr_rank_alpha 0.
- `configs/exp_b2_kaggle.yaml`: manifests `data/manifests_znorm`, SAME cache
  `/kaggle/temp/encoder_cache_b`, ckpt `/kaggle/working/checkpoints/exp_b2`, acr_rank_alpha 0.

### 6. Notebook `notebooks/kaggle_approach_b.ipynb` (gen `make_kaggle_b_notebook.py`)
probe → clone → token → download `nisqa tmhint` → build raw manifests
(`--datasets nisqa tmhint`) + znorm manifests (`... --normalize per_source_z --output_dir
data/manifests_znorm`); assert `source` col + print composition → dev set → cache once (from
`data/manifests`) → set HF offline → train B1 → predict B1 (`--score-mode clamp`) → train B2
→ predict B2 (`--score-mode rescale`) → summary with both submission paths + pretrain_dev
SRCCs. **Recommended run mode: "Save & Run All (Commit)" (headless)** — avoids the
interactive-session fragility seen on 2026-06-09; subprocess train/predict get fresh CUDA
contexts (no fork-from-live-kernel hang).

## Error Handling

Reuse `train.py` safety net (smoke check, NaN diagnostic, always-`last.pt`, `best.pt`
fallback, early stopping). NISQA parser asserts schema (fail-loud) and skips missing wavs.
B1's untouched clamp path is the control: a z-norm/rescale bug in B2 cannot mask a genuine
data-composition win.

## Testing (TDD, local, no GPU)

- `parse_nisqa`: fake `NISQA_corpus_file.csv` (`db,filepath_deg,mos`) + fake wavs → rows
  tagged `source="nisqa"`, correct `acr`/`path`; `NISQA_TEST*` rows excluded.
- `build_manifests` `--datasets` selection: only requested sources present.
- `normalize_per_source`: each source's acr mean≈0/std≈1 after; raw mode unchanged.
- `rescale_to_range`: min→lo, max→hi, monotonic; degenerate max==min → lo.
- Regression: existing 49 tests stay green.
- Manual (Kaggle): probe shows disk/GPU; both `[smoke]` non-NaN; B1+B2 submissions built;
  upload both to CodaBench; compare to 0.574.

## Risks & Fallbacks

- **NISQA download size/time** (~few GB): probe + skip-guard; Zenodo is reliable (hosts BVCC).
- **NISQA path/dir/CSV-name drift**: glob-locate CSV + corpus root; assert columns.
- **Disk** (~57.6 GB cap): NISQA + TMHINT audio + one cache — well under (BVCC dropped frees
  room).
- **z-norm/rescale bug**: B1 raw control protects the experiment from a false negative.
