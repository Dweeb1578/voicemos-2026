# Kaggle A1 Runner — Design

- **Date:** 2026-06-09
- **Status:** Approved (conversational)
- **Author:** Vrishab + Claude

## Context / Problem

Approach A experiment **A1** (rank-aware ACR loss, mel-off, final Whisper layer) has
never produced a usable checkpoint on **free Colab**:
- Free-T4 sessions disconnect frequently; the ~45 GB feature cache lives in ephemeral
  `/content` and is too big for the 15 GB Drive, so every cold start re-caches (~1–2h).
- The first complete run hit a silent-save bug (now fixed in `src/train.py`, commit
  `8adc50e`: smoke check, per-epoch NaN diagnostic, always-`last.pt`, `best.pt` fallback,
  non-empty-manifest asserts).

We are switching to **Kaggle Notebooks**, which give a more stable 12h GPU session and
~30 GPU-hrs/week — enough to run A1 end-to-end in a single session.

**Profiling-by-reading finding:** `data/cache_features.py` is the wall-clock bottleneck,
and it is **CPU-bound, not GPU-bound**. Its loop decodes audio + computes log-mel on a
single thread, then runs the GPU encoder, then saves — fully serial, so the GPU idles
while one CPU core decodes. Therefore **multi-GPU (DDP) would not help**; the correct
lever is **parallel data loading** to keep the single GPU fed.

## Goals

1. Run **A1 only** end-to-end on Kaggle in one session → `submission_a1.zip` for CodaBench.
2. Cut caching wall-clock by parallelizing audio decode/mel extraction across Kaggle's
   ~4 CPU cores (overlap CPU and GPU instead of alternating).
3. **No assumptions about Kaggle resources** — the notebook probes `df -h` + `nvidia-smi`
   in its first cell so we size the run against the *real* disk/GPU.

## Non-Goals

- A2 (layer-12 cache) and A3 (no-rank baseline) — deferred until A1 is verified good.
  A3 shares A1's cache and becomes a cheap add-on later.
- **Dual-GPU / DDP** — explicitly rejected: neither caching (CPU-bound) nor head training
  (tiny model, IO-bound) is GPU-bound, so a second GPU buys ~nothing for real complexity.
- Cross-session resume / Google Drive — single-session run; if it dies we re-run.

## Verified Kaggle facts (2026-06)

| Spec | Value |
|---|---|
| GPU | P100 16 GB (chosen) or T4×2 16 GB each |
| Weekly quota | ~30 GPU-hrs, resets Sat 00:00 UTC |
| Session length | 12 h (GPU/CPU) |
| Idle timeout | 20 min (interactive) |
| `/kaggle/working` | ~20 GB, persistent across runs of the same notebook |
| `/kaggle/temp` | ephemeral scratch, **size undocumented → must probe** |
| Internet | toggle in settings (needs phone-verified account) — ON |
| Secrets | `UserSecretsClient().get_secret("HF_TOKEN")` |

The ~45 GB cache cannot fit `/kaggle/working` (20 GB); it must live in `/kaggle/temp`,
whose capacity we confirm via the cell-1 probe before committing to the long cache step.

## Design

### Change 1 — Parallelize `data/cache_features.py`

Replace the serial per-batch decode loop with a `DataLoader(num_workers=N)` so worker
processes prefetch and decode batches while the GPU encodes.

- New small `Dataset` (e.g. `_CacheClips`) over the deduped, not-yet-cached path list.
  `__getitem__` returns `{"input_features": (80,3000) float32, "path": str}` — it does the
  `resample_and_normalize` + `trim_and_pad` + `WhisperFeatureExtractor` work (the CPU cost),
  exactly mirroring `MOSDataset`'s no-cache path.
- Main loop iterates the loader, runs `encoder(...)` (final or intermediate layer), then
  `np.save` each item by `cache_key(path)`. Saving stays on the main process.
- Add `--num_workers` CLI arg (default 4). Keep `--batch_size`, `--layer`, `--manifests`,
  `--cache_dir`, `--whisper_model` unchanged. Resume-by-skip (already-cached filter) stays.
- Expected effect: caching wall-clock ~1–2h → ~15–30 min (bounded by overlapped 4-core
  decode + GPU, not 1-core serial).

### Change 2 — `configs/exp_a1_kaggle.yaml`

Clone of `exp_a1_rankloss.yaml` with Kaggle-appropriate paths:
- `cache_dir: /kaggle/temp/encoder_cache`
- `checkpoint_dir: /kaggle/working/checkpoints/exp_a1`
- **no** `drive_checkpoint_dir` key (so train.py skips all Drive copy logic)
- `num_workers: 4`
- everything else identical (mel-off, `encoder_layer: -1`, `acr_rank_alpha: 1.0`, epochs 15).

### Change 3 — `notebooks/make_kaggle_notebook.py` → `notebooks/kaggle_train.ipynb`

A Kaggle-specific run-all generator (separate from the Colab `make_notebook.py`; the
platforms diverge enough — Drive vs Secrets, paths, probe — that a separate generator is
clearer than parameterizing one). Cells:

1. **Probe (no assumptions):** `!df -h`, `!nvidia-smi`, print torch/CUDA versions.
2. **Repo + deps:** `git clone` → `/kaggle/working/voicemos-2026`, `cd`, `pip install -r
   requirements.txt -q`. Define a fail-fast `sh()` with `PYTHONUNBUFFERED=1`.
3. **HF token:** `from kaggle_secrets import UserSecretsClient` →
   `os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')`; enable `hf_transfer`.
4. **Download datasets:** BVCC + TMHINT → `/kaggle/temp/datasets`.
5. **Manifests:** build → assert `source` column present → print columns.
6. **Dev set:** `data/prepare_dev.py` → dev_acr/dev_ccr manifests.
7. **Cache (final layer):** `data.cache_features … --cache_dir /kaggle/temp/encoder_cache
   --num_workers 4`; skip-guard if already populated.
8. **Train A1:** `src.train --config configs/exp_a1_kaggle.yaml`; watch `[smoke]` + epoch lines.
9. **Predict + submit:** `scripts.predict_dev … --zip /kaggle/working/submission_a1.zip`;
   print offline `dev_srcc` and the zip location (downloadable from the Output tab).

### Persistence model

- `/kaggle/working`: repo, checkpoints (`exp_a1/best.pt`, `last.pt`), `submission_a1.zip`
  — small, persists, well under 20 GB.
- `/kaggle/temp`: raw datasets + ~45 GB cache — disposable, single-session.
- No Drive, no cross-session resume.

## Error Handling

Already shipped in `src/train.py` (`8adc50e`) and reused here: non-empty-manifest asserts,
`[smoke]` pre-loop dev forward (NaN/constant detection in seconds), per-epoch NaN
diagnostic, always-`last.pt` + `best.pt` fallback so the pipeline always yields a
submittable checkpoint. The notebook's `sh()` is fail-fast with live (unbuffered) output.

## Testing

- **Unit (no GPU, local):** `_CacheClips.__getitem__` returns `input_features` shape
  `(80, 3000)` and the correct `path`; dedupe/skip-already-cached logic preserved.
- **Regression:** existing 43 tests stay green.
- **Manual (Kaggle):** cell-1 probe shows `/kaggle/temp` ≥ ~50 GB free and a live GPU;
  read the `[smoke]` line and `Epoch 001` to confirm non-NaN training; confirm
  `submission_a1.zip` is produced.

## Risks & Fallbacks

- **`/kaggle/temp` too small for ~45 GB cache** → revealed by cell-1 probe before the long
  step. Fallback: trim the training set (e.g. cap clips), or cache to `/kaggle/working`
  with a reduced manifest. Decide *after* seeing the real number.
- **DataLoader workers crash (shared-memory `/dev/shm` limits on Kaggle)** → fall back to
  `--num_workers 2` (still better than serial). Surface clearly, don't mask.
- **Dataset download size** competes with cache for `/kaggle/temp` — probe accounts for both.
