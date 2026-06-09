# Approach B v1 Implementation Plan

> Executed inline with TDD in-session (controller holds full context). Steps tracked via TaskCreate.

**Goal:** Train on NISQA+TMHINT (drop BVCC); one Kaggle session, shared cache, two submissions — B1 (raw MOS) and B2 (per-source z-norm, rank-preserving rescale).

**Architecture:** Additions only — NISQA downloader + parser, `--datasets`/`--normalize` flags in manifest build, `--score-mode rescale` in predict, two configs, one notebook. No model/loss/train changes.

---

### Task 1 — `download_nisqa()` in `data/download.py`
- Constant `NISQA_ZENODO_ZIP = "https://zenodo.org/record/4728081/files/NISQA_Corpus.zip"`.
- `download_nisqa(output_dir)`: `out=join(output_dir,"nisqa")`; if a `NISQA_corpus*.csv` already exists under it, skip; else `_download_file` the zip to `out/NISQA_Corpus.zip` then `_extract`.
- Add `"nisqa"` to the `--datasets` choices and wire `if "nisqa" in args.datasets: download_nisqa(args.output)`.
- No unit test (network/IO, matches existing download fns which are untested).

### Task 2 — `parse_nisqa()` + flags in `data/build_manifests.py`
- `parse_nisqa(nisqa_dir)`: glob `**/NISQA_corpus*.csv` under `nisqa_dir`; if none → `return []`. corpus_root = csv dir. Read CSV; assert `{"db","filepath_deg","mos"} <= columns`. Keep `db` starting `NISQA_TRAIN`/`NISQA_VAL`. Row: `path=abspath(join(corpus_root, filepath_deg))` (skip if missing), `acr=float(mos)`, `ccr=nan`, `language="en"`, `system=str(db)`, `source="nisqa"`.
- `normalize_per_source(train_rows, dev_rows)`: per source, mean/std of `acr` over TRAIN rows (`std or 1.0` guard); apply `(acr-mean)/std` to train and dev (dev source missing from train → leave). Mutates in place.
- `main()`: add `--datasets` (choices bvcc/tmhint/audiomos/nisqa, `nargs="+"`, default all four), `--normalize {none,per_source_z}` default none. Gate each parser on membership in `--datasets`. Add nisqa block (`split_rows` like others). If normalize==per_source_z → `normalize_per_source(all_train, all_dev)` before write.
- **Tests** (`tests/test_build_manifests.py`): parse_nisqa tags source + excludes `NISQA_TEST*`; normalize makes each source mean≈0/std≈1; `--normalize none` leaves acr unchanged.

### Task 3 — `--score-mode` in `scripts/predict_dev.py`
- Helper `rescale_to_range(values, lo, hi)`: `mn,mx=min,max`; if `mx==mn` return `[lo]*len`; else linear map each → `lo+(v-mn)/(mx-mn)*(hi-lo)`. Pure, unit-tested (`tests/test_predict_rescale.py`).
- `--score-mode {clamp,rescale}` default clamp. Refactor: gather raw ACR preds list, then clamp→`min(5,max(1,·))` or rescale→`rescale_to_range(·,1,5)`; same for CCR diffs with `[-3,3]`. Keep arcname `predictions.csv`.

### Task 4 — configs `exp_b1_kaggle.yaml` / `exp_b2_kaggle.yaml`
mel-on, `acr_rank_alpha 0`, `encoder_layer -1`, `epochs 10`, `early_stop_patience 3`,
`checkpoint_every_n_epochs 2`, `num_workers 4`, cache `/kaggle/temp/encoder_cache_b` (both).
B1: manifests `data/manifests`, ckpt `/kaggle/working/checkpoints/exp_b1`.
B2: manifests `data/manifests_znorm`, ckpt `/kaggle/working/checkpoints/exp_b2`.

### Task 5 — `notebooks/make_kaggle_b_notebook.py` → `kaggle_approach_b.ipynb`
Cells: probe → clone+pip+sh() → token → `download.py --datasets nisqa tmhint` → build raw
(`build_manifests --datasets nisqa tmhint`) + znorm (`--datasets nisqa tmhint --normalize
per_source_z --output_dir data/manifests_znorm`) + assert source/print composition → dev set
→ cache once (from data/manifests) → set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE → train B1 →
predict B1 (`--score-mode clamp`, zip `submission_b1.zip`) → train B2 → predict B2
(`--score-mode rescale`, zip `submission_b2.zip`) → summary. Validate generated nb content.

### Task 6 — merge + push
FF-merge to main, `pytest -q` green, push origin (Kaggle clones main).
