# Kaggle A1 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Approach A experiment A1 end-to-end on Kaggle (single P100, one session) producing `submission_a1.zip`, with caching parallelized so it stops starving the GPU.

**Architecture:** Three changes — (1) rewrite `data/cache_features.py` to decode audio + extract log-mel in `DataLoader` worker processes so CPU work overlaps the GPU encoder; (2) add `configs/exp_a1_kaggle.yaml` (cache→`/kaggle/temp`, checkpoints→`/kaggle/working`, no Drive); (3) add `notebooks/make_kaggle_notebook.py` that generates a turnkey `notebooks/kaggle_train.ipynb` whose first cell probes real disk/GPU.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers (Whisper), Kaggle Notebooks (`kaggle_secrets`), pytest.

---

### Task 1: Parallelize `data/cache_features.py`

**Files:**
- Modify: `data/cache_features.py` (full rewrite of `main()`, add `_CacheClips`)
- Test: `tests/test_cache_features.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_features.py`:

```python
import numpy as np
import soundfile as sf
import torch

from data.cache_features import _CacheClips, cache_key


def test_cache_clips_returns_features_and_path(tmp_path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.random.randn(32000).astype("float32"), 16000)
    ds = _CacheClips([str(wav)], whisper_model="openai/whisper-tiny")
    item = ds[0]
    assert item["input_features"].shape == (80, 3000)
    assert item["input_features"].dtype == torch.float32
    assert item["path"] == str(wav)


def test_cache_key_deterministic_and_npy():
    assert cache_key("/a/b.wav") == cache_key("/a/b.wav")
    assert cache_key("/a/b.wav").endswith(".npy")
    assert cache_key("/a/b.wav") != cache_key("/a/c.wav")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache_features.py -v`
Expected: FAIL with `ImportError: cannot import name '_CacheClips'`.

- [ ] **Step 3: Rewrite `data/cache_features.py`**

Replace the entire file with:

```python
"""Pre-extract frozen Whisper encoder outputs to disk (one-time cost).

Saves (1500, hidden_size) float16 arrays keyed by MD5 of audio path.
~3 MB/sample for whisper-medium -> ~47 GB for 15.5K samples.

Audio decode + log-mel extraction (the CPU cost) run in DataLoader worker
processes so they overlap the GPU encoder forward; the loop is otherwise
CPU-bound and starves the GPU.

Usage:
    python -m data.cache_features \
        --manifests data/manifests/pretrain_train.csv data/manifests/pretrain_dev.csv \
        --cache_dir data/encoder_cache \
        --whisper_model openai/whisper-medium --num_workers 4

    # Intermediate layer (e.g. layer 12 of 24 for whisper-medium)
    python -m data.cache_features --manifests ... --cache_dir data/encoder_cache_layer12 \
        --whisper_model openai/whisper-medium --layer 12 --num_workers 4
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

from data.preprocess import resample_and_normalize, trim_and_pad


def cache_key(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode()).hexdigest() + ".npy"


class _CacheClips(Dataset):
    """Label-free dataset: decode audio + log-mel in worker procs, yield (feats, path)."""

    def __init__(self, paths, whisper_model):
        self.paths = list(paths)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        audio, _ = resample_and_normalize(p)
        audio = trim_and_pad(audio)
        feats = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        return {"input_features": feats.input_features.squeeze(0), "path": p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--whisper_model", default="openai/whisper-medium")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Encoder layer to extract (-1 = final layer, 0 = embedding output, "
                             "1..N = after transformer layer N). Whisper-medium has 24 layers.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  model: {args.whisper_model}")

    whisper = WhisperModel.from_pretrained(args.whisper_model)
    encoder = whisper.encoder.to(device)
    encoder.train(False)

    # Deduplicate paths across manifests
    all_paths = []
    seen = set()
    for m in args.manifests:
        for path in pd.read_csv(m)["path"]:
            if path not in seen:
                seen.add(path)
                all_paths.append(path)

    # Filter already-cached (resume-by-skip)
    todo = [p for p in all_paths if not os.path.exists(os.path.join(args.cache_dir, cache_key(p)))]
    print(f"Total unique paths: {len(all_paths)}  to cache: {len(todo)}")
    if not todo:
        print("Nothing to cache.")
        return

    use_intermediate = args.layer != -1
    print(f"Extracting {'intermediate layer ' + str(args.layer) if use_intermediate else 'final layer'}")

    ds = _CacheClips(todo, args.whisper_model)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(loader, desc="Extracting"):
            inp = batch["input_features"].to(device)
            if use_intermediate:
                out = encoder(inp, output_hidden_states=True).hidden_states[args.layer]
            else:
                out = encoder(inp).last_hidden_state
            out_np = out.half().cpu().numpy()
            for path, feat in zip(batch["path"], out_np):
                np.save(os.path.join(args.cache_dir, cache_key(path)), feat)

    print(f"Done. Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_cache_features.py -v`
Expected: PASS (2 tests). Downloads the whisper-tiny feature extractor once, like the existing dataset tests.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `python -m pytest -q`
Expected: all previously-green tests still pass (43 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add data/cache_features.py tests/test_cache_features.py
git commit -m "perf: parallelize feature caching with DataLoader workers"
```

---

### Task 2: Kaggle A1 config

**Files:**
- Create: `configs/exp_a1_kaggle.yaml`

- [ ] **Step 1: Create the config**

Create `configs/exp_a1_kaggle.yaml` (clone of `exp_a1_rankloss.yaml` with Kaggle paths and **no** `drive_checkpoint_dir`):

```yaml
# Approach A / Experiment A1 on KAGGLE: rank-aware ACR loss, mel branch OFF, final layer.
# Cache -> ephemeral /kaggle/temp (~45 GB, too big for the 20 GB /kaggle/working).
# Checkpoints -> persistent /kaggle/working. No Google Drive (Kaggle has none).
model:
  whisper_model: "openai/whisper-medium"
  proj_dim: 256
  dropout: 0.1
  encoder_layer: -1
  use_mel_branch: false

training:
  train_manifest: "data/manifests/pretrain_train.csv"
  dev_manifest: "data/manifests/pretrain_dev.csv"
  batch_size: 16
  epochs: 15
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_gamma: 0.9999
  ccr_lambda: 0.0
  ccr_lambda_ramp_epochs: 0
  acr_rank_alpha: 1.0
  cache_dir: "/kaggle/temp/encoder_cache"
  checkpoint_dir: "/kaggle/working/checkpoints/exp_a1"
  checkpoint_every_n_epochs: 5
  num_workers: 4

logging:
  run_name: "exp-a1-rankloss-nomel-kaggle"
```

- [ ] **Step 2: Verify it loads and has the right shape**

Run:
```bash
python -c "import yaml; c=yaml.safe_load(open('configs/exp_a1_kaggle.yaml')); t=c['training']; assert 'drive_checkpoint_dir' not in t, 'drive key must be absent on Kaggle'; assert t['cache_dir']=='/kaggle/temp/encoder_cache'; assert t['checkpoint_dir']=='/kaggle/working/checkpoints/exp_a1'; assert c['model']['use_mel_branch'] is False; assert t['acr_rank_alpha']==1.0; print('config ok')"
```
Expected: prints `config ok`.

- [ ] **Step 3: Commit**

```bash
git add configs/exp_a1_kaggle.yaml
git commit -m "feat: Kaggle A1 config (cache in /kaggle/temp, no Drive)"
```

---

### Task 3: Kaggle notebook generator

**Files:**
- Create: `notebooks/make_kaggle_notebook.py`
- Generate: `notebooks/kaggle_train.ipynb` (output of running the generator)

- [ ] **Step 1: Create the generator**

Create `notebooks/make_kaggle_notebook.py`:

```python
"""Run once to (re)generate kaggle_train.ipynb.

Source of truth for the Kaggle notebook. Edit cells here, then run:
    python notebooks/make_kaggle_notebook.py

Turnkey, run-all Approach A *A1* pipeline for Kaggle (single P100, one session):
probe disk/GPU -> clone repo -> install -> HF token from Kaggle Secrets ->
download data -> manifests -> dev set -> cache features (parallel) -> train A1
-> produce /kaggle/working/submission_a1.zip. Nothing to edit.
"""
import json
import os

cells = [
    ("markdown", '''# VoiceMOS 2026 -- Track 1 A1 on Kaggle (run-all)

**Run every cell top to bottom -- nothing to edit.**

**Prerequisites**
- Settings -> Accelerator -> **GPU P100** (T4 x2 also fine; we use one card).
- Settings -> **Internet ON** (needs a phone-verified Kaggle account).
- Add-ons -> Secrets -> add **`HF_TOKEN`** (a HuggingFace *read* token), attached to this notebook.

**Produces:** `/kaggle/working/submission_a1.zip` -- download it from the **Output** tab (right panel) and upload to CodaBench for the dev UTT-SRCC (vs the 0.574 baseline / 0.662 top).

**Heads-up:** the feature cache (~45 GB) goes in ephemeral `/kaggle/temp`; checkpoints + the zip go in persistent `/kaggle/working`. If the session ends, re-run all cells.'''),

    ("code", '''# Probe the REAL environment first -- no assumptions about disk or GPU.
import subprocess, torch
print(subprocess.run(['df', '-h'], capture_output=True, text=True).stdout)
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')'''),

    ("code", '''# Clone/refresh repo into persistent /kaggle/working; define constants + fail-fast shell helper.
import os, sys, subprocess

REPO_URL = 'https://github.com/Dweeb1578/voicemos-2026.git'
REPO_DIR = '/kaggle/working/voicemos-2026'
DATA_DIR = '/kaggle/temp/datasets'        # ephemeral raw audio
CACHE_DIR = '/kaggle/temp/encoder_cache'  # ephemeral ~45 GB cache (matches exp_a1_kaggle.yaml)

if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)
os.chdir(REPO_DIR)

def sh(*cmd):
    """Run a command, streaming output; raise on failure (fail-fast)."""
    cmd = [sys.executable if c == 'python' else c for c in cmd]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

sh('python', '-m', 'pip', 'install', '-r', 'requirements.txt', '-q')
print('Repo ready at', REPO_DIR)'''),

    ("code", '''# HuggingFace token from Kaggle Secrets (never hardcode -- repo is public).
import os
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
print('HF_TOKEN loaded from Kaggle Secrets.')'''),

    ("code", '''# Download training datasets (BVCC + TMHINT-QI) into ephemeral /kaggle/temp.
sh('python', 'data/download.py', '--output', DATA_DIR, '--datasets', 'bvcc', 'tmhint')'''),

    ("code", '''# Build unified training manifests, then ASSERT the `source` column exists
# (within-source rank masking is a silent no-op without it).
import pandas as pd
sh('python', '-m', 'data.build_manifests', '--data_dir', DATA_DIR, '--output_dir', 'data/manifests')
cols = pd.read_csv('data/manifests/pretrain_train.csv', nrows=1).columns.tolist()
assert 'source' in cols, f'source column missing -- regenerate manifests. cols={cols}'
print('Manifests built. Columns:', cols)'''),

    ("code", '''# Materialize the official dev set (FLAC) + dev manifests
# -> data/manifests/dev_acr.csv (1008), data/manifests/dev_ccr.csv (2520)
sh('python', 'data/prepare_dev.py', '--output', DATA_DIR + '/track1_dev')'''),

    ("code", '''# One-time frozen-encoder feature cache (final layer), parallelized across CPU workers.
# Skipped if already populated (resume-by-skip). Watch df -h above: this writes ~45 GB to /kaggle/temp.
import os
have_cache = os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) >= 100
if have_cache:
    print('Cache exists (%d files), skipping.' % len(os.listdir(CACHE_DIR)))
else:
    sh('python', '-m', 'data.cache_features',
       '--manifests', 'data/manifests/pretrain_train.csv', 'data/manifests/pretrain_dev.csv',
       '--cache_dir', CACHE_DIR, '--whisper_model', 'openai/whisper-medium',
       '--batch_size', '32', '--num_workers', '4')'''),

    ("code", '''# Train A1 (rank-aware ACR loss). Watch for the [smoke] line and per-epoch srcc.
sh('python', '-m', 'src.train', '--config', 'configs/exp_a1_kaggle.yaml')'''),

    ("code", '''# Predict the dev set and build the CodaBench submission (persisted in /kaggle/working).
import os, torch
SUB = '/kaggle/working/submission_a1.zip'
if not os.path.exists(SUB):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', '/kaggle/working/checkpoints/exp_a1/best.pt', '--config', 'configs/exp_a1_kaggle.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', '/kaggle/working/predictions_a1.csv', '--zip', SUB)
srcc = torch.load('/kaggle/working/checkpoints/exp_a1/best.pt', map_location='cpu', weights_only=True)['dev_srcc']
print('A1 offline pretrain_dev SRCC (relative guardrail, NOT the CodaBench number):', srcc)
print('Submission ready at', SUB, '-- download from the Output tab (right panel), upload to CodaBench.')'''),
]

nb_cells = []
for i, (ctype, src) in enumerate(cells):
    cell = {"cell_type": ctype, "metadata": {}, "source": src, "id": str(i)}
    if ctype == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nb_cells.append(cell)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": nb_cells,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_train.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Written: {out}  ({len(nb_cells)} cells)")
```

- [ ] **Step 2: Generate the notebook**

Run: `python notebooks/make_kaggle_notebook.py`
Expected: prints `Written: .../notebooks/kaggle_train.ipynb  (10 cells)`.

- [ ] **Step 3: Validate the generated notebook**

Run:
```bash
python -c "import json; nb=json.load(open('notebooks/kaggle_train.ipynb')); src='\n'.join(''.join(c['source']) for c in nb['cells']); assert 'kaggle_secrets' in src, 'missing Kaggle secrets'; assert 'df' in src and 'nvidia-smi' in src, 'missing probe'; assert 'google.colab' not in src, 'Colab leftover'; assert 'configs/exp_a1_kaggle.yaml' in src; assert '/kaggle/working/submission_a1.zip' in src; print('notebook ok', len(nb['cells']), 'cells')"
```
Expected: prints `notebook ok 10 cells`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/make_kaggle_notebook.py notebooks/kaggle_train.ipynb
git commit -m "feat: turnkey Kaggle A1 notebook (probe disk/GPU, parallel cache)"
```

---

### Task 4: Push to origin (gated on user confirmation)

**Files:** none (git only)

- [ ] **Step 1: Confirm with the user before pushing**

Pushing is outward-facing (shared repo with Shubham). Ask the user to confirm. Do **not** push without an explicit yes.

- [ ] **Step 2: Push**

```bash
git push origin main
```
Expected: `main -> main` updated. Kaggle's first cell `git clone`/`git pull` then picks up all three changes.

- [ ] **Step 3: Hand off to the user**

Tell the user: open `notebooks/kaggle_train.ipynb` on Kaggle (File -> Import Notebook, or open from GitHub), confirm GPU P100 + Internet ON + `HF_TOKEN` secret, then Run all. Report back the cell-1 `df -h` (`/kaggle/temp` free space) and the `[smoke]` line.

---

## Self-Review

**Spec coverage:**
- Change 1 (parallel caching) → Task 1. ✓
- Change 2 (`exp_a1_kaggle.yaml`) → Task 2. ✓
- Change 3 (notebook generator + `kaggle_train.ipynb`) → Task 3. ✓
- "No assumptions" disk/GPU probe → Task 3 cell 1. ✓
- HF token via `UserSecretsClient` → Task 3 cell 3. ✓
- Cache→`/kaggle/temp`, checkpoints/zip→`/kaggle/working`, no Drive → Tasks 2 & 3. ✓
- Fail-loud training (smoke/NaN/last.pt) → already shipped in `src/train.py` (`8adc50e`), reused as-is; no task needed. ✓
- A1-only scope (A2/A3 deferred, dual-GPU rejected) → reflected; no tasks for them. ✓
- Testing: unit test for `_CacheClips` (Task 1), config-shape check (Task 2), notebook-content check (Task 3), full suite (Task 1 Step 5). ✓

**Placeholder scan:** none — every code/command step shows full content.

**Type/name consistency:** `_CacheClips(paths, whisper_model)`, `cache_key`, `--num_workers`, `CACHE_DIR=/kaggle/temp/encoder_cache` match between the rewritten script, the config, and the notebook cells. The notebook's `CACHE_DIR` equals the config's `cache_dir`. `best.pt` path `/kaggle/working/checkpoints/exp_a1/best.pt` matches the config's `checkpoint_dir`.

**Known out-of-scope cost (not a gap):** `scripts/predict_dev.py` still runs the dev set through the *live* encoder single-threaded (~6k clips). It's correct, just not parallelized; left as a possible follow-up to keep this plan focused on the caching bottleneck.
