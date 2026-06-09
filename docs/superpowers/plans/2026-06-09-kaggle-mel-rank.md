# Kaggle Mel-on Rank Experiments (M1/M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two Kaggle experiment configs (mel-on + rank, and mel-on + no-rank control) plus a turnkey notebook that trains and submits both on a shared encoder cache.

**Architecture:** Pure configuration + notebook generation. No `src/` changes — mel-on training with cached encoder features and rank loss already work. Both configs reuse the same final-layer encoder cache; the notebook sets HF offline mode before train/predict and relies on the already-shipped parallel predict + `predictions.csv` arcname fixes.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers (Whisper), Kaggle Notebooks (`kaggle_secrets`), YAML, pytest.

---

### Task 1: Mel-on experiment configs (M1 rank, M2 control)

**Files:**
- Create: `configs/exp_mel_rank_kaggle.yaml`
- Create: `configs/exp_mel_norank_kaggle.yaml`

- [ ] **Step 1: Create the rank config**

Create `configs/exp_mel_rank_kaggle.yaml`:

```yaml
# Approach A / mel-on + rank on KAGGLE: restore the mel-CNN branch + rank-aware ACR loss.
# Tests whether rank helps when the mel branch is present (A1 mel-off regressed to 0.534).
# Reuses the shared final-layer encoder cache; mel branch decodes waveforms per epoch
# (parallelized via num_workers). Cache -> /kaggle/temp, checkpoints -> /kaggle/working, no Drive.
model:
  whisper_model: "openai/whisper-medium"
  proj_dim: 256
  dropout: 0.1
  encoder_layer: -1
  use_mel_branch: true

training:
  train_manifest: "data/manifests/pretrain_train.csv"
  dev_manifest: "data/manifests/pretrain_dev.csv"
  batch_size: 16
  epochs: 8
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_gamma: 0.9999
  ccr_lambda: 0.0
  ccr_lambda_ramp_epochs: 0
  acr_rank_alpha: 1.0
  cache_dir: "/kaggle/temp/encoder_cache"
  checkpoint_dir: "/kaggle/working/checkpoints/exp_mel_rank"
  checkpoint_every_n_epochs: 2
  num_workers: 4

logging:
  run_name: "exp-mel-rank-kaggle"
```

- [ ] **Step 2: Create the no-rank control config**

Create `configs/exp_mel_norank_kaggle.yaml`:

```yaml
# Approach A / mel-on + NO rank on KAGGLE: clean control for the rank effect.
# Identical to exp_mel_rank_kaggle.yaml except acr_rank_alpha=0.0. M1 vs M2 isolates rank.
model:
  whisper_model: "openai/whisper-medium"
  proj_dim: 256
  dropout: 0.1
  encoder_layer: -1
  use_mel_branch: true

training:
  train_manifest: "data/manifests/pretrain_train.csv"
  dev_manifest: "data/manifests/pretrain_dev.csv"
  batch_size: 16
  epochs: 8
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_gamma: 0.9999
  ccr_lambda: 0.0
  ccr_lambda_ramp_epochs: 0
  acr_rank_alpha: 0.0
  cache_dir: "/kaggle/temp/encoder_cache"
  checkpoint_dir: "/kaggle/working/checkpoints/exp_mel_norank"
  checkpoint_every_n_epochs: 2
  num_workers: 4

logging:
  run_name: "exp-mel-norank-kaggle"
```

- [ ] **Step 3: Verify both configs load and have the right shape**

Run:
```bash
python -c "import yaml
for f, alpha, ck in [('configs/exp_mel_rank_kaggle.yaml',1.0,'exp_mel_rank'),('configs/exp_mel_norank_kaggle.yaml',0.0,'exp_mel_norank')]:
    c=yaml.safe_load(open(f)); t=c['training']
    assert c['model']['use_mel_branch'] is True, f'{f}: mel must be on'
    assert c['model']['encoder_layer']==-1, f'{f}: final layer'
    assert t['acr_rank_alpha']==alpha, f'{f}: alpha {t[\"acr_rank_alpha\"]} != {alpha}'
    assert 'drive_checkpoint_dir' not in t, f'{f}: no Drive key on Kaggle'
    assert t['cache_dir']=='/kaggle/temp/encoder_cache', f'{f}: shared cache'
    assert t['checkpoint_dir']==f'/kaggle/working/checkpoints/{ck}', f'{f}: ckpt dir'
    assert t['epochs']==8 and t['num_workers']==4 and t['checkpoint_every_n_epochs']==2, f'{f}: training knobs'
print('both configs ok')"
```
Expected: prints `both configs ok`.

- [ ] **Step 4: Commit**

```bash
git add configs/exp_mel_rank_kaggle.yaml configs/exp_mel_norank_kaggle.yaml
git commit -m "feat: Kaggle mel-on configs (rank M1 + no-rank control M2)"
```

---

### Task 2: Kaggle mel notebook generator

**Files:**
- Create: `notebooks/make_kaggle_mel_notebook.py`
- Generate: `notebooks/kaggle_mel.ipynb` (output of running the generator)

- [ ] **Step 1: Create the generator**

Create `notebooks/make_kaggle_mel_notebook.py`:

```python
"""Run once to (re)generate kaggle_mel.ipynb.

Source of truth for the Kaggle mel-on notebook. Edit cells here, then run:
    python notebooks/make_kaggle_mel_notebook.py

Turnkey run-all: trains mel-on+rank (M1) and mel-on+no-rank control (M2) on a shared
final-layer encoder cache, and produces one CodaBench submission per run. Bakes in today's
fixes: HF offline before train/predict (no network hang), parallel predict, predictions.csv
arcname (scores instead of NA).
"""
import json
import os

cells = [
    ("markdown", '''# VoiceMOS 2026 -- Track 1 mel-on experiments on Kaggle (run-all)

**Run every cell top to bottom -- nothing to edit.** Trains two models on one shared cache:
- **M1 = mel-on + rank** -> `submission_mel_rank.zip`
- **M2 = mel-on + no-rank** (control) -> `submission_mel_norank.zip`

**Prerequisites**
- Settings -> Accelerator -> **GPU T4 x2** (P100 is NOT supported by Kaggle's PyTorch -- sm_60).
- Settings -> **Internet ON** (phone-verified account).
- Add-ons -> Secrets -> **`HF_TOKEN`** (HuggingFace read token), attached.

**Produces:** two zips in `/kaggle/working` -- download from the **Output** tab and upload each
to CodaBench. Compare M1 vs M2 (rank effect) and both vs the 0.574 baseline.'''),

    ("code", '''# Probe the REAL environment first -- no assumptions about disk or GPU.
import subprocess, torch
print(subprocess.run(['df', '-h'], capture_output=True, text=True).stdout)
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')'''),

    ("code", '''# Clone/refresh repo into persistent /kaggle/working; constants + fail-fast shell helper.
import os, sys, subprocess

REPO_URL = 'https://github.com/Dweeb1578/voicemos-2026.git'
REPO_DIR = '/kaggle/working/voicemos-2026'
DATA_DIR = '/kaggle/temp/datasets'        # ephemeral raw audio
CACHE_DIR = '/kaggle/temp/encoder_cache'  # ephemeral ~45 GB cache, shared by M1 & M2

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

    ("code", '''# Build manifests, then ASSERT the `source` column exists (rank masking needs it).
import pandas as pd
sh('python', '-m', 'data.build_manifests', '--data_dir', DATA_DIR, '--output_dir', 'data/manifests')
cols = pd.read_csv('data/manifests/pretrain_train.csv', nrows=1).columns.tolist()
assert 'source' in cols, f'source column missing -- regenerate manifests. cols={cols}'
print('Manifests built. Columns:', cols)'''),

    ("code", '''# Materialize the official dev set (FLAC) + dev manifests.
sh('python', 'data/prepare_dev.py', '--output', DATA_DIR + '/track1_dev')'''),

    ("code", '''# Shared one-time encoder feature cache (final layer), parallel across CPU workers.
# Both M1 and M2 reuse this (same Whisper branch). Skip-guarded.
import os
have_cache = os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) >= 100
if have_cache:
    print('Cache exists (%d files), skipping.' % len(os.listdir(CACHE_DIR)))
else:
    sh('python', '-m', 'data.cache_features',
       '--manifests', 'data/manifests/pretrain_train.csv', 'data/manifests/pretrain_dev.csv',
       '--cache_dir', CACHE_DIR, '--whisper_model', 'openai/whisper-medium',
       '--batch_size', '32', '--num_workers', '4')'''),

    ("code", '''# Model is cached now -> force HF OFFLINE for all train/predict subprocesses below.
# This prevents from_pretrained from network-hanging on a Hub revision check.
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
print('HF offline mode ON for train/predict.')'''),

    ("code", '''# M1: mel-on + rank. Watch for the [smoke] line and per-epoch srcc (mel decodes per epoch).
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_mel_rank_kaggle.yaml')
SUB = '/kaggle/working/submission_mel_rank.zip'
if not os.path.exists(SUB):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', '/kaggle/working/checkpoints/exp_mel_rank/best.pt',
       '--config', 'configs/exp_mel_rank_kaggle.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', '/kaggle/working/predictions_mel_rank.csv', '--zip', SUB)
print('M1 mel-rank pretrain_dev SRCC:',
      torch.load('/kaggle/working/checkpoints/exp_mel_rank/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("code", '''# M2: mel-on + no-rank control (reuses the same cache).
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_mel_norank_kaggle.yaml')
SUB = '/kaggle/working/submission_mel_norank.zip'
if not os.path.exists(SUB):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', '/kaggle/working/checkpoints/exp_mel_norank/best.pt',
       '--config', 'configs/exp_mel_norank_kaggle.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', '/kaggle/working/predictions_mel_norank.csv', '--zip', SUB)
print('M2 mel-norank pretrain_dev SRCC:',
      torch.load('/kaggle/working/checkpoints/exp_mel_norank/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("code", '''# Summary: offline pretrain_dev guardrail (NOT the CodaBench number) + submission paths.
import os, torch
for name, ckpt, zipf in [
    ('M1 mel+rank',   'checkpoints/exp_mel_rank/best.pt',   '/kaggle/working/submission_mel_rank.zip'),
    ('M2 mel+norank', 'checkpoints/exp_mel_norank/best.pt', '/kaggle/working/submission_mel_norank.zip'),
]:
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location='cpu', weights_only=True)['dev_srcc']
        print(f'  {name:14s} pretrain_dev SRCC {s:.4f}  -> {zipf} ({"ready" if os.path.exists(zipf) else "MISSING"})')
print('\\nDownload both zips from the Output tab; upload each to CodaBench. Compare M1 vs M2 (rank effect) and vs 0.574.')'''),
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_mel.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Written: {out}  ({len(nb_cells)} cells)")
```

- [ ] **Step 2: Generate the notebook**

Run: `python notebooks/make_kaggle_mel_notebook.py`
Expected: prints `Written: .../notebooks/kaggle_mel.ipynb  (12 cells)`.

- [ ] **Step 3: Validate the generated notebook**

Run:
```bash
python -c "import json; nb=json.load(open('notebooks/kaggle_mel.ipynb')); src='\n'.join(''.join(c['source']) for c in nb['cells']); assert 'kaggle_secrets' in src; assert 'df' in src and 'nvidia-smi' in src; assert 'google.colab' not in src; assert 'HF_HUB_OFFLINE' in src; assert 'configs/exp_mel_rank_kaggle.yaml' in src and 'configs/exp_mel_norank_kaggle.yaml' in src; assert 'submission_mel_rank.zip' in src and 'submission_mel_norank.zip' in src; print('notebook ok', len(nb['cells']), 'cells')"
```
Expected: prints `notebook ok 12 cells`.

- [ ] **Step 4: Confirm the existing suite still passes (no src changes, sanity only)**

Run: `python -m pytest -q`
Expected: `46 passed` (this task adds no tests and changes no `src/`).

- [ ] **Step 5: Commit**

```bash
git add notebooks/make_kaggle_mel_notebook.py notebooks/kaggle_mel.ipynb
git commit -m "feat: Kaggle mel-on notebook (M1 rank + M2 control, offline train/predict)"
```

---

### Task 3: Push to origin (gated on user confirmation)

**Files:** none (git only)

- [ ] **Step 1: Confirm with the user before pushing**

Pushing is outward-facing (shared repo with Shubham). Ask the user to confirm. Do **not** push without an explicit yes.

- [ ] **Step 2: Push**

```bash
git push origin main
```
Expected: `main -> main` updated. Kaggle's `git clone`/`git pull` then picks up the configs + notebook.

- [ ] **Step 3: Hand off to the user**

Tell the user: open `notebooks/kaggle_mel.ipynb` on Kaggle (File -> Import Notebook -> GitHub, URL `https://github.com/Dweeb1578/voicemos-2026/blob/main/notebooks/kaggle_mel.ipynb`), set **GPU T4 x2** + Internet ON + `HF_TOKEN` secret, then Run All. Report back the cell-1 probe, the two `[smoke]` lines, and the two `pretrain_dev` SRCCs; then upload both zips to CodaBench.

---

## Self-Review

**Spec coverage:**
- M1 mel-on+rank config → Task 1 Step 1. ✓
- M2 mel-on+no-rank control config → Task 1 Step 2. ✓
- Shared `/kaggle/temp/encoder_cache`, checkpoints in `/kaggle/working`, no Drive, epochs 8, `checkpoint_every_n 2`, `num_workers 4` → Task 1 (both configs) + Step 3 asserts. ✓
- Notebook generator + `kaggle_mel.ipynb` (probe, secrets, download, manifests+assert, dev, shared cache skip-guard, train+predict M1, train+predict M2, summary) → Task 2. ✓
- Offline mode set after cache, before train/predict → Task 2 cell 9 ("HF offline"). ✓
- Parallel predict + `predictions.csv` arcname → already shipped (`0ce0321`, `0cd8db4`); used as-is, no task. ✓
- A1's overfit lesson → `epochs: 8`; pretrain_dev-may-not-transfer hedge → `checkpoint_every_n_epochs: 2`. ✓
- GPU must be T4 (P100 unsupported) → notebook markdown prerequisite. ✓
- Testing: config-shape check (Task 1 Step 3), notebook-content asserts (Task 2 Step 3), full suite (Task 2 Step 4). ✓

**Placeholder scan:** none — every step has full file content or an exact command + expected output.

**Type/name consistency:** config filenames (`exp_mel_rank_kaggle.yaml`, `exp_mel_norank_kaggle.yaml`), checkpoint dirs (`/kaggle/working/checkpoints/exp_mel_rank`, `.../exp_mel_norank`), submission zips (`submission_mel_rank.zip`, `submission_mel_norank.zip`), and `CACHE_DIR=/kaggle/temp/encoder_cache` are identical across the configs, the notebook train/predict cells, and the validation asserts. `best.pt` paths match each config's `checkpoint_dir`.

**Note (no src changes):** mel-on + cached encoder feats + rank loss are already supported by `src/model.py` / `src/train.py` / `src/losses.py`; this plan is configs + notebook only, so no new unit tests are warranted beyond the shape/content checks.
