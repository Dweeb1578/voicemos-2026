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
