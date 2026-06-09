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
