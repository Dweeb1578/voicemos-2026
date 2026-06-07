"""Run once to (re)generate colab_train.ipynb.

This is the source of truth for the Colab notebook. Edit cells here, then run:
    python notebooks/make_notebook.py

The generated notebook is a turnkey, run-all Approach A pipeline: it regenerates
manifests, caches frozen Whisper features, trains the three Approach A experiments
(A1 rank loss, A2 rank+layer12, A3 rank+no-mel), and produces one CodaBench
submission zip per experiment -- with nothing for the user to edit.
"""
import json
import os

# Each entry is (cell_type, source). cell_type is "markdown" or "code".
cells = [
    ("markdown", '''# VoiceMOS 2026 -- Track 1 Approach A (run-all)

**Run every cell top to bottom -- nothing to edit.** This trains and submits all three
Approach A experiments and downloads a CodaBench submission zip for each.

**Prerequisites**
- Runtime -> Change runtime type -> **GPU** (T4 is fine).
- Add a Colab **Secret** named `HF_TOKEN` (a HuggingFace *read* token), notebook access ON.
- Google Drive is mounted for checkpoint backup/resume.

**Produces:** `submission_a1.zip` (rank loss), `submission_a2.zip` (rank loss + layer 12),
`submission_a3.zip` (rank loss, mel branch removed). Upload each to CodaBench for its dev UTT-SRCC.

**Heads-up:** the full run is long (feature caching ~1-2h each + three short cached trainings).
Checkpoints back up to Drive and caches are reused, so **if the session disconnects, just
re-run all cells -- it resumes automatically.**'''),

    ("code", '''# Setup: mount Drive, clone/refresh repo, define a fail-fast shell helper.
from google.colab import drive
drive.mount('/content/drive')

import os, sys, subprocess

REPO_URL = 'https://github.com/Dweeb1578/voicemos-2026.git'
REPO_DIR = '/content/voicemos-2026'
CKPT_DIR = '/content/drive/MyDrive/voicemos2026/checkpoints'

if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)
os.chdir(REPO_DIR)
os.makedirs(CKPT_DIR, exist_ok=True)

def sh(*cmd):
    """Run a command, streaming output; raise on failure (fail-fast)."""
    cmd = [sys.executable if c == 'python' else c for c in cmd]
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

print('Repo ready at', REPO_DIR)'''),

    ("code", '''# Install dependencies.
sh('python', '-m', 'pip', 'install', '-r', 'requirements.txt', '-q')'''),

    ("code", '''# Faster, authenticated HuggingFace downloads (token from Colab Secrets; never hardcoded).
import os
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print('HF_TOKEN loaded from Colab Secrets.')
except Exception as e:
    print('No HF_TOKEN secret found -- downloads will be slower/unauthenticated.', e)

sh('python', '-m', 'pip', 'install', '-q', 'hf_transfer')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # inherited by the subprocess calls below'''),

    ("code", '''# Download training datasets (BVCC + TMHINT-QI; AudioMOS-T3 skipped gracefully if unavailable).
sh('python', 'data/download.py', '--output', 'data/datasets',
   '--datasets', 'bvcc', 'tmhint', 'audiomos25t3')'''),

    ("code", '''# Build unified training manifests, then ASSERT the `source` column is present --
# within-source rank masking is a silent no-op without it.
import pandas as pd
sh('python', '-m', 'data.build_manifests', '--data_dir', 'data/datasets', '--output_dir', 'data/manifests')
cols = pd.read_csv('data/manifests/pretrain_train.csv', nrows=1).columns.tolist()
assert 'source' in cols, f'source column missing -- regenerate manifests. cols={cols}'
print('Manifests built. Columns:', cols)'''),

    ("code", '''# Materialize the official dev set (FLAC) + build dev manifests
# -> data/manifests/dev_acr.csv (1008), data/manifests/dev_ccr.csv (2520)
sh('python', 'data/prepare_dev.py', '--output', 'data/datasets/track1_dev')'''),

    ("markdown", '''## Experiments A1 & A3 -- final-layer features (shared cache)
Both run on the final Whisper layer, so they share one feature cache.
A1 = rank-aware ACR loss. A3 = same, with the mel-CNN branch removed.'''),

    ("code", '''# One-time frozen-encoder feature cache (final layer). ~1-2h on T4.
# Skipped if it already exists, or if A1 & A3 are already trained (so re-runs after a
# disconnect don't needlessly re-cache after the cache was freed for A2 below).
import os
done_a1a3 = os.path.exists('checkpoints/exp_a1/best.pt') and os.path.exists('checkpoints/exp_a3_nomel/best.pt')
have_cache = os.path.exists('data/encoder_cache') and len(os.listdir('data/encoder_cache')) >= 100
if done_a1a3:
    print('A1 & A3 already trained -- final-layer cache not needed, skipping.')
elif have_cache:
    print('Final-layer cache exists (%d files), skipping.' % len(os.listdir('data/encoder_cache')))
else:
    sh('python', '-m', 'data.cache_features',
       '--manifests', 'data/manifests/pretrain_train.csv', 'data/manifests/pretrain_dev.csv',
       '--cache_dir', 'data/encoder_cache', '--whisper_model', 'openai/whisper-medium',
       '--batch_size', '32')'''),

    ("code", '''# Experiment A1: rank-aware ACR loss. Training resumes from Drive if interrupted.
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_a1_rankloss.yaml')
if not os.path.exists('submission_a1.zip'):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', 'checkpoints/exp_a1/best.pt', '--config', 'configs/exp_a1_rankloss.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', 'predictions_a1.csv', '--zip', 'submission_a1.zip')
print('A1 offline pretrain_dev SRCC:', torch.load('checkpoints/exp_a1/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("code", '''# Experiment A3: rank loss + mel branch removed (reuses the final-layer cache).
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_a3_nomel.yaml')
if not os.path.exists('submission_a3.zip'):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', 'checkpoints/exp_a3_nomel/best.pt', '--config', 'configs/exp_a3_nomel.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', 'predictions_a3.csv', '--zip', 'submission_a3.zip')
print('A3 offline pretrain_dev SRCC:', torch.load('checkpoints/exp_a3_nomel/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("markdown", '''## Experiment A2 -- layer-12 features
Uses an intermediate Whisper layer (often better for quality than the final ASR layer).
This needs a *different* feature cache; to stay within free-T4 disk, the final-layer cache
is freed first (A1 & A3 are already done by now).'''),

    ("code", '''# Free the final-layer cache (A1 & A3 are done) and build the layer-12 cache.
import os, shutil
if os.path.exists('checkpoints/exp_a1/best.pt') and os.path.exists('checkpoints/exp_a3_nomel/best.pt'):
    shutil.rmtree('data/encoder_cache', ignore_errors=True)
    print('Freed final-layer cache.')
else:
    print('WARNING: A1/A3 not complete; skipping cache free.')

have_l12 = os.path.exists('data/encoder_cache_layer12') and len(os.listdir('data/encoder_cache_layer12')) >= 100
if have_l12:
    print('Layer-12 cache exists (%d files), skipping.' % len(os.listdir('data/encoder_cache_layer12')))
else:
    sh('python', '-m', 'data.cache_features',
       '--manifests', 'data/manifests/pretrain_train.csv', 'data/manifests/pretrain_dev.csv',
       '--cache_dir', 'data/encoder_cache_layer12', '--whisper_model', 'openai/whisper-medium',
       '--layer', '12', '--batch_size', '8')'''),

    ("code", '''# Experiment A2: rank loss + encoder layer 12.
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_a2_layer.yaml')
if not os.path.exists('submission_a2.zip'):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', 'checkpoints/exp_a2_l12/best.pt', '--config', 'configs/exp_a2_layer.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', 'predictions_a2.csv', '--zip', 'submission_a2.zip')
print('A2 offline pretrain_dev SRCC:', torch.load('checkpoints/exp_a2_l12/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("markdown", '''## Results -- offline guardrail + downloads'''),

    ("code", '''# Offline pretrain_dev SRCC is the GUARDRAIL (relative comparison across configs on held-out
# training-distribution data). It is NOT the CodaBench number -- upload each zip for the dev
# UTT-SRCC comparable to the 0.574 baseline / 0.662 leaderboard top.
import os, torch, shutil
CKPT_DIR = '/content/drive/MyDrive/voicemos2026/checkpoints'
runs = [
    ('A1 rank-loss',    'checkpoints/exp_a1/best.pt',       'submission_a1.zip'),
    ('A2 rank+layer12', 'checkpoints/exp_a2_l12/best.pt',   'submission_a2.zip'),
    ('A3 rank+no-mel',  'checkpoints/exp_a3_nomel/best.pt', 'submission_a3.zip'),
]
print('Offline pretrain_dev SRCC (relative guardrail -- NOT the 0.574 CodaBench number):')
for name, ckpt, zipf in runs:
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location='cpu', weights_only=True)['dev_srcc']
        tag = os.path.basename(os.path.dirname(ckpt))
        shutil.copy(ckpt, f'{CKPT_DIR}/{tag}_best.pt')
        print(f'  {name:16s} {s:.4f}   -> {zipf}')
print('\\nUpload each submission_*.zip to CodaBench for the real dev UTT-SRCC (vs 0.574 / 0.662).')

from google.colab import files
for _, _, zipf in runs:
    if os.path.exists(zipf):
        files.download(zipf)'''),
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_train.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Written: {out}  ({len(nb_cells)} cells)")
