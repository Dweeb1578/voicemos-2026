"""Run once to (re)generate kaggle_approach_b.ipynb.

Source of truth for the Kaggle Approach B notebook. Edit cells here, then run:
    python notebooks/make_kaggle_b_notebook.py

Turnkey run-all: trains on NISQA + TMHINT (BVCC dropped) and produces two CodaBench
submissions from one shared encoder cache -- B1 (raw MOS) and B2 (per-source z-norm,
rank-preserving rescale). Bakes in the 2026-06-09 fixes: HF offline before train/predict,
parallel predict, predictions.csv arcname. Recommended run mode: Save & Run All (Commit).
"""
import json
import os

cells = [
    ("markdown", '''# VoiceMOS 2026 -- Track 1 Approach B on Kaggle (run-all)

**Run every cell top to bottom -- nothing to edit.** Trains on domain-closer data
(**NISQA + TMHINT**, BVCC dropped) and makes two submissions from one shared cache:
- **B1 = raw MOS** -> `submission_b1.zip`
- **B2 = per-source z-norm** (rank-preserving rescale) -> `submission_b2.zip`

**Prerequisites**
- Settings -> Accelerator -> **GPU T4 x2** (P100 is NOT supported by Kaggle's PyTorch -- sm_60).
- Settings -> **Internet ON** (phone-verified account).
- Add-ons -> Secrets -> **`HF_TOKEN`** (HuggingFace read token), attached.

**Best run mode:** Save Version -> **"Save & Run All (Commit)"** (headless, survives tab/socket
death). Download both zips from the version's **Output**, upload each to CodaBench, compare to 0.574.'''),

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
DATA_DIR = '/kaggle/temp/datasets'          # ephemeral raw audio
CACHE_DIR = '/kaggle/temp/encoder_cache_b'  # ephemeral cache, shared by B1 & B2

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

    ("markdown", '''### NISQA via Kaggle Dataset (no download)
**Add Input** (right panel) -> search **`nisqa-corpus`** -> add **`pratt3000/nisqa-corpus`**.
It mounts read-only at `/kaggle/input/...` -- instant, and costs ZERO writable disk. We do
NOT download NISQA: the 16 GB download fills Kaggle's disk and hangs the kernel.'''),

    ("code", '''# NISQA: link the attached read-only dataset into the path build_manifests expects.
# TMHINT (~2.2 GB) is small and downloads fine, so we still fetch that. BVCC is excluded.
import os, glob
os.makedirs(DATA_DIR, exist_ok=True)
hits = glob.glob('/kaggle/input/**/NISQA_corpus*.csv', recursive=True)
assert hits, ('NISQA dataset not attached. Right panel -> Add Input -> search '
              '"nisqa-corpus" -> add pratt3000/nisqa-corpus, then re-run this cell.')
nisqa_src = os.path.dirname(hits[0])          # dir holding NISQA_corpus_file.csv
nisqa_link = os.path.join(DATA_DIR, 'nisqa')
if not os.path.islink(nisqa_link) and not os.path.exists(nisqa_link):
    os.symlink(nisqa_src, nisqa_link)
print('NISQA linked:', nisqa_link, '->', nisqa_src)

sh('python', 'data/download.py', '--output', DATA_DIR, '--datasets', 'tmhint')'''),

    ("code", '''# Build manifests TWICE from the same NISQA+TMHINT data:
#   data/manifests        -> raw MOS        (B1)
#   data/manifests_znorm  -> per-source z-norm (B2)
# --max-per-source 7500 caps each source (seeded) so the shared cache stays ~15k clips
# (~45 GB, the size that fit on the mel runs) -- NISQA TRAIN+VAL + TMHINT uncapped would
# be ~25k clips (~75 GB) and blow Kaggle's ~57.6 GB disk cap. The fixed seed makes both
# builds pick the SAME clips, so the path sets are identical => the cache is shared.
import pandas as pd
CAP = '7500'
sh('python', '-m', 'data.build_manifests', '--data_dir', DATA_DIR,
   '--output_dir', 'data/manifests', '--datasets', 'nisqa', 'tmhint', '--max-per-source', CAP)
sh('python', '-m', 'data.build_manifests', '--data_dir', DATA_DIR,
   '--output_dir', 'data/manifests_znorm', '--datasets', 'nisqa', 'tmhint',
   '--max-per-source', CAP, '--normalize', 'per_source_z')
df = pd.read_csv('data/manifests/pretrain_train.csv')
assert 'source' in df.columns, f'source column missing. cols={list(df.columns)}'
print('Train composition:', df['source'].value_counts().to_dict())
assert set(df['source'].unique()) == {'nisqa', 'tmhint'}, \
    f"expected both nisqa+tmhint, got {set(df['source'].unique())} (NISQA link/paths wrong?)"
# The shared cache requires raw and znorm manifests to reference the SAME audio paths.
for split in ['pretrain_train.csv', 'pretrain_dev.csv']:
    p_raw = set(pd.read_csv(f'data/manifests/{split}')['path'])
    p_zn = set(pd.read_csv(f'data/manifests_znorm/{split}')['path'])
    assert p_raw == p_zn, f'{split}: raw vs znorm path mismatch -> cache would not cover B2'
print('Manifests built (raw + znorm); path sets identical -> cache is shared.')'''),

    ("code", '''# Materialize the official dev set (FLAC) + dev manifests.
sh('python', 'data/prepare_dev.py', '--output', DATA_DIR + '/track1_dev')'''),

    ("code", '''# Shared one-time encoder feature cache (final layer), parallel across CPU workers.
# Built from data/manifests (paths identical to data/manifests_znorm) -> serves B1 and B2.
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
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
print('HF offline mode ON for train/predict.')'''),

    ("code", '''# B1: NISQA+TMHINT raw MOS. Predict with clamp (raw-MOS model).
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_b1_kaggle.yaml')
SUB = '/kaggle/working/submission_b1.zip'
if not os.path.exists(SUB):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', '/kaggle/working/checkpoints/exp_b1/best.pt', '--config', 'configs/exp_b1_kaggle.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', '/kaggle/working/predictions_b1.csv', '--zip', SUB, '--score-mode', 'clamp')
print('B1 pretrain_dev SRCC:',
      torch.load('/kaggle/working/checkpoints/exp_b1/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("code", '''# B2: NISQA+TMHINT per-source z-norm. Predict with RESCALE (rank-preserving, z-norm model).
import os, torch
sh('python', '-m', 'src.train', '--config', 'configs/exp_b2_kaggle.yaml')
SUB = '/kaggle/working/submission_b2.zip'
if not os.path.exists(SUB):
    sh('python', '-m', 'scripts.predict_dev',
       '--checkpoint', '/kaggle/working/checkpoints/exp_b2/best.pt', '--config', 'configs/exp_b2_kaggle.yaml',
       '--acr-manifest', 'data/manifests/dev_acr.csv', '--ccr-manifest', 'data/manifests/dev_ccr.csv',
       '--output', '/kaggle/working/predictions_b2.csv', '--zip', SUB, '--score-mode', 'rescale')
print('B2 pretrain_dev SRCC:',
      torch.load('/kaggle/working/checkpoints/exp_b2/best.pt', map_location='cpu', weights_only=True)['dev_srcc'])'''),

    ("code", '''# Summary: offline pretrain_dev guardrail (NOT the CodaBench number) + submission paths.
import os, torch
for name, ckpt, zipf in [
    ('B1 raw',    'checkpoints/exp_b1/best.pt', '/kaggle/working/submission_b1.zip'),
    ('B2 z-norm', 'checkpoints/exp_b2/best.pt', '/kaggle/working/submission_b2.zip'),
]:
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location='cpu', weights_only=True)['dev_srcc']
        print(f'  {name:9s} pretrain_dev SRCC {s:.4f}  -> {zipf} ({"ready" if os.path.exists(zipf) else "MISSING"})')
print('\\nDownload both submission_b*.zip from the Output tab; upload each to CodaBench. Compare vs 0.574.')'''),
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_approach_b.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Written: {out}  ({len(nb_cells)} cells)")
