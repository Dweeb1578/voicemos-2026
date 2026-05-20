"""Run once to generate colab_train.ipynb."""
import json
import os

cells = [
    # Cell 0: mount Drive and clone repo
    [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "import os, subprocess\n",
        "REPO_URL = 'https://github.com/YOUR_USERNAME/voicemos-2026.git'\n",
        "REPO_DIR = '/content/voicemos-2026'\n",
        "CKPT_DIR = '/content/drive/MyDrive/voicemos2026/checkpoints'\n",
        "if not os.path.exists(REPO_DIR):\n",
        "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)\n",
        "else:\n",
        "    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)\n",
        "os.chdir(REPO_DIR)\n",
        "os.makedirs(CKPT_DIR, exist_ok=True)\n",
    ],
    # Cell 1: install dependencies
    [
        "import subprocess\n",
        "subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'], check=True)\n",
    ],
    # Cell 2: download datasets
    [
        "import subprocess\n",
        "subprocess.run(\n",
        "    ['python', 'data/download.py', '--output', 'data/datasets',\n",
        "     '--datasets', 'bvcc', 'tmhint', 'audiomos25t3'],\n",
        "    check=True\n",
        ")\n",
    ],
    # Cell 3: build manifests
    [
        "import subprocess\n",
        "subprocess.run(\n",
        "    ['python', 'data/build_manifests.py',\n",
        "     '--data_dir', 'data/datasets', '--output_dir', 'data/manifests'],\n",
        "    check=True\n",
        ")\n",
    ],
    # Cell 4: pretrain
    [
        "import subprocess\n",
        "subprocess.run(['python', 'src/train.py', '--config', 'configs/pretrain.yaml'], check=True)\n",
    ],
    # Cell 5: save checkpoint to Drive
    [
        "import shutil, os\n",
        "CKPT_DIR = '/content/drive/MyDrive/voicemos2026/checkpoints'\n",
        "shutil.copy('checkpoints/pretrain/best.pt', f'{CKPT_DIR}/pretrain_best.pt')\n",
        "print('Checkpoint saved to Drive.')\n",
    ],
    # Cell 6: evaluate and compare to baseline
    [
        "import torch\n",
        "from torch.utils.data import DataLoader\n",
        "from src.model import WhisperMOSNet\n",
        "from src.dataset import MOSDataset\n",
        "from src.evaluate import compute_metrics\n",
        "\n",
        "device = torch.device('cuda')\n",
        "model = WhisperMOSNet(whisper_model='openai/whisper-medium', proj_dim=256).to(device)\n",
        "ckpt = torch.load('checkpoints/pretrain/best.pt', map_location=device)\n",
        "model.load_state_dict(ckpt['model_state'])\n",
        "model.train(False)\n",
        "\n",
        "ds = MOSDataset('data/manifests/pretrain_dev.csv', whisper_model='openai/whisper-medium')\n",
        "loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)\n",
        "\n",
        "acr_preds, acr_targets = [], []\n",
        "with torch.no_grad():\n",
        "    for batch in loader:\n",
        "        acr, _ = model(batch['input_features'].to(device), batch['waveform'].to(device))\n",
        "        acr_preds.extend(acr.cpu().tolist())\n",
        "        acr_targets.extend(batch['acr'].tolist())\n",
        "\n",
        "m = compute_metrics(acr_preds, acr_targets)\n",
        "print(f\"Dev ACR  SRCC={m['srcc']:.4f}  LCC={m['lcc']:.4f}  MSE={m['mse']:.4f}\")\n",
        "beat = 'BEAT' if m['srcc'] > 0.780 else 'not yet'\n",
        "print(f'Baseline (MOSA-Net+): SRCC=0.780 -- {beat}')\n",
    ],
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": [
        {"cell_type": "code", "source": src, "metadata": {}, "outputs": [], "execution_count": None, "id": str(i)}
        for i, src in enumerate(cells)
    ],
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_train.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=2)
print(f"Written: {out}")
