"""Generate CodaBench submission CSV from a trained checkpoint.

Usage:
    python scripts/predict.py \
        --checkpoint checkpoints/finetune/best.pt \
        --manifest data/manifests/evalset.csv \
        --config configs/finetune.yaml \
        --output submission.csv
"""

import argparse
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MOSDataset
from src.model import WhisperMOSNet


def predict(checkpoint_path: str, manifest_path: str, config_path: str, output_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg = cfg["model"]

    model = WhisperMOSNet(
        whisper_model=m_cfg["whisper_model"], proj_dim=m_cfg["proj_dim"]
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.train(False)  # inference mode

    ds = MOSDataset(manifest_path, whisper_model=m_cfg["whisper_model"])
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    df = pd.read_csv(manifest_path)
    acr_preds, ccr_preds = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            acr, ccr = model(batch["input_features"].to(device), batch["waveform"].to(device))
            acr_preds.extend(acr.cpu().tolist())
            ccr_preds.extend(ccr.cpu().tolist())

    pd.DataFrame({
        "filename": df["path"].apply(os.path.basename),
        "acr_pred": acr_preds,
        "ccr_pred": ccr_preds,
    }).to_csv(output_path, index=False)
    print(f"Submission -> {output_path} ({len(acr_preds)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    predict(args.checkpoint, args.manifest, args.config, args.output)


if __name__ == "__main__":
    main()
