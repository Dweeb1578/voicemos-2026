"""Predict the Track 1 dev set for one or more saved checkpoints and build CodaBench
submissions -- so a fresh Kaggle session can score uploaded models without re-training.

Usage (run from the repo root):
    python -m scripts.predict_saved <ckpt1.pt> [<ckpt2.pt> ...]

Each checkpoint -> /kaggle/working/submission_<stem>.zip (containing predictions.csv).
Assumes the mel-on experiment architecture (final layer, proj_dim 256) -- matches both
exp_mel_rank and exp_mel_norank. Needs data/manifests/dev_acr.csv + dev_ccr.csv present
(run data/prepare_dev.py first) and internet/HF_TOKEN to fetch whisper-medium once.
"""
import os
import sys
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.predict_dev import _InferenceClips
from src.model import WhisperMOSNet

WHISPER_MODEL = "openai/whisper-medium"
OUT_DIR = "/kaggle/working"


def _predict_acr(model, paths, device, desc):
    unique = list(dict.fromkeys(paths))
    loader = DataLoader(_InferenceClips(unique, WHISPER_MODEL),
                        batch_size=32, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            acr, _ = model(batch["input_features"].to(device), batch["waveform"].to(device))
            preds.extend(acr.cpu().tolist())
    return dict(zip(unique, preds))


def main():
    checkpoints = sys.argv[1:]
    assert checkpoints, "usage: python -m scripts.predict_saved <ckpt.pt> [<ckpt.pt> ...]"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acr_df = pd.read_csv("data/manifests/dev_acr.csv")
    ccr_df = pd.read_csv("data/manifests/dev_ccr.csv")

    for ckpt in checkpoints:
        model = WhisperMOSNet(whisper_model=WHISPER_MODEL, proj_dim=256,
                              encoder_layer=-1, use_mel_branch=True).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"], strict=False)
        model.eval()
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        print(f"\n>> {ckpt} | dev_srcc {state.get('dev_srcc')}")

        acr_scores = _predict_acr(model, acr_df["path"], device, "ACR")
        acr_rows = [{"sample_id": r["sample_id"], "pred_score": min(5.0, max(1.0, acr_scores[r["path"]]))}
                    for _, r in acr_df.iterrows()]
        ccr_scores = _predict_acr(model, list(ccr_df["path_a"]) + list(ccr_df["path_b"]), device, "CCR")
        ccr_rows = [{"sample_id": r["sample_id"],
                     "pred_score": min(3.0, max(-3.0, ccr_scores[r["path_a"]] - ccr_scores[r["path_b"]]))}
                    for _, r in ccr_df.iterrows()]

        out_csv = os.path.join(OUT_DIR, f"predictions_{stem}.csv")
        zip_path = os.path.join(OUT_DIR, f"submission_{stem}.zip")
        pd.DataFrame(acr_rows + ccr_rows, columns=["sample_id", "pred_score"]).to_csv(out_csv, index=False)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_csv, arcname="predictions.csv")  # CodaBench-required name
        del model
        torch.cuda.empty_cache()
        print(f"-> {zip_path} | {len(acr_rows) + len(ccr_rows)} rows")

    print("\nDONE -- download submission_*.zip from the Output tab.")


if __name__ == "__main__":
    main()
