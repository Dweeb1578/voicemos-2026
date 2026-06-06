"""Generate a CodaBench submission for the VoiceMOS 2026 Track 1 dev set.

Submission format (single flat file, all rows stacked):
    sample_id,pred_score
    vmc2026-track1-dev-acr_489,3.42
    vmc2026-track1-dev-ccr_7233,-0.15

ACR is predicted directly by the (pretrained) ACR head, clamped to [1, 5].
CCR has no trained head yet, so it is derived from the ACR head as a
comparative score:  ccr = clamp(acr_a - acr_b, -3, 3).  Because SRCC is
rank-based, an ACR model that ranks quality well also ranks A-vs-B
preferences reasonably -- a legitimate baseline to validate the pipeline.

Subtraction order (a - b, not b - a) was corrected after the 2026-06-06 dev
submission scored CCR UTT-SRCC = -0.3685: the ground-truth convention is
quality(A) - quality(B), so b - a was sign-inverted.  No CCR value saturates
the +/-3 clamp, so flipping the order negates each prediction exactly and
flips the SRCC to +0.3685.

Usage:
    python scripts/predict_dev.py \
        --checkpoint checkpoints/pretrain/best.pt \
        --config configs/pretrain.yaml \
        --acr-manifest data/manifests/dev_acr.csv \
        --ccr-manifest data/manifests/dev_ccr.csv \
        --output predictions.csv
"""

import argparse
import os
import zipfile

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from data.preprocess import resample_and_normalize, trim_and_pad
from src.model import WhisperMOSNet


class _InferenceClips(Dataset):
    """Minimal label-free dataset: yields Whisper features + waveform per path."""

    def __init__(self, paths, whisper_model):
        self.paths = list(paths)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, _ = resample_and_normalize(self.paths[idx])
        audio = trim_and_pad(audio)
        features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        return {
            "input_features": features.input_features.squeeze(0),
            "waveform": torch.tensor(audio, dtype=torch.float32),
        }


def _predict_acr(model, paths, whisper_model, device, batch_size):
    """Return a dict {path: acr_score} for every unique path."""
    unique = list(dict.fromkeys(paths))  # dedupe, preserve order
    ds = _InferenceClips(unique, whisper_model)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ACR inference"):
            acr, _ = model(
                batch["input_features"].to(device),
                batch["waveform"].to(device),
            )
            preds.extend(acr.cpu().tolist())
    return dict(zip(unique, preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain/best.pt")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--acr-manifest", default="data/manifests/dev_acr.csv")
    parser.add_argument("--ccr-manifest", default="data/manifests/dev_ccr.csv")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--zip", default="submission.zip")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["model"]
    whisper_model = m_cfg["whisper_model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperMOSNet(whisper_model=whisper_model, proj_dim=m_cfg["proj_dim"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- ACR: direct prediction, clamped to [1, 5] ---
    acr_df = pd.read_csv(args.acr_manifest)
    acr_scores = _predict_acr(model, acr_df["path"], whisper_model, device, args.batch_size)
    acr_rows = [
        {"sample_id": r["sample_id"], "pred_score": min(5.0, max(1.0, acr_scores[r["path"]]))}
        for _, r in acr_df.iterrows()
    ]

    # --- CCR: derive from ACR head, ccr = clamp(acr_a - acr_b, -3, 3) ---
    ccr_df = pd.read_csv(args.ccr_manifest)
    all_ccr_paths = list(ccr_df["path_a"]) + list(ccr_df["path_b"])
    ccr_acr = _predict_acr(model, all_ccr_paths, whisper_model, device, args.batch_size)
    ccr_rows = [
        {
            "sample_id": r["sample_id"],
            "pred_score": min(3.0, max(-3.0, ccr_acr[r["path_a"]] - ccr_acr[r["path_b"]])),
        }
        for _, r in ccr_df.iterrows()
    ]

    out_df = pd.DataFrame(acr_rows + ccr_rows, columns=["sample_id", "pred_score"])
    out_df.to_csv(args.output, index=False)
    print(f"predictions -> {args.output} "
          f"({len(acr_rows)} ACR + {len(ccr_rows)} CCR = {len(out_df)} rows)")

    # zip -j equivalent: store predictions.csv at archive root, no folders
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.output, arcname=os.path.basename(args.output))
    print(f"submission -> {args.zip}")


if __name__ == "__main__":
    main()
