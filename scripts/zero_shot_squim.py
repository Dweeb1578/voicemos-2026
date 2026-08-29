"""Zero-shot SQUIM predictions for the Track 1 dev set.

torchaudio's SQUIM_OBJECTIVE estimates reference-free objective enhancement
metrics (STOI, PESQ, SI-SDR) -- a third enhancement-domain quality signal,
decorrelated from DNSMOS by architecture and training target. Primary score
for the submission: **predicted PESQ** (the most perceptual of the three);
the clip CSV keeps all three for later use as separate ensemble members.

ACR rows = pesq(clip). CCR rows = pesq(A) - pesq(B) (convention
quality(A) - quality(B), matching scripts/predict_dev.py).

Runs on GPU if available (the model is plain PyTorch; torchaudio's wheel
being +cpu doesn't matter), else CPU. Resumable via scripts/clip_cache.

Usage:
    python -m scripts.zero_shot_squim --limit 5     # smoke
    python -m scripts.zero_shot_squim               # full pass + submission
"""

import argparse

import pandas as pd
import torch

from scripts.clip_cache import score_paths
from scripts.submission import rescale_to_range, write_submission_zip
from scripts.zero_shot_dnsmos import load_audio_16k

COLS = ("stoi", "pesq", "si_sdr")
PRIMARY = "pesq"


def make_squim_scorer(device=None):
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SQUIM_OBJECTIVE.get_model().eval().to(device)

    def scorer(path):
        wav = torch.from_numpy(load_audio_16k(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            stoi, pesq, si_sdr = model(wav)
        return float(stoi.item()), float(pesq.item()), float(si_sdr.item())

    return scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acr-manifest", default="data/manifests/dev_acr.csv")
    parser.add_argument("--ccr-manifest", default="data/manifests/dev_ccr.csv")
    parser.add_argument("--clip-scores", default="outputs/squim_clip_scores.csv")
    parser.add_argument("--output", default="outputs/squim_predictions.csv")
    parser.add_argument("--zip", default="outputs/submission_squim.zip")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke mode: score only the first N ACR clips, "
                             "print them, and write NO submission")
    args = parser.parse_args()

    scorer = make_squim_scorer(args.device)
    acr_df = pd.read_csv(args.acr_manifest)
    ccr_df = pd.read_csv(args.ccr_manifest)
    idx = COLS.index(PRIMARY)

    if args.limit:
        paths = list(acr_df["path"])[: args.limit]
        scores = score_paths(scorer, paths, args.clip_scores, COLS, desc="SQUIM")
        for p in paths:
            stoi, pesq, si_sdr = scores[p]
            print(f"{p} -> stoi {stoi:.3f} pesq {pesq:.3f} si_sdr {si_sdr:.2f}")
        return

    all_paths = (list(acr_df["path"]) + list(ccr_df["path_a"])
                 + list(ccr_df["path_b"]))
    unique = list(dict.fromkeys(all_paths))
    print(f"{len(unique)} unique clips "
          f"({len(acr_df)} ACR rows, {len(ccr_df)} CCR pairs)")
    scores = score_paths(scorer, unique, args.clip_scores, COLS, desc="SQUIM")
    quality = {p: v[idx] for p, v in scores.items()}

    acr_scores = rescale_to_range(
        [quality[r["path"]] for _, r in acr_df.iterrows()], 1.0, 5.0)
    ccr_scores = rescale_to_range(
        [quality[r["path_a"]] - quality[r["path_b"]] for _, r in ccr_df.iterrows()],
        -3.0, 3.0)

    acr_rows = [{"sample_id": r["sample_id"], "pred_score": s}
                for (_, r), s in zip(acr_df.iterrows(), acr_scores)]
    ccr_rows = [{"sample_id": r["sample_id"], "pred_score": s}
                for (_, r), s in zip(ccr_df.iterrows(), ccr_scores)]
    out_df = pd.DataFrame(acr_rows + ccr_rows, columns=["sample_id", "pred_score"])
    assert not out_df["pred_score"].isna().any(), "NaN in predictions"
    out_df.to_csv(args.output, index=False)
    print(f"predictions -> {args.output} "
          f"({len(acr_rows)} ACR + {len(ccr_rows)} CCR = {len(out_df)} rows)")
    write_submission_zip(args.output, args.zip)
    print(f"submission -> {args.zip}")


if __name__ == "__main__":
    main()
