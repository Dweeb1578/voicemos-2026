"""Zero-shot Audiobox-Aesthetics predictions for the Track 1 dev set.

facebookresearch/audiobox-aesthetics scores audio on four AESTHETIC axes --
CE (Content Enjoyment), CU (Content Usefulness), PC (Production Complexity),
PQ (Production Quality) -- learned from a broad speech+music+sound corpus
(LibriTTS, Common Voice, EARS, MUSDB18, MusicCaps, AudioSet, PAM). Its training
objective is *aesthetic quality*, NOT enhancement/communication MOS like every
existing FRANK member (whisper students, DNSMOS, NISQA, SQUIM, SCOREQ,
Distill-MOS). That distinct objective is the bet: a genuinely orthogonal error
direction, the profile that ADDS to the ensemble (Distill-MOS +0.0098,
SCOREQ +0.0049) rather than a same-data twin (WavLM diluted -0.0048).

Run zero-shot on CPU. Each axis becomes a candidate member: ACR = axis(clip),
CCR = axis(A) - axis(B) (quality(A)-quality(B), matching scripts/predict_dev.py
and zero_shot_dnsmos.py). All four axis CSVs are written so they can be screened
(SRCC vs urgent2025-sqa real MOS x decorrelation vs ensemble) before adoption.

Clip scores append to --clip-scores row-by-row (flushed); already-scored clips
are skipped on rerun, so a crash midway through ~6k clips resumes for free.

Usage:
    python -m scripts.zero_shot_audiobox --limit 5     # smoke
    python -m scripts.zero_shot_audiobox               # full pass + 4 axis CSVs
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from scripts.submission import rescale_to_range, write_submission_zip

AXES = ["CE", "CU", "PC", "PQ"]


def load_item(path):
    """Load a FLAC/WAV as an audiobox input dict carrying a pre-decoded tensor.

    Passing a tensor + explicit sample_rate makes AesPredictor use it directly
    (infer.py audio_resample_mono branch), bypassing its torchcodec/ffmpeg path
    loader -- which has no usable DLLs on this machine. soundfile + torch is the
    same decode stack every other zero-shot member already uses here."""
    import torch
    audio, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
    wav = torch.from_numpy(audio.T.copy())  # -> (ch, frames)
    return {"path": wav, "sample_rate": sr}


def load_clip_scores(path):
    """{clip_path: {axis: score}} for clips already scored (resume support)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    # A torn last row (crash mid-write) drops to NaN -> rescored on resume.
    df = pd.read_csv(path).dropna(subset=AXES)
    return {r["path"]: {a: float(r[a]) for a in AXES} for _, r in df.iterrows()}


def score_all(predictor, clip_paths, clip_scores_csv, batch=16):
    """Score every path not already cached, appending row-by-row (flushed).
    Returns {path: {axis: score}} for ALL requested paths."""
    done = load_clip_scores(clip_scores_csv)
    todo = [p for p in clip_paths if p not in done]
    os.makedirs(os.path.dirname(clip_scores_csv) or ".", exist_ok=True)
    new_file = (not os.path.exists(clip_scores_csv)
                or os.path.getsize(clip_scores_csv) == 0)
    if not new_file:
        # Terminate a torn last line before appending (see zero_shot_dnsmos.py).
        with open(clip_scores_csv, "rb+") as fb:
            fb.seek(-1, os.SEEK_END)
            if fb.read(1) != b"\n":
                fb.write(b"\n")
    with open(clip_scores_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["path"] + AXES)
        for i in tqdm(range(0, len(todo), batch), desc="Audiobox"):
            chunk = todo[i:i + batch]
            res = predictor.forward([load_item(p) for p in chunk])
            for p, r in zip(chunk, res):
                writer.writerow([p] + [r[a] for a in AXES])
                done[p] = {a: float(r[a]) for a in AXES}
            f.flush()
    return {p: done[p] for p in clip_paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acr-manifest", default="data/manifests/dev_acr.csv")
    parser.add_argument("--ccr-manifest", default="data/manifests/dev_ccr.csv")
    parser.add_argument("--clip-scores", default="outputs/audiobox_clip_scores.csv")
    parser.add_argument("--out-prefix", default="outputs/audiobox",
                        help="per-axis CSVs written as <prefix>_<AXIS>_predictions.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke mode: score first N ACR clips, print, no CSVs")
    args = parser.parse_args()

    from audiobox_aesthetics.infer import initialize_predictor
    predictor = initialize_predictor()

    acr_df = pd.read_csv(args.acr_manifest)
    ccr_df = pd.read_csv(args.ccr_manifest)

    if args.limit:
        paths = list(acr_df["path"])[: args.limit]
        scores = score_all(predictor, paths, args.clip_scores)
        for p in paths:
            print(f"{p} -> " + " ".join(f"{a}={scores[p][a]:.3f}" for a in AXES))
        return

    all_paths = (list(acr_df["path"]) + list(ccr_df["path_a"])
                 + list(ccr_df["path_b"]))
    unique = list(dict.fromkeys(all_paths))
    print(f"{len(unique)} unique clips "
          f"({len(acr_df)} ACR rows, {len(ccr_df)} CCR pairs)")
    sc = score_all(predictor, unique, args.clip_scores)

    for axis in AXES:
        acr_scores = rescale_to_range(
            [sc[r["path"]][axis] for _, r in acr_df.iterrows()], 1.0, 5.0)
        ccr_scores = rescale_to_range(
            [sc[r["path_a"]][axis] - sc[r["path_b"]][axis]
             for _, r in ccr_df.iterrows()], -3.0, 3.0)
        acr_rows = [{"sample_id": r["sample_id"], "pred_score": s}
                    for (_, r), s in zip(acr_df.iterrows(), acr_scores)]
        ccr_rows = [{"sample_id": r["sample_id"], "pred_score": s}
                    for (_, r), s in zip(ccr_df.iterrows(), ccr_scores)]
        out_df = pd.DataFrame(acr_rows + ccr_rows,
                              columns=["sample_id", "pred_score"])
        assert not out_df["pred_score"].isna().any(), f"NaN in {axis}"
        out_path = f"{args.out_prefix}_{axis}_predictions.csv"
        out_df.to_csv(out_path, index=False)
        write_submission_zip(out_path, f"{args.out_prefix}_{axis}_submission.zip")
        print(f"{axis}: {out_path} ({len(out_df)} rows) + submission zip")


if __name__ == "__main__":
    main()
