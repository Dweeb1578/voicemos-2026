"""Zero-shot DNSMOS P.835 predictions for the Track 1 dev set.

DNSMOS (microsoft/DNS-Challenge) predicts subjective quality of *enhanced
speech* -- the Track 1 target domain -- so it is run zero-shot on CPU as
(a) a standalone diagnostic submission and (b) an ensemble member next to B1.

ACR rows = OVRL(clip). CCR rows = OVRL(A) - OVRL(B) (convention
quality(A) - quality(B), matching scripts/predict_dev.py -- the sign was
once inverted and cost a submission, see the 2026-06-06 entry there).

Windowing mirrors the reference dnsmos_local.py: resample to 16 kHz, tile
clips shorter than 9.01 s, slide a 9.01 s window at a 1 s hop, average the
per-window scores after the published polynomial calibration. One deliberate
deviation: short clips are tiled to exactly ONE window (the reference's
doubling loop overshoots and scores up to 7 phase-shifted copies of the same
looped content -- pure redundancy at up to 7x the inference cost).

Clip scores append to --clip-scores as they compute; already-scored clips
are skipped on rerun, so a crash midway through ~6k clips resumes for free.

Usage:
    python -m scripts.zero_shot_dnsmos --limit 5     # smoke
    python -m scripts.zero_shot_dnsmos               # full pass + submission
"""

import argparse
import csv
import math
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from scripts.submission import rescale_to_range, write_submission_zip

SAMPLING_RATE = 16000
INPUT_LENGTH_S = 9.01  # fixed model input length, seconds

# Published P.835 calibration polynomials (microsoft/DNS-Challenge
# dnsmos_local.py, non-personalized branch). Monotone over the raw-score
# range, so SRCC-neutral; applied anyway so stored scores read as 1-5 MOS.
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def load_audio_16k(path):
    """Load any-rate FLAC/WAV as mono float32 at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLING_RATE:
        g = math.gcd(sr, SAMPLING_RATE)
        audio = resample_poly(audio, SAMPLING_RATE // g, sr // g).astype(np.float32)
    return audio


def segment_audio(audio, fs=SAMPLING_RATE, input_length_s=INPUT_LENGTH_S, hop_s=1.0):
    """Reference windowing: tile short clips, then 1 s-hop fixed windows."""
    if len(audio) == 0:
        # The tiling loop below never grows an empty array -- a corrupt clip
        # must crash (resumable) rather than hang an unattended pass.
        raise ValueError("zero-length audio")
    len_samples = int(input_length_s * fs)
    if len(audio) < len_samples:
        while len(audio) < len_samples:
            audio = np.concatenate([audio, audio])
        audio = audio[:len_samples]
    num_hops = int(np.floor(len(audio) / fs) - input_length_s) + 1
    hop_samples = int(hop_s * fs)
    segments = []
    for idx in range(num_hops):
        seg = audio[idx * hop_samples : idx * hop_samples + len_samples]
        if len(seg) == len_samples:
            segments.append(seg)
    return segments


def score_clip(infer, audio):
    """Calibrated (sig, bak, ovrl) averaged over all windows.

    `infer` maps one float32 segment -> raw (sig, bak, ovr) triple; injected
    so tests run without onnxruntime."""
    sigs, baks, ovrs = [], [], []
    for seg in segment_audio(audio):
        sig_raw, bak_raw, ovr_raw = infer(seg)
        sigs.append(float(P_SIG(sig_raw)))
        baks.append(float(P_BAK(bak_raw)))
        ovrs.append(float(P_OVR(ovr_raw)))
    return float(np.mean(sigs)), float(np.mean(baks)), float(np.mean(ovrs))


def load_clip_scores(path):
    """{clip_path: ovrl} for clips already scored (resume support)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    # A crash can tear the last row (missing fields -> NaN). Dropping it here
    # makes the cache self-healing: the clip is simply rescored on resume.
    df = pd.read_csv(path).dropna(subset=["ovrl"])
    return dict(zip(df["path"], df["ovrl"]))


def score_all(infer, clip_paths, clip_scores_csv):
    """Score every path not already in clip_scores_csv, appending row-by-row
    (flushed) so progress survives a crash. Returns {path: ovrl} for ALL
    requested paths, cached and new."""
    done = load_clip_scores(clip_scores_csv)
    todo = [p for p in clip_paths if p not in done]
    os.makedirs(os.path.dirname(clip_scores_csv) or ".", exist_ok=True)
    new_file = (not os.path.exists(clip_scores_csv)
                or os.path.getsize(clip_scores_csv) == 0)
    if not new_file:
        # If a crash tore the last line mid-write (no trailing newline),
        # appending would merge onto it and corrupt the CSV beyond what
        # load_clip_scores can heal. Terminate the torn line first.
        with open(clip_scores_csv, "rb+") as fb:
            fb.seek(-1, os.SEEK_END)
            if fb.read(1) != b"\n":
                fb.write(b"\n")
    with open(clip_scores_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["path", "sig", "bak", "ovrl"])
        for p in tqdm(todo, desc="DNSMOS"):
            sig, bak, ovrl = score_clip(infer, load_audio_16k(p))
            writer.writerow([p, sig, bak, ovrl])
            f.flush()
            done[p] = ovrl
    return {p: done[p] for p in clip_paths}


def make_onnx_infer(model_path):
    """Wrap the ONNX session as a segment -> raw (sig, bak, ovr) callable.
    onnxruntime is imported here, not module-top, so the pure helpers stay
    importable (and testable) without it."""
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name  # 'input_1'

    def infer(segment):
        out = sess.run(None, {input_name: segment[np.newaxis, :].astype(np.float32)})
        sig_raw, bak_raw, ovr_raw = out[0][0]
        return float(sig_raw), float(bak_raw), float(ovr_raw)

    return infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/dnsmos/sig_bak_ovr.onnx")
    parser.add_argument("--acr-manifest", default="data/manifests/dev_acr.csv")
    parser.add_argument("--ccr-manifest", default="data/manifests/dev_ccr.csv")
    parser.add_argument("--clip-scores", default="outputs/dnsmos_clip_scores.csv")
    parser.add_argument("--output", default="outputs/dnsmos_predictions.csv")
    parser.add_argument("--zip", default="outputs/submission_dnsmos.zip")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke mode: score only the first N ACR clips, "
                             "print them, and write NO submission")
    args = parser.parse_args()

    infer = make_onnx_infer(args.model)
    acr_df = pd.read_csv(args.acr_manifest)
    ccr_df = pd.read_csv(args.ccr_manifest)

    if args.limit:
        paths = list(acr_df["path"])[: args.limit]
        scores = score_all(infer, paths, args.clip_scores)
        for p in paths:
            print(f"{p} -> ovrl {scores[p]:.3f}")
        return

    all_paths = (list(acr_df["path"]) + list(ccr_df["path_a"])
                 + list(ccr_df["path_b"]))
    unique = list(dict.fromkeys(all_paths))
    print(f"{len(unique)} unique clips "
          f"({len(acr_df)} ACR rows, {len(ccr_df)} CCR pairs)")
    ovrl = score_all(infer, unique, args.clip_scores)

    acr_scores = rescale_to_range(
        [ovrl[r["path"]] for _, r in acr_df.iterrows()], 1.0, 5.0)
    ccr_scores = rescale_to_range(
        [ovrl[r["path_a"]] - ovrl[r["path_b"]] for _, r in ccr_df.iterrows()],
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
