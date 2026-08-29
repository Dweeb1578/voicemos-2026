"""Zero-shot DNSMOS P.808 scores for the Track 1 dev set.

model_v8.onnx is the SECOND model in microsoft/DNS-Challenge -- a different
network from the P.835 one (mel-spectrogram input, trained on crowdsourced
ITU-T P.808 ACR ratings rather than P.835 SIG/BAK/OVRL). Candidate ensemble
member; admission is decided OFFLINE by the pre-registered rule (decent solo
prior + agreement <= ~0.75 vs existing members) before spending any upload.

Mirrors the reference dnsmos_local.py exactly: same 9.01 s / 1 s-hop windows
as P.835, mel of segment[:-160] (320-sample frame, 160 hop, 120 mels,
power_to_db ref=max then (db+40)/40), raw model output averaged over windows
(no calibration polynomial for P.808).

Output: outputs/p808_clip_scores.csv (path, p808). Convert with
scripts/member_predictions.py --score-col p808.

Usage:
    python -m scripts.zero_shot_p808 [--limit 5]
"""

import argparse

import librosa
import numpy as np
import pandas as pd

from scripts.clip_cache import score_paths
from scripts.zero_shot_dnsmos import load_audio_16k, segment_audio

COLS = ("p808",)


def audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000):
    """Reference audio_melspec, to_db branch included (the reference default)."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels)
    mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
    return mel.T


def make_p808_scorer(model_path="data/dnsmos/model_v8.onnx"):
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name  # 'input_1', (N, 900, 120)

    def scorer(path):
        audio = load_audio_16k(path)
        seg_scores = []
        for seg in segment_audio(audio):
            feats = audio_melspec(seg[:-160]).astype(np.float32)[np.newaxis, :, :]
            seg_scores.append(float(sess.run(None, {input_name: feats})[0][0][0]))
        return (float(np.mean(seg_scores)),)

    return scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/dnsmos/model_v8.onnx")
    parser.add_argument("--acr-manifest", default="data/manifests/dev_acr.csv")
    parser.add_argument("--ccr-manifest", default="data/manifests/dev_ccr.csv")
    parser.add_argument("--clip-scores", default="outputs/p808_clip_scores.csv")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scorer = make_p808_scorer(args.model)
    acr_df = pd.read_csv(args.acr_manifest)
    ccr_df = pd.read_csv(args.ccr_manifest)

    if args.limit:
        paths = list(acr_df["path"])[: args.limit]
        scores = score_paths(scorer, paths, args.clip_scores, COLS, desc="P808")
        for p in paths:
            print(f"{p} -> p808 {scores[p][0]:.3f}")
        return

    all_paths = (list(acr_df["path"]) + list(ccr_df["path_a"])
                 + list(ccr_df["path_b"]))
    unique = list(dict.fromkeys(all_paths))
    print(f"{len(unique)} unique clips")
    score_paths(scorer, unique, args.clip_scores, COLS, desc="P808")
    df = pd.read_csv(args.clip_scores)
    print(f"clip scores -> {args.clip_scores} ({len(df)} rows, "
          f"min {df.p808.min():.3f} max {df.p808.max():.3f} std {df.p808.std():.3f})")


if __name__ == "__main__":
    main()
