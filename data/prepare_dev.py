"""Download the VoiceMOS 2026 Track 1 dev set and materialize it locally.

The HuggingFace dataset ships audio embedded inside parquet. We disable HF's
audio auto-decoder (which requires torchcodec/FFmpeg) and write the raw FLAC
bytes straight to disk -- lossless, no re-encode -- so the rest of the pipeline
(librosa-based MOSDataset) can read plain file paths, exactly like the
BVCC / TMHINT sources.

Two manifests are produced:
    data/manifests/dev_acr.csv  -> sample_id, path            (1008 rows)
    data/manifests/dev_ccr.csv  -> sample_id, path_a, path_b  (2520 rows)

Usage:
    python data/prepare_dev.py --output data/datasets/track1_dev
"""

import argparse
import os

import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm

HF_REPO = "urgent-challenge/vmc2026-track1-dev"


def _save_clip(audio_obj, dest: str) -> None:
    """Write the raw (undecoded) FLAC bytes of an audio field to dest."""
    if os.path.exists(dest):
        return
    with open(dest, "wb") as f:
        f.write(audio_obj["bytes"])


def prepare_acr(output_dir: str, manifest_dir: str) -> None:
    wav_dir = os.path.join(output_dir, "acr")
    os.makedirs(wav_dir, exist_ok=True)
    ds = load_dataset(HF_REPO, "acr", split="dev").cast_column("audio", Audio(decode=False))

    rows = []
    for row in tqdm(ds, desc="ACR"):
        sid = row["sample_id"]
        dest = os.path.join(wav_dir, f"{sid}.flac")
        _save_clip(row["audio"], dest)
        rows.append({"sample_id": sid, "path": dest})

    out = os.path.join(manifest_dir, "dev_acr.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ACR manifest -> {out} ({len(rows)} rows)")


def prepare_ccr(output_dir: str, manifest_dir: str) -> None:
    wav_dir = os.path.join(output_dir, "ccr")
    os.makedirs(wav_dir, exist_ok=True)
    ds = load_dataset(HF_REPO, "ccr", split="dev")
    ds = ds.cast_column("audio_a", Audio(decode=False)).cast_column("audio_b", Audio(decode=False))

    rows = []
    for row in tqdm(ds, desc="CCR"):
        sid = row["sample_id"]
        path_a = os.path.join(wav_dir, f"{sid}_a.flac")
        path_b = os.path.join(wav_dir, f"{sid}_b.flac")
        _save_clip(row["audio_a"], path_a)
        _save_clip(row["audio_b"], path_b)
        rows.append({"sample_id": sid, "path_a": path_a, "path_b": path_b})

    out = os.path.join(manifest_dir, "dev_ccr.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"CCR manifest -> {out} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/datasets/track1_dev")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--configs", nargs="+", choices=["acr", "ccr"],
                        default=["acr", "ccr"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.manifest_dir, exist_ok=True)

    if "acr" in args.configs:
        prepare_acr(args.output, args.manifest_dir)
    if "ccr" in args.configs:
        prepare_ccr(args.output, args.manifest_dir)
    print("Dev set prepared.")


if __name__ == "__main__":
    main()
