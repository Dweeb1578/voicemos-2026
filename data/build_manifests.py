"""Build unified manifest CSVs from each downloaded dataset.

Usage:
    python data/build_manifests.py --data_dir data/datasets --output_dir data/manifests
"""

import argparse
import os
import random

import pandas as pd

from data.preprocess import write_manifest


def parse_bvcc(bvcc_dir: str) -> list:
    """Parse BVCC dataset (VoiceMOS 2022 main track).

    Expected structure:
        bvcc_dir/sets/{TRAINSET,DEVSET}/mydata_system.csv
        bvcc_dir/sets/{TRAINSET,DEVSET}/wav/{system_ID}_{file_name}.wav
    CSV columns: system_ID, file_name, mean_score
    """
    rows = []
    for split_name in ("TRAINSET", "DEVSET"):
        split_dir = os.path.join(bvcc_dir, "sets", split_name)
        label_path = os.path.join(split_dir, "mydata_system.csv")
        wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(label_path):
            continue
        df = pd.read_csv(label_path)
        for _, row in df.iterrows():
            wav_name = f"{row['system_ID']}_{row['file_name']}.wav"
            wav_path = os.path.join(wav_dir, wav_name)
            if not os.path.exists(wav_path):
                continue
            rows.append({
                "path": os.path.abspath(wav_path),
                "acr": float(row["mean_score"]),
                "ccr": float("nan"),
                "language": "en",
                "system": str(row["system_ID"]),
                "split": "train" if split_name == "TRAINSET" else "dev",
            })
    return rows


def parse_tmhint(tmhint_dir: str, label_csv: str = "VOICEMOS2023_DISTRO/train_avg_mos.csv", wav_subdir: str = "train") -> list:
    """Parse TMHINT-QI dataset (VoiceMOS 2023 Track 3).

    Uses train_avg_mos.csv from VOICEMOS2023_DISTRO (columns: uttID, avgMOS).
    Audio lives in the InQSS dataset 'train/' subdirectory.
    Falls back to generic 'filename'/'quality' columns if those aren't present.
    """
    label_path = os.path.join(tmhint_dir, label_csv)
    wav_dir = os.path.join(tmhint_dir, wav_subdir)
    df = pd.read_csv(label_path)
    rows = []

    # Normalise column names to (filename_col, mos_col)
    if "uttID" in df.columns and "avgMOS" in df.columns:
        filename_col, mos_col = "uttID", "avgMOS"
        # uttID is bare stem (e.g. "FCN_snr0_babble_TMHINT_g1_14_01"); append .wav
        df["_wavfile"] = df[filename_col] + ".wav"
    else:
        filename_col, mos_col = "filename", "quality"
        df["_wavfile"] = df[filename_col]

    for _, row in df.iterrows():
        wav_path = os.path.join(wav_dir, row["_wavfile"])
        if not os.path.exists(wav_path):
            continue
        fname = row["_wavfile"]
        rows.append({
            "path": os.path.abspath(wav_path),
            "acr": float(row[mos_col]),
            "ccr": float("nan"),
            "language": "zh",
            "system": str(fname).split("_")[0],
            "split": "train",
        })
    return rows


def parse_audiomos25t3(audiomos_dir: str, label_csv: str = "labels.csv", wav_subdir: str = "wav") -> list:
    """Parse AudioMOS 2025 Track 3 dataset.
    CSV columns: filename, mos, sr
    """
    label_path = os.path.join(audiomos_dir, label_csv)
    wav_dir = os.path.join(audiomos_dir, wav_subdir)
    df = pd.read_csv(label_path)
    rows = []
    for _, row in df.iterrows():
        wav_path = os.path.join(wav_dir, row["filename"])
        if not os.path.exists(wav_path):
            continue
        rows.append({
            "path": os.path.abspath(wav_path),
            "acr": float(row["mos"]),
            "ccr": float("nan"),
            "language": "en",
            "system": "unknown",
            "split": "train",
        })
    return rows


def split_rows(rows: list, train_ratio: float = 0.9, seed: int = 42):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    for r in shuffled[:n_train]:
        r["split"] = "train"
    for r in shuffled[n_train:]:
        r["split"] = "dev"
    return shuffled[:n_train], shuffled[n_train:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/datasets")
    parser.add_argument("--output_dir", default="data/manifests")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_train, all_dev = [], []

    bvcc_dir = os.path.join(args.data_dir, "bvcc")
    if os.path.exists(bvcc_dir):
        rows = parse_bvcc(bvcc_dir)
        train, dev = split_rows(rows)
        all_train.extend(train)
        all_dev.extend(dev)
        print(f"BVCC: {len(train)} train, {len(dev)} dev")

    tmhint_dir = os.path.join(args.data_dir, "tmhint")
    if os.path.exists(tmhint_dir):
        rows = parse_tmhint(tmhint_dir)
        train, dev = split_rows(rows)
        all_train.extend(train)
        all_dev.extend(dev)
        print(f"TMHINT: {len(train)} train, {len(dev)} dev")

    audiomos_dir = os.path.join(args.data_dir, "audiomos25t3")
    if os.path.exists(audiomos_dir):
        rows = parse_audiomos25t3(audiomos_dir)
        train, dev = split_rows(rows)
        all_train.extend(train)
        all_dev.extend(dev)
        print(f"AudioMOS 2025 T3: {len(train)} train, {len(dev)} dev")

    write_manifest(all_train, os.path.join(args.output_dir, "pretrain_train.csv"))
    write_manifest(all_dev, os.path.join(args.output_dir, "pretrain_dev.csv"))
    print(f"Total: {len(all_train)} train, {len(all_dev)} dev -> {args.output_dir}")


if __name__ == "__main__":
    main()
