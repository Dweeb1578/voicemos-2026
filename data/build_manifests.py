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

    Actual structure after extraction:
        bvcc_dir/main/DATA/sets/{TRAINSET,DEVSET}  -- no header, cols: system_id,filename,score,...
        bvcc_dir/main/DATA/wav/                    -- all wav files flat
    """
    data_dir = os.path.join(bvcc_dir, "main", "DATA")
    wav_dir = os.path.join(data_dir, "wav")
    if not os.path.exists(wav_dir):
        return []

    rows = []
    for split_name, split_label in (("TRAINSET", "train"), ("DEVSET", "dev")):
        split_path = os.path.join(data_dir, "sets", split_name)
        if not os.path.exists(split_path):
            continue
        df = pd.read_csv(split_path, header=None,
                         names=["system_id", "filename", "score", "session_id", "listener_info"])
        avg = df.groupby("filename")["score"].mean().reset_index()
        for _, row in avg.iterrows():
            wav_path = os.path.join(wav_dir, row["filename"])
            if not os.path.exists(wav_path):
                continue
            rows.append({
                "path": os.path.abspath(wav_path),
                "acr": float(row["score"]),
                "ccr": float("nan"),
                "language": "en",
                "system": str(row["filename"]).split("-")[0],
                "split": split_label,
                "source": "bvcc",
            })
    return rows


def parse_tmhint(tmhint_dir: str, label_csv: str = "TMHINTQI/raw_data.csv", wav_subdir: str = "TMHINTQI/train") -> list:
    """Parse TMHINT-QI dataset (VoiceMOS 2023 Track 3).

    Uses raw_data.csv (columns: file_name, quality_score per listener).
    Averages quality_score by file_name to get per-utterance MOS.
    Audio lives in TMHINTQI/train/.
    """
    label_path = os.path.join(tmhint_dir, label_csv)
    wav_dir = os.path.join(tmhint_dir, wav_subdir)
    df = pd.read_csv(label_path)

    # Average per-listener ratings to utterance-level MOS
    avg = df.groupby("file_name")["quality_score"].mean().reset_index()
    avg.columns = ["file_name", "avg_mos"]

    rows = []
    for _, row in avg.iterrows():
        wav_path = os.path.join(wav_dir, row["file_name"] + ".wav")
        if not os.path.exists(wav_path):
            continue
        rows.append({
            "path": os.path.abspath(wav_path),
            "acr": float(row["avg_mos"]),
            "ccr": float("nan"),
            "language": "zh",
            "system": str(row["file_name"]).split("_")[0],
            "split": "train",
            "source": "tmhint",
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
            "source": "audiomos",
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
    if os.path.exists(os.path.join(audiomos_dir, "labels.csv")):
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
