"""Pre-extract frozen Whisper encoder outputs to disk (one-time cost).

Saves (1500, hidden_size) float16 arrays keyed by MD5 of audio path.
~3 MB/sample for whisper-medium → ~47 GB for 15.5K samples.

Usage:
    python -m data.cache_features \
        --manifests data/manifests/pretrain_train.csv data/manifests/pretrain_dev.csv \
        --cache_dir data/encoder_cache \
        --whisper_model openai/whisper-medium \
        --batch_size 32
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

from data.preprocess import resample_and_normalize, trim_and_pad


def cache_key(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode()).hexdigest() + ".npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--whisper_model", default="openai/whisper-medium")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  model: {args.whisper_model}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_model)
    whisper = WhisperModel.from_pretrained(args.whisper_model)
    encoder = whisper.encoder.to(device)
    encoder.train(False)

    # Deduplicate paths across manifests
    all_paths = []
    seen = set()
    for m in args.manifests:
        for path in pd.read_csv(m)["path"]:
            if path not in seen:
                seen.add(path)
                all_paths.append(path)

    # Filter already-cached
    todo = [p for p in all_paths if not os.path.exists(os.path.join(args.cache_dir, cache_key(p)))]
    print(f"Total unique paths: {len(all_paths)}  to cache: {len(todo)}")

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="Extracting"):
            batch_paths = todo[i:i + args.batch_size]
            wavs = []
            for p in batch_paths:
                audio, _ = resample_and_normalize(p)
                wavs.append(trim_and_pad(audio))

            feats = feature_extractor(wavs, sampling_rate=16000, return_tensors="pt")
            inp = feats.input_features.to(device)
            out = encoder(inp).last_hidden_state  # (B, T, hidden)
            out_np = out.half().cpu().numpy()

            for j, p in enumerate(batch_paths):
                np.save(os.path.join(args.cache_dir, cache_key(p)), out_np[j])

    print(f"Done. Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
