"""Pre-extract frozen Whisper encoder outputs to disk (one-time cost).

Saves (1500, hidden_size) float16 arrays keyed by MD5 of audio path.
~3 MB/sample for whisper-medium -> ~47 GB for 15.5K samples.

Audio decode + log-mel extraction (the CPU cost) run in DataLoader worker
processes so they overlap the GPU encoder forward; the loop is otherwise
CPU-bound and starves the GPU.

Usage:
    python -m data.cache_features \
        --manifests data/manifests/pretrain_train.csv data/manifests/pretrain_dev.csv \
        --cache_dir data/encoder_cache \
        --whisper_model openai/whisper-medium --num_workers 4

    # Intermediate layer (e.g. layer 12 of 24 for whisper-medium)
    python -m data.cache_features --manifests ... --cache_dir data/encoder_cache_layer12 \
        --whisper_model openai/whisper-medium --layer 12 --num_workers 4
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

from data.preprocess import resample_and_normalize, trim_and_pad


def cache_key(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode()).hexdigest() + ".npy"


class _CacheClips(Dataset):
    """Label-free dataset: decode audio + log-mel in worker procs, yield (feats, path)."""

    def __init__(self, paths, whisper_model):
        self.paths = list(paths)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        audio, _ = resample_and_normalize(p)
        audio = trim_and_pad(audio)
        feats = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        return {"input_features": feats.input_features.squeeze(0), "path": p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--whisper_model", default="openai/whisper-medium")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Encoder layer to extract (-1 = final layer, 0 = embedding output, "
                             "1..N = after transformer layer N). Whisper-medium has 24 layers.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  model: {args.whisper_model}")

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

    # Filter already-cached (resume-by-skip)
    todo = [p for p in all_paths if not os.path.exists(os.path.join(args.cache_dir, cache_key(p)))]
    print(f"Total unique paths: {len(all_paths)}  to cache: {len(todo)}")
    if not todo:
        print("Nothing to cache.")
        return

    use_intermediate = args.layer != -1
    print(f"Extracting {'intermediate layer ' + str(args.layer) if use_intermediate else 'final layer'}")

    ds = _CacheClips(todo, args.whisper_model)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(loader, desc="Extracting"):
            inp = batch["input_features"].to(device)
            if use_intermediate:
                out = encoder(inp, output_hidden_states=True).hidden_states[args.layer]
            else:
                out = encoder(inp).last_hidden_state
            out_np = out.half().cpu().numpy()
            for path, feat in zip(batch["path"], out_np):
                np.save(os.path.join(args.cache_dir, cache_key(path)), feat)

    print(f"Done. Cache written to {args.cache_dir}")


if __name__ == "__main__":
    main()
