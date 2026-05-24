import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from data.preprocess import resample_and_normalize, trim_and_pad

TARGET_SAMPLES = 160000  # 10s at 16kHz


def _cache_key(audio_path: str) -> str:
    return hashlib.md5(audio_path.encode()).hexdigest() + ".npy"


class MOSDataset(Dataset):
    """Unified dataset for all MOS prediction sources.

    If cache_dir is given and contains pre-extracted encoder outputs
    (from data/cache_features.py), returns encoder_feats (T, hidden) instead
    of input_features (80, 3000) -- skips Whisper encoder in training.

    Each item always contains:
        waveform:  (160000,) raw audio at 16kHz for the mel CNN branch
        acr:       scalar float32 tensor
        ccr:       scalar float32 tensor, NaN if not available

    Plus one of:
        encoder_feats: (1500, hidden_size) float16  -- if cache hit
        input_features: (80, 3000) float32          -- otherwise
    """

    def __init__(self, manifest_csv: str, whisper_model: str = "openai/whisper-medium",
                 cache_dir: str = None):
        self.df = pd.read_csv(manifest_csv)
        self.cache_dir = cache_dir
        if cache_dir is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        else:
            self.feature_extractor = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio, _ = resample_and_normalize(row["path"])
        audio = trim_and_pad(audio)
        waveform = torch.tensor(audio, dtype=torch.float32)

        if self.cache_dir is not None:
            enc = np.load(os.path.join(self.cache_dir, _cache_key(row["path"])))
            enc_tensor = torch.from_numpy(enc).float()  # (T, hidden)
            feat_dict = {"encoder_feats": enc_tensor}
        else:
            features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            feat_dict = {"input_features": features.input_features.squeeze(0)}

        acr = torch.tensor(float(row["acr"]), dtype=torch.float32)
        ccr_val = row["ccr"]
        ccr = torch.tensor(
            float("nan") if pd.isna(ccr_val) else float(ccr_val),
            dtype=torch.float32,
        )

        return {**feat_dict, "waveform": waveform, "acr": acr, "ccr": ccr}
