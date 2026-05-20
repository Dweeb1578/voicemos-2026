import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from data.preprocess import resample_and_normalize, trim_and_pad

TARGET_SAMPLES = 160000  # 10s at 16kHz


class MOSDataset(Dataset):
    """Unified dataset for all MOS prediction sources.

    Each item:
        input_features: (80, 3000) Whisper log-mel features (padded to 30s)
        waveform:       (160000,)  raw audio at 16kHz for the mel CNN branch
        acr:            scalar float32 tensor
        ccr:            scalar float32 tensor, NaN if not available
    """

    def __init__(self, manifest_csv: str, whisper_model: str = "openai/whisper-medium"):
        self.df = pd.read_csv(manifest_csv)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        audio, _ = resample_and_normalize(row["path"])
        audio = trim_and_pad(audio)  # (160000,)

        # Whisper log-mel: (80, 3000) padded to 30s internally
        features = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = features.input_features.squeeze(0)  # (80, 3000)

        waveform = torch.tensor(audio, dtype=torch.float32)

        acr = torch.tensor(float(row["acr"]), dtype=torch.float32)
        ccr_val = row["ccr"]
        ccr = torch.tensor(
            float("nan") if pd.isna(ccr_val) else float(ccr_val),
            dtype=torch.float32,
        )

        return {"input_features": input_features, "waveform": waveform, "acr": acr, "ccr": ccr}
