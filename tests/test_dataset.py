import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.dataset import MOSDataset

WHISPER_MEL_FRAMES = 3000
TARGET_SAMPLES = 160000


def make_manifest(tmpdir, n=4, include_ccr=False):
    wav_dir = os.path.join(tmpdir, "wav")
    os.makedirs(wav_dir)
    rows = []
    for i in range(n):
        path = os.path.join(wav_dir, f"utt{i:03d}.wav")
        sf.write(path, np.random.randn(48000).astype(np.float32), 16000)
        rows.append({
            "path": path, "acr": float(i + 1),
            "ccr": float(i * 0.5) if include_ccr else float("nan"),
            "language": "en", "system": "s1", "split": "train",
        })
    csv_path = os.path.join(tmpdir, "manifest.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_dataset_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = MOSDataset(make_manifest(tmpdir, n=4), whisper_model="openai/whisper-tiny")
        assert len(ds) == 4


def test_item_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = MOSDataset(make_manifest(tmpdir, n=2), whisper_model="openai/whisper-tiny")
        item = ds[0]
        assert item["input_features"].shape == (80, WHISPER_MEL_FRAMES)
        assert item["waveform"].shape == (TARGET_SAMPLES,)
        assert item["input_features"].dtype == torch.float32
        assert item["waveform"].dtype == torch.float32


def test_acr_value():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = MOSDataset(make_manifest(tmpdir, n=3), whisper_model="openai/whisper-tiny")
        assert ds[0]["acr"].item() == pytest.approx(1.0)
        assert ds[2]["acr"].item() == pytest.approx(3.0)


def test_ccr_nan_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = MOSDataset(make_manifest(tmpdir, n=2, include_ccr=False), whisper_model="openai/whisper-tiny")
        assert torch.isnan(ds[0]["ccr"])


def test_ccr_value_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = MOSDataset(make_manifest(tmpdir, n=3, include_ccr=True), whisper_model="openai/whisper-tiny")
        assert not torch.isnan(ds[1]["ccr"])
        assert ds[2]["ccr"].item() == pytest.approx(1.0)
