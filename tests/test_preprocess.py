import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from data.preprocess import resample_and_normalize, trim_and_pad, write_manifest


def make_sine_wav(path, sr=44100, duration=5.0):
    t = np.linspace(0, duration, int(sr * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(path, audio, sr)


def test_resample_changes_sr():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.wav")
        make_sine_wav(src, sr=44100, duration=2.0)
        audio, sr = resample_and_normalize(src, target_sr=16000)
        assert sr == 16000
        assert len(audio) == 32000


def test_trim_and_pad_clips_long_audio():
    long_audio = np.random.randn(16000 * 15).astype(np.float32)
    result = trim_and_pad(long_audio, sr=16000, max_duration=10.0)
    assert len(result) == 160000


def test_trim_and_pad_zero_pads_short_audio():
    short_audio = np.random.randn(16000 * 3).astype(np.float32)
    result = trim_and_pad(short_audio, sr=16000, max_duration=10.0)
    assert len(result) == 160000
    assert np.all(result[16000 * 3 :] == 0.0)


def test_write_manifest_creates_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = [
            {"path": "/a.wav", "acr": 3.5, "ccr": float("nan"), "language": "en", "system": "s1", "split": "train"},
            {"path": "/b.wav", "acr": 4.0, "ccr": 0.3, "language": "zh", "system": "s2", "split": "dev"},
        ]
        out = os.path.join(tmpdir, "manifest.csv")
        write_manifest(rows, out)
        df = pd.read_csv(out)
        assert len(df) == 2
        assert set(df.columns) >= {"path", "acr", "ccr", "language", "system", "split"}
        assert pd.isna(df.iloc[0]["ccr"])
        assert df.iloc[1]["ccr"] == pytest.approx(0.3)


def test_write_manifest_preserves_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = [
            {"path": "/a.wav", "acr": 3.5, "ccr": float("nan"), "language": "en",
             "system": "s1", "split": "train", "source": "bvcc"},
            {"path": "/b.wav", "acr": 4.0, "ccr": 0.3, "language": "zh",
             "system": "s2", "split": "dev", "source": "tmhint"},
        ]
        out = os.path.join(tmpdir, "manifest.csv")
        write_manifest(rows, out)
        df = pd.read_csv(out)
        assert "source" in df.columns
        assert df.iloc[0]["source"] == "bvcc"
        assert df.iloc[1]["source"] == "tmhint"


def test_write_manifest_source_defaults_to_unknown_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = [
            {"path": "/a.wav", "acr": 3.5, "ccr": float("nan"), "language": "en",
             "system": "s1", "split": "train"},  # no source key
        ]
        out = os.path.join(tmpdir, "manifest.csv")
        write_manifest(rows, out)
        df = pd.read_csv(out)
        assert df.iloc[0]["source"] == "unknown"
