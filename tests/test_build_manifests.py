import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from data.build_manifests import parse_tmhint, parse_audiomos25t3


def make_wav(path, sr=16000, duration=1.0):
    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    sf.write(path, audio, sr)


def test_parse_tmhint_returns_required_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_dir = os.path.join(tmpdir, "wav")
        os.makedirs(wav_dir)
        make_wav(os.path.join(wav_dir, "sys1_utt001.wav"))
        make_wav(os.path.join(wav_dir, "sys1_utt002.wav"))
        pd.DataFrame({
            "filename": ["sys1_utt001.wav", "sys1_utt002.wav"],
            "quality": [3.5, 4.0],
            "intelligibility": [0.8, 0.9],
        }).to_csv(os.path.join(tmpdir, "scores.csv"), index=False)

        rows = parse_tmhint(tmpdir, label_csv="scores.csv", wav_subdir="wav")
        assert len(rows) == 2
        assert all(k in rows[0] for k in ["path", "acr", "ccr", "language", "system", "split"])
        assert rows[0]["acr"] == pytest.approx(3.5)
        assert pd.isna(rows[0]["ccr"])


def test_parse_audiomos25t3_returns_required_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_dir = os.path.join(tmpdir, "wav")
        os.makedirs(wav_dir)
        make_wav(os.path.join(wav_dir, "utt001.wav"))
        pd.DataFrame({
            "filename": ["utt001.wav"],
            "mos": [3.2],
            "sr": [16000],
        }).to_csv(os.path.join(tmpdir, "labels.csv"), index=False)

        rows = parse_audiomos25t3(tmpdir, label_csv="labels.csv", wav_subdir="wav")
        assert len(rows) == 1
        assert rows[0]["acr"] == pytest.approx(3.2)
        assert pd.isna(rows[0]["ccr"])
