import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from data.build_manifests import (
    parse_tmhint, parse_audiomos25t3, parse_nisqa, normalize_per_source, cap_rows,
)


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
            "file_name": ["sys1_utt001", "sys1_utt002"],  # no .wav — function appends it
            "quality_score": [3.5, 4.0],
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


def _make_nisqa(tmpdir):
    """Build a fake NISQA corpus: NISQA_Corpus/NISQA_corpus_file.csv + deg wavs."""
    corpus = os.path.join(tmpdir, "nisqa", "NISQA_Corpus")
    deg = os.path.join(corpus, "deg")
    os.makedirs(deg)
    for name in ["a.wav", "b.wav", "c.wav"]:
        make_wav(os.path.join(deg, name))
    pd.DataFrame({
        "db": ["NISQA_TRAIN_SIM", "NISQA_VAL_LIVE", "NISQA_TEST_P501"],
        "filepath_deg": ["deg/a.wav", "deg/b.wav", "deg/c.wav"],
        "mos": [3.1, 4.2, 2.0],
    }).to_csv(os.path.join(corpus, "NISQA_corpus_file.csv"), index=False)
    return os.path.join(tmpdir, "nisqa")


def test_parse_nisqa_tags_source_and_excludes_test_split():
    with tempfile.TemporaryDirectory() as tmpdir:
        rows = parse_nisqa(_make_nisqa(tmpdir))
        # TEST split (c.wav) excluded; only TRAIN + VAL kept
        assert len(rows) == 2
        assert {r["source"] for r in rows} == {"nisqa"}
        assert all(os.path.exists(r["path"]) for r in rows)
        mos_by = {os.path.basename(r["path"]): r["acr"] for r in rows}
        assert mos_by["a.wav"] == pytest.approx(3.1)
        assert mos_by["b.wav"] == pytest.approx(4.2)
        assert all(pd.isna(r["ccr"]) for r in rows)


def test_parse_nisqa_missing_dir_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert parse_nisqa(os.path.join(tmpdir, "nope")) == []


def test_normalize_per_source_standardizes_each_source():
    train = [
        {"source": "nisqa", "acr": 1.0}, {"source": "nisqa", "acr": 3.0}, {"source": "nisqa", "acr": 5.0},
        {"source": "tmhint", "acr": 2.0}, {"source": "tmhint", "acr": 4.0},
    ]
    dev = [{"source": "nisqa", "acr": 3.0}]
    normalize_per_source(train, dev)
    nisqa = [r["acr"] for r in train if r["source"] == "nisqa"]
    assert np.mean(nisqa) == pytest.approx(0.0, abs=1e-9)
    assert np.std(nisqa) == pytest.approx(1.0, abs=1e-6)
    # dev uses TRAIN stats: nisqa train mean 3.0 -> dev acr 3.0 maps to 0.0
    assert dev[0]["acr"] == pytest.approx(0.0, abs=1e-9)


def test_normalize_per_source_constant_source_no_div0():
    train = [{"source": "x", "acr": 2.0}, {"source": "x", "acr": 2.0}]
    normalize_per_source(train, [])
    assert all(np.isfinite(r["acr"]) for r in train)  # std==0 guard -> no NaN/inf


def test_cap_rows_caps_and_is_deterministic():
    rows = [{"i": i} for i in range(100)]
    out = cap_rows(rows, 10)
    assert len(out) == 10
    assert cap_rows(rows, 10) == out  # seeded -> same subsample every call


def test_cap_rows_noop_when_unset_or_small():
    rows = [{"i": i} for i in range(5)]
    assert cap_rows(rows, 0) is rows      # 0 = no cap, returns original
    assert cap_rows(rows, None) is rows   # None = no cap
    assert len(cap_rows(rows, 50)) == 5   # cap larger than len -> unchanged
