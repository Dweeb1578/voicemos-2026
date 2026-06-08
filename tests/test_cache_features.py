import numpy as np
import soundfile as sf
import torch

from data.cache_features import _CacheClips, cache_key


def test_cache_clips_returns_features_and_path(tmp_path):
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.random.randn(32000).astype("float32"), 16000)
    ds = _CacheClips([str(wav)], whisper_model="openai/whisper-tiny")
    item = ds[0]
    assert item["input_features"].shape == (80, 3000)
    assert item["input_features"].dtype == torch.float32
    assert item["path"] == str(wav)


def test_cache_key_deterministic_and_npy():
    assert cache_key("/a/b.wav") == cache_key("/a/b.wav")
    assert cache_key("/a/b.wav").endswith(".npy")
    assert cache_key("/a/b.wav") != cache_key("/a/c.wav")
