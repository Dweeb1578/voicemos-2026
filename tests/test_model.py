import torch
import pytest

from src.model import WhisperMOSNet


@pytest.fixture
def tiny_model():
    return WhisperMOSNet(whisper_model="openai/whisper-tiny", proj_dim=64)


def make_batch(B=2):
    return (
        torch.randn(B, 80, 3000),
        torch.randn(B, 160000),
    )


def test_forward_output_shapes(tiny_model):
    inp, wav = make_batch(B=2)
    acr, ccr = tiny_model(inp, wav)
    assert acr.shape == (2,)
    assert ccr.shape == (2,)


def test_forward_output_dtype(tiny_model):
    inp, wav = make_batch(B=2)
    acr, ccr = tiny_model(inp, wav)
    assert acr.dtype == torch.float32
    assert ccr.dtype == torch.float32


def test_whisper_encoder_frozen(tiny_model):
    for p in tiny_model.whisper_encoder.parameters():
        assert not p.requires_grad


def test_adapter_is_trainable(tiny_model):
    params = list(tiny_model.adapter.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_batch_size_1(tiny_model):
    inp, wav = make_batch(B=1)
    acr, ccr = tiny_model(inp, wav)
    assert acr.shape == (1,)
    assert ccr.shape == (1,)
