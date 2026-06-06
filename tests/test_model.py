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


def test_forward_with_encoder_feats(tiny_model):
    # Bypass Whisper encoder using pre-extracted feats (as in cached training)
    from src.model import WHISPER_HIDDEN_SIZES
    hidden = WHISPER_HIDDEN_SIZES["openai/whisper-tiny"]
    B = 2
    encoder_feats = torch.randn(B, 1500, hidden)
    wav = torch.randn(B, 160000)
    acr, ccr = tiny_model(None, wav, encoder_feats=encoder_feats)
    assert acr.shape == (B,)
    assert ccr.shape == (B,)


def test_encoder_not_called_when_feats_provided(tiny_model):
    # When encoder_feats is given, Whisper encoder should not run
    from src.model import WHISPER_HIDDEN_SIZES
    hidden = WHISPER_HIDDEN_SIZES["openai/whisper-tiny"]
    encoder_feats = torch.randn(1, 1500, hidden)
    wav = torch.randn(1, 160000)
    # Pass None for input_features — would crash if encoder ran
    acr, ccr = tiny_model(None, wav, encoder_feats=encoder_feats)
    assert acr.shape == (1,)


def test_encoder_layer_param_changes_live_features():
    import torch
    from src.model import WhisperMOSNet
    inp, wav = make_batch(B=1)
    torch.manual_seed(0)
    m_final = WhisperMOSNet(whisper_model="openai/whisper-tiny", proj_dim=64,
                            encoder_layer=-1)
    torch.manual_seed(0)
    m_mid = WhisperMOSNet(whisper_model="openai/whisper-tiny", proj_dim=64,
                          encoder_layer=2)
    # Same weights (same seed), different layer => different ACR output
    m_mid.load_state_dict(m_final.state_dict())
    a_final, _ = m_final(inp, wav)
    a_mid, _ = m_mid(inp, wav)
    assert not torch.allclose(a_final, a_mid)


def test_mel_branch_can_be_disabled():
    from src.model import WhisperMOSNet
    m = WhisperMOSNet(whisper_model="openai/whisper-tiny", proj_dim=64,
                      use_mel_branch=False)
    assert not hasattr(m, "mel_cnn")
    inp, wav = make_batch(B=2)
    acr, ccr = m(inp, wav)
    assert acr.shape == (2,)


def test_mel_branch_enabled_by_default():
    from src.model import WhisperMOSNet
    m = WhisperMOSNet(whisper_model="openai/whisper-tiny", proj_dim=64)
    assert hasattr(m, "mel_cnn")
