import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import WhisperModel

WHISPER_HIDDEN_SIZES = {
    "openai/whisper-tiny": 384,
    "openai/whisper-base": 512,
    "openai/whisper-small": 768,
    "openai/whisper-medium": 1024,
    "openai/whisper-large-v3": 1280,
}

WHISPER_SEQ_LEN = 1500  # encoder output frames for 30s padded audio


class WhisperMOSNet(nn.Module):
    """Two-branch MOS predictor: frozen Whisper encoder + mel-spectrogram CNN.

    Args:
        whisper_model: HuggingFace model name for the Whisper backbone
        proj_dim:      projection dimension shared by both branches
    """

    def __init__(self, whisper_model: str = "openai/whisper-medium", proj_dim: int = 256,
                 dropout: float = 0.1, encoder_layer: int = -1,
                 use_mel_branch: bool = True):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.use_mel_branch = use_mel_branch

        whisper = WhisperModel.from_pretrained(whisper_model)
        self.whisper_encoder = whisper.encoder
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        whisper_hidden = WHISPER_HIDDEN_SIZES.get(whisper_model, 1024)

        # Adapter: bridges Whisper hidden dim to proj_dim
        self.adapter = nn.Sequential(
            nn.Linear(whisper_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        if self.use_mel_branch:
            # Mel spectrogram transform (matches Whisper's internal preprocessing)
            self.mel_transform = T.MelSpectrogram(
                sample_rate=16000, n_fft=400, hop_length=160, n_mels=80,
            )
            self.amplitude_to_db = T.AmplitudeToDB()

            # CNN branch: (B, 1, 80, T_mel) -> (B, proj_dim)
            self.mel_cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.mel_proj = nn.Linear(128, proj_dim)

        # BiLSTM over fused (Whisper + mel) features
        bilstm_input = proj_dim * 2 if self.use_mel_branch else proj_dim
        self.bilstm = nn.LSTM(
            input_size=bilstm_input,
            hidden_size=proj_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(proj_dim * 2, 1)
        self.acr_head = nn.Linear(proj_dim * 2, 1)
        self.ccr_head = nn.Linear(proj_dim * 2, 1)

    def forward(self, input_features: torch.Tensor, waveforms: torch.Tensor,
                encoder_feats: torch.Tensor = None):
        """
        Args:
            input_features: (B, 80, 3000) Whisper log-mel features  -- used when no cache
            waveforms:      (B, 160000) raw audio at 16kHz
            encoder_feats:  (B, 1500, hidden) pre-extracted encoder outputs  -- skips encoder

        Returns:
            acr: (B,) predicted ACR scores
            ccr: (B,) predicted CCR scores
        """
        B = waveforms.size(0)
        device = waveforms.device

        if encoder_feats is not None:
            whisper_feats = self.adapter(encoder_feats)            # (B, 1500, proj_dim)
        else:
            with torch.no_grad():
                if self.encoder_layer == -1:
                    encoder_out = self.whisper_encoder(input_features).last_hidden_state
                else:
                    hs = self.whisper_encoder(
                        input_features, output_hidden_states=True
                    ).hidden_states
                    encoder_out = hs[self.encoder_layer]
            whisper_feats = self.adapter(encoder_out)              # (B, 1500, proj_dim)

        # Mel branch and BiLSTM run in float32 -- AmplitudeToDB uses log10 which
        # underflows to -inf for near-zero values in float16, producing NaN.
        with torch.amp.autocast('cuda', enabled=False):
            whisper_feats_f32 = whisper_feats.float()
            if self.use_mel_branch:
                mel = self.mel_transform(waveforms.float())
                mel_db = self.amplitude_to_db(mel)
                mel_out = self.mel_cnn(mel_db.unsqueeze(1))
                mel_feats = self.mel_proj(mel_out.view(B, -1))         # (B, proj_dim)
                mel_feats = mel_feats.unsqueeze(1).expand(-1, WHISPER_SEQ_LEN, -1)
                fused = torch.cat([whisper_feats_f32, mel_feats], dim=-1)  # (B, 1500, 2*proj_dim)
            else:
                fused = whisper_feats_f32                                  # (B, 1500, proj_dim)
            lstm_out, _ = self.bilstm(fused)
        attn = torch.softmax(self.attention(lstm_out), dim=1)   # (B, 1500, 1)
        pooled = self.dropout((lstm_out * attn).sum(dim=1))     # (B, 2*proj_dim)

        return self.acr_head(pooled).squeeze(-1), self.ccr_head(pooled).squeeze(-1)
