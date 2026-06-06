# Track 1 Approach A — Model/Loss/Feature Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise Track-1 ACR UTT-SRCC (0.574 → toward 0.662) by adding a rank-aware ACR loss, enabling encoder-layer selection, and making the mel-CNN branch ablatable — all on the frozen-feature path that fits free Colab T4.

**Architecture:** Keep `WhisperMOSNet` (frozen Whisper encoder, cached features, lightweight trainable head). Three independently-measurable changes: (A1) `MOSLoss` gains a within-source pairwise-ranking term on ACR; (A2) the model can read any encoder layer (not just the final one); (A3) the mel branch becomes a config toggle. Each is gated by a dev-SRCC measurement before the next.

**Tech Stack:** PyTorch, HuggingFace `transformers` (WhisperModel), pandas, scipy.stats, pytest. Training/inference on Colab; unit tests run locally (CPU, whisper-tiny).

**Spec:** `docs/superpowers/specs/2026-06-07-voicemos-track1-improvements-design.md`

**Methodology reminder (applies to every experiment task):** change ONE variable, regenerate the submission, upload to CodaBench, record dev ACR UTT-SRCC, AND record the offline `pretrain_dev` SRCC printed by `train.py`. Keep a change only if dev SRCC rises without the offline SRCC collapsing (guards against leaderboard overfitting under unlimited submissions).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/losses.py` | Loss math | Add `acr_rank_alpha` term + within-source masking |
| `src/train.py` | Training loop | Build per-batch source ids; pass to loss; read new config keys |
| `src/model.py` | Model | `encoder_layer` (live-path layer pick) + `use_mel_branch` toggle |
| `src/dataset.py` | Data loading | Return `source` string per item |
| `data/build_manifests.py` | Manifest building | Add `source` column to every parser |
| `scripts/predict_dev.py` | Submission generation | Build model with `encoder_layer` + `use_mel_branch` from config |
| `configs/*.yaml` | Experiment configs | New keys: `acr_rank_alpha`, `encoder_layer`, `use_mel_branch` |
| `tests/test_losses.py` | Loss tests | Rank-term + masking coverage |
| `tests/test_model.py` | Model tests | Layer-pick + mel-toggle coverage |
| `tests/test_dataset.py` | Dataset tests | `source` field coverage |
| `tests/test_train_utils.py` | New | `build_source_ids` coverage |

---

## Task 1: Source field in dataset + manifests

**Why:** The rank loss (Task 2) must only compare utterances from the same dataset — cross-dataset MOS values aren't comparable. The manifest needs a `source` column and the dataset must surface it.

**Files:**
- Modify: `data/build_manifests.py` (each parser)
- Modify: `src/dataset.py:48-69` (`__getitem__`)
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dataset.py`:

```python
import pandas as pd
from src.dataset import MOSDataset


def test_dataset_returns_source(tmp_path):
    # Manifest with a source column
    wav = tmp_path / "a.wav"
    import soundfile as sf, numpy as np
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)
    df = pd.DataFrame([{"path": str(wav), "acr": 3.0, "ccr": float("nan"),
                        "language": "en", "system": "x", "split": "train",
                        "source": "bvcc"}])
    man = tmp_path / "m.csv"
    df.to_csv(man, index=False)
    ds = MOSDataset(str(man), whisper_model="openai/whisper-tiny")
    assert ds[0]["source"] == "bvcc"


def test_dataset_source_defaults_when_missing(tmp_path):
    wav = tmp_path / "a.wav"
    import soundfile as sf, numpy as np
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)
    df = pd.DataFrame([{"path": str(wav), "acr": 3.0, "ccr": float("nan"),
                        "language": "en", "system": "x", "split": "train"}])
    man = tmp_path / "m.csv"
    df.to_csv(man, index=False)
    ds = MOSDataset(str(man), whisper_model="openai/whisper-tiny")
    assert ds[0]["source"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py::test_dataset_returns_source -v`
Expected: FAIL with `KeyError: 'source'`.

- [ ] **Step 3: Implement — dataset returns source**

In `src/dataset.py`, change the return in `__getitem__` (currently line 69) to include `source`. Replace:

```python
        return {**feat_dict, "waveform": waveform, "acr": acr, "ccr": ccr}
```

with:

```python
        source = row["source"] if "source" in self.df.columns else "unknown"
        return {**feat_dict, "waveform": waveform, "acr": acr, "ccr": ccr,
                "source": source}
```

- [ ] **Step 4: Add `source` to manifest parsers**

In `data/build_manifests.py`, add `"source": "bvcc"` to the dict appended in `parse_bvcc` (the dict at line 40-47), `"source": "tmhint"` in `parse_tmhint` (line 71-78), and `"source": "audiomos"` in `parse_audiomos25t3` (line 94-101). Example for `parse_bvcc`:

```python
            rows.append({
                "path": os.path.abspath(wav_path),
                "acr": float(row["score"]),
                "ccr": float("nan"),
                "language": "en",
                "system": str(row["filename"]).split("-")[0],
                "split": split_label,
                "source": "bvcc",
            })
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (both new tests, plus existing ones).

- [ ] **Step 6: Commit**

```bash
git add src/dataset.py data/build_manifests.py tests/test_dataset.py
git commit -m "feat: add source field to dataset items and manifests"
```

---

## Task 2: Rank-aware ACR loss with within-source masking

**Why:** SRCC is rank-based, and ranks transfer across domains better than absolute MOS. Add a pairwise-ranking term on ACR, masked to same-source pairs. Defaults to off (`acr_rank_alpha=0.0`) so existing behavior/tests are unchanged.

**Files:**
- Modify: `src/losses.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_losses.py`:

```python
def test_acr_rank_zero_when_correctly_ordered():
    loss_fn = MOSLoss(ccr_lambda=0.0, acr_rank_alpha=1.0)
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])  # correct order + zero MSE
    src = torch.tensor([0, 0, 0])
    loss = loss_fn(pred, torch.zeros(3), target, torch.full((3,), float("nan")),
                   source_ids=src)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_acr_rank_penalizes_wrong_order():
    loss_fn = MOSLoss(ccr_lambda=0.0, acr_rank_alpha=1.0)
    pred = torch.tensor([2.0, 1.0])     # reversed vs target
    target = torch.tensor([1.0, 2.0])
    same = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")),
                   source_ids=torch.tensor([0, 0]))
    diff = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")),
                   source_ids=torch.tensor([0, 1]))
    # same-source: MSE(=1) + rank(=1); diff-source: only MSE(=1), pair masked out
    assert same.item() == pytest.approx(2.0, abs=1e-4)
    assert diff.item() == pytest.approx(1.0, abs=1e-4)


def test_acr_rank_alpha_zero_is_pure_mse():
    loss_fn = MOSLoss(ccr_lambda=0.0, acr_rank_alpha=0.0)
    pred = torch.tensor([2.0, 1.0])
    target = torch.tensor([1.0, 2.0])
    loss = loss_fn(pred, torch.zeros(2), target, torch.full((2,), float("nan")))
    assert loss.item() == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_losses.py::test_acr_rank_penalizes_wrong_order -v`
Expected: FAIL with `TypeError` (unexpected `acr_rank_alpha` / `source_ids`).

- [ ] **Step 3: Implement the rank term**

Replace the body of `src/losses.py` with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MOSLoss(nn.Module):
    """ACR MSE (+ optional within-source rank term) + CCR pairwise ranking, NaN-masked.

    Args:
        ccr_lambda:     weight for the CCR pairwise-ranking term.
        acr_rank_alpha: weight for the ACR pairwise-ranking term (0 = pure MSE).
    """

    def __init__(self, ccr_lambda: float = 0.0, acr_rank_alpha: float = 0.0):
        super().__init__()
        self.ccr_lambda = ccr_lambda
        self.acr_rank_alpha = acr_rank_alpha

    @staticmethod
    def _pairwise_rank(pred, target, source_ids=None):
        """Mean margin-ranking loss over all pairs; masked to same-source pairs."""
        n = pred.size(0)
        if n < 2:
            return pred.sum() * 0.0
        i, j = torch.triu_indices(n, n, offset=1)
        keep = torch.sign(target[i] - target[j]) != 0
        if source_ids is not None:
            keep = keep & (source_ids[i] == source_ids[j])
        if not keep.any():
            return pred.sum() * 0.0
        target_sign = torch.sign(target[i] - target[j])[keep]
        return F.margin_ranking_loss(pred[i][keep], pred[j][keep], target_sign, margin=0.0)

    def forward(self, acr_pred, ccr_pred, acr_target, ccr_target, source_ids=None):
        acr_mask = ~torch.isnan(acr_target)
        if acr_mask.any():
            acr_loss = F.mse_loss(acr_pred[acr_mask], acr_target[acr_mask])
            if self.acr_rank_alpha > 0.0:
                src = source_ids[acr_mask] if source_ids is not None else None
                acr_loss = acr_loss + self.acr_rank_alpha * self._pairwise_rank(
                    acr_pred[acr_mask], acr_target[acr_mask], src
                )
        else:
            acr_loss = acr_pred.sum() * 0.0

        if self.ccr_lambda == 0.0:
            return acr_loss

        ccr_mask = ~torch.isnan(ccr_target)
        cp, ct = ccr_pred[ccr_mask], ccr_target[ccr_mask]
        ccr_loss = self._pairwise_rank(cp, ct)
        return acr_loss + self.ccr_lambda * ccr_loss
```

Note: this refactors the existing CCR ranking into the shared `_pairwise_rank` helper — behavior is identical for CCR (no source mask passed), so existing CCR tests still pass.

- [ ] **Step 4: Run the full loss suite**

Run: `pytest tests/test_losses.py -v`
Expected: PASS (all old + 3 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: add within-source rank-aware ACR loss term"
```

---

## Task 3: Wire source ids + rank alpha into the training loop

**Files:**
- Create: `src/train_utils.py`
- Modify: `src/train.py:34-63` (`run_epoch`), `src/train.py:126` (loss construction)
- Test: `tests/test_train_utils.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_utils.py`:

```python
import torch
from src.train_utils import build_source_ids


def test_build_source_ids_groups_same_strings():
    ids = build_source_ids(["bvcc", "tmhint", "bvcc"])
    assert ids[0] == ids[2]
    assert ids[0] != ids[1]
    assert ids.dtype == torch.long


def test_build_source_ids_length():
    ids = build_source_ids(["a", "b", "c", "a"])
    assert ids.shape == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train_utils'`.

- [ ] **Step 3: Implement the helper**

Create `src/train_utils.py`:

```python
import torch


def build_source_ids(sources):
    """Map a list of source strings to batch-local integer ids (for rank masking)."""
    mapping = {s: i for i, s in enumerate(sorted(set(sources)))}
    return torch.tensor([mapping[s] for s in sources], dtype=torch.long)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_utils.py -v`
Expected: PASS.

- [ ] **Step 5: Use it in `run_epoch`**

In `src/train.py`, add to the imports near line 15:

```python
from src.train_utils import build_source_ids
```

Inside `run_epoch`, in the batch loop (after `enc = ...` at line 45), build source ids and pass them to the loss. Change:

```python
        with torch.cuda.amp.autocast():
            acr_p, ccr_p = model(inp, wav, encoder_feats=enc)
            loss = loss_fn(acr_p, ccr_p, acr_t, ccr_t)
```

to:

```python
        src_ids = None
        if "source" in batch:
            src_ids = build_source_ids(batch["source"]).to(device)

        with torch.cuda.amp.autocast():
            acr_p, ccr_p = model(inp, wav, encoder_feats=enc)
            loss = loss_fn(acr_p, ccr_p, acr_t, ccr_t, source_ids=src_ids)
```

- [ ] **Step 6: Read `acr_rank_alpha` from config**

In `src/train.py`, change the loss construction (line 126) from:

```python
        loss_fn = MOSLoss(ccr_lambda=get_ccr_lambda(epoch, cfg))
```

to:

```python
        loss_fn = MOSLoss(
            ccr_lambda=get_ccr_lambda(epoch, cfg),
            acr_rank_alpha=cfg["training"].get("acr_rank_alpha", 0.0),
        )
```

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q`
Expected: PASS (no regressions).

- [ ] **Step 8: Commit**

```bash
git add src/train.py src/train_utils.py tests/test_train_utils.py
git commit -m "feat: pass source ids and acr_rank_alpha through training loop"
```

---

## Task 4: Encoder-layer selection in the model

**Why:** Intermediate Whisper layers carry more acoustic/quality information than the ASR-oriented final layer. `cache_features.py` already supports `--layer N`; the model's *live* inference path (used by `predict_dev.py`) must read the same layer. Default `-1` keeps current behavior.

**Files:**
- Modify: `src/model.py:25-34` (`__init__`), `src/model.py:91-94` (live encoder path)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_encoder_layer_param_changes_live_features -v`
Expected: FAIL with `TypeError` (unexpected `encoder_layer`).

- [ ] **Step 3: Add the `encoder_layer` parameter**

In `src/model.py`, change the `__init__` signature (line 25-26) from:

```python
    def __init__(self, whisper_model: str = "openai/whisper-medium", proj_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
```

to:

```python
    def __init__(self, whisper_model: str = "openai/whisper-medium", proj_dim: int = 256,
                 dropout: float = 0.1, encoder_layer: int = -1):
        super().__init__()
        self.encoder_layer = encoder_layer
```

- [ ] **Step 4: Use it in the live encoder path**

In `forward`, change the `else` branch (lines 91-94) from:

```python
        else:
            with torch.no_grad():
                encoder_out = self.whisper_encoder(input_features).last_hidden_state
            whisper_feats = self.adapter(encoder_out)              # (B, 1500, proj_dim)
```

to:

```python
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
```

- [ ] **Step 5: Run model tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (new test + all existing).

- [ ] **Step 6: Pass `encoder_layer` from config in train + predict_dev**

In `src/train.py` (model construction, lines 75-78) add `encoder_layer=m_cfg.get("encoder_layer", -1)`:

```python
    model = WhisperMOSNet(
        whisper_model=m_cfg["whisper_model"], proj_dim=m_cfg["proj_dim"],
        dropout=m_cfg.get("dropout", 0.0),
        encoder_layer=m_cfg.get("encoder_layer", -1),
    ).to(device)
```

In `scripts/predict_dev.py` (model construction, line 92) change:

```python
    model = WhisperMOSNet(whisper_model=whisper_model, proj_dim=m_cfg["proj_dim"]).to(device)
```

to:

```python
    model = WhisperMOSNet(
        whisper_model=whisper_model, proj_dim=m_cfg["proj_dim"],
        encoder_layer=m_cfg.get("encoder_layer", -1),
    ).to(device)
```

- [ ] **Step 7: Commit**

```bash
git add src/model.py src/train.py scripts/predict_dev.py tests/test_model.py
git commit -m "feat: configurable encoder layer for live feature extraction"
```

---

## Task 5: Mel-branch toggle

**Why:** The mel-CNN branch collapses the spectrogram to one vector and broadcasts it identically across all 1500 frames — likely low-value and the reason for the float32/AMP-disable path. Make it ablatable to A/B whether removing it improves OOD transfer. Default `True` keeps current behavior.

**Files:**
- Modify: `src/model.py` (`__init__` and `forward`)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_mel_branch_can_be_disabled -v`
Expected: FAIL with `TypeError` (unexpected `use_mel_branch`).

- [ ] **Step 3: Make the branch conditional in `__init__`**

In `src/model.py`, extend the signature:

```python
    def __init__(self, whisper_model: str = "openai/whisper-medium", proj_dim: int = 256,
                 dropout: float = 0.1, encoder_layer: int = -1,
                 use_mel_branch: bool = True):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.use_mel_branch = use_mel_branch
```

Then guard the mel-branch modules with the toggle. Replace the four mel module definitions (`mel_transform`, `amplitude_to_db`, `mel_cnn`, `mel_proj`) with:

```python
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
```

Then set the BiLSTM input size based on the toggle — replace the `self.bilstm = nn.LSTM(input_size=proj_dim * 2, ...)` block with:

```python
        bilstm_input = proj_dim * 2 if self.use_mel_branch else proj_dim
        self.bilstm = nn.LSTM(
            input_size=bilstm_input,
            hidden_size=proj_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
```

(Heads/attention stay `proj_dim * 2` — the BiLSTM is bidirectional, so its output is `2*proj_dim` regardless.)

- [ ] **Step 4: Make `forward` skip the mel branch**

Replace the float32 fusion block (current lines 98-107) with:

```python
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
```

- [ ] **Step 5: Run model tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (new + existing).

- [ ] **Step 6: Pass `use_mel_branch` from config in train + predict_dev**

In `src/train.py` model construction add `use_mel_branch=m_cfg.get("use_mel_branch", True)`. In `scripts/predict_dev.py` model construction add the same. Final `predict_dev.py` construction:

```python
    model = WhisperMOSNet(
        whisper_model=whisper_model, proj_dim=m_cfg["proj_dim"],
        encoder_layer=m_cfg.get("encoder_layer", -1),
        use_mel_branch=m_cfg.get("use_mel_branch", True),
    ).to(device)
```

- [ ] **Step 7: Run the full suite + commit**

Run: `pytest -q`
Expected: PASS.

```bash
git add src/model.py src/train.py scripts/predict_dev.py tests/test_model.py
git commit -m "feat: make mel-CNN branch an ablatable config toggle"
```

---

## Task 6: Experiment configs + run/measure protocol

**Why:** Turn the three changes into measured experiments on Colab. Each experiment = one config, one training run, one submission, one recorded dev SRCC. **Baseline to beat: ACR 0.574.**

**Files:**
- Create: `configs/exp_a1_rankloss.yaml`, `configs/exp_a2_layer.yaml`, `configs/exp_a3_nomel.yaml`

- [ ] **Step 1: Create the rank-loss config (Experiment A1)**

Create `configs/exp_a1_rankloss.yaml` (identical to `pretrain_local.yaml` plus the new key):

```yaml
model:
  whisper_model: "openai/whisper-medium"
  proj_dim: 256
  dropout: 0.1
  encoder_layer: -1
  use_mel_branch: true

training:
  train_manifest: "data/manifests/pretrain_train.csv"
  dev_manifest: "data/manifests/pretrain_dev.csv"
  batch_size: 16
  epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_gamma: 0.9999
  ccr_lambda: 0.0
  ccr_lambda_ramp_epochs: 0
  acr_rank_alpha: 1.0
  cache_dir: "data/encoder_cache"
  checkpoint_dir: "checkpoints/exp_a1"
  checkpoint_every_n_epochs: 5
  num_workers: 0

logging:
  run_name: "exp-a1-rankloss"
```

- [ ] **Step 2: Regenerate manifests so `source` is populated (Colab)**

Run: `python data/build_manifests.py --data_dir data/datasets --output_dir data/manifests`
Expected: prints per-dataset counts; `pretrain_train.csv` / `pretrain_dev.csv` now contain a `source` column. (The existing feature cache stays valid — it is keyed by audio path, unaffected by the new column.)

- [ ] **Step 3: Train + measure A1**

Run: `python -m src.train --config configs/exp_a1_rankloss.yaml`
Expected: per-epoch lines printing `dev srcc=...`; final `Best dev SRCC`. Record the offline best dev SRCC.

Then generate + score the submission:

Run:
```bash
python scripts/predict_dev.py --checkpoint checkpoints/exp_a1/best.pt --config configs/exp_a1_rankloss.yaml --output predictions.csv --zip submission.zip
```
Upload `submission.zip` to CodaBench; record **Track 1 ACR UTT-SRCC**. Keep `acr_rank_alpha` only if dev ACR rises vs 0.574 without offline SRCC collapsing. If helpful, also try `acr_rank_alpha: 0.5` and `2.0` (one run each) and keep the best.

- [ ] **Step 4: Create + run the layer experiment (Experiment A2)**

Create `configs/exp_a2_layer.yaml` as a copy of the best A1 config with `encoder_layer: 12` and `cache_dir: "data/encoder_cache_layer12"`, `checkpoint_dir: "checkpoints/exp_a2_l12"`, `run_name: "exp-a2-layer12"`.

Cache features for that layer, then train + submit:
```bash
python -m data.cache_features --manifests data/manifests/pretrain_train.csv data/manifests/pretrain_dev.csv --cache_dir data/encoder_cache_layer12 --whisper_model openai/whisper-medium --layer 12
python -m src.train --config configs/exp_a2_layer.yaml
python scripts/predict_dev.py --checkpoint checkpoints/exp_a2_l12/best.pt --config configs/exp_a2_layer.yaml --output predictions.csv --zip submission.zip
```
Record dev ACR SRCC. Repeat for one or two more layers (e.g. 18, 20) if 12 helps; keep the best layer. **Storage note:** delete a layer's `encoder_cache_layer*` dir before caching the next if disk is tight.

- [ ] **Step 5: Create + run the mel-ablation experiment (Experiment A3)**

Create `configs/exp_a3_nomel.yaml` as a copy of the best config so far with `use_mel_branch: false`, `checkpoint_dir: "checkpoints/exp_a3_nomel"`, `run_name: "exp-a3-nomel"`. (Cache is reusable — the mel branch reads the raw waveform, which the dataset always returns.)

```bash
python -m src.train --config configs/exp_a3_nomel.yaml
python scripts/predict_dev.py --checkpoint checkpoints/exp_a3_nomel/best.pt --config configs/exp_a3_nomel.yaml --output predictions.csv --zip submission.zip
```
Record dev ACR SRCC; keep `use_mel_branch: false` only if it helps or is neutral (prefer the simpler model on ties).

- [ ] **Step 6: Commit the configs + a results note**

```bash
git add configs/exp_a1_rankloss.yaml configs/exp_a2_layer.yaml configs/exp_a3_nomel.yaml
git commit -m "chore: Approach A experiment configs"
```

Record the measured dev SRCCs (A1/A2/A3 vs the 0.574 baseline) in the PR description or a short `docs/` note.

---

## Decision gate (end of Approach A)

After A1–A3 are measured: if the best combination reaches **≥ 0.662**, optionally stop or push for the stretch target. Otherwise proceed to **Approach B** (domain-closer data: NISQA + TMHINT, de-weight BVCC), which will be planned in a separate document once these numbers are in — the measured deltas tell us the best backbone/layer/loss settings to carry into B.

---

## Self-Review Notes

- **Spec coverage:** A1 rank loss (§4.2 A1) → Tasks 2–3; A2 layer selection (§4.2 A2, constraint-adjusted to single-layer per §6 storage) → Task 4; A3 mel toggle (§4.2 A3) → Task 5; within-source masking (§4.2 correctness) → Tasks 1–2; methodology/measurement + offline guardrail (§4.1) → Task 6 + the methodology reminder. Approach B (§4.3) and C (§4.4) are intentionally deferred to a later plan per the staged gate.
- **Backward compatibility:** all new params default to current behavior (`acr_rank_alpha=0.0`, `encoder_layer=-1`, `use_mel_branch=True`), so existing configs and the existing feature cache keep working.
- **Type consistency:** `build_source_ids` returns a `long` tensor; `MOSLoss.forward(..., source_ids=None)` and `_pairwise_rank(..., source_ids=None)` accept it; `model(..., encoder_layer, use_mel_branch)` names match across `model.py`, `train.py`, `predict_dev.py`, and configs.
