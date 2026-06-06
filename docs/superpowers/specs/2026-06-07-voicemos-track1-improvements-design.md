# VoiceMOS 2026 Track 1 — Improvement Plan (Domain-Generalization Regime)

**Date:** 2026-06-07
**Status:** Approved design (brainstormed with Vrishab)
**Builds on:** `2026-05-20-voicemos-track1-design.md`
**Team:** Vrishab + Shubham
**Target metric:** Track-1 ACR UTT-SRCC ≥ 0.662 (match/beat current leaderboard)

---

## 1. Context & current state

| Item | Value |
|---|---|
| First dev submission | ACR UTT-SRCC **0.574**, CCR **−0.369** |
| CCR sign bug | Fixed (`acr_a − acr_b`); CCR now ≈ **+0.369**. Confirmed on dev. |
| Leaderboard top (training phase, same dev set) | ACR **0.662** / CCR **0.411** (two identical entries → likely a shared/official baseline) |

The 0.574 model is `WhisperMOSNet`: **frozen whisper-medium encoder** + a mel-spectrogram CNN branch + BiLSTM + attention pooling → ACR/CCR heads. It was trained **Stage-1 ("pretrain") only**, on **BVCC + TMHINT-QI** (AudioMOS-2025-T3 dropped out — download failed silently), with **MSE on ACR only and `ccr_lambda = 0`** (the CCR head never received a gradient). Stage-2 finetune was never run — `configs/finetune.yaml` points at `finetune_train.csv` / `finetune_dev.csv` that do not exist, and there is no `track1_train` dataset, only `track1_dev`.

**Key fact that defines this plan:** the challenge provides **no labeled in-domain training data** for Track 1. The only feedback on the target distribution (multilingual speech-enhancement quality, 9 languages, top-6 ICASSP-2026 URGENT systems) is the **unlimited** CodaBench dev UTT-SRCC. Therefore this is a **domain-generalization problem**: we train on out-of-domain (OOD) labeled data and must maximize transfer to the target.

So 0.574 is a *triple-handicapped* number — frozen encoder, zero in-domain data, and an MSE objective that fights for an absolute scale that does not transfer. That implies real headroom.

---

## 2. Constraints

| Constraint | Value | Implication |
|---|---|---|
| Compute | Free Colab T4 (16 GB), time-limited sessions | Encoder stays **frozen**; train a lightweight head on **cached features**. No encoder fine-tuning. |
| Dev evaluations | **Unlimited** | Dev UTT-SRCC is a live A/B signal — but guard against leaderboard overfitting. |
| Storage | Free-tier Drive (~15 GB) + ephemeral session disk | Cached features must be **small** (see §6). |

---

## 3. Goal & non-goals

**Goal:** raise Track-1 ACR UTT-SRCC from 0.574 to ≥ 0.662; CCR follows via the sign-fixed `clamp(acr_a − acr_b, −3, 3)` derivation.

**Non-goals:** encoder fine-tuning; an in-domain CCR head (no CCR labels exist anywhere in our data); Track 2; any architecture that cannot run from cached frozen features.

**Guiding principle — optimize rank, not absolute scale.** MOS *values* do not transfer across datasets (different listener pools use 1–5 differently), but quality *ordering* is far more domain-invariant, and SRCC only measures rank. Every design choice favors rank-robustness: rank-aware losses and training data whose *degradation types* match the target.

---

## 4. Strategy — staged A → B, C in reserve

Execute Approach A (cheap, fast-validating), measure on dev; if at/above target, optionally stop. Otherwise execute Approach B (the domain-gap fix). Approach C triggers only if B plateaus short of target.

### 4.1 Methodology — the measured climb (part of the design, not an afterthought)
- **One variable per experiment**, each behind a config flag.
- **Primary metric:** dev UTT-SRCC. **Guardrail:** a fixed **offline OOD holdout** (TMHINT-QI test split) scored on every run. If dev rises while the holdout collapses, we are overfitting the leaderboard (a real risk with unlimited submissions: enough tries will surface noise that won't survive the July-31 eval set). The holdout has labels we control, so it cannot be gamed.
- **Decision gates:** A → measure → (≥ 0.662 and holdout stable? → optional stop) → B → (plateau below target? → C).

### 4.2 Approach A — model / loss / feature changes (existing data)
- **A1 — Rank-aware ACR loss** (`src/losses.py`): `L = MSE + α · pairwise_ranking(ACR)`, reusing the `margin_ranking_loss` pattern already present for CCR. **Correctness requirement:** ranking is computed **within-source only** (mask cross-dataset pairs) — cross-dataset MOS values are not comparable. `α` is tuned on dev.
- **A2 — Encoder layer selection** (`src/model.py`, `data/cache_features.py`): replace the hard-coded final encoder layer with a configurable *set* of cached layers combined by a learnable softmax-weighted sum (s3prl/SUPERB-style). This subsumes both the existing layer-12 experiment and single-best-layer selection. Rationale: intermediate SSL layers retain acoustic/quality information that the ASR-oriented final layer discards.
- **A3 — Mel-branch toggle** (`src/model.py`): gate the mel-CNN branch behind `use_mel_branch`. It currently collapses the spectrogram to a single vector and broadcasts it identically across all 1500 frames — effectively a constant bias term, at the cost of a float32/AMP-disable path. A/B it; expectation is that removal helps (simpler ⇒ better OOD transfer, and it removes the AMP hack).

### 4.3 Approach B — close the domain gap with data
- **Add the NISQA Corpus** (~14k clips with real degradation MOS: noisiness, coloration, discontinuity, loudness — almost exactly the Track-1 quality axis): new `data/download.py` target + a parser that normalizes it into the unified manifest (`path, acr, ccr=NaN, language, system, split`).
- **Keep TMHINT-QI** (noisy/enhanced-speech quality — already on-domain). **Demote BVCC** to optional/low-weight (synthesis naturalness is the furthest source from the target); keep-or-drop decided on dev.
- **Per-source score normalization** (z-score per dataset) so pooled MSE is coherent across differing scales; the rank loss already handles ordering within-source.

### 4.4 Approach C — reserve (only if B plateaus)
- Add **XLS-R-300M** (multilingual wav2vec2 — quality-sensitive *and* covers the 9 languages, unlike English WavLM or ASR-invariant Whisper) as an alternative frozen backbone; pick the winner on dev.
- **Ensemble** Whisper + XLS-R predictions (averaging reliably lifts SRCC at negligible inference cost).

---

## 5. Architecture (after A, frozen-feature path)

```
cached encoder features (subset of layers, frame-downsampled, fp16)
        │
 learnable layer-weighting (softmax over cached layers)
        │
     adapter (Linear → LayerNorm → Dropout)
        │
 [optional mel-CNN branch — A/B via use_mel_branch]
        │
   BiLSTM → attention pooling
        │
   ┌────┴─────┐
 ACR head   CCR head (untrained; inference uses derived CCR)
```

---

## 6. Storage & compute plan (the real free-T4 constraint)

Full-resolution caching does not fit: whisper-medium at 1500×1024 fp16 ≈ 3 MB / clip / layer; ~30k clips × several layers ≈ hundreds of GB. Mitigations, applied together:
- **Frame-downsample** cached features (~50 fps → ~10–15 fps via average-pool over time; quality is near-stationary, so fine temporal resolution is unnecessary).
- **fp16** storage and a **small layer set** (e.g., 2–3 layers).
- **Default to whisper-small (768-d)** for Phase B to halve storage; A/B against medium on dev.
- Target a cache **≲ 10–15 GB** so it persists on Drive; otherwise re-extract once per session on ephemeral disk.

The same feature configuration (backbone, layer set, frame-pool factor) is shared by training (`cache_features.py`, `dataset.py`) and inference (`predict_dev.py`) so dev predictions are produced from identical features.

---

## 7. Files touched (kept isolated)

| File | Change |
|---|---|
| `src/losses.py` | Add rank-aware ACR term with within-source masking |
| `src/model.py` | Learnable layer-weighting module; `use_mel_branch` toggle; head input-dim adjusts |
| `data/cache_features.py` | Extract/cache a configurable layer subset; frame-downsampling |
| `data/download.py` | NISQA download target |
| `data/build_manifests.py` | NISQA parser; per-source z-norm; dataset weights/inclusion |
| `src/dataset.py` | Carry `source` id; per-source normalization; return selected-layer feats |
| `configs/*.yaml` | New flags: layer set, frame-pool, `use_mel_branch`, `acr_rank_alpha`, dataset weights |
| `scripts/predict_dev.py` | Consume the shared feature config |
| `tests/` | Coverage for the new loss (within-source masking) and the NISQA parser |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Cache too big for free tier | Frame-downsample + fp16 + whisper-small + small layer set (§6) |
| Leaderboard overfitting (unlimited submissions) | Fixed OOD holdout guardrail (§4.1) |
| Cross-dataset scale mismatch | Within-source rank loss + per-source z-norm (§4.2/§4.3) |
| NISQA skews English/German | Degradation-type coverage is what transfers, not language; Whisper handles language |
| Layer-caching rework diverges train vs inference | Single shared feature config used by both (§6) |

---

## 9. Success criteria

- **Primary:** dev ACR UTT-SRCC ≥ 0.662 (match/beat leaders) with no OOD-holdout collapse.
- **Stretch:** ≥ 0.70.
- CCR remains the sign-fixed derived score (expected to track ACR gains).

---

## 10. Decisions deferred to the implementation plan (resolved empirically on dev)

These are intentionally left to the plan/experiment phase, not unknowns in the design:
- Exact cached layer set and frame-pool factor (sized to the storage budget in §6).
- Backbone size (whisper-small vs medium).
- Rank-loss weight `α` and its exact form (pairwise margin vs batch-correlation).
- Whether BVCC stays in the mix and at what weight.
- Whether the mel branch is kept (A3 A/B outcome).
