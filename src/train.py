"""Training loop for WhisperMOSNet.

Usage:
    python -m src.train --config configs/pretrain.yaml
    python -m src.train --config configs/finetune.yaml
"""

import argparse
import math
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import MOSDataset
from src.evaluate import compute_metrics
from src.losses import MOSLoss
from src.model import WhisperMOSNet
from src.train_utils import build_source_ids


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_ccr_lambda(epoch: int, cfg: dict) -> float:
    target = cfg["training"]["ccr_lambda"]
    ramp = cfg["training"].get("ccr_lambda_ramp_epochs", 0)
    if ramp == 0:
        return target
    return min(target, target * (epoch / ramp))


def trainable_state_dict(model):
    """Model state excluding the frozen Whisper encoder (it is reloaded from HF on
    construction, so persisting its ~1.4 GB every checkpoint is pure waste).
    Load back with strict=False -- the encoder comes from the HF init."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("whisper_encoder.")}


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss = 0.0
    acr_preds, acr_targets = [], []

    for batch in loader:
        wav = batch["waveform"].to(device)
        acr_t = batch["acr"].to(device)
        ccr_t = batch["ccr"].to(device)

        inp = batch["input_features"].to(device) if "input_features" in batch else None
        enc = batch["encoder_feats"].to(device) if "encoder_feats" in batch else None

        src_ids = None
        if "source" in batch:
            src_ids = build_source_ids(batch["source"]).to(device)

        with torch.cuda.amp.autocast():
            acr_p, ccr_p = model(inp, wav, encoder_feats=enc)
            loss = loss_fn(acr_p, ccr_p, acr_t, ccr_t, source_ids=src_ids)

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        acr_preds.extend(acr_p.detach().cpu().tolist())
        acr_targets.extend(acr_t.cpu().tolist())

    return total_loss / len(loader), compute_metrics(acr_preds, acr_targets), acr_preds, acr_targets


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    m_cfg, t_cfg = cfg["model"], cfg["training"]
    cache_dir = t_cfg.get("cache_dir", None)
    if cache_dir:
        print(f"Using encoder cache: {cache_dir}")

    model = WhisperMOSNet(
        whisper_model=m_cfg["whisper_model"], proj_dim=m_cfg["proj_dim"],
        dropout=m_cfg.get("dropout", 0.0),
        encoder_layer=m_cfg.get("encoder_layer", -1),
        use_mel_branch=m_cfg.get("use_mel_branch", True),
    ).to(device)

    if "pretrained_checkpoint" in t_cfg:
        ckpt = torch.load(t_cfg["pretrained_checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded: {t_cfg['pretrained_checkpoint']}")

    load_waveform = m_cfg.get("use_mel_branch", True)
    train_ds = MOSDataset(t_cfg["train_manifest"], whisper_model=m_cfg["whisper_model"],
                          cache_dir=cache_dir, load_waveform=load_waveform)
    dev_ds = MOSDataset(t_cfg["dev_manifest"], whisper_model=m_cfg["whisper_model"],
                        cache_dir=cache_dir, load_waveform=load_waveform)

    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=t_cfg.get("num_workers", 0), pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg.get("num_workers", 0), pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t_cfg["lr"], weight_decay=t_cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=t_cfg["lr_gamma"])
    scaler = torch.cuda.amp.GradScaler()

    assert len(train_ds) > 0, f"train manifest is empty: {t_cfg['train_manifest']}"
    assert len(dev_ds) > 0, f"dev manifest is empty: {t_cfg['dev_manifest']}"
    print(f"train={len(train_ds)} dev={len(dev_ds)} samples")

    os.makedirs(t_cfg["checkpoint_dir"], exist_ok=True)
    drive_dir = t_cfg.get("drive_checkpoint_dir", None)
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
    best_srcc = -1.0

    # Resume from latest Drive checkpoint if available
    start_epoch = 1
    if drive_dir:
        import glob
        drive_ckpts = sorted(glob.glob(os.path.join(drive_dir, "epoch_*.pt")))
        if drive_ckpts:
            latest = drive_ckpts[-1]
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            start_epoch = ckpt["epoch"] + 1
            best_srcc = ckpt.get("best_srcc", -1.0)
            print(f"Resumed from {latest} (epoch {ckpt['epoch']}, best SRCC so far: {best_srcc:.4f})")

    # Smoke check: one dev forward BEFORE the long loop, so a NaN/constant
    # prediction bug surfaces in seconds rather than after a full training run
    # that would silently save nothing (a NaN SRCC never beats the -1.0 init).
    model.eval()
    with torch.no_grad():
        b = next(iter(dev_loader))
        inp0 = b["input_features"].to(device) if "input_features" in b else None
        enc0 = b["encoder_feats"].to(device) if "encoder_feats" in b else None
        acr0, _ = model(inp0, b["waveform"].to(device), encoder_feats=enc0)
        a0 = acr0.detach().cpu().float().numpy()
        print(f"[smoke] dev ACR preds: n={a0.size} nan={int(np.isnan(a0).sum())} "
              f"std={np.nanstd(a0):.4g} min={np.nanmin(a0):.4g} max={np.nanmax(a0):.4g}")
        if np.isnan(a0).any():
            print("[smoke] WARNING: model emits NaN before any training -- inspect the "
                  "cached features for inf/NaN and the fp16 forward path.")

    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        loss_fn = MOSLoss(
            ccr_lambda=get_ccr_lambda(epoch, cfg),
            acr_rank_alpha=cfg["training"].get("acr_rank_alpha", 0.0),
        )

        tr_loss, tr_m, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, train=True)
        dv_loss, dv_m, dv_preds, dv_targets = run_epoch(model, dev_loader, loss_fn, optimizer, scaler, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr_loss:.4f} srcc={tr_m['srcc']:.4f} | "
            f"dev loss={dv_loss:.4f} srcc={dv_m['srcc']:.4f}"
        )

        # A NaN dev SRCC is why a run can "succeed" yet save nothing. Print what
        # made it NaN -- degenerate predictions (all-NaN/constant) or targets.
        if math.isnan(dv_m["srcc"]):
            dp, dt = np.array(dv_preds), np.array(dv_targets)
            print(f"  !! dev SRCC is NaN @ epoch {epoch}. "
                  f"preds: nan={int(np.isnan(dp).sum())}/{dp.size} std={np.nanstd(dp):.4g} "
                  f"min={np.nanmin(dp):.4g} max={np.nanmax(dp):.4g} | "
                  f"targets: nan={int(np.isnan(dt).sum())}/{dt.size} std={np.nanstd(dt):.4g}")

        # Always keep the latest model so a run is never wasted, even when SRCC is NaN.
        torch.save(
            {"epoch": epoch, "model_state": trainable_state_dict(model), "dev_srcc": dv_m["srcc"]},
            os.path.join(t_cfg["checkpoint_dir"], "last.pt"),
        )

        if dv_m["srcc"] > best_srcc:
            best_srcc = dv_m["srcc"]
            best_path = os.path.join(t_cfg["checkpoint_dir"], "best.pt")
            torch.save(
                {"epoch": epoch, "model_state": trainable_state_dict(model), "dev_srcc": best_srcc},
                best_path,
            )
            if drive_dir:
                import shutil
                shutil.copy(best_path, os.path.join(drive_dir, "pretrain_best.pt"))
                print(f"  -> best.pt saved to Drive (epoch {epoch}, SRCC {best_srcc:.4f})")

        save_every = t_cfg.get("checkpoint_every_n_epochs", 5)
        if epoch % save_every == 0:
            ckpt_data = {"epoch": epoch, "model_state": trainable_state_dict(model), "best_srcc": best_srcc}
            torch.save(ckpt_data, os.path.join(t_cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pt"))
            if drive_dir:
                import shutil
                shutil.copy(
                    os.path.join(t_cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pt"),
                    os.path.join(drive_dir, f"epoch_{epoch:03d}.pt"),
                )
                print(f"  -> saved to Drive: epoch_{epoch:03d}.pt")

    # Never finish a run with no submittable checkpoint: if SRCC was NaN every
    # epoch, best.pt was never written -- fall back to last.pt with a loud warning
    # so predict_dev can still run and the smoke/NaN diagnostics above are actionable.
    best_path = os.path.join(t_cfg["checkpoint_dir"], "best.pt")
    if not os.path.exists(best_path):
        import shutil
        last_path = os.path.join(t_cfg["checkpoint_dir"], "last.pt")
        print("WARNING: dev SRCC never beat the -1.0 init (NaN every epoch?). "
              "Copying last.pt -> best.pt so the pipeline completes; the model is "
              "likely degenerate -- act on the NaN diagnostic above before submitting.")
        shutil.copy(last_path, best_path)

    print(f"Done. Best dev SRCC: {best_srcc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
