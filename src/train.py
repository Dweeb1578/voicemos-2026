"""Training loop for WhisperMOSNet.

Usage:
    python -m src.train --config configs/pretrain.yaml
    python -m src.train --config configs/finetune.yaml
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import MOSDataset
from src.evaluate import compute_metrics
from src.losses import MOSLoss
from src.model import WhisperMOSNet


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_ccr_lambda(epoch: int, cfg: dict) -> float:
    target = cfg["training"]["ccr_lambda"]
    ramp = cfg["training"].get("ccr_lambda_ramp_epochs", 0)
    if ramp == 0:
        return target
    return min(target, target * (epoch / ramp))


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

        with torch.cuda.amp.autocast():
            acr_p, ccr_p = model(inp, wav, encoder_feats=enc)
            loss = loss_fn(acr_p, ccr_p, acr_t, ccr_t)

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

    return total_loss / len(loader), compute_metrics(acr_preds, acr_targets)


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
    ).to(device)

    if "pretrained_checkpoint" in t_cfg:
        ckpt = torch.load(t_cfg["pretrained_checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded: {t_cfg['pretrained_checkpoint']}")

    train_ds = MOSDataset(t_cfg["train_manifest"], whisper_model=m_cfg["whisper_model"],
                          cache_dir=cache_dir)
    dev_ds = MOSDataset(t_cfg["dev_manifest"], whisper_model=m_cfg["whisper_model"],
                        cache_dir=cache_dir)

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
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt["epoch"] + 1
            best_srcc = ckpt.get("best_srcc", -1.0)
            print(f"Resumed from {latest} (epoch {ckpt['epoch']}, best SRCC so far: {best_srcc:.4f})")

    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        loss_fn = MOSLoss(ccr_lambda=get_ccr_lambda(epoch, cfg))

        tr_loss, tr_m = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, train=True)
        dv_loss, dv_m = run_epoch(model, dev_loader, loss_fn, optimizer, scaler, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr_loss:.4f} srcc={tr_m['srcc']:.4f} | "
            f"dev loss={dv_loss:.4f} srcc={dv_m['srcc']:.4f}"
        )

        if dv_m["srcc"] > best_srcc:
            best_srcc = dv_m["srcc"]
            best_path = os.path.join(t_cfg["checkpoint_dir"], "best.pt")
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "dev_srcc": best_srcc},
                best_path,
            )
            if drive_dir:
                import shutil
                shutil.copy(best_path, os.path.join(drive_dir, "pretrain_best.pt"))
                print(f"  -> best.pt saved to Drive (epoch {epoch}, SRCC {best_srcc:.4f})")

        save_every = t_cfg.get("checkpoint_every_n_epochs", 5)
        if epoch % save_every == 0:
            ckpt_data = {"epoch": epoch, "model_state": model.state_dict(), "best_srcc": best_srcc}
            torch.save(ckpt_data, os.path.join(t_cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pt"))
            if drive_dir:
                import shutil
                shutil.copy(
                    os.path.join(t_cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pt"),
                    os.path.join(drive_dir, f"epoch_{epoch:03d}.pt"),
                )
                print(f"  -> saved to Drive: epoch_{epoch:03d}.pt")

    print(f"Done. Best dev SRCC: {best_srcc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
