"""Training entry point.

    python -m aletheia.train --config configs/crnn.yaml \
        --data-root /content/data --output-dir /content/drive/MyDrive/aletheia/runs

Checkpoints go to `<output-dir>/<run-name>/`. If `last.pt` already exists
there, training resumes from it automatically (useful when Colab disconnects).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import SARDataset
from .models import build_model


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    # Different numpy RNG per dataloader worker (otherwise identical crops).
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    losses, probs, labels = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        losses.append(criterion(logits.float(), y).item() * len(y))
        probs.append(logits.float().softmax(1)[:, 1].cpu())
        labels.append(y.cpu())
    p, t = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    metrics = {"loss": float(np.sum(losses) / len(t)), "acc": float(((p > 0.5) == t).mean())}
    metrics["auc"] = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else float("nan")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", help="overrides data.root")
    ap.add_argument("--output-dir", help="overrides output_dir")
    ap.add_argument("--run-name", help="defaults to the config file name")
    ap.add_argument("--epochs", type=int, help="overrides train.epochs")
    ap.add_argument("--batch-size", type=int, help="overrides train.batch_size")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    run_name = args.run_name or Path(args.config).stem

    dcfg, tcfg = cfg["data"], cfg["train"]
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    out_dir = Path(cfg.get("output_dir", "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    # ---- data -------------------------------------------------------------
    common = dict(
        root=dcfg["root"],
        class_names=dcfg["class_names"],
        img_size=dcfg["img_size"],
        log_transform=dcfg.get("log_transform", True),
    )
    train_ds = SARDataset(split="train", train=True, **common)
    val_ds = SARDataset(split="val", train=False, **common)
    print(f"train: {len(train_ds)} {train_ds.class_counts()} | val: {len(val_ds)} {val_ds.class_counts()}")

    loader_kw = dict(
        batch_size=tcfg["batch_size"],
        num_workers=tcfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init,
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ---- model / optim ----------------------------------------------------
    model = build_model(cfg["model"], num_classes=len(dcfg["class_names"])).to(device)
    print(f"Model: {cfg['model']['name']} — {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    # Class weights compensate an unbalanced real/fake ratio.
    counts = torch.tensor(train_ds.class_counts(), dtype=torch.float)
    weights = (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights if tcfg.get("class_weights", True) else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch, best_auc = 0, -1.0
    last_ckpt = out_dir / "last.pt"
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_auc = ckpt["epoch"] + 1, ckpt["best_auc"]
        print(f"Resumed from {last_ckpt} at epoch {start_epoch}")

    # ---- loop -------------------------------------------------------------
    log_path = out_dir / "log.csv"
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{tcfg['epochs']}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            n += len(y)
            pbar.set_postfix(loss=f"{running / n:.4f}")
        scheduler.step()

        val = evaluate(model, val_dl, device, criterion)
        row = {"epoch": epoch + 1, "train_loss": running / n, **{f"val_{k}": v for k, v in val.items()},
               "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t0}
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}))
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}, out_dir / "best.pt")
            print(f"  -> new best val AUC {best_auc:.4f}")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch, "best_auc": best_auc},
            last_ckpt,
        )

    print(f"Done. Best val AUC: {best_auc:.4f} — checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
