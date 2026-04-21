import argparse
import random
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.dataset import CCCDDataset, collate_fn
from src.evaluate import full_evaluation
from src.model import build_model
from src.visualize import save_epoch_plot, save_kfold_plot



def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(dataloader, desc="  Train", leave=False):
        images     = [img.to(device) for img in images]
        targets    = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss       = sum(model(images, targets).values())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    for images, targets in tqdm(dataloader, desc="  Val  ", leave=False):
        images = [img.to(device) for img in images]
        preds  = model(images)
        metric.update(
            [{"boxes":  p["boxes"].cpu(), "scores": p["scores"].cpu(),
              "labels": p["labels"].cpu()} for p in preds],
            [{"boxes":  t["boxes"].cpu(),
              "labels": t["labels"].cpu()} for t in targets],
        )
    r = metric.compute()
    return {"map_50":  round(r["map_50"].item(),  4),
            "map":     round(r["map"].item(),     4),
            "mar_100": round(r["mar_100"].item(), 4)}


def run_training(model, train_loader, val_loader, cfg, device,
                 weights_dir, label=""):
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    writer    = SummaryWriter(log_dir=str(weights_dir / "tensorboard"))
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 0.005), momentum=0.9, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 50), eta_min=1e-5,
    )

    best_map50 = 0.0
    no_improve = 0
    PATIENCE   = 10
    history    = []

    for epoch in range(1, cfg.get("epochs", 50) + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n{label} Epoch {epoch}/{cfg.get('epochs', 50)}"
              f"  lr={lr:.6f}")

        loss    = train_one_epoch(model, train_loader, optimizer, device)
        metrics = validate(model, val_loader, device)
        scheduler.step()

        print(f"  loss={loss:.4f}  mAP@0.5={metrics['map_50']}"
              f"  mAP={metrics['map']}  mAR={metrics['mar_100']}")

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Metrics/mAP@0.5", metrics["map_50"], epoch)
        writer.add_scalar("Metrics/mAP@0.5:0.95", metrics["map"], epoch)
        writer.add_scalar("Metrics/mAR@100", metrics["mar_100"], epoch)
        writer.add_scalar("LR", lr, epoch)
        writer.add_scalars("Compare/mAP", {
            "current": metrics["map_50"],
            "best":    max(best_map50, metrics["map_50"]),
        }, epoch)

        history.append({
            "epoch": epoch,
            "loss": round(loss, 6),
            "lr": round(lr,   8),
            "map_50": metrics["map_50"],
            "map": metrics["map"],
            "mar_100": metrics["mar_100"],
            "best_map50": round(max(best_map50, metrics["map_50"]), 4),
        })

        save_epoch_plot(history, save_dir=weights_dir, label=label)

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   weights_dir / "last.pth")

        if metrics["map_50"] > best_map50:
            best_map50 = metrics["map_50"]
            no_improve = 0
            torch.save(model.state_dict(), weights_dir / "best.pth")
            print(f"  [OK] Best saved  mAP@0.5={best_map50}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping.")
                break

    print(f"\n{label} Đánh giá cuối cùng...")
    full_evaluation(model, val_loader, device, cfg,
                    weights_dir, writer=writer)
    writer.close()
    return best_map50, history

def _make_loaders(dataset, train_idx, val_idx, batch_size):
    return (
        DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                   shuffle=True,  num_workers=2, collate_fn=collate_fn),
        DataLoader(Subset(dataset, val_idx),   batch_size=1,
                   shuffle=False, num_workers=2, collate_fn=collate_fn),
    )


def _build_model(cfg, device):
    return build_model(cfg["num_classes"],
                       cfg.get("max_size", 800),
                       cfg.get("score_thresh", 0.4)).to(device)


def train_normal(cfg, device, args):
    print("\n[Chế độ: Train thường]")
    dataset = CCCDDataset(cfg["img_dir"], cfg["ann_dir"], cfg["class_map"])
    indices = list(range(len(dataset)))

    random.seed(42); random.shuffle(indices)
    n_val = max(1, int(len(indices) * 0.15))
    train_loader, val_loader = _make_loaders(
        dataset, indices[n_val:], indices[:n_val], cfg.get("batch_size", 2)
    )
    print(f"  Train: {len(indices) - n_val}  |  Val: {n_val}")

    model = _build_model(cfg, device)
    if args.resume:
        model.load_state_dict(
            torch.load(args.resume, map_location=device)["model"])
        print(f"  Resumed: {args.resume}")

    best, _ = run_training(model, train_loader, val_loader,
                           cfg, device, cfg["weights_dir"])
    wd = cfg["weights_dir"]
    print(f"\n[OK] Best mAP@0.5={best:.4f}  →  {wd}/best.pth")
    print(f"     tensorboard --logdir {wd}  →  http://localhost:6006")


def train_kfold(cfg, device, k):
    print(f"\n[Chế độ: {k}-Fold Cross Validation]")
    dataset = CCCDDataset(cfg["img_dir"], cfg["ann_dir"], cfg["class_map"])
    indices = list(range(len(dataset)))
    random.seed(42); random.shuffle(indices)

    fold_size = len(indices) // k
    folds = [indices[i * fold_size : (i+1)*fold_size if i<k-1 else None]
                 for i in range(k)]

    all_maps, all_histories = [], []

    for i in range(k):
        print(f"\n{'═'*45}\n  FOLD {i+1}/{k}\n{'═'*45}")
        val_idx   = folds[i]
        train_idx = [x for j, f in enumerate(folds) if j != i for x in f]
        train_loader, val_loader = _make_loaders(
            dataset, train_idx, val_idx, cfg.get("batch_size", 2)
        )
        print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

        best, history = run_training(
            _build_model(cfg, device),
            train_loader, val_loader, cfg, device,
            weights_dir = Path(cfg["weights_dir"]) / f"fold_{i+1}",
            label       = f"[Fold {i+1}/{k}]",
        )
        all_maps.append(best); all_histories.append(history)

    best_i = all_maps.index(max(all_maps))
    shutil.copy2(Path(cfg["weights_dir"]) / f"fold_{best_i+1}" / "best.pth",
                 Path(cfg["weights_dir"]) / "best.pth")

    save_kfold_plot(all_histories, cfg["weights_dir"], k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--kfold",  type=int, default=None)
    parser.add_argument("--resume", default=None)
    args   = parser.parse_args()
    cfg    = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config: {args.config}  |  Device: {device}")

    if args.kfold:
        train_kfold(cfg, device, args.kfold)
    else:
        train_normal(cfg, device, args)


if __name__ == "__main__":
    main()