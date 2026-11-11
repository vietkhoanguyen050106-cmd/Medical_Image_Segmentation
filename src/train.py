import argparse
import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .dataloader import build_dataloaders
from .model import UNet, count_parameters
from .losses import get_loss
from .metrics import MetricTracker
from .utils import seed_everything, get_device, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", type=str, default="bce_dice", choices=["bce", "dice", "bce_dice"])
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_name", type=str, default="unet_baseline")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tracker = MetricTracker()
    total_loss, n = 0.0, 0
    use_amp = device.type == "cuda"

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks, _ in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            loss = criterion(logits.float(), masks)  # loss tính ở fp32 cho ổn định
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        tracker.update(logits.detach().float(), masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / n, tracker.compute()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out = Path(args.out_dir) / args.exp_name
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(
        args.image_size, args.batch_size, args.num_workers)

    model = UNet(base=args.base).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = get_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path = out / "history.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_dice", "val_loss",
                                "val_dice", "val_iou", "val_precision", "val_recall"])

    best_dice, bad_epochs = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_m = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.6f}", f"{tr_loss:.4f}", f"{tr_m['dice']:.4f}",
                                    f"{va_loss:.4f}", f"{va_m['dice']:.4f}", f"{va_m['iou']:.4f}",
                                    f"{va_m['precision']:.4f}", f"{va_m['recall']:.4f}"])

        print(f"Epoch {epoch:03d}/{args.epochs} | {time.time()-t0:.0f}s | lr {lr:.2e} | "
              f"train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
              f"val_dice {va_m['dice']:.4f} | val_iou {va_m['iou']:.4f}")

        if va_m["dice"] > best_dice:
            best_dice, bad_epochs = va_m["dice"], 0
            save_checkpoint(out / "best.pth", model, optimizer, epoch, best_dice, vars(args))
            print(f"  -> Saved best model (val_dice={best_dice:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping tại epoch {epoch}")
                break

    save_checkpoint(out / "last.pth", model, optimizer, epoch, best_dice, vars(args))
    print(f"Hoàn tất. Best val Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()