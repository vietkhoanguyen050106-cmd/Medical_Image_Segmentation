import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .dataloader import build_dataloaders
from .model import UNet
from .metrics import MetricTracker
from .utils import get_device, visualize_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/unet_baseline/best.pth")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]

    _, _, test_loader = build_dataloaders(cfg["image_size"], args.batch_size, args.num_workers)
    model = UNet(base=cfg["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tracker = MetricTracker()
    first_batch = None
    for images, masks, _ in tqdm(test_loader, desc="test"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        tracker.update(logits, masks)
        if first_batch is None:
            preds = (torch.sigmoid(logits) > args.threshold).float()
            first_batch = (images.cpu(), masks.cpu(), preds.cpu())

    results = {k: round(v, 4) for k, v in tracker.compute().items()}
    print("Test results:", results)

    out_dir = Path(args.ckpt).parent
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    visualize_batch(*first_batch, n=6, save_path=out_dir / "test_predictions.png")


if __name__ == "__main__":
    main()