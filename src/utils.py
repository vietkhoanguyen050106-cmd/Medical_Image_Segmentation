import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .transforms import IMAGENET_MEAN, IMAGENET_STD


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer, epoch, best_dice, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "config": config,
    }, path)


def denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_batch(images, masks, preds=None, n=4, save_path=None):
    """Hàng: Image | Ground truth | (Prediction nếu có)."""
    n = min(n, images.size(0))
    cols = 3 if preds is not None else 2
    fig, axes = plt.subplots(n, cols, figsize=(3.2 * cols, 3.2 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[i, 0].imshow(denormalize(images[i])); axes[i, 0].set_title("Image")
        axes[i, 1].imshow(masks[i, 0].cpu(), cmap="gray"); axes[i, 1].set_title("Ground truth")
        if preds is not None:
            axes[i, 2].imshow(preds[i, 0].cpu(), cmap="gray"); axes[i, 2].set_title("Prediction")
        for a in axes[i]:
            a.axis("off")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_history(csv_path, save_path=None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["epoch"], df["train_loss"], label="train")
    ax[0].plot(df["epoch"], df["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(df["epoch"], df["val_dice"], label="Dice")
    ax[1].plot(df["epoch"], df["val_iou"], label="IoU")
    ax[1].set_title("Validation metrics"); ax[1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()