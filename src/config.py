"""Cau hinh tap trung: doi 1 cho, ap dung toan project (reproducibility)."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # data
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 2
    # model
    base_channels: int = 32          # 32 nhe, 64 la U-Net goc
    # training
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    bce_weight: float = 0.5          # loss = bce_weight*BCE + (1-bce_weight)*Dice
    patience: int = 15               # early stopping tren val Dice
    threshold: float = 0.5
    seed: int = 42
    amp: bool = True
    out_dir: str = "outputs"
    run_name: str = "unet_baseline"

    def to_dict(self):
        return asdict(self)
