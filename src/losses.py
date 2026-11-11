import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        inter = (probs * target).sum(1)
        dice = (2 * inter + self.smooth) / (probs.sum(1) + target.sum(1) + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bw, self.dw = bce_weight, dice_weight

    def forward(self, logits, target):
        return self.bw * self.bce(logits, target) + self.dw * self.dice(logits, target)


def get_loss(name: str = "bce_dice"):
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss: {name}")