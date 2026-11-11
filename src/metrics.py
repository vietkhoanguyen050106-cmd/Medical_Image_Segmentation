import torch

EPS = 1e-7


@torch.no_grad()
def batch_metrics(logits, target, threshold=0.5):
    """Trả về dict các tensor (B,) - metric theo từng ảnh."""
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = target.flatten(1)
    tp = (pred * target).sum(1)
    fp = (pred * (1 - target)).sum(1)
    fn = ((1 - pred) * target).sum(1)
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
    }


class MetricTracker:
    """Tích lũy metric theo ảnh rồi lấy trung bình toàn tập."""
    KEYS = ("dice", "iou", "precision", "recall")

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {k: 0.0 for k in self.KEYS}
        self.n = 0

    def update(self, logits, target):
        m = batch_metrics(logits, target)
        for k in self.KEYS:
            self.sums[k] += m[k].sum().item()
        self.n += target.size(0)

    def compute(self):
        return {k: self.sums[k] / max(self.n, 1) for k in self.KEYS}