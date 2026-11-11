"""pytest -q  : kiem tra nhanh shape, loss, metric, augmentation dong bo."""
import numpy as np, torch
from src.model import UNet
from src.losses import BCEDiceLoss
from src.metrics import batch_metrics
from src.transforms import get_train_transforms


def test_unet_shape():
    m = UNet(3, 1, 16); assert m(torch.randn(2, 3, 128, 128)).shape == (2, 1, 128, 128)


def test_loss_backward():
    m = UNet(3, 1, 16); x = torch.randn(2, 3, 64, 64); y = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = BCEDiceLoss()(m(x), y); loss.backward(); assert torch.isfinite(loss)


def test_perfect_prediction_metrics():
    y = (torch.rand(2, 1, 32, 32) > 0.5).float(); logits = (y * 2 - 1) * 20
    assert torch.allclose(batch_metrics(logits, y), torch.ones(2, 4), atol=1e-3)


def test_augmentation_sync():
    img = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    mask = np.zeros((300, 400), np.uint8); mask[100:200, 150:300] = 1
    out = get_train_transforms(128)(image=img, mask=mask)
    assert out["image"].shape == (3, 128, 128) and out["mask"].shape == (128, 128)
    assert set(np.unique(out["mask"].numpy())) <= {0, 1}
