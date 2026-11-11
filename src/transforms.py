"""Data augmentation dong bo Image-Mask (albumentations).

Albumentations ap dung CUNG phep bien hinh hoc len image va mask,
nhung phep mau (ColorJitter...) chi tac dong len image.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms(size: int = 256):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20,
                           border_mode=0, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_eval_transforms(size: int = 256):
    """Val/Test: KHONG augmentation, chi resize + normalize."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
