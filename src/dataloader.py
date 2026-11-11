from datasets import load_dataset
from torch.utils.data import DataLoader

from .dataset import PolypSegmentationDataset
from .transforms import get_train_transforms, get_eval_transforms

HF_NAME = "MedOtter/KvasirSEG"


def _pick_split(ds, candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    raise KeyError(f"Không tìm thấy split nào trong {candidates}. Có: {list(ds.keys())}")


def build_dataloaders(image_size=256, batch_size=16, num_workers=2,
                      hf_name=HF_NAME, pin_memory=True):
    ds = load_dataset(hf_name)
    train_hf = _pick_split(ds, ["train"])
    val_hf = _pick_split(ds, ["validation", "val", "valid"])
    test_hf = _pick_split(ds, ["test"])

    train_set = PolypSegmentationDataset(train_hf, get_train_transforms(image_size))
    val_set = PolypSegmentationDataset(val_hf, get_eval_transforms(image_size))
    test_set = PolypSegmentationDataset(test_hf, get_eval_transforms(image_size))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader