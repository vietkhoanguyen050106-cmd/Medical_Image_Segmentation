
from torch.utils.data import DataLoader

from src.dataset import PolypSegmentationDataset


def build_dataloaders(
    hf_dataset,
    image_size=(256, 256),
    batch_size=4,
    num_workers=0,
    pin_memory=False
):
    """
    Build PyTorch DataLoaders from Hugging Face dataset splits.

    Expected splits:
        train, val, test
    """

    required_splits = {"train", "val", "test"}

    missing_splits = required_splits - set(hf_dataset.keys())

    if missing_splits:
        raise ValueError(
            f"Missing dataset splits: {missing_splits}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    # Step 1: Build PyTorch Dataset objects.
    datasets = {}

    for split_name in ["train", "val", "test"]:

        datasets[split_name] = PolypSegmentationDataset(
            hf_dataset=hf_dataset[split_name],
            image_size=image_size
        )

    # Step 2: Build DataLoader objects.
    loaders = {}

    for split_name, dataset in datasets.items():

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

    return loaders
