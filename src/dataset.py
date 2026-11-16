
import numpy as np
from torch.utils.data import Dataset


class PolypSegmentationDataset(Dataset):
    """
    Dataset wrapper for Hugging Face Kvasir-SEG.

    Returns:
        image: FloatTensor (3, H, W)
        mask: FloatTensor (1, H, W), values {0, 1}
        image_id: str
    """

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load RGB image
        image = np.array(
            sample["image"].convert("RGB")
        )

        # Load original grayscale mask
        raw_mask = np.array(
            sample["mask"].convert("L")
        )

        # Support both binary encodings:
        # 0/1 and 0/255
        if raw_mask.max() <= 1:
            mask = (raw_mask > 0).astype(np.uint8)
        else:
            mask = (raw_mask > 127).astype(np.uint8)

        # Apply synchronized image-mask transformations
        if self.transform is not None:
            out = self.transform(
                image=image,
                mask=mask
            )
            image = out["image"]
            mask = out["mask"]

        mask = mask.unsqueeze(0).float()

        image_id = str(sample.get("image_id", idx))

        return image, mask, image_id
