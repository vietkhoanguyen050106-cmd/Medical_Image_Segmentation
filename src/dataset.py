import numpy as np
from torch.utils.data import Dataset


class PolypSegmentationDataset(Dataset):
    """
    Bọc một split của Hugging Face dataset (fields: image, mask, image_id).
    Trả về:
        image: FloatTensor (3, H, W)
        mask : FloatTensor (1, H, W), giá trị {0, 1}
        image_id: str
    """

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = np.array(sample["image"].convert("RGB"))
        mask = np.array(sample["mask"].convert("L"))
        mask = (mask > 127).astype(np.uint8)  # nhị phân hóa, tránh nhiễu JPEG

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        mask = mask.unsqueeze(0).float()
        image_id = str(sample.get("image_id", idx))
        return image, mask, image_id