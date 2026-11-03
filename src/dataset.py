
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class PolypSegmentationDataset(Dataset):

    def __init__(
        self,
        hf_dataset,
        image_size=(256, 256)
    ):
        """
        Args:
            hf_dataset: A Hugging Face dataset split.
            image_size: Target size as (height, width).
        """

        self.data = hf_dataset
        self.image_height, self.image_width = image_size

    def __len__(self):
        """
        Return the number of samples.
        """
        return len(self.data)

    def _resize_and_pad(self, image, mask):
        """
        Resize image and mask while preserving aspect ratio.
        Pad both to the target size.
        """

        target_h = self.image_height
        target_w = self.image_width

        original_w, original_h = image.size

        scale = min(
            target_w / original_w,
            target_h / original_h
        )

        new_w = round(original_w * scale)
        new_h = round(original_h * scale)

        # Resize image and mask using different interpolation.
        image = image.resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (new_w, new_h),
            Image.Resampling.NEAREST
        )

        # Calculate padding.
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2

        # Create empty canvases.
        padded_image = Image.new(
            "RGB",
            (target_w, target_h),
            (0, 0, 0)
        )

        padded_mask = Image.new(
            "L",
            (target_w, target_h),
            0
        )

        # Place resized image and mask.
        padded_image.paste(image, (left, top))
        padded_mask.paste(mask, (left, top))

        return padded_image, padded_mask

    def __getitem__(self, index):
        """
        Load and process one image-mask pair.
        """

        sample = self.data[index]

        # 1. Read image and mask.
        image = sample["image"].convert("RGB")
        mask = sample["mask"].convert("L")

        image_id = str(sample["image_id"])

        # 2. Verify spatial alignment before processing.
        if image.size != mask.size:
            raise ValueError(
                f"Image-mask size mismatch: {image_id}"
            )

        # 3. Resize and pad.
        image, mask = self._resize_and_pad(
            image, mask
        )

        # 4. Convert image to NumPy.
        image_array = np.asarray(
            image,
            dtype=np.float32
        ).copy()

        image_array /= 255.0

        # 5. Convert mask to binary values.
        mask_array = np.asarray(mask).copy()

        if mask_array.max() <= 1:
            binary_mask = mask_array > 0
        else:
            binary_mask = mask_array >= 128

        mask_array = binary_mask.astype(np.float32)

        # 6. Convert NumPy arrays to PyTorch tensors.
        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1).contiguous()

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0).contiguous()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": image_id
        }
