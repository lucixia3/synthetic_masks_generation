import os
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset


def _load_image(path: str, mode: Optional[str] = None) -> Image.Image:
    image = Image.open(path)
    if mode is not None:
        image = image.convert(mode)
    return image


class MaskDataset(Dataset):
    """Dataset that loads segmentation masks only.

    Masks are expected to be single-channel images with integer labels per pixel.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.root = root
        self.paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        if not self.paths:
            raise ValueError(f"No mask files found in {root}")
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = _load_image(self.paths[idx], mode="L")
        if self.transforms:
            return self.transforms(image)
        return torch.as_tensor(image, dtype=torch.long)


class PairedMaskCTDataset(Dataset):
    """Dataset for paired mask and CT images stored in separate directories.

    Expected directory structure:
    - mask_root/
        mask_000.png
        ...
    - ct_root/
        mask_000.png  # matching filename for CT slice
    """

    def __init__(
        self,
        mask_root: str,
        ct_root: str,
        mask_transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        ct_transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.mask_root = mask_root
        self.ct_root = ct_root
        self.mask_paths = sorted(
            [p for p in os.listdir(mask_root) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        if not self.mask_paths:
            raise ValueError(f"No mask files found in {mask_root}")
        self.mask_transforms = mask_transforms
        self.ct_transforms = ct_transforms

    def __len__(self) -> int:
        return len(self.mask_paths)

    def __getitem__(self, idx: int):
        name = self.mask_paths[idx]
        mask_image = _load_image(os.path.join(self.mask_root, name), mode="L")
        ct_image = _load_image(os.path.join(self.ct_root, name), mode="L")

        if self.mask_transforms:
            mask = self.mask_transforms(mask_image)
        else:
            mask = torch.as_tensor(mask_image, dtype=torch.long)

        if self.ct_transforms:
            ct = self.ct_transforms(ct_image)
        else:
            ct = torch.as_tensor(ct_image, dtype=torch.float32) / 255.0

        return mask, ct
