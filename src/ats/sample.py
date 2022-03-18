import torch
from ats import ops


def extract_patches(
    img: torch.Tensor, offsets: torch.Tensor, patch_size
) -> torch.Tensor:
    return ops.extract_patches(img, offsets, patch_size)
