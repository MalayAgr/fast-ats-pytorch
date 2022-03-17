import torch

import extract_patches

img = torch.rand(1, 100, 100, 3, dtype=torch.float32, device="cuda")
offsets = torch.zeros(1, 10, 2, dtype=torch.float32, device="cuda")

print(img)

patches = extract_patches.extract_patches(img, offsets, (6, 6))

print(patches)
