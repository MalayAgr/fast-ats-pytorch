import torch

import extract_patches

img = torch.rand(16, 4000, 4000, 3, dtype=torch.float32, device="cuda")
offsets = torch.zeros(16, 10, 2, dtype=torch.float32, device="cuda")

print(img)

patches = extract_patches.extract_patches(img, offsets, (6, 6))

print(patches)
