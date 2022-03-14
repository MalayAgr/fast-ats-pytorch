#include <torch/extension.h>

#include <iostream>

using namespace torch::indexing;

torch::Tensor extract_patches(
    torch::Tensor img,
    torch::Tensor offsets,
    c10::ArrayRef<long int> patch_size)
{
    img = img.permute({0, 3, 1, 2});

    auto pad_const = (int)(patch_size[0] / 2.0);
    img = torch::constant_pad_nd(img, {pad_const, pad_const, pad_const, pad_const}, 0.0);

    offsets = offsets + pad_const;

    auto patch_H = patch_size[0];
    auto patch_W = patch_size[1];

    auto offset_acc = offsets.accessor<float, 3>();

    std::vector<torch::Tensor> patches;

    for (int b = 0; b < img.size(0); b++)
    {
        for (int n = 0; n < offset_acc.size(1); n++)
        {
            auto x_start = (int)(offset_acc[b][n][0]);
            auto x_end = x_start + patch_H;
            auto y_start = (int)(offset_acc[b][n][1]);
            auto y_end = y_start + patch_W;

            auto patch = img.index({b, Slice(), Slice(x_start, x_end), Slice(y_start, y_end)});

            patches.push_back(patch);
        }
    }

    return torch::stack(patches);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("extract_patches", &extract_patches, "Patch extractor");
}
