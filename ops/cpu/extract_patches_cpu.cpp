#include <torch/extension.h>

#include <iostream>

using namespace torch::indexing;

torch::Tensor extract_patches_cpu(torch::Tensor img, torch::Tensor offsets, c10::ArrayRef<long int> patch_size)
{
    img = img.permute({0, 3, 1, 2});

    auto pad_const = (int)(patch_size[0] / 2.0);
    img = torch::constant_pad_nd(img, {pad_const, pad_const, pad_const, pad_const}, 0.0);

    offsets = offsets + pad_const;

    auto patch_H = patch_size[0];
    auto patch_W = patch_size[1];

    auto offset_acc = offsets.accessor<float, 3>();

    std::vector<torch::Tensor> patches;

    at::parallel_for(0, img.size(0), 0, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; b++)
        {
            for (int64_t n = 0; n < offset_acc.size(1); n++)
            {
                auto x_start = (int)(offset_acc[b][n][0]);
                auto x_end = x_start + patch_H;
                auto y_start = (int)(offset_acc[b][n][1]);
                auto y_end = y_start + patch_W;

                auto patch = img.index({b, Slice(), Slice(x_start, x_end), Slice(y_start, y_end)});

                patches.push_back(patch);
            }
        }
    });

    return torch::stack(patches);
}
