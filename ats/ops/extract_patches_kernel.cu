#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void extract_patches_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> img,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> offsets, const int64_t batch_size,
    int64_t n_samples, int64_t channels, const int64_t patch_H, const int64_t patch_W,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> patches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int b = idx / (n_samples * channels);
    const int n = (idx % (n_samples * channels)) / channels;
    const int c = idx - n * channels - b * n_samples * channels;

    if ((b >= batch_size) || (n >= n_samples) || (c >= channels))
        return;

    auto patch_idx = (b + 1) * n;

    printf("%l %l %l %l", b, n, c, patc_idx);

    auto x_start = (int)(offsets[b][n][0]);
    auto x_end = x_start + patch_H;
    auto y_start = (int)(offsets[b][n][1]);
    auto y_end = y_start + patch_W;

    for (auto i = x_start; i < x_end; i++)
    {
        for (auto j = y_start; j < y_end; j++)
        {
            patches[patch_idx][c][i][j] += img[b][c][i][j];
        }
    }
}

torch::Tensor extract_patches_cuda(torch::Tensor img, torch::Tensor offsets, c10::ArrayRef<long int> patch_size)
{
    img = img.permute({0, 3, 1, 2});

    auto pad_const = (int)(patch_size[0] / 2.0);
    img = torch::constant_pad_nd(img, {pad_const, pad_const, pad_const, pad_const}, 0.0);

    offsets = offsets + pad_const;

    auto patch_H = patch_size[0];
    auto patch_W = patch_size[1];

    auto offset_acc = offsets.accessor<float, 3>();

    auto batch_size = img.size(0);
    auto channels = img.size(1);
    auto n_samples = offsets.size(1);

    auto patches = torch::zeros({batch_size * n_samples, channels, patch_H, patch_W});

    const int n = batch_size * n_samples * channels;
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(img.type(), "extract_patch", ([&] {
                                   extract_patches_kernel<scalar_t><<<blocks, threads>>>(
                                       img.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                       offsets.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       batch_size, channels, n_samples, patch_H,
                                       patch_W, patches.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
                               }));

    return torch::stack(patches);
}
