#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void extract_patches_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> img,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> offsets, const int64_t batch_size,
    const int64_t n_samples, const int64_t channels, const int64_t patch_H, const int64_t patch_W,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> patches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int b = idx / (n_samples * channels);
    const int n = (idx % (n_samples * channels)) / channels;
    const int c = idx - n * channels - b * n_samples * channels;

    if ((b >= batch_size) || (n >= n_samples) || (c >= channels))
        return;

    const int patch_idx = (b * n_samples) + n;


    auto x_start = (int)(offsets[b][n][0]);
    auto x_end = x_start + patch_H;
    auto y_start = (int)(offsets[b][n][1]);
    auto y_end = y_start + patch_W;

    for (auto img_i = x_start, patch_i = 0; img_i < x_end; img_i++, patch_i++)
        for (auto img_j = y_start, patch_j = 0; img_j < y_end; img_j++, patch_j++)
            patches[patch_idx][c][patch_i][patch_j] = img[b][c][img_i][img_j];
}

torch::Tensor extract_patches_cuda(torch::Tensor img, torch::Tensor offsets, c10::ArrayRef<long int> patch_size)
{
    img = img.permute({0, 3, 1, 2});

    auto batch_size = img.size(0);
    auto channels = img.size(1);
    auto n_samples = offsets.size(1);

    auto pad_const = (int)(patch_size[0] / 2.0);
    img = torch::constant_pad_nd(img, {pad_const, pad_const, pad_const, pad_const}, 0.0);

    offsets = offsets + pad_const;

    auto patch_H = patch_size[0];
    auto patch_W = patch_size[1];

    auto options = torch::TensorOptions().device(torch::kCUDA, 0);

    auto patches = torch::zeros({batch_size * n_samples, channels, patch_H, patch_W}, options);

    const int n = batch_size * n_samples * channels;
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(img.scalar_type(), "extract_patch", ([&] {
                                   extract_patches_kernel<scalar_t><<<blocks, threads>>>(
                                       img.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                                       offsets.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                       batch_size, n_samples, channels, patch_H,
                                       patch_W, patches.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
                               }));

    return patches;
}
