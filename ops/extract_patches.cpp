#include <iostream>
#include <torch/extension.h>

torch::Tensor extract_patches_cpu(torch::Tensor img, torch::Tensor offsets, c10::ArrayRef<long int> patch_size);

#ifdef USE_CUDA

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

torch::Tensor extract_patches_cuda(torch::Tensor img, torch::Tensor offsets, c10::ArrayRef<long int> patch_size);

torch::Tensor extract_patches(torch::Tensor img, torch::Tensor offsets, c10::ArrayRef<long int> patch_size)
{
    auto cuda_available = torch::cuda::is_available();

    if (!cuda_available)
        return extract_patches_cpu(img, offsets, patch_size);

    CHECK_INPUT(img);
    CHECK_INPUT(offsets);

    return extract_patches_cuda(img, offsets, patch_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("extract_patches", &extract_patches, "Patch extractor");
}

#else

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("extract_patches", &extract_patches_cpu, "Patch extractor");
}

#endif
