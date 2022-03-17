from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ops",
    ext_modules=[
        CUDAExtension(
            name="extract_patches",
            sources=[
                "extract_patches.cpp",
                "extract_patches_kernel.cu",
            ],
            extra_compile_args={"nvcc": ["-g", "-G"], "cxx": ["-g"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
