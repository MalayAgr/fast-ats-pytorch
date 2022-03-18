import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

DEBUG = os.environ.get("ADD_DEBUG_FLAGS", "False") == "True"

extra_compile_args = {"nvcc": ["-g", "-G"], "cxx": ["-g"]} if DEBUG else {}


setup(
    name="ops",
    ext_modules=[
        CUDAExtension(
            name="extract_patches",
            sources=[
                "extract_patches.cpp",
                "extract_patches_kernel.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
