import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

USE_CUDA = os.environ.get("USE_CUDA", "True") == "True"

source_files = [
    os.path.join("ops", "extract_patches.cpp"),
    os.path.join("ops", "extract_patches_cpu.cpp"),
]

with open("README.md", "r") as f:
    long_description = f.read()


def launch_setup():
    if USE_CUDA is True:
        Extension = CUDAExtension
        kernel = os.path.join("ops", "extract_patches_kernel.cu")
        source_files.append(kernel)
        macro = [("USE_CUDA", None)]
    else:
        Extension = CppExtension
        macro = []

    setup(
        name="ats",
        version="0.3.0",
        author="Malay Agarwal",
        author_email="malay.agarwal261016@outlook.com",
        description="Fast Attention Sampling For PyTorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/MalayAgr/fast-ats-pytorch",
        install_requires=["torch>=1.11.0"],
        ext_modules=[
            Extension(
                name="ats.ops",
                sources=source_files,
                define_macros=macro,
            )
        ],
        package_dir={"": "src"},
        packages=["ats"],
        cmdclass={"build_ext": BuildExtension},
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    launch_setup()
