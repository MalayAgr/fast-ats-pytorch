from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    name="extract_patches",
    ext_modules=[
        cpp_extension.CppExtension("extract_patches", ["extract_patches.cpp"])
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
