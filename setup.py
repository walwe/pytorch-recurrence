from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from warnings import warn

# TODO: always returns false
if True or torch.cuda.is_available():
    ext_modules = [
        CUDAExtension('recurrence_matrix', [
            'src/recurrence.cpp',
            'src/recurrence_kernel_gpu.cu',
        ])
    ]
else:
    warn("CUDA not found: Using CPU version")
    ext_modules = [
        CppExtension('recurrence_matrix', [
            'src/recurrence.cpp',
            'src/recurrence_kernel_cpu.cpp',
        ])
    ]

setup(
    name='recurrence_matrix',
    version='1.0.0',
    ext_modules=ext_modules,
    install_requires=[
        'torch'
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })
