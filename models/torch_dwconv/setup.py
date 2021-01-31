from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torch_dwconv',
    ext_modules=[
        CUDAExtension('torch_dwconv_C',
                      ['dwconv.cpp', 'dwconv_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)