from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='adaptive_conv_cuda',
    ext_modules=[
        CUDAExtension('torch_dwconv_CC',
                      ['dwconv.cpp', 'dwconv_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)