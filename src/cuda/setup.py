from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="paged_attn",
    ext_modules=[
        CUDAExtension(
            name="paged_attn",
            sources=["decode.cu", "prefill.cu"],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2'],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
