# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import nvdiffrast
import setuptools
import os
import platform
import torch
import logging
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

logger = logging.getLogger(__name__)

def check_cuda_availability():
    """Check if CUDA is available and properly configured"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available - falling back to CPU")
        return False
        
    # Check CUDA_HOME without requiring it
    cuda_home = os.environ.get('CUDA_HOME', None)
    if cuda_home is None:
        if platform.system() == "Linux":
            default_cuda_path = "/usr/local/cuda"
            if os.path.exists(default_cuda_path):
                os.environ["CUDA_HOME"] = default_cuda_path
                logger.info(f"Set CUDA_HOME to default path: {default_cuda_path}")
                return True
            else:
                logger.warning(f"Default CUDA path {default_cuda_path} not found")
                return False
        elif platform.system() == "Windows":
            # Try common Windows CUDA paths
            cuda_versions = ["12.4", "12.3", "12.2", "12.1", "12.0", 
                           "11.8", "11.7", "11.6", "11.5", "11.4", "11.3", "11.2"]
            base_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
            
            for version in cuda_versions:
                cuda_path = os.path.join(base_path, f"v{version}")
                if os.path.exists(cuda_path):
                    os.environ["CUDA_HOME"] = cuda_path
                    logger.info(f"Set CUDA_HOME to Windows default path: {cuda_path}")
                    return True
            
            logger.warning("Could not find CUDA installation in default Windows paths")
            return False
            
    return True

def get_cuda_version():
    """Get CUDA version from nvcc if available"""
    cuda_home = os.environ.get('CUDA_HOME', None)
    if not cuda_home:
        return None
        
    try:
        import subprocess
        nvcc_path = os.path.join(cuda_home, "bin/nvcc")
        if not os.path.exists(nvcc_path):
            return None
            
        nvcc_out = subprocess.check_output([nvcc_path, "--version"]).decode()
        version_line = [l for l in nvcc_out.split('\n') if "release" in l][0]
        version = version_line.split("release")[-1].strip().split(",")[0].split("V")[-1]
        return version
    except:
        return None

# Check CUDA availability before proceeding
cuda_available = check_cuda_availability()
cuda_version = get_cuda_version()

if cuda_available and cuda_version:
    logger.info(f"CUDA {cuda_version} is available")
    ext_modules = [
        CUDAExtension(
            name="nvdiffrast.torch._C",
            sources=[
                'nvdiffrast/common/cudaraster/impl/Buffer.cpp',
                'nvdiffrast/common/cudaraster/impl/CudaRaster.cpp',
                'nvdiffrast/common/cudaraster/impl/RasterImpl_.cu',
                'nvdiffrast/common/cudaraster/impl/RasterImpl.cpp',
                'nvdiffrast/common/common.cpp',
                'nvdiffrast/common/rasterize.cu',
                'nvdiffrast/common/interpolate.cu',
                'nvdiffrast/common/texture_.cu',
                'nvdiffrast/common/texture.cpp',
                'nvdiffrast/common/antialias.cu',
                'nvdiffrast/torch/torch_bindings.cpp',
                'nvdiffrast/torch/torch_rasterize.cpp',
                'nvdiffrast/torch/torch_interpolate.cpp',
                'nvdiffrast/torch/torch_texture.cpp',
                'nvdiffrast/torch/torch_antialias.cpp',
            ],
            extra_compile_args={
                'cxx': ['-DNVDR_TORCH'],
                'nvcc': ['-DNVDR_TORCH', '-lineinfo'],
            },
        )
    ]
    cmdclass = {'build_ext': BuildExtension}
else:
    logger.warning("CUDA not available - building without CUDA extensions")
    ext_modules = []
    cmdclass = {}

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvdiffrast",
    version=nvdiffrast.__version__,
    author="Samuli Laine",
    author_email="slaine@nvidia.com",
    description="nvdiffrast - modular primitives for high-performance differentiable rendering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/nvdiffrast",
    packages=setuptools.find_packages(),
    # package_data={
    #     'nvdiffrast': [
    #         'common/*.h',
    #         'common/*.inl',
    #         'common/*.cu',
    #         'common/*.cpp',
    #         'common/cudaraster/*.hpp',
    #         'common/cudaraster/impl/*.cpp',
    #         'common/cudaraster/impl/*.hpp',
    #         'common/cudaraster/impl/*.inl',
    #         'common/cudaraster/impl/*.cu',
    #         'lib/*.h',
    #         'torch/*.h',
    #         'torch/*.inl',
    #         'torch/*.cpp',
    #         'tensorflow/*.cu',
    #     ] + (['lib/*.lib'] if os.name == 'nt' else [])
    # },
    # include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=['numpy'],  # note: can't require torch here as it will install torch even for a TensorFlow container
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
