
# ComfyUI-IF_Trellis

ComfyUI TRELLIS is a large 3D asset generation in various formats, such as Radiance Fields, 3D Gaussians, and meshes. The cornerstone of TRELLIS is a unified Structured LATent (SLAT) representation that allows decoding to different output formats and Rectified Flow Transformers tailored for SLAT as the powerful backbones. 

![teaser (1)](https://github.com/user-attachments/assets/6eee56bd-0936-44a5-b843-be4e87be649f)

# Installation 
ONLY tested on windows but it should work easier in Linux without any issues or needing such specific stuffs.

You need to set up the environment first
follow this guide for the first part

Set the VSCode Cpp Envirronment as in the guide

https://ko-fi.com/post/Installing-Triton-and-Sage-Attention-Flash-Attenti-P5P8175434

Setting up ComfyUI with the Xformers, flash attention, Sage-attention(Optional Recommended for Hunyuan and other Video models)

<!-- Installation -->
## 📦 Second part of the installation

Activate youur comfy environment
```
(gen) PS D:\ComfyUI\custom_nodes\ComfyUI-IF_Trellis> micromamba activate gen
```
If you haven't set your vars or for some reason it can't compile some of this specially `nvdiffrast`
it doesn't hurt if you do it again now.
```
cmd.exe /c "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 "&&" powershell
```
You will see some message like this:

**********************************************************************
** Visual Studio 2019 Developer Command Prompt v16.11.41
** Copyright (c) 2021 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

```
pip install -r requirements.txt
```

```bash
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
New-Item -ItemType Directory -Force -Path C:\tmp\extensions
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git C:\tmp\extensions\diffoctreerast
pip install C:\tmp\extensions\diffoctreerast
git clone https://github.com/autonomousvision/mip-splatting.git C:\tmp\extensions\mip-splatting
pip install C:\tmp\extensions\mip-splatting\submodules\diff-gaussian-rasterization\
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
git clone https://github.com/NVlabs/nvdiffrast.git C:\tmp\extensions\nvdiffrast
pip install C:\tmp\extensions\nvdiffrast
```

### Prerequisites
- **System**: The code is currently tested only on **Linux**.  For windows setup, you may refer to [#3](https://github.com/microsoft/TRELLIS/issues/3) (not fully tested).
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A100 and A6000 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  This repo use **CUDA 12.4**.
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

  Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

### Installation Steps
1. Clone the repo:
    ```
    git clone --recurse-submodules https://github.com/sdbds/TRELLIS-for-windows.git
    ```
## MUST HAVE `--recurse-submodules`
![blender_Ny9GX9TsN1](https://github.com/user-attachments/assets/ac0d8c65-a9f4-4277-8453-ee62a9f7aef9)
![thorium_lou9h3r4uF](https://github.com/user-attachments/assets/3e81cc5e-898b-48ff-8ab2-d6394cd1ff8f)


## 🌟 Features
- **High Quality**: It produces diverse 3D assets at high quality with intricate shape and texture details.
- **Versatility**: It takes text or image prompts and can generate various final 3D representations including but not limited to *Radiance Fields*, *3D Gaussians*, and *meshes*, accommodating diverse downstream requirements.
- **Flexible Editing**: It allows for easy editings of generated 3D assets, such as generating variants of the same object or local editing of the 3D asset.

<!-- TODO List -->
## 🚧 TODO List
- [x] Release comfyUI-IF_Trellis
- [ ] 3D Viewport 
- [ ] OPT mode
- [ ] Fix BUGS
- [ ] Installation
