# Environment Preparation

[Chinese version](environment_preparation-zh-CN.md)

This guide lists the dependency set used in our production training environment. We recommend installing the following versions exactly.

| Component | Version |
|---|---|
| OS | Linux x86_64 |
| CUDA driver | >= 12.1 |
| Python | 3.10.18 |
| PyTorch | 2.4.1 + cu121 |
| Hardware | 8 x A800 |

## Step 1 - Create a conda environment

```bash
conda create -n flashtalk_train python=3.10.18 -y
conda activate flashtalk_train
```

## Step 2 - Install PyTorch (cu121)

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

## Step 3 - Install project dependencies

```bash
pip install -r requirements.txt
```

## Step 4 - Install flash-attn separately (**required**)

```bash
# A-series GPUs (A100 / A800 / A40 / A30 / A6000 ...): use flash-attn v2.
pip install flash-attn==2.7.4.post1 --no-build-isolation
# H-series GPUs (H100 / H800 / H200 ...): flash-attn v3 usually performs better.
# If online pip installation fails, download a wheel matching your environment from
# https://github.com/Dao-AILab/flash-attention/releases?page=3 and install it offline.
```

## Step 5 - Install FFmpeg

```bash
# Ubuntu / Debian
apt-get install ffmpeg
```

or

```bash
# Conda (no root required)
conda install -c conda-forge ffmpeg==7
```