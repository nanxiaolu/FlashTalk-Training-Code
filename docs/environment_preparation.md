# 环境配置

本指南给出本仓库在生产环境中实际使用的依赖组合，建议所有库严格按照以下的版本安装，本仓库调试环境如下。
| 组件 | 版本 |
|---|---|
| OS | Linux x86_64 |
| CUDA driver | >= 12.1 |
| Python | 3.10.18 |
| PyTorch | 2.4.1 + cu121 |
| 硬件 | 8 × A800|

## Step 1 — 创建 conda 环境

```bash
conda create -n flashtalk_train python=3.10.18 -y
conda activate flashtalk_train
```

## Step 2 — 安装 PyTorch（cu121）

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

## Step 3 — 安装项目依赖

```bash
pip install -r requirements.txt
```

## Step 4 — 单独安装 flash-attn（**必装**）
```bash
# A 系列（A100 / A800 / A40 / A30 / A6000 ...）：使用 flash-attn v2
pip install flash-attn==2.7.4.post1 --no-build-isolation
# H 系列（H100 / H800 / H200 ...）：使用 flash-attn v3 性能更佳.
# 如果pip在线安装不了，可以到“https://github.com/Dao-AILab/flash-attention/releases?page=3”选择适用于你环境的版本下载到服务器，再pip离线install
```

## Step 5 — FFmpeg installation

```bash
# Ubuntu / Debian
apt-get install ffmpeg
```
or
```bash
# Conda (no root required) 
conda install -c conda-forge ffmpeg==7
```