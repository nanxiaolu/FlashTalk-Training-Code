# 环境配置

本指南给出本仓库在生产环境中实际使用的依赖组合，建议所有库严格按照以下的版本安装，本仓库调试环境如下。
| 组件 | 版本 |
|---|---|
| OS | Linux x86_64 |
| CUDA driver | >= 12.1 |
| Python | 3.10.18 |
| PyTorch | 2.4.1 + cu121 |
| 硬件 | 8 × A800|

## 步骤 1：创建 conda 环境

```bash
conda create -n flashtalk_train python=3.10.18 -y
conda activate flashtalk_train
```

## 步骤 2：安装 PyTorch（cu121）

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

## 步骤 3：安装项目依赖

```bash
pip install -r requirements.txt
```

## 步骤 4：单独安装 flash-attn（**必装**）
```bash
# A 系列（A100 / A800 / A40 / A30 / A6000 ...）：使用 flash-attn v2
pip install flash-attn==2.7.4.post1 --no-build-isolation
# H 系列（H100 / H800 / H200 …）：使用 flash-attn v3 性能更佳。
# 若 pip 在线安装失败，可到「https://github.com/Dao-AILab/flash-attention/releases?page=3」下载适配本机的预编译 wheel，再在服务器上 pip 离线安装。
```

## 步骤 5：安装 FFmpeg

```bash
# Ubuntu / Debian
apt-get install ffmpeg
```

或

```bash
# Conda（无需 root）
conda install -c conda-forge ffmpeg==7
```
