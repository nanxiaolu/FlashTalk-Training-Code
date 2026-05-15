# 模型权重准备

请将下载的权重统一放置在仓库根目录的 `weights/` 文件夹下，根据自己需要下载模型权重。

## 1. 权重与模型一览

| 模型目录名 | 说明 | 下载来源 |
|---|---|---|
| chinese-wav2vec2-base | （可选）运行 Stage 1/2 特征提取过程时需要，否则不需要。提取音频特征 | [HuggingFace](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |
| face_det | （可选）运行 Stage 1 特征提取过程时需要，否则不需要。配合 insightface 计算脸部区域 mask | [Resnet50](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth), [bisenet](https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth) |
| **InfiniteTalk** | **（必选）用于 Stage 1/2 训练。语音相关权重，只需要下载其中的 single 文件夹** | [HuggingFace](https://huggingface.co/MeiGen-AI/InfiniteTalk/tree/main) |
| insightface | （可选）运行 Stage 1 特征提取过程时需要，否则不需要。脸部检测模型 | [HuggingFace](https://huggingface.co/FrancisRing/StableAnimator/tree/main/models) |
| q-align | （可选）运行 Stage 1/2 val 之后的 eval 时需要，否则不需要 | [HuggingFace](https://huggingface.co/q-future/one-align/) |
| **rvm_model** | **（必选）用于 Stage 1 特征提取过程以及 Stage 2 训练。人像分割模型，第一阶段用于筛选数据的背景稳定性（默认不开启），第二阶段用于计算背景一致性loss（提高模型的背景一致性能力）** | [GitHub Release](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth) |
| syncnet | （可选）运行 Stage 1/2 val 之后的 eval 时需要，否则不需要 | [HuggingFace](https://huggingface.co/lithiumice/syncnet) |
| **Wan2.1-I2V-14B-480P** | **（必选）用于 Stage 1/2 训练。包括 VAE 与文本/视觉编码器 (T5、CLIP)，预训练 DiT** | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Ours Stage 1 & Stage 2 训练 ckpt |（可选）我们训练好的两个阶段权重，供快速验证/对比 | [Stage1](https://pan.baidu.com/s/1PNg-QS61aV0pbD1oiGPjxQ?pwd=1426) <br>[Stage2](https://pan.baidu.com/s/1mludcQgg7w3Z014gDYvxPg?pwd=0960) |

## 2. 目录结构预期

下载并解压所有必需文件后，仓库根目录下 `weights/` 的预期结构如下（可选模型缺失不影响训练）：

```
weights/
├── chinese-wav2vec2-base/
│   ├── chinese-wav2vec2-base-fairseq-ckpt.pt               # 音频编码器
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── face_det/                                               # 仅评估时
│   ├── detection_Resnet50_Final.pth
│   └── parsing_bisenet.pth
├── InfiniteTalk/
│   └── single/
│       └── infinitetalk.safetensors                        # 基础 InfiniteTalk DiT
├── insightface/
│   └── models/antelopev2/                                  # 脸部检测 / id / 关键点
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── genderage.onnx
│       ├── glintr100.onnx
│       └── scrfd_10g_bnkps.onnx
├── q-align/                                                # 仅评估时
│   ├── config.json
│   ├── configuration_mplug_owl2.py
|   └── (...other files...)
├── rvm_model/
│   └── rvm_mobilenetv3.pth                                 # 背景稳定性过滤器
├── syncnet/                                                # 仅评估时
│   ├── sfd_face.pth
│   └── syncnet_v2.model
└── Wan2.1-I2V-14B-480P/                                    # HuggingFace 下载的整个目录
    ├── google/umt5-xxl/                                    # T5 tokenizer
    ├── xlm-roberta-large/                                  # CLIP-XLMR tokenizer
    ├── config.json
    ├── diffusion_pytorch_model-0000{1..7}-of-00007.safetensors
    ├── diffusion_pytorch_model.safetensors.index.json
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── Wan2.1_VAE.pth
```

## 3. 如果使用我们训练好的 Checkpoint(可选)

假设模型放在 `weights/flashtalk_reproduce/`：

```
weights/flashtalk_reproduce/
├── flashtalk_stage1.safetensors    # Stage 1 训练终点权重
└── flashtalk_stage2.safetensors    # Stage 2 训练终点权重
```

* **用 Stage 1 ckpt 作为 Stage 2 训练起点**：把路径写到 `config/train_stage2.yaml` 的 `init_stage1_full`。
* **用 Stage 2 ckpt 直接验证**：把路径写到 `config/val_stage2.yaml` 的 `resume`。
* **用 Stage 2 ckpt 部署推理**：先用 `tools/export_stage2_model_to_flashtalk_style.py` 转成 Diffusers 分片格式，再用 SoulX-FlashTalk 推理（详见 [推理章节](train_val_inference.md#5-推理-inference)）。
