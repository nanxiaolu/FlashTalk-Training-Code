# 模型权重准备

本项目需要依赖多个基础模型和组件进行特征提取、训练和评估。请将下载的权重放置在项目根目录的 `weights/` 文件夹下。

| 模型/组件名称 | 作用简述 | 下载链接 |
|---|---|---|
| **InfiniteTalk Base** | Stage 1 的初始生成器权重。 | [暂未提供](#) |
| **Wan2.1-I2V-14B-480P** | 提供 VAE 以及文本/视觉编码器 (T5, CLIP 等)。 | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| **chinese-wav2vec2-base** | 用于提取音频特征的网络。 | [HuggingFace](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |
| **InsightFace Models** | 包含脸部检测、关键点、身份提取等模型，用于特征处理。 | [暂未提供](#) |
| **Stage 1 & 2 Checkpoints** | 我们训练好的第一、二阶段权重，供快速体验、参考对比或跳过训练使用。 | [暂未提供](#) |
| ***(可选)* Eval 模型** | 包含 face_det, syncnet, q-align 等，**仅在运行指标评估代码时需要**，普通训练无需下载。 | [暂未提供](#) |

## 目录结构预期

下载并解压所有文件后，您的 `weights/` 目录结构应该如下所示（未使用的可选模型可以不存在）：

```
weights/
├── InfiniteTalk/
│   └── single/
│       └── infinitetalk.safetensors                        # 基础 InfiniteTalk DiT
├── Wan2.1-I2V-14B-480P/                                    # HuggingFace 下载的文件夹
│   ├── Wan2.1_VAE.pth
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── diffusion_pytorch_model-0000{1..7}-of-00007.safetensors
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── google/umt5-xxl/                                    # T5 tokenizer
│   └── xlm-roberta-large/                                  # CLIP-XLMR tokenizer
├── chinese-wav2vec2-base/
│   ├── chinese-wav2vec2-base-fairseq-ckpt.pt               # 音频编码器
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── insightface/
│   └── models/antelopev2/                                  # 脸部检测 / id / 关键点
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── genderage.onnx
│       ├── glintr100.onnx
│       └── scrfd_10g_bnkps.onnx
├── rvm_model/
│   └── rvm_mobilenetv3.pth                                 # 背景稳定性过滤器
├── face_det/                                               # (可选) 评估时的人脸裁剪
│   ├── detection_Resnet50_Final.pth
│   ├── detection_mobilenet0.25_Final.pth
│   ├── parsing_bisenet.pth
│   └── parsing_parsenet.pth
├── syncnet/                                                # (可选) 评估时的 SyncNet
│   ├── sfd_face.pth
│   └── syncnet_v2.model
└── q-align/                                                # (可选) 评估时的 Q-Align
    ├── config.json
    ├── pytorch_model-0000{1,2}-of-00002.bin
    ├── pytorch_model.bin.index.json
    └── tokenizer.model
```
