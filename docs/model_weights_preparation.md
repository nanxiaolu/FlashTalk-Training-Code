# Model Weights Preparation

[Chinese version](model_weights_preparation-zh-CN.md)

Place all downloaded weights under the `weights/` directory at the repository root. Download the models you need for your workflow.

## 1. Weights and models

| Model directory | Description | Source |
|---|---|---|
| chinese-wav2vec2-base | Optional. Required only for Stage 1/2 feature extraction. Used for audio feature extraction. | [HuggingFace](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |
| face_det | Optional. Required only for Stage 1 feature extraction. Used with insightface to compute face-region masks. | [Resnet50](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth), [bisenet](https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth) |
| **InfiniteTalk** | **Required for Stage 1/2 training. For audio-related weights, only the `single` folder is needed.** | [HuggingFace](https://huggingface.co/MeiGen-AI/InfiniteTalk/tree/main) |
| insightface | Optional. Required only for Stage 1 feature extraction. Used for face detection. | [HuggingFace](https://huggingface.co/FrancisRing/StableAnimator/tree/main/models) |
| q-align | Optional. Required only for evaluation after Stage 1/2 validation. | [HuggingFace](https://huggingface.co/q-future/one-align/) |
| **rvm_model** | **Required for Stage 1 feature extraction and Stage 2 training. This portrait matting model is used in Stage 1 to filter samples by background stability (disabled by default), and in Stage 2 to compute the background-consistency loss.** | [GitHub Release](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth) |
| syncnet | Optional. Required only for evaluation after Stage 1/2 validation. | [HuggingFace](https://huggingface.co/lithiumice/syncnet) |
| **Wan2.1-I2V-14B-480P** | **Required for Stage 1/2 training. Includes the VAE, text/vision encoders (T5 and CLIP), and pretrained DiT.** | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Our Stage 1 & Stage 2 training checkpoints | Optional. Our trained two-stage weights for quick validation, comparison, or fine-tuning. | [Stage1](https://modelscope.cn/models/youngsx/FlashTalk_Reproduction/files) <br>[Stage2](https://modelscope.cn/models/youngsx/FlashTalk_Reproduction/files) |

## 2. Expected directory structure

After downloading and extracting all required files, the expected `weights/` structure is shown below. Missing optional models do not affect training.

```
weights/
├── chinese-wav2vec2-base/
│   ├── chinese-wav2vec2-base-fairseq-ckpt.pt               # Audio encoder
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── face_det/                                               # Evaluation only
│   ├── detection_Resnet50_Final.pth
│   └── parsing_bisenet.pth
├── InfiniteTalk/
│   └── single/
│       └── infinitetalk.safetensors                        # Base InfiniteTalk DiT
├── insightface/
│   └── models/antelopev2/                                  # Face detection / ID / landmarks
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── genderage.onnx
│       ├── glintr100.onnx
│       └── scrfd_10g_bnkps.onnx
├── q-align/                                                # Evaluation only
│   ├── config.json
│   ├── configuration_mplug_owl2.py
|   └── (...other files...)
├── rvm_model/
│   └── rvm_mobilenetv3.pth                                 # Background-stability filter
├── syncnet/                                                # Evaluation only
│   ├── sfd_face.pth
│   └── syncnet_v2.model
└── Wan2.1-I2V-14B-480P/                                    # Full HuggingFace directory
    ├── google/umt5-xxl/                                    # T5 tokenizer
    ├── xlm-roberta-large/                                  # CLIP-XLMR tokenizer
    ├── config.json
    ├── diffusion_pytorch_model-0000{1..7}-of-00007.safetensors
    ├── diffusion_pytorch_model.safetensors.index.json
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── Wan2.1_VAE.pth
```

## 3. Using our trained checkpoints (optional)

Assume the checkpoints are placed in `weights/flashtalk_reproduce/`:

```
weights/flashtalk_reproduce/
├── flashtalk_stage1.safetensors    # Final Stage 1 checkpoint
└── flashtalk_stage2.safetensors    # Final Stage 2 checkpoint
```

* **Use the Stage 1 checkpoint as the Stage 2 initialization**: set `init_stage1_full` in `config/train_stage2.yaml`.
* **Validate directly with the Stage 2 checkpoint**: set `resume` in `config/val_stage2.yaml`.
* **Deploy the Stage 2 checkpoint for inference**: first convert it to Diffusers sharded format with `tools/export_stage2_model_to_flashtalk_style.py`, then run inference with SoulX-FlashTalk. See [Inference](train_val_inference.md#5-inference).
