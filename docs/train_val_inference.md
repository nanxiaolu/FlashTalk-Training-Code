# Training, Validation, and Inference Guide

[Chinese version](train_val_inference-zh-CN.md)

This guide covers Stage 1 / Stage 2 training and validation, plus final inference deployment. All commands assume 8 GPUs by default.

> **Prerequisites**: complete the following before starting:
>
> 1. [Environment preparation](environment_preparation.md)
> 2. [Data preparation](data_preparation.md)
> 3. [Model weights preparation](model_weights_preparation.md)

> **GPU-count reminder**: the default configuration targets 8 GPUs. For any other GPU count, read [hardware_scaling.md](hardware_scaling.md) first.

---

## 1. Stage 1 training

**Goal**: adapt the base model, InfiniteTalk, from talking-head-only data to a data distribution that includes gestures and upper-body motion.

**Config file**: `config/train_stage1.yaml`.  
Update `lmdb_path` in `config/train_stage1.yaml`. By default, it points to the pre-extracted features we provide.

> **Note**: Because our dataset and some hyperparameters differ from the FlashTalk paper, the required iteration count is also different. In this repository, `max_steps = 500` is enough for the best result, while the original FlashTalk paper uses 1,000 iterations.

**Launch**:

```bash
bash script/train_stage1.sh
```

**Outputs**: saved to `outputs/flashtalk_stage1/<run_name>/`. This directory contains several single-file `model_<step>.safetensors` checkpoints. Stage 2 later references one of them through `init_stage1_full` as initialization.

---

## 2. Stage 1 validation

> ⚠️ **Important**: The Stage 1 model has not been DMD-distilled. For engineering convenience, we validate it directly with the Stage 2 4-step path with CFG. **The visual quality here does not represent the true capability of Stage 1**. This step is only for quickly checking whether training collapsed and **must not** be used as the final quality comparison.

**Config file**: `config/val_stage1.yaml`. Before running, you **must** fill in `init_stage1_full` and point it to the `model_<step>.safetensors` trained in the previous step, or to the Stage 1 checkpoint we provide.

**Note**: `script/val_stage1.sh` runs `train_flashtalk_stage2.py`. **This is intentional**: all validation is executed through `train_flashtalk_stage2.py`.

**Launch**:

```bash
bash script/val_stage1.sh
```

**Outputs**: generated videos are written to `outputs/val_stage1/<run_name>/`.

---

## 3. Stage 2 training

**Goal**: run **DMD distillation + Self-Forcing++** on top of the Stage 1 model, giving the model three capabilities:

1. **4-step inference**: compress the original 40-step Flow-Matching inference into 4 steps;
2. **CFG-free generation**: remove classifier-free guidance so one forward path is enough;
3. **Noise self-correction**: Self-Forcing++ teaches the student model to recover from accumulated errors while rolling out its own motion latent.

**Config file**: `config/train_stage2.yaml`. Before running, confirm or update:

- `init_stage1_full`: **required**. Point it to the Stage 1 `model_<step>.safetensors`, or to the Stage 1 checkpoint we provide.
- `lmdb_path`: by default, this points to the large-scale TalkCuts LMDB at `processed_data/talkcuts/train/stage2_sample_6400.lmdb`. For the example data, change it to the example LMDB you extracted. For custom data, change it to your packed LMDB.
- `gen_grad_accum_steps` / `critic_grad_accum_steps`: default `4/4`, corresponding to 8 GPUs and global batch size 32.

> **Note**: For the same reason, because the dataset and some hyperparameters differ, `max_steps = 100` is enough in the Stage 2 config. The original FlashTalk paper uses 200 iterations.

**Launch**:

```bash
bash script/train_stage2.sh
```

**Outputs**: saved to `outputs/flashtalk_stage2/<run_name>/`. The directory contains files such as `generator_<step>.safetensors` and `critic_100.safetensors`, with save intervals changing by training phase. Final inference only uses `generator_*.safetensors`.

---

## 4. Stage 2 validation

This uses the real inference path: 4 steps, CFG-free, with motion injection and self-forcing chunked rollout. Besides generated videos, the validation script also runs four objective metrics: **Sync-C / Sync-D / IQA / Aesthe**. Evaluation models must be downloaded first; see [model_weights_preparation.md](model_weights_preparation.md).

**Config file**: `config/val_stage2.yaml`. Before running, you **must** fill in `resume_from`. Validation uses the model pointed to by `resume_from`; set it to the checkpoint directory produced by Stage 2 training. The script automatically loads `generator_<step>.safetensors` from that directory.

**Launch**:

```bash
bash script/val_stage2.sh
```

> Evaluation runs automatically after validation. You can also run `run_evaluate_gt_standalone.py` separately to evaluate all videos under a folder. See the comments at the top of `run_evaluate_gt_standalone.py` for detailed usage.

**Outputs**: generated videos and per-sample/global metric JSON files are written to `outputs/val_stage2/<run_name>/`.

---

## 5. Inference

This repository **only contains training and validation code**. For inference, we recommend using the official **[SoulX-FlashTalk repository](https://github.com/Soul-AILab/SoulX-FlashTalk)**, which includes inference-specific engineering optimizations. We strongly recommend using their code for deployment.

### 5.1 Model format conversion

Our training saves a single-file `generator_<step>.safetensors` checkpoint, with all parameters stored in one file. The SoulX-FlashTalk inference code expects HuggingFace Diffusers sharded safetensors format. We provide a conversion tool:

```bash
python tools/export_stage2_model_to_flashtalk_style.py \
    --src         outputs/flashtalk_stage2/<run_name>/generator_<step>.safetensors \
    --output_dir  outputs/models/SoulX-FlashTalk-14B \
    --num_shards  4
```

Outputs:

```
<output_dir>/
├── diffusion_pytorch_model-00001-of-00004.safetensors
├── diffusion_pytorch_model-00002-of-00004.safetensors
├── diffusion_pytorch_model-00003-of-00004.safetensors
├── diffusion_pytorch_model-00004-of-00004.safetensors
└── diffusion_pytorch_model.safetensors.index.json
```

Overwrite the corresponding files in the [SoulX-FlashTalk](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B) model directory with these files. Keep all other config files unchanged, as shown below:

> ```text
> FlashTalk/
> ├── ...other file...                                          # Keep
> ├── config.json                                               # Keep
> ├── configuration.json                                        # Keep
> ├── diffusion_pytorch_model-00001-of-00004.safetensors        # <- Replace
> ├── diffusion_pytorch_model-00002-of-00004.safetensors        # <- Replace
> ├── diffusion_pytorch_model-00003-of-00004.safetensors        # <- Replace
> ├── diffusion_pytorch_model-00004-of-00004.safetensors        # <- Replace
> ├── diffusion_pytorch_model.safetensors.index.json            # <- Replace
> ├── LICENSE.txt                                               # Keep
> ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   # Keep
> └── ...other file...                                          # Keep
> ```

### 5.2 Run inference with SoulX-FlashTalk

> ⚠️ Note: this project's environment is not compatible with FlashTalk's `torch.compile` acceleration. Please set up a separate environment and run inference according to the [official SoulX-FlashTalk repository](https://github.com/Soul-AILab/SoulX-FlashTalk).

