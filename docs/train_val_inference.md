# 训练、验证与推理指南

本指南覆盖 Stage 1 / Stage 2 的训练与验证，以及最终的推理部署。所有命令均默认使用 8 GPU。

> **前置依赖**：开始之前请先完成：
>
> 1. [环境配置](environment_preparation.md)
> 2. [数据准备](data_preparation.md)
> 3. [模型权重准备](model_weights_preparation.md)

> **多卡数量提醒**：默认配置基于 8 GPU。其它卡数请先阅读 [hardware_scaling.md](hardware_scaling.md)。

---

## 1. Stage 1 训练

**目的**：让基础模型（InfiniteTalk）从纯头部说话数据**适配到包含手势、上半身运动的数据分布**。

**配置文件**：`config/train_stage1.yaml`。  
请修改 `config/train_stage1.yaml` 的 `lmdb_path` ，默认指向我们提供的预提取特征。

> **注**：由于我们的数据集与官方不同，训练的迭代步数与FlashTalk论文里的不一致，这里采用 `max_steps` = 500 即可达到最优效果（原版 FlashTalk 论文中为 1000 iters）。

**启动**：

```bash
bash script/train_stage1.sh
```

**产物**：保存在 `outputs/flashtalk_stage1/<run_name>/`，其中包含若干 `models_<step>.safetensors`（单文件），后面 Stage 2 通过 `init_stage1_full` 字段引用其中一份作为初始权重。

---

## 2. Stage 1 验证

> ⚠️ **重要**：Stage 1 模型尚未经过 DMD 蒸馏，但出于工程便利我们直接用 Stage 2 的 4 步含 CFG 来验证它。**这一步的画质并不代表 Stage 1 真实能力**，仅用来快速判断模型是否训崩，**不能**作为最终效果对比依据。

**配置文件**：`config/val_stage1.yaml`。运行前**必须**补充 `init_stage1_full` 参数，将其指向你上一步训出的 `generator_<step>.safetensors` 路径（也可以是我们提供的预训练 Stage 1 ckpt）。

**注意**：在script/val_stage1.sh里运行的是 `train_flashtalk_stage2.py`，**这不是笔误**——val过程都用`train_flashtalk_stage2.py`脚本来执行。`config/val_stage1.yaml` 通过 `val_only` 字段告诉它"我只是想加载一个 Stage 1 权重做验证，别开始训练"。

**启动**：

```bash
bash script/val_stage1.sh
```

**产物**：生成视频写到 `outputs/val_stage1/<run_name>/`。

---

## 3. Stage 2 训练

**目的**：在 Stage 1 模型基础上做 **DMD 蒸馏 + Self-Forcing++**，让模型具备三种能力：

1. **4 步推理**：把原本 40 步的 Flow-Matching 推理压到 4 步；
2. **CFG-free**：去掉分类器引导，单次前向即可；
3. **噪声自纠正**：Self-Forcing++ 让学生模型在 rollout 自己的 motion latent 时学会从累积误差中恢复。

**配置文件**：`config/train_stage2.yaml`。运行前需要确认/修改：

- `init_stage1_full`：**必须补充**，指向 Stage 1 训练产出的 `generator_<step>.safetensors`（或我们提供的 Stage 1 ckpt）。
- `lmdb_path`：默认指向大规模 TalkCuts 数据集 `processed_data/talkcuts/train/stage2_sample_6400.lmdb`。如果是跑示例，请修改为你提取的示例 LMDB；如果是自定义数据，请修改为你 pack 出来的 LMDB。
- `gen_grad_accum_steps` / `critic_grad_accum_steps`：默认 4/4 对应 8 GPU (batchsize=32)。

> **注**：同理，由于数据集差异，Stage 2 配置文件中的 `max_steps` 设为 100 即可（原版 FlashTalk 论文中为 200 iters）。

**启动**：

```bash
bash script/train_stage2.sh
```

**产物**：保存在 `outputs/flashtalk_stage2/<run_name>/`，包含按训练阶段切换保存间隔的 `generator_<step>.safetensors`、`critic_100.safetensors` 等。最终推理只关心 `generator_*.safetensors`。

---

## 4. Stage 2 验证

走真实推理路径：4 步、CFG-free、含动作注入、含 self-forcing chunked rollout。除生成视频外，验证脚本还会跑 **Sync-C / Sync-D / IQA / Aesthe** 四个客观指标（需要先下载评估模型，见 [model_weights_preparation.md](model_weights_preparation.md)）。

**配置文件**：`config/val_stage2.yaml`。运行前**必须**补充 `resume_from` 参数，val 的是`resume_from` 参数指向的模型。需要将其指向 Stage 2 训练输出的 checkpoint 目录（脚本会自动从该目录加载 `generator_<step>.safetensors`）。

**启动**：

```bash
bash script/val_stage2.sh
```

> val 后会自动运行eval，也可以单独运行run_evaluate_gt_standalone.py评估一个文件夹下所有视频，详细用法可以看run_evaluate_gt_standalone.py最开头的注释。

**产物**：生成视频与逐样本/整体指标 JSON 写到 `outputs/val_stage2/<run_name>/`。

---

## 5. 推理 (Inference)

本仓库**只包含训练与验证代码**。推理推荐使用 **[SoulX-FlashTalk 官方仓库](https://github.com/Soul-AILab/SoulX-FlashTalk)** ，他们对推理做了一些工程优化，强烈建议部署时直接用他们的代码。

### 5.1 模型格式转换

我们训练保存的是单文件 `generator_<step>.safetensors`（全部参数挤在一个 file 里），而 SoulX-FlashTalk 推理代码要求 HuggingFace Diffusers 的分片 safetensors 格式。提供了一个转换工具：

```bash
python tools/export_stage2_model_to_flashtalk_style.py \
    --src         outputs/flashtalk_stage2/<run_name>/generator_<step>.safetensors \
    --output_dir  outputs/models/SoulX-FlashTalk-14B \
    --num_shards  4
```

产出物：

```
<output_dir>/
├── diffusion_pytorch_model-00001-of-00004.safetensors
├── diffusion_pytorch_model-00002-of-00004.safetensors
├── diffusion_pytorch_model-00003-of-00004.safetensors
├── diffusion_pytorch_model-00004-of-00004.safetensors
└── diffusion_pytorch_model.safetensors.index.json
```

把这些文件**覆盖**到 [SoulX-FlashTalk](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B) 的对应模型目录即可（除上述文件外，其它配置文件保持不变），如下图所示：

> ```text
> FlashTalk/
> ├── ...other file...                                          # 保留
> ├── config.json                                               # 保留
> ├── configuration.json                                        # 保留
> ├── diffusion_pytorch_model-00001-of-00004.safetensors        # <- 替换
> ├── diffusion_pytorch_model-00002-of-00004.safetensors        # <- 替换
> ├── diffusion_pytorch_model-00003-of-00004.safetensors        # <- 替换
> ├── diffusion_pytorch_model-00004-of-00004.safetensors        # <- 替换
> ├── diffusion_pytorch_model.safetensors.index.json            # <- 替换
> ├── LICENSE.txt                                               # 保留
> ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   # 保留
> └── ...other file...                                          # 保留
> ```

### 5.2 用 SoulX-FlashTalk 推理

> ⚠️ 注：本项目的环境无法兼容FlashTalk的torch.compile加速，请按照[SoulX-FlashTalk 官方仓库](https://github.com/Soul-AILab/SoulX-FlashTalk)指示重新配置环境并推理。

