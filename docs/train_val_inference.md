# 训练、验证与推理指南

本指南将按时间顺序带您走完整个数据预处理、模型训练、验证及最终的推理部署流程。所有脚本启动命令均默认使用 `torchrun --nproc_per_node=8 --standalone`。

## 0. 数据提取与特征准备

在开始训练之前，需要将您的原始视频与音频数据处理成模型可读取的特征文件。

> **提示**：我们随代码提供了 32 条短视频（位于 `processed_data/example/`）用于帮助您快速跑通整个预处理流程或进行格式调试。如果您想要复现最终的模型性能，**您必须使用我们提供的大规模 TalkCuts 提取特征数据集**，或者用本流程处理您自己足够多且质量高的数据。

### Stage 1 预处理
将视频和音频转换为 VAE 潜变量、CLIP/T5 文本特征以及音频特征。

1. **配置**：复制并修改配置文件。
   ```bash
   cp config/preprocess_stage1_example.yaml config/preprocess_stage1.yaml
   $EDITOR config/preprocess_stage1.yaml
   ```
2. **运行特征提取**：
   ```bash
   OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
     train_flashtalk_stage1.py --config config/preprocess_stage1.yaml
   ```
3. **打包 LMDB**：提取完成后，将所有 `.pt` 有效载荷文件打包成单个 LMDB，以加速训练时的读取。
   ```bash
   python tools/payload_files_to_lmdb.py \
       --payload_dir processed_data/talkcuts/train/my_stage1.payloads \
       --output_lmdb_path processed_data/talkcuts/train/my_stage1.lmdb \
       --num_samples 25030 \
       --shuffle_k_groups false
   ```

### Stage 2 预处理
由于 Stage 2 使用分块自强制（Self-Forcing++），我们需要对数据进行重新切片，并分配特定的生成块长度（`selected_k`）。

1. **配置**：复制并修改配置文件。
   ```bash
   cp config/preprocess_stage2_example.yaml config/preprocess_stage2.yaml
   $EDITOR config/preprocess_stage2.yaml
   ```
2. **运行提取与重新切片**：
   ```bash
   OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
     train_flashtalk_stage2.py --config config/preprocess_stage2.yaml
   ```
3. **打包 LMDB (⚠️ 注意 GPU 数量绑定)**：
   **极度重要**：这里打包时 `--group_size` 必须严格等于您将要用于 Stage 2 训练的 GPU 数量！
   ```bash
   python tools/payload_files_to_lmdb.py \
       --payload_dir processed_data/talkcuts/train/my_stage2.payloads \
       --output_lmdb_path processed_data/talkcuts/train/my_stage2.lmdb \
       --num_samples 6400 \
       --shuffle_k_groups true \
       --group_size 8
   ```

---

## 1. Stage 1 训练

Stage 1 的目的是将基础模型分布从纯头部说话数据迁移到包含手势和身体运动的数据分布上。

* **配置**：编辑 `config/train_stage1.yaml`，确认 `lmdb_path` 指向您的 Stage 1 训练集。
* **运行**：
  ```bash
  bash script/train_stage1.sh
  ```
* 产物将保存在 `outputs/flashtalk_stage1/` 目录下。

## 2. Stage 1 验证

> ⚠️ **重要提示**：Stage 1 验证是直接使用第一阶段的模型进行 **4 步推理**。由于该模型还未经过 Stage 2 蒸馏，生成的视频质量可能并不理想。这里的验证**仅用于快速参考模型是否发生训练崩溃**，并不代表最终效果。

* **配置**：编辑 `config/val_stage1.yaml`，设置 `init_stage1_full` 为刚训练好的（或下载好的）Stage 1 safetensors 路径。
* **运行**：
  ```bash
  bash script/val_stage1.sh
  ```

## 3. Stage 2 训练

Stage 2 通过 DMD 蒸馏和自强制（Self-Forcing++）让模型能够在 4 步、无 CFG 下工作，且能从累积噪声中恢复。

* **配置**：编辑 `config/train_stage2.yaml`，将 `init_stage1_full` 指向 Stage 1 生成的权重，并确认数据路径正确。
* **运行**：
  ```bash
  bash script/train_stage2.sh
  ```

## 4. Stage 2 验证

对最终蒸馏好的模型进行全面的测试。它复用了 Stage 2 的真实推理路径（动作注入、特定降噪步数等）。验证代码不仅会生成视频，还会计算 "Sync-C", "Sync-D", "IQA", "Aesthe" 等评估指标。

* **配置**：编辑 `config/val_stage2.yaml`，设置 `resume_from` 为您的 Stage 2 checkpoint 目录。
* **运行**：
  ```bash
  bash script/val_stage2.sh
  ```

## 5. 推理 (Inference)

本仓库**仅包含训练与验证**代码。为了获得更好的推理速度及显存优化，实际的推理服务由原版 **[SoulX-FlashTalk 官方仓库](https://github.com/Soul-AILab/SoulX-FlashTalk)** 负责支持。由于官方推理代码对底层生成过程做了诸多优化，强烈建议在最终部署时使用。

**导出权重并推理的步骤**：

1. **转换格式**：使用本项目提供的脚本，将保存的单文件 safetensors 转换为 Diffusers 分片格式。
   ```bash
   python tools/export_stage2_model_to_flashtalk_style.py \
       --src outputs/flashtalk_stage2/.../generator_xxx.safetensors \
       --output_dir <FlashTalk克隆目录>/models/SoulX-FlashTalk-14B \
       --num_shards 4
   ```
2. **克隆推理仓库并运行**：
   ```bash
   git clone https://github.com/Soul-AILab/SoulX-FlashTalk.git
   cd SoulX-FlashTalk
   # 按照其 README 安装依赖后执行推理
   bash inference_script_multi_gpu.sh
   ```