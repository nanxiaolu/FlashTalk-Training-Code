<h1 align="center">FlashTalk 训练代码</h1>

> **非官方实现。** 本仓库是在 [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) 基础模型之上对 [FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) 训练方案的独立复现。它未经 [FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) 或 [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) 原作者的审查或认可。超参数、消融实验和工程选择可能与官方发布有所不同（详见 **[Tips: 与官方 FlashTalk 实现的核心差异及原因](docs/tips.md#一-与官方-flashtalk-实现的核心差异及原因)**）。

[FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) 是一个能够 Real-Time 生成无限长度的语音驱动数字人模型，它将原本需要 40 步、依赖 CFG（分类器引导）的 InfiniteTalk 压缩为 **4 步、无 CFG** 的 Self-Correcting 模型，同时在包含手部的半身数据上保持较好的性能。本仓库包含了完整的训练代码。

## 🌟 核心亮点 (Key Features)

* **完整的训练流水线**：提供 Stage 1（全参数适应性微调）和 Stage 2（引入 Self-Forcing++ 和 DMD 蒸馏）的完整训练代码。
* **全流程数据支持**：提供 30k 提取好特征的 TalkCuts 样本作为训练集，12 个样本作为验证集。额外提供 32 条未处理短视频作为toy example用于帮助用户快速跑通数据处理流程，方便用到处理自己的数据集。
* **预训练权重开源**：我们公开了通过此流程训练得出的 Stage 1 和 Stage 2 完整模型权重。用户可直接下载用于推理测试、验证对比或作为自身数据的微调起点。
* **全面的评估验证**：包含验证模型及 "Sync-C", "Sync-D", "IQA", "Aesthe" 等评估指标的代码。

以下是FlashTalk 官方开源模型（左）与 我们复现的模型（右）在近似输入条件下的效果对比。复现模型完全基于我们提供的开源训练数据特征训练得出。

  <table align="center">
    <tr>
      <td align="center"><b>Original FlashTalk</b></td>
      <td align="center"><b>Our Reproduction</b></td>
    </tr>
    <tr>
      <td align="center">
          <video src="https://github.com/user-attachments/assets/00b7fbf7-8787-41d8-b252-7a637f953b7f" width="320" controls loop></video>
      </td>
      <td align="center">
          <video src="https://github.com/user-attachments/assets/b0646708-1a95-4906-9c86-beaa5ce97d6f" width="320" controls loop></video>
      </td>
    </tr>
  </table>

## 💻 硬件要求 (Hardware Requirements)

本仓库的默认训练配置基于 **8× NVIDIA A800 (80 GB)**。
* **显存底线**：4 张 80G 显卡（如 A800/H800）会 OOM，8 * 80G是最低要求。当前项目的参数按照 8 GPU 配置。
* **不同卡数扩展**：如果您使用 16、32、64 等其他显卡数量，请参考 **[硬件扩展配置指南](docs/hardware_scaling.md)** 修改对应参数。
* **内存 (RAM) 峰值**：大约需要 **1.6 TB**。内存峰值出现在 Stage 2 同时初始化三个 14B 参数的模型。

## 🛠️ 环境与资源准备 (Preparation)

在开始任何训练或推理之前，请**按顺序**完成以下准备工作：
1. **[环境配置](docs/environment_preparation.md)**：Conda 环境构建、特定依赖库的安装。
2. **[数据准备](docs/data_preparation.md)**：下载训练和验证需要的特征，或准备处理自己的数据集。
3. **[模型权重准备](docs/model_weights_preparation.md)**：所有必要或可选的底层预训练模型以及我们提供的训练 Checkpoints 汇总。

## 🚀 训练流程全览 (Training Pipeline)

整个训练管线分为以下阶段，所有步骤详细的运行命令及配置方法请参考 **[训练、验证与推理指南](docs/train_val_inference.md)**。
> **💡 提示**：为了方便大家对比最终训练效果和快速体验，我们开放了已提取好的大规模数据特征和每个阶段结束后的 Checkpoints 权重。如果您的目标是复现模型性能，您可以灵活利用这些产物跳过某些耗时的训练步骤。

## ⚠️ 避坑指南 (Important Tips)

我们在开发和训练中遇到了许多隐藏的陷阱，为避免您浪费大量时间排查错误，在开始大规模运行之前，**强烈建议**您阅读：**[Tips](docs/tips.md)**

## 🙇 Acknowledgement
[FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/): 这个项目复现的论文。

[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) and [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon.

[Self forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus): FlashTalk使用的 key distillation technique.

[StableAvatar](https://github.com/Francis-Rings/StableAvatar): Stage1训练过程参考他们部分数据处理和loss设计.

[DMD2](https://github.com/tianweiy/DMD2/tree/main?tab=readme-ov-file), [CausVid](https://github.com/tianweiy/CausVid),[Self-Forcing](https://github.com/guandeh17/Self-Forcing) ，[Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus): 参考他们DMD训练的代码与技术

## 📜 License
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, 
granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. 
You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, 
causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. 