# FlashTalk Training Code

[Chinese version](README-zh-CN.md)

> **Unofficial implementation.** This repository is an independent reproduction of the [FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) training recipe built on top of the [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) base model. It has not been reviewed or endorsed by the original authors of [FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) or [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk). Hyperparameters, ablations, and engineering choices may differ from the official release. See **[Tips: key differences from the official FlashTalk implementation](docs/tips.md#1-key-differences-from-the-official-flashtalk-implementation)** for details.

[FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) is a speech-driven digital human model that can generate real-time videos of unlimited length. It compresses InfiniteTalk from a 40-step, CFG-based model into a **4-step, CFG-free** self-correcting model while preserving strong performance on half-body data with hand motion. This repository provides the full training code.

## 🌟 Key Features

- **Complete training pipeline**: Stage 1 full-parameter fine-tuning and Stage 2 training with Self-Forcing++ and DMD distillation.
- **End-to-end data support**: around 30k pre-extracted TalkCuts training samples and 12 validation samples are provided. We also include 32 raw short videos as toy examples so users can quickly run through the preprocessing pipeline and adapt it to your own datasets.
- **Open pretrained weights**: we release the Stage 1 and Stage 2 model weights trained with this pipeline. They can be used for inference tests, validation comparisons, or as fine-tuning starting points for custom data.
- **Evaluation support**: validation and metric code for "Sync-C", "Sync-D", "IQA", "Aesthe", and related checks.

The comparison below shows the official FlashTalk open model on the left and our reproduced model on the right under similar input conditions. Our reproduced model is trained entirely from the open training-data features provided in this repository.

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

## 💻 Hardware Requirements

The default training configuration targets **8 x NVIDIA A800 (80 GB)**.

- **Minimum GPU memory**: 4 x 80 GB GPUs such as A800/H800 will run out of memory. 8 x 80 GB is the minimum supported setup for the default configuration.
- **Scaling to other GPU counts**: if you use 16, 32, 64, or another number of GPUs, see the **[hardware scaling guide](docs/hardware_scaling.md)** and update the corresponding parameters.
- **Peak RAM usage**: about **1.6 TB**. The peak occurs in Stage 2, where three 14B-parameter models are initialized at the same time.

## 🛠️ Preparation

Before running any training or inference workflow, complete the following steps **in order**:

1. **[Environment preparation](docs/environment_preparation.md)**: create the Conda environment and install required dependencies.
2. **[Data preparation](docs/data_preparation.md)**: download the training/validation features or preprocess your own dataset.
3. **[Model weights preparation](docs/model_weights_preparation.md)**: download required or optional base models and the checkpoints we provide.

## 🚀 Training Pipeline

The training pipeline is divided into multiple stages. For detailed commands and configuration instructions, see the **[training, validation, and inference guide](docs/train_val_inference.md)**.

> **💡 Tip**: To make final-quality comparison and quick experiments easier, we release pre-extracted large-scale data features and checkpoints from the end of each stage. If your goal is to reproduce model performance, you can use these artifacts to skip some time-consuming training steps.

## ⚠️ Important Tips

We encountered many hidden pitfalls during development and training. Before starting large-scale runs, we strongly recommend reading **[Tips](docs/tips.md)** to avoid spending a lot of time debugging known issues.

## 🙇 Acknowledgement

[FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/): the paper reproduced by this project.

[InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) and [Wan](https://github.com/Wan-Video/Wan2.1): the base models we build upon.

[Self forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus): the key distillation technique used by FlashTalk.

[StableAvatar](https://github.com/Francis-Rings/StableAvatar): part of our Stage 1 data processing and loss design references their work.

[DMD2](https://github.com/tianweiy/DMD2), [CausVid](https://github.com/tianweiy/CausVid), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), and [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus): our DMD training code references these projects.

## 📜 License

The models in this repository are licensed under the Apache 2.0 License. We claim no rights over your generated contents, 
granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. 
You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. 

## Contact

If you have any questions, feel free to open an issue. I will reply within 24 hours.