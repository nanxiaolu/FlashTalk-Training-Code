# 数据准备

本仓库的数据准备共有两条可选路径：

- **路径 A：直接使用我们提供的特征**。如果你只想复现训练或快速验证模型，可以跳过预处理，直接下载 TalkCuts 的预提取特征（两阶段约 30k 训练样本 + 12 条验证集）并解压使用。
- **路径 B：处理自己的数据**。如果想在自有数据集上做适应性微调，需要走完整的 `preprocess → pack → train` 流程；本仓库附带 32 条示例视频用于跑通整个流水线。

强烈建议先用我们的数据跑通后，再尝试在自己的数据上运行。

---

## 1. 数据处理pipeline

```
              ┌───────────────────────────┐          ┌──────────────────────┐        ┌───────────┐
raw mp4/wav   │        preprocess         │ payloads │        pack          │  LMDB  │   train   │
─────────────►│                           ├─────────►│                      │───────►│           │
              └───────────────────────────┘          └──────────────────────┘        └───────────┘
```

- 将raw video/audio处理为 `.payload` 文件，存放该样本的 VAE latent、CLIP / T5 文本特征、wav2vec 音频特征、face mask、运动元信息、以及 Stage 2 的 `selected_k`。位于 `<payload_dir>/<key>.payload`。
- 将所有payload文件pack为单个lmdb文件，提高数据读取速度。

数据路径可以在配置文件自行修改，默认放在仓库根目录的 `processed_data/` 下，结构如下：

```
processed_data/
├── example/                                   # 32 + 12 条示例原始素材（路径 B）
│   ├── train/
│   │   ├── audio/<clip_xxx>.wav
│   │   └── video/<clip_xxx>.mp4
│   ├── val/
│   │   ├── audio/<clip_xxx>.wav
│   │   └── video/<clip_xxx>.mp4
│   ├── train_data.csv
│   └── val_data.csv
└── talkcuts/                                  # 大规模 TalkCuts 数据
    ├── train/                                 # talkcuts.tar.gz里初始为空，请将下载的 LMDB 放入此文件夹
    │   ├── stage1_sample_25030.lmdb           # Stage 1 训练 LMDB
    │   └── stage2_sample_6400.lmdb            # Stage 2 训练 LMDB（按 8 GPU 打包）
    ├── val/
    │   ├── audio/                             # 原始音频
    │   ├── feature/                           # 12 条验证样本的特征
    │   │   ├── <sample_key_0>/...
    │   │   ├── ...
    │   │   └── context_null.pt                # n_prompt 共享的negative文本编码
    │   └── video/                             # 原始视频（验证集色彩校正步骤需要）
    └── val_data.csv                           # 验证集标注
```

---

## 2. 路径 A：使用我们提供的预提取特征

我们已经把 TalkCuts 数据集预处理好并打包到百度网盘，下载并解压后即可跳过 preprocess、pack 直接训练。


| 文件名                        | 用途                                                       | 解压目标                             | 下载链接                                                             |
| -------------------------- | -------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------- |
| `talkcuts.tar.gz`          | 包含 12 条验证样本（预提取特征及原始音视频）、`val_data.csv` 以及空的 `train` 文件夹 | `processed_data/`                | [百度网盘](https://pan.baidu.com/s/1uEF6QpVih9EotDPKY7W99g?pwd=dc7b) |
| `stage1_sample_25030.lmdb` | Stage 1 训练用 LMDB（≈25,000 样本）                             | `processed_data/talkcuts/train/` | [百度网盘](https://pan.baidu.com/s/1zlFI70481g5HHt1FoFJwag?pwd=f94a) |
| `stage2_sample_6400.lmdb`  | Stage 2 训练用 LMDB（6,400 样本，**按 8 GPU 打包**）                | `processed_data/talkcuts/train/` | [百度网盘](https://pan.baidu.com/s/14EM0PQsNVLvSPO7TOrUMpw?pwd=d49e) |


> ⚠️ `stage2_sample_6400.lmdb` 非 8 卡训练**必须**重新 pack。详见[hardware_scaling.md](hardware_scaling.md)。
> 配置文件 `config/train_stage1.yaml` 与 `config/train_stage2.yaml` 默认就指向上述路径，如果修改了路径请对应修改配置文件中的路径。

---

## 3. 路径 B：处理自己的数据

> 以下内容只有在需要处理自己数据集的时候才需要看

### 3.1 用 32 条示例数据跑通整套流程

提前准备了32条原始视频➕音频作为最小可运行示例供大家跑通处理数据过程[百度网盘](https://pan.baidu.com/s/1GBubHORr7Zb9o09_pMwGHg?pwd=1983)，解压放到 `processed_data/example`。

```bash
# Stage 1：预处理 + 打包
bash script/preprocess_stage1.sh
bash script/pack_stage1.sh

# Stage 2：预处理 + 打包（同样的 32 条片段重新切片）
bash script/preprocess_stage2.sh
bash script/pack_stage2.sh
```

上述程序跑通约 5 分钟（绝大部分时间花在模型初始化，32 条样本本身很快）。输出会写到 `processed_data/example/train/example_stage{1,2}.{payloads,lmdb}`。

> ⚠️ 这只是可执行测试用的数据，**远不足以训练出可用模型**。要复现性能，请走路径 A 或扩展到自己的大规模数据集。

### 3.2 Stage 1 预处理（自己的数据集）

你的数据集需要一个 CSV 文件来描述，至少包含以下三列（多余列不会出错）：

```csv
video,input_audio,prompt
relative/or/absolute/path/to/clip_001.mp4,relative/or/absolute/path/to/clip_001.wav,"A man in a blue shirt gestures with his hands while explaining a chart..."
...
```

- `video`：mp4 文件。训练实际使用滑动窗口 33 帧（约 1.3s @ 25fps），原视频长度需要大于33帧，预处理会随机采样33帧的窗口。
- `input_audio`：wav 文件。
- `prompt`：T5 输入文本。描述整个视频的内容。

CSV 中路径的解析规则：

- 如果预处理 YAML 里 `dataset_dir` 留空（默认），CSV 中所有路径会按"**相对于仓库根目录**"解析。
- 如果 `dataset_dir` 非空，则相对该目录解析。

```bash
cp config/preprocess_stage2_example.yaml config/preprocess_stage2.yaml
```

配置文件里需要关注的字段：


| 字段                          | 说明                                                |
| --------------------------- | ------------------------------------------------- |
| `lmdb_num_samples`          | **过滤后最终留下的样本数**，会shuffle，循环遍历数据集直到样本数达到该值         |
| `payload_dir`               | 所有样本特征的输出目录                                       |
| `(val_)annotation_file`     | 你的数据标注 CSV                                        |
| `dataset_dir`               | 只作用于(val_)annotation_file，空串表示相对仓库根目录；填值表示相对该目录   |
| `enable_background_filter`  | 开启后会自动跳过背景晃动过大的样本                                 |
| `background_ssim_threshold` | 仅在上面 flag 设置true时生效，表示背景相似度（SSIM）阈值，默认 0.96，越大越严格 |


然后启动（按你的 GPU 数调整 `--nproc_per_node`）：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage1.py --config config/preprocess_stage1.yaml
```

#### 3.3 打包 Stage 1 LMDB

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir       processed_data/talkcuts/train/my_stage1.payloads \
    --output_lmdb_path  processed_data/talkcuts/train/my_stage1.lmdb \
    --shuffle_k_groups  false
```

Stage 1 没有 K 维度概念，`--shuffle_k_groups false` 是正确选项。改过程会把可能不连续的 key 重排为连续的 `0..N-1`。

### 3.4 Stage 2 预处理（自己的数据集）

Stage 2 引入了 **self-forcing chunked** 训练：每个样本会持有一个 `selected_k ∈ {1..K_max}`，决定学生模型在一次训练步中要 rollout 多少个 33 帧chunk。这个预处理过程会做：

1. 把每条原始视频再切成有重叠的小块；
2. 给每个样本分配 K 值，保证每个 K 的值尽可能均衡。

```bash
cp config/preprocess_stage2_example.yaml config/preprocess_stage2.yaml
```

同样，修改自己的路径（lmdb_num_samples，payload_dir，annotation_file，output_dir）。

启动命令：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage2.py --config config/preprocess_stage2.yaml
```

同步逻辑与 Stage 1 一致；日志中会打印各 K 桶的目标分布，例如：
`Stage2 K counts target snapshot: {1: 1280, 2: 1280, 3: 1280, 4: 1280, 5: 1280}`（`K_max=5`、`num_samples=6400` 时）。

#### 3.5 打包 Stage 2 LMDB（**注意 GPU 数量绑定**）

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir       processed_data/talkcuts/train/my_stage2.payloads \
    --output_lmdb_path  processed_data/talkcuts/train/my_stage2.lmdb \
    --shuffle_k_groups  true \
    --group_size        8
```

- `--shuffle_k_groups true`：true时程序执行stage2时的lmdb打包。
- `--group_size 8`：**必须等于你打算训练用的 GPU 数量，或者group_size % GPU_num == 0**。因为FSDP要求多GPU时不同rank的前向传播次数需要一致，否则程序会卡住，所以这里需要限制所有rank的样本的chunk数K一致。后续如果调整 GPU 数，必须重打包。

---

## 4. 验证集特征

Stage 2 的验证是真实推理路径（动作注入、特定降噪步数等），所以验证视频也需要先做一次特征提取。**⚠️**：无论是验证 Stage 1 还是 Stage 2 的模型，验证集的特征**必须**使用 `train_flashtalk_stage1.py` 提取，以保证两阶段验证时输入的特征格式完全一致。验证集和训练集走同一条 preprocess 路径，只需要在[config](config/preprocess_val_example.yaml)里设置：

```yaml
val_annotation_file: your csv file
val_features_dir: your feature output path
```

完成后每条验证样本会有一个子目录存放它的所有特征，并在 `val_features_dir/context_null.pt` 写入共享的negtive文本编码（推理时 CFG 用）。

如果你想，也可以直接用路径 A 提供的特征。