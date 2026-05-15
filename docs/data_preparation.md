# Data Preparation

[Chinese version](data_preparation-zh-CN.md)

You ultimately need three data artifacts:

1. features for Stage 1 training (`.lmdb`);
2. features for Stage 2 training (`.lmdb`);
3. validation features shared by Stage 1 and Stage 2 (folder).

This repository supports two data-preparation paths:

- **Path A: use the features we provide directly**. If you only want to reproduce training or quickly validate the model, you can skip preprocessing and download our pre-extracted TalkCuts features, including about 30k training samples across both stages and 12 validation samples.
- **Path B: process your own data**. If you want to adaptively fine-tune on your own dataset, run the full `preprocess -> pack -> train` pipeline. This repository includes 32 example videos to help you verify the whole pipeline.

We strongly recommend running the pipeline with our data first, then trying your own data.

---

## 1. Data-processing pipeline

```
              ┌───────────────────────────┐          ┌──────────────────────┐        ┌───────────┐
raw mp4/wav   │        preprocess         │ payloads │        pack          │  LMDB  │   train   │
─────────────►│                           ├─────────►│                      │───────►│           │
              └───────────────────────────┘          └──────────────────────┘        └───────────┘
```

- Convert raw video/audio into `.payload` files. Each payload stores the sample's VAE latent, CLIP/T5 text features, wav2vec audio features, face mask, motion metadata, and Stage 2 `selected_k`. Files are written to `<payload_dir>/<key>.payload`.
- Pack all payload files into a single LMDB to improve data-loading speed, then use the LMDB for training.

Data paths can be changed in the configuration files. By default, data is stored under `processed_data/` at the repository root:

```
processed_data/
├── example/                                   # 32 + 12 raw example samples (Path B)
│   ├── train/
│   │   ├── audio/<clip_xxx>.wav
│   │   └── video/<clip_xxx>.mp4
│   ├── val/
│   │   ├── audio/<clip_xxx>.wav
│   │   └── video/<clip_xxx>.mp4
│   ├── train_data.csv
│   └── val_data.csv
└── talkcuts/                                  # Large-scale TalkCuts data
    ├── train/                                 # Empty in talkcuts.tar.gz; put downloaded LMDBs here.
    │   ├── stage1_sample_25030.lmdb           # Stage 1 training LMDB
    │   └── stage2_sample_6400.lmdb            # Stage 2 training LMDB (packed for 8 GPUs)
    ├── val/
    │   ├── audio/                             # Raw audio
    │   ├── feature/                           # Features for 12 validation samples
    │   │   ├── <sample_key_0>/...
    │   │   ├── ...
    │   │   └── context_null.pt                # Shared negative text encoding for n_prompt
    │   └── video/                             # Raw video, needed for validation color correction
    └── val_data.csv                           # Validation annotations
```

---

## 2. Path A: use our pre-extracted features

We have preprocessed and packed the TalkCuts data into Baidu Netdisk files. After downloading and extracting them, you can skip preprocessing and packing and start training directly.

| File | Purpose | Extract to | Download |
| --- | --- | --- | --- |
| `talkcuts.tar.gz` | Contains 12 validation samples with pre-extracted features and raw audio/video, `val_data.csv`, and an empty `train` folder. | `processed_data/` | [Baidu Netdisk](https://pan.baidu.com/s/1uEF6QpVih9EotDPKY7W99g?pwd=dc7b) |
| `stage1_sample_25030.lmdb` | Stage 1 training LMDB, about 25,000 samples. | `processed_data/talkcuts/train/` | [Baidu Netdisk](https://pan.baidu.com/s/1zlFI70481g5HHt1FoFJwag?pwd=f94a) |
| `stage2_sample_6400.lmdb` | Stage 2 training LMDB, 6,400 samples, **packed for 8 GPUs**. | `processed_data/talkcuts/train/` | [Baidu Netdisk](https://pan.baidu.com/s/1ksSThfOVuccOnHmAZwYU-Q?pwd=v9fa) |

> ⚠️ If you are not training with 8 GPUs, `stage2_sample_6400.lmdb` **must be repacked**. See [hardware_scaling.md](hardware_scaling.md).
> The default `config/train_stage1.yaml` and `config/train_stage2.yaml` already point to these paths. If you change the paths, update the corresponding config fields.

---

## 3. Path B: process your own data

> This section is only needed if you want to process your own dataset.

### 3.1 Run the full pipeline with 32 example samples

We provide [32 raw video + audio samples](https://pan.baidu.com/s/1GBubHORr7Zb9o09_pMwGHg?pwd=1983) as a minimal runnable example for the data-processing pipeline. Extract them to `processed_data/example`.

```bash
# Stage 1: preprocess + pack
bash script/preprocess_stage1.sh
bash script/pack_stage1.sh

# Stage 2: preprocess + pack (the same 32 clips are sliced again)
bash script/preprocess_stage2.sh
bash script/pack_stage2.sh

# Validation set: preprocess
bash script/preprocess_val.sh
```

The example pipeline should finish within 10 minutes, with most time spent on model initialization. Validation features are written to `processed_data/example/val/feature`, and training outputs are written to `processed_data/example/train/example_stage{1,2}.{payloads,lmdb}`.

> ⚠️ These samples are only for runnable testing and are **far from enough to train a usable model**. To reproduce performance, use Path A or scale up to your own large dataset.

### 3.2 Stage 1 preprocessing for your own dataset

Your dataset must be described by a CSV file with at least the following three columns. Extra columns are allowed.

```csv
video,input_audio,prompt
relative/or/absolute/path/to/clip_001.mp4,relative/or/absolute/path/to/clip_001.wav,"A man in a blue shirt gestures with his hands while explaining a chart..."
...
```

- `video`: an mp4 file. Training uses a 33-frame sliding window, about 1.3 seconds at 25 fps. The original video must be longer than 33 frames; preprocessing randomly samples a 33-frame window.
- `input_audio`: a wav file.
- `prompt`: T5 input text describing the full video.

CSV path resolution rules:

- If `dataset_dir` is empty in the preprocessing YAML, which is the default, all CSV paths are resolved **relative to the repository root**.
- If `dataset_dir` is not empty, paths are resolved relative to that directory.

```bash
cp config/preprocess_stage2_example.yaml config/preprocess_stage2.yaml
```

Important config fields:

| Field | Description |
| --- | --- |
| `lmdb_num_samples` | **Final number of samples kept after filtering**. The data is shuffled and iterated repeatedly until this count is reached. |
| `payload_dir` | Output directory for all sample features. |
| `(val_)annotation_file` | Your data annotation CSV. |
| `dataset_dir` | Applies only to `(val_)annotation_file`. Empty string means paths are relative to the repository root; otherwise paths are relative to this directory. |
| `enable_background_filter` | When enabled, samples with excessive background motion are skipped automatically. |
| `background_ssim_threshold` | Effective only when the flag above is true. This is the background similarity (SSIM) threshold. The default is 0.96; larger values are stricter. |

Then launch preprocessing. Adjust `--nproc_per_node` to your GPU count.

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage1.py --config config/preprocess_stage1.yaml
```

#### 3.3 Pack the Stage 1 LMDB

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir       processed_data/talkcuts/train/my_stage1.payloads \
    --output_lmdb_path  processed_data/talkcuts/train/my_stage1.lmdb \
    --shuffle_k_groups  false
```

Stage 1 does not have a K dimension, so `--shuffle_k_groups false` is the correct option. This process also remaps possibly non-contiguous keys into a contiguous `0..N-1` range.

### 3.4 Stage 2 preprocessing for your own dataset

Stage 2 introduces **self-forcing chunked** training. Each sample has a `selected_k ∈ {1..K_max}`, which determines how many 33-frame chunks the student model rolls out in one training step. This preprocessing step:

1. slices each raw video into overlapping chunks;
2. assigns a K value to each sample while keeping the distribution across K values as balanced as possible.

```bash
cp config/preprocess_stage2_example.yaml config/preprocess_stage2.yaml
```

Similarly, update your own paths, including `lmdb_num_samples`, `payload_dir`, `annotation_file`, and `output_dir`.

Launch command:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage2.py --config config/preprocess_stage2.yaml
```

#### 3.5 Pack the Stage 2 LMDB (**GPU-count binding**)

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir       processed_data/talkcuts/train/my_stage2.payloads \
    --output_lmdb_path  processed_data/talkcuts/train/my_stage2.lmdb \
    --shuffle_k_groups  true \
    --group_size        8
```

- `--shuffle_k_groups true`: enables the Stage 2 LMDB packing behavior.
- `--group_size 8`: configuration for 8 GPUs. For other GPU counts, see [hardware_scaling](hardware_scaling.md).

---

## 4. Validation features

Stage 2 validation uses the real inference path, including motion injection and the specified number of denoising steps, so validation videos must also go through feature extraction first. **⚠️** Whether validating a Stage 1 or Stage 2 model, validation features **must** be extracted with `train_flashtalk_stage1.py` to ensure both stages receive the same feature format. Validation and training use the same preprocessing path. You only need to set the following fields in [config](../config/preprocess_val_example.yaml):

```yaml
val_annotation_file: your csv file
val_features_dir: your feature output path
```

After preprocessing, each validation sample has its own subdirectory containing all features, and the shared negative text encoding for CFG inference is written to `val_features_dir/context_null.pt`.

You can also directly use the validation features provided in Path A.