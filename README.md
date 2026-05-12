# FlashTalk Training

> **Non-official implementation.** This repository is an independent
> re-implementation of the FlashTalk training recipe on top of the
> [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) base model. It is
> not endorsed by, affiliated with, nor reviewed by the original FlashTalk or
> InfiniteTalk authors. Hyper-parameters, ablations and engineering choices
> may differ from any "official" release.

FlashTalk turns the multi-step, CFG-dependent InfiniteTalk inference pipeline
into a **4-step, CFG-free** audio-driven talking-video generator while keeping
hand- and body-region fidelity. This repo contains the full training code,
data preprocessing pipeline, and a minimum-viable example that lets you
verify everything runs end-to-end on a single 8×GPU node.

---

## What stages 1 and 2 actually do

The base InfiniteTalk model already works well at the resolutions and window
sizes used here (33-frame, ~480p clips), but its quality collapses when the
training distribution contains hands and large body motion — it was tuned on
head-only data. The two stages target two *separate* problems:

* **Stage 1 — Dataset adaptation (full-parameter flow-matching fine-tune).**
  Stage 1 is **not** about changing the window size or resolution. It is
  about pushing the base distribution from head-only talking footage to the
  TalkCuts-style data used here, which includes hand gestures and torso
  motion. We fine-tune all parameters of the generator with a flow-matching
  objective plus a face-region loss and a temporal-consistency loss; the
  output is a single generator checkpoint
  (`stage1_save_model.safetensors`) suitable for both downstream use and
  Stage-2 initialization.

* **Stage 2 — Distillation, CFG removal, and error correction (DMD).**
  Stage 2 takes the Stage-1 generator and trains it under Distribution
  Matching Distillation with three networks (generator / real-score /
  fake-score). The goals are:
  1. shrink inference from ~40 denoising steps down to **4**,
  2. learn to operate **CFG-free** so we no longer need text/audio
     classifier-free guidance at inference, and
  3. inject self-forcing chunked rollouts so the student learns to
     **recover from accumulated noise** rather than collapsing on long
     auto-regressive generation. The dataset is packed in chunks of variable
     length (`selected_k ∈ {1..K_max}`) precisely so the student can train
     the recovery behaviour rather than only single-chunk inference.

The output of Stage 2 is the deployable generator
(`stage2_save_model.safetensors`).

---

## Hardware

The released checkpoints and the entire two-stage pipeline were trained on
**8× NVIDIA A800 (80 GB)**. That is the *only* configuration we ran
end-to-end and the only one we have measured wall-clock times for. A few
notes if you deviate:

* **4× A800 will OOM.** We tried. The DiT-14B Generator + 3-network DMD
  state in Stage 2 does not fit on 4 cards even with FSDP CPU-offload and
  gradient checkpointing both enabled. 8 cards is effectively the floor.
* **16 / 32 / 64 / 128 GPUs**: the launchers and configs do generalize,
  but you need to update three things together so the global batch stays
  the same and Stage-2 K-homogeneity is preserved:

  | GPUs | `grad_accum_steps` (stage1) | `gen_grad_accum_steps` / `critic_grad_accum_steps` (stage2) | `--group_size` at stage-2 pack | Repreprocess stage-2 LMDB? |
  |---:|---:|---:|---:|---|
  |   8 | 4 | 4 / 4 | 8   | no — released LMDB fits |
  |  16 | 2 | 2 / 2 | 16  | no |
  |  32 | 1 | 1 / 1 | 32  | no |
  |  64 | 1 | 1 / 1 | 64  | **yes** — 140 steps × 64 ranks > 6400 keys |
  | 128 | 1 | 1 / 1 | 128 | **yes** — same overflow, plus K-bucket tails get larger |

  Stage 1 is more forgiving than Stage 2 because there is no K-grouping
  constraint and the released `stage1_sample_25030.lmdb` holds 25 030
  keys, enough for 195 steps even at 128 GPUs.

* If you have to repreprocess Stage 2 for ≥64 GPUs, bump
  `lmdb_num_samples` so that `step * world_size > lmdb_num_samples`
  never trips during training. As a rule of thumb,
  `lmdb_num_samples >= 2 * max_steps * world_size` and
  `(num_samples_per_K_bucket) % group_size == 0` (else the packer drops
  the tail of each K bucket — see
  [docs/stage2_k_grouping.md](docs/stage2_k_grouping.md)).

* See [docs/training.md](docs/training.md) for the per-stage scaling
  rationale and where the global batch size enters each loss.

---

## Things to read **before** you launch a long run

These are the things that tripped us up. Please skim them, especially the
first item — it is structural rather than a one-line fix.

### 1. Stage-2 packs LMDB samples in K-consistent groups; `group_size` must equal your GPU count

Stage 2 uses FSDP and rolls out exactly `selected_k` forward passes per
sample (`selected_k ∈ {1..K_max}`, stored in each LMDB entry). FSDP
synchronizes gradients across ranks after every forward pass, so **every
rank in a global batch must perform the same number of forwards**. We
enforce this by packing the LMDB in groups of `--group_size` consecutive
samples that share the same `selected_k`. The shipped
`stage2_sample_6400.lmdb` is packed with `--group_size 8` (i.e. every 8
consecutive samples share `selected_k`), which is hard-coded for 8 GPUs.

If you train Stage 2 with a different GPU count (16, 32, …), **re-pack
the LMDB** (`script/pack_stage2.sh` with `--group_size <your_gpu_count>`),
or the run will hang on the first global-batch where ranks took
different numbers of forward passes. (Going below 8 GPUs is not viable
— see the **Hardware** section above.) The same table tells you what
`--group_size` and `grad_accum_steps` to pick.

### 2. *(reserved)* — your own pitfalls go here

<!--
Author note: free-form slots for the original trainer (chenxiaoyong) to
fill in personal experience. Suggested template:

#### Title (one sentence summarizing the symptom)
**Symptom**: what you saw / how it failed.
**Root cause**: what was actually happening underneath.
**Fix**: the smallest change that made it go away.
-->

### 3. *(reserved)*

### 4. *(reserved)*

### 5. *(reserved)*

### 6. *(reserved)*

### 7. *(reserved)*

---

## Quick links to detailed docs

| Topic | File |
|---|---|
| Conda + pip environment from scratch | [`docs/environment.md`](docs/environment.md) |
| What's on Baidu NetDisk and what to do with each archive | [`docs/baidu_assets.md`](docs/baidu_assets.md) |
| Run validation on the bundled 12 example clips | [`docs/validation.md`](docs/validation.md) |
| Run Stage-1 / Stage-2 training | [`docs/training.md`](docs/training.md) |
| Preprocess your own raw videos into LMDB | [`docs/data_preparation.md`](docs/data_preparation.md) |
| Stage-2 K-grouping internals (FSDP rationale, repacking) | [`docs/stage2_k_grouping.md`](docs/stage2_k_grouping.md) |
| Export a Stage-2 checkpoint for SoulX-FlashTalk inference | [`docs/inference.md`](docs/inference.md) |

---

## Repository layout

```
flashtalk-training-dev/
├── train_flashtalk_stage1.py      # stage-1 entry (also handles stage-1 preprocess)
├── train_flashtalk_stage2.py      # stage-2 entry (training + validation for both stages)
├── infinitetalk_dmd.py            # InfiniteTalkDMD model (G / real-score / fake-score)
├── config/
│   ├── train_stage1.yaml          # stage-1 training (full LMDB)
│   ├── train_stage2.yaml          # stage-2 training (full LMDB)
│   ├── preprocess_stage1_example.yaml  # stage-1 preprocess on the bundled 32-clip example
│   ├── preprocess_stage2_example.yaml  # stage-2 preprocess on the bundled 32-clip example
│   ├── val_stage1.yaml            # validation of a stage-1 checkpoint
│   └── val_stage2.yaml            # validation of a stage-2 checkpoint
├── script/
│   ├── preprocess_stage{1,2}.sh   # raw video -> per-sample payload files
│   ├── pack_stage{1,2}.sh         # payload files -> single LMDB
│   ├── train_stage{1,2}.sh        # training launchers (torchrun, 8×GPU)
│   └── val_stage{1,2}.sh          # validation launchers
├── src/
│   ├── data_processor_flashtalk.py
│   ├── validation_inference.py
│   └── _warning_filters.py        # silences noisy 3rd-party deprecation warnings
├── tools/
│   └── payload_files_to_lmdb.py   # stage-A -> stage-B packer
├── wan/                           # InfiniteTalk modules (VAE, CLIP, T5, ...)
├── processed_data/
│   ├── talkcuts/                       # full TalkCuts dataset (features only)
│   │   ├── val_data.csv                # 12-sample validation CSV (sample_ids match val/feature/)
│   │   ├── train/                      # holds the full-dataset training LMDB (download to here)
│   │   │   ├── stage1_sample_25030.lmdb
│   │   │   └── stage2_sample_6400.lmdb
│   │   └── val/
│   │       └── feature/                # precomputed per-sample validation features
│   │           ├── context_null.pt     # shared n_prompt encoding
│   │           └── <sample_id>/...     # context.pt, full_emb.pt, clip_fea.pt, ...
│   └── example/                        # 32 train + 12 val raw clips for smoke testing
│       ├── train_data.csv
│       ├── val_data.csv
│       ├── train/{video,audio}/
│       └── val/{video,audio}/          # feature/ is generated by preprocess_split=val
└── weights/                            # gitignored; populate it with the
                                        # Baidu `weights.tar` contents (or a
                                        # symlink to an existing tree). The
                                        # expected layout is documented in
                                        # the "Where the downloaded assets go"
                                        # section below.
```

---

## Where the downloaded assets go

Git tracks **none** of the binaries listed below — `weights/` and
everything under `processed_data/` are gitignored. The directory names
themselves are intentionally not committed either (the placeholder
`.gitkeep` files keep `outputs/` / `processed_data/` alive in the repo;
`weights/` you create yourself, see step 1). After downloading the
Baidu NetDisk archives, your tree should look like:

### `weights/` (extract `weights.tar` into this directory)

```
weights/
├── InfiniteTalk/
│   └── single/
│       └── infinitetalk.safetensors                        # base InfiniteTalk DiT
├── Wan2.1-I2V-14B-480P/                                    # HuggingFace snapshot
│   ├── Wan2.1_VAE.pth
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── diffusion_pytorch_model-0000{1..7}-of-00007.safetensors
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── google/umt5-xxl/                                    # T5 tokenizer
│   └── xlm-roberta-large/                                  # CLIP-XLMR tokenizer
├── chinese-wav2vec2-base/
│   ├── chinese-wav2vec2-base-fairseq-ckpt.pt               # audio encoder
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── insightface/
│   └── models/antelopev2/                                  # face det / id / landmark
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── genderage.onnx
│       ├── glintr100.onnx
│       └── scrfd_10g_bnkps.onnx
├── rvm_model/
│   └── rvm_mobilenetv3.pth                                 # background-stability filter
├── face_det/                                               # (optional) eval-time face crop
│   ├── detection_Resnet50_Final.pth
│   ├── detection_mobilenet0.25_Final.pth
│   ├── parsing_bisenet.pth
│   └── parsing_parsenet.pth
├── syncnet/                                                # (optional) eval-time SyncNet
│   ├── sfd_face.pth
│   └── syncnet_v2.model
└── q-align/                                                # (optional) eval-time Q-Align
    ├── config.json
    ├── pytorch_model-0000{1,2}-of-00002.bin
    ├── pytorch_model.bin.index.json
    └── tokenizer.model
```

Locally you can either populate `weights/` directly or replace it with
a symlink to an existing tree — git is configured to ignore the entire
path (`/weights`), so either choice is fine.

The three `(optional)` blocks (`face_det/`, `syncnet/`, `q-align/`) are
only needed if you run the bundled evaluation scripts; they are *not*
required for training or for normal validation.

### `processed_data/` (extract the data archives here)

```
processed_data/
├── talkcuts/                                               # full TalkCuts dataset
│   ├── val_data.csv                                        # from talkcuts.tar.gz
│   ├── train/                                              # the two big training LMDBs
│   │   ├── stage1_sample_25030.lmdb                        # from stage1_sample_25030_lmdb/
│   │   └── stage2_sample_6400.lmdb                         # from stage2_sample_6400_lmdb/
│   └── val/
│       └── feature/                                        # from talkcuts.tar.gz
│           ├── context_null.pt
│           └── <sample_id>/{context,full_emb,clip_fea,first_frame_latent,cond_latents}.pt + metadata.json
└── example/                                                # from example_data.tar.gz
    ├── train_data.csv
    ├── val_data.csv
    ├── train/{video,audio}/                                # 32 raw mp4 + wav pairs
    └── val/{video,audio}/                                  # 12 raw mp4 + wav pairs
```

See [`docs/baidu_assets.md`](docs/baidu_assets.md) for the exact
archive names, sizes, and one-liner extraction commands.

---

## End-to-end checklist (existing trained weights)

If you have downloaded the Baidu NetDisk assets and just want to **run** the
trained model:

```bash
# 1. Environment (see docs/environment.md for details, including flash-attn)
conda create -n flashtalk_cxy python=3.10.18 -y
conda activate flashtalk_cxy
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation   # A-series

# 2. Drop the Baidu NetDisk archives into the layouts shown above.

# 3. Validate the released stage-2 checkpoint on the 12 bundled clips
#    (edit config/val_stage2.yaml: resume_from=<dir containing
#    generator_xxx.safetensors and training_state.pt>)
bash script/val_stage2.sh
```

---

## End-to-end checklist (training from scratch)

```bash
# 1. Environment (same as above)
# 2. Stage-1 preprocess + pack on YOUR full dataset (see docs/data_preparation.md)
# 3. Stage-1 training
bash script/train_stage1.sh
# 4. Stage-2 preprocess + pack on YOUR full dataset
# 5. Stage-2 training (point init_stage1_full at the stage-1 safetensors)
bash script/train_stage2.sh
```

A smoke-test loop on the bundled 32-clip example takes ~5 min per
preprocess stage on 8×A800; see [`docs/training.md`](docs/training.md).

---

## Inference

This repository contains the **training and validation** pipeline only —
once a Stage-2 checkpoint is produced, the actual serving / streaming
inference is performed by the upstream
[SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk)
repository (it ships a single-GPU script, a multi-GPU script, and a
Gradio app).

A trained Stage-2 checkpoint produced by this repo is a single
`generator_<step>.safetensors` file (alongside `training_state.pt` and
its config); SoulX-FlashTalk however expects the HF-Diffusers sharded
layout
`diffusion_pytorch_model-XXXXX-of-YYYYY.safetensors` +
`diffusion_pytorch_model.safetensors.index.json`. We provide
`tools/export_stage2_model_to_flashtalk_style.py` to perform the
conversion (it shards the tensors, casts floats to bf16, and writes the
weight-map index expected by Diffusers).

End-to-end recipe:

```bash
# 1. Convert a trained Stage-2 checkpoint into the SoulX-FlashTalk layout.
#    Point --src at the single-file generator safetensors saved by training
#    (see outputs/flashtalk_stage2/.../checkpoint_<step>/generator_<step>.safetensors).
python tools/export_stage2_model_to_flashtalk_style.py \
    --src outputs/flashtalk_stage2/<run>/checkpoint_<step>/generator_<step>.safetensors \
    --output_dir <SoulX-FlashTalk-root>/models/SoulX-FlashTalk-14B \
    --num_shards 4

# 2. (Optional but recommended) Drop the sharded files in place of the
#    matching files inside the official SoulX-FlashTalk-14B checkpoint
#    directory so that auxiliary configs / tokenizers come along for free.
#    --output_dir above already does this when you point it at the
#    SoulX-FlashTalk-14B directory you downloaded from Huggingface.

# 3. Clone & install SoulX-FlashTalk, then run their inference script
#    with --ckpt_dir pointing at the (now-overwritten) model directory.
git clone https://github.com/Soul-AILab/SoulX-FlashTalk.git
cd SoulX-FlashTalk
# follow their README to install requirements, then:
bash inference_script_multi_gpu.sh   # edit --ckpt_dir inside the .sh
# or
bash inference_script_single_gpu.sh
```

If you used the included `tools/export_stage2_model_to_flashtalk_style.py
--index_path` mode, the shard layout will *exactly* match the upstream
release, so you can drop the new shards in next to (or on top of) the
existing `diffusion_pytorch_model-*.safetensors` files. Use `--num_shards
4` to match the official 4-shard release if you do not have the
original index.

---

## License

Code released under the project [LICENSE](LICENSE.txt). The base
InfiniteTalk weights and the upstream model components remain under their
respective licenses; please consult them before using this code in any
commercial or downstream project.
