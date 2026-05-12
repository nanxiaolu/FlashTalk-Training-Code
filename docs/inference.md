# Inference (export to SoulX-FlashTalk)

This repository covers **training** of the two-stage FlashTalk distillation pipeline.
The final serving / streaming inference path is **not** included here — for
that we rely on the upstream
[SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk) project,
which ships:

* `inference_script_single_gpu.sh` — single-GPU inference, ~64 GB VRAM
  (`--cpu_offload` brings it down to ~40 GB).
* `inference_script_multi_gpu.sh` — multi-GPU inference with sub-second
  start-up latency on 8×H800.
* `gradio_app.py` — interactive demo.

The only manual step needed to plug a checkpoint trained in this repo
into SoulX-FlashTalk is **reformatting the safetensors layout**.

## 1. Checkpoint formats

* What this repo saves:
  ```
  outputs/flashtalk_stage2/<run>/checkpoint_<step>/
  ├── generator_<step>.safetensors      # single file, ~71 GB, our format
  ├── training_state.pt                 # optimizer / LR / step bookkeeping
  ├── real_score_<step>.safetensors     # critic, not used at inference
  └── fake_score_<step>.safetensors     # critic, not used at inference
  ```
* What SoulX-FlashTalk expects under `--ckpt_dir`:
  ```
  models/SoulX-FlashTalk-14B/
  ├── diffusion_pytorch_model-00001-of-00004.safetensors
  ├── diffusion_pytorch_model-00002-of-00004.safetensors
  ├── diffusion_pytorch_model-00003-of-00004.safetensors
  ├── diffusion_pytorch_model-00004-of-00004.safetensors
  ├── diffusion_pytorch_model.safetensors.index.json
  └── (config / tokenizer / scheduler files from the HF release)
  ```

`tools/export_stage2_model_to_flashtalk_style.py` converts the first
layout into the second. It (a) shards the generator tensors into N
balanced files, (b) casts floats to bf16 (matching the upstream release
storage dtype), and (c) writes the `weight_map` index json.

## 2. Convert and copy in place

The cleanest workflow is to point `--output_dir` at the
SoulX-FlashTalk-14B directory you downloaded from Huggingface — the
exporter overwrites just the safetensors + index and leaves the rest of
the directory (configs, tokenizer, scheduler) intact.

```bash
# Download the official model bundle once.
git clone https://github.com/Soul-AILab/SoulX-FlashTalk.git
cd SoulX-FlashTalk
huggingface-cli download Soul-AILab/SoulX-FlashTalk-14B \
    --local-dir ./models/SoulX-FlashTalk-14B
huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
    --local-dir ./models/chinese-wav2vec2-base
cd -

# Convert the trained Stage-2 checkpoint into the same sharded layout.
python tools/export_stage2_model_to_flashtalk_style.py \
    --src outputs/flashtalk_stage2/<run>/checkpoint_<step>/generator_<step>.safetensors \
    --output_dir <SoulX-FlashTalk>/models/SoulX-FlashTalk-14B \
    --num_shards 4
```

Tips:

* Use `--num_shards 4` to match the official 4-shard release; the
  upstream model loader allocates buffers based on the shard count, and
  4 is the value the team trained / benchmarked with.
* `--index_path <path_to_official>/diffusion_pytorch_model.safetensors.index.json`
  reuses the *exact* upstream weight map — useful if you want
  bit-for-bit shard alignment with the official release. Note that all
  keys must match exactly; missing or extra tensors will raise.

## 3. Run SoulX-FlashTalk inference

Follow the upstream README for environment setup (note: it uses
`torch==2.7.1 + CUDA 12.8` and `flash_attn==2.8.0.post2`, different from
the training environment in this repo). Then edit the
`--ckpt_dir` inside the chosen inference shell script to point at the
overwritten model directory, and launch:

```bash
cd SoulX-FlashTalk
# Edit --ckpt_dir in this file to point at ./models/SoulX-FlashTalk-14B
bash inference_script_multi_gpu.sh
# or
bash inference_script_single_gpu.sh
```

## Why two environments?

SoulX-FlashTalk is optimized for H-series GPUs and uses a newer CUDA /
PyTorch combo that doesn't match the A-series training environment.
Keeping the two environments separate (one
`flashtalk_cxy` for training in this repo, another `flashtalk` for
inference in the upstream repo) is the path with fewest surprises. The
exported safetensors themselves are interchangeable.
