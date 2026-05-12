# Training

Both Stage-1 and Stage-2 launchers use `torchrun --nproc_per_node=8
--standalone`. All paths in the YAMLs are relative to the project root.

---

## Stage 1 — full-parameter flow-matching

### Configuration

`config/train_stage1.yaml`. The defaults match what we used to produce
`stage1_save_model.safetensors`:

| Field | Default | Notes |
|---|---|---|
| `lmdb_path` | `processed_data/talkcuts/train/stage1_sample_25030.lmdb` | Full TalkCuts split (download or build via preprocess + pack). |
| `max_steps` | 700 | One step = one global batch = 8 samples × `grad_accum_steps`. |
| `save_interval` | 100 | Single cadence for Stage 1. |
| `gen_lr` | `2.0e-6` | Generator learning rate. |
| `warmup_steps` | 0 | Warmup helps if you lower `gen_lr` further. |
| `grad_accum_steps` | 4 | Effective global batch = 8 × 4 = 32. |
| `flow_match_weight` | 0.75 | Flow-matching loss weight. |
| `face_loss_weight` | 0.75 | Face-region L2 loss weight. |
| `temporal_loss_weight` | 0.25 | Temporal-consistency loss weight. |
| `gradient_checkpointing` | true | Required to fit 14B + ✕8 ranks on 80GB. |
| `fsdp_cpu_offload` | false | Stage 1 fits without CPU offload. |
| `debug` | false | Set to `true` to skip pretrained weight loading and shrink the model — useful for smoke-testing. |

### Launch

```bash
bash script/train_stage1.sh
```

Outputs land under
`outputs/flashtalk_stage1/<auto_run_name>/<timestamp>/`, with subfolders:

```
checkpoint_<step>/model_<step>.safetensors   # generator-only snapshot
tensorboard/                                  # scalars + image samples
train.log
```

### Resume

```yaml
# config/train_stage1.yaml
resume_from: outputs/flashtalk_stage1/<run>/<ts>/checkpoint_500/
```

The optimizer / scheduler state is restored from `training_state.pt` next
to the safetensors.

---

## Stage 2 — DMD distillation + self-forcing++

### Configuration

`config/train_stage2.yaml`. Notable fields:

| Field | Default | Notes |
|---|---|---|
| `lmdb_path` | `processed_data/talkcuts/train/stage2_sample_6400.lmdb` | Must have been packed with `--group_size 8` for 8-GPU training (see [stage2_k_grouping.md](stage2_k_grouping.md)). |
| `init_stage1_full` | `""` | Path to Stage-1 generator safetensors (`stage1_save_model.safetensors`). If empty, DMD starts from the base InfiniteTalk weights. |
| `max_steps` | 140 | Stage 2 converges fast. |
| `save_interval_first_stage` | 50 | Cadence for `step < save_interval_switch_step`. |
| `save_interval_switch_step` | 100 | Phase boundary. |
| `save_interval_second_stage` | 20 | Cadence for `step >= switch_step`. Tighter because the best-quality window is narrow. |
| `gen_lr` / `critic_lr` | `2.0e-6` / `4.0e-7` | Generator vs fake-score. |
| `gen_betas` / `critic_betas` | `[0.9, 0.95]` / `[0.9, 0.99]` | |
| `dmd_loss_weight` | 1.0 | Score-difference KL. |
| `temporal_align_weight` | 1.0 | Cross-K consistency. |
| `denoising_step_list` | `"1000,750,500,250"` | The 4 student timesteps after timestep shifting. |
| `text_guide_scale` / `audio_guide_scale` | 3.0 / 4.0 | Teacher CFG scales (student becomes CFG-free implicitly). |
| `use_inject_motion_frames` | true | Inject motion-warped frames into the student input. |
| `keep_k_chunks` | -1 | -1 means use the LMDB's `selected_k`; positive values clamp it. |
| `fsdp_cpu_offload` | true | Stage 2 has 3 networks under FSDP; CPU offload helps. |

### Launch

```bash
# Edit config/train_stage2.yaml:
#   init_stage1_full: outputs/flashtalk_stage1/.../checkpoint_700/model_700.safetensors
bash script/train_stage2.sh
```

Outputs:

```
outputs/flashtalk_stage2/<auto_run_name>/<timestamp>/
├── checkpoint_<step>/
│   ├── generator_<step>.safetensors
│   ├── critic_<step>.safetensors
│   └── training_state.pt
├── iter_<step>/rank_<r>_gen_exit_*  # per-rank debug dumps from the rollout
├── tensorboard/
└── train.log
```

### Resume

```yaml
# config/train_stage2.yaml
resume_from: outputs/flashtalk_stage2/<run>/<ts>/checkpoint_120/
```

When `resume_from` is set, `init_stage1_full` is ignored (resume takes
precedence so you do not double-load the generator).

### About FSDP and K consistency

This is the biggest gotcha in Stage 2. Please read
[stage2_k_grouping.md](stage2_k_grouping.md) before changing the GPU count.

---

## Smoke test (debug mode)

Both training scripts honour `debug: true`, which:

1. Sets `num_layers=1` so the DiT trunk is tiny.
2. Skips loading pretrained generator / real-score weights.

This lets you verify a single training step end-to-end in ~3-5 min,
including FSDP wrapping. A typical smoke session looks like:

```bash
# 1. Preprocess + pack on the 32-clip example
bash script/preprocess_stage1.sh && bash script/pack_stage1.sh
bash script/preprocess_stage2.sh && bash script/pack_stage2.sh

# 2. Copy the train config and override the LMDB + debug flag:
#    lmdb_path: processed_data/example/train/example_stage1.lmdb
#    max_steps: 1
#    debug: true
cp config/train_stage1.yaml outputs/smoke_train_stage1.yaml
$EDITOR outputs/smoke_train_stage1.yaml
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone \
  train_flashtalk_stage1.py --config outputs/smoke_train_stage1.yaml

# Same idea for stage 2.
```

The smoke run should produce a `checkpoint_1/` directory and exit
cleanly.
