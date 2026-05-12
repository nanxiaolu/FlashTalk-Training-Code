# Baidu NetDisk assets

Everything except this source tree is hosted on Baidu NetDisk:

> *(link to be filled in by the maintainer)*

A single folder contains all the binary artefacts. Below is what each file
is for, what to do with it, and which downstream script consumes it.

| Archive / dir | Size | Description | Where it goes |
|---|---:|---|---|
| `weights.tar` | ~104 GB | All model weights required at training / inference time: Wan2.1 (VAE, DiT, T5, CLIP), chinese-wav2vec2-base, InsightFace antelopev2, RVM (rvm_mobilenetv3.pth), SyncNet/S3FD/OneAlign checkpoints, etc. | `tar -xf weights.tar -C ./` and confirm the `weights/` layout. |
| `stage1_save_model.safetensors` | ~71 GB | Pre-trained Stage-1 generator (output of full-parameter flow-matching fine-tune on the full LMDB). | (a) use as `init_stage1_full` in `config/train_stage2.yaml` to start Stage-2; (b) use as `init_stage1_full` in `config/val_stage1.yaml` to validate Stage-1 quality. |
| `stage2_save_model.safetensors` | ~71 GB | Pre-trained Stage-2 generator (after DMD distillation). The "release" checkpoint. | Place it under `outputs/flashtalk_stage2/<your_run>/checkpoint_<step>/generator_<step>.safetensors`, copy or symlink a `training_state.pt` next to it (or accept that resume won't restore optimizer state), then point `resume_from` in `config/val_stage2.yaml` at that directory. |
| `stage1_sample_25030_lmdb/` | ~280 GB | Stage-1 training LMDB: 25 030 preprocessed samples (33-frame clips) packed from the full TalkCuts split. | Move / symlink to `processed_data/talkcuts/train/stage1_sample_25030.lmdb` (the path that `config/train_stage1.yaml` defaults to). |
| `stage2_sample_6400_lmdb/` | ~280 GB | Stage-2 training LMDB: 6 400 preprocessed samples chunked with `K_max=5`, packed with `--group_size 8` (i.e. for 8-GPU FSDP). | Move / symlink to `processed_data/talkcuts/train/stage2_sample_6400.lmdb`. If you train on a different GPU count, re-pack with `script/pack_stage2.sh --group_size <gpus>` — see [stage2_k_grouping.md](stage2_k_grouping.md). |
| `talkcuts.tar.gz` | ~200 MB | Preprocessed TalkCuts validation feature cache. Contains `val_data.csv` and `val/feature/` with one folder per validation sample (`context.pt` / `full_emb.pt` / `clip_fea.pt` / ...) plus a shared `val/feature/context_null.pt`. The training LMDB is uploaded separately (see `stage{1,2}_sample_*_lmdb/` above). | `tar -xzf talkcuts.tar.gz -C processed_data/` — should land directly under `processed_data/talkcuts/{val_data.csv,val/feature/}`. |
| `example_data.tar.gz` | ~199 MB | 32 training + 12 validation raw video/audio pairs from TalkCuts, used to smoke-test the preprocess pipeline on a single node. | `tar -xzf example_data.tar.gz -C processed_data/` — should land under `processed_data/example/train/{video,audio}/` and `processed_data/example/val/{video,audio}/`. The CSVs (`processed_data/example/{train,val}_data.csv`) already reference these paths. |

## Minimal deployment (inference only)

If you only want to run the released Stage-2 model:

1. `weights.tar` → extract.
2. `stage2_save_model.safetensors` → place under a checkpoint dir as
   described above.
3. `talkcuts.tar.gz` → extract for the validation CSV.
4. `example_data.tar.gz` → optional, for the bundled 12-clip val set.
5. `bash script/val_stage2.sh` after editing `config/val_stage2.yaml`'s
   `resume_from`.

## Full training reproduction

You additionally need both LMDB folders (`stage1_sample_25030_lmdb/` and
`stage2_sample_6400_lmdb/`). Without them you'd have to re-run preprocess on the
full TalkCuts dataset, which is a multi-day GPU job — see
[data_preparation.md](data_preparation.md) if you want to go that route
(e.g. on a different dataset).
