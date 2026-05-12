#!/usr/bin/env bash
# Stage-1 preprocess (minimum viable example).
# Walks the 32-video demo set bundled under processed_data/example/train/ and writes
# per-sample payload files into processed_data/example/train/example_stage1.payloads.
# After this script finishes, run script/pack_stage1.sh to pack the payloads
# into a single LMDB suitable for training.
#
# To preprocess your own dataset, copy config/preprocess_stage1_example.yaml,
# point ``annotation_file`` at your CSV (with columns video,input_audio,prompt)
# and update ``payload_dir`` / ``lmdb_num_samples`` accordingly.
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage1.py \
  --config config/preprocess_stage1_example.yaml
