#!/usr/bin/env bash
# Stage-2 preprocess (minimum viable example).
# Walks the 32-video demo set bundled under processed_data/example/train/ and writes
# per-sample payload files into processed_data/example/train/example_stage2.payloads.
# After this script finishes, run script/pack_stage2.sh to pack the payloads
# into a single LMDB suitable for stage-2 DMD training.
#
# To preprocess your own dataset, copy config/preprocess_stage2_example.yaml,
# point ``annotation_file`` at your CSV and update ``payload_dir`` /
# ``lmdb_num_samples`` / ``stage2_k_max`` accordingly.
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage2.py \
  --config config/preprocess_stage2_example.yaml
