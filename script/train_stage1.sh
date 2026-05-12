#!/usr/bin/env bash
# Stage-1 training: full-parameter flow-matching fine-tune for FlashTalk.
# Reads config/train_stage1.yaml (mode=train). For the preprocess step,
# use script/preprocess_stage1.sh + config/preprocess_stage1_example.yaml.
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage1.py \
  --config config/train_stage1.yaml
