#!/usr/bin/env bash
# Stage-2 training: full-parameter DMD self-forcing++ for FlashTalk.
# Reads config/train_stage2.yaml (mode=train). For the preprocess step,
# use script/preprocess_stage2.sh + config/preprocess_stage2_example.yaml.
# Initializes generator/real_score/fake_score from stage-1 weights when
# ``init_stage1_full`` is set inside config/train_stage2.yaml.
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage2.py \
  --config config/train_stage2.yaml
