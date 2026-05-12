#!/usr/bin/env bash
# Validation for STAGE-2 trained weights.
#
# Driver: train_flashtalk_stage2.py with `val_only=true` and `resume_from=<stage-2 checkpoint dir>`.
# The checkpoint directory must contain generator_{step}.safetensors and training_state.pt.
#
# Before running, edit config/val_stage2.yaml and set:
#   resume_from: outputs/flashtalk_stage2/.../checkpoint_{step}/
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage2.py \
  --config config/val_stage2.yaml
