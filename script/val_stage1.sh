#!/usr/bin/env bash
# Validation for STAGE-1 trained weights.
#
# IMPORTANT: We invoke train_flashtalk_stage2.py here on purpose (NOT a typo).
# That script also drives validation by reusing the stage-2 inference path
# (CFG, denoising_step_list, motion injection, ...). With `val_only=true` and
# `init_stage1_full=<your stage-1 safetensors>` in config/val_stage1.yaml, only
# the generator is loaded with the stage-1 weights and run through inference.
#
# Before running, edit config/val_stage1.yaml and set:
#   init_stage1_full: outputs/flashtalk_stage1/.../checkpoint_{step}/model_{step}.safetensors
set -euo pipefail

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --standalone train_flashtalk_stage2.py \
  --config config/val_stage1.yaml
