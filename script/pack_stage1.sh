#!/usr/bin/env bash
# Stage-A -> Stage-B: pack stage-1 payload files into a single LMDB database.
# Adjust paths to match your stage-1 preprocess output (payload_dir from
# config/preprocess_stage1_example.yaml or your custom preprocess config).
set -euo pipefail

python tools/payload_files_to_lmdb.py \
    --payload_dir processed_data/example/train/example_stage1.payloads \
    --output_lmdb_path processed_data/example/train/example_stage1.lmdb \
    --shuffle_k_groups false
