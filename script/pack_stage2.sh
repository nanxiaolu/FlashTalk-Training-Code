#!/usr/bin/env bash
# Stage-A -> Stage-B: pack stage-2 payload files into a single LMDB database.
#
# IMPORTANT: ``--group_size`` MUST equal the number of GPUs used at stage-2
# training time. Stage-2 trains under FSDP with self-forcing++ chunked
# generation: each sample stores its own ``selected_k`` (1..K_max) and the
# student rolls out exactly ``selected_k`` forward passes. FSDP synchronizes
# gradients across ranks after every forward pass, so every rank in a global
# batch must perform the same number of forwards. By packing the LMDB in
# groups of ``group_size`` consecutive samples that share the same K, we
# guarantee that a global batch of ``group_size`` ranks operates on a
# uniform K. If you change the GPU count, re-run this script with the new
# ``--group_size`` (or repack with a different group size).
set -euo pipefail

# Defaults below target the bundled 32-video minimum viable example and 8 GPUs.
# For real training, swap to the LMDB you produced from the full dataset, e.g.
#   --payload_dir     processed_data/talkcuts/train/stage2_sample_6400.payloads \
#   --output_lmdb_path processed_data/talkcuts/train/stage2_sample_6400.lmdb \
#   --num_samples     6400
python tools/payload_files_to_lmdb.py \
    --payload_dir processed_data/example/train/example_stage2.payloads \
    --output_lmdb_path processed_data/example/train/example_stage2.lmdb \
    --num_samples 32 \
    --shuffle_k_groups true \
    --group_size 8
