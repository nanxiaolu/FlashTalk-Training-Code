# Hardware Scaling Guide

[Chinese version](hardware_scaling-zh-CN.md)

The default configuration in this repository targets **8 x A800 (80 GB)**. If you train with 16, 32, 64, or any other number of GPUs, you **must** adjust two categories of settings:

1. **Gradient accumulation steps in the training YAML files**: the default global batch size is 8 GPUs x 4 gradient accumulation steps = 32. Adjust this according to your setup.
2. **The Stage 2 LMDB `--group_size`**: `group_size` means every `group_size` samples in the LMDB share the same K value. When changing the GPU count, `group_size` must satisfy `group_size % GPU_num == 0`; we recommend `group_size == GPU_num`. You can set `group_size` when converting payloads to LMDB with [payloads -> LMDB](../script/pack_stage2.sh), or convert an existing LMDB with [LMDB -> LMDB](../tools/repack_stage2_lmdb_group_size.py). This requirement comes from FSDP: during every forward/backward pass, all GPU ranks must participate in collective communication. If different GPUs receive samples with different window counts $K$, ranks with fewer windows finish early while ranks with more windows continue backpropagation and wait for synchronization, which causes a deadlock.

> Stage 1 does not have the K-grouping issue.

## Steps

### Stage 1 (YAML only)

* Update `grad_accum_steps` in `config/train_stage1.yaml`.
* Set `--nproc_per_node` in `script/train_stage1.sh` to the actual number of GPUs.

### Stage 2 (YAML + LMDB)

1. Update `gen_grad_accum_steps` and `critic_grad_accum_steps` in `config/train_stage2.yaml`.
2. Update the LMDB. You can either set `group_size` when converting `payload_dir` to LMDB, or convert an existing LMDB to a new `group_size`.

Payloads -> LMDB:

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir       processed_data/talkcuts/train/stage2_sample_6400.payloads \
    --output_lmdb_path  processed_data/talkcuts/train/stage2_sample_6400_gs<N>.lmdb \
    --shuffle_k_groups  true \
    --group_size        <N>          # Must exactly match the actual GPU count.
```

LMDB -> LMDB:

```bash
python tools/repack_stage2_lmdb_group_size.py \
    --input_lmdb_path processed_data/talkcuts/train/stage2_sample_6400.lmdb \
    --input_group_size 8 \
    --output_group_size 32 \
    --output_lmdb_path processed_data/talkcuts/train/stage2_sample_6400_gs32.lmdb
```

3. Update `lmdb_path` in `config/train_stage2.yaml` to the new LMDB.
4. Set `--nproc_per_node` in `script/train_stage2.sh` to the actual number of GPUs.
