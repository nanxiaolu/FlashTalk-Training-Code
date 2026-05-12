# Stage-2 K grouping & FSDP

This is the single most important deployment detail in the repo. If you
are about to launch a long Stage-2 run on a non-default GPU count, please
read all of it.

## Why each LMDB row stores a `selected_k`

Stage 2 trains the student under **self-forcing chunked generation**: a
single sample is a sequence of `selected_k` overlapping 33-frame chunks.
The student auto-regressively generates them, conditioning each chunk on
the motion frames of the previously generated chunk, so it learns to
*recover* from accumulated noise rather than collapsing.

`selected_k ∈ {1, 2, ..., K_max}` is fixed at preprocess time. We pick
`K_max=5` for the released `stage2_sample_6400.lmdb`. The distribution of K
values is approximately uniform: every K bucket holds the same number of
samples (`quota_per_k = ceil(num_samples / K_max)`).

## Why FSDP cares

Under FSDP, after every forward pass each rank issues an
`all_reduce` / `all_gather` for the sharded parameters and gradients.
This means **every rank in the global batch must execute the same number
of forward passes**, otherwise some ranks will busy-wait for sync
primitives that never arrive on the other ranks → deadlock.

In Stage 2 the number of forward passes per sample equals its
`selected_k`. So in a global batch of N ranks, **all N samples must share
the same `selected_k`**.

## How we enforce it

In `tools/payload_files_to_lmdb.py`, when `--shuffle_k_groups true`, we
write the LMDB in *K-homogeneous groups* of size `--group_size`:

```
K=1: [sample_a, sample_b, sample_c, sample_d, sample_e, ...]   <-- N consecutive K=1 samples
K=2: [sample_f, sample_g, sample_h, sample_i, sample_j, ...]   <-- N consecutive K=2 samples
K=1: [sample_k, sample_l, ...]
...
```

If `--group_size 8` matches the number of training GPUs, then any
contiguous slice of 8 consecutive LMDB rows is K-homogeneous, which is
exactly what FSDP requires.

The shuffle randomizes *across groups* (so adjacent groups don't share
metadata) but never *within* a group.

## What the dataloader does at train time

In `mode=train` we bypass `DistributedSampler` entirely and read by a
deterministic key formula (see `_prepare_batch_tensors_from_lmdb` in
`train_flashtalk_stage2.py`):

```python
lmdb_key = (step - 1) * world_size * grad_accum_steps \
         + accum_idx * world_size + rank
```

So for any fixed `(step, accum_idx)` the `world_size` ranks read
`world_size` **consecutive** LMDB rows. With `world_size == group_size`,
those rows are a K-homogeneous group by construction. The
`grad_accum_steps` factor only moves the window forward by groups of
`group_size`, so micro-batch forwards (which is where FSDP syncs) stay
K-homogeneous.

`batch_size=1` is asserted at startup; combined with the formula above
each rank consumes one LMDB row per micro-batch.

The Stage-2 train log prints a one-line summary so you can verify:

```
LMDB train mode enabled: path=processed_data/talkcuts/train/stage2_sample_6400.lmdb, num_samples=6400
```

If you see ranks diverge in their `Step <s> | k=<...>` log lines, your
LMDB was packed for a different `group_size`. Repack.

## Changing the GPU count

Note: Stage-2 training does **not** fit on 4×A800 (the activations from
the self-forcing rollouts plus the real-score / fake-score critics OOM).
8×A800 is the minimum we have validated. Pick a `group_size` that
matches your actual world size; the example below shows scaling up to
16 GPUs.

```bash
# Repack the same payloads for 16-GPU training:
python tools/payload_files_to_lmdb.py \
    --payload_dir      processed_data/talkcuts/train/stage2_sample_6400.payloads \
    --output_lmdb_path processed_data/talkcuts/train/stage2_sample_6400_gpu16.lmdb \
    --num_samples      6400 \
    --shuffle_k_groups true \
    --group_size       16
```

Then point `config/train_stage2.yaml`'s `lmdb_path` at the new file and
launch with `--nproc_per_node=16`. For ≥64 GPUs you should also rerun
preprocess with a larger `lmdb_num_samples` so that
`max_steps * world_size <= num_samples` — see the **Hardware** table in
the top-level `README.md`.

Tail samples (`< group_size` per K bucket) are dropped during repack;
this is reported as `dropped_tail` in the packer output. Expect to lose
up to `K_max * (group_size - 1)` samples — for `K_max=5, group_size=16`
that is at most 75 samples out of 6 400.

## What if I really want non-uniform batches?

You can disable the requirement by changing the training code to **not**
use FSDP (set `use_fsdp: false`), at the cost of pulling the full model
onto every rank. We never tested that path with the released weights and
do not recommend it.
