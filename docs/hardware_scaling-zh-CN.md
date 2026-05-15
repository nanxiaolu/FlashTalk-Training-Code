# 硬件扩展配置指南

本仓库的默认配置基于 **8× A800 (80 GB)**。如果你要在 16/32/64 等其它卡数上跑，**必须**修改两类参数：

1. **训练 YAML 里的梯度累积步数**：项目默认是 8 卡 × 4 步梯度累积，即有效 batch size = 32，请按目标卡数调整。
2. **Stage 2 LMDB 的 `--group_size`**：`group_size` 表示 LMDB 中每 `group_size` 个样本共享相同的 $K$。更换 GPU 数量时，需满足 `group_size % GPU_num == 0`，并推荐 **`group_size` 与实际 GPU 数一致**。可在由 **[payload 目录打包为 LMDB](../script/pack_stage2.sh)** 时指定 `group_size`，也可将已有 LMDB **[重打包为新的 `group_size`](../tools/repack_stage2_lmdb_group_size.py)**。原因在于 FSDP 等分布式框架在每次前向/反向中要求所有 GPU（rank）参与梯度同步等**集合通信（Collective Communication）**；若各卡分配的样本窗口数 $K$ 不一致，样本少的卡会率先结束循环，其余卡仍在反向并等待同步，容易导致进程**死锁（挂起）**。

> Stage 1 不存在 K-grouping 问题。

## 操作步骤

### Stage 1（只改 YAML）

* 修改 `config/train_stage1.yaml` 中的 `grad_accum_steps` 。
* 启动脚本 `script/train_stage1.sh` 里的 `--nproc_per_node` 改成实际卡数。

### Stage 2（YAML + 更新 LMDB）

1. 修改 `config/train_stage2.yaml` 中的 `gen_grad_accum_steps` 与 `critic_grad_accum_steps`。
2. 更新 LMDB  
   可在将 **payload 目录打包为 LMDB** 时指定 `group_size`，或将已有 LMDB **重打包**为新的 `group_size`。

由 payload 目录生成 LMDB：

```bash
python tools/payload_files_to_lmdb.py \
    --payload_dir       processed_data/talkcuts/train/stage2_sample_6400.payloads \
    --output_lmdb_path  processed_data/talkcuts/train/stage2_sample_6400_gs<N>.lmdb \
    --shuffle_k_groups  true \
    --group_size        <N>          # 与实际 GPU 数严格一致
```

由已有 LMDB 重打包（LMDB → LMDB）：

```bash
python tools/repack_stage2_lmdb_group_size.py \
    --input_lmdb_path processed_data/talkcuts/train/stage2_sample_6400.lmdb \
    --input_group_size 8 \
    --output_group_size 32 \
    --output_lmdb_path processed_data/talkcuts/train/stage2_sample_6400_gs32.lmdb
```

3. 把 `config/train_stage2.yaml` 的 `lmdb_path` 改成新 LMDB。
4. `script/train_stage2.sh` 里的 `--nproc_per_node` 同步改成实际卡数。
