# 硬件扩展配置指南

本仓库的默认配置基于 **8× A800 (80 GB)**。如果你要在 16/32/64 等其它卡数上跑，**必须**修改两类参数：

1. **训练 YAML 里的梯度累积步数**：项目默认是 8卡 * 4 步梯度累积，即batchsize=32，根据你的需要修改。
2. **Stage 2 LMDB 的 `--group_size`**：由于 FSDP 的原因，第二阶段训练中不同GPU的样本的chunk数 K 必须一致，`group_size`的含义是lmdb里每group_size个样本的K一致。该值必须等于实际训练用的 GPU 数。如果你换了GPU数，group_size需要满足(group_size % GPU_num == 0， 推荐group_size == GPU_num)。可以在[pyloads->lmdb](script/pack_stage2.sh)时指定group_size，也可以在[lmdb->lmdb](tools/repack_stage2_lmdb_group_size.py)时转化group_size。

> Stage 1 不存在 K-grouping 问题。

## 操作步骤

### Stage 1（只改 YAML）

* 修改 `config/train_stage1.yaml` 中的 `grad_accum_steps` 。
* 启动脚本 `script/train_stage1.sh` 里的 `--nproc_per_node` 改成实际卡数。

### Stage 2（YAML + 更新 LMDB）

1. 修改 `config/train_stage2.yaml` 中的 `gen_grad_accum_steps` 与 `critic_grad_accum_steps`。
2. 更新 LMDB
可以在payload_dir -> LMDB时候指定group_size，或者将已有LMDB转化新的group_size
payload_dir -> LMDB：
   ```bash
   python tools/payload_files_to_lmdb.py \
       --payload_dir       processed_data/talkcuts/train/stage2_sample_6400.payloads \
       --output_lmdb_path  processed_data/talkcuts/train/stage2_sample_6400_gs<N>.lmdb \
       --shuffle_k_groups  true \
       --group_size        <N>          # 与实际 GPU 数严格一致
   ```

LMDB -> LMDB:
   ```bash
    python tools/repack_stage2_lmdb_group_size.py \
    --input_lmdb_path processed_data/talkcuts/train/stage2_sample_6400.lmdb \
    --input_group_size 8 \
    --output_group_size 32 \
    --output_lmdb_path processed_data/talkcuts/train/stage2_sample_6400_gs32.lmdb
   ```

3. 把 `config/train_stage2.yaml` 的 `lmdb_path` 改成新 LMDB。
4. `script/train_stage2.sh` 里的 `--nproc_per_node` 同步改成实际卡数。
