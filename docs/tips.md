# 避坑指南与常见问题 (Tips)

在这里我们列出了复现训练与调试过程中最容易踩的坑，请在遇到异常时首先检查此处。

## 1. Stage 2 训练死锁/卡死：`group_size` 必须等于 GPU 数量

**症状**：Stage 2 训练启动后一直 hang 住，或者在第一个 global-batch 同步时报错/死锁。

**根本原因**：
Stage 2 采用了 Self-Forcing++ 的分块展开策略。这意味着在一个 Batch 中的不同样本可能对应不同的生成分块数量（`selected_k ∈ {1..K_max}`）。
Stage 2 同时使用了 FSDP (Fully Sharded Data Parallel)，FSDP 会在每次前向传播（forward pass）后同步梯度。
如果处于同一个 global batch 的各个 GPU 取到的样本 `selected_k` 不同，就会导致不同 rank 执行前向传播的次数不一样。当一个 rank 在等第 $N+1$ 次同步，而另一个 rank 已经执行完了 $N$ 次时，整个训练就会永久死锁。

**解决方案**：
为了解决这个问题，我们在打包数据为 LMDB 格式时，会通过 `--group_size` 参数强制将具有相同 `selected_k` 值的连续样本聚类打包。
因此，**`--group_size` 的值必须严格等于您进行 Stage 2 训练所使用的总 GPU 数量**。
* 如果您使用的是 8 张卡，可以直接使用我们随项目发布（或示例代码中默认打包好的）的基于 `--group_size 8` 的数据集。
* 如果您使用非 8 卡（如 16、32、64卡），**您必须**修改数据处理脚本，使用新的显卡数量作为 `--group_size` 重新执行 `tools/payload_files_to_lmdb.py` 对 Stage 2 的 payload 文件进行重新打包。

## 2. 内存 (RAM) 峰值过高导致进程被杀 (OOM Killer)

**症状**：在运行 Stage 2 初始化加载模型（且还没开始训练）时，系统提示 `Killed` 或 dmesg 显示 OOM-killer 杀死了 Python 进程。

**根本原因**：
Stage 2 初始化时需要实例化生成器、真实分数模型和假分数模型三个 DiT 14B 实例。在进行 FSDP 划分之前，所有参数同时存在于内存中，导致物理内存占用在短时间内飙升至约 **1.6 TB**。

**解决方案**：
* 确保您的机器具有超过 1.6TB 的可用物理内存。
* **内存优化捷径**：如果您受到物理内存限制，可以修改初始化代码逻辑：在实例化并深拷贝每个大模型实例之后、进行下一个实例化之前，**尽早通过 FSDP 对其进行包装 (wrap)**。这样 FSDP 会自动将参数分片并抛弃无用部分，能极其显著地削平初始化阶段的内存使用峰值。正常训练开启后，显存总占用一般只需保持在 400GB+。

## 3. 多卡训练报错或无故停止

**症状**：使用非 8 卡训练配置时，程序由于数据无法整除报错，或者丢弃大量分块尾部数据。

**根本原因**：
除了 `group_size` 外，预处理 LMDB 的 `lmdb_num_samples` (总样本数) 在计算分桶时对卡数也很敏感。如果数据量不足以填满 `K_max` 个组，打包脚本会自动丢弃 (drop tail) 这些填不满的部分，导致最终实际可用样本变少甚至不足以分配给所有 rank。

**解决方案**：
* 重新打包时，请参考 [`docs/hardware_scaling.md`](hardware_scaling.md) 设置与卡数完全匹配的梯度累积步数 (`grad_accum_steps`)。
* 确保预处理时的数据总量 `lmdb_num_samples` 足够大（作为经验法则，应满足 `lmdb_num_samples >= 2 * max_steps * world_size` 且每个 K 桶能被 `group_size` 整除）。