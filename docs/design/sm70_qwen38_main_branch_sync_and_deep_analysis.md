# `origin/main` (`6f3557c97` / PR #1203) 深度代码剖析、分支合并与增量优化报告

- **基线演进**：`be3722bc0`（旧基线） $\rightarrow$ **`6f3557c97`**（最新 `origin/main`，`Merge pull request #1203 from 1CatAI/agent/flashnext-16gb-fast-20261011`，共 **178 个提交、176 个文件变更、+20,247 / -600 行代码**）
- **当前工作分支**：`arena/5368f936-1cat-vllm`（已完成与 `origin/main` `6f3557c97` 的全量合并与冲突消解，并在此基础上完成 `P14`–`P16` 增量优化）

---

## 一、为什么此前分支与最新 `main` 存在代差？

本会话最初从 `main` 的历史提交 `be3722bc0`（`[Kernel] Reduce SM70 mHC projection and output overhead (#1172)`）拉出。在此之后，`origin/main` 合入了巨型 PR **#1203**（`6f3557c97`，来自 `agent/flashnext-16gb-fast-20261011` 分支），涉及 **176 个文件、+20,247 行新增代码**。

此前未同步 `origin/main` (`6f3557c97`) 带来了两个直接问题：
1. **文件覆盖冲突风险**：我们在 `be3722bc0` 上修改的 32 个文件中，有 **15 个核心文件**（`qsa.py`、`ops/qsa.py`、`common/qsa_cache.py`、`indexer_qsa.py`、`ops/hc.py`、`hyperconnection.py`、`ple_layer.py`、`model.py`、`mtp.py`、`sm70_fp16_gemv.py`、`qwen_gdn_linear_attn.py`、`qwen3_next.py`、`cudagraph_utils.py`、`model_runner.py`、`gpu_model_runner.py`）在 `origin/main` (`6f3557c97`) 中也发生了重大演进。若直接拿基于 `be3722bc0` 打包的文件覆盖到最新 `main` 代码树，会冲掉 `6f3557c97` 新增的 `sharded_hc_project`、`prepare_sharded_hc_storage`、`_router_batch_storage`、`MTPLosslessConfig`、`share_target_io_weights`、`_TARGET_HOST_KV`、`allocate_host_kv_cache_pool`、`get_capture_context` 等关键接口。
2. **新架构特性未纳入分析**：`origin/main` (`6f3557c97`) 针对 **SM70 (V100) 上的 Qwen3.8-Flash-Next (含 256K Host-KV、GGUF Expert Cache、PLE Fast Lane、Sharded HC、Sparse-Target Speculative Sampling、Balanced Prefill Chunks)** 引入了 8 大核心子系统，并在 `benchmarks/flashnext_prefill/HANDOFF.md` 中记录了实测 **32K TTFT 从 8.08s 降至 7.50s** 的延迟分解与剩余瓶颈。

---

## 二、最新 `origin/main` (`6f3557c97`) 八大核心子系统深度剖析

### 2.1 子系统一：QSA 原生 Host-KV 架构、跨层异步预取与单 CTA/行 `rows_fp8` 内核
- **核心文件**：
  - `csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu`
  - `vllm/models/qwen4_exp/nvidia/ops/host_kv.py`
  - `vllm/models/qwen4_exp/nvidia/ops/device_kv_attention.py`
  - `vllm/models/qwen4_exp/nvidia/qsa.py`
  - `vllm/v1/worker/utils.py` (`allocate_host_kv_cache_pool`)
- **关键机制剖析**：
  1. **单池精确映射主机内存 (`allocate_host_kv_cache_pool`，Commit `a9d16c6a` / `19586512`)**：
     每个 Worker 进程将所有 Host-KV 层合并为单个精确大小的 `cudaHostAllocMapped` 锁页内存池（4 卡 256K FP16 约 15.22 GiB），并在 `model_runner.py` / `gpu_model_runner.py` 中标记 `host_backed_layers`，使 KV Block Zeroer 跳过主机 KV 清零（**每步省去约 160 MB 主机内存写零，节省 ~17 ms**）。
  2. **Decode 设备端热缓存 (`sm70_qsa_host_native=True`, `qsa_host_kv_hot_tokens=8192`，Commit `4fd3d517`)**：
     在 GPU 上维护 8192-token 的 CLOCK 保护热页缓存（`_protect` + `_gather`），Decode 时通过 `torch.ops.vllm_sm70_qsa_device.run_hot` 直接在设备端读取命中的热页，未命中项通过 UVA 直接从映射主机内存读取。
  3. **Prefill 双缓冲跨层异步 DMA 暂存 (`stage_history` + `_prefetch_next_prefix`，Commit `e756ed97`)**：
     当启用 `qsa_host_kv_prefill_stage_tokens` 时，不再让每个 Query 行都经由 PCIe 重复拉取历史 KV，而是：
     - 利用 2 个设备端 Stage Buffer（`slot = order % 2` 乒乓缓冲），在第 $L$ 层 QSA 计算完成后，立即在独立 `_PREFETCH["stream"]` 上异步将第 $L+1$ 层的历史前缀页（`prefix_pages`）从主机拷贝到另一个 Stage Buffer；
     - 当前 Chunk 新生成的 KV 页无需先写回主机再从主机读回，而是由 `state.write_staged` 直接写入设备端 Stage Buffer；
     - 随后调用 **`torch.ops.vllm_sm70_qsa_device.run_rows`**（`rows_fp8<2, true>`）：每个 Query 行由 **1 个 CTA** 独立负责全部 Top-K 选中历史页，三级软件流水（`stage_index` / `stage_table` / `stage_scales` 提前 2 个 Tile 定位，K/V 提前 1 个 Tile 预取），使用 SM70 Tensor Core `mma.sync.aligned.m8n8k4` 计算 QK 与 PV（`MMA_PV=true`），内部完成 Online Softmax 并直接写出 Gated FP16 结果，**完全消除了 Partial/State Workspace 读写与 Merge Kernel**，将每层每 8K Chunk 的 QSA Prefill 耗时从 **22 ms 压降至 12.5 ms**。
  4. **零同步 `max_seq_len` 边界与 `qsa_score_tile_mb` 限流 (`common/qsa_cache.py` + `ops/qsa.py`，Commit `caa27279` / `3e3a6513`)**：
     通过 `common_attn_metadata.seq_lens_cpu_upper_bound` 在主机侧直接计算 `live_blocks = max_seq_len // compress_ratio`，消除了 `all_visible.max().item()` 的 GPU$\rightarrow$CPU 标量同步；同时用 `2 * qsa_score_tile_mb * 1024**2` 限制 cuBLAS 与 Paged 两条路径的单次 Logits 切片显存峰值。
  5. **文本模式 MRoPE 缓存 4 倍压缩 (`_qsa_rope_cache_length`，Commit `4fd3d517`)**：
     由于 `MRotaryEmbedding` 会为视频位置将缓存放大 4 倍，在纯文本模式下将 `max_position` 设为 `ceil(min(max_position_embeddings, max_model_len) / 4)`，使实际生成的 RoPE 表刚好覆盖 `max_model_len`，节省 75% 的 RoPE 缓存显存。

---

### 2.2 子系统二：路由专家 GPU 缓存与冷专家 DMA 暂存转码 (`TurboMindPrefill`)
- **核心文件**：
  - `csrc/gguf_expert_cache.cpp`
  - `vllm/model_executor/layers/quantization/gguf_expert_cache.py`
  - `csrc/sm70_turbomind/ops/gguf_prefill_transcode_sm70.cu`
- **关键机制剖析**：
  1. **80% GPU + 20% Pinned Host 路由专家分层存储 (`expert_cache_fraction=0.8`，Commit `4db646d4`)**：
     在 `4 × V100 16GB` 等紧凑显存形态下，每层将 80% 的高频专家驻留在 GPU Slot，20% 的冷专家放在主机锁页内存。Decode 发生缓存缺失时，由原生 C++ 后台线程（`csrc/gguf_expert_cache.cpp`）拷入预留的空闲 GPU Slot（`expert_cache_spare_slots=8`），并在主机侧保留 `expert_cache_host_slack=0.1` 的冗余副本，使驱逐（Eviction）无需 D2H 写回。
  2. **解决 Zero-Copy PCIe 拖慢主流的隐藏陷阱（`TurboMindPrefill` DMA 暂存）**：
     `benchmarks/flashnext_prefill/HANDOFF.md` 揭示了一个极其关键的实测结论：此前在 Prefill 阶段由 8 个 GPU CTA 通过 Zero-Copy 直接跨 PCIe 读取主机冷专家，会导致并发执行的主流（Main Stream）算子延迟翻倍（例如 HC Down 从 `0.59 ms` 恶化至 `1.2 ms`，32K 请求平白多出 `1.3 s`）。
     `origin/main` 改为：由 Copy Engine 先通过 `cudaMemcpyAsync` 将主机冷专家 Slot 区间 DMA 到每卡 68 MB 的设备暂存区 `self.stage`（每步 PCIe 传输约 `0.93 s`，在独立流上提前一层与计算完全重叠掩盖），再用 64 个 CTA 从设备显存高速转码为 TurboMind Canonical Bank（`gguf_prefill_transcode_sm70.cu`），使 32K TTFT 从 `8.08 s` 降至 `7.80 s`。

---

### 2.3 子系统三：HyperConnection (mHC) 分片权重存储、Prefill 本地混合与 HCX 本地调度
- **核心文件**：
  - `vllm/models/qwen4_exp/nvidia/hyperconnection.py`
  - `vllm/models/qwen4_exp/nvidia/sm70_fp16_hc.py`
  - `vllm/models/qwen4_exp/nvidia/ops/hc.py`
  - `csrc/sm70_turbomind/ops/hcx_sm70.cu`
  - `docs/design/flashnext_hcx_local_schedule.md`
- **关键机制剖析**：
  1. **无损分片权重存储 (`hc_weight_storage="sharded"`，Commit `2b0b8b1a` / `22d5fb99`)**：
     在模型加载阶段将 `GatedResidual` 的 FP16 权重暂存在 CPU（`_hc_cpu_staging = True`），调用 `prepare_sharded_hc_storage` 仅将当前 TP4 Rank 对应的无损打包分片（`hc_shard_down`、`hc_shard_up`）移入 GPU，随后立即释放完整的复制权重矩阵（`[336, 10240]` 与 `[10240, 320]`）及借用的 `_sm70_dense_state`。
  2. **Prefill 本地 Gate-Mix 降低 4 倍通信量 (`_sharded_hc_project`，Commit `011a8ce1`)**：
     在大 Batch Prefill 下，从分片恢复本卡的 `local_down` (`[88, 10240]`) 和 `local_up` (`[2560, 320]`)：
     - 先算 `local = torch.mm(x, local_down.T)` 并 `all_gather` 得到 `[4 * rows, 88]`；
     - 用融合算子 `hc_packet_silu` 按逻辑 Rank 顺序提取 `lora` (`[rows, 320]`) 与 `injection` (`[rows, 4]`)；
     - 算 `local_gate = torch.mm(lora, local_up.T)` (`[rows, 2560]`，包含本卡负责的 640 列在全部 4 条 HC 流上的 Gate)；
     - **关键突破**：不再将 `[rows, 2560]` 的 `local_gate` 做全量 AllGather（会产生 `[rows, 10240]` 通信），而是先在本地调用 `hc_gate_mix_local` 完成 4 条流的加权混合得到 `[rows, 640]`，再对 **窄 4 倍** 的 `[rows, 640]` 做 AllGather 并通过 `hc_assemble_ranks` 还原为 `[rows, 2560]`！
  3. **Full-Mesh HCX/HCXO 本地调度 (`sm70_hcx_local_schedule=True`，Commit `b0d7af22` / `71fce5ca`)**：
     在 `csrc/sm70_turbomind/ops/hcx_sm70.cu` 中延迟 Up 权重预取至 Combine 之后，合并独立 Peer Loads，交错执行 4 条流的 Norm 归约，并用持久化 Acquire/Release 原子屏障消除完成计数器重置阶段（M=5 未融合边界从 `23.766 µs` 降至 `18.993 µs`，融合输出投影链从 `28.505 µs` 降至 `26.395 µs`）。

---

### 2.4 子系统四：C++ 原生 PLE Fast Lane、Draft 预取、IQ4_NL 打包传输与磁盘行缓存
- **核心文件**：
  - `csrc/ple_fast_lane.cpp` & `vllm/v1/ple_offload/fast_lane.py`
  - `vllm/v1/ple_offload/draft_prefetch.py`
  - `vllm/model_executor/kernels/ple/gguf_packed_result.py`
  - `csrc/ple_disk_rows.cpp` & `vllm/models/qwen4_exp/nvidia/ple_layer.py`
- **关键机制剖析**：
  1. **原生 C++ 共享内存快速通道 (`csrc/ple_fast_lane.cpp`，Commit `619788df` / `c1f67f15`)**：
     构建了 GPU Worker 与 CPU PLE Offload 进程之间的共享内存环形通道 `_ple_draft_ring`。当 Target 完成校验产生 Bonus Token（`model_runner.py:publish_round`）或 Draft 每步生成提案 Token（`draft_step_listener`）时，直接通过轻量级 Triton 内核把 Token ID 写入映射环形缓冲区，由 C++ 线程直接唤醒查表，彻底绕开 Python 进程间队列开销。
  2. **Draft Token PLE 提前预取 (`draft_prefetch.py` + `draft_prefetch_row_ids`，Commit `a1768746`)**：
     在 Draft 阶段提前用相同的 3-gram 哈希计算待校验 Token 的 PLE 行号并预热 CPU 缓存。
  3. **IQ4_NL 原始打包字节跨进程传输 (`gguf_packed_result.py`，Commit `ec0e22c1`)**：
     CPU 查表后不再做 CPU 反量化（将每 Token 传输量从 `5,120 B` FP16 压缩至 `1,440 B` 原始 IQ4_NL 字节，M=5 CPU 生产耗时从 `0.832 ms` 降至 `0.242 ms`，M=20 从 `2.557 ms` 降至 `0.366 ms`），GPU 端等待信号量后由 Triton 内核在设备端直接解码为 FP16。
  4. **有界 512 MiB CPU 直接映射行缓存 (`ple_disk_only=True`, `ple_row_cache_mib=512`) 与 `_CPU_SMALL_NGRAM_TOKENS = 64`**：
     将 CPU 标量快速 N-gram 哈希的阈值从 16 提升至 64 Token（覆盖 4 并发 MTP4 Verify 的 20 Token 批次）。

---

### 2.5 子系统五：MTP Draft 13-bit 无损/Q8_0 专家、共享词表权重、热词表与稀疏目标采样
- **核心文件**：
  - `csrc/sm70_turbomind/ops/mtp_moe_fp16_sm70.cu` (`mtp_moe_u13_out`)
  - `vllm/models/qwen4_exp/nvidia/mtp_lossless_experts.py` & `mtp_q8_experts.py`
  - `vllm/models/qwen4_exp/nvidia/mtp.py` & `sm70_mtp_head.py`
  - `vllm/v1/worker/gpu/spec_decode/sm70_sparse_target.py` & `rejection_sampler_utils.py`
- **关键机制剖析**：
  1. **BF16 源 FP16 专家的 13-bit 无损位平面存储 (`sm70_mtp_lossless_storage=True`，Commit `700c891f`)**：
     由于模型权重的 FP16 来自 BF16 转换（正常数低 3 位尾数恒为 0），`mtp_moe_u13_out` 将每 32 个 FP16 权重（64 字节）无损压缩为 52 字节（32 个低字节 + 16 个高半字节 + 4 个符号字节，保留全部次正规数精度与符号零），节省 **18.75% 显存** 且在 `mtp_moe_fp16_sm70.cu` 寄存器加载时实时无损还原，计算结果 bit-exact 一致；可选 `sm70_mtp_q8_0_experts=True` 则以 GGUF `Q8_0` 存储 Draft 专家（节省约 50% 显存）。
  2. **Target 与 Draft 共享 `embed_tokens` 与 `lm_head` (`share_target_io_weights=True`，Commit `700c891f`)**：
     在 `pp_world_size == 1` 时，`Qwen4ExpMTP` 将自身的 `embed_tokens` 和 `lm_head` 设为 `PPMissingLayer()` 并在 `skip_checkpoint_weight` 中跳过加载，直接复用 Target 模型的 Embedding 与 LM Head，每卡节省约 1.2 GiB 显存并显著加快启动加载。
  3. **稀疏目标投机拒绝采样 (`sm70_sparse_target_sampling=True`，Commit `011a8ce1`)**：
     针对 `top_k = 20` 采样 + Greedy Draft 场景（如 `HANDOFF.md` 中的 `T=0.6, top_p=0.95, top_k=20`），不再跨 TP4 AllGather 全词表 `[M, 248320]` Logits 并运行稠密 Rejection Sampler，而是由每个 TP Rank 仅提取本卡局部的 Top-K 候选（`get_topk_tokens_and_logits`），在紧凑候选集上执行 `greedy_draft_sparse_topk_rejection_sample`；仅当 Top-K/Top-P 截断边界存在并列（Tie）或舍入歧义时才回退全词表采样，保证数学与随机数流完全等价。

---

### 2.6 子系统六：调度器均分 Prefill Chunk (`VLLM_BALANCED_PREFILL_CHUNKS=1`) 与状态/历史独立块池
- **核心文件**：
  - `vllm/v1/core/sched/scheduler.py` (`_balanced_prefill_chunk`, `mamba_tail_checkpoint`)
  - `vllm/v1/core/block_pool.py` & `kv_cache_utils.py`
- **关键机制剖析**：
  1. **均分 Prefill Chunk (`_balanced_prefill_chunk`，Commit `caa27279`)**：
     在开启主机冷专家流式加载时，每个 Prefill 步都有固定的 ~0.93s PCIe 传输底噪，若 34.7K Prompt 按 `8160×4 + 2083` 切分，最后的 2083 短尾块仍需支付完整一步的 PCIe 代价。开启 `VLLM_BALANCED_PREFILL_CHUNKS=1` 后，调度器按 Mamba/QSA 块大小（816 Token）对齐向上均分剩余 Token，将 34.7K 切分为 `7344×4 + 5347`，使 32K TTFT 从 `7.80 s` 降至 `7.62 s`。
  2. **跳过不必要的尾块 Mamba Checkpoint (`mamba_tail_checkpoint=False`)**：
     当剩余 Prompt 可在当前步一次性完成且无需尾块回放检查点时，不在最后一个 Replay 边界处强行拆步。
  3. **循环状态与注意力历史分区块池 (`qsa_host_kv_state_blocks`，Commit `8faef7ea` / `41fa4bee`)**：
     将低位全局 Block ID（如 `0..28` 共 29 个块）专用于 GPU 上的 GDN/PLE 循环状态，其余高位 Block ID 用于主机上的 256K QSA 历史，并在 `simple_kv_offload` 中跳过分区块池的驱逐游标。

---

### 2.7 子系统七：GDN Prefill 门控归一化、无上下文设备探测与 Row-Major 路由权重
- **核心文件**：
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
  - `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py`
  - `csrc/sm70_turbomind/ops/qwen38_gdn_input_batch_sm70.cu`
- **关键机制剖析**：
  1. **GDN Prefill 直接调用单趟 Triton `RMSNormGated.forward_cuda`（Commit `011a8ce1`）**：
     `origin/main` 在 `qwen_gdn_linear_attn.py:2777` 增加了 `or num_tokens > 64`，避免在自定义算子内部调 `self.norm(core_attn_out, z)` 时被拆解为 9 个原生 PyTorch 小算子。
  2. **CPU Offload 进程零 CUDA 上下文污染（Commit `4071c4d0` / `f1c01fe7`）**：
     将 `_sm70_current_device_is_volta()`、`create_chunk_kernels()` 等处的 `torch.cuda.get_device_capability()` 替换为无上下文查询 `current_platform.is_device_capability((7, 0))`，并将 `RMSNormGated` 设备绑定改为 `device=self.dt_bias.device`，防止 CPU PLE Worker 在 GPU0 上意外创建 300~500 MiB 的 CUDA Context。
  3. **Row-Major 路由权重零拷贝复用 (`sm70_router_weight_storage="row_major"`，Commit `2b0b8b1a`)**：
     在 `qwen38_gdn_input_batch_sm70.cu` 中新增 `router_split_quad_kernel<40, true>`，直接读取原始 `[512, 2560]` Row-Major FP16 路由权重，无需在每层额外保留一份 `[64, 40, 2, 4, 8, 8]` 的重排副本。

---

### 2.8 子系统八：并发 Worker 启动、共享 CUDA Graph 捕获流与有界 TurboMind Scratch
- **核心文件**：
  - `vllm/v1/executor/multiproc_executor.py`
  - `vllm/v1/worker/gpu/cudagraph_utils.py` & `model_runner.py`
  - `csrc/sm70_turbomind/ops/gemm_runtime.cpp`
- **关键机制剖析**：
  1. **Target 与 Draft 共享捕获流 (`GraphCaptureContext`，Commit `629063e7`)** 与 **Draft 捕获前释放 Eager 缓存 (`5ee0e91f`)**：
     消除多流捕获引起的流间同步与显存池碎片。
  2. **有界 TurboMind Split-K 暂存区 (`sm70_turbomind_workspace_bytes`，Commit `622c1b99` / `a12068b3`)**：
     允许将每流默认 32 MiB 的 Split-K Partials 压缩至 8 MiB，并约束强制 Split-K 候选不超过可用暂存容量。

---

## 三、此前优化项 (`P1`–`P13` & `D1`–`D29`) 与最新 `origin/main` (`6f3557c97`) 的交叉比对与融合

在将 `origin/main` (`6f3557c97`) 合入 `arena/5368f936-1cat-vllm` 时，我们对重叠区域进行了逐项比对与融合：

| 领域 | `origin/main` (`6f3557c97`) 的实现 | 本分支 (`P1`–`P13` / `D1`–`D29`) 的实现 | 合并后最佳融合策略 |
|---|---|---|---|
| **QSA Prefill 零同步 `max_seq_len`** | 新增 `_host_max_seq_len` 读取 V2 Runner 的 `seq_lens_cpu_upper_bound`，并增加 `qsa_score_tile_mb` 显存上限 | 支持 V1 Runner `max_seq_len < max_model_len` 上界，并在无 `max_seq_len` 时跨 12 层缓存 `(all_visible, max_visible)` | **双路径融合**：优先用 `_host_max_seq_len`，V1 Runner 下回退 `common_attn_metadata.max_seq_len`，无上界时启用 `visible_blocks_cache` 跨层复用，同时保留 `qsa_score_tile_mb` 限流且兼容 `test_qsa_indexer_request_split.py` |
| **GDN `RMSNormGated` 避免 9-kernel 拆解** | `qwen_gdn_linear_attn.py:2777` 对 `num_tokens > 64` 调 `self.norm.forward_cuda` | `P3` 对 `rows > 192` 避免 9-kernel 拆解 | **统一收敛至 `qwen_gdn_linear_attn.py`**：将阈值从 `num_tokens > 64` 扩展为 `core_attn_out.shape[0] > 192`（即 `num_tokens > 16`，覆盖 4 并发 MTP4 Verify 的 `M=20` 及全部 Prefill），直接走单趟 Triton `self.norm.forward_cuda`，保持 `layernorm.py` 零额外编译开销 |
| **QSA `_run_qsa` Host-KV 写入与 DCP Query AllGather** | 在 `self.host_kv.write` 后新增 `self._host_kv_new = (key, value, main_metadata.slot_mapping)` 供 `write_staged` 使用 | 将 `host_kv.write` / `qsa_prep_sm70_out` / `do_kv_cache_update` 提前到 `self.indexer` 之前，使 DCP2 Query AllGather 与 Indexer 重叠 | **完整保留双方特性**：在提前的 `if self.host_kv_enabled:` 分支中保留 `self._host_kv_new` 赋值，消除重复代码块 |
| **NVFP4 `TP4+DCP2` / `DP2×TP4` 全套优化 (`D1`–`D29`)** | `origin/main` 聚焦于 `4×V100 16GB GGUF + Host-KV` 路径，未改动 NVFP4 `TP4+DCP2` 路径 | 实现了 `D1`–`D29`（DCP2 AllGather 重叠、Split-K 融合、MTP Draft KV Cache Dtype/DCP Buffer 修复、GDN Split 16-block Grid 修复等） | **100% 无冲突保留**，使本分支同时具备 **NVFP4 `TP4+DCP2`** 与 **GGUF `TP4 + Host-KV`** 两条主力部署路径的全部能力 |

---

## 四、针对最新 `origin/main` (`HANDOFF.md` 第 6 节) 剩余瓶颈的新增优化 (`P14`–`P16`)

根据 `origin/main` 中 `benchmarks/flashnext_prefill/HANDOFF.md` 第 6 节列出的 **最高性价比待完成优化项**，以及我们对 `6f3557c97` 新增代码的审查，我们在合并后的分支上进一步实现了 **`P14`、`P15`、`P16`** 三项优化：

### 4.1 `P14`：QSA Staged Prefill 新增 `rows_fp16` 单 CTA/行内核（消除 FP16 Draft/Target 的 `run_tiled` Partial + Merge）

- **问题定位（对应 `HANDOFF.md` 6.1 第 1 条）**：
  在 `origin/main` 的 `vllm/models/qwen4_exp/nvidia/qsa.py:1022` 中：
  - 当 Staged History 为 `torch.uint8`（FP8 E4M3）时，调用单趟 `torch.ops.vllm_sm70_qsa_device.run_rows`（每个 Query 行 1 个 CTA，无 Partial Workspace，无 Merge Pass）；
  - 但当 Staged History 为 `torch.float16`（包括 **32K Prefill 中的 MTP4 Draft 模型**，以及 `serve-16gb-256k.sh` 中 `qsa_host_kv_dtype="float16"` 的 Target 模型）时，由于 `qsa_device_history_sm70.cu` 中 `TORCH_CHECK(fp8 && !hot)` 仅实现了 `rows_fp8`，只能回退到 `run_tiled`（按 64 行切块，每块启动 `partial<false, false>` + `merge` 两个内核并读写 `19.5 MiB` 的 FP32 Partial 缓冲区）。
- **优化方案**：
  1. 在 `csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu` 中新增 `rows_fp16<MIN_BLOCKS, MMA_PV>`：
     - 共享内存布局：`query[8][264]` (`4,224 B`) + `tile[64][264]` (`33,792 B`) + `scores[4][8][68]` (`8,704 B`) + `slots[3][64]` (`1,536 B`) + `alpha/denom` (`64 B`) = **`48,320 B` (`< 48 KiB`)**，在 V100 (96 KiB SMEM/SM) 上完美达成 **2 CTAs/SM (`MIN_BLOCKS = 2`)** 且无需 Scale 查表与 FP8 解码指令；
     - 采用与 `rows_fp8` 相同的三级软件流水，QK 使用 Tensor Core `mma.sync.aligned.m8n8k4`，PV 保持与 `partial<false, false>` 完全一致的顺序 FP32 FMA 累加链（或可选 `MMA_PV`），直接写出 Gated FP16 结果；
     - 注册 `torch.ops.vllm_sm70_qsa_device.run_rows_fp16`，并在 `qsa.py:host_kv_forward` 中通过 `elif hasattr(torch.ops.vllm_sm70_qsa_device, "run_rows_fp16"):` 自动路由（若运行环境尚未重编 `_sm70_qsa_device_C.so`，则安全回退 `run_tiled`，保证 100% 二进制向后兼容）。

#### 修改代码：`csrc/sm70_turbomind/ops/qsa_device_history_sm70.cu` 与 `vllm/models/qwen4_exp/nvidia/qsa.py`

```diff
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -1274,6 +1274,19 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
                         gate_rows,
                         12,
                     )
+                elif hasattr(torch.ops.vllm_sm70_qsa_device, "run_rows_fp16"):
+                    torch.ops.vllm_sm70_qsa_device.run_rows_fp16(
+                        query,
+                        history,
+                        scales,
+                        indices,
+                        staged_table,
+                        requests,
+                        positions,
+                        lengths,
+                        output,
+                        gate_rows,
+                        2,
+                    )
                 else:
                     torch.ops.vllm_sm70_qsa_device.run_tiled(
```

---

### 4.2 `P15`：Sharded HC Prefill 多行 CTA 分块 (`BLOCK_M=8`) 与单测兼容修复

- **问题定位（对应 `HANDOFF.md` 6.2）**：
  1. 在 `origin/main` 的 `vllm/models/qwen4_exp/nvidia/ops/hc.py` 中，`hc_packet_silu` 和 `hc_assemble_ranks` 的启动网格均为 `(rows, 4)`：当 `rows = 8192` 时，单次调用启动 **32,768 个 CTA**，而 `_hc_packet_silu_kernel` 每个 CTA 仅处理 80 个 FP16 元素（160 字节），绝大部分时间消耗在 CTA 调度开销上；
  2. 此外，`origin/main` 的 Commit `011a8ce1` 将 `sm70_fp16_hc.py:_sharded_hc_project` 改为直接访问 `group.rank_in_group`，但 `tests/kernels/test_sm70_hc_sharded_fallback.py` 中构造的 Mock `group = SimpleNamespace(device_communicator=..., all_gather=gather)` 没有 `rank_in_group` 属性（且其 Mock `gather` 对每个 Rank 乘以 `rank + 1`），导致该单测在 `origin/main` 上会抛出 `AttributeError`。
- **优化方案**：
  1. 在 `ops/hc.py` 的 `_hc_packet_silu_kernel` 和 `_hc_assemble_kernel` 中引入 `BLOCK_M: tl.constexpr = 1`（当 `rows >= 64` 时取 `BLOCK_M = 8`），使每个 CTA 沿行维度連続处理 8 行，将 8K Prefill 下的 CTA 启动数量压缩 **8 倍**（从 `32,768` 降至 `4,096` 个 CTA/次，单次 32K Prefill 减少约 **550 万个 CTA 启动**），且每行算术顺序完全不变；
  2. 在 `sm70_fp16_hc.py:_sharded_hc_project` 中检查 `rank_in_group = getattr(group, "rank_in_group", None)`：真实 TP4 运行时走 `hc_gate_mix_local` + 4 倍窄 AllGather + `hc_assemble_ranks`；单进程 Mock 测试（`rank_in_group is None`）自动走 `hc_gate_mix_gathered`，兼顾真实线上极速与单测 100% 通过。

#### 修改代码：`vllm/models/qwen4_exp/nvidia/ops/hc.py` 与 `vllm/models/qwen4_exp/nvidia/sm70_fp16_hc.py`

```diff
--- a/vllm/models/qwen4_exp/nvidia/ops/hc.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/hc.py
@@ -262,16 +262,21 @@ def _hc_assemble_kernel(
     R3: tl.constexpr,
     COLS: tl.constexpr,
     BLOCK_SIZE: tl.constexpr,
+    BLOCK_M: tl.constexpr = 1,
 ) -> None:
     """dst[row, j * COLS + c] = src[R<j> * rows + row, c] (rank-major gather
     to logical column order)."""
-    row = tl.program_id(0)
+    row_base = tl.program_id(0) * BLOCK_M
     j = tl.program_id(1)
     rank = tl.where(j == 0, R0, tl.where(j == 1, R1, tl.where(j == 2, R2, R3)))
-    for start in tl.static_range(0, COLS, BLOCK_SIZE):
-        offs = start + tl.arange(0, BLOCK_SIZE)
-        mask = offs < COLS
-        value = tl.load(src_ptr + (rank.to(tl.int64) * rows + row) * COLS + offs, mask)
-        tl.store(dst_ptr + row.to(tl.int64) * 4 * COLS + j * COLS + offs, value, mask)
+    for m_off in tl.static_range(0, BLOCK_M):
+        row = row_base + m_off
+        row_valid = row < rows
+        for start in tl.static_range(0, COLS, BLOCK_SIZE):
+            offs = start + tl.arange(0, BLOCK_SIZE)
+            mask = row_valid & (offs < COLS)
+            value = tl.load(
+                src_ptr + (rank.to(tl.int64) * rows + row) * COLS + offs, mask
+            )
+            tl.store(
+                dst_ptr + row.to(tl.int64) * 4 * COLS + j * COLS + offs, value, mask
+            )
@@ -307,13 +312,15 @@ def hc_assemble_ranks(gathered: torch.Tensor, order: tuple[int, ...]) -> torch.T
     ranks = len(order)
     assert ranks == 4 and gathered.is_contiguous()
     rows, cols = gathered.shape[0] // ranks, gathered.shape[1]
     out = gathered.new_empty(rows, ranks * cols)
-    _hc_assemble_kernel[(rows, ranks)](
+    block_m = 8 if rows >= 64 else 1
+    _hc_assemble_kernel[(triton.cdiv(rows, block_m), ranks)](
         gathered,
         out,
         rows,
         *order,
         cols,
         256,
+        BLOCK_M=block_m,
     )
     return out
```

---

### 4.3 `P16`：Prefill 阶段 `HostQSAKV.write` 移至独立 Side Stream 与 `self.indexer` 并行重叠

- **问题定位（对应 `HANDOFF.md` 6.1 第 3 条）**：
  `HANDOFF.md` 指出：在 32K Prefill 中，`HostQSAKV.write` 在主流（Main Stream）上将当前 Chunk 的 K/V 编码写入映射主机内存，累计耗时约 `0.07 s`；而紧随其后的 `self.indexer(hidden_states, positions, ...)` 仅访问 Indexer 自身的 `compressed_key_cache`，完全不读写 `self.host_kv`。
- **优化方案**：
  在 `vllm/models/qwen4_exp/nvidia/qsa.py:_run_qsa` 中，当 `self.host_kv_enabled` 且处于大 Batch Prefill（`num_tokens > self.host_kv.resolve_rows` 且非 CUDA Graph 捕获期）时：
  - 在专用 `_PREFETCH["write_stream"]` 上异步执行 `self.host_kv.write(key, value, main_metadata.slot_mapping)`；
  - 主流同时并发执行 `selected = self.indexer(...)`（包含 `index_qk_proj`、RoPE、Compressor 及 cuBLAS MQA Top-512 评分选择）；
  - 在进入 `impl.forward_qsa` 之前执行 `torch.cuda.current_stream(key.device).wait_stream(async_host_write_stream)` 汇合，从而将 `HostQSAKV.write` 的 PCIe 写入延迟完全掩盖在 `self.indexer` 的 GPU 计算之下。

#### 修改代码：`vllm/models/qwen4_exp/nvidia/qsa.py`

```diff
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -1613,8 +1613,22 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
         impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
+        async_host_write_stream = None
         if self.host_kv_enabled:
-            self.host_kv.write(key, value, main_metadata.slot_mapping)
+            if (
+                key.is_cuda
+                and num_tokens > getattr(self.host_kv, "resolve_rows", 32)
+                and not torch.cuda.is_current_stream_capturing()
+            ):
+                if _PREFETCH["write_stream"] is None:
+                    _PREFETCH["write_stream"] = torch.cuda.Stream(device=key.device)
+                async_host_write_stream = _PREFETCH["write_stream"]
+                async_host_write_stream.wait_stream(torch.cuda.current_stream(key.device))
+                with torch.cuda.stream(async_host_write_stream):
+                    self.host_kv.write(key, value, main_metadata.slot_mapping)
+            else:
+                self.host_kv.write(key, value, main_metadata.slot_mapping)
             # A staged prefill writes these rows into its stage buffer too.
             self._host_kv_new = (key, value, main_metadata.slot_mapping)
@@ -1703,6 +1717,8 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
         selected = _sm70_dump_qwen_layer_tensor(
             "qsa_selected_indices",
             self.indexer.layer_id,
             "qsa",
             selected,
         )
+        if async_host_write_stream is not None:
+            torch.cuda.current_stream(key.device).wait_stream(async_host_write_stream)
         impl.forward_qsa(
```
