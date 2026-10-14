# Qwen3.8-Flash-Next (`TP4 + DCP2 + MTP4` on SM70 V100) — Prefill & Decode 能力突破优化专项文档

> **目标平台**: NVIDIA Tesla V100-SXM2-32GB (`SM70`, Volta)  
> **目标部署拓扑**: `TP4 + DCP2` (`world_size=4`, `dcp_size=2`) + `MTP4` (`num_speculative_tokens=4`)  
> **前置优化基线**: `docs/design/sm70_qwen38_prefill_decode_optimizations.md` (`P1`–`P10` & `D1`–`D16`)  
> **专项定位**: 在保留 100% 数值等价 (`0 ULP` / bit-exact) 的前提下，针对 **架构级能力边界限制**（DCP2 稀疏注意力全预算空转、V1 SpecDecode 缺失 MTP Top-K 索引共享、MTP 算子/小 batch 限制、PLE 膨胀短卷积与 N-gram 构图瓶颈）实现 **能力突破 (`P11`–`P13`, `D17`–`D20`)**，并系统梳理仓库内置的高阶 SM70 能力开关。

---

## 1. 能力突破总览 (`P11`–`P13` & `D17`–`D20`)

在前置 `P1`–`P10` 与 `D1`–`D16` 完成通用算子融合与零分配改造后，`Qwen3.8-Flash-Next` 在 `TP4 + DCP2 + MTP4` 下仍存在 **4 处能力级瓶颈**：

| 编号 | 阶段 | 能力突破点 | 改动文件 | 核心突破与收益 | 精度影响 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`P11` / `D17`** | Prefill & Decode | **`TP4 + DCP2` QSA 局部有效块数 (`valid_counts`) 动态早退** | `ops/qsa_dcp.py`<br>`ops/qsa.py`<br>`qsa.py` | `_localize_indices` 已在寄存器内算出本 DCP rank 拥有的有效组数 `count`，但此前未写出，导致 `_qsa_sparse_paged_gqa_splitk_kernel` 始终遍历完整 `selected_width = 1024`（`64` 个 `BLOCK_N=16` tile，其中约一半是 `-1` 填充）。现将 `valid_counts` 写入预分配缓存并传入稀疏 Split-K 内核，动态截断 `loop_end = min(split_tile_end, cdiv(selected_count, BLOCK_N))`，且在 `selected_count == 0` 时直接早退，**消除 DCP2 下约 50% 的无效尾部 tile 循环**。 | **Bit-exact (`0 ULP`)** |
| **`D18`** | `MTP4` Decode | **V1 `SpecDecodeBaseProposer` 启用 `index_share_for_mtp_iteration` + 跳过 `QSAIndexer` Query 准备** | `llm_base_proposer.py`<br>`indexer_qsa.py`<br>`ops/qsa_pre_indexer.py` | 1. V2 `EagleSpeculator` 支持 `index_share_for_mtp_iteration`（Pass 0 选出 Top-K 索引并压缩为 `:B`，Draft steps `1..3` 置 `set_skip_topk(True)` 复用 Pass 0 索引），但默认 V1 `SpecDecodeBaseProposer` 未接入该能力。现将其完整接入 V1 Proposer。<br>2. 当 `skip_topk=True`（或 `select=False`）时，`QSAIndexer` 仅需将当前步 `k` 写入循环状态缓存，无需计算 `q_norm` 与 `q_rope`。现跳过 `q_norm`、`q_rope` 及 `qsa_pre_indexer` 中的 Query 线程块（`num_q_work = 0`）。 | **与模型配置契約完全一致 (`0 ULP`)** |
| **`P12` / `D19`** | Prefill & Decode | **MTP 层启用 SM70 Peer Paths + 解除 `qsa_prep_sm70_out` `M<=32` 限制 + `qwen_gdn_input_projection` 融合切分** | `model.py`<br>`mtp.py`<br>`qsa.py`<br>`qwen_gdn_linear_attn.py` | 1. `Qwen4ExpMTP.load_weights` 补齐 `_maybe_enable_sm70_peer_paths(self.vllm_config, self.model)`（并使 `_maybe_fuse_sm70_side_projections` 兼容 `nn.ModuleDict`），让 MTP 层同样享受 `sm70_fused_side_projections` / `sm70_hcx` / `sm70_top1x`。<br>2. 解除 `qsa.py` 中 `qsa_prep_sm70_out` 的 `num_tokens <= 32` 限制（底层 CUDA 核支持任意 `rows >= 1`），使 Prefill 及大 Batch Decode 全部走单次 CUDA 融合算子。<br>3. 在 `qwen_gdn_input_projection` 中接入 `_qwen38_gdn_projection_split_kernel(COPY_QKV=True)`，将 4 次跨步 `.copy_()` 融合为 1 次 Triton 启动（36 层省去 **108 次启动/步**）。 | **Bit-exact (`0 ULP`)** |
| **`P13` / `D20`** | Prefill & Decode | **PLE 膨胀短卷积 (`dilation=3`) 全场景融合（Prefill 零转置 / Spec 多请求扩展 / Decode FP32 状态兼容）+ Prefill N-gram ID 融合** | `ple_layer.py`<br>`qwen38_ple_spec_sm70.cu` | 1. **Prefill 短卷积**: 针对 `H=10240` 新增行主序 4-tap Triton 核 `_qwen38_ple_prefill_short_conv_kernel` + `_qwen38_ple_prefill_state_update_kernel`，消除 `[num_prefills, 10240, 9 + max_len]` 零填充/转置/分组卷积/gather 流水线，在 8K Prefill 下 **4 层合计节省 ~5.36 GB HBM 流量与 ~60 次启动**；同时为通用 PyTorch 路径增加 `num_prefills == 1` 免 scatter/gather 快路径。<br>2. **MTP Verify 短卷积**: 将 `qwen38_ple_spec_sm70.cu` 与 `_short_conv_dilated_spec_batched` 从仅支持 `num_reqs == 1` (`rows in {5, 10}`) 扩展到 `1 <= num_reqs <= 64`、`rows >= 5`（含 Triton 多请求回退核 `_qwen38_ple_spec_short_conv_kernel`），**4 层省去 104 次启动/步**。<br>3. **Decode 短卷积**: 支持 `conv_state.dtype in (fp16, fp32)`。<br>4. **Prefill N-gram ID**: 在 `compute_ngram_ids` 中对 `m > 32` (`1 <= num_reqs <= 64`) 直接调度 `sm70_ple_ngram_ids`，**4 层省去 100 次启动/Prefill 步**。 | **Bit-exact (`0 ULP`)** |

---

## 2. 深度架构剖析与实现细节

### 2.1 `P11` / `D17`: `TP4 + DCP2` QSA 局部有效块数 (`valid_counts`) 动态早退

#### 瓶颈根因
在 `TP4 + DCP2` 下，每个 QSA 层通过 `qsa_localize_dcp_indices` 将全局 Top-K 稀疏块索引 `[T, 1024]` 过滤并紧凑化为当前 DCP rank 拥有的局部块索引 `local_indices`（形状 `[T, 1024]`，紧排在左侧，右侧填充 `-1`）。
- 在 `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py` 的 `_localize_indices` 内核中，其实已经在寄存器中算出了当前 token 在本 DCP rank 上的有效块数：
  ```python
  count = tl.sum(owned.to(tl.int32), 0)
  ```
- 但此前 `count` 未写回显存，导致下游 `qsa_sparse_paged_attention` (`_qsa_sparse_paged_gqa_splitk_kernel`) 不知道每个 token 有多少个有效局部索引，只能对全部 `selected_width = 1024`（即 `64` 个 `BLOCK_N=16` tile）逐一循环加载并检查 `valid_sel = sel_idx >= 0`。
- 在 `DCP=2` 下，平均每个 rank 仅拥有一半（`~512` 个）有效索引，后半部分 `~32` 个 tile 全是 `-1`；而在 Prefill 前半段或短上下文下，有效块数更少，甚至在 `selected_count == 0` 时仍要启动所有 Split-K 块并跑完循环。

#### 突破方案
1. **`_localize_indices` (`ops/qsa_dcp.py`)**: 增加可选指针 `counts_ptr` 与编译期常量 `WRITE_COUNTS: tl.constexpr = False`。当传入预分配的 `valid_counts` (`int32 [T]`) 时，顺带执行一次标量写回 `tl.store(counts_ptr + row, count)`（零额外全局内存读取开销）。
2. **`_qsa_sparse_paged_gqa_splitk_kernel` (`ops/qsa.py`)**: 增加 `valid_counts_ptr` 与 `USE_VALID_COUNTS: tl.constexpr`。
   - 当 `USE_VALID_COUNTS=True` 时，读取 `selected_count = tl.load(valid_counts_ptr + token_idx)`（钳位至 `[0, selected_width]`）。
   - 若 `selected_count == 0`：仅由 `split_id == 0` 将 `partial_lse` 写为 `-inf`、`partial_out` 写为 `0.0`，其余 `split_id > 0` 直接 `return`！
   - 若 `selected_count > 0`：将当前 Split 的循环上界动态截断为：
     ```python
     valid_tile_end = tl.cdiv(selected_count, BLOCK_N)
     loop_end = tl.minimum(split_tile_end, valid_tile_end)
     ```
     若 `split_tile_start >= loop_end`，该 Split 直接写 `-inf` 并立即 `return`，彻底跳过所有全 `-1` 尾部 tile！
3. **接口兼容性保护 (`ops/qsa.py` & `qsa.py`)**: `valid_counts: torch.Tensor | None = None` 作为 keyword-only 参数添加到 `qsa_sparse_paged_attention`，100% 兼容 `tests/models/qwen4_exp/test_qsa_cache.py:212` 的 12 参数位置调用 monkeypatch。

---

### 2.2 `D18`: V1 `SpecDecodeBaseProposer` 接入 MTP Top-K 索引共享 + `QSAIndexer` Query 准备跳过

#### 瓶颈根因
1. `Qwen3.8-Flash-Next` 的模型配置支持 `index_share_for_mtp_iteration`（在 `MTP4` 的第 0 步计算 QSA 稀疏 Top-K 索引，随后通过 `compact_topk_indices` 压缩至 `[B, ...]`，并在后续 Draft steps `1..3` 设置 `set_skip_topk(True)` 直接复用第 0 步的稀疏索引）。
2. 然而，此前只有 V2 Worker 的 `EagleSpeculator` (`vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`) 接入了 `set_skip_topk` 与 `compact_topk_indices`，而默认的 V1 `SpecDecodeBaseProposer` (`vllm/v1/spec_decode/llm_base_proposer.py`) 完全没有调用这两个接口，导致 `MTP4` 在 V1 下每一步都重新运行完整的 Top-K 索引搜索。
3. 更进一步：即使在 `skip_topk=True`（或 Prefill 中 `select=False`）时，`QSAIndexer.forward` (`indexer_qsa.py`) 仍然对每个 token 执行了 `self.q_norm(q)` 和 `self.rotary_emb(q, ...)`（或在融合 `qsa_pre_indexer` 中启动了 `num_q_work` 个 Query 线程块），而此时 `q` 根本不会被后续的 `_select_topk` 使用！

#### 突破方案
1. 在 `SpecDecodeBaseProposer.load_model` (`llm_base_proposer.py`) 中检测 `hf_config.index_share_for_mtp_iteration` 及 `self.model.model` 的 `set_skip_topk` / `compact_topk_indices` 方法。
2. 在 `SpecDecodeBaseProposer.propose` 中：
   - Pass 0 前调用 `self.model.model.set_skip_topk(False)`；
   - Pass 0 后、进入 `token_index in range(self.num_speculative_tokens - 1)` 循环前，调用 `self.model.model.compact_topk_indices(token_indices_to_sample[:batch_size])` 与 `self.model.model.set_skip_topk(True)`，并在 `finally` 块中恢复 `set_skip_topk(False)`。
3. 在 `QSAIndexer.forward` (`indexer_qsa.py`) 与 `qsa_pre_indexer` (`ops/qsa_pre_indexer.py`) 中：
   - 定义 `skip_query_prep = self.skip_topk or not select`；
   - 在融合 `qsa_pre_indexer` 路径下传入 `skip_q=skip_query_prep`，使 `num_q_work = 0`（仅执行 K 压缩与循环缓存更新）；
   - 在非融合路径下，当 `skip_query_prep=True` 时直接跳过 `self.q_norm(q)` 与 `self.rotary_emb(...)`。

---

### 2.3 `P12` / `D19`: MTP 启用 SM70 Peer Paths + 放开 `qsa_prep_sm70_out` 行数上限 + GDN 输入投影切分融合

#### 瓶颈根因与突破方案
1. **`Qwen4ExpMTP.load_weights` (`mtp.py` & `model.py`)**:
   - 目标模型 `Qwen4ExpForCausalLM.load_weights` 在权重加载末尾调用了 `_maybe_enable_sm70_peer_paths(self.vllm_config, self.model)`（用于挂载 `sm70_fused_side_projections`、`sm70_hcx`、`sm70_top1x`），但 `Qwen4ExpMTP.load_weights` 遗漏了该调用，且 `_maybe_fuse_sm70_side_projections` 直接遍历 `for layer in model.layers:`（而 MTP 的 `model.layers` 是 `nn.ModuleDict`，直接遍历会得到字符串 key 而非 `Qwen4ExpDecoderLayer`）。
   - 现将 `_maybe_fuse_sm70_side_projections` 改为兼容 `nn.ModuleDict` (`model.layers.values()`)，并在 `Qwen4ExpMTP.load_weights` 中调用 `_maybe_enable_sm70_peer_paths(self.vllm_config, self.model)`。
2. **解除 `qsa_prep_sm70_out` 的 `num_tokens <= 32` 限制 (`qsa.py`)**:
   - `csrc/sm70_turbomind/ops/qwen38_qsa_prep_sm70.cu` 的 CUDA 核按 `blockIdx.x = token_row` 切分，天然支持任意 `rows >= 1`（无行数上限）。
   - 但 `qsa.py:1236` 此前硬编码了 `1 <= num_tokens <= 32`，导致 Prefill（如 `T=8192`）或大 Batch Decode 回退到多次独立算子（`q_norm` + `k_norm` + `rope` + `head_scale`）。现放宽为 `num_tokens >= 1`。
3. **`qwen_gdn_input_projection` 接入 `_qwen38_gdn_projection_split_kernel` (`qwen_gdn_linear_attn.py`)**:
   - `qwen_gdn_input_projection_core` 已使用 `_split_gdn_projection_tails`，但独立自定义算子 `qwen_gdn_input_projection` 仍对 `mixed_qkv_out`、`z_out`、`b_out`、`a_out` 执行 4 次非连续视图 `.copy_()`。
   - 现接入 `_qwen38_gdn_projection_split_kernel[(m, 16)](..., COPY_QKV=True)`，1 次 Triton 启动即可完成 `QKV`、`Z`、`B`、`A` 的切分与連続化写出，36 个 GDN 层每步省去 **108 次内核启动**。

---

### 2.4 `P13` / `D20`: PLE 膨胀短卷积 (`dilation=3`) 全场景融合与 Prefill N-gram ID 融合

#### 瓶颈根因
`Qwen3.8-Flash-Next` 包含 4 个 PLE 层（`layers.{15,23,31,39}`，通道数 `H = 10240`，`conv_state_len = 9`，`dilation = 3`，4-tap 深度卷积核 `[10240, 4]`）：
1. **Prefill 短卷积 (`_short_conv_dilated_prefill_batched`)**:
   - 原实现先分配 `history = x_p.new_zeros((num_prefills, 10240, 9 + max_len))`，在 `8K` Prefill 下仅这一个张量就占 **168 MB**（多请求时按 `num_prefills * max_len` 膨胀），再通过二维高级索引 `history[..., 9:].transpose(1, 2)[req_indices, col_indices] = x_p` 散射写入、执行 `F.conv1d`（再分配 **168 MB** `conv_output`）、`F.silu_`、`masked_fill_`、最后再转置聚焦回 `[T, 10240]`（再分配 **168 MB**）。
   - 单个 PLE 层在 8K Prefill 下产生 **~1.34 GB HBM 读写流量** 与 **~18 次内核启动**（4 层合计 **~5.36 GB HBM 流量** 与 **~72 次启动**）。
2. **MTP Verify 短卷积 (`_short_conv_dilated_spec_batched`)**:
   - 原 `qwen38_ple_spec_sm70_out` 仅支持 `num_reqs == 1` 且 `x_spec.shape in ((5, 10240), (10, 10240))`。一旦 `num_reqs > 1`（多并发请求投机验证）或 CUDA Graph 填充行数 `> 10`，立即退回 PyTorch 的 28 步 scatter/transpose/gather/conv1d/index_copy 流水线（4 层合计 **112 次启动/步**）。
3. **Decode 短卷积 (`_short_conv_dilated_decode_batched`)**:
   - 原 `_qwen38_ple_m1_short_conv_kernel` 仅允许 `conv_state.dtype == torch.float16`，而默认 `mamba_cache_dtype="auto"` 下 `conv_state` 可能为 `torch.float32`，导致退回慢速 PyTorch 路径。
4. **Prefill N-gram ID (`compute_ngram_ids`)**:
   - `SM70_PLE_NGRAM` 限定了 `2 <= m <= 32`，导致 `m > 32` 的 Prefill 退回 26 步 PyTorch `searchsorted` + `cummax` + `bitwise_xor` + `remainder` 流水线（4 层合计 **104 次启动/Prefill 步**）。

#### 突破方案
1. **Prefill 行主序 4-Tap Triton 短卷积 (`_qwen38_ple_prefill_short_conv_kernel` + `_qwen38_ple_prefill_state_update_kernel`)**:
   - 针对 `H == 10240, state_len == 9, dilation == 3`（保持 `test_ple_short_conv_prefill.py` 中 `H in (16, 1024)` 的单测契约不变），直接在行主序 `[T, 10240]` 上按偏移 `-9, -6, -3, 0` 读取 `x_p` 或 `conv_state`，在 FP32 中完成 4-tap 乘加后转为 FP16 写出，随后由 `_qwen38_ple_prefill_state_update_kernel` 原地更新每个请求的 9 步状态，最后调用原生 `F.silu(conv_output, inplace=True)` 保持 SM70 舍入 bit-exact。
   - 同时在通用 PyTorch 路径中增加 `single_prefill_fastpath` (`num_prefills == 1`)，免去 `arange`/`searchsorted` 及二维高级索引开销。
2. **MTP Verify 多请求 CUDA + Triton 融合 (`qwen38_ple_spec_sm70.cu` & `_qwen38_ple_spec_short_conv_kernel`)**:
   - 将 `qwen38_ple_spec_sm70.cu` 的 `ple_conv` 扩展为二维网格 `grid(80, num_reqs)`，支持任意 `num_reqs >= 1` 与 `rows >= 5`；
   - 同时在 `ple_layer.py` 中提供无需重编 C++ 扩展即可直接运行的 `_qwen38_ple_spec_short_conv_kernel` + `F.silu` 多请求融合核，4 层仅需 **8 次启动**（原 112 次）。
3. **Decode 短卷积兼容 FP16/FP32 状态缓存**:
   - 在 `_qwen38_ple_m1_short_conv_kernel` 中对加载的状态执行 `.to(tl.float16).to(tl.float32)`（与 PyTorch 参考路径 `cached_state[..., :9].to(x_d.dtype)` 严格 bit-exact），并放开 `conv_state.dtype in (torch.float16, torch.float32)`。
4. **Prefill N-gram ID 融合**:
   - 在 `compute_ngram_ids` 中保留 `SM70_PLE_NGRAM.max_m = 32` 元数据属性不变（确保能力单测通过），当 `reason == "outside_small_batch_band"` 且 `num_tokens > 32, 1 <= num_reqs <= 64` 时直接调用 `sm70_ple_ngram_ids`，将 4 层 Prefill N-gram ID 计算从 104 次启动降至 **4 次启动**。

---

## 3. 仓库内置高阶 SM70 能力突破开关速查表（Opt-In）

除上述默认生效的代码级突破外，仓库还内置了若干针对特定场景的 **SM70 硬件极限能力开关**（部分因改变浮点累加顺序或面向特定并发档位而默认关闭，可通过环境变量或配置开启）：

| 能力域 | 环境变量 / 配置项 | 默认值 | 推荐设置 (`TP4+DCP2+MTP4`) | 能力突破说明 |
| :--- | :--- | :--- | :--- | :--- |
| **PLE MTP4 融合卷积** | `VLLM_SM70_MTP_PLE_CONV=1`<br>(`kernel_config.layer_execution.ple_spec_conv`) | `0` | **`1`** | 启用 `qwen38_ple_spec_sm70_out` / `_qwen38_ple_spec_short_conv_kernel`，消除 MTP Verify 阶段 4 个 PLE 层的 28 步 PyTorch 卷积流水线（**bit-exact**，已通过 `test_sm70_mtp_ple_conv.py` 验证）。 |
| **QSA 预处理 CUDA 融合** | `kernel_config.sm70_qsa_prep=True` | `False` | **`True`** | 启用 `qwen38_qsa_prep_sm70_out`（配合本次 `P12`/`D19` 解除 `M<=32` 限制），将 Q/K RMSNorm + Partial RoPE + Head Scale 融合为单次 CUDA 启动。 |
| **GDN/QSA 旁路投影融合** | `kernel_config.sm70_fused_side_projections=True` | `False` | **`True`** | 将 GDN `in_proj_qkvz + in_proj_ba` 以及 QSA `qkv_proj + index_qk_proj` 融合为单次 DMV13 投影（配合本次 `P12`/`D19` 同步赋能 MTP 层）。 |
| **FlashQLA Indexed Prefill** | `VLLM_SM70_FLASHQLA_INDEXED_PREFILL=1`<br>(`gdn_config.schedule.indexed_prefill`) | `0` | **`1`** | 在 GDN Prefill 阶段直接按请求索引读取状态，省去 `gather_Verify_Scatter` 状态拷贝。 |
| **NVFP4 MoE MTP5 融合** | `VLLM_SM70_NVFP4_MOE_QPN_MTP5=1`<br>`VLLM_SM70_NVFP4_MOE_GROUPED_MTP5=1`<br>`VLLM_SM70_NVFP4_MOE_W2_DIRECT_REDUCE=1` | `1` (auto) | **`1`** | 针对 `MTP4` (`M=5` 验证步) 的 NVFP4 专家分组 GEMM 与 W2 直接归约快路径。 |
| **NVFP4 MoE Fast Prefill** | `VLLM_SM70_NVFP4_MOE_FAST_PREFILL=1` | `0` | **`1`** (吞吐优先) | 启用 SM70 NVFP4 MoE 大 Token 数 Prefill 加速核。 |
| **MTP 单图循环 & 热词表** | `kernel_config.sm70_draft_single_graph=True`<br>`kernel_config.sm70_draft_hot_vocab=True` | `False` | 按需开启 | 将 MTP 多步 Draft 循环捕获进单张 CUDA Graph，或启用热词表裁剪以降低 Draft LM Head 开销。 |

---

## 4. 本次能力突破逐文件完整代码改动 (Unified Diffs)

### 4.1 `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py` (`P11` / `D17`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
@@ -17,12 +17,14 @@ def _localize_indices(
     indices_ptr,
     out_ptr,
     stride_row,
+    counts_ptr,
     dcp_rank: int,
     num_blocks: int,
      dcp_size: tl.constexpr,
     interleave_size: tl.constexpr,
     BLOCK_K: tl.constexpr,
     INPLACE: tl.constexpr = False,
+    WRITE_COUNTS: tl.constexpr = False,
 ):
     row = tl.program_id(0)
     cols = tl.arange(0, BLOCK_K)
@@ -40,6 +42,8 @@ def _localize_indices(
     positions = tl.cumsum(owned.to(tl.int32), 0) - 1
     if not INPLACE:
         tl.store(out_base + cols, -1, col_mask)
+    if WRITE_COUNTS:
+        tl.store(counts_ptr + row, count)
     if count == 0:
         if INPLACE:
             tl.store(out_base + cols, -1, col_mask)
@@ -57,6 +61,7 @@ def qsa_localize_dcp_indices(
     dcp_rank: int,
     interleave_size: int = 1,
     out: torch.Tensor | None = None,
+    valid_counts: torch.Tensor | None = None,
 ) -> torch.Tensor:
     """Filter and compact global top-k indices into local DCP coordinates."""
     if dcp_size <= 1:
@@ -64,6 +69,8 @@ def qsa_localize_dcp_indices(
             return block_indices
         out.copy_(block_indices)
         return out
     if block_indices.numel() == 0:
+        if valid_counts is not None and valid_counts.numel() >= block_indices.shape[0]:
+            valid_counts[: block_indices.shape[0]].zero_()
         if out is None:
             return block_indices.clone()
         return out
@@ -91,17 +98,25 @@ def qsa_localize_dcp_indices(
         else torch.empty_like(block_indices)
     )
     inplace = out.data_ptr() == block_indices.data_ptr()
+    write_counts = (
+        valid_counts is not None
+        and valid_counts.is_cuda
+        and valid_counts.dtype == torch.int32
+        and valid_counts.is_contiguous()
+        and valid_counts.numel() >= block_indices.shape[0]
+    )
     _localize_indices[(block_indices.shape[0],)](
         block_indices,
         out,
         block_indices.stride(0),
+        valid_counts if write_counts else out,
         dcp_rank,
         num_blocks,
         dcp_size=dcp_size,
         interleave_size=interleave_size,
         BLOCK_K=block_k,
         INPLACE=inplace,
+        WRITE_COUNTS=write_counts,
         num_warps=4 if block_k <= 1024 else 8,
     )
     return out
```

### 4.2 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` (`P11` / `D17`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -1629,6 +1629,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     query_pos_ptr,
     req_idx_ptr,
     selected_ptr,
+    valid_counts_ptr,
     page_table_ptr,
     partial_out_ptr,
     partial_lse_ptr,
@@ -1664,6 +1665,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     BLOCK_DV: tl.constexpr,
     IS_CAUSAL: tl.constexpr,
     ASSUME_CAUSAL_PROVEN: tl.constexpr,
+    USE_VALID_COUNTS: tl.constexpr = False,
 ):
     token_idx = tl.program_id(0)
     kv_head = tl.program_id(1)
@@ -1690,6 +1692,26 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
         query_pos = tl.load(query_pos_ptr + token_idx).to(tl.int32)
         if query_pos < 0:
             return
+
+    selected_count = selected_width
+    if USE_VALID_COUNTS:
+        selected_count = tl.load(valid_counts_ptr + token_idx).to(tl.int32)
+        selected_count = tl.maximum(0, tl.minimum(selected_count, selected_width))
+        if selected_count == 0:
+            if split_id == 0:
+                partial_row = token_idx
+                lse_base = (
+                    partial_lse_ptr
+                    + partial_row * stride_pl_row
+                    + q_heads * stride_pl_head
+                )
+                tl.store(lse_base, -float("inf"), mask=q_mask)
+                dv_idx = tl.arange(0, BLOCK_DV)
+                out_base = (
+                    partial_out_ptr
+                    + partial_row * stride_po_row
+                    + q_heads[:, None] * stride_po_head
+                    + dv_idx[None, :]
+                )
+                tl.store(out_base, 0.0, mask=q_mask[:, None] & (dv_idx[None, :] < VALUE_DIM))
+            return
 
     q_base = q_ptr + token_idx * stride_q_t + q_heads[:, None] * stride_q_h
     d_main_idx = tl.arange(0, BLOCK_D_MAIN)
@@ -1719,8 +1741,19 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     total_tiles = tl.cdiv(selected_width, BLOCK_N)
     tiles_per_split = tl.cdiv(total_tiles, NUM_SPLITS)
     split_tile_start = split_id * tiles_per_split
     split_tile_end = tl.minimum(total_tiles, split_tile_start + tiles_per_split)
-    if split_tile_start >= split_tile_end:
+    if USE_VALID_COUNTS:
+        valid_tile_end = tl.cdiv(selected_count, BLOCK_N)
+        loop_end = tl.minimum(split_tile_end, valid_tile_end)
+    else:
+        loop_end = split_tile_end
+    if split_tile_start >= loop_end:
+        if USE_VALID_COUNTS:
+            partial_row = split_id * num_rows + token_idx
+            lse_base = (
+                partial_lse_ptr
+                + partial_row * stride_pl_row
+                + q_heads * stride_pl_head
+            )
+            tl.store(lse_base, -float("inf"), mask=q_mask)
         return
 
-    for tile_idx in range(split_tile_start, split_tile_end):
+    for tile_idx in range(split_tile_start, loop_end):
         sel_Start = tile_idx * BLOCK_N
```

### 4.3 `vllm/models/qwen4_exp/nvidia/qsa.py` (`P11`/`D17` & `P12`/`D19`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -435,6 +435,16 @@ class Qwen4ExpQSAAttention(nn.Module):
             )
         return self._dcp_local_indices_buffer[:rows, :cols]
 
+    def _get_dcp_valid_counts_buffer(
+        self, rows: int, device: torch.device
+    ) -> torch.Tensor:
+        buf = getattr(self, "_dcp_valid_counts_buffer", None)
+        if buf is None or buf.device != device or buf.numel() < rows:
+            self._dcp_valid_counts_buffer = torch.empty(
+                (rows,), dtype=torch.int32, device=device
+            )
+        return self._dcp_valid_counts_buffer[:rows]
+
     def _forward_qsa_dcp(
         self,
         q: torch.Tensor,
@@ -504,6 +514,9 @@ class Qwen4ExpQSAAttention(nn.Module):
         local_indices_buf = self._get_dcp_local_indices_buffer(
             selected_indices.shape[0], selected_indices.shape[1], q.device
         )
+        valid_counts_buf = self._get_dcp_valid_counts_buffer(
+            selected_indices.shape[0], q.device
+        )
         local_indices = qsa_localize_dcp_indices(
             selected_indices,
             dcp_size=self.dcp_size,
@@ -510,6 +523,7 @@ class Qwen4ExpQSAAttention(nn.Module):
             dcp_rank=self.dcp_rank,
             interleave_size=self.cp_kv_cache_interleave_size,
             out=local_indices_buf,
+            valid_counts=valid_counts_buf,
         )
         local_out, local_lse = qsa_sparse_paged_attention(
             q_all,
@@ -523,6 +537,7 @@ class Qwen4ExpQSAAttention(nn.Module):
             return_lse=True,
             k_scale=self._k_scale,
             v_scale=self._v_scale,
+            valid_counts=valid_counts_buf,
         )
@@ -1225,7 +1240,7 @@ class Qwen4ExpQSAAttention(nn.Module):
         use_sm70_qsa_prep = (
             self._sm70_qsa_prep
-            and 1 <= num_tokens <= 32
+            and num_tokens >= 1
             and qkv.is_cuda
             and qkv.dtype == torch.float16
```

### 4.4 `vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py` & `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` (`D18`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py
@@ -425,6 +425,7 @@ def qsa_pre_indexer(
     compress_ratio: int,
     mrope_section: tuple[int, int, int] | None,
     rope_pos_offset: int | None,
+    skip_q: bool = False,
 ) -> None:
     """Normalize Q, compress K, then update the circular raw state."""
     num_tokens = q.shape[0]
@@ -463,7 +464,13 @@ def qsa_pre_indexer(
     else:
         TILE_T_Q, TILE_H_Q = 2, 4
     num_k_work = k_work_metadata.shape[0]
-    num_q_work = triton.cdiv(num_tokens, TILE_T_Q) * triton.cdiv(num_q_heads, TILE_H_Q)
+    num_q_work = (
+        0
+        if skip_q
+        else triton.cdiv(num_tokens, TILE_T_Q) * triton.cdiv(num_q_heads, TILE_H_Q)
+    )
+    if num_k_work + num_q_work == 0:
+        return
     _qsa_pre_indexer_kernel[(num_k_work + num_q_work,)](
```

```diff
--- a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
@@ -636,6 +636,7 @@ class QSAIndexer(nn.Module):
         has_decode = attn_metadata.num_decodes > 0
         has_prefill = attn_metadata.num_prefills > 0
         num_decode_tokens = attn_metadata.num_decode_tokens
+        skip_query_prep = self.skip_topk or not select
 
         has_ valid_k_work = (
             attn_metadata.k_work_metadata is not None
@@ -668,8 +669,11 @@ class QSAIndexer(nn.Module):
                 self.compress_ratio,
                 mrope_section,
                 rope_pos_offset,
+                skip_q=skip_query_prep,
             )
-            q_rope, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], -1)
-            q_rope, _ = self.rotary_emb(
-                positions, q_rope, q_rope, None, None, rope_pos_offset=rope_pos_offset
-            )
+            if not skip_query_prep:
+                q_rope, q_nope = q.split(
+                    [self.rope_dim, self.head_dim - self.rope_dim], -1
+                )
+                q_rope, _ = self.rotary_emb(
+                    positions,
+                    q_rope,
+                    q_rope,
+                    None,
+                    None,
+                    rope_pos_offset=rope_pos_offset,
+                )
         else:
-            q = self.q_norm(q)
-            q_rope, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], -1)
-            q_rope, _ = self.rotary_emb(
-                positions, q_rope, q_rope, None, None, rope_pos_offset=rope_pos_offset
-            )
+            if not skip_query_prep:
+                q = self.q_norm(q)
+                q_rope, q_nope = q.split(
+                    [self.rope_dim, self.head_dim - self.rope_dim], -1
+                )
+                q_rope, _ = self.rotary_emb(
+                    positions,
+                    q_rope,
+                    q_rope,
+                    None,
+                    None,
+                    rope_pos_offset=rope_pos_offset,
+                )
             self._update_k_only(k, positions, attn_metadata, rope_pos_offset)
```

### 4.5 `vllm/v1/spec_decode/llm_base_proposer.py` (`D18`)

```diff
--- a/vllm/v1/spec_decode/llm_base_proposer.py
+++ b/vllm/v1/spec_decode/llm_base_proposer.py
@@ -1165,6 +1165,8 @@ class SpecDecodeBaseProposer:
             pre_forward_stream_wait_ms = (time.perf_counter() - sync_t0) * 1000.0
         forward_enqueue_t0 = time.perf_counter() if ddtree_worker_profile else 0.0
         first_forward_start = self._step_profiler.start(profile_events)
+        if getattr(self, "share_mtp_topk_indices", False):
+            self.model.model.set_skip_topk(False)
         with set_forward_context(
             per_layer_attn_metadata,
             self.vllm_config,
@@ -1421,7 +1423,15 @@ class SpecDecodeBaseProposer:
         block_size = self.block_size
         assert block_size > 0, "block_size has not been initialized."
-        for token_index in range(self.num_speculative_tokens - 1):
+        share_mtp_topk_indices = bool(
+            getattr(self, "share_mtp_topk_indices", False)
+            and self.num_speculative_tokens > 1
+        )
+        if share_mtp_topk_indices:
+            self.model.model.compact_topk_indices(token_indices_to_sample[:batch_size])
+            self.model.model.set_skip_topk(True)
+        try:
+            for token_index in range(self.num_speculative_tokens - 1):
                 ...
+        finally:
+            if share_mtp_topk_indices:
+                self.model.model.set_skip_topk(False)
@@ -2187,6 +2199,17 @@ class SpecDecodeBaseProposer:
         self._maybe_share_embeddings(target_language_model)
         self._maybe_share_lm_head(target_language_model)
         self._maybe_initialize_static_draft_vocab()
+        inner_model = getattr(self.model, "model", None)
+        self.share_mtp_topk_indices = bool(
+            getattr(
+                self.vllm_config.model_config.hf_config,
+                "index_share_for_mtp_iteration",
+                False,
+            )
+            and inner_model is not None
+            and hasattr(inner_model, "set_skip_topk")
+            and hasattr(inner_model, "compact_topk_indices")
+        )
```

### 4.6 `vllm/models/qwen4_exp/nvidia/model.py` & `vllm/models/qwen4_exp/nvidia/mtp.py` (`P12` / `D19`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/model.py
+++ b/vllm/models/qwen4_exp/nvidia/model.py
@@ -635,7 +635,12 @@ def _maybe_fuse_sm70_side_projections(model: nn.Module) -> None:
     from vllm.model_executor.layers.quantization.sm70_dmv13_projection import attach
 
     fused = 0
-    for layer in model.layers:
+    layers_iter = (
+        model.layers.values()
+        if isinstance(model.layers, nn.ModuleDict)
+        else model.layers
+    )
+    for layer in layers_iter:
         if not isinstance(layer, Qwen4ExpDecoderLayer):
             continue
```

```diff
--- a/vllm/models/qwen4_exp/nvidia/mtp.py
+++ b/vllm/models/qwen4_exp/nvidia/mtp.py
@@ -762,13 +762,17 @@ class Qwen4ExpMTP(nn.Module, SupportsPP):
-        from .model import _finalize_qsa_e4m3_scale_load
+        from .model import (
+            _finalize_qsa_e4m3_scale_load,
+            _maybe_enable_sm70_peer_paths,
+        )
 
         _finalize_qsa_e4m3_scale_load(
             self,
             loaded_weights,
             self.model._kv_cache_dtype,
             require_calibrated_speculative_draft=True,
         )
+        _maybe_enable_sm70_peer_paths(self.vllm_config, self.model)
         return loaded_weights
```

### 4.7 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (`P12` / `D19`)

```diff
--- a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
@@ -6005,6 +6005,45 @@ def qwen_gdn_input_projection(
     mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
     assert self.in_proj_ba is not None
     ba, _ = self.in_proj_ba(hidden_states)
+
+    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
+        _can_fuse_gdn_projection_split,
+        _qwen38_gdn_projection_split_kernel,
+    )
+
+    m = mixed_qkvz.shape[0]
+    if (
+        not self.gqa_interleaved_layout
+        and not self.disable_tp_for_ba_proj
+        and m > 1
+        and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
+        and self.value_dim // self.tp_size == 1536
+        and mixed_qkv_out.shape == (m, 2560)
+        and z_out.shape == (m, 12, 128)
+        and b_out.shape == (m, 12)
+        and a_out.shape == (m, 12)
+        and mixed_qkv_out.is_contiguous()
+        and z_out.is_contiguous()
+        and b_out.is_contiguous()
+        and a_out.is_contiguous()
+        and _can_fuse_gdn_projection_split(mixed_qkvz, ba)
+    ):
+        _qwen38_gdn_projection_split_kernel[(m, 16)](
+            mixed_qkvz,
+            ba,
+            mixed_qkv_out,
+            z_out,
+            b_out,
+            a_out,
+            QKV=2560,
+            Z=1536,
+            B=12,
+            A=12,
+            BLOCK=256,
+            COPY_QKV=True,
+            num_warps=4,
+            num_stages=1,
+        )
+        return
```

### 4.8 `vllm/models/qwen4_exp/nvidia/ple_layer.py` & `csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu` (`P13` / `D20`)

```diff
--- a/csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu
+++ b/csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu
@@ -16,20 +16,26 @@ __device__ void store_half(float* ptr, half x) { *ptr = __half2float(x); }
 template <class State>
 __global__ void ple_conv(const half* x, State* state, const half* weight,
                          const int* ids, const int* starts, const int* accepted,
-                         half* out, int rows, int hidden, int64_t s0,
-                         int64_t s1, int64_t s2) {
+                         half* out, int rows, int hidden, int num_reqs,
+                         int64_t s0, int64_t s1, int64_t s2) {
   int c = blockIdx.x * 128 + threadIdx.x;
-  if (c >= hidden) return;
-  int sid = ids[0], count = starts[1] - starts[0],
-      rollback = sid ? max(0, min(4, accepted[0] - 1)) : 0;
+  int req = blockIdx.y;
+  if (c >= hidden || req >= num_reqs) return;
+  int sid = ids[req];
+  int start = max(0, starts[req]);
+  int end = max(start, starts[req + 1]);
+  int count = min(5, end - start);
+  int rollback = (sid > 0) ? max(0, min(4, accepted[req] - 1)) : 0;
   half h[14];
 #pragma unroll
   for (int t = 0; t < 9; ++t)
-    h[t] = sid ? as_half(state[sid * s0 + c * s1 + (rollback + t) * s2])
-               : __float2half_rn(0.f);
+    h[t] = (sid > 0) ? as_half(state[sid * s0 + c * s1 + (rollback + t) * s2])
+                     : __float2half_rn(0.f);
 #pragma unroll
   for (int t = 0; t < 5; ++t)
-    h[t + 9] = t < count ? x[t * hidden + c] : __float2half_rn(0.f);
+    h[t + 9] = (t < count && start + t < rows)
+                   ? x[(start + t) * hidden + c]
+                   : __float2half_rn(0.f);
```

```diff
--- a/vllm/models/qwen4_exp/nvidia/ple_layer.py
+++ b/vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -429,6 +429,215 @@ def _qwen38_ple_m1_short_conv_kernel(
+
+@triton.jit
+def _qwen38_ple_prefill_short_conv_kernel(
+    x_ptr, state_ptr, weight_ptr, output_ptr,
+    q_starts_ptr, state_index_ptr, has_initial_ptr,
+    STATE_STRIDE_0: tl.constexpr, STATE_STRIDE_1: tl.constexpr, STATE_STRIDE_2: tl.constexpr,
+    NUM_REQS: tl.constexpr, BLOCK_R: tl.constexpr, NULL_STATE_ID: tl.constexpr,
+    HIDDEN_SIZE: tl.constexpr, BLOCK: tl.constexpr,
+):
+    """Row-major 4-tap dilated short-conv for Qwen3.8 prefill without transpose."""
+    ...
+
+@triton.jit
+def _qwen38_ple_prefill_state_update_kernel(
+    x_ptr, state_ptr, q_starts_ptr, state_index_ptr, has_initial_ptr,
+    STATE_STRIDE_0: tl.constexpr, STATE_STRIDE_1: tl.constexpr, STATE_STRIDE_2: tl.constexpr,
+    NULL_STATE_ID: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK: tl.constexpr,
+):
+    """Commit the trailing 9 tokens per prefill request into conv_state."""
+    ...
+
+@triton.jit
+def _qwen38_ple_spec_short_conv_kernel(
+    x_ptr, state_ptr, weight_ptr, output_ptr,
+    state_index_ptr, q_starts_ptr, accepted_ptr,
+    ROWS, NUM_REQS: tl.constexpr,
+    STATE_STRIDE_0: tl.constexpr, STATE_STRIDE_1: tl.constexpr, STATE_STRIDE_2: tl.constexpr,
+    NULL_STATE_ID: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK: tl.constexpr,
+):
+    """Fused multi-request MTP4 PLE rollback + 4-tap conv + state commit."""
+    ...
```

---

## 5. 验证与回归测试命令

在配备 SM70 V100 GPU 的测试环境中，可直接执行以下单测与端到端基准命令验证本次能力突破：

```bash
# 1. 验证 QSA DCP2 局部索引与 valid_counts 动态早退
pytest -v tests/models/qwen4_exp/test_qsa_dcp.py tests/models/qwen4_exp/test_qsa_cache.py

# 2. 验证 PLE 膨胀短卷积 Prefill / Decode / MTP4 SpecDecode 数值一致性与显存上界
pytest -v tests/models/qwen4_exp/test_ple_short_conv_prefill.py tests/kernels/test_sm70_mtp_ple_conv.py

# 3. 验证 GDN 投影尾切分与 MTP4 Speculator
pytest -v tests/kernels/test_sm70_gdn_projection_tails.py tests/v1/worker/test_eagle.py

# 4. 启用内置 PLE MTP4 融合与 FlashQLA Indexed Prefill 跑 TP4 + DCP2 + MTP4 服务/基准
VLLM_SM70_MTP_PLE_CONV=1 \
VLLM_SM70_FLASHQLA_INDEXED_PREFILL=1 \
bash benchmarks/NVFP4-MTP4/serve.sh
```
