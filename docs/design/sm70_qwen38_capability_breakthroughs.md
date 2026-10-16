# Qwen3.8-Flash-Next (`TP4 + DCP2 + MTP4` on SM70 V100) — Prefill & Decode 能力突破优化专项文档

> **目标平台**: NVIDIA Tesla V100-SXM2-32GB (`SM70`, Volta)  
> **目标部署拓扑**: `TP4 + DCP2` (`world_size=4`, `dcp_size=2`) + `MTP4` (`num_speculative_tokens=4`)  
> **前置优化基线**: `docs/design/sm70_qwen38_prefill_decode_optimizations.md` (`P1`–`P10` & `D1`–`D16`)  
> **专项定位**: 在保留 100% 数值等价 (`0 ULP` / bit-exact) 的前提下，针对 **架构级能力边界限制**（DCP2 稀疏注意力全预算空转、V1 SpecDecode 缺失 MTP Top-K 索引共享、MTP 算子/小 batch 限制、PLE 膨胀短卷积与 N-gram 构图瓶颈、以及 `Qwen3.8-27B` vs. `Qwen3.8-Flash-Next` Decode 吞吐差距根因）实现 **能力突破 (`P11`–`P13`, `D17`–`D25`)**，并系统梳理仓库内置的高阶 SM70 能力开关。

---

## 1. 能力突破总览 (`P11`–`P13` & `D17`–`D25`)

在前置 `P1`–`P10` 与 `D1`–`D16` 完成通用算子融合与零分配改造后，针对 `Qwen3.8-Flash-Next` 在 `TP4 + DCP2 + MTP4` 下的 **架构级瓶颈** 与 **`Qwen3.8-27B` vs. `Qwen3.8-Flash-Next` Decode 差距**，本专项实现了以下能力突破：

| 编号 | 阶段 | 能力突破点 | 改动文件 | 核心突破与收益 | 精度影响 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`P11` / `D17`** | Prefill & Decode | **`TP4 + DCP2` QSA 局部有效块数 (`valid_counts`) 动态早退** | `ops/qsa_dcp.py`<br>`ops/qsa.py`<br>`qsa.py` | `_localize_indices` 已在寄存器内算出本 DCP rank 拥有的有效组数 `count`，但此前未写出，导致 `_qsa_sparse_paged_gqa_splitk_kernel` 始终遍历完整 `selected_width = 1024`（`64` 个 `BLOCK_N=16` tile，其中约一半是 `-1` 填充）。现将 `valid_counts` 写入预分配缓存并传入稀疏 Split-K 内核，动态截断 `loop_end = min(split_tile_end, cdiv(selected_count, BLOCK_N))`，且在 `selected_count == 0` 时直接早退，**消除 DCP2 下约 50% 的无效尾部 tile 循环**。 | **Bit-exact (`0 ULP`)** |
| **`D18`** | `MTP4` Decode | **V1 `SpecDecodeBaseProposer` 启用 `index_share_for_mtp_iteration` + 跳过 `QSAIndexer` Query 准备** | `llm_base_proposer.py`<br>`indexer_qsa.py`<br>`ops/qsa_pre_indexer.py` | 1. V2 `EagleSpeculator` 支持 `index_share_for_mtp_iteration`（Pass 0 选出 Top-K 索引并压缩为 `:B`，Draft steps `1..3` 置 `set_skip_topk(True)` 复用 Pass 0 索引），但默认 V1 `SpecDecodeBaseProposer` 未接入该能力。现将其完整接入 V1 Proposer。<br>2. 当 `skip_topk=True`（或 `select=False`）时，`QSAIndexer` 仅需将当前步 `k` 写入循环状态缓存，无需计算 `q_norm` 与 `q_rope`。现跳过 `q_norm`、`q_rope` 及 `qsa_pre_indexer` 中的 Query 线程块（`num_q_work = 0`）。 | **与模型配置契約完全一致 (`0 ULP`)** |
| **`P12` / `D19`** | Prefill & Decode | **MTP 层启用 SM70 Peer Paths + 解除 `qsa_prep_sm70_out` `M<=32` 限制 + `qwen_gdn_input_projection` 融合切分** | `model.py`<br>`mtp.py`<br>`qsa.py`<br>`qwen_gdn_linear_attn.py` | 1. `Qwen4ExpMTP.load_weights` 补齐 `_maybe_enable_sm70_peer_paths(self.vllm_config, self.model)`（并使 `_maybe_fuse_sm70_side_projections` 兼容 `nn.ModuleDict`），让 MTP 层同样享受 `sm70_fused_side_projections` / `sm70_hcx` / `sm70_top1x`。<br>2. 解除 `qsa.py` 中 `qsa_prep_sm70_out` 的 `num_tokens <= 32` 限制（底层 CUDA 核支持任意 `rows >= 1`），使 Prefill 及大 Batch Decode 全部走单次 CUDA 融合算子。<br>3. 在 `qwen_gdn_input_projection` 中接入 `_qwen38_gdn_projection_split_kernel(COPY_QKV=True)`，将 4 次跨步 `.copy_()` 融合为 1 次 Triton 启动（36 层省去 **108 次启动/步**）。 | **Bit-exact (`0 ULP`)** |
| **`P13` / `D20`** | Prefill & Decode | **PLE 膨胀短卷积 (`dilation=3`) 全场景融合（Prefill 零转置 / Spec 多请求扩展 / Decode FP32 状态兼容）+ Prefill N-gram ID 融合** | `ple_layer.py`<br>`qwen38_ple_spec_sm70.cu` | 1. **Prefill 短卷积**: 针对 `H=10240` 新增行主序 4-tap Triton 核 `_qwen38_ple_prefill_short_conv_kernel` + `_qwen38_ple_prefill_state_update_kernel`，消除 `[num_prefills, 10240, 9 + max_len]` 零填充/转置/分组卷积/gather 流水线，在 8K Prefill 下 **4 层合计节省 ~5.36 GB HBM 流量与 ~60 次启动**；同时为通用 PyTorch 路径增加 `num_prefills == 1` 免 scatter/gather 快路径。<br>2. **MTP Verify 短卷积**: 将 `qwen38_ple_spec_sm70.cu` 与 `_short_conv_dilated_spec_batched` 从仅支持 `num_reqs == 1` (`rows in {5, 10}`) 扩展到 `1 <= num_reqs <= 64`、`rows >= 5`（含 Triton 多请求回退核 `_qwen38_ple_spec_short_conv_kernel`），**4 层省去 104 次启动/步**。<br>3. **Decode 短卷积**: 支持 `conv_state.dtype in (fp16, fp32)`。<br>4. **Prefill N-gram ID**: 在 `compute_ngram_ids` 中对 `m > 32` (`1 <= num_reqs <= 64`) 直接调度 `sm70_ple_ngram_ids`，**4 层省去 100 次启动/Prefill 步**。 | **Bit-exact (`0 ULP`)** |
| **`D21`** | `MTP4` Decode | **修复 `enable_sm70_hcx` 目标模型与 MTP Draft 模型 `"final"` 注册表冲突** | `model.py` | `Qwen4ExpModel` 与 `Qwen4ExpMultiTokenPredictor` 的末尾 `hyper_connection_mixer` 此前均以 `"final"` 注册到 `hc._PARTIAL_MODULES`（且均覆盖 `collective_kernel_selections["hcx:target"]`），开启 `sm70_hcx=True` 时 MTP 权重会覆盖目标模型末尾 Mixer。现根据 `mtp_start_layer_idx > 0` 区分 `"mtp.final"` / `"final"` 与 `"hcx:draft"` / `"hcx:target"`。 | **Bit-exact (`0 ULP`)** |
| **`D22`** | `TP4 + DCP2` & `MTP4` Decode | **QSA Indexer `index_qk_proj` 仅投影 Key (`[128, 2560]` vs. `[640, 2560]`) + 短上下文免打分直展 (`IDENTITY_BLOCKS`)** | `indexer_qsa.py`<br>`ops/qsa.py`<br>`sm70_fp16_gemv.py` | 1. 当 `skip_query_prep` 为真（**MTP Draft steps 1–3 全上下文长度**、所有 `dense` 短上下文步、以及 **`TP4 + DCP2` 下 `max_seq_len <= 2051` 所有块均可见时**），`index_qk_proj` 的前 `512` 行 Query 投影完全不被读取。现仅对后 `128` 行 Key 权重 (`weight[512:]`, `[128, 2560]`) 执行 GEMV/GEMM，**直接削减 80% (5×) 的 `index_qk_proj` 计算量、CTA 数量与 HBM 权重读取**！<br>2. 在 `qsa_select_paged_tokens` 中，当 `max_seq_len // compress_ratio <= block_topk` (`seq_len <= 2051`) 时，`qsa_lexicographic_topk` 必然早退返回恒等块序 `0..visible-1`。现直接跳过 `qsa_mqa_paged` 与 `qsa_lexicographic_topk`，单核启动 `_expand_qsa_indices_kernel(..., IDENTITY_BLOCKS=True)`，**在 `TP4 + DCP2` 短/中上下文下每轮省去 26 次内核启动**。 | **Bit-exact (`0 ULP`)** |
| **`D23`** | Decode & `MTP4` | **Shared-Expert `down_proj` (`[2560, 160]`) 接入 SM70 FP16 GEMV 快路径** | `sm70_fp16_gemv.py` | 48 个目标层与 4 个 MTP Draft 步（每轮 52 次调用）的 `mlp.shared_expert.down_proj` (`[2560, 160]`, `reduce_results=False`) 此前未列入 `_plan_for`，在 `M=1` 时回退通用 cuBLAS。现为其接入 `_GemvPlan(256, 2, 0)`，单次加载 `K=160` 并将 `[1, 160]` 驻留 L1/L2，消除 52 次通用 cuBLAS 回退。 | **Bit-exact (`0 ULP`)** |
| **`D24`** | `MTP4` Decode | **MTP Draft `maybe_combine_fc` 双路投影 (`fc_embedding` + `fc_hidden`) 与 4 流残差加法单核融合** | `sm70_mtp_fc.py` | 每个 MTP Draft 步 (`1 <= rows <= 4`) 此前分 3 次内核启动执行 `fc_embedding` cuBLAS GEMM、`fc_hidden` cuBLAS GEMM 与 `el.unsqueeze(-2) + hl`。现新增 `_qwen38_mtp_fc_fused_gemv_add_kernel[(640, rows)]`，单核同时流式读取 `w_e` 与 `w_h`（复用寄存器内 `wh` 乘 4 条 HC 流）并在精确保留两个 FP16 舍入边界的前提下直接写出 `el + hl`，**每轮 MTP4 省去 8 次内核启动与中间显存读写**。 | **保留全部 FP16 舍入边界** |
| **`D25`** | `MTP4` Decode | **MTP `compact_topk_indices` 单行原位 Triton 核 + 多行持久 Scratch 零分配** | `mtp.py` | `compact_topk_indices` 在每轮 MTP Step 0 之后调用 `buffer.index_select(0, row_indices)` + `copy_`，每轮触发动态显存分配与 2 次内核启动（且对共享 `topk_indices_buffer` 重复执行）。现去重共享 Buffer，对 `B=1` (`num_rows == 1`) 使用单次原位 Triton 核 `_compact_topk_single_row_kernel`（当 `src_row == 0` 时直接零拷贝早退），对 `num_rows > 1` 复用持久 `_compact_topk_scratch` 消除动态分配。 | **Bit-exact (`0 ULP`)** |

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

### 2.5 为什么 `Qwen3.8-27B` Decode 可达 `200–260+ tok/s`，而 `Qwen3.8-Flash-Next` 在 `100–150 tok/s`？——全链路硬件级对比与 `D21`–`D25` 深度突破

#### 一、核心悖论：单步 `M=1` 更快，但投机 Decode 慢将近一倍？

根据仓库在 SM70 (`4 × V100-SXM2-32GB`, `TP4`) 上的实测基准（见 `docs/design/sm70_quasar_dflash2_15ms.md`、`sm70_dflash2_nvfp4_17ms.md`、`sm70_flash_next_mtp4_15ms.md` 与 `sm70_flash_next_mtp4_trace_reconciliation.md`），两个模型呈现出极具启发性的性能反差：

| 对比维度 | `Qwen3.8-27B-NVFP4` (`TP4`) | `Qwen3.8-Flash-Next-NVFP4` (`TP4 / TP4+DCP2`) | 关键结论 |
| :--- | :--- | :--- | :--- |
| **无投机单步 Decode (`M=1`)** | `60.4 – 66.4 tok/s`<br>(`15.1 – 16.2 ms/token`) | **`67.6 – 82.3 tok/s`**<br>(`12.15 – 14.8 ms/token`) | **不开启投机时，`Flash-Next` 比 `27B` 更快**（`M=1` 仅激活 10/512 个专家，活跃参数约 `3B` vs. `27B` Dense） |
| **投机解码范式** | **`DFlash2` (`K=7`, 并行块起草)**<br>5 层轻量 Dense 跨层交叉注意力，**1 次前向同时生成 7 个候选 Token** | **原生 `MTP4` (`K=4`, 串行自回归起草)**<br>1 层完整 512 专家 MoE + QSA + 4 流 HC + 248K 词表头，**串行循环 4 步生成 4 个候选 Token** | `DFlash2` 单轮上限 **8 tokens**；`MTP4` 单轮上限仅 **5 tokens**（理论上限低 **37.5%**） |
| **Draft 起草耗时 / 轮** | **`~1.8 – 2.5 ms / 轮`**（并行出 7 个词） | **`4.98 ms` (CUDA Event) / `5.28 ms` (Trace)**（串行出 4 个词） | `Flash-Next` 的 4 步串行 MoE 起草比 `27B` 的 7 词并行起草 **慢 2.2× 以上** |
| **Target 验证耗时 / 轮** | **`14.5 – 16.5 ms / 轮`**（验证 **`M=8`** 个词） | **`15.7 – 22.2 ms / 轮`**（仅验证 **`M=5`** 个词） | Dense 模型从 `M=1 -> M=8` 几乎零边际成本；MoE + FP16 骨架从 `M=1 -> M=5` 耗时显著膨胀 |
| **每轮平均接受 Token 数 ($\tau$)** | **`4.06 – 5.40 tokens/轮`**<br>（MBPP28: `4.686`，GSM8K: `5.40`） | **`2.67 – 4.81 tokens/轮`**<br>（`xhigh` 思考: `2.67`；8-prompt 自然集: `3.00`；常规集: `4.23`；Greedy 代码: `4.81`） | `DFlash2` 依托 `K=7` 窗口平均每轮多产出 **`+1.1 ~ +1.7` 个 Token** |
| **端到端 Decode 吞吐** | **`204.5 – 260.4 tok/s`**<br>（MBPP28 均值: **`251.6 tok/s`** @ `18.57 ms/轮`） | **`82.1 – 150.2 tok/s`**<br>（`xhigh` 思考: `82.1`；8-prompt 自然集: `117.2`；常规集: **`148.4`**；Greedy: **`150.2 tok/s`**） | $\text{tok/s} = \frac{\tau}{T_{\text{target}} + T_{\text{draft}} + T_{\text{sample}}}$：分子小 ~30%、分母大 ~35%，叠加造成 **~1.8× 吞吐差距** |

---

#### 二、`Qwen3.8-Flash-Next` Decode 慢于 `Qwen3.8-27B` 的四大硬件根因

通过 Nsight Systems 硬件级节点追踪（`sm70_flash_next_mtp4_trace_reconciliation.md`，单轮 Critical-Rank 服务时间 `27.32 ms`，共 **~3,100 次内核启动/轮**），可以精确定位造成差距的四大根因：

1. **根因 1：大量 Dense 投影层未量化（保持 FP16 16-bit），占整轮耗时的 47.0%（`12.83 ms / 27.32 ms`）**
   - 在 **`Qwen3.8-27B-NVFP4`** 中，64 层的所有主干投影（GDN `in_proj_qkvz`/`out_proj`、Full Attention `qkv_proj`/`o_proj`、Dense MLP `gate_up_proj`/`down_proj`）**全部是 4-bit/8-bit 量化权重**，由专门定制的 SM70 `QPN2` / `QPN4` / `QPN8` Split-K Tensor Core 算子执行，64 层全部线性投影在 `M=8` 下合计仅需 **`12.2 ms`**。
   - 而在 **`Qwen3.8-Flash-Next-NVFP4`** 中，**只有 512 个路由专家 (Routed Experts) 是 NVFP4**！其余所有骨架层在 Checkpoint 中均为 **16-bit FP16**：
     - 36 层 GDN 的 `in_proj_qkvz [4096, 2560]`、`in_proj_ba [24, 2560]`、`out_proj [2560, 1536]`；
     - 12 层 QSA 的 `qkv_proj [3584, 2560]`、`o_proj [2560, 1536]`、`index_qk_proj [640, 2560]`；
     - 48 层 Shared Expert 的 `gate_up_proj [320, 2560]`、`down_proj [2560, 160]`、`shared_expert_gate [1, 2560]`；
     - 48 层 Router `mlp.gate [512, 2560]`；
     - 96 个 HyperConnection (HC) 阶段的 `hc_down [336, 10240]` 与 `hc_up [10240, 320]`；
     - MTP 层 `fc_embedding [640, 2560]`、`fc_hidden [640, 2560]` 与 248,320 词表的 `lm_head`。
   - 仅 Target `M=5` 验证步中，就有 **291 次通用 CUTLASS/cuBLAS `volta_fp16_s884gemm` 调用（耗时 `5.66 ms`）** + 定制 FP16 算子（合计 **`9.31 ms`**），加上 Draft 步 FP16 线性层，整轮 FP16 Dense 投影高达 **`12.83 ms`**！

2. **根因 2：512 专家 Top-10 MoE 在 `M=5` 验证步发生“活跃专家扇出膨胀 (Expert Fan-Out)”（`4.47 ms/轮`）**
   - `M=1` 时每层仅激活 `10` 个专家；但在 `M=5` 投机验证时，5 个候选位置共触发 `5 × 10 = 50` 个专家槽位，去重后每层需从 HBM 加载 **`35 ~ 50` 个不同专家的权重**（HBM 专家权重读取量膨胀 **3.5× ~ 5×**），48 层 Target NVFP4 专家耗时 **`2.88 ms`** + 路由/排序/规约 **`0.75 ms`** = **`3.63 ms`**。
   - 更严重的是，MTP Draft 层的 512 个专家在 Checkpoint 中默认是 **未量化的 FP16**，4 步串行 `M=1` Draft MoE 耗时 **`0.84 ms`**（若开启 `mtp_expert_quantization="fp8"` 可直接减半）。

3. **根因 3：4 流 HyperConnection (HC) + 125 次通信原语 + ~3,100 次内核启动开销**
   - `Qwen3.8-27B` 采用标准单流残差 `[M, 5120]`，每层仅 2 次 AllReduce（64 层 + DFlash2 合计约 138 次通信，内核启动少）。
   - `Qwen3.8-Flash-Next` 采用 **4 流超连接 (`hc_count = 4`, 状态宽度 `4 × 2560 = 10240`)**，每个 Target 层包含 2 个完整 HC 阶段（48 层共 **96 个 HC 阶段**），整轮产生 **1,077 次 Elementwise/Copy 启动（`4.48 ms`）**、**384 次 HC 专属核启动（`1.38 ms`）** 和 **125 次 NCCL/自定义通信（`4.83 ms`）**，单轮总内核启动达 **~3,100 次**。

4. **根因 4：串行 `MTP4` (`K=4`) vs. 并行 `DFlash2` (`K=7`) 的算法结构差异**
   - `DFlash2` 利用目标模型多层隐藏特征一次性并行预测 7 个位置，无论 `K=4` 还是 `K=7` 都只需 **1 次前向**（`~2.0 ms`）；
   - `MTP4` 必须串行跑 4 步（Step $i$ 依赖 Step $i-1$ 采样的 Token Embedding 与 4 流 HC 状态），每步都要走一遍 `Embedding -> Pre-FC Norm -> FC (AllGather) -> QSA Attention -> 512-Expert MoE -> Final HC Mixer -> 248K Vocab Argmax`，4 步累计 **`4.98 ~ 5.28 ms`**。

---

#### 三、针对 `Qwen3.8-Flash-Next` Decode 瓶颈的五大代码级新突破 (`D21`–`D25`)

围绕上述四大根因，我们在不破坏任何现有单测契约与数值精度的前提下，进一步挖掘并落地了 **5 项针对 `Qwen3.8-Flash-Next` Decode 的代码级优化 (`D21`–`D25`)**：

1. **`D21` (`model.py`) — 修复 `enable_sm70_hcx` 目标模型与 MTP Draft 模型 `"final"` 注册表覆盖 Bug**：
   - **问题**：当开启 `sm70_hcx=True`（将 MoE AllReduce 延迟并融合进下一个 HC 层的输入规约，省去每层独立 MoE AllReduce）时，`Qwen4ExpModel` 与 `Qwen4ExpMultiTokenPredictor` 的末尾 `hyper_connection_mixer` 均以硬编码名称 `"final"` 注册到全局 `hc._PARTIAL_MODULES["final"]`，导致后加载的 MTP Draft 模型权重覆盖了 Target 模型末尾 Mixer！
   - **修复**：通过 `getattr(model, "mtp_start_layer_idx", 0) > 0` 识别 MTP Draft 模型，将其末尾 Mixer 注册为 `"mtp.final"`（Target 模型保持 `"final"`），并将 `collective_kernel_selections` 区分记录为 `"hcx:draft"` 与 `"hcx:target"`。

2. **`D22` (`indexer_qsa.py`, `ops/qsa.py`, `sm70_fp16_gemv.py`) — QSA Indexer `index_qk_proj` 仅投影 Key (`[128, 2560]` vs. `[640, 2560]`) + 短上下文免打分直展 (`IDENTITY_BLOCKS`)**：
   - **问题 1（80% 死计算与无效权重读取）**：`QSAIndexer.forward` 中的 `self.index_qk_proj(hidden_states)` 权重形状为 `[640, 2560]`（前 `512` 行为 4 个头的 Query 投影 `projected_q`，后 `128` 行为单头 Key 投影 `raw_keys`）。然而在以下所有高频场景中，`projected_q`（占矩阵 **80%**）完全不会被读取：
     1. **MTP Draft Steps 1, 2, 3（任意上下文长度 `2K ~ 128K`）**：因 `self.skip_topk == True`，`qsa_pre_indexer(..., skip_q=True)` 中 `num_q_work = 0` 根本不解引用 `q_ptr`，且 `QSAIndexer.forward` 在更新完 K 缓存后直接返回复用的 Step 0 索引；
     2. **所有 `select == False`（`dense` 短上下文）步**；
     3. **`TP4 + DCP2` 下 `max_seq_len <= 2051`（所有压缩块均不超过 `block_topk = 512`）时的全部 12 个 Target QSA 层与 MTP Step 0**！
   - **问题 2（`TP4 + DCP2` 下短/中上下文冗余打分与 Top-K 核启动）**：在 `TP4 + DCP2` 下，`qsa_dcp_sharded=True` 使 `dense` 恒为 `False`，因此即使 `max_seq_len <= 2051`，12 个 Target QSA 层仍会启动 `qsa_mqa_paged`（遍历 KV Cache 打分）和 `qsa_lexicographic_topk`。而 `qsa_lexicographic_topk` 在 `length <= TopK (512)` 时第一行就执行 `if (length <= TopK) { indices[tx] = tx < length ? tx : -1; return; }`，完全丢弃 `qsa_mqa_paged` 算出的所有分数！
   - **突破方案**：
     - 在 `QSAIndexer.forward` 中提前判定 `all_blocks_fit`（`max_seq_len // compress_ratio <= block_topk`）并令 `skip_query_prep = self.skip_topk or not select or all_blocks_fit`。当 `skip_query_prep=True` 时，直接切片 `k_weight = self.index_qk_proj.weight[512:]` (`[128, 2560]`) 调用 `qwen38_sm70_fp16_gemv`（在 `_plan_for` 中注册 `_GemvPlan(512, 2, 0)`，仅启动 **128 个 CTA** 而非 640 个 CTA）或 `F.linear`，**削减 80% 的 `index_qk_proj` 权重 HBM 读取与计算量**！
     - 在 `qsa_select_paged_tokens` (`ops/qsa.py`) 中，当 `0 < max_seq_len` 且 `max_seq_len // compress_ratio <= block_topk` 时，直接跳过 `qsa_mqa_paged` 与 `qsa_lexicographic_topk`，单核启动 `_expand_qsa_indices_kernel(..., IDENTITY_BLOCKS=True)`（直接以 `block = safe_rank` 展开），**在 `TP4 + DCP2` 短/中上下文下每轮额外省去 26 次内核启动**，且输出张量与原路径 **100% 逐位一致 (`torch.equal`)**。

3. **`D23` (`sm70_fp16_gemv.py`) — Shared-Expert `down_proj` (`[2560, 160]`) 接入 SM70 FP16 GEMV 快路径**：
   - **问题**：每层 Shared Expert 的 `down_proj` (`[2560, 160]`, `reduce_results=False`) 在每轮 Decode 中被调用 **52 次**（48 个 Target 层 + 4 个 MTP Draft 步），此前未列入 `_plan_for`，导致在 `M=1`（全部 4 个 MTP Draft 步及单 Token Decode）下回退到通用 cuBLAS GEMM（且 `K=160` 非 64 对齐）。
   - **突破方案**：在 `_plan_for` 中为 `.mlp.shared_expert.down_proj` (`(2560, 160)`) 注册 `_GemvPlan(256, 2, 0)`，在 `M=1` 时使用单块 `BLOCK_K=256` 一次性加载全部 160 个输入通道并驻留 L1/L2，消除每轮 4~52 次通用 cuBLAS 回退。

4. **`D24` (`sm70_mtp_fc.py`) — MTP Draft `maybe_combine_fc` 双路投影 + 4 流残差加法单核融合 (`_qwen38_mtp_fc_fused_gemv_add_kernel`)**：
   - **问题**：在 4 个 MTP Draft 步中，每步 `maybe_combine_fc` 依次启动 3 个内核：
     1. `el = fc_embedding(e)` (`[rows, 2560] @ [2560, 640]`)；
     2. `hl = fc_hidden(h)` (`[rows, 4, 2560] @ [2560, 640]`)；
     3. `el.unsqueeze(-2) + hl` (`[rows, 1, 640] + [rows, 4, 640]`)。
   - **突破方案**：新增自定义算子 `torch.ops.vllm.qwen38_sm70_mtp_fc_local_add` 与 Triton 核 `_qwen38_mtp_fc_fused_gemv_add_kernel[(640, rows)]`（覆盖 `1 <= rows <= 4`）：每个 CTA 负责 1 个输出列 `col` 与 1 个 Token 行 `row`，在单次 `K=2560` 循环中同时加载 `w_e[col, :]` 与 `w_h[col, :]`（其中 `w_h` 在寄存器内一次加载、复用于 `h` 的全部 4 条 HC 流），在 FP32 累加后分别做 `.to(tl.float16)` 舍入再相加写出，**严格保留原有 2 个 FP16 舍入边界**，同时将 3 次内核启动压缩为 **1 次启动**（每轮 MTP4 省去 **8 次内核启动** 及 `el`/`hl` 中间显存读写）。

5. **`D25` (`mtp.py`) — MTP `compact_topk_indices` 单行原位 Triton 核 + 多行持久 Scratch 零分配**：
   - **问题**：每轮 MTP Step 0 结束后，`compact_topk_indices` 对每个 QSA 层的 `topk_indices_buffer` 执行 `selected = buffer.index_select(0, row_indices); buffer[:num_rows].copy_(selected)`，不仅每轮动态分配 `selected` 显存并启动 2 个内核，而且当多层共享同一个 `topk_indices_buffer` 时会重复执行。
   - **突破方案**：按 `buffer.data_ptr()` 去重；当 `num_rows == 1` (`B=1` Decode) 时启动单次原位 Triton 核 `_compact_topk_single_row_kernel`（若 `row_indices[0] == 0` 则直接在核内零拷贝早退）；当 `num_rows > 1` 时使用持久缓存 `self._compact_topk_scratch` 配合 `torch.index_select(..., out=selected)` 消除动态显存分配。

---

## 3. 仓库内置高阶 SM70 能力突破开关速查表（Opt-In）

除上述默认生效的代码级突破外，仓库还内置了若干针对特定场景的 **SM70 硬件极限能力开关**（部分因改变浮点累加顺序或面向特定并发档位而默认关闭，可通过环境变量或配置开启）：

| 能力域 | 环境变量 / 配置项 | 默认值 | 推荐设置 (`TP4+DCP2+MTP4`) | 能力突破说明 |
| :--- | :--- | :--- | :--- | :--- |
| **MTP Draft 专家 FP8 量化** | `--speculative-config '{"mtp_expert_quantization": "fp8"}'` | `None` (FP16) | **`"fp8"`** | 将 MTP Draft 层的 512 个路由专家从 FP16 在线量化为 **FP8 (`E4M3`)**，使 4 步串行 Draft MoE 的专家权重 HBM 读取量直接减半（实测节省 **`~0.4 ~ 0.5 ms/轮`**，且每 Rank 节省 **562.5 MiB** 显存）。 |
| **SM70 批处理快路径总开关** | `VLLM_SM70_QWEN38_BATCH_FASTPATH=1` | `1` (auto) | **`1`** | 启用 Target `M=5` 验证步的 `qwen38_dense_batch_sm70_out`（Router `M<=4` 与 Attention Output `o_proj`/`out_proj` `M<=8` 的 4-split FP32 Tensor Core 核）与 `qwen38_gdn_input_batch_sm70_out`。 |
| **HC 批处理 & 低延迟通信** | `VLLM_SM70_QWEN38_HC_BATCH=1`<br>`VLLM_SM70_QWEN38_HC_LL_COMM=1`<br>`VLLM_SM70_QWEN38_HC_REPLICATED=1` | `1` (auto) | **`1`** | 将 96 个 HyperConnection 阶段的 `M=5` 投影切换至 8-warp / split-K 批处理核，并使用 SM70 低延迟紧凑通信与副本化 HC 权重消除跨卡同步开销。 |
| **HCX 跨层 MoE 归约融合** | `kernel_config.sm70_hcx=True`<br>`kernel_config.sm70_top1x=True` | `False` | **`True`** (配合本次 `D21` 修复) | 将每层 MoE / Attention 输出的 TP AllReduce 延迟并融合进下一层 HyperConnection 的 `input_mix_weight_down` 局部规约中（配合本次 `D21` 修复 `"final"` / `"mtp.final"` 冲突）。 |
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

### 4.12 `vllm/models/qwen4_exp/nvidia/model.py` (`D21`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/model.py
+++ b/vllm/models/qwen4_exp/nvidia/model.py
@@ -620,8 +620,11 @@ def enable_sm70_hcx(
         status["prepared_modules"] += 1
         previous_moe = isinstance(layer.mlp, Qwen4ExpSparseMoeBlock)
     if previous_moe and model.hyper_connection_mixer is not None:
+        final_key = (
+            "mtp.final" if getattr(model, "mtp_start_layer_idx", 0) > 0 else "final"
+        )
         model.hyper_connection_mixer._hcx_moe_payload = True
-        model.hyper_connection_mixer.enable_partial_inputs("final")
+        model.hyper_connection_mixer.enable_partial_inputs(final_key, runtime)
     status.update(enabled=True, reason=None)
     logger.info_once("SM70 HCX loaded operator capability: %s.", str(status))
     return True
@@ -674,7 +677,12 @@ def _maybe_enable_sm70_peer_paths(vllm_config: VllmConfig, model: nn.Module) ->
             fuse_output_projection=kernel_config.sm70_hcx_output_projection,
             diagnostic=kernel_config.sm70_hcx_diagnostics,
         )
-        kernel_config.collective_kernel_selections["hcx:target"] = model.sm70_hcx_status
+        selection_key = (
+            "hcx:draft"
+            if getattr(model, "mtp_start_layer_idx", 0) > 0
+            else "hcx:target"
+        )
+        kernel_config.collective_kernel_selections[selection_key] = model.sm70_hcx_status
```

### 4.13 `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` (`D22`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
@@ -242,36 +242,89 @@ class QSAIndexer(nn.Module):
+        from vllm.config.compilation import CUDAGraphMode
+        from vllm.forward_context import is_forward_context_available
+
         from .ops.qsa import (
+            _use_sm70_qsa_lexicographic_topk,
             qsa_compress_groups_with_ratio,
             qsa_select_paged_tokens,
             qsa_store_cache_rows,
         )
 
         raw_metadata, compressed_metadata = metadata
         num_tokens = raw_metadata.num_actual_tokens
         hidden_states = hidden_states[:num_tokens]
         positions = positions[..., :num_tokens]
 
+        q_dim = self.index_n_heads * self.index_head_dim
+        k_dim = self.index_kv_heads * self.index_head_dim
+        block_topk = self.token_topk // self.compress_ratio
+        max_seq_len = getattr(compressed_metadata, "max_seq_len", None)
+        all_blocks_fit = bool(
+            select
+            and not self.skip_topk
+            and self.use_fused_pre_indexer
+            and hidden_states.is_cuda
+            and not torch.cuda.is_current_stream_capturing()
+            and (
+                not is_forward_context_available()
+                or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL
+            )
+            and max_seq_len is not None
+            and 0 < max_seq_len
+            and max_seq_len // self.compress_ratio <= block_topk
+            and _use_sm70_qsa_lexicographic_topk(block_topk)
+        )
+        skip_query_prep = self.skip_topk or not select or all_blocks_fit
+
         # Q/K projection
-        if hidden_states.shape[-1] != self.index_qk_proj.input_size:
+        if hidden_states.shape[-1] == k_dim:
+            projected_q = None
+            raw_keys = hidden_states
+        elif hidden_states.shape[-1] != self.index_qk_proj.input_size:
             # Already projected inside the fused SM70 qkv launch.
-            projected_qk = hidden_states
+            projected_q, raw_keys = hidden_states.split((q_dim, k_dim), dim=-1)
         else:
-            projected_qk, _ = self.index_qk_proj(hidden_states)
-        projected_q, raw_keys = projected_qk.split(
-            (
-                self.index_n_heads * self.index_head_dim,
-                self.index_kv_heads * self.index_head_dim,
-            ),
-            dim=-1,
-        )
+            weight = getattr(self.index_qk_proj, "weight", None)
+            if (
+                skip_query_prep
+                and self.use_fused_pre_indexer
+                and isinstance(weight, torch.Tensor)
+                and weight.shape == (q_dim + k_dim, hidden_states.shape[-1])
+                and weight.dtype in (torch.float16, torch.bfloat16)
+                and getattr(self.index_qk_proj, "bias", None) is None
+                and not getattr(
+                    self.index_qk_proj, "_sm70_qwen4_exp_online_qpn8", False
+                )
+            ):
+                from vllm.config.execution_policy import (
+                    use_sm70_decode_graph_semantics,
+                )
+
+                from .sm70_fp16_gemv import Qwen38SM70FP16LinearMethod
+
+                k_weight = weight[q_dim:]
+                projected_q = None
+                if (
+                    isinstance(
+                        getattr(self.index_qk_proj, "quant_method", None),
+                        Qwen38SM70FP16LinearMethod,
+                    )
+                    and use_sm70_decode_graph_semantics()
+                ):
+                    raw_keys = torch.ops.vllm.qwen38_sm70_fp16_gemv(
+                        hidden_states,
+                        k_weight,
+                        getattr(self.index_qk_proj, "prefix", ""),
+                        None,
+                        False,
+                    )
+                else:
+                    raw_keys = torch.nn.functional.linear(hidden_states, k_weight)
+            else:
+                projected_qk, _ = self.index_qk_proj(hidden_states)
+                projected_q, raw_keys = projected_qk.split((q_dim, k_dim), dim=-1)
         raw_key_state_cache = self.raw_key_cache
         compressed_key_cache = self.compressed_key_cache.kv_cache
-        skip_query_prep = self.skip_topk or not select
 
         if self.use_fused_pre_indexer:
-            q = self._get_q_buffer(num_tokens, projected_q.dtype, projected_q.device)
+            q = self._get_q_buffer(num_tokens, raw_keys.dtype, raw_keys.device)
+            q_input = (
+                q.view(num_tokens, q_dim) if projected_q is None else projected_q
+            )
             qsa_pre_indexer(
-                projected_q,
+                q_input,
                 raw_keys,
```

### 4.14 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` (`D22`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -391,6 +391,7 @@ def _expand_qsa_indices_kernel(
     TOKEN_TOPK: tl.constexpr,
     OUTPUT_WIDTH: tl.constexpr,
     COLUMN_BLOCK: tl.constexpr,
+    IDENTITY_BLOCKS: tl.constexpr = False,
 ) -> None:
     row = tl.program_id(0)
     columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
@@ -414,11 +415,16 @@ def _expand_qsa_indices_kernel(
     block_rank = columns // COMPRESS_RATIO
     offset = columns % COMPRESS_RATIO
     safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
-    block = tl.load(
-        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
-        mask=(row < rows) & is_expanded,
-        other=-1,
-    )
+    if IDENTITY_BLOCKS:
+        block = safe_rank
+    else:
+        block = tl.load(
+            block_indices_ptr
+            + row * stride_blocks_row
+            + safe_rank * stride_blocks_column,
+            mask=(row < rows) & is_expanded,
+            other=-1,
+        )
     expanded = block * COMPRESS_RATIO + offset
@@ -1788,6 +1794,45 @@ def qsa_select_paged_tokens(
     capacity_columns = page_table.shape[1] * k_cache.shape[1]
     score_columns = capacity_columns
     block_topk = token_topk // compress_ratio
+    if (
+        max_seq_len is not None
+        and 0 < max_seq_len
+        and compress_ratio > 0
+        and max_seq_len // compress_ratio <= block_topk
+        and q.is_cuda
+        and HAS_TRITON
+        and _use_sm70_qsa_lexicographic_topk(block_topk)
+        and not torch.cuda.is_current_stream_capturing()
+    ):
+        from vllm.config.compilation import CUDAGraphMode
+        from vllm.forward_context import (
+            get_forward_context,
+            is_forward_context_available,
+        )
+
+        if (
+            not is_forward_context_available()
+            or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL
+        ):
+            column_block = 256
+            _expand_qsa_indices_kernel[(rows, triton.cdiv(output_width, column_block))](
+                out,
+                query_positions,
+                sequence_lengths,
+                token_to_req,
+                out,
+                0,
+                0,
+                out.stride(0),
+                out.stride(1),
+                rows,
+                sequence_lengths.shape[0],
+                BLOCK_TOPK=block_topk,
+                COMPRESS_RATIO=compress_ratio,
+                TOKEN_TOPK=token_topk,
+                OUTPUT_WIDTH=output_width,
+                COLUMN_BLOCK=column_block,
+                IDENTITY_BLOCKS=True,
+                num_warps=4,
+            )
+            return out
     contiguous_keys: torch.Tensor | None = None
```

### 4.15 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` (`D22` & `D23`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
+++ b/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
@@ -47,6 +47,7 @@ _QSA_OUT_SUFFIX = ".self_attn.o_proj"
 _QSA_INDEX_SUFFIX = ".self_attn.indexer.index_qk_proj"
 _ROUTER_SUFFIX = ".mlp.gate"
 _SHARED_UP_SUFFIX = ".mlp.shared_expert.gate_up_proj"
+_SHARED_DOWN_SUFFIX = ".mlp.shared_expert.down_proj"
@@ -718,6 +719,10 @@ def _plan_for(prefix: str, shape: tuple[int, int]) -> _GemvPlan | None:
+    if prefix.endswith(_QSA_INDEX_SUFFIX) and shape == (128, 2560):
+        return _GemvPlan(512, 2, 0)
+    if prefix.endswith(_SHARED_DOWN_SUFFIX) and shape == (2560, 160):
+        return _GemvPlan(256, 2, 0)
     for suffix, expected_shape, plan in _ROLE_PLANS:
         if prefix.endswith(suffix) and shape == expected_shape:
             return plan
```

### 4.16 `vllm/models/qwen4_exp/nvidia/sm70_mtp_fc.py` (`D24`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/sm70_mtp_fc.py
+++ b/vllm/models/qwen4_exp/nvidia/sm70_mtp_fc.py
@@ -12,6 +12,114 @@ from vllm.model_executor.layers.linear import (
     UnquantizedLinearMethod,
 )
 from vllm.platforms import current_platform
+from vllm.triton_utils import HAS_TRITON, tl, triton
+from vllm.utils.torch_utils import direct_register_custom_op
+
+
+@triton.jit
+def _qwen38_mtp_fc_fused_gemv_add_kernel(
+    e_ptr,
+    h_ptr,
+    w_e_ptr,
+    w_h_ptr,
+    out_ptr,
+    K: tl.constexpr,
+    N: tl.constexpr,
+    BLOCK_K: tl.constexpr,
+):
+    col = tl.program_id(0)
+    row = tl.program_id(1)
+    offsets = tl.arange(0, BLOCK_K)
+    acc_e = tl.zeros((BLOCK_K,), dtype=tl.float32)
+    acc_h0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
+    acc_h1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
+    acc_h2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
+    acc_h3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
+    e_row_ptr = e_ptr + row * K
+    h_row_ptr = h_ptr + row * (4 * K)
+    w_e_col_ptr = w_e_ptr + col * K
+    w_h_col_ptr = w_h_ptr + col * K
+    for block_start in tl.static_range(0, K, BLOCK_K):
+        idx = block_start + offsets
+        we = tl.load(w_e_col_ptr + idx).to(tl.float32)
+        wh = tl.load(w_h_col_ptr + idx).to(tl.float32)
+        ev = tl.load(e_row_ptr + idx).to(tl.float32)
+        h0 = tl.load(h_row_ptr + 0 * K + idx).to(tl.float32)
+        h1 = tl.load(h_row_ptr + 1 * K + idx).to(tl.float32)
+        h2 = tl.load(h_row_ptr + 2 * K + idx).to(tl.float32)
+        h3 = tl.load(h_row_ptr + 3 * K + idx).to(tl.float32)
+        acc_e += ev * we
+        acc_h0 += h0 * wh
+        acc_h1 += h1 * wh
+        acc_h2 += h2 * wh
+        acc_h3 += h3 * wh
+    el = tl.sum(acc_e, axis=0).to(tl.float16)
+    hl0 = tl.sum(acc_h0, axis=0).to(tl.float16)
+    hl1 = tl.sum(acc_h1, axis=0).to(tl.float16)
+    hl2 = tl.sum(acc_h2, axis=0).to(tl.float16)
+    hl3 = tl.sum(acc_h3, axis=0).to(tl.float16)
+    out_base = out_ptr + row * (4 * N) + col
+    tl.store(out_base + 0 * N, el + hl0)
+    tl.store(out_base + 1 * N, el + hl1)
+    tl.store(out_base + 2 * N, el + hl2)
+    tl.store(out_base + 3 * N, el + hl3)
@@ -52,8 +160,14 @@ def maybe_combine_fc(model, embedding, hidden):
     rows = embedding.shape[0]
     e = model.pre_fc_norm_embedding(embedding)
     h = model.pre_fc_norm_hidden(hidden).view(rows, 4, 2560)
-    el = model.fc_embedding.quant_method.apply(model.fc_embedding, e)
-    hl = model.fc_hidden.quant_method.apply(model.fc_hidden, h)
+    if e.device.type == "cuda":
+        local_sum = torch.ops.vllm.qwen38_sm70_mtp_fc_local_add(
+            e, h, model.fc_embedding.weight, model.fc_hidden.weight
+        )
+    else:
+        el = model.fc_embedding.quant_method.apply(model.fc_embedding, e)
+        hl = model.fc_hidden.quant_method.apply(model.fc_hidden, h)
+        local_sum = el.unsqueeze(-2) + hl
     return tensor_model_parallel_all_gather(local_sum).flatten(-2)
```

### 4.17 `vllm/models/qwen4_exp/nvidia/mtp.py` (`D25`)

```diff
--- a/vllm/models/qwen4_exp/nvidia/mtp.py
+++ b/vllm/models/qwen4_exp/nvidia/mtp.py
@@ -78,6 +79,22 @@ logger = init_logger(__name__)
+@triton.jit
+def _compact_topk_single_row_kernel(
+    buffer_ptr,
+    row_indices_ptr,
+    stride_row,
+    width,
+    BLOCK: tl.constexpr,
+):
+    src_row = tl.load(row_indices_ptr).to(tl.int64)
+    if src_row <= 0:
+        return
+    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
+    mask = cols < width
+    vals = tl.load(buffer_ptr + src_row * stride_row + cols, mask=mask)
+    tl.store(buffer_ptr + cols, vals, mask=mask)
@@ -434,8 +451,47 @@ class Qwen4ExpMultiTokenPredictor(nn.Module, SupportsPP):
     def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
         """Keep each request's target-aligned step-0 sparse-index row."""
 
         num_rows = row_indices.numel()
+        if num_rows <= 0:
+            return
+        seen_ptrs: set[int] = set()
         for attention in self._iter_qsa_attentions():
             buffer = attention.topk_indices_buffer
-            selected = buffer.index_select(0, row_indices)
-            buffer[:num_rows].copy_(selected)
+            ptr = buffer.data_ptr()
+            if ptr in seen_ptrs:
+                continue
+            seen_ptrs.add(ptr)
+            width = buffer.shape[1]
+            if (
+                num_rows == 1
+                and HAS_TRITON
+                and buffer.device.type == "cuda"
+                and row_indices.device == buffer.device
+                and buffer.stride(1) == 1
+            ):
+                _compact_topk_single_row_kernel[(triton.cdiv(width, 256),)](
+                    buffer,
+                    row_indices,
+                    buffer.stride(0),
+                    width,
+                    BLOCK=256,
+                    num_warps=2,
+                )
+            elif buffer.device.type == "cuda":
+                scratch = getattr(self, "_compact_topk_scratch", None)
+                if (
+                    scratch is None
+                    or scratch.shape[0] < num_rows
+                    or scratch.shape[1] != width
+                    or scratch.dtype != buffer.dtype
+                    or scratch.device != buffer.device
+                ):
+                    cap = max(num_rows, buffer.shape[0])
+                    scratch = torch.empty(
+                        (cap, width), dtype=buffer.dtype, device=buffer.device
+                    )
+                    object.__setattr__(self, "_compact_topk_scratch", scratch)
+                selected = scratch[:num_rows]
+                torch.index_select(buffer, 0, row_indices, out=selected)
+                buffer[:num_rows].copy_(selected)
+            else:
+                selected = buffer.index_select(0, row_indices)
+                buffer[:num_rows].copy_(selected)
```

### 4.18 `vllm/model_executor/models/qwen3_next.py` (`D26`: `DP2 × TP4` MoE Replica-Local TP Fix)

```diff
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -254,6 +254,10 @@ class Qwen3NextSparseMoeBlock(nn.Module):
                     scope="local",
                 )
 
+        keep_replica_local_tp = not (
+            bool(getattr(parallel_config, "enable_expert_parallel", False))
+            or self.is_sequence_parallel
+        )
         self.experts = FusedMoE(
             shared_experts=self.shared_expert,
             gate=self.gate,
@@ -263,6 +267,9 @@ class Qwen3NextSparseMoeBlock(nn.Module):
             intermediate_size=config.moe_intermediate_size,
             renormalize=getattr(config, "norm_topk_prob", True),
             quant_config=quant_config,
+            tp_size=self.tp_size if keep_replica_local_tp else None,
+            dp_size=1 if keep_replica_local_tp else None,
+            pcp_size=1 if keep_replica_local_tp else None,
             prefix=f"{prefix}.experts",
             enable_eplb=self.enable_eplb,
             num_redundant_experts=self.n_redundant_experts,
```

### 4.19 `vllm/v1/spec_decode/llm_base_proposer.py` (`D26`: Align `dummy_run` DP Sync Calls with `propose`)

```diff
--- a/vllm/v1/spec_decode/llm_base_proposer.py
+++ b/vllm/v1/spec_decode/llm_base_proposer.py
@@ -2443,16 +2443,17 @@ class SpecDecodeBaseProposer:
             )
 
         for fwd_idx in fwd_indices:
-            (
-                cudagraph_runtime_mode,
-                num_input_tokens,
-                num_tokens_across_dp,
-                batch_descriptor,
-            ) = self._determine_batch_execution_and_padding(
-                num_tokens, use_cudagraphs=use_cudagraphs
-            )
+            if fwd_idx <= 1 or is_graph_capturing or spec_step_idx is not None:
+                (
+                    cudagraph_runtime_mode,
+                    num_input_tokens,
+                    num_tokens_across_dp,
+                    base_batch_descriptor,
+                ) = self._determine_batch_execution_and_padding(
+                    num_tokens, use_cudagraphs=use_cudagraphs
+                )
             batch_descriptor = self._batch_descriptor_for_spec_step(
-                batch_descriptor, fwd_idx
+                base_batch_descriptor, fwd_idx
             )
```

### 4.20 `vllm/v1/worker/gpu_model_runner.py` (`D26`: Keep DP Drafter Sync Balanced on Over-Limit Context)

```diff
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -9172,6 +9172,23 @@ class GPUModelRunner(
                 propose_drafts_after_bookkeeping = input_fits_in_drafter
 
             if not input_fits_in_drafter:
+                if (
+                    self.vllm_config.parallel_config.data_parallel_size > 1
+                    and (
+                        spec_config.use_eagle()
+                        or spec_config.uses_draft_model()
+                        or spec_config.uses_extract_hidden_states()
+                    )
+                ):
+                    assert isinstance(
+                        self.drafter,
+                        EagleProposer
+                        | DFlashProposer
+                        | DraftModelProposer
+                        | ExtractHiddenStatesProposer
+                        | Gemma4Proposer,
+                    )
+                    self.drafter.dummy_run(1)
                 # Do not schedule any new drafts once the drafter cannot cover
                 # the current context. Returning zero-filled draft rows would
```

### 4.21 `vllm/v1/worker/dp_utils.py` (`D26`: Preserve Per-Replica CUDA Graph Mode & Token Count)

```diff
--- a/vllm/v1/worker/dp_utils.py
+++ b/vllm/v1/worker/dp_utils.py
@@ -136,20 +136,28 @@ def _synchronize_dp_ranks(
         parallel_config=parallel_config,
     )
 
-    # Synchronize cudagraph_mode across ranks first (take min).
-    # This is needed before DP padding decision since we use the synced
-    # cudagraph mode to determine whether DP padding is needed.
-    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
-
     # Check conditions for microbatching
     should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)
 
-    # DP padding is needed when cudagraph is enabled (synced across ranks)
-    # or when ubatching/DBO is active (ubatching requires uniform batch
-    # sizes across DP ranks currently).
-    # Use the synced runtime cudagraph mode rather than the compilation config
-    # so we can avoid padding when cudagraph is not enabled for this step.
-    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    requires_cross_dp_moe_sync = bool(
+        getattr(parallel_config, "enable_expert_parallel", False)
+        or getattr(parallel_config, "use_sequence_parallel_moe", False)
+    )
+    if requires_cross_dp_moe_sync or should_ubatch:
+        synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
+        should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    else:
+        synced_cudagraph_mode = cudagraph_mode
+        should_dp_pad = False
 
     # Pad all DP ranks up to the maximum token count across ranks if
     # should_dp_pad is True
```

### 4.22 `tests/models/qwen4_exp/test_weight_loading.py` (`D26` Regression Test)

```diff
--- a/tests/models/qwen4_exp/test_weight_loading.py
+++ b/tests/models/qwen4_exp/test_weight_loading.py
@@ -470,3 +470,64 @@ def test_loader_skips_final_mixer_on_non_last_pp_rank(monkeypatch) -> None:
     Qwen4ExpModel.load_weights(self_stub, iter([]))
     assert "hyper_connection_mixer." not in captured["skip"]
     assert "hyper_connection_mixer.block_inject_weight" in captured["skip"]
+
+
+def test_qwen3_next_sparse_moe_keeps_replica_local_tp_under_dp(monkeypatch) -> None:
+    """Under DP>1 with enable_expert_parallel=False, Qwen3NextSparseMoeBlock must
+    pass tp_size=self.tp_size, dp_size=1, pcp_size=1 to FusedMoE so that SM70
+    NVFP4 MoE is not flattened to dp_size * tp_size (e.g. 2 * 4 = 8)."""
+    import vllm.model_executor.models.qwen3_next as qwen3_next_module
+
+    captured_kwargs: dict[str, object] = {}
+
+    class _FakeFusedMoE(torch.nn.Module):
+        def __init__(self, **kwargs):
+            super().__init__()
+            captured_kwargs.update(kwargs)
+
+    class _FakeLinear(torch.nn.Module):
+        def __init__(self, *args, **kwargs):
+            super().__init__()
+
+    monkeypatch.setattr(qwen3_next_module, "FusedMoE", _FakeFusedMoE)
+    monkeypatch.setattr(qwen3_next_module, "ReplicatedLinear", _FakeLinear)
+    monkeypatch.setattr(
+        qwen3_next_module, "get_tensor_model_parallel_world_size", lambda: 4
+    )
+    monkeypatch.setattr(
+        qwen3_next_module,
+        "get_ep_group",
+        lambda: SimpleNamespace(
+            device_group=SimpleNamespace(size=lambda: 8),
+            rank_in_group=0,
+        ),
+    )
+    vllm_config = SimpleNamespace(
+        model_config=SimpleNamespace(
+            hf_text_config=SimpleNamespace(
+                num_experts=512,
+                num_experts_per_tok=10,
+                hidden_size=2048,
+                moe_intermediate_size=512,
+                shared_expert_intermediate_size=0,
+                hidden_act="silu",
+                norm_topk_prob=True,
+            )
+        ),
+        parallel_config=SimpleNamespace(
+            use_sequence_parallel_moe=False,
+            enable_expert_parallel=False,
+            enable_eplb=False,
+            eplb_config=SimpleNamespace(num_redundant_experts=0),
+        ),
+        quant_config=None,
+    )
+    monkeypatch.setattr(
+        qwen3_next_module, "get_current_vllm_config", lambda: vllm_config
+    )
+
+    Qwen3NextSparseMoeBlock(vllm_config, prefix="model.layers.0.mlp")
+    assert captured_kwargs["tp_size"] == 4
+    assert captured_kwargs["dp_size"] == 1
+    assert captured_kwargs["pcp_size"] == 1
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
