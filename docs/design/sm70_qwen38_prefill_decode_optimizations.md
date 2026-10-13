# Qwen3.8-Flash-Next (`qwen4_exp`) SM70 Prefill & Decode Optimization Guide

- **Branch**: `arena/5368f936-1cat-vllm`
- **Base Commit**: `be3722bc095b9792bf9833068595f0b1ef70ad37` (`#1172`)
- **Target Hardware**: NVIDIA Tesla V100-SXM2-32GB (`SM70`, 900 GB/s HBM2, 125 TFLOPS FP16 Tensor Core)
- **Target Model**: `Qwen3.8-Flash-Next-NVFP4` (`Qwen4ExpForCausalLM`, `qwen4_exp`)

---

## 1. Model Architecture & Memory Footprint (`TP4` on 4×V100-32GB)

| Component | Configuration | Notes |
| :--- | :--- | :--- |
| **Backbone** | 48 layers (36 GDN + 12 QSA), `hidden_size = 2560` | 3:1 `linear_attention` : `full_attention` pattern |
| **HyperConnection (mHC)** | `hc_count = 4`, `hc_lowrank = 320` | 4-stream residual state (`4 × 2560 = 10240`) per token |
| **QSA (12 layers)** | 24 Q heads, 2 KV heads (`head_dim = 256`, `rope_dim = 64`) | Indexer: `4` heads, `dim = 128`, `compress_ratio = 4`, `budget = 2048` |
| **GDN (36 layers)** | 16 Q/K heads, 48 V heads (`head_dim = 128`), `conv_kernel = 4` | TP4 local: 4 Q/K heads (`512`), 12 V heads (`1536`) |
| **MoE (48 layers)** | 512 routed experts, `top_k = 10`, `moe_intermediate_size = 640` | + 1 shared expert (`640`) with sigmoid gate per layer |
| **PLE (Per-Layer Embeddings)** | 12 QSA layers, `ple_vocab_size = 250880`, `ple_dim = 640`, `ngram = 3` | ~7.18 GiB total (~1.79 GiB/rank), offloaded to pinned host RAM / disk |
| **MTP Draft** | 1 full `GatedDeltaNet + MoE` layer (`num_speculative_tokens = 4`) | FP8 E4M3 routed experts (`mtp_fp8_experts.py`) |

### Per-GPU VRAM Breakdown (`TP4`, `gpu_memory_utilization = 0.92`)

| Memory Consumer | Size per GPU (`TP4`) |
| :--- | :--- |
| **Target + MTP4 Model Weights** (NVFP4 routed experts + FP16 dense/QSA/GDN/shared + FP8 MTP) | **~11.75 GiB** |
| **SM70 Pre-packed Weight Buffers** (GDN input, HC target/draft, router, shared expert) | **~0.83 GiB** |
| **PyTorch / Inductor / CUDA Graph Workspaces & Pinned PLE Staging** | **~2.70 GiB** |
| **Available KV + GDN/QSA State Cache Pool** | **~14.22 GiB** (~359K tokens FP16 KV / ~590K tokens FP8 E4M3 KV) |

---

## 2. Measured Baseline vs. Optimization Ceiling (4×V100 `TP4`)

### 2.1 End-to-End Throughput Summary

| Scenario | Context Length | Current Baseline (4×V100 `TP4`) | Post-Optimization Ceiling (4×V100 `TP4`) | Expected Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill (No Prefix Cache, Warm PLE)** | **8K** | **6,950 – 7,056 tok/s** (`1.16 s`) | **7,800 – 8,300 tok/s** (`0.99 – 1.05 s`) | **+12% ~ +18%** |
| **Prefill (No Prefix Cache, Warm PLE)** | **64K** | **6,246 – 6,327 tok/s** (`10.4 s`) | **7,000 – 7,450 tok/s** (`8.8 – 9.3 s`) | **+12% ~ +18%** |
| **Prefill (No Prefix Cache, Warm PLE)** | **128K** | **5,830 – 5,927 tok/s** (`22.1 s`) | **6,500 – 6,950 tok/s** (`18.8 – 20.1 s`) | **+11% ~ +17%** |
| **Prefill (No Prefix Cache, Warm PLE)** | **256K** | **5,170 – 5,249 tok/s** (`50.0 s`) | **5,700 – 6,100 tok/s** (`42.9 – 46.0 s`) | **+10% ~ +16%** |
| **Prefill (Sparse Prefix Cache)** | **8K / 128K** | **5,356 / 4,666 tok/s** | **6,400 / 5,600 tok/s** | **+19% ~ +20%** |
| **Decode (`C=1`, No-MTP)** | **8K** | **97.7 – 99.0 tok/s** (`10.10 ms/tok`) | **112 – 120 tok/s** (`8.3 – 8.9 ms/tok`) | **+15% ~ +21%** |
| **Decode (`C=1`, No-MTP)** | **256K** | **74.0 tok/s** (`13.52 ms/tok`) | **85 – 92 tok/s** (`10.9 – 11.8 ms/tok`) | **+15% ~ +24%** |
| **Decode (`C=1`, MTP4)** | **8K** | **265 – 275 tok/s** (`18.38 ms/round`) | **325 – 355 tok/s** (`13.8 – 15.0 ms/round`) | **+22% ~ +30%** |
| **Batched Decode (`C=16`, No-MTP)** | **8K (FP16 KV)** | **441.5 tok/s** | **600 – 680 tok/s** | **+36% ~ +54%** |
| **Batched Decode (`C=16`, No-MTP)** | **1K (E4M3 KV)** | **772.6 tok/s** | **950 – 1,080 tok/s** | **+23% ~ +40%** |

### 2.2 Decode `C=1` with MTP4 Latency Breakdown (`18.38 ms/round` @ 8K)

| Stage | Current Latency (`ms/round`) | Share | Primary Bottleneck | Optimized Target (`ms/round`) |
| :--- | :--- | :--- | :--- | :--- |
| **Target Verify (`M = 5`, 48 layers)** | **`14.38 ms`** | **78.2%** | NVFP4 MoE (`3.40 ms`), TP4 AllReduce (`2.55 ms` / 96 calls), HyperConnection (`1.82 ms` / 96 calls), GDN (`2.35 ms`), QSA (`1.58 ms`), Shared Expert + Router (`1.38 ms`), PLE + `lm_head` (`1.30 ms`) | **`11.0 – 11.8 ms`** |
| **MTP Draft (`4 × M = 1` steps)** | **`3.32 ms`** | **18.1%** | 4 steps × `0.83 ms/step`: Draft `lm_head` FP16 GEMV (`1.44 ms` = `4 × 0.36 ms`, reading `318 MB/step`), Draft MoE + GDN + HC + AllReduce (`1.88 ms`) | **`2.2 – 2.5 ms`** |
| **Rejection Sampler & Host Bookkeeping** | **`0.68 ms`** | **3.7%** | Speculative metadata preparation & greedy verification | **`0.5 – 0.6 ms`** |
| **Total per MTP4 Round (`~4.886 tok`)** | **`18.38 ms`** | **100%** | **265.8 tok/s** | **`13.8 – 15.0 ms` (325 – 355 tok/s)** |

---

## 3. Phase-1 Implemented Optimizations (100% Bitwise-Exact, Active on Current Branch)

All Phase-1 optimizations are **strictly bit-exact** (preserving identical floating-point rounding and reduction order so `benchmarks/benchmark_sm70_qwen38_baseline.py` greedy token ID verification passes without modification) and require **no C++ `_C.abi3.so` rebuild**.

### 3.1 QSA Indexer Prefill cuBLAS Visibility Cache (Eliminates 11/12 Host Syncs per Chunk)

- **Files Modified**:
  - `vllm/models/qwen4_exp/common/qsa_cache.py`
  - `vllm/models/qwen4_exp/nvidia/indexer_qsa.py`
  - `vllm/models/qwen4_exp/nvidia/ops/qsa.py`
- **Problem**:
  - `QSAMetadataBuilder.build()` constructs a single shared `QSAForwardMetadata` instance for all 12 QSA layers in a forward step, with identical `logical_positions`, `token_to_req`, and `seq_lens`.
  - In `qsa_select_paged_tokens` (`ops/qsa.py`), when `_use_sm70_qsa_indexer_cublas` is taken during Prefill, every QSA layer launched `_qsa_visible_blocks(...)`, ran `all_visible.max()`, and called `.item()` — forcing **12 host-device pipeline synchronizations**, 12 `_qsa_visible_blocks_kernel` launches, and 12 `max()` reductions per prefill chunk.
- **Solution**:
  - Added `visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]]` to `QSAForwardMetadata`, keyed by `(start_row, end_row, int(compress_ratio))`.
  - The first QSA layer computes `(all_visible, max_visible)` once and caches it on `compressed_metadata.visible_blocks_cache`; the remaining 11 QSA layers reuse both the device tensor `all_visible` and host int `max_visible` with zero kernel launches and zero host syncs.
- **Savings**: Eliminates **11/12 `.item()` CPU–GPU pipeline stalls**, **11/12 `_qsa_visible_blocks_kernel` launches**, and **11/12 `max()` reductions** per prefill chunk.

### 3.2 QSA Output Gate 3D Strided View Passthrough (Eliminates Contiguous Copy in Decode & Prefill)

- **File Modified**:
  - `vllm/models/qwen4_exp/nvidia/qsa.py`
- **Problem**:
  - In `Qwen4ExpQSAAttention.forward`, both the decode fastpath (`self._sm70_qsa_prep and num_tokens <= 32`) and the prefill/fallback path (`_project_qkv_gate`) reshaped the strided slice `gate = q_gate[..., self.head_dim :]` from `(num_tokens, num_heads, head_dim)` (strides `(2 * num_heads * head_dim, 2 * head_dim, 1)`) to 2D `(num_tokens, q_size)`.
  - Because `q_gate` interleaves `query` and `gate` per head, flattening `num_heads × head_dim` across non-contiguous dim 1 forced PyTorch to allocate a new tensor and launch an `aten::copy_` kernel on **every QSA layer**, only for `_run_qsa` to immediately reshape `output_gate.view_as(query)` back to 3D `(num_tokens, num_heads, head_dim)`.
  - All downstream QSA attention consumers (`qsa_dense_decode_sm70_out`, `vllm_sm70_qsa_device.run`, `_qsa_output_gate_kernel`, `_qsa_sparse_paged_gqa_splitk_kernel`, `host_qsa_attention`, `device_history_attention`) already accept a 3D strided `gate` (`gate.sizes() == q.sizes() && gate.stride(2) == 1`) and index via `gate.stride(0)` and `gate.stride(1)`.
- **Solution**:
  - Added keyword parameter `flatten_gate: bool = True` to `_project_qkv_gate` (preserving backwards compatibility for direct unit-test callers in `tests/models/qwen4_exp/test_config.py`).
  - In `Qwen4ExpQSAAttention.forward`, pass the 3D strided view `gate = q_gate[..., self.head_dim :]` (and `flatten_gate=False` in `_project_qkv_gate`) directly into `qwen4_exp_qsa_with_output` / `_run_qsa`.
- **Savings**:
  - **Decode / MTP4 Verify**: Eliminates **16 `aten::copy_` kernel launches per MTP4 round** (12 Target QSA + 4 Draft QSA steps) and **12 launches per No-MTP step**.
  - **8K Prefill**: Eliminates **`302 MB` of HBM read+write copy traffic** (`[8192, 6, 256]` FP16 × 12 layers) and 12 kernel launches per 8K prefill chunk.

### 3.3 NVFP4 MoE Prefill Workspace & `moe_permute` / `moe_unpermute` Dispatch Cleanup

- **Files Modified**:
  - `vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py`
  - `vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py`
- **Problem**:
  1. In `NvFp4MoEWorkspace._eager_buffers` (used for `num_tokens > 18` across all 48 MoE layers during prefill), every layer launched two GPU `torch.arange` kernels (`token_expert_indices` of size `slots` and `compact_offsets` of size `slots + 1`) even though both tensors are read-only `[0, 1, ..., N]` index sequences.
  2. In `ModelOptNvFp4SM70MoEMethod._apply_routed_experts_sm70`, the general `moe_permute` / `moe_unpermute` path unconditionally executed:
     - `output.zero_()` (`[8192, 2560]` FP16 = `41.94 MB` per layer × 48 layers = **`2.01 GB` HBM zero-fill per 8K prefill chunk**), even though `finalizeMoeRoutingKernel` (`moe_unpermute` with `CHECK_SKIPPED=false` when `layer.expert_map is None`) initializes `thread_output.fill(0)` in registers and overwrites 100% of `output`.
     - `buffers["permuted_idx"].fill_(slots)` (`15.7 MB` per 8K chunk + 48 kernel launches), even though `get_source_row_id` (`CHECK_SKIPPED=false` when `layer.expert_map is None`) overwrites all `0..slots-1` entries.
     - `topk_ids_i32.copy_(topk_ids)` (`15.7 MB` per 8K chunk + 48 kernel launches), even when `topk_ids` is already a contiguous `int32` tensor (`moe_permute` takes `const torch::Tensor& topk_ids` and never mutates it).
- **Solution**:
  - Cached a per-device read-only `int32` sequence buffer (`NvFp4MoEWorkspace._get_eager_arange(device, slots + 1)`) and sliced `arange_buf[:slots].view(num_tokens, top_k)` and `arange_buf[: slots + 1]` in `_eager_buffers`.
  - Guarded `output.zero_()` and `buffers["permuted_idx"].fill_(slots)` behind `if layer.expert_map is not None:`, and bypassed `topk_ids_i32.copy_(topk_ids)` when `topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()`.
- **Savings**:
  - Eliminates **`2.04 GB` of HBM traffic** and **240 kernel launches** (`96 arange + 48 zero_ + 48 fill_ + 48 copy_`) per 8K prefill chunk.

### 3.4 GDN Input Projection Tail-Split Grid Shrink & Prefill Admission

- **Files Modified**:
  - `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py`
  - `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- **Problem**:
  1. In `_split_gdn_projection_tails` (`COPY_QKV=False`), `_qwen38_gdn_projection_split_kernel` was launched with grid `(m, triton.cdiv(4096, 256)) = (m, 16)` and executed `tl.load(qkvz + ...)` across all 4096 columns even though `COPY_QKV=False` only writes the `Z = 1536` tail (`cols 2560..4095`) and `B + A = 24` gate columns on `program_id(1) == 0`. As a result, 9 out of 16 CTAs per row (`program_id(1) in 1..9`) were pure no-ops that loaded `qkvz[:, 256:2560]` from HBM and discarded it.
  2. In `qwen_gdn_input_projection_core` (the opaque custom op outside Inductor used when `mtp == 4`), `copied_tails` restricted `_split_gdn_projection_tails` to `mixed_qkvz.shape[0] in (5, 20)`. Consequently, during Prefill (`M = 8192`), `qwen_gdn_input_projection_core` fell back to 3 separate eager `aten::copy_` launches per GDN layer (`b.contiguous()`, `a.contiguous()`, `z_out.copy_(z)` = 108 launches per prefill chunk).
- **Solution**:
  - In `_qwen38_gdn_projection_split_kernel`, specialized the `else` (`COPY_QKV=False`) branch to index `z_col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)` starting at offset `QKV = 2560`, and reduced the launch grid in `_split_gdn_projection_tails` from `(m, 16)` to `(m, triton.cdiv(1536, 256)) = (m, 6)` (**62.5% fewer CTAs**, zero wasted `qkv` HBM loads).
  - In `qwen_gdn_input_projection_core`, relaxed `mixed_qkvz.shape[0] in (5, 20)` to `mixed_qkvz.shape[0] > 1` (bounded by `_can_fuse_gdn_projection_split`).
- **Savings**:
  - Cuts CTA count and HBM loads of `_split_gdn_projection_tails` by **62.5%** during MTP4 Verify (`M = 5, 20`) and replaces **108 separate eager `aten::copy_` launches** per 8K prefill chunk with 36 fused launches.

---

## 4. Phase-2 Kernel Optimization Roadmap (Requiring CUDA/C++ Extension Rebuild)

To reach the full theoretical ceiling on 4×V100 (`7,800–8,300 tok/s` 8K prefill; `325–355 tok/s` MTP4 decode), the following C++/CUDA kernel fusions are recommended for Phase 2:

1. **Prefill MoE W13 Fused SwiGLU for `M > 128` (`csrc/sm70_turbomind/ops/nvfp4_sm70_gemm.cu`, `fp4_stages.py`)**:
   - Currently `fused_indexed_w13` (`nvfp4_moe_indexed_w13_fused_swiglu_sm70_out`) is gated by `num_tokens <= 128` (`indexed_w13_max_tokens = 128`).
   - Extending the indexed/grouped W13 HMMA epilogue to fuse `SwiGLU` in registers for `M > 128` eliminates writing `[81920, 320]` (`52.43 MB/layer`) and reading `[81920, 320]` + writing `[81920, 160]` (`26.21 MB/layer`) in `silu_and_mul`, saving **`3.77 GB` of HBM traffic per 8K prefill chunk** (~`4.2 ms`/chunk).
2. **Sharded HyperConnection + TP4 AllReduce Fusion (`csrc/sm70_turbomind/ops/`, `sm70_hcx.py`)**:
   - Currently each layer runs 2× TP4 AllReduce (`2.55 ms/round`) and 2× 4-stream HyperConnection mixing (`1.82 ms/round`).
   - Sharding the 4-stream residual state `[M, 4, 2560]` across the 4 TP ranks (`[M, 4, 640]` per rank) andreplacing `AllReduce(2560)` + full `hc_down(10240 -> 320)` with local `ReduceScatter(640)` + sharded `hc_down(2560 -> 320)` + `AllGather` cuts HC + collective HBM & NVLink traffic by ~60% (saving **`~1.2–1.5 ms/round`** in MTP4 decode).
3. **MTP Draft `lm_head` Quantization (FP8 E4M3 or W4A16) + Top-1 Fusion (`vllm/models/qwen4_exp/nvidia/mtp.py`)**:
   - The MTP Draft model executes 4 sequential steps per round, reading its FP16 `lm_head` (`[62080, 2560]` per rank = `317.8 MB`) 4 times (`1.27 GB/round`, `~1.44 ms/round`).
   - Quantizing the Draft `lm_head` to FP8 E4M3 (matching the Draft's FP8 MoE experts) or W4A16 cuts Draft `lm_head` bandwidth from `1.27 GB/round` to `0.32–0.64 GB/round` (**saving `~0.7–1.0 ms/round`**) with zero impact on Target verification exactness (since Target verification always uses the exact FP16 Target `lm_head`).
4. **Fused Router (`512`) + Shared-Expert Input (`320` + `1` gate) Projection (`sm70_fp16_gemv.py`)**:
   - Both the MoE router (`[512, 2560]`) and shared expert (`[320, 2560]` + `[1, 2560]`) read the exact same normalized `hidden_states` (`[M, 2560]`). Fusing them into a single `[833, 2560]` packed projection + shared-expert `SiLU` epilogue eliminates 96 kernel launches per MTP4 round (**saving `~0.6–0.8 ms/round`**).

---

## 5. `TP4 + DCP2` Decode Path Optimization Analysis & Implemented Changes

Under **`TP4 + DCP2`** (`--tensor-parallel-size 4 --decode-context-parallel-size 2`), the 4 TP ranks are split into two DCP groups of size 2 (`{0, 1}` and `{2, 3}`), with `dcp_comm_backend="a2a"` by default (`vllm/model_executor/models/config.py:772-779`).
- **Sharded Target QSA Main KV Cache (`qsa_dcp_sharded = True`)**: Only the 12 target QSA main KV caches are sharded across DCP (`local_block_size = 200` for scheduler `block_size = 400`, `cp_kv_cache_interleave_size = 1`), packed 2 layers per physical page alongside **6 GDN state groups** of 6 layers each (vs. 3 GDN groups at `DCP1`).
- **Replicated Selector & Draft Caches (`dcp_sharded = False`)**: All 13 QSA selectors (raw circular buffer + compressed key cache) and the MTP draft QSA layer (`layers.48`) stay replicated across DCP so top-$k$ block selection and MTP draft generation remain communication-free.
- **Per-Layer Target QSA Decode Execution (`_forward_qsa_dcp`)**:
  1. `qsa_localize_dcp_indices` compacts the rank-owned subset of the `[B, 2051]` global token selection into local page coordinates.
  2. `group.all_gather(query.contiguous(), dim=1)` gathers Q heads from `[B, 6, 256]` to `[B, 12, 256]` across the 2-rank DCP group.
  3. `qsa_sparse_paged_attention` runs local sparse GQA (`12` Q heads, `1` KV head, `group_size=12`, `local_width=1026`) and emits FP32 `partial_output` (`[B, 12, 256]`) and base-2 `partial_lse` (`[B, 12]`).
  4. `dcp_a2a_lse_reduce` packs `partial_output` + `partial_lse`, exchanges them via `dist.all_to_all_single`, and combines the 2 ranks' partial outputs weighted by base-2 LSE back to `[B, 6, 256]`, followed by FP16 rounding and sigmoid output gating.

### 5.1 Bit-Exact `TP4 + DCP2` Decode Optimizations Implemented

| # | Subsystem | File(s) Modified | Bottleneck Identified on `TP4 + DCP2` Decode | Optimization Implemented | Bit-Exact? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | **QSA DCP Combine + Round + Sigmoid Gate Fusion** | `vllm/v1/attention/ops/dcp_alltoall.py`, `vllm/models/qwen4_exp/nvidia/qsa.py` | After `all_to_all_single`, `_dcp_a2a_unpack_combine` allocated an intermediate FP32 `out` (`[B, 6, 256]`) and a dummy `(1, 1)` `out_lse` tensor on the GPU, then `_forward_qsa_dcp` launched `output.copy_(merged)` (`FP32 -> FP16` cast kernel) followed by `_qsa_output_gate` (a separate Triton sigmoid gate kernel). | Extended `dcp_a2a_lse_reduce` / `_dcp_a2a_unpack_combine_kernel` with optional `out` and `output_gate` parameters (`HAS_OUTPUT_GATE: tl.constexpr`) that round `acc` to `out.dtype` (`FP16`) in registers, multiply by `tl.sigmoid(gate)` in `FP32`, and store directly to `output`, while eliminating the dummy `(1, 1)` `out_lse` allocation when `return_lse=False`. Saves **24 kernel launches + 24 tensor allocations + 36 HBM round-trips per decode step**. | **Yes (100% bitwise identical)** |
| **D2** | **Half-Width Output Write in `qsa_localize_dcp_indices`** | `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py`, `vllm/models/qwen4_exp/nvidia/qsa.py` | `_forward_qsa_dcp` called `qsa_localize_dcp_indices` with the full `[B, 2051]` `local_indices_buffer` before computing `local_width = 1026`, writing `-1` padding across all `2051` columns even though only `:1026` can hold owned tokens or are read by `qsa_sparse_paged_attention`. | Compute `local_width` first and pass `local_indices_buffer[:B, :local_width]` (`OUT_WIDTH = 1026`) into `qsa_localize_dcp_indices`, cutting padding stores in half (`1026` vs. `2051` columns per query row). | **Yes (100% bitwise identical)** |
| **D3** | **Persistent Split-K Scratch Workspace in `qsa_sparse_paged_attention`** | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | Under `TP4 + DCP2`, `_forward_qsa_dcp` passes `lse=partial_lse` (`group_size=12`, `page_size=200`), so `qsa_sparse_paged_attention` always takes the Triton split-K path and dynamically allocated `partial_output` (`[num_splits, B, 12, 256]` FP32) and `partial_lse` (`[num_splits, B, 12]` FP32) on every QSA layer (`24` allocations/step). | Added `_qsa_splitk_workspace(q, num_splits)` backed by `workspace_cache("qsa_splitk", _SM70_QSA_SPLITK_WORKSPACES)` and `retain_for_capture`, reusing per-stream persistent FP32 scratch buffers across all 12 QSA layers and eliminating **24 dynamic GPU allocations per decode step**. | **Yes (100% bitwise identical)** |
| **D4** | **Skip Full-Buffer `output.zero_()` on DCP QSA Decode** | `vllm/models/qwen4_exp/nvidia/qsa.py` | `forward_qsa` called `output.zero_()` on the entire buffer for all non-host-KV layers, even though `_forward_qsa_dcp` (`qsa_dcp_sharded=True`) unconditionally overwrites every active row `output[:num_tokens]` (including empty selections). | Extended the padding-only zero guard to `qsa_dcp_sharded`, only zeroing `output[num_tokens:]` when `num_tokens < output.shape[0]` (eliminating **12 `zero_()` kernel launches per decode step** when unpadded). | **Yes (100% bitwise identical)** |
| **D5** | **Vectorized Replicated-Draft Block-Table Expansion in `DCP2`** | `vllm/models/qwen4_exp/nvidia/qsa.py` | `Qwen4ExpQSAMetadataBuilder.build` expanded the replicated MTP draft layer's block table via a Python `for sub_block in range(expansion):` loop, launching `2 * expansion` small strided elementwise kernels per step. | Replaced the Python loop with a single broadcasted `torch.mul(table.unsqueeze(-1), expansion, out=expanded_view)` + `expanded_view.add_(sub_block_offsets)` over a cached `(1, 1, expansion)` offset tensor. | **Yes (100% bitwise identical)** |
| **D6** | **Lazy `compute_num_computed_tokens()` across 6 GDN Groups on Decode** | `vllm/v1/attention/backends/gdn_attn.py` | Because `TP4 + DCP2` packs 2 QSA layers per physical page, the 36 GDN layers are partitioned into **6 GDN KV-cache groups** (vs. 3 at `DCP1`). `GDNAttentionMetadataBuilder.build` unconditionally called `m.compute_num_computed_tokens()` (`torch.diff(query_start_loc)` + `seq_lens - query_len`), even though `context_lens_tensor` is only read inside `if num_prefills > 0:`. | Moved `context_lens_tensor = m.compute_num_computed_tokens()` inside `if num_prefills > 0:`, eliminating **12 wasted GPU kernel launches and 12 tensor allocations per decode step** (and up to 60 in uncached MTP4 fallback builds). | **Yes (100% bitwise identical)** |
| **D7** | **Persistent QSA Selector, Indexer & Prep Workspaces in Opaque Custom Op** | `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `ops/qsa_shared_key.py`, `indexer_qsa.py`, `qsa.py` | Because `qwen4_exp_qsa_with_output` is an opaque custom op outside Inductor's memory pool, every QSA layer (`12` target + `1..4` draft) dynamically allocated `scores`/`logits`, `visible_blocks`, `blocks_buffer`, `indexer.q`, and `prepared` query tensors, plus repeated `kv_cache.unbind(1)` and `.to(torch.int64)` calls. | Added `_qsa_selector_workspace`, persistent `_q_buffer` in `QSAIndexer`, persistent `_prepared_q_buffer` + cached `kv_cache.unbind(1)` + dtype-guarded `.to(torch.int64)` in `Qwen4ExpQSAAttention`, eliminating **~65 dynamic GPU allocations per decode step**. | **Yes (100% bitwise identical)** |
| **D8** | **Zero-Allocation / Zero-Copy DCP Query `all_gather`** | `vllm/models/qwen4_exp/nvidia/qsa.py` | `group.all_gather(query.contiguous(), dim=1)` went through `torch.ops.vllm.all_gather` custom-op boxing, dynamically allocated `output_tensor` on every layer, and for `B > 1` (e.g. `B = 5` in MTP4 verify) `.movedim(0, 1).reshape(...)` allocated a second tensor and launched an `aten::copy_` transpose kernel. | Added `_all_gather_dcp_query` backed by persistent `_dcp_ag_recv_buffer` and `_dcp_ag_out_buffer` calling `dist.all_gather_into_tensor` directly: zero-copy view at `B == 1`, zero dynamic allocations at `B > 1`. | **Yes (100% bitwise identical)** |
| **D9** | **Dynamic Live-Context `score_columns` Bounding on Decode** | `vllm/models/qwen4_exp/common/qsa_cache.py`, `nvidia/indexer_qsa.py`, `nvidia/ops/qsa.py` | On decode (`not _use_sm70_qsa_indexer_cublas`), `qsa_select_paged_tokens` set `score_columns = capacity_columns` (the full `max_model_len` table width, e.g. `65,600` columns at 256K), launching `2,050` CTAs per QSA layer in `mqa_multiquery` / `_qsa_mqa_paged_kernel` even when active `seq_len` was only 4K–8K (`32–64` active CTAs). | Propagated `max_seq_len` on `QSAForwardMetadata` and added `_qsa_bounded_decode_score_columns` to bound `score_columns` to `min(capacity_columns, max(block_topk, ceil(bound_seq_len // compress_ratio, 64)))` in eager/piecewise and context-bucketed FULL graph modes, cutting empty scorer CTAs by **80%–98%** on short/medium context decode. | **Yes (100% bitwise identical)** |
| **D10** | **Fused Physical Page Resolution in `qsa_localize_dcp_indices`** | `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py`, `nvidia/ops/qsa.py`, `nvidia/qsa.py` | `_qsa_localize_dcp_indices_kernel` wrote rank-local logical token IDs, requiring all 64 split CTAs of `_qsa_sparse_paged_gqa_splitk_kernel` to load `block_table` from global memory on every tile. | Added `RESOLVE_PHYSICAL: tl.constexpr` to `_localize_indices` so physical KV slots (`physical_page * PAGE_SIZE + page_offset`) are resolved in registers during DCP index localization and consumed with `RESOLVED_INDICES=True` in `qsa_sparse_paged_attention` with zero extra kernel launches. | **Yes (100% bitwise identical)** |
| **D11** | **Empty-Tile Short-Circuit + Two-Warp `G=12` Launch in Sparse Attention** | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | In `_qsa_sparse_paged_gqa_splitk_kernel`, compacted DCP2 selections leave trailing tiles filled with `-1`, yet empty tiles still executed masked `k_cache`/`v_cache` loads and 2× Tensor Core `tl.dot` MMAs; additionally `_use_sm70_qsa_two_warp_partial` only admitted `group_size == 6` (`DCP1`) and excluded `group_size == 12` (`DCP2`). | Guarded the tile loop body with `if tl.max(logical_token, axis=0) >= 0:` to skip KV loads and both MMAs on empty tiles, and extended `_use_sm70_qsa_two_warp_partial` to `group_size in (6, 12)` so `TP4 + DCP2` uses the 2-warp SM70 partial launch. | **Yes (100% bitwise identical)** |
| **D12** | **Skip MLA-Only `dcp_local_seq_lens` Preparation on `Qwen4Exp`** | `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/cudagraph_utils.py` | When `dcp_world_size > 1`, `ModelRunnerV1` computed and H2D-copied `dcp_local_seq_lens` (and `SpecDecodeBaseProposer` cloned it on GPU) and `ModelRunnerV2` launched `_dcp_local_seq_lens_kernel` on every step, even though only `MLAAttentionMetadataBuilder` reads `dcp_local_seq_lens`. | Gated `dcp_local_seq_lens` preparation on `is_deepseek_mla` / `ag.backend.is_mla()`, eliminating wasted CPU work, H2D copies, Triton kernel launches, and MTP `.clone()` calls on `TP4 + DCP2`. | **Yes (100% bitwise identical)** |

---

## 6. 8×V100 Deployment & `TP8` Feasibility Analysis

### 6.1 Why Pure `TP8` Fails on Startup Today

1. **`ModelOptNvFp4SM70MoEMethod` Hardcoded TP Gate (`vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py:594-624`)**:
   - `_SUPPORTED_TP_SIZES = (1, 2, 4)` explicitly raises `NotImplementedError` when `tp_size == 8`.
2. **MoE `intermediate_size = 640` Sharding Alignment**:
   - At `TP8`, `local_intermediate_size = 640 / 8 = 80` (`gate_up = 160`).
   - In `nvfp4_sm70_moe.py`, `80 % 16 == 0` satisfies the 16-column tile alignment, so removing the `tp_size == 8` gate allows NVFP4 MoE to run at `TP8`.
   - However, in `gguf_turbomind_moe.py:619,712` (`80 % 64 != 0` and `80 % 32 != 0`) and `gguf_moe.py:106` (`160 % 256 != 0`), `TP8` violates the kernel tile alignment and cannot run without weight padding.
3. **QSA KV Heads (`num_key_value_heads = 2`) & TP4-Specialized Fastpaths**:
   - At `TP8`, KV heads must be replicated (`4` ranks per KV head), and Q heads drop to `24 / 8 = 3` per rank.
   - All SM70 TP4-specialized fastpaths (`flash_decode_paged.cu` XQA `H=6, D=256`, `sm70_fp16_gemv.py` GDN `H_v=12`, `sm70_ring.py` / `sm70_hc_ll.py` 4-rank collectives) fall back to generic Triton/NCCL paths, and 8-GPU ring AllReduce latency (`~4.5 ms/round`) exceeds the compute savings on V100.

### 6.2 Recommended 8×V100 Topologies

| 8×V100 Topology | Configuration | Best Use Case | Performance vs. 4×V100 `TP4` |
| :--- | :--- | :--- | :--- |
| **`TP4 + DP2` (Recommended for Throughput)** | 2 independent `TP4` instances (`GPU 0–3` and `GPU 4–7`) | Online serving / high concurrency / batch processing | **2.0× aggregate Prefill & Decode throughput** (`~14,000+ tok/s` prefill, `~540+ tok/s` two-stream MTP4 decode) while preserving 100% of `TP4` SM70 custom kernels |
| **`TP4 + DCP2` (Recommended for Ultra-Long Context)** | `--tensor-parallel-size 4 --decode-context-parallel-size 2` | Single-request 512K–1M+ ultra-long context | **2.0× KV cache capacity** (~718K FP16 / ~1.18M E4M3 KV tokens) while keeping `TP4` local head geometry (`6` QSA Q heads, `12` GDN V heads) and ~97% of `TP4` decode speed |

---

## 7. Verification & Benchmark Commands on V100

```bash
# 1. Run unit tests covering QSA, DCP2, GDN metadata/tail split, NVFP4 MoE workspace, and baseline contract
pytest -q \
  tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_qsa_dcp_indices.py \
  tests/models/qwen4_exp/test_qsa_dcp_local_width.py \
  tests/models/qwen4_exp/test_qsa_dcp_attention.py \
  tests/models/qwen4_exp/test_qsa_dcp_packed_cache.py \
  tests/models/qwen4_exp/test_qsa_metadata_padding.py \
  tests/models/qwen4_exp/test_qsa_ops.py \
  tests/distributed/test_dcp_a2a.py \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/models/qwen4_exp/test_sm70_gdn_projection_split.py \
  tests/kernels/test_sm70_gdn_projection_tails.py \
  tests/kernels/test_sm70_qsa_prep.py \
  tests/kernels/test_sm70_qsa_shared_key.py \
  tests/quantization/test_sm70_fp4_mainflow.py \
  tests/quantization/test_sm70_nvfp4_integration.py \
  tests/quantization/test_sm70_warmup.py \
  tests/benchmarks/test_sm70_qwen38_baseline.py

# 2. Run reproducible TP4 + DCP2 benchmark / quality gate
pytest -q tests/benchmarks/test_qwen38_dcp_quality.py

# 3. Run reproducible TP4 + MTP4 benchmark (verifies bit-exact greedy token IDs)
python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --mtp 4 \
  --output /tmp/qwen38_mtp4_opt.json

# 4. Run reproducible TP4 + No-MTP benchmark
python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --mtp 0 \
  --output /tmp/qwen38_nomtp_opt.json
```

---

## 8. File-by-File Code Modification Details (`be3722b..HEAD`)

Below is the complete unified diff and explanation for every modified file across Phase-1 (`d68b5b3`) and the `TP4 + DCP2` Decode optimizations (`D1`–`D12`).

---

### 8.1 `vllm/v1/attention/ops/dcp_alltoall.py` (Optimization D1: Fused DCP A2A Unpack + LSE Combine + FP16 Round + Sigmoid Gate)

- **Why**: Eliminates 2 kernel launches (`output.copy_(merged)` and `_qsa_output_gate`), 2 GPU tensor allocations (`out` and dummy `(1, 1)` `out_lse`), and 3 HBM round-trips per QSA layer on every `TP4 + DCP2` decode step.
- **Diff**:
```diff
diff --git a/vllm/v1/attention/ops/dcp_alltoall.py b/vllm/v1/attention/ops/dcp_alltoall.py
index 1469a5c..5fc18b2 100644
--- a/vllm/v1/attention/ops/dcp_alltoall.py
+++ b/vllm/v1/attention/ops/dcp_alltoall.py
@@ -198,6 +198,7 @@ def _dcp_a2a_unpack_combine_kernel(
     recv_ptr,
     out_ptr,
     out_lse_ptr,
+    gate_ptr,
     recv_stride_N,
     recv_stride_B,
     recv_stride_H,
@@ -207,11 +208,15 @@ def _dcp_a2a_unpack_combine_kernel(
     out_stride_D,
     out_lse_stride_B,
     out_lse_stride_H,
+    gate_stride_B,
+    gate_stride_H,
+    gate_stride_D,
     N: tl.constexpr,
     HEAD_DIM: tl.constexpr,
     IS_BASE_E: tl.constexpr,
     RETURN_LSE: tl.constexpr,
     LSE_PACK_DIM: tl.constexpr,
+    HAS_OUTPUT_GATE: tl.constexpr = False,
 ):
     batch_idx = tl.program_id(0).to(tl.int64)
     head_idx = tl.program_id(1).to(tl.int64)
@@ -310,6 +315,18 @@ def _dcp_a2a_unpack_combine_kernel(
     final_offsets = (
         batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
     )
+    if HAS_OUTPUT_GATE:
+        # Round combined attention output to out_ptr's dtype first, matching
+        # DCP1's attention output and the unfused output.copy_(merged) +
+        # _qsa_output_gate sequence bit-for-bit.
+        rounded = acc.to(out_ptr.dtype.element_ty)
+        gate_offsets = (
+            batch_idx * gate_stride_B
+            + head_idx * gate_stride_H
+            + d_offsets * gate_stride_D
+        )
+        gate = tl.load(gate_ptr + gate_offsets).to(tl.float32)
+        acc = rounded.to(tl.float32) * tl.sigmoid(gate)
     tl.store(out_ptr + final_offsets, acc)

     if RETURN_LSE:
@@ -353,23 +370,32 @@ def _dcp_a2a_unpack_combine(
     lse_pack_dim: int,
     return_lse: bool,
     is_lse_base_on_e: bool,
+    out: torch.Tensor | None = None,
+    output_gate: torch.Tensor | None = None,
 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
     world_size, num_tokens, h_per_rank, _ = recv_buffer.shape
-    out = torch.empty(
-        (num_tokens, h_per_rank, head_dim),
-        device=recv_buffer.device,
-        dtype=recv_buffer.dtype,
-    )
-    out_lse = torch.empty(
-        (num_tokens, h_per_rank) if return_lse else (1, 1),
-        device=recv_buffer.device,
-        dtype=torch.float32 if return_lse else recv_buffer.dtype,
+    if out is None:
+        out = torch.empty(
+            (num_tokens, h_per_rank, head_dim),
+            device=recv_buffer.device,
+            dtype=recv_buffer.dtype,
+        )
+    out_lse = (
+        torch.empty(
+            (num_tokens, h_per_rank),
+            device=recv_buffer.device,
+            dtype=torch.float32,
+        )
+        if return_lse
+        else None
     )
+    has_gate = output_gate is not None
     grid = (num_tokens, h_per_rank, 1)
     _dcp_a2a_unpack_combine_kernel[grid](
         recv_buffer,
         out,
         out_lse,
+        output_gate,
         recv_buffer.stride(0),
         recv_buffer.stride(1),
         recv_buffer.stride(2),
@@ -377,15 +403,20 @@ def _dcp_a2a_unpack_combine(
         out.stride(0),
         out.stride(1),
         out.stride(2),
-        out_lse.stride(0),
-        out_lse.stride(1),
+        out_lse.stride(0) if out_lse is not None else 0,
+        out_lse.stride(1) if out_lse is not None else 0,
+        output_gate.stride(0) if output_gate is not None else 0,
+        output_gate.stride(1) if output_gate is not None else 0,
+        output_gate.stride(2) if output_gate is not None else 0,
         N=world_size,
         HEAD_DIM=head_dim,
         IS_BASE_E=is_lse_base_on_e,
         RETURN_LSE=return_lse,
         LSE_PACK_DIM=lse_pack_dim,
+        HAS_OUTPUT_GATE=has_gate,
     )
     if return_lse:
+        assert out_lse is not None
         return out, out_lse
     return out

@@ -397,6 +428,8 @@ def dcp_a2a_lse_reduce(
     ctx: CPTritonContext | None = None,
     return_lse: bool = False,
     is_lse_base_on_e: bool = True,
+    out: torch.Tensor | None = None,
+    output_gate: torch.Tensor | None = None,
 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
     """
     Combine partial attention outputs across DCP ranks using All-to-All.
@@ -411,6 +444,9 @@ def dcp_a2a_lse_reduce(
         ctx: CPTritonContext (unused, for signature compatibility)
         return_lse: If True, also return the combined global LSE
         is_lse_base_on_e: If True, LSE is base e; if False, base 2
+        out: Optional preallocated [B, H/N, D] output buffer
+        output_gate: Optional [B, H/N, D] sigmoid gate applied after rounding to
+            out.dtype

     Returns:
         Combined output [B, H/N, D] (head-scattered)
@@ -419,9 +455,17 @@ def dcp_a2a_lse_reduce(
     world_size = cp_group.world_size

     if world_size == 1:
+        result = cp_attn_out
+        if out is not None:
+            out.copy_(cp_attn_out)
+            result = out
+        if output_gate is not None:
+            result.copy_(
+                (result.float() * torch.sigmoid(output_gate.float())).to(result.dtype)
+            )
         if return_lse:
-            return cp_attn_out, cp_attn_lse
-        return cp_attn_out
+            return result, cp_attn_lse
+        return result

     B, H, D = cp_attn_out.shape
     if H % world_size != 0:
@@ -454,5 +498,11 @@ def dcp_a2a_lse_reduce(
     work.wait()

     return _dcp_a2a_unpack_combine(
-        recv_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
+        recv_buffer,
+        D,
+        lse_pack_dim,
+        return_lse,
+        is_lse_base_on_e,
+        out=out,
+        output_gate=output_gate,
     )
```

---

### 8.2 `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py` (Optimizations D2 & D10: Half-Width Output Write + Fused Physical Page Resolution in `qsa_localize_dcp_indices`)

- **Why**: Writes only `:local_width` (`1026` columns instead of `2051`) and optionally resolves physical KV cache page slots (`physical_page * PAGE_SIZE + page_offset`) directly inside `_localize_indices` (`RESOLVE_PHYSICAL=True`) so `_qsa_sparse_paged_gqa_splitk_kernel` skips per-tile `block_table` global memory indirection.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
index 050f072..b70e969 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
@@ -13,28 +13,64 @@ import torch
 from vllm.triton_utils import tl, triton


-@triton.jit
+@triton.jit(do_not_specialize=["num_cache_blocks", "num_requests"])
 def _localize_indices(
     src,
     dst,
     src_stride,
     dst_stride,
+    block_table_ptr,
+    token_to_req_ptr,
+    table_stride_req,
+    num_cache_blocks,
+    num_requests,
     WIDTH: tl.constexpr,
+    OUT_WIDTH: tl.constexpr,
     WORLD: tl.constexpr,
     RANK: tl.constexpr,
     INTERLEAVE: tl.constexpr,
     BLOCK: tl.constexpr,
+    PAGE_SIZE: tl.constexpr = 1,
+    PAGE_TABLE_WIDTH: tl.constexpr = 1,
+    RESOLVE_PHYSICAL: tl.constexpr = False,
 ):
     row = tl.program_id(0)
     col = tl.arange(0, BLOCK)
     token = tl.load(src + row * src_stride + col, col < WIDTH, other=-1)
     owned = (col < WIDTH) & (token >= 0) & ((token // INTERLEAVE) % WORLD == RANK)
     local = token // (WORLD * INTERLEAVE) * INTERLEAVE + token % INTERLEAVE
+    if RESOLVE_PHYSICAL:
+        request = tl.load(token_to_req_ptr + row)
+        safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
+        safe_local = tl.maximum(local, 0)
+        logical_page = safe_local // PAGE_SIZE
+        page_offset = safe_local % PAGE_SIZE
+        valid_page = (
+            owned
+            & (request >= 0)
+            & (request < num_requests)
+            & (logical_page < PAGE_TABLE_WIDTH)
+        )
+        physical_page = tl.load(
+            block_table_ptr
+            + safe_request * table_stride_req
+            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
+            mask=valid_page,
+            other=-1,
+        )
+        valid_phys = (
+            valid_page & (physical_page >= 0) & (physical_page < num_cache_blocks)
+        )
+        local = tl.where(valid_phys, physical_page * PAGE_SIZE + page_offset, -1)
     position = tl.cumsum(owned.to(tl.int32), 0) - 1
     count = tl.sum(owned.to(tl.int32), 0)
     # Disjoint writes avoid races between warp-local compaction and padding.
-    tl.store(dst + row * dst_stride + col, -1, (col < WIDTH) & (col >= count))
-    tl.store(dst + row * dst_stride + position, local, owned)
+    tl.store(dst + row * dst_stride + col, -1, (col < OUT_WIDTH) & (col >= count))
+    tl.store(
+        dst + row * dst_stride + position,
+        local,
+        owned & (position < OUT_WIDTH),
+    )


 def qsa_dcp_local_selection_width(
@@ -69,6 +105,9 @@ def qsa_localize_dcp_indices(
     dcp_rank: int,
     interleave_size: int,
     local_block_size: int,
+    block_table: torch.Tensor | None = None,
+    token_to_req: torch.Tensor | None = None,
+    num_cache_blocks: int | None = None,
 ) -> torch.Tensor:
     """Compact this rank's selected IDs into an independent persistent buffer.

@@ -77,7 +116,14 @@ def qsa_localize_dcp_indices(
     Requiring interleave to divide a local page makes flat localization agree
     with block-table slot mapping at every global page boundary.
     """
-    if indices.ndim != 2 or out.shape != indices.shape or indices.shape[1] == 0:
+    if (
+        indices.ndim != 2
+        or out.ndim != 2
+        or out.shape[0] != indices.shape[0]
+        or indices.shape[1] == 0
+        or out.shape[1] == 0
+        or out.shape[1] > indices.shape[1]
+    ):
         raise ValueError("QSA DCP needs matching [rows, selection_width] buffers")
     if indices.dtype != torch.int32 or out.dtype != torch.int32:
         raise ValueError("QSA DCP indices must be int32")
@@ -100,19 +146,33 @@ def qsa_localize_dcp_indices(
     if indices.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
         raise ValueError("QSA DCP output must not alias the MTP selection buffer")
     if dcp_world_size == 1:
-        out.copy_(indices)
+        out.copy_(indices[:, : out.shape[1]])
         return out
     if not indices.is_cuda:
         raise ValueError("QSA DCP localization requires CUDA buffers")
+    resolve_physical = (
+        block_table is not None
+        and token_to_req is not None
+        and num_cache_blocks is not None
+    )
     _localize_indices[(indices.shape[0],)](
         indices,
         out,
         indices.stride(0),
         out.stride(0),
+        block_table,
+        token_to_req,
+        block_table.stride(0) if block_table is not None else 0,
+        num_cache_blocks if num_cache_blocks is not None else 0,
+        block_table.shape[0] if block_table is not None else 0,
         WIDTH=indices.shape[1],
+        OUT_WIDTH=out.shape[1],
         WORLD=dcp_world_size,
         RANK=dcp_rank,
         INTERLEAVE=interleave_size,
         BLOCK=triton.next_power_of_2(indices.shape[1]),
+        PAGE_SIZE=local_block_size,
+        PAGE_TABLE_WIDTH=block_table.shape[1] if block_table is not None else 1,
+        RESOLVE_PHYSICAL=resolve_physical,
     )
     return out
```

---

### 8.3 `vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py` (Optimization D7: Preallocated Output Buffers in `maybe_shared_key_scores`)

- **Why**: Accepts optional `scores` and `visible` workspace views from `qsa_mqa_paged` / `qsa_select_paged_tokens` to avoid allocating `[B, columns]` FP32 and `[B]` INT32 tensors on every QSA layer.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py b/vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py
index 8b5f9b8..51e7a11 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py
@@ -59,14 +59,38 @@ def shared_key_reason(q, cache, table, requests, positions, lengths):


 def maybe_shared_key_scores(
-    q, cache, table, requests, positions, lengths, compress, columns, divisor
+    q,
+    cache,
+    table,
+    requests,
+    positions,
+    lengths,
+    compress,
+    columns,
+    divisor,
+    scores: torch.Tensor | None = None,
+    visible: torch.Tensor | None = None,
 ):
     reason = shared_key_reason(q, cache, table, requests, positions, lengths)
     if reason is not None:
         logger.info_once("SM70 QSA shared-key scorer fallback: %s.", reason)
         return None
-    scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
-    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
+    if (
+        scores is None
+        or scores.shape != (q.shape[0], columns)
+        or scores.dtype != torch.float32
+        or scores.device != q.device
+        or not scores.is_contiguous()
+    ):
+        scores = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
+    if (
+        visible is None
+        or visible.shape != (q.shape[0],)
+        or visible.dtype != torch.int32
+        or visible.device != q.device
+        or not visible.is_contiguous()
+    ):
+        visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
     torch.ops.vllm_sm70_qsa_indexer.run(
         scores,
         visible,
```

---

### 8.4 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` (Optimizations 3.1, D3, D7, D9, D10 & D11: Prefill Visibility Cache, Split-K & Selector Workspaces, Live-Context `score_columns` Bounding, Empty-Tile Skip, and Two-Warp `G=12` Launch)

- **Why**: Caches `(all_visible, max_visible)` across QSA layers in prefill; adds persistent 1D workspaces `_qsa_splitk_workspace` and `_qsa_selector_workspace`; bounds decode `score_columns` to active context length (`_qsa_bounded_decode_score_columns`); skips KV loads and Tensor Core MMAs on empty `-1` tiles (`if tl.max(logical_token, axis=0) >= 0:`); accepts `pre_resolved_indices`; and enables 2-warp SM70 partial kernel launches for `TP4 + DCP2` (`group_size == 12`).
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa.py b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
index aa5e6a6..9347a36 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -70,6 +70,14 @@ _SM70_QSA_GROUPED_PAGE4_WORKSPACES: dict[
         torch.Tensor,
     ],
 ] = {}
+_SM70_QSA_SPLITK_WORKSPACES: dict[
+    tuple[int, int],
+    tuple[int, int, torch.Tensor, torch.Tensor],
+] = {}
+_SM70_QSA_SELECTOR_WORKSPACES: dict[
+    tuple[int, int],
+    tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor],
+] = {}
 _SM70_QSA_GROUPED_PAGE4_ABI_CACHE: tuple[object, int] | None = None


@@ -703,89 +711,92 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
             mask=columns < TOPK,
             other=-1,
         )
-        safe_token = tl.maximum(logical_token, 0)
-        logical_page = safe_token // PAGE_SIZE
-        page_offset = safe_token % PAGE_SIZE
-        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0)
-        if DEVICE_HISTORY:
-            valid &= (logical_token <= position) & (logical_token < length)
-        if RESOLVED_INDICES:
-            physical_page = logical_page
-        else:
-            valid &= logical_page < PAGE_TABLE_WIDTH
-            physical_page = tl.load(
-                block_table_ptr
-                + safe_request * stride_table_req
-                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
-                mask=valid,
-                other=-1,
-            )
-        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
-        # physical_page * block stride can overflow int32 for large caches.
-        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
-        if HOST_CACHE:
-            hot_tokens = tl.load(
-                HOST_INDICES + row * TOPK + columns, columns < TOPK, other=-2
-            )
-            valid &= hot_tokens >= -1
-        if HOST_CACHE:
-            keys, values = load_host_kv(
-                k_cache_ptr,
-                v_cache_ptr,
-                hot_tokens,
-                row,
-                columns,
-                valid,
-                dim_offsets,
-                tl.cdiv(TOPK, 4) * 4,
-                HEAD_DIM,
-            )
-        else:
-            keys = tl.load(
-                k_cache_ptr
-                + safe_page[None, :] * stride_k_block
-                + page_offset[None, :] * stride_k_token
-                + kv_head * stride_k_head
-                + dim_offsets[:, None],
-                mask=valid[None, :],
-                other=0.0,
-            )
-            values = tl.load(
-                v_cache_ptr
-                + safe_page[:, None] * stride_v_block
-                + page_offset[:, None] * stride_v_token
-                + kv_head * stride_v_head
-                + dim_offsets[None, :],
-                mask=valid[:, None],
-                other=0.0,
+        if tl.max(logical_token, axis=0) >= 0:
+            safe_token = tl.maximum(logical_token, 0)
+            logical_page = safe_token // PAGE_SIZE
+            page_offset = safe_token % PAGE_SIZE
+            valid = (request >= 0) & (request < num_requests) & (logical_token >= 0)
+            if DEVICE_HISTORY:
+                valid &= (logical_token <= position) & (logical_token < length)
+            if RESOLVED_INDICES:
+                physical_page = logical_page
+            else:
+                valid &= logical_page < PAGE_TABLE_WIDTH
+                physical_page = tl.load(
+                    block_table_ptr
+                    + safe_request * stride_table_req
+                    + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
+                    mask=valid,
+                    other=-1,
+                )
+            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
+            # physical_page * block stride can overflow int32 for large caches.
+            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
+            if HOST_CACHE:
+                hot_tokens = tl.load(
+                    HOST_INDICES + row * TOPK + columns, columns < TOPK, other=-2
+                )
+                valid &= hot_tokens >= -1
+            if HOST_CACHE:
+                keys, values = load_host_kv(
+                    k_cache_ptr,
+                    v_cache_ptr,
+                    hot_tokens,
+                    row,
+                    columns,
+                    valid,
+                    dim_offsets,
+                    tl.cdiv(TOPK, 4) * 4,
+                    HEAD_DIM,
+                )
+            else:
+                keys = tl.load(
+                    k_cache_ptr
+                    + safe_page[None, :] * stride_k_block
+                    + page_offset[None, :] * stride_k_token
+                    + kv_head * stride_k_head
+                    + dim_offsets[:, None],
+                    mask=valid[None, :],
+                    other=0.0,
+                )
+                values = tl.load(
+                    v_cache_ptr
+                    + safe_page[:, None] * stride_v_block
+                    + page_offset[:, None] * stride_v_token
+                    + kv_head * stride_v_head
+                    + dim_offsets[None, :],
+                    mask=valid[:, None],
+                    other=0.0,
+                )
+            if DEVICE_HISTORY and HISTORY_E4M3:
+                slot = safe_page * PAGE_SIZE + page_offset
+                key_scales = tl.load(HISTORY_SCALES + slot * 2, valid, other=1.0)
+                value_scales = tl.load(HISTORY_SCALES + slot * 2 + 1, valid, other=1.0)
+                keys = (fp8_e4m3fn_bits_to_fp32(keys) * key_scales[None, :]).to(
+                    query.dtype
+                )
+                values = (fp8_e4m3fn_bits_to_fp32(values) * value_scales[:, None]).to(
+                    query.dtype
+                )
+            if KV_E4M3:
+                keys = fp8_e4m3fn_bits_to_fp32(keys).to(query.dtype)
+                values = fp8_e4m3fn_bits_to_fp32(values).to(query.dtype)
+            scores = tl.dot(query, keys)
+            # Scaling scores avoids re-quantizing a scaled query to BF16.
+            scores *= qk_scale_log2
+            scores = tl.where(valid[None, :], scores, -1.0e20)
+            next_max = tl.maximum(max_value, tl.max(scores, axis=1))
+            alpha = tl.math.exp2(max_value - next_max)
+            probabilities = tl.where(
+                valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
             )
-        if DEVICE_HISTORY and HISTORY_E4M3:
-            slot = safe_page * PAGE_SIZE + page_offset
-            key_scales = tl.load(HISTORY_SCALES + slot * 2, valid, other=1.0)
-            value_scales = tl.load(HISTORY_SCALES + slot * 2 + 1, valid, other=1.0)
-            keys = (fp8_e4m3fn_bits_to_fp32(keys) * key_scales[None, :]).to(query.dtype)
-            values = (fp8_e4m3fn_bits_to_fp32(values) * value_scales[:, None]).to(
-                query.dtype
+            accumulator = tl.dot(
+                probabilities.to(values.dtype),
+                values,
+                acc=accumulator * alpha[:, None],
             )
-        if KV_E4M3:
-            keys = fp8_e4m3fn_bits_to_fp32(keys).to(query.dtype)
-            values = fp8_e4m3fn_bits_to_fp32(values).to(query.dtype)
-        scores = tl.dot(query, keys)
-        # Scaling scores avoids re-quantizing a scaled query to BF16.
-        scores *= qk_scale_log2
-        scores = tl.where(valid[None, :], scores, -1.0e20)
-        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
-        alpha = tl.math.exp2(max_value - next_max)
-        probabilities = tl.where(
-            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
-        )
-        accumulator = tl.dot(
-            probabilities.to(values.dtype),
-            values,
-            acc=accumulator * alpha[:, None],
-        )
-        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
-        max_value = next_max
+            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
+            max_value = next_max

     has_values = normalizer > 0
     normalized_output = tl.where(
@@ -1174,6 +1185,8 @@ def qsa_mqa_paged(
     num_columns: int | None = None,
     score_scale: float | None = None,
     shared_key_scoring: bool = False,
+    out_logits: torch.Tensor | None = None,
+    out_visible_blocks: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     """Compute QSA scores directly from a paged compressed-key cache."""

@@ -1217,11 +1230,31 @@ def qsa_mqa_paged(
             compress_ratio,
             columns,
             score_divisor,
+            scores=out_logits,
+            visible=out_visible_blocks,
         )
         if result is not None:
             return result
-    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
-    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
+    if (
+        out_logits is not None
+        and out_logits.shape == (q.shape[0], columns)
+        and out_logits.dtype == torch.float32
+        and out_logits.device == q.device
+        and out_logits.is_contiguous()
+    ):
+        logits = out_logits
+    else:
+        logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
+    if (
+        out_visible_blocks is not None
+        and out_visible_blocks.shape == (q.shape[0],)
+        and out_visible_blocks.dtype == torch.int32
+        and out_visible_blocks.device == q.device
+        and out_visible_blocks.is_contiguous()
+    ):
+        visible_blocks = out_visible_blocks
+    else:
+        visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
     if not q.shape[0] or not columns:
         return logits, visible_blocks
     sm70_single_token = q.shape[0] == 1 and current_platform.is_device_capability(70)
@@ -1523,6 +1556,8 @@ def _qsa_select_by_request(
     compress_ratio: int,
     out: torch.Tensor,
     query_start_loc_cpu: torch.Tensor,
+    visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]]
+    | None = None,
 ) -> bool:
     """Select a multi-request batch one long request at a time.

@@ -1565,6 +1600,8 @@ def _qsa_select_by_request(
                 token_topk,
                 compress_ratio,
                 out[start:end],
+                visible_blocks_cache=visible_blocks_cache,
+                _cache_row_range=(start, end),
             )
         else:
             qsa_select_paged_tokens(
@@ -1577,10 +1614,93 @@ def _qsa_select_by_request(
                 token_topk,
                 compress_ratio,
                 out[start:end],
+                visible_blocks_cache=visible_blocks_cache,
+                _cache_row_range=(start, end),
             )
     return True


+def _qsa_bounded_decode_score_columns(
+    q: torch.Tensor,
+    capacity_columns: int,
+    block_topk: int,
+    compress_ratio: int,
+    max_seq_len: int | None,
+) -> int:
+    if capacity_columns <= 0 or compress_ratio <= 0:
+        return capacity_columns
+    from vllm.config.compilation import CUDAGraphMode
+    from vllm.forward_context import (
+        get_forward_context,
+        is_forward_context_available,
+    )
+
+    capturing = bool(q.is_cuda and torch.cuda.is_current_stream_capturing())
+    bound_seq_len: int | None = None
+    if is_forward_context_available():
+        context = get_forward_context()
+        descriptor = context.batch_descriptor
+        bucket = getattr(descriptor, "attention_context_bucket", None)
+        if capturing or context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
+            bound_seq_len = bucket
+        else:
+            bound_seq_len = max_seq_len if max_seq_len is not None else bucket
+    elif not capturing:
+        bound_seq_len = max_seq_len
+    if bound_seq_len is None or bound_seq_len <= 0:
+        return capacity_columns
+    live_columns = ((bound_seq_len // compress_ratio + 63) // 64) * 64
+    return min(capacity_columns, max(block_topk, live_columns))
+
+
+def _qsa_selector_workspace(
+    q: torch.Tensor,
+    chunk_rows: int,
+    score_columns: int,
+    block_topk: int,
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+    device_index = q.device.index if q.device.index is not None else -1
+    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
+    key = (device_index, stream_id)
+    cache = workspace_cache("qsa_selector", _SM70_QSA_SELECTOR_WORKSPACES)
+    workspace = cache.get(key)
+    needed_logits = max(chunk_rows * score_columns, 1)
+    needed_visible = max(chunk_rows, 1)
+    needed_blocks = max(chunk_rows * block_topk, 1)
+    if (
+        workspace is None
+        or workspace[0] < needed_logits
+        or workspace[1] < needed_visible
+        or workspace[2] < needed_blocks
+    ):
+        logits_cap = 1 << (needed_logits - 1).bit_length()
+        visible_cap = 1 << (needed_visible - 1).bit_length()
+        blocks_cap = 1 << (needed_blocks - 1).bit_length()
+        if workspace is not None:
+            logits_cap = max(logits_cap, workspace[0])
+            visible_cap = max(visible_cap, workspace[1])
+            blocks_cap = max(blocks_cap, workspace[2])
+        logits_buf = torch.empty((logits_cap,), dtype=torch.float32, device=q.device)
+        visible_buf = torch.empty((visible_cap,), dtype=torch.int32, device=q.device)
+        blocks_buf = torch.empty((blocks_cap,), dtype=torch.int32, device=q.device)
+        workspace = (
+            logits_cap,
+            visible_cap,
+            blocks_cap,
+            logits_buf,
+            visible_buf,
+            blocks_buf,
+        )
+        cache[key] = workspace
+    retain_for_capture(cache, workspace, q)
+    _, _, _, logits_buf, visible_buf, blocks_buf = workspace
+    return (
+        logits_buf,
+        visible_buf,
+        blocks_buf[: chunk_rows * block_topk].view(chunk_rows, block_topk),
+    )
+
+
 def qsa_select_paged_tokens(
     q: torch.Tensor,
     k_cache: torch.Tensor,
@@ -1593,6 +1713,10 @@ def qsa_select_paged_tokens(
     out: torch.Tensor | None = None,
     query_start_loc_cpu: torch.Tensor | None = None,
     shared_key_scoring: bool = False,
+    visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]]
+    | None = None,
+    _cache_row_range: tuple[int, int] | None = None,
+    max_seq_len: int | None = None,
 ) -> torch.Tensor:
     """Score, select, and expand QSA indices without host synchronization."""

@@ -1615,6 +1739,7 @@ def qsa_select_paged_tokens(
         compress_ratio,
         out,
         query_start_loc_cpu,
+        visible_blocks_cache=visible_blocks_cache,
     ):
         return out

@@ -1625,19 +1750,34 @@ def qsa_select_paged_tokens(
     key_valid: torch.Tensor | None = None
     all_visible: torch.Tensor | None = None
     if _use_sm70_qsa_indexer_cublas(q, k_cache, page_table):
-        all_visible = _qsa_visible_blocks(
-            token_to_req,
-            query_positions,
-            sequence_lengths,
-            compress_ratio,
+        start_row, end_row = (
+            _cache_row_range if _cache_row_range is not None else (0, rows)
+        )
+        cache_key = (start_row, end_row, int(compress_ratio))
+        cached_visible = (
+            visible_blocks_cache.get(cache_key)
+            if visible_blocks_cache is not None
+            else None
         )
+        if cached_visible is None:
+            all_visible = _qsa_visible_blocks(
+                token_to_req,
+                query_positions,
+                sequence_lengths,
+                compress_ratio,
+            )
+            max_visible = int(all_visible.max().item())
+            if visible_blocks_cache is not None:
+                visible_blocks_cache[cache_key] = (all_visible, max_visible)
+        else:
+            all_visible, max_visible = cached_visible
         # cuBLAS has a rectangular host-side launch shape, unlike the paged
         # Triton kernel's device-side early exit. Bound it to this chunk's live
         # prefix so early-context prefill does not multiply the unused 140K
-        # model-capacity tail. This is one scalar sync per QSA layer.
+        # model-capacity tail. Cached across QSA layers in the same forward pass.
         score_columns = min(
             capacity_columns,
-            max(block_topk, int(all_visible.max().item())),
+            max(block_topk, max_visible),
         )
         if _qsa_indexer_cublas_work_supported(rows, score_columns):
             logger.info_once(
@@ -1653,13 +1793,26 @@ def qsa_select_paged_tokens(
                 k_cache, page_table, score_columns
             )
         else:
-            score_columns = capacity_columns
+            score_columns = _qsa_bounded_decode_score_columns(
+                q, capacity_columns, block_topk, compress_ratio, max_seq_len
+            )
             all_visible = None
+    else:
+        score_columns = _qsa_bounded_decode_score_columns(
+            q, capacity_columns, block_topk, compress_ratio, max_seq_len
+        )
     rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(score_columns * 4, 1))
     chunk_rows = min(rows, rows_per_chunk)
-    blocks_buffer = torch.empty(
-        (chunk_rows, block_topk), dtype=torch.int32, device=q.device
-    )
+    if q.is_cuda:
+        logits_buf, visible_buf, blocks_buffer = _qsa_selector_workspace(
+            q, chunk_rows, score_columns, block_topk
+        )
+    else:
+        logits_buf = None
+        visible_buf = None
+        blocks_buffer = torch.empty(
+            (chunk_rows, block_topk), dtype=torch.int32, device=q.device
+        )
     exact_topk = _use_sm70_qsa_lexicographic_topk(block_topk)
     topk_workspace = (
         None
@@ -1669,6 +1822,7 @@ def qsa_select_paged_tokens(
     for row_start in range(0, rows, rows_per_chunk):
         row_end = min(row_start + rows_per_chunk, rows)
         row_slice = slice(row_start, row_end)
+        chunk_n = row_end - row_start
         if contiguous_keys is not None and key_valid is not None:
             assert all_visible is not None
             logits, visible_blocks = _qsa_mqa_cublas(
@@ -1678,6 +1832,16 @@ def qsa_select_paged_tokens(
                 all_visible[row_slice],
             )
         else:
+            paged_kwargs: dict[str, object] = {
+                "shared_key_scoring": shared_key_scoring,
+            }
+            if score_columns != capacity_columns:
+                paged_kwargs["num_columns"] = score_columns
+            if logits_buf is not None and visible_buf is not None:
+                paged_kwargs["out_logits"] = logits_buf[
+                    : chunk_n * score_columns
+                ].view(chunk_n, score_columns)
+                paged_kwargs["out_visible_blocks"] = visible_buf[:chunk_n]
             logits, visible_blocks = qsa_mqa_paged(
                 q[row_slice],
                 k_cache,
@@ -1686,9 +1850,9 @@ def qsa_select_paged_tokens(
                 query_positions[row_slice],
                 sequence_lengths,
                 compress_ratio,
-                shared_key_scoring=shared_key_scoring,
+                **paged_kwargs,  # type: ignore[arg-type]
             )
-        blocks = blocks_buffer[: row_end - row_start]
+        blocks = blocks_buffer[:chunk_n]
         use_cooperative_topk = (
             blocks.shape[0] <= 32
             and logits.stride(0) % 4 == 0
@@ -1976,6 +2140,40 @@ def _qsa_grouped_page4_workspace(
     )


+def _qsa_splitk_workspace(
+    q: torch.Tensor,
+    num_splits: int,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    device_index = q.device.index if q.device.index is not None else -1
+    stream_id = int(torch.cuda.current_stream(q.device).cuda_stream)
+    key = (device_index, stream_id)
+    cache = workspace_cache("qsa_splitk", _SM70_QSA_SPLITK_WORKSPACES)
+    workspace = cache.get(key)
+    rows, heads, head_dim = q.shape
+    needed_lse = num_splits * rows * heads
+    needed_out = needed_lse * head_dim
+    if workspace is None or workspace[0] < needed_out or workspace[1] < needed_lse:
+        out_capacity = 1 << (max(needed_out, 1) - 1).bit_length()
+        lse_capacity = 1 << (max(needed_lse, 1) - 1).bit_length()
+        if workspace is not None:
+            out_capacity = max(out_capacity, workspace[0])
+            lse_capacity = max(lse_capacity, workspace[1])
+        partial_output_buf = torch.empty(
+            (out_capacity,), dtype=torch.float32, device=q.device
+        )
+        partial_lse_buf = torch.empty(
+            (lse_capacity,), dtype=torch.float32, device=q.device
+        )
+        workspace = (out_capacity, lse_capacity, partial_output_buf, partial_lse_buf)
+        cache[key] = workspace
+    retain_for_capture(cache, workspace, q)
+    _, _, partial_output_buf, partial_lse_buf = workspace
+    return (
+        partial_output_buf[:needed_out].view(num_splits, rows, heads, head_dim),
+        partial_lse_buf[:needed_lse].view(num_splits, rows, heads),
+    )
+
+
 def _qsa_xqa_page4_physical_kv(
     q: torch.Tensor,
     k_cache: torch.Tensor,
@@ -2349,6 +2547,7 @@ def qsa_sparse_paged_attention(
     k_scale: float = 1.0,
     v_scale: float = 1.0,
     lse: torch.Tensor | None = None,
+    pre_resolved_indices: bool = False,
 ) -> torch.Tensor:
     """Run sparse GQA, optionally returning base-2 LSE for a cross-rank merge.

@@ -2453,10 +2652,10 @@ def qsa_sparse_paged_attention(
                 _qsa_output_gate(xqa_output, output_gate_view)
             return xqa_output

-    resolved_indices = _use_sm70_qsa_resolved_indices(
+    resolved_indices = pre_resolved_indices or _use_sm70_qsa_resolved_indices(
         q, k_cache, logical_indices, kv_cache_dtype
     )
-    if resolved_indices:
+    if resolved_indices and not pre_resolved_indices:
         physical_indices = torch.empty_like(logical_indices)
         _qsa_resolve_physical_indices_kernel[(1, triton.cdiv(2051, 256))](
             logical_indices,
@@ -2500,15 +2699,9 @@ def qsa_sparse_paged_attention(
         partial_lse = out
     else:
         # FP32 partials preserve accuracy when merging independently normalized
-        # splits.
-        partial_output = torch.empty(
-            (num_splits, *q.shape), dtype=torch.float32, device=q.device
-        )
-        partial_lse = torch.empty(
-            (num_splits, q.shape[0], q.shape[1]),
-            dtype=torch.float32,
-            device=q.device,
-        )
+        # splits. Reuse per-stream scratch buffers to avoid 24 dynamic tensor
+        # allocations per step across the 12 target QSA layers on TP4+DCP2.
+        partial_output, partial_lse = _qsa_splitk_workspace(q, num_splits)

     partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
     _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
@@ -2624,7 +2817,7 @@ def _use_sm70_qsa_two_warp_partial(
     """Gate the bitwise small-batch SM70 sparse-QSA launch policy."""
     return bool(
         0 < num_query_tokens <= 16
-        and group_size == 6
+        and group_size in (6, 12)
         and head_dim == 256
         and current_platform.is_device_capability(70)
     )
```

---

### 8.5 `vllm/models/qwen4_exp/nvidia/qsa.py` (Optimizations 3.2, D1, D2, D4, D5, D7, D8 & D10: 3D Gate Passthrough, Fused DCP Combine, Persistent DCP AllGather & Prep Buffers, Fused Physical Index Resolution, and Vectorized Draft Block-Table Expansion)

- **Why**: Passes 3D strided `output_gate` views without contiguous copies; fuses DCP LSE combine + FP16 round + gate; reuses persistent `_dcp_ag_recv_buffer` / `_dcp_ag_out_buffer` in `_all_gather_dcp_query` and `_prepared_q_buffer` / cached `kv_cache.unbind(1)` in `_run_qsa`; passes `block_table` into `qsa_localize_dcp_indices` for fused physical resolution; skips redundant `output.zero_()` on `qsa_dcp_sharded`; and vectorizes replicated-draft block-table expansion.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/qsa.py b/vllm/models/qwen4_exp/nvidia/qsa.py
index 7826024..d668aa9 100644
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -153,10 +153,20 @@ class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
             ):
                 raise RuntimeError("QSA draft block-table buffer is too small")
             expanded = self.draft_block_table_buffer[:rows, : columns * expansion]
-            expanded_view = expanded.view(rows, columns, expansion)
-            for sub_block in range(expansion):
-                torch.mul(table, expansion, out=expanded_view[:, :, sub_block])
-                expanded_view[:, :, sub_block].add_(sub_block)
+            expanded_view = expanded.unflatten(1, (columns, expansion))
+            sub_block_offsets = getattr(self, "_draft_sub_block_offsets", None)
+            if (
+                sub_block_offsets is None
+                or sub_block_offsets.shape[-1] != expansion
+                or sub_block_offsets.device != table.device
+                or sub_block_offsets.dtype != table.dtype
+            ):
+                sub_block_offsets = torch.arange(
+                    expansion, dtype=table.dtype, device=table.device
+                ).view(1, 1, expansion)
+                self._draft_sub_block_offsets = sub_block_offsets
+            torch.mul(table.unsqueeze(-1), expansion, out=expanded_view)
+            expanded_view.add_(sub_block_offsets)
             common_attn_metadata = common_attn_metadata.replace(
                 block_table_tensor=expanded
             )
@@ -268,10 +278,14 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             raise NotImplementedError("QSA does not support sliding-window attention")

         num_tokens = attn_metadata.num_actual_tokens
-        if num_tokens and getattr(layer, "host_kv_enabled", False):
-            # Direct host QSA writes every active row, including empty
-            # selections. Only graph padding requires explicit initialization.
-            output[num_tokens:].zero_()
+        if num_tokens and (
+            getattr(layer, "host_kv_enabled", False)
+            or getattr(layer, "qsa_dcp_sharded", False)
+        ):
+            # Direct host QSA and DCP QSA write every active row, including
+            # empty selections. Only graph padding requires explicit initialization.
+            if num_tokens < output.shape[0]:
+                output[num_tokens:].zero_()
         else:
             output.zero_()
         if num_tokens == 0:
@@ -300,9 +314,15 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             return output
         # This tree's FlashAttention cache ABI keeps K/V on dimension 1:
         # [num_blocks, 2, block_size, num_kv_heads, head_size].
-        key_cache, value_cache = kv_cache.unbind(1)
-        key_cache = canonicalize_singleton_dim_strides(key_cache)
-        value_cache = canonicalize_singleton_dim_strides(value_cache)
+        kv_cache_key = (kv_cache.data_ptr(), kv_cache.shape, kv_cache.stride())
+        if getattr(self, "_cached_kv_unbind_key", None) == kv_cache_key:
+            key_cache, value_cache = self._cached_kv_unbind_pair
+        else:
+            key_cache, value_cache = kv_cache.unbind(1)
+            key_cache = canonicalize_singleton_dim_strides(key_cache)
+            value_cache = canonicalize_singleton_dim_strides(value_cache)
+            self._cached_kv_unbind_key = kv_cache_key
+            self._cached_kv_unbind_pair = (key_cache, value_cache)
         if query.dtype not in (torch.float16, torch.bfloat16):
             raise NotImplementedError("Qwen4Exp QSA requires FP16/BF16 queries")
         if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
@@ -373,6 +393,62 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         )
         return output

+    def _all_gather_dcp_query(
+        self,
+        query: torch.Tensor,
+        group: Any,
+    ) -> torch.Tensor:
+        q_contig = query if query.is_contiguous() else query.contiguous()
+        device_group = getattr(group, "device_group", None)
+        if device_group is None and hasattr(group, "device_communicator"):
+            device_group = getattr(group.device_communicator, "device_group", None)
+        if device_group is None:
+            return group.all_gather(q_contig, dim=1)
+        rows, heads, dim = q_contig.shape
+        world_size = self.dcp_world_size
+        needed = world_size * rows * heads * dim
+        recv_buf = getattr(self, "_dcp_ag_recv_buffer", None)
+        if (
+            recv_buf is None
+            or recv_buf.numel() < needed
+            or recv_buf.dtype != q_contig.dtype
+            or recv_buf.device != q_contig.device
+        ):
+            cap = 1 << (max(needed, 1) - 1).bit_length()
+            if (
+                recv_buf is not None
+                and recv_buf.device == q_contig.device
+                and recv_buf.dtype == q_contig.dtype
+            ):
+                cap = max(cap, recv_buf.numel())
+            recv_buf = torch.empty((cap,), dtype=q_contig.dtype, device=q_contig.device)
+            self._dcp_ag_recv_buffer = recv_buf
+        recv_view = recv_buf[:needed].view(world_size * rows, heads, dim)
+        torch.distributed.all_gather_into_tensor(
+            recv_view, q_contig, group=device_group
+        )
+        if rows == 1:
+            return recv_view.view(1, world_size * heads, dim)
+        out_buf = getattr(self, "_dcp_ag_out_buffer", None)
+        if (
+            out_buf is None
+            or out_buf.numel() < needed
+            or out_buf.dtype != q_contig.dtype
+            or out_buf.device != q_contig.device
+        ):
+            cap = 1 << (max(needed, 1) - 1).bit_length()
+            if (
+                out_buf is not None
+                and out_buf.device == q_contig.device
+                and out_buf.dtype == q_contig.dtype
+            ):
+                cap = max(cap, out_buf.numel())
+            out_buf = torch.empty((cap,), dtype=q_contig.dtype, device=q_contig.device)
+            self._dcp_ag_out_buffer = out_buf
+        out_view = out_buf[:needed].view(rows, world_size, heads, dim)
+        out_view.copy_(recv_view.view(world_size, rows, heads, dim).movedim(0, 1))
+        return out_view.view(rows, world_size * heads, dim)
+
     def _forward_qsa_dcp(
         self,
         layer: torch.nn.Module,
@@ -386,6 +462,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         output_gate: torch.Tensor,
     ) -> None:
         from vllm.distributed import get_dcp_group
+        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

         from .ops.qsa import _qsa_output_gate, qsa_sparse_paged_attention
         from .ops.qsa_dcp import (
@@ -405,33 +482,37 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             )
         ):
             raise RuntimeError("QSA DCP target workspaces are not initialized")
+        # Only this prefix of the compacted columns can hold an owned token.
+        local_width = qsa_dcp_local_selection_width(
+            layer.indexer.token_topk,
+            layer.indexer.compress_ratio,
+            self.dcp_world_size,
+            layer.cp_kv_cache_interleave_size,
+            logical_indices.shape[1],
+        )
+        resolve_physical = bool(key_cache.shape[0] * key_cache.shape[1] < 2**31)
         local_indices = qsa_localize_dcp_indices(
             logical_indices,
-            local_indices_buffer[: query.shape[0]],
+            local_indices_buffer[: query.shape[0], :local_width],
             dcp_world_size=self.dcp_world_size,
             dcp_rank=self.dcp_rank,
             interleave_size=layer.cp_kv_cache_interleave_size,
             local_block_size=key_cache.shape[1],
+            block_table=block_table if resolve_physical else None,
+            token_to_req=token_to_req if resolve_physical else None,
+            num_cache_blocks=key_cache.shape[0] if resolve_physical else None,
         )
         group = get_dcp_group()
-        gathered_query = group.all_gather(query.contiguous(), dim=1)
+        gathered_query = self._all_gather_dcp_query(query, group)
         partial_output = partial_output_buffer[: query.shape[0]]
         partial_lse = partial_lse_buffer[: query.shape[0]]
         if partial_output.shape != gathered_query.shape:
             raise RuntimeError("QSA DCP partial output has wrong head geometry")
-        # Only this prefix of the compacted columns can hold an owned token.
-        local_width = qsa_dcp_local_selection_width(
-            layer.indexer.token_topk,
-            layer.indexer.compress_ratio,
-            self.dcp_world_size,
-            layer.cp_kv_cache_interleave_size,
-            local_indices.shape[1],
-        )
         qsa_sparse_paged_attention(
             gathered_query,
             key_cache,
             value_cache,
-            local_indices[:, :local_width],
+            local_indices,
             block_table,
             token_to_req,
             partial_output,
@@ -439,7 +520,19 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             k_scale=layer._k_scale_float,
             v_scale=layer._v_scale_float,
             lse=partial_lse,
+            pre_resolved_indices=resolve_physical,
         )
+        gate_view = output_gate.view_as(output)
+        if self.dcp_combine is dcp_a2a_lse_reduce:
+            dcp_a2a_lse_reduce(
+                partial_output,
+                partial_lse,
+                group,
+                is_lse_base_on_e=False,
+                out=output,
+                output_gate=gate_view,
+            )
+            return
         merged = cast(
             torch.Tensor,
             self.dcp_combine(
@@ -453,7 +546,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         # apply the Triton gate DCP1's page4 route uses. This replaces a chain
         # of five elementwise kernels (12.2 us per decode layer on V100).
         output.copy_(merged)
-        _qsa_output_gate(output, output_gate.view_as(output))
+        _qsa_output_gate(output, gate_view)


 def _verify_e4m3_kv_requirements(
@@ -862,6 +955,8 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
         self,
         qkv: torch.Tensor,
         positions: torch.Tensor,
+        *,
+        flatten_gate: bool = True,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
         """Split, normalize, and rotate Q/K using this tree's Qwen3 API."""
         q_gate, key, value = qkv.split(
@@ -875,7 +970,9 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             key.view(*token_shape, self.num_kv_heads, self.head_dim)
         ).reshape(*token_shape, self.kv_size)
         query, key = self.rotary_emb(positions, query, key)
-        return query, key, value, gate.reshape(*token_shape, self.q_size)
+        if flatten_gate:
+            gate = gate.reshape(*token_shape, self.q_size)
+        return query, key, value, gate

     def _use_dense_short_context(
         self, num_tokens: int, side_metadata: QSAForwardMetadata
@@ -978,12 +1075,48 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             self.host_kv.write(key, value, main_metadata.slot_mapping)
         elif query.dim() == 2:
             # SM70 prep: raw qkv rows -> normalized/rotated query + cache write.
-            key_cache, value_cache = self.kv_cache.unbind(1)
-            prepared = query.new_empty((num_tokens, self.num_heads, self.head_dim))
-            text_positions = positions[0] if positions.dim() == 2 else positions
+            kv_cache_key = (
+                self.kv_cache.data_ptr(),
+                self.kv_cache.shape,
+                self.kv_cache.stride(),
+            )
+            if getattr(self, "_cached_raw_kv_unbind_key", None) == kv_cache_key:
+                key_cache, value_cache = self._cached_raw_kv_unbind_pair
+            else:
+                key_cache, value_cache = self.kv_cache.unbind(1)
+                self._cached_raw_kv_unbind_key = kv_cache_key
+                self._cached_raw_kv_unbind_pair = (key_cache, value_cache)
+            needed_q = num_tokens * self.num_heads * self.head_dim
+            prep_buf = getattr(self, "_prepared_q_buffer", None)
+            if (
+                prep_buf is None
+                or prep_buf.numel() < needed_q
+                or prep_buf.dtype != query.dtype
+                or prep_buf.device != query.device
+            ):
+                cap = 1 << (max(needed_q, 1) - 1).bit_length()
+                if (
+                    prep_buf is not None
+                    and prep_buf.device == query.device
+                    and prep_buf.dtype == query.dtype
+                ):
+                    cap = max(cap, prep_buf.numel())
+                prep_buf = torch.empty((cap,), dtype=query.dtype, device=query.device)
+                self._prepared_q_buffer = prep_buf
+            prepared = prep_buf[:needed_q].view(
+                num_tokens, self.num_heads, self.head_dim
+            )
+            text_positions = (positions[0] if positions.dim() == 2 else positions)[
+                :num_tokens
+            ]
+            if text_positions.dtype != torch.int64:
+                text_positions = text_positions.to(torch.int64)
+            slot_mapping = main_metadata.slot_mapping[:num_tokens]
+            if slot_mapping.dtype != torch.int64:
+                slot_mapping = slot_mapping.to(torch.int64)
             torch.ops._C.qsa_prep_sm70_out(
                 query[:num_tokens],
-                text_positions[:num_tokens].to(torch.int64),
+                text_positions,
                 self.rotary_emb.cos_sin_cache,
                 self.q_norm.weight,
                 self.k_norm.weight,
@@ -991,7 +1124,7 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
                 prepared,
                 key_cache,
                 value_cache,
-                main_metadata.slot_mapping[:num_tokens].to(torch.int64),
+                slot_mapping,
             )
             query = key = value = prepared
         else:
@@ -1043,12 +1176,14 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             q_gate = qkv[:, : self.q_size * 2].view(
                 num_tokens, self.num_heads, 2 * self.head_dim
             )
-            gate = q_gate[..., self.head_dim :].reshape(num_tokens, self.q_size)
+            gate = q_gate[..., self.head_dim :]
             query = qkv
             key = value = qkv
             attn_output = qkv.new_empty((num_tokens, self.num_heads, self.head_dim))
         else:
-            q, k, v, gate = self._project_qkv_gate(qkv, positions)
+            q, k, v, gate = self._project_qkv_gate(
+                qkv, positions, flatten_gate=False
+            )
             query = q.view(num_tokens, self.num_heads, self.head_dim)
             key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
             value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
```

---

### 8.6 `vllm/models/qwen4_exp/common/qsa_cache.py` (Optimizations 3.1 & D9: Shared `visible_blocks_cache` and Live `max_seq_len` Upper Bound on `QSAForwardMetadata`)

- **Why**: Adds `visible_blocks_cache` and `max_seq_len` (tightened from `seq_lens_cpu_upper_bound` on non-dummy batches) to `QSAForwardMetadata`.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/common/qsa_cache.py b/vllm/models/qwen4_exp/common/qsa_cache.py
index f087d0b..b33e090 100644
--- a/vllm/models/qwen4_exp/common/qsa_cache.py
+++ b/vllm/models/qwen4_exp/common/qsa_cache.py
@@ -14,7 +14,7 @@ shared by the generic cache-layout planner.
 from __future__ import annotations

 import math
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 from functools import cache
 from typing import ClassVar

@@ -606,6 +606,10 @@ class QSAForwardMetadata(AttentionMetadata):
     # Host copy of query_start_loc. It lets QSA selection split a mixed batch
     # by request without a device synchronization.
     query_start_loc_cpu: torch.Tensor | None = None
+    max_seq_len: int | None = None
+    visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]] = field(
+        default_factory=dict
+    )


 class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
@@ -624,6 +628,7 @@ class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
         block_table_width: int,
     ) -> None:
         super().__init__(kv_cache_spec, layer_names, vllm_config, device)
+        self.max_model_len = vllm_config.model_config.max_model_len
         self.is_circular_buffer = isinstance(kv_cache_spec, CircularBufferSpec)
         self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
         self.has_sharded_main_owner = any(
@@ -788,6 +793,16 @@ class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
             self.is_circular_buffer or self.compress_ratio != 1
         ):
             slot_mapping.fill_(PAD_SLOT_ID)
+        max_seq_len = common_attn_metadata.max_seq_len
+        if (
+            not common_attn_metadata.is_dummy_batch
+            and max_seq_len >= self.max_model_len
+            and common_attn_metadata.seq_lens_cpu_upper_bound is not None
+            and common_attn_metadata.seq_lens_cpu_upper_bound.numel() > 0
+        ):
+            upper_bound = int(common_attn_metadata.seq_lens_cpu_upper_bound.max().item())
+            if upper_bound > 0:
+                max_seq_len = min(max_seq_len, upper_bound)
         return QSAForwardMetadata(
             block_table=block_table,
             slot_mapping=slot_mapping,
@@ -800,6 +815,7 @@ class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
             storage_block_size=self.storage_block_size,
             compress_ratio=self.compress_ratio,
             query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
+            max_seq_len=max_seq_len,
         )
```

---

### 8.7 `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` (Optimizations 3.1, D7 & D9: Persistent Indexer Query Buffer and `max_seq_len` Passthrough in `QSAIndexer.forward`)

- **Why**: Reuses a persistent 1D `_q_buffer` instead of allocating `projected_q.new_empty(num_tokens, 4, 128)` per layer, and forwards `visible_blocks_cache` and `max_seq_len` into `qsa_select_paged_tokens`.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
index 0369fac..d894ed6 100644
--- a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
@@ -166,6 +166,28 @@ class QSAIndexer(nn.Module):
             cache_config=cache_config,
             vllm_config=vllm_config,
         )
+        self._q_buffer: torch.Tensor | None = None
+
+    def _get_q_buffer(
+        self,
+        num_tokens: int,
+        dtype: torch.dtype,
+        device: torch.device,
+    ) -> torch.Tensor:
+        needed = num_tokens * self.index_n_heads * self.index_head_dim
+        buf = self._q_buffer
+        if (
+            buf is None
+            or buf.numel() < needed
+            or buf.dtype != dtype
+            or buf.device != device
+        ):
+            cap = 1 << (max(needed, 1) - 1).bit_length()
+            if buf is not None and buf.device == device and buf.dtype == dtype:
+                cap = max(cap, buf.numel())
+            buf = torch.empty((cap,), dtype=dtype, device=device)
+            self._q_buffer = buf
+        return buf[:needed].view(num_tokens, self.index_n_heads, self.index_head_dim)

     @property
     def output_width(self) -> int:
@@ -248,11 +270,7 @@ class QSAIndexer(nn.Module):
         compressed_key_cache = self.compressed_key_cache.kv_cache

         if self.use_fused_pre_indexer:
-            q = projected_q.new_empty(
-                num_tokens,
-                self.index_n_heads,
-                self.index_head_dim,
-            )
+            q = self._get_q_buffer(num_tokens, projected_q.dtype, projected_q.device)
             qsa_pre_indexer(
                 projected_q,
                 raw_keys,
@@ -361,6 +379,8 @@ class QSAIndexer(nn.Module):
             out,
             query_start_loc_cpu=compressed_metadata.query_start_loc_cpu,
             shared_key_scoring=self.shared_key_scoring,
+            visible_blocks_cache=compressed_metadata.visible_blocks_cache,
+            max_seq_len=compressed_metadata.max_seq_len,
         )
```

---

### 8.8 `vllm/v1/attention/backends/gdn_attn.py` (Optimization D6: Lazy `compute_num_computed_tokens()` on Decode in `GDNAttentionMetadataBuilder.build`)

- **Why**: Moves `context_lens_tensor = m.compute_num_computed_tokens()` inside `if num_prefills > 0:`, saving 12 GPU kernel launches and 12 tensor allocations per decode step across the 6 GDN cache groups in `TP4 + DCP2`.
- **Diff**:
```diff
diff --git a/vllm/v1/attention/backends/gdn_attn.py b/vllm/v1/attention/backends/gdn_attn.py
index 3fb9816..34f5ce9 100644
--- a/vllm/v1/attention/backends/gdn_attn.py
+++ b/vllm/v1/attention/backends/gdn_attn.py
@@ -1188,7 +1188,6 @@ class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]
             if fast_metadata is not None:
                 return fast_metadata
         self._ddtree_fast_tail_key = None
-        context_lens_tensor = m.compute_num_computed_tokens()
         nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
         block_table_tensor = mamba_get_block_table_tensor(
             m.block_table_tensor,
@@ -1589,6 +1588,7 @@ class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]
                 ).to(device=gpu_device, non_blocking=True)

         if num_prefills > 0:
+            context_lens_tensor = m.compute_num_computed_tokens()
             has_initial_state = context_lens_tensor > 0
             if spec_sequence_masks_cpu is not None:
                 has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
```

---

### 8.9 `vllm/v1/worker/gpu_model_runner.py` (Optimization D12: Gate `dcp_local_seq_lens` CPU Calculation & H2D Copy on `is_deepseek_mla` in `ModelRunnerV1`)

- **Why**: Skips computing and copying `dcp_local_seq_lens` (and cloning it in `SpecDecodeBaseProposer`) when `is_deepseek_mla` is `False`.
- **Diff**:
```diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 001204b..44268e2 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -5293,7 +5293,7 @@ class GPUModelRunner(
             ]
             return current_mamba_state_block_ids_by_gid[kv_cache_gid]

-        if self.dcp_world_size > 1:
+        if self.dcp_world_size > 1 and self.model_config.is_deepseek_mla:
             self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                 self.optimistic_seq_lens_cpu[:num_reqs],
                 self.dcp_world_size,
```

---

### 8.10 `vllm/v1/worker/gpu/model_runner.py` (Optimization D12: Gate `prepare_dcp_local_seq_lens` Triton Kernel on `is_deepseek_mla` in `ModelRunnerV2`)

- **Why**: Skips launching `_dcp_local_seq_lens_kernel` on every step when `is_deepseek_mla` is `False`.
- **Diff**:
```diff
diff --git a/vllm/v1/worker/gpu/model_runner.py b/vllm/v1/worker/gpu/model_runner.py
index 324b32e..7f46a44 100644
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -1085,7 +1085,7 @@ class GPUModelRunner(LoRAModelRunnerMixin):
         seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

         dcp_local_seq_lens = None
-        if self.use_dcp:
+        if self.use_dcp and self.model_config.is_deepseek_mla:
             # Prepare dcp local seq_lens.
             prepare_dcp_local_seq_lens(
                 self.input_buffers.dcp_local_seq_lens,
```

---

### 8.11 `vllm/v1/worker/gpu/cudagraph_utils.py` (Optimization D12: Gate Dummy `prepare_dcp_local_seq_lens` on `ag.backend.is_mla()` during Graph Capture)

- **Why**: Skips launching `_dcp_local_seq_lens_kernel` during CUDA graph capture when no attention group uses MLA.
- **Diff**:
```diff
diff --git a/vllm/v1/worker/gpu/cudagraph_utils.py b/vllm/v1/worker/gpu/cudagraph_utils.py
index 1b04912..adf13da 100644
--- a/vllm/v1/worker/gpu/cudagraph_utils.py
+++ b/vllm/v1/worker/gpu/cudagraph_utils.py
@@ -809,7 +809,9 @@ def prepare_inputs_to_capture(
     )

     # HACK(woosuk): Special handling for DCP.
-    if block_tables.cp_size > 1:
+    if block_tables.cp_size > 1 and any(
+        ag.backend.is_mla() for group in attn_groups for ag in group
+    ):
         prepare_dcp_local_seq_lens(
             input_buffers.dcp_local_seq_lens,
             input_batch.seq_lens,
```

---

### 8.12 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` (Optimization 3.4: GDN Input Projection Tail-Split Grid Shrink (`(m, 16)` -> `(m, 6)`))

- **Why**: Offsets `COPY_QKV=False` column indexing directly to `QKV = 2560` and shrinks the 2D launch grid from 16 column blocks to 6, eliminating 62.5% of CTAs and all unused `qkv` HBM loads.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py b/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
index 14b4750..ecedc69 100644
--- a/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
+++ b/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
@@ -83,11 +83,15 @@ def _qwen38_gdn_projection_split_kernel(
     COPY_QKV: tl.constexpr,
 ):
     row = tl.program_id(0)
-    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
-    value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
     if COPY_QKV:
+        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
+        value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
         tl.store(qkv + row * QKV + col, value, col < QKV)
-    tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
+        tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
+    else:
+        z_col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
+        value = tl.load(qkvz + row * (QKV + Z) + QKV + z_col, z_col < Z, other=0)
+        tl.store(z + row * Z + z_col, value, z_col < Z)
     if tl.program_id(1) == 0:
         tl.static_assert(B + A <= BLOCK)
         gate_col = tl.arange(0, BLOCK)
@@ -120,7 +124,7 @@ def _split_gdn_projection_tails(qkvz, ba, z_out):
     m = qkvz.shape[0]
     qkv = qkvz[:, :2560]
     b, a = (qkvz.new_empty((m, 12)) for _ in range(2))
-    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
+    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(1536, 256))](
         qkvz,
         ba,
         qkv,
```

---

### 8.13 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (Optimization 3.4: Enable Fused `_split_gdn_projection_tails` for Prefill (`M > 1`))

- **Why**: Relaxes `mixed_qkvz.shape[0] in (5, 20)` to `mixed_qkvz.shape[0] > 1` so prefill batches also fuse the `(b, a, z)` tail extraction into a single kernel launch.
- **Diff**:
```diff
diff --git a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
index a82ba15..f517f21 100644
--- a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
@@ -5813,7 +5813,7 @@ def qwen_gdn_input_projection_core(
         copied_tails = (
             not self.gqa_interleaved_layout
             and not self.disable_tp_for_ba_proj
-            and mixed_qkvz.shape[0] in (5, 20)
+            and mixed_qkvz.shape[0] > 1
             and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
             and self.value_dim // self.tp_size == 1536
             and z_out.shape == (mixed_qkvz.shape[0], 12, 128)
```

---

### 8.14 `vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py` (Optimization 3.3: Persistent Read-Only `arange` Buffer in `NvFp4MoEWorkspace._eager_buffers`)

- **Why**: Replaces 96 per-layer `torch.arange` GPU kernel launches and allocations per prefill chunk with views into a cached per-device `int32` sequence buffer.
- **Diff**:
```diff
diff --git a/vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py b/vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py
index 15b070d..5f397a2 100644
--- a/vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py
+++ b/vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py
@@ -18,6 +18,16 @@ from vllm.model_executor.layers.quantization.utils.sm70_layer_workspaces import


 class NvFp4MoEWorkspace:
+    _eager_arange_cache: dict[torch.device, torch.Tensor] = {}
+
+    @classmethod
+    def _get_eager_arange(cls, device: torch.device, length: int) -> torch.Tensor:
+        buf = cls._eager_arange_cache.get(device)
+        if buf is None or buf.numel() < length:
+            buf = torch.arange(length, dtype=torch.int32, device=device)
+            cls._eager_arange_cache[device] = buf
+        return buf
+
     @classmethod
     def allocate(cls, layer: RoutedExperts) -> None:
         device = layer.w13_tm_weight.device
@@ -123,9 +133,9 @@ class NvFp4MoEWorkspace:
             "active_expert_ids": layer._nvfp4_sm70_active_expert_ids[:slots],
         }

-    @staticmethod
+    @classmethod
     def _eager_buffers(
-        layer: RoutedExperts, num_tokens: int, indexed_w13: bool
+        cls, layer: RoutedExperts, num_tokens: int, indexed_w13: bool
     ) -> dict[str, torch.Tensor]:
         device = layer.w13_tm_weight.device
         top_k = int(layer.sm70_nvfp4_top_k)
@@ -136,6 +146,7 @@ class NvFp4MoEWorkspace:
         workspace_size = torch.ops._moe_C.moe_permute_sort_workspace_size(
             slots, layer.global_num_experts
         )
+        arange_buf = cls._get_eager_arange(device, slots + 1)
         return {
             "output": torch.empty(
                 num_tokens, hidden, dtype=torch.float16, device=device
@@ -171,9 +182,7 @@ class NvFp4MoEWorkspace:
             "topk_ids": torch.empty(
                 num_tokens, top_k, dtype=torch.int32, device=device
             ),
-            "token_expert_indices": torch.arange(
-                slots, dtype=torch.int32, device=device
-            ).view(num_tokens, top_k),
+            "token_expert_indices": arange_buf[:slots].view(num_tokens, top_k),
             "permuted_idx": torch.empty(slots, dtype=torch.int32, device=device),
             "sort_workspace": torch.empty(
                 workspace_size, dtype=torch.int8, device=device
@@ -182,9 +191,7 @@ class NvFp4MoEWorkspace:
             "sorted_row_idx": torch.empty(slots, dtype=torch.int32, device=device),
             "topk_ids_for_sort": torch.empty(slots, dtype=torch.int32, device=device),
             "dense_expert_ids": layer._nvfp4_sm70_dense_expert_ids,
-            "compact_offsets": torch.arange(
-                slots + 1, dtype=torch.int32, device=device
-            ),
+            "compact_offsets": arange_buf[: slots + 1],
             "active_expert_ids": torch.empty(slots, dtype=torch.int32, device=device),
         }
```

---

### 8.15 `vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py` (Optimization 3.3: Eliminate Redundant `output.zero_()`, `permuted_idx.fill_()`, and `topk_ids` Copy in NVFP4 MoE)

- **Why**: Guards `output.zero_()` and `permuted_idx.fill_(slots)` behind `if layer.expert_map is not None:` and reuses contiguous `int32` `topk_ids` without copying.
- **Diff**:
```diff
diff --git a/vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py b/vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py
index 2a39f1e..c2edaaf 100644
--- a/vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py
+++ b/vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py
@@ -1411,10 +1411,14 @@ class ModelOptNvFp4SM70MoEMethod(ModelOptNvFp4FusedMoE):
             stage_expert_ids = buffers["active_expert_ids"]
             stage_experts = slots
         else:
-            output.zero_()
-            topk_ids_i32 = buffers["topk_ids"]
-            topk_ids_i32.copy_(topk_ids, non_blocking=True)
-            buffers["permuted_idx"].fill_(slots)
+            if layer.expert_map is not None:
+                output.zero_()
+                buffers["permuted_idx"].fill_(slots)
+            if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
+                topk_ids_i32 = topk_ids
+            else:
+                topk_ids_i32 = buffers["topk_ids"]
+                topk_ids_i32.copy_(topk_ids, non_blocking=True)
             if indexed_w13:
                 self.native_ops.moe_permute_metadata_with_scratch(
                     x,
```
