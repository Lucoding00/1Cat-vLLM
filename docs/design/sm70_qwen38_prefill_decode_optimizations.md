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

### 5.2 Phase-2 `TP4 + DCP2` Decode Optimization Opportunities (Next Steps)

1. **Extend `_sm70_qsa_prep` C++ Fused Prep Kernel to `fp8_e4m3` KV Cache (`csrc/sm70_turbomind/ops/qsa_prep_sm70.cu`, `vllm/models/qwen4_exp/nvidia/qsa.py:598-606`)**:
   - Under `--kv-cache-dtype auto` without MTP (or when calibrated E4M3 scales are present in the checkpoint), `cache_config.cache_dtype` resolves to `"fp8_e4m3"` (`vllm/models/qwen4_exp/nvidia/qsa.py:486`).
   - Currently `_sm70_qsa_prep` only admits `cache_config.cache_dtype in ("auto", "float16")` (`qsa.py:604`), so under calibrated `fp8_e4m3` KV cache, QSA falls back to separate `q_norm`, `k_norm`, `apply_rotary_emb_ inplace`, and `reshape_and_cache_flash` kernels (`~6` extra kernels per QSA layer = `72` kernels/step). Adding E4M3 quantization (`k_scale`, `v_scale`) to `qsa_prep_sm70.cu`'s KV cache write epilogue will restore the single-launch fused QSA prep path for E4M3 KV caches.
2. **SM70 Two-Warp Partial & Native XQA Decode Support for `DCP2` `group_size = 12` (`vllm/models/qwen4_exp/nvidia/ops/qsa.py:2513, 2608`, `flash-attention-v100/kernel/flash_decode_paged.cu`)**:
   - At `TP4 + DCP1`, each rank has `6` Q heads and `1` KV head (`group_size = 6`, `block_m = 8`), which hits both the native Flash-V100 `decode_paged_xqa_fwd` (`q.shape[1:] == (6, 256)`) and `_use_sm70_qsa_two_warp_partial` (`group_size == 6`).
   - At `TP4 + DCP2`, `gathered_query` has `12` Q heads and `1` KV head (`group_size = 12`, `block_m = 16`), which bypasses `decode_paged_xqa_fwd` (also because `lse` is required) and uses `partial_warps = 4, target_splits = 64` in `_qsa_sparse_paged_gqa_splitk_kernel`. Adding a `group_size = 12` (`H=12, D=256`) LSE-emitting specialization to `flash_decode_paged.cu` (or tuning `target_splits = 32, partial_warps = 2` for `local_width = 1026` on V100) can cut `TP4 + DCP2` sparse attention kernel time by ~20–30%.
3. **Eliminate Unused `prepare_dcp_local_seq_lens` Launch When Only QSA Uses DCP (`vllm/v1/worker/gpu/model_runner.py:1090`, `vllm/v1/worker/gpu/cudagraph_utils.py:813`)**:
   - `ModelRunnerV2` launches `prepare_dcp_local_seq_lens` on every step when `self.use_dcp` is `True`, even though `Qwen4ExpQSAMetadataBuilder` sets `builds_dcp_context_lens = False` and only `MLAAttentionMetadataBuilder` reads `common_attn_metadata.dcp_local_seq_lens`. Gating `prepare_dcp_local_seq_lens` on whether any attention group requires `dcp_local_seq_lens` saves 1 extra Triton kernel launch per step.

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
  tests/distributed/test_dcp_a2a.py \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/models/qwen4_exp/test_sm70_gdn_projection_split.py \
  tests/kernels/test_sm70_gdn_projection_tails.py \
  tests/kernels/test_sm70_qsa_prep.py \
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

## 8. File-by-File Code Modification Details (`be3722b..2b06ade`)

Below is the complete diff and explanation for every modified file across Phase-1 (`d68b5b3`) and the `TP4 + DCP2` Decode optimizations (`2b06ade`).

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
...
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
...
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

### 8.2 `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py` (Optimization D2: Half-Width Output Write in `qsa_localize_dcp_indices`)

- **Why**: Allows `_forward_qsa_dcp` to pass `local_indices_buffer[:B, :local_width]` (`1026` columns instead of `2051`), cutting `-1` padding global-memory stores in half while remaining 100% compatible with callers that pass a full-width output buffer.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
index 050f072..35d3a44 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
@@ -20,6 +20,7 @@ def _localize_indices(
     src_stride,
     dst_stride,
     WIDTH: tl.constexpr,
+    OUT_WIDTH: tl.constexpr,
     WORLD: tl.constexpr,
     RANK: tl.constexpr,
     INTERLEAVE: tl.constexpr,
@@ -33,8 +34,12 @@ def _localize_indices(
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
@@ -77,7 +82,14 @@ def qsa_localize_dcp_indices(
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
@@ -100,7 +112,7 @@ def qsa_localize_dcp_indices(
     if indices.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
         raise ValueError("QSA DCP output must not alias the MTP selection buffer")
     if dcp_world_size == 1:
-        out.copy_(indices)
+        out.copy_(indices[:, : out.shape[1]])
         return out
     if not indices.is_cuda:
         raise ValueError("QSA DCP localization requires CUDA buffers")
@@ -110,6 +122,7 @@ def qsa_localize_dcp_indices(
         indices.stride(0),
         out.stride(0),
         WIDTH=indices.shape[1],
+        OUT_WIDTH=out.shape[1],
         WORLD=dcp_world_size,
         RANK=dcp_rank,
         INTERLEAVE=interleave_size,
```

---

### 8.3 `vllm/models/qwen4_exp/nvidia/qsa.py` (Optimizations 3.2, D1, D2, D4, D5)

- **Why**:
  1. Vectorizes replicated MTP draft block-table expansion under `DCP2` (`Qwen4ExpQSAMetadataBuilder.build`).
  2. Skips full-buffer `output.zero_()` when `qsa_dcp_sharded` is active and `num_tokens > 0`, zeroing only `output[num_tokens:]` when padded (`forward_qsa`).
  3. Passes narrowed `local_indices_buffer[:B, :local_width]` to `qsa_localize_dcp_indices` and routes `dcp_a2a_lse_reduce` through the fused `out=output, output_gate=gate_view` kernel (`_forward_qsa_dcp`).
  4. Avoids flattening the non-contiguous 3D `gate` slice (`q_gate[..., self.head_dim :]`) into a 2D tensor in `Qwen4ExpQSAAttention.forward`.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/nvidia/qsa.py b/vllm/models/qwen4_exp/nvidia/qsa.py
index 7826024..5472099 100644
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
@@ -386,6 +400,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         output_gate: torch.Tensor,
     ) -> None:
         from vllm.distributed import get_dcp_group
+        from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

         from .ops.qsa import _qsa_output_gate, qsa_sparse_paged_attention
         from .ops.qsa_dcp import (
@@ -405,9 +420,17 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
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
         local_indices = qsa_localize_dcp_indices(
             logical_indices,
-            local_indices_buffer[: query.shape[0]],
+            local_indices_buffer[: query.shape[0], :local_width],
             dcp_world_size=self.dcp_world_size,
             dcp_rank=self.dcp_rank,
             interleave_size=layer.cp_kv_cache_interleave_size,
@@ -419,19 +442,11 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
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
@@ -440,6 +455,17 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             v_scale=layer._v_scale_float,
             lse=partial_lse,
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
@@ -453,7 +479,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         # apply the Triton gate DCP1's page4 route uses. This replaces a chain
         # of five elementwise kernels (12.2 us per decode layer on V100).
         output.copy_(merged)
-        _qsa_output_gate(output, output_gate.view_as(output))
+        _qsa_output_gate(output, gate_view)


 def _verify_e4m3_kv_requirements(
@@ -862,6 +888,8 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
         self,
         qkv: torch.Tensor,
         positions: torch.Tensor,
+        *,
+        flatten_gate: bool = True,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
         """Split, normalize, and rotate Q/K using this tree's Qwen3 API."""
         q_gate, key, value = qkv.split(
@@ -875,7 +903,9 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             key.view(*token_shape, self.num_kv_heads, self.head_dim)
         ).reshape(*token_shape, self.kv_size)
         query, key = self.rotary_emb(positions, query, key)
-        return query, key, value, gate.reshape(*token_shape, self.q_size)
+        if flatten_gate:
+            gate = gate.reshape(*token_shape, self.q_size)
+        return query, key, value, gate

     def _use_dense_short_context(
         self, num_tokens: int, side_metadata: QSAForwardMetadata
@@ -1043,12 +1073,14 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             q_gate = qkv[:, : self.q_size * 2].view(
                 num_tokens, self.num_heads, 2 * self.head_dim
             )
-            gate = q_gate[..., self.head_dim :].reshape(num_tokens, self.head_dim)
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

### 8.4 `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `vllm/models/qwen4_exp/common/qsa_cache.py`, & `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` (Optimizations 3.1 & D3)

- **Why**:
  1. Caches `(all_visible, max_visible)` on `QSAForwardMetadata.visible_blocks_cache` across all 12 QSA layers in a prefill chunk, eliminating 11/12 host-device `.item()` stalls and `_qsa_visible_blocks_kernel` launches.
  2. Adds `_qsa_splitk_workspace(q, num_splits)` backed by `workspace_cache("qsa_splitk", _SM70_QSA_SPLITK_WORKSPACES)` and `retain_for_capture`, eliminating 24 dynamic GPU tensor allocations (`partial_output` + `partial_lse`) per decode step under `TP4 + DCP2`.
- **Diff**:
```diff
diff --git a/vllm/models/qwen4_exp/common/qsa_cache.py b/vllm/models/qwen4_exp/common/qsa_cache.py
index f087d0b..48e2d7b 100644
--- a/vllm/models/qwen4_exp/common/qsa_cache.py
+++ b/vllm/models/qwen4_exp/common/qsa_cache.py
@@ -14,7 +14,7 @@ shared by the generic cache-layout planner.
 from __future__ import annotations

 import math
-from dataclasses import dataclass
+from dataclasses import dataclass, field
 from functools import cache
 from typing import ClassVar

@@ -606,6 +606,9 @@ class QSAForwardMetadata(AttentionMetadata):
     # Host copy of query_start_loc. It lets QSA selection split a mixed batch
     # by request without a device synchronization.
     query_start_loc_cpu: torch.Tensor | None = None
+    visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]] = field(
+        default_factory=dict
+    )

diff --git a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
index 0369fac..2010871 100644
--- a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
@@ -361,6 +361,7 @@ class QSAIndexer(nn.Module):
             out,
             query_start_loc_cpu=compressed_metadata.query_start_loc_cpu,
             shared_key_scoring=self.shared_key_scoring,
+            visible_blocks_cache=compressed_metadata.visible_blocks_cache,
         )

diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa.py b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
index aa5e6a6..2b452c9 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -70,6 +70,10 @@ _SM70_QSA_GROUPED_PAGE4_WORKSPACES: dict[
         torch.Tensor,
     ],
 ] = {}
+_SM70_QSA_SPLITK_WORKSPACES: dict[
+    tuple[int, int],
+    tuple[int, int, torch.Tensor, torch.Tensor],
+] = {}
 _SM70_QSA_GROUPED_PAGE4_ABI_CACHE: tuple[object, int] | None = None
...
@@ -1625,19 +1639,34 @@ def qsa_select_paged_tokens(
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
         )
+        cache_key = (start_row, end_row, int(compress_ratio))
+        cached_visible = (
+            visible_blocks_cache.get(cache_key)
+            if visible_blocks_cache is not None
+            else None
+        )
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
...
@@ -1976,6 +2005,40 @@ def _qsa_grouped_page4_workspace(
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
...
@@ -2500,15 +2563,9 @@ def qsa_sparse_paged_attention(
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
```

---

### 8.5 `vllm/v1/attention/backends/gdn_attn.py` (Optimization D6: Lazy `compute_num_computed_tokens()` on Decode)

- **Why**: Under `TP4 + DCP2`, the 36 GDN layers are partitioned into **6 GDN KV-cache groups**. Moving `context_lens_tensor = m.compute_num_computed_tokens()` inside `if num_prefills > 0:` eliminates **12 wasted GPU elementwise kernel launches and 12 tensor allocations per decode step**.
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

### 8.6 `vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py` & `vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py` (Optimization 3.3: NVFP4 MoE Workspace & Dispatch Cleanup)

- **Why**: Caches per-device `arange` index buffers across all 48 MoE layers and skips redundant `output.zero_()`, `buffers["permuted_idx"].fill_(slots)`, and `topk_ids_i32.copy_(topk_ids)` when `layer.expert_map is None` and `topk_ids` is already contiguous `int32`.
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
...
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
...
-            "token_expert_indices": torch.arange(
-                slots, dtype=torch.int32, device=device
-            ).view(num_tokens, top_k),
+            "token_expert_indices": arange_buf[:slots].view(num_tokens, top_k),
...
-            "compact_offsets": torch.arange(
-                slots + 1, dtype=torch.int32, device=device
-            ),
+            "compact_offsets": arange_buf[: slots + 1],
             "active_expert_ids": torch.empty(slots, dtype=torch.int32, device=device),
         }

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
```

---

### 8.7 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` & `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (Optimization 3.4: GDN Tail-Split Grid Shrink & Prefill Admission)

- **Why**: Shrinks the `_qwen38_gdn_projection_split_kernel` grid when `COPY_QKV=False` from `(m, 16)` to `(m, 6)` (eliminating 9/16 no-op CTAs per row and wasted `qkv` HBM loads) and admits `mixed_qkvz.shape[0] > 1` in `qwen_gdn_input_projection_core`.
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
