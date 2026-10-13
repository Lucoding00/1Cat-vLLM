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

## 5. 8×V100 Deployment & `TP8` Feasibility Analysis

### 5.1 Why Pure `TP8` Fails on Startup Today

1. **`ModelOptNvFp4SM70MoEMethod` Hardcoded TP Gate (`vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py:594-624`)**:
   - `_SUPPORTED_TP_SIZES = (1, 2, 4)` explicitly raises `NotImplementedError` when `tp_size == 8`.
2. **MoE `intermediate_size = 640` Sharding Alignment**:
   - At `TP8`, `local_intermediate_size = 640 / 8 = 80` (`gate_up = 160`).
   - In `nvfp4_sm70_moe.py`, `80 % 16 == 0` satisfies the 16-column tile alignment, so removing the `tp_size == 8` gate allows NVFP4 MoE to run at `TP8`.
   - However, in `gguf_turbomind_moe.py:619,712` (`80 % 64 != 0` and `80 % 32 != 0`) and `gguf_moe.py:106` (`160 % 256 != 0`), `TP8` violates the kernel tile alignment and cannot run without weight padding.
3. **QSA KV Heads (`num_key_value_heads = 2`) & TP4-Specialized Fastpaths**:
   - At `TP8`, KV heads must be replicated (`4` ranks per KV head), and Q heads drop to `24 / 8 = 3` per rank.
   - All SM70 TP4-specialized fastpaths (`flash_decode_paged.cu` XQA `H=6, D=256`, `sm70_fp16_gemv.py` GDN `H_v=12`, `sm70_ring.py` / `sm70_hc_ll.py` 4-rank collectives) fall back to generic Triton/NCCL paths, and 8-GPU ring AllReduce latency (`~4.5 ms/round`) exceeds the compute savings on V100.

### 5.2 Recommended 8×V100 Topologies

| 8×V100 Topology | Configuration | Best Use Case | Performance vs. 4×V100 `TP4` |
| :--- | :--- | :--- | :--- |
| **`TP4 + DP2` (Recommended for Throughput)** | 2 independent `TP4` instances (`GPU 0–3` and `GPU 4–7`) | Online serving / high concurrency / batch processing | **2.0× aggregate Prefill & Decode throughput** (`~14,000+ tok/s` prefill, `~540+ tok/s` two-stream MTP4 decode) while preserving 100% of `TP4` SM70 custom kernels |
| **`TP4 + DCP2` (Recommended for Ultra-Long Context)** | `--tensor-parallel-size 4 --decode-context-parallel-size 2` | Single-request 512K–1M+ ultra-long context | **2.0× KV cache capacity** (~718K FP16 / ~1.18M E4M3 KV tokens) while keeping `TP4` local head geometry (`6` QSA Q heads, `12` GDN V heads) and ~97% of `TP4` decode speed |

---

## 6. Verification & Benchmark Commands on V100

```bash
# 1. Run unit tests covering QSA, GDN tail split, NVFP4 MoE workspace, and baseline contract
pytest -q \
  tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_sm70_gdn_projection_split.py \
  tests/kernels/test_sm70_gdn_projection_tails.py \
  tests/kernels/test_sm70_qsa_prep.py \
  tests/quantization/test_sm70_fp4_mainflow.py \
  tests/quantization/test_sm70_nvfp4_integration.py \
  tests/quantization/test_sm70_warmup.py \
  tests/benchmarks/test_sm70_qwen38_baseline.py

# 2. Run reproducible TP4 + MTP4 benchmark (verifies bit-exact greedy token IDs)
python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --mtp 4 \
  --output /tmp/qwen38_mtp4_opt.json

# 3. Run reproducible TP4 + No-MTP benchmark
python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --mtp 0 \
  --output /tmp/qwen38_nomtp_opt.json
```
