# Commit `048ac68` 专项说明文档：`indexer_qsa` 导入修复、`IDENTITY_BLOCKS` 多请求边界防护与 `fp8_e4m3` 单位 Scale `"duct"` 乱码排查指南

> **对应提交**：`048ac687f233bb4ea8c5730b0c0ed3830eefea14`（分支 `arena/5368f936-1cat-vllm`）  
> **提交标题**：`Fix indexer_qsa import, guard IDENTITY_BLOCKS ordering, and document E4M3 unit-scale precision collapse`

---

## 1. 本次提交内容总览

本次提交（`048ac68`）共包含 **2 项代码修复** 与 **1 项精度坍缩（输出重复 `"duct"` 乱码）根因排查及生产启动配置更新**：

| 序号 | 模块 / 文件 | 变更类型 | 解决的问题 |
| :--- | :--- | :--- | :--- |
| **1** | `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` | **Bug 修复** | 修复 `D22` 短上下文 `skip_query_prep` 路径中 `use_sm70_decode_graph_semantics` 的导入模块路径错误（`vllm.config.execution_policy` $\rightarrow$ `vllm.compilation.sm70_decode_graph`），消除 `ImportError` |
| **2** | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | **边界防护修复** | 将 `qsa_select_paged_tokens` 中的 `IDENTITY_BLOCKS=True`（`max_seq_len // compress_ratio <= block_topk`）快速路径移至 `_qsa_select_by_request` 之前，防止多请求短上下文 Batch（`num_requests > 1, rows >= 64, max_seq_len <= 2048`）在跳过 `q` 计算后误入 `_qsa_select_by_request` 读取未初始化的 `q` 缓冲区 |
| **3** | `docs/design/sm70_qwen38_master_summary.md`<br>`docs/design/sm70_qwen38_dp2_tp4_dcp2_fix.md`<br>`docs/design/sm70_qwen38_capability_breakthroughs.md` | **精度排查与启动脚本更新** | 查明未校准 `RadixArk/Qwen3.8-Flash-Next-NVFP4` 直接开启 `--kv-cache-dtype fp8_e4m3` 时因回退到单位缩放因子（`scale = 1.0`）导致输出全部坍缩为 `"duct"` 乱码的根因；更新默认生产启动脚本为 `--kv-cache-dtype auto`（原生 FP16），并给出离线校准开启 `fp8_e4m3` 的完整操作步骤 |

---

## 2. 代码修复详解与完整 Diff

### 2.1 修复 1：`vllm/models/qwen4_exp/nvidia/indexer_qsa.py` 导入路径修正

- **触发条件**：
  当序列长度满足 `max_seq_len <= 2048`（即 `all_blocks_fit = True`，所有压缩块均在 `block_topk = 128` 窗口内）时，`QSAIndexer.forward` 触发 `D22`（`skip_query_prep = True`）优化，跳过 `index_qk_proj` 的前 `q_dim = 2048` 行投影，仅对后 `k_dim = 64` 行执行投影。
- **错误原因**：
  `use_sm70_decode_graph_semantics` 定义在 `vllm/compilation/sm70_decode_graph.py` 中，而 `indexer_qsa.py:326` 误从 `vllm.config.execution_policy` 导入，导致触发 `ImportError: cannot import name 'use_sm70_decode_graph_semantics' from 'vllm.config.execution_policy'`。
- **代码修改 Diff**：

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
index b026a79..0349d37 100644
--- a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
@@ -323,7 +323,7 @@ class QSAIndexer(nn.Module):
                     self.index_qk_proj, "_sm70_qwen4_exp_online_qpn8", False
                 )
             ):
-                from vllm.config.execution_policy import (
+                from vllm.compilation.sm70_decode_graph import (
                     use_sm70_decode_graph_semantics,
                 )
 
```

---

### 2.2 修复 2：`vllm/models/qwen4_exp/nvidia/ops/qsa.py` 中 `IDENTITY_BLOCKS=True` 与 `_qsa_select_by_request` 顺序防护

- **触发条件与风险链路**：
  1. 在 `indexer_qsa.py` 中，当 `all_blocks_fit` 为 `True`（`max_seq_len // compress_ratio <= block_topk`，即 `max_seq_len <= 2048`）时，`skip_query_prep = True` 会跳过 `qsa_pre_indexer` 中对 `q`（`self.q_buffer[:num_tokens]`）的计算与写入，因为此时所有可见压缩块都不超过 `block_topk = 128`，无需计算 `Q @ K^T` 分数即可直接按恒等块号 `0..block_topk-1` 展开选中的 Token 索引。
  2. 但在 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` 的 `qsa_select_paged_tokens` 中，原本先检查了：
     ```python
     if query_start_loc_cpu is not None and _qsa_select_by_request(...):
         return out
     ```
     然后才在后面检查 `max_seq_len // compress_ratio <= block_topk`（`IDENTITY_BLOCKS=True`）。
  3. 当一个 Batch 同时包含多个短请求（`num_requests > 1` 且总 Token 数 `rows >= 64`，同时 `max_seq_len <= 2048`）时，`_qsa_select_by_request` 会命中并在内部递归调用 `qsa_select_paged_tokens`（且递归调用未透传 `max_seq_len`），进而对未初始化的 `q_buffer` 执行 `_qsa_mqa_cublas` 打分！
- **修复方案**：
  将 `IDENTITY_BLOCKS=True` 快速路径移到 `_qsa_select_by_request` **之前**。只要整个 Batch 的 `max_seq_len // compress_ratio <= block_topk`，无论单请求还是多请求，直接通过单次 `_expand_qsa_indices_kernel(..., IDENTITY_BLOCKS=True)` 完成全部行的索引展开并返回，既避免了按请求切分循环，也彻底杜绝了访问未初始化 `q` 的隐患。
- **代码修改 Diff**：

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa.py b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
index 968e4a2..5e90388 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -1771,23 +1771,6 @@ def qsa_select_paged_tokens(
         raise ValueError("QSA selection output has an invalid shape")
     if not rows:
         return out
-    if query_start_loc_cpu is not None and _qsa_select_by_request(
-        q,
-        k_cache,
-        page_table,
-        token_to_req,
-        query_positions,
-        sequence_lengths,
-        token_topk,
-        compress_ratio,
-        out,
-        query_start_loc_cpu,
-        visible_blocks_cache=visible_blocks_cache,
-    ):
-        return out
-
-    capacity_columns = page_table.shape[1] * k_cache.shape[1]
-    score_columns = capacity_columns
     block_topk = token_topk // compress_ratio
     if (
         max_seq_len is not None
@@ -1831,6 +1814,23 @@ def qsa_select_paged_tokens(
                 num_warps=4,
             )
             return out
+    if query_start_loc_cpu is not None and _qsa_select_by_request(
+        q,
+        k_cache,
+        page_table,
+        token_to_req,
+        query_positions,
+        sequence_lengths,
+        token_topk,
+        compress_ratio,
+        out,
+        query_start_loc_cpu,
+        visible_blocks_cache=visible_blocks_cache,
+    ):
+        return out
+
+    capacity_columns = page_table.shape[1] * k_cache.shape[1]
+    score_columns = capacity_columns
     contiguous_keys: torch.Tensor | None = None
     key_valid: torch.Tensor | None = None
     all_visible: torch.Tensor | None = None
```

---

### 2.3 一键应用本次代码修复的 `git apply` 命令

```bash
cat << 'EOF' | git apply -
diff --git a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
--- a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
@@ -323,7 +323,7 @@ class QSAIndexer(nn.Module):
                     self.index_qk_proj, "_sm70_qwen4_exp_online_qpn8", False
                 )
             ):
-                from vllm.config.execution_policy import (
+                from vllm.compilation.sm70_decode_graph import (
                     use_sm70_decode_graph_semantics,
                 )
 
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa.py b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
@@ -1771,23 +1771,6 @@ def qsa_select_paged_tokens(
         raise ValueError("QSA selection output has an invalid shape")
     if not rows:
         return out
-    if query_start_loc_cpu is not None and _qsa_select_by_request(
-        q,
-        k_cache,
-        page_table,
-        token_to_req,
-        query_positions,
-        sequence_lengths,
-        token_topk,
-        compress_ratio,
-        out,
-        query_start_loc_cpu,
-        visible_blocks_cache=visible_blocks_cache,
-    ):
-        return out
-
-    capacity_columns = page_table.shape[1] * k_cache.shape[1]
-    score_columns = capacity_columns
     block_topk = token_topk // compress_ratio
     if (
         max_seq_len is not None
@@ -1831,6 +1814,23 @@ def qsa_select_paged_tokens(
                 num_warps=4,
             )
             return out
+    if query_start_loc_cpu is not None and _qsa_select_by_request(
+        q,
+        k_cache,
+        page_table,
+        token_to_req,
+        query_positions,
+        sequence_lengths,
+        token_topk,
+        compress_ratio,
+        out,
+        query_start_loc_cpu,
+        visible_blocks_cache=visible_blocks_cache,
+    ):
+        return out
+
+    capacity_columns = page_table.shape[1] * k_cache.shape[1]
+    score_columns = capacity_columns
     contiguous_keys: torch.Tensor | None = None
     key_valid: torch.Tensor | None = None
     all_visible: torch.Tensor | None = None
EOF
```

---

## 3. 为什么未校准的 `RadixArk/Qwen3.8-Flash-Next-NVFP4` 开启 `--kv-cache-dtype fp8_e4m3` 会输出重复 `"duct"` 乱码？

### 3.1 启动日志中的关键线索

当对原始 `RadixArk/Qwen3.8-Flash-Next-NVFP4` 权重直接传入 `--kv-cache-dtype fp8_e4m3` 时，启动日志会打印以下两行告警：

```text
WARNING: No KV-cache quantization metadata found in hf_quant_config.json. KV-cache scales will not be loaded and default to 1.0, which may cause accuracy degradation.
WARNING: QSA E4M3 scale overlay is incomplete: 0/24 local K/V scales loaded. Falling back to unit scales for missing tensors: language_model.model.layers.3.self_attn.k_scale, language_model.model.layers.3.self_attn.v_scale, ...
```

### 3.2 精度坍缩的数学与代码链路

1. **原始 Checkpoint 不含 QSA `k_scale` / `v_scale`**：
   - `RadixArk/Qwen3.8-Flash-Next-NVFP4` 的 `hf_quant_config.json` 将 `*.self_attn.*` 和 `*.linear_attn.*` 列在 `exclude_modules` 中，权重中没有任何 `layers.{3,7,11,15,19,23,27,31,35,39,43,47}.self_attn.{k,v}_scale` 张量（`0/24`）。
2. **`_finalize_qsa_e4m3_scale_load` 回退到单位缩放因子 `scale = 1.0`**：
   - 在 `vllm/models/qwen4_exp/nvidia/model.py:189-199` 中，当检测到 `0/24` 个 scale 被加载时，会调用 `module.adopt_default_kv_scales()` 将 `_k_scale_float` 和 `_v_scale_float` 全部设为 **`1.0`**。
3. **本仓库 QSA `fp8_e4m3` 的缩放契约为 `e4m3fn(x / scale)`，其中 `scale = max_abs / 448.0 ≈ 0.009 ~ 0.034`**：
   - 参见 `tools/qwen4_exp/materialize_qsa_scale_overlay.py`、`tests/models/qwen4_exp/test_e4m3_mtp_envelope.py` 以及 `tools/qwen4_exp/README.md`：
     $$\text{scale} = \frac{\max |x|}{448.0} \approx 0.009 \sim 0.034$$
   - 经过 K-norm/RoPE 后的 QSA Key/Value 幅值通常在 `4.0 ~ 15.0` 之间。正常校准时，先除以 `scale ≈ 0.02` 将数值拉伸到 `[-448, 448]`，从而充分利用 FP8 E4M3 的全部指数档位；读取时再乘回 `scale ≈ 0.02`。
   - 一旦回退到单位 `scale = 1.0`（`tools/qwen4_exp/README.md:32` 明确将其作为 `--unit-scale-negative-control` 负向对照），原本应利用 `[-448, 448]` 动态范围的 K/V 被直接按原值存储在 `[-15, 15]` 区间内，**浪费了 30～100 倍的动态范围**，且经过 12 层 QSA 注意力逐层放大后，主模型与 MTP 草稿模型的 Logits 彻底失真，解码输出全部坍缩为同一个高频乱码 Token **`"duct"`**。

---

## 4. 生产环境解决方案与启动脚本

### 4.1 方案 A（推荐开箱即用）：使用 `--kv-cache-dtype auto`（原生 FP16 KV Cache）

#### 为什么在 `TP4 + DCP2` 下 `--kv-cache-dtype auto` 依然能跑满 256K 上下文？
- `Qwen3.8-Flash-Next` 共 48 层，其中 **36 层是 GDN 线性注意力**（固定状态大小，不随序列长度增长），**只有 12 层是 QSA 稀疏注意力**。
- 在每层 QSA 中，除了主 K/V Cache 外，还有一个 **恒为 FP16 的 Indexer Cache（每 Token 每卡 `3.75 KB`）**。
- 在 `TP4 + DCP2` 下，每 Token 在单张 GPU 上的 KV 显存占用为：
  - `fp8_e4m3`：`3.0 KB (主 KV) + 3.75 KB (Indexer) = 6.75 KB/token/GPU` $\rightarrow$ 实测容量 **`377,795 tokens`**。
  - `auto`（`float16`）：`6.0 KB (主 KV) + 3.75 KB (Indexer) = 9.75 KB/token/GPU` $\rightarrow$ 在 `--gpu-memory-utilization 0.90` 下容量为：
    $$377,795 \times \frac{6.75}{9.75} \approx \mathbf{261,550\text{ tokens}}$$
    若将 `--gpu-memory-utilization` 设为 `0.91` 或 `0.92`，单副本 KV 容量即达 **`268,000+ tokens`**，完全满足 **256K（`262,144`）** 上下文长度需求！
- **额外性能收益**：在 `vllm/models/qwen4_exp/nvidia/qsa.py:886-895` 中，`self._sm70_qsa_prep`（`qsa_prep_sm70_out` C++ 融合算子 + `D14` 异步 DCP Query AllGather 通信重叠）仅在 `cache_dtype in ("auto", "float16")` 时激活。使用 `auto` 可以同时获得原生 FP16 精度与 `_sm70_qsa_prep` 加速。

#### `8 × V100 (32GB)` 单端口 `DP2 × TP4 + DCP2 + MTP4 + 256K` 完整启动脚本（`--kv-cache-dtype auto`）

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/ubuntukv/lw/data/model/RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 关闭默认 Disk-Mmap PLE，启用 Pinned-Host UVA（8 Worker × 10 GiB = 80 GiB Host RAM）
export VLLM_PLE_CPU_OFFLOAD=0
export VLLM_PLE_DISK_OFFLOAD=0
export VLLM_SM70_QWEN38_HYBRID_PLE=0
export VLLM_QWEN4EXP_PLE_HOST_GIB="${VLLM_QWEN4EXP_PLE_HOST_GIB:-10}"

# 清理之前 fp8_e4m3 启动时生成的编译缓存，避免跨 KV dtype 复用
rm -rf /tmp/vllm-compile-cache-dp*

exec vllm serve "${MODEL_PATH}" \
  --served-model-name Qwen3.8-Flash-Next \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --data-parallel-size 2 \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --dtype half \
  --kv-cache-dtype auto \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.91 \
  --no-async-scheduling \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --no-language-model-only \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --kernel-config '{"ple_disk_cascade":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --seed 0
```

---

### 4.2 方案 B（需要 `377,795 tokens` 容量时）：合入离线校准包 `model-kvscales.safetensors` 后开启 `fp8_e4m3`

如果需要开启 `--kv-cache-dtype fp8_e4m3`，必须先为模型目录生成并合并 `model-kvscales.safetensors`（包含 12 层 QSA 共 24 个校准后的 `k_scale` / `v_scale` 张量）：

1. **方式 1：若机器上已有校准清单文件**（例如仓库测试代码 `tests/models/qwen4_exp/test_qsa_kv_calibration_mtp.py:137` 中引用的 `/home/l/models/qwen38-flash-next-e4m3-kvscales/kvscales-manifest.json`）：
   运行仓库内置的 `materialize_qsa_scale_overlay.py`（秒级完成，不修改原始权重分片，仅生成 `model-kvscales.safetensors` 并在 `model.safetensors.index.json` 中注册这 24 个张量）：
   ```bash
   python3 tools/qwen4_exp/materialize_qsa_scale_overlay.py \
     --manifest /home/l/models/qwen38-flash-next-e4m3-kvscales/kvscales-manifest.json \
     --base-index "${MODEL_PATH}/model.safetensors.index.json" \
     --output-dir "${MODEL_PATH}"
   ```
2. **方式 2：若机器上没有现成校准清单文件**：
   使用仓库内置的离线校准工具 `tools/qwen4_exp/qsa_kv_calibration.py` 采集校准样本统计每层 K/V 的 `max_abs / 448.0` 并生成 `model-kvscales.safetensors`：
   ```bash
   python3 tools/qwen4_exp/qsa_kv_calibration.py \
     --model "${MODEL_PATH}" \
     --dataset /path/to/calibration_texts.jsonl \
     --output-dir "${MODEL_PATH}" \
     --tp-size 4 \
     --include-mtp
   ```
3. **合入后启动 `fp8_e4m3`**：
   建议同时设置 `export VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1`（强制校验 `24/24` 个 scale 全部从权重加载成功，一旦缺失立即报错拒绝启动，防止静默退化为单位 `scale=1.0`）：
   ```bash
   export VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1
   rm -rf /tmp/vllm-compile-cache-dp*
   ```
   并在启动参数中使用：
   ```bash
   --kv-cache-dtype fp8_e4m3 \
   --speculative-config '{"method":"mtp","num_speculative_tokens":4,"kv_cache_dtype":"float16"}'
   ```
