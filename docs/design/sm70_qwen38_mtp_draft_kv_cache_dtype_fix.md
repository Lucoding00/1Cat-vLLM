# `Qwen3.8-Flash-Next` 主模型 `fp8_e4m3` + MTP 草稿 `float16` KV Cache 类型继承错误（`Loaded 0/2`）修复方案（`D27`）

---

## 1. 一句话总结与报错现象

### 1.1 一句话总结
**MTP 草稿模型的 KV cache 数据类型（`dtype`）继承错了，被设成了主模型的 `fp8_e4m3`，而草稿层没有对应的 FP8 缩放因子（`scale`），导致加载权重时报 `"Loaded 0/2"` 直接拒绝启动。**

### 1.2 报错日志现象
当使用 `--kv-cache-dtype fp8_e4m3` 并通过 `--speculative-config '{"method":"mtp","num_speculative_tokens":4,"kv_cache_dtype":"float16"}'` 启动未包含草稿层校准 scale 的 `Qwen3.8-Flash-Next` 权重时，启动阶段会在 `Qwen4ExpMTP.load_weights` 抛出如下异常：

```text
ValueError: QSA E4M3 speculative draft scale overlay is incomplete: loaded 0/2 scale tensors.
Missing: model.layers.0.self_attn.k_scale, model.layers.0.self_attn.v_scale
```

---

## 2. 细节根因链路（5 步完整追踪）

1. **启动参数显式分离了主模型与草稿模型的 KV Cache 类型**：
   - 主模型 12 层 QSA 指定使用 `--kv-cache-dtype fp8_e4m3`（显存占用与带宽减半）；
   - 草稿模型（MTP 1 层 QSA）明确指定使用 `float16`（`--speculative-config '{"method":"mtp","num_speculative_tokens":4,"kv_cache_dtype":"float16"}'`）。
2. **`mtp.py` 中 `_make_draft_vllm_config()` 漏传了 `speculative_config.kv_cache_dtype`**：
   - 在 `vllm/models/qwen4_exp/nvidia/mtp.py`（以及 `vllm/models/qwen4_exp/amd/mtp.py`）中，`Qwen4ExpMultiTokenPredictor.__init__` 调用 `_make_draft_vllm_config(vllm_config, self.mtp_start_layer_idx)` 构造草稿模型的专属 `VllmConfig`。
   - 原实现直接调用了：
     ```python
     draft_vllm_config = replace(
         vllm_config,
         model_config=speculative_config.draft_model_config,
         is_speculative_draft=True,
     )
     ```
   - 该调用从主配置 `vllm_config` 整体继承了 `cache_config`，**完全漏掉了 `speculative_config.kv_cache_dtype="float16"` 这一项覆盖逻辑**。
3. **草稿模型与草稿 QSA 层拿到的 `cache_dtype` 仍是 `fp8_e4m3`**：
   - `Qwen4ExpMultiTokenPredictor` 记录了 `self._kv_cache_dtype = draft_vllm_config.cache_config.cache_dtype`（值为 `"fp8_e4m3"`）；
   - 草稿层的 `Qwen4ExpQSAAttention` 也以 `cache_dtype="fp8_e4m3"` 初始化，不仅注册了待校验的 `k_scale = -1.0` / `v_scale = -1.0` 参数，还被迫关闭了仅对 `float16` 生效的 `self._sm70_qsa_prep` 融合预处理算子。
4. **加载权重时触发 `_finalize_qsa_e4m3_scale_load` 强制校验**：
   - 在 `Qwen4ExpMTP.load_weights()`（`vllm/models/qwen4_exp/nvidia/mtp.py:828`）末尾调用了：
     ```python
     _finalize_qsa_e4m3_scale_load(
         self,
         loaded_weights,
         self.model._kv_cache_dtype,
         require_calibrated_speculative_draft=True,
     )
     ```
   - 因为 `self.model._kv_cache_dtype` 被错误设成了 `"fp8_e4m3"`，该函数开启严格校验，要求草稿层必须加载 K/V 的 E4M3 缩放因子（`mtp.layers.0.self_attn.k_scale` 与 `mtp.layers.0.self_attn.v_scale`）。
5. **Checkpoint 中无草稿层 K/V scale，直接拒绝启动**：
   - 模型权重文件中仅校准了主模型或不含草稿层的 `k_scale`/`v_scale`，草稿层实际加载数为 `0/2`，从而触发 `ValueError: QSA E4M3 speculative draft scale overlay is incomplete: loaded 0/2 scale tensors.`。

---

## 3. 代码修复方案（`D27`）

### 3.1 修改文件清单

| 序号 | 文件路径 | 修改说明 |
| :--- | :--- | :--- |
| 1 | `vllm/models/qwen4_exp/nvidia/mtp.py` | 在 `_make_draft_vllm_config()` 中读取 `speculative_config.kv_cache_dtype`，若已指定则将其覆盖至 `draft_vllm_config.cache_config.cache_dtype`（不影响主模型 `vllm_config.cache_config`） |
| 2 | `vllm/models/qwen4_exp/amd/mtp.py` | 同步修复 AMD 后端 `_make_draft_vllm_config()` 的 `speculative_config.kv_cache_dtype` 继承逻辑 |
| 3 | `tests/models/qwen4_exp/test_e4m3_mtp.py` | 新增回归单测 `test_make_draft_vllm_config_honors_speculative_kv_cache_dtype`，验证草稿配置正确设为 `"float16"` 且主配置保持 `"fp8_e4m3"` |

---

### 3.2 各文件详细 Diff

#### (1) `vllm/models/qwen4_exp/nvidia/mtp.py`

```diff
--- a/vllm/models/qwen4_exp/nvidia/mtp.py
+++ b/vllm/models/qwen4_exp/nvidia/mtp.py
@@ -15,6 +15,9 @@
 """
 
 from collections.abc import Iterable
+import copy
+from dataclasses import is_dataclass
+from typing import Any
 
 import regex as re
 import torch
@@ -241,10 +244,25 @@ def _make_draft_vllm_config(
                 _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
             )
 
+    replace_kwargs: dict[str, Any] = {
+        "model_config": speculative_config.draft_model_config,
+        "is_speculative_draft": True,
+    }
+    draft_cache_dtype = getattr(speculative_config, "kv_cache_dtype", None)
+    draft_cache_config = getattr(vllm_config, "cache_config", None)
+    if draft_cache_dtype is not None and draft_cache_config is not None:
+        if is_dataclass(draft_cache_config):
+            draft_cache_config = replace(
+                draft_cache_config,
+                cache_dtype=draft_cache_dtype,
+            )
+        else:
+            draft_cache_config = copy.copy(draft_cache_config)
+            draft_cache_config.cache_dtype = draft_cache_dtype
+        replace_kwargs["cache_config"] = draft_cache_config
+
     draft_vllm_config = replace(
         vllm_config,
-        model_config=speculative_config.draft_model_config,
-        is_speculative_draft=True,
+        **replace_kwargs,
     )
     # VllmConfig post-init derives the target quant config, so restore the
     # independently resolved draft quant config after replacement.
```

---

#### (2) `vllm/models/qwen4_exp/amd/mtp.py`

```diff
--- a/vllm/models/qwen4_exp/amd/mtp.py
+++ b/vllm/models/qwen4_exp/amd/mtp.py
@@ -15,6 +15,9 @@
 """
 
 from collections.abc import Iterable
+import copy
+from dataclasses import is_dataclass
+from typing import Any
 
 import regex as re
 import torch
@@ -172,9 +175,24 @@ def _make_draft_vllm_config(
                 _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
             )
 
+    replace_kwargs: dict[str, Any] = {
+        "model_config": speculative_config.draft_model_config,
+    }
+    draft_cache_dtype = getattr(speculative_config, "kv_cache_dtype", None)
+    draft_cache_config = getattr(vllm_config, "cache_config", None)
+    if draft_cache_dtype is not None and draft_cache_config is not None:
+        if is_dataclass(draft_cache_config):
+            draft_cache_config = replace(
+                draft_cache_config,
+                cache_dtype=draft_cache_dtype,
+            )
+        else:
+            draft_cache_config = copy.copy(draft_cache_config)
+            draft_cache_config.cache_dtype = draft_cache_dtype
+        replace_kwargs["cache_config"] = draft_cache_config
+
     draft_vllm_config = replace(
         vllm_config,
-        model_config=speculative_config.draft_model_config,
+        **replace_kwargs,
     )
     # VllmConfig post-init derives the target quant config, so restore the
     # independently resolved draft quant config after replacement.
```

---

#### (3) `tests/models/qwen4_exp/test_e4m3_mtp.py`

```diff
--- a/tests/models/qwen4_exp/test_e4m3_mtp.py
+++ b/tests/models/qwen4_exp/test_e4m3_mtp.py
@@ -247,3 +247,38 @@ def test_automatic_e4m3_refuses_unit_scale_fallback(monkeypatch):
         require_calibrated_target=True,
     )
     assert stub._qsa_kv_scales_finalized is True
+
+
+def test_make_draft_vllm_config_honors_speculative_kv_cache_dtype(monkeypatch):
+    from dataclasses import dataclass
+
+    from vllm.models.qwen4_exp.nvidia import mtp
+
+    @dataclass
+    class _CacheCfg:
+        cache_dtype: str = E4M3
+
+    @dataclass
+    class _VllmCfg:
+        model_config: object
+        speculative_config: object
+        cache_config: _CacheCfg
+        quant_config: object = None
+        is_speculative_draft: bool = False
+
+    monkeypatch.setattr(mtp, "get_draft_quant_config", lambda _: None)
+    target_model = SimpleNamespace(architectures=["Qwen4ExpForCausalLM"])
+    draft_model = SimpleNamespace(architectures=["Qwen4ExpMTP"])
+    spec_cfg = SimpleNamespace(
+        draft_model_config=draft_model,
+        kv_cache_dtype="float16",
+    )
+    target_cfg = _VllmCfg(
+        model_config=target_model,
+        speculative_config=spec_cfg,
+        cache_config=_CacheCfg(cache_dtype=E4M3),
+    )
+
+    draft_cfg = mtp._make_draft_vllm_config(target_cfg, 48)
+    assert draft_cfg.cache_config.cache_dtype == "float16"
+    assert target_cfg.cache_config.cache_dtype == E4M3
+    assert draft_cfg.is_speculative_draft is True
```

---

### 3.3 修改后的 `_make_draft_vllm_config` 完整函数代码（`vllm/models/qwen4_exp/nvidia/mtp.py`）

```python
def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )

    replace_kwargs: dict[str, Any] = {
        "model_config": speculative_config.draft_model_config,
        "is_speculative_draft": True,
    }
    draft_cache_dtype = getattr(speculative_config, "kv_cache_dtype", None)
    draft_cache_config = getattr(vllm_config, "cache_config", None)
    if draft_cache_dtype is not None and draft_cache_config is not None:
        if is_dataclass(draft_cache_config):
            draft_cache_config = replace(
                draft_cache_config,
                cache_dtype=draft_cache_dtype,
            )
        else:
            draft_cache_config = copy.copy(draft_cache_config)
            draft_cache_config.cache_dtype = draft_cache_dtype
        replace_kwargs["cache_config"] = draft_cache_config

    draft_vllm_config = replace(
        vllm_config,
        **replace_kwargs,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    from vllm.model_executor.layers.quantization.sm70_turbomind import (
        is_exact_sm70_cuda_platform,
    )

    from .mtp_fp8_experts import MTPExpertFp8Config, checkpoint_fp8_prefixes

    online_fp8 = getattr(speculative_config, "mtp_expert_quantization", None) == "fp8"
    checkpoint_prefixes = set()
    if draft_quant_config is not None and is_exact_sm70_cuda_platform():
        config = draft_vllm_config.model_config.hf_text_config
        prefixes = {
            f"mtp.layers.{mtp_start_layer_idx + index}.mlp.experts"
            for index in range(getattr(config, "mtp_num_hidden_layers", 1))
        }
        checkpoint_prefixes = checkpoint_fp8_prefixes(draft_quant_config, prefixes)
    if online_fp8 or checkpoint_prefixes:
        if draft_quant_config is None or not _mtp_fp8_experts_supported(
            draft_vllm_config,
            draft_quant_config,
            speculative_config,
            is_exact_sm70_cuda_platform(),
            online_fp8,
        ):
            raise ValueError(
                "MTP FP8 experts require SM70, FP16, an AWQ/ModelOpt/FP8 draft "
                "checkpoint, no expert parallelism, standard rejection sampling, "
                "and pipeline-parallel size 1 for online conversion"
            )
        draft_quant_config = MTPExpertFp8Config(
            draft_quant_config,
            checkpoint_prefixes,
            quantize_unquantized=online_fp8,
        )
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config
```

---

### 3.4 在干净 `main` 分支上一键应用本问题修复的 `git apply` 命令

```bash
cat << 'EOF' | git apply
diff --git a/vllm/models/qwen4_exp/nvidia/mtp.py b/vllm/models/qwen4_exp/nvidia/mtp.py
index 62eb948..b1a4d52 100644
--- a/vllm/models/qwen4_exp/nvidia/mtp.py
+++ b/vllm/models/qwen4_exp/nvidia/mtp.py
@@ -15,6 +15,9 @@
 """
 
 from collections.abc import Iterable
+import copy
+from dataclasses import is_dataclass
+from typing import Any
 
 import regex as re
 import torch
@@ -241,10 +244,25 @@ def _make_draft_vllm_config(
                 _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
             )
 
+    replace_kwargs: dict[str, Any] = {
+        "model_config": speculative_config.draft_model_config,
+        "is_speculative_draft": True,
+    }
+    draft_cache_dtype = getattr(speculative_config, "kv_cache_dtype", None)
+    draft_cache_config = getattr(vllm_config, "cache_config", None)
+    if draft_cache_dtype is not None and draft_cache_config is not None:
+        if is_dataclass(draft_cache_config):
+            draft_cache_config = replace(
+                draft_cache_config,
+                cache_dtype=draft_cache_dtype,
+            )
+        else:
+            draft_cache_config = copy.copy(draft_cache_config)
+            draft_cache_config.cache_dtype = draft_cache_dtype
+        replace_kwargs["cache_config"] = draft_cache_config
+
     draft_vllm_config = replace(
         vllm_config,
-        model_config=speculative_config.draft_model_config,
-        is_speculative_draft=True,
+        **replace_kwargs,
     )
     # VllmConfig post-init derives the target quant config, so restore the
     # independently resolved draft quant config after replacement.
diff --git a/vllm/models/qwen4_exp/amd/mtp.py b/vllm/models/qwen4_exp/amd/mtp.py
index 7b5be1e..285e81f 100644
--- a/vllm/models/qwen4_exp/amd/mtp.py
+++ b/vllm/models/qwen4_exp/amd/mtp.py
@@ -15,6 +15,9 @@
 """
 
 from collections.abc import Iterable
+import copy
+from dataclasses import is_dataclass
+from typing import Any
 
 import regex as re
 import torch
@@ -172,9 +175,24 @@ def _make_draft_vllm_config(
                 _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
             )
 
+    replace_kwargs: dict[str, Any] = {
+        "model_config": speculative_config.draft_model_config,
+    }
+    draft_cache_dtype = getattr(speculative_config, "kv_cache_dtype", None)
+    draft_cache_config = getattr(vllm_config, "cache_config", None)
+    if draft_cache_dtype is not None and draft_cache_config is not None:
+        if is_dataclass(draft_cache_config):
+            draft_cache_config = replace(
+                draft_cache_config,
+                cache_dtype=draft_cache_dtype,
+            )
+        else:
+            draft_cache_config = copy.copy(draft_cache_config)
+            draft_cache_config.cache_dtype = draft_cache_dtype
+        replace_kwargs["cache_config"] = draft_cache_config
+
     draft_vllm_config = replace(
         vllm_config,
-        model_config=speculative_config.draft_model_config,
+        **replace_kwargs,
     )
     # VllmConfig post-init derives the target quant config, so restore the
     # independently resolved draft quant config after replacement.
EOF
```

---

## 4. 配套启动命令（主模型 `fp8_e4m3` + MTP 草稿 `float16`）

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 关闭默认 Disk-Mmap PLE，使用 Pinned-Host UVA 内存直读
export VLLM_PLE_CPU_OFFLOAD=0
export VLLM_PLE_DISK_OFFLOAD=0
export VLLM_SM70_QWEN38_HYBRID_PLE=0
export VLLM_QWEN4EXP_PLE_HOST_GIB="${VLLM_QWEN4EXP_PLE_HOST_GIB:-12}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name Qwen3.8-Flash-Next \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --dtype half \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"kv_cache_dtype":"float16"}' \
  --seed 0
```
