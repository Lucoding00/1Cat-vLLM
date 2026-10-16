# SM70 V100×8 `Qwen3.8-Flash-Next` (`DP2 × TP4 + DCP2 + MTP4`) 报错根因分析与完整修复方案

本文档针对在 **8 × NVIDIA V100 (32GB, SM70)** 上以 `--data-parallel-size 2 --tensor-parallel-size 4 --decode-context-parallel-size 2`（`DP2 × TP4 + DCP2 + MTP4`）启动 `Qwen3.8-Flash-Next` 时出现的启动报错：

```text
NotImplementedError: SM70 TurboMind NVFP4 MoE currently supports tensor parallel sizes (1, 2, 4), plus GLM-5.3 TP8; got 8.
```

提供完整的根因溯源、代码级修复补丁（含完整 Unified Diff）以及在 `8 × V100` 满卡部署 **256K + 多模态 + 工具调用 + MTP4** 的两种生产启动方案。

---

## 1. 为什么 `TP4 + DCP2` 只用了 4 张卡，而加 `--data-parallel-size 2` 会报错 `got 8`？

### 1.1 为什么 `--tensor-parallel-size 4 --decode-context-parallel-size 2` 只占用 4 张 GPU？

在 vLLM 的并行拓扑定义（`vllm/config/parallel.py:328-331, 493-502`）中，`decode_context_parallel_size`（`DCP`）**不会增加 `world_size`**，而是直接复用 `TP` 通信组内的 GPU（要求 `tp_size % dcp_size == 0`）：

```python
# vllm/config/parallel.py:328-331
decode_context_parallel_size: int = 1
"""Number of decode context parallel groups, because the world size does
not change by dcp, it simply reuse the GPUs of TP group, and tp_size
needs to be divisible by dcp_size."""
```

因此：
- 单纯指定 `--tensor-parallel-size 4 --decode-context-parallel-size 2` 时，每个副本的 `world_size = pp_size × tp_size = 1 × 4 = 4` 张卡（在该 4 张 TP 卡内部按 `DCP=2` 切分 KV Cache）。
- 要把整机 **8 张 V100** 全部用满，同时保留 **SM70 NVFP4 MoE 的 `TP=4` 最优算子合同**与 **`DCP=2` 的 KV Cache 显存减半能力**，8 张卡的标准拓扑正是 **`DP=2 × TP=4`（每个 4 卡 TP 组内部开启 `DCP=2`）**：
  - **DP 副本 0（GPUs 0, 1, 2, 3）**：`TP=4, DCP=2`
  - **DP 副本 1（GPUs 4, 5, 6, 7）**：`TP=4, DCP=2`

---

### 1.2 报错 `got 8` 的直接根因：`FusedMoEParallelConfig` 将 `DP × TP` 展平为了 `flatten_tp_size = 8`

当传入 `--data-parallel-size 2 --tensor-parallel-size 4`（且未开启 `--enable-expert-parallel`）时，模型初始化链路发生了以下冲突：

1. **`FusedMoE` 默认读取全局 `dp_size = 2` 并把 `TP` 展平到 `DP × TP = 8`**：
   - 在 `vllm/model_executor/layers/fused_moe/layer.py:164-175` 中，若调用方未显式传入 `dp_size=1`，`FusedMoE.__init__` 默认取 `dp_size_ = get_dp_group().world_size = 2`。
   - 随后在 `vllm/model_executor/layers/fused_moe/config.py:1111, 1188-1201` 中，`FusedMoEParallelConfig.make()` 调用 `flatten_tp_across_dp_and_pcp()`：
     ```python
     # vllm/model_executor/layers/fused_moe/config.py:1111
     def flatten_tp_across_dp_and_pcp(...) -> tuple[int, int]:
         flatten_tp_size = dp_size * pcp_size * tp_size  # 2 * 1 * 4 = 8 !
         flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
         return flatten_tp_size, flatten_tp_rank
     ```
   - 当 `use_ep=False` 时，`FusedMoEParallelConfig` 把 `self.tp_size` 改写成了 `flatten_tp_size = 8`，`self.dp_size = 2`！

2. **`Qwen3NextSparseMoeBlock`（`Qwen4ExpSparseMoeBlock` 父类）漏传 `dp_size=1, pcp_size=1`**：
   - 在 `vllm/model_executor/models/qwen3_next.py:232-277` 中：
     - `self.shared_expert = Qwen3NextMLP(...)` 按 **4 卡 `TP` 组**（`self.tp_size = 4`）切分权重。
     - `forward()` 最后通过 `tensor_model_parallel_all_reduce` 在 **4 卡 `TP` 组**内对 `shared_expert + routed_experts` 做规约。
     - 但构造 `self.experts = FusedMoE(...)` 时未传入 `tp_size=self.tp_size, dp_size=1, pcp_size=1`，导致 `self.experts` 的 `moe_parallel_config.tp_size` 被展平为 `8`！

3. **这在 `Qwen3.8-Flash-Next` 上同时触发了 3 处致命冲突**：
   - **冲突 A（Target NVFP4 MoE 启动崩溃）**：
     `Qwen3.8-Flash-Next` 的 `moe_intermediate_size = 512`，NVFP4 group scale 的 `group_size = 128`。若 `moe.tp_size = 8`，`w2` (`down_proj`) 的 `K` 维会被切分为 `512 / 8 = 64 < 128`（出现半 Group 切分），因此 `validate_nvfp4_sm70_moe_contract()`（`vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py:591-597`）直接抛出：
     `NotImplementedError: SM70 TurboMind NVFP4 MoE currently supports tensor parallel sizes (1, 2, 4), plus GLM-5.3 TP8; got 8.`
   - **冲突 B（MTP4 Draft FP8 MoE 权重形状冲突）**：
     `Qwen4ExpMultiTokenPredictor`（`vllm/models/qwen4_exp/nvidia/mtp.py:327`）设置 `self.fp8_mtp_tp_size = draft_vllm_config.parallel_config.tensor_parallel_size = 4`，而 `MTPFp8SM70MoEMethod`（`vllm/models/qwen4_exp/nvidia/mtp_fp8_experts.py:80`）读取 `moe_parallel_config.tp_size = 8`，导致 MTP 权重加载与算子分片维度不一致。
   - **冲突 C（跨 DP `all_gatherv` / `reduce_scatterv` 通信错位）**：
     `moe.dp_size = 2` 会激活 `DefaultMoERunner.do_naive_dispatch_combine = True`（`default_moe_runner.py:310`），在每一层 MoE 前后跨 8 张卡执行 `all_gatherv` / `reduce_scatterv`，与按 4 卡 TP 切分的 `shared_expert` 完全错位。

---

## 2. 除启动报错外，`DP2 × TP4 + MTP4` 运行期的 2 个深层隐患（一并修复）

即使修好了 `Qwen3NextSparseMoeBlock` 的 `tp_size=4, dp_size=1`，在开启 `--data-parallel-size 2` + `MTP4`（`num_speculative_tokens = 4`）后，运行期还有 **2 个跨 DP 同步问题**必须同步修复：

### 2.1 运行期隐患 1：`LLMBaseProposer.dummy_run()` 与 `propose()` 的 DP `all_reduce` 次数不一致导致死锁

在 `DP=2` 模式下，当 `DP0` 有请求而 `DP1` 空闲时，`DPEngineCoreProc`（`vllm/v1/engine/core.py:2080`）会让空闲的 `DP1` 执行 `execute_dummy_batch()` -> `_dummy_run(1)` -> `self.drafter.dummy_run(1)`：
- **在 `DP0`（`LLMBaseProposer.propose()`，`llm_base_proposer.py:1137, 1396`）**：
  对于 `MTP4`（`num_speculative_tokens = 4`），`_determine_batch_execution_and_padding()`（内部调用 `coordinate_batch_across_dp` 的 `dist.all_reduce`）在 Step 0 前调用 **1 次**，在 Step 1..3 循环前调用 **1 次**，**共计调用 2 次**。
- **在 `DP1`（`LLMBaseProposer.dummy_run()`，`llm_base_proposer.py:2445-2453`）**：
  原代码在 `for fwd_idx in range(4):` 循环体内部每次迭代都调用 `_determine_batch_execution_and_padding()`，**共计调用 4 次**！
- **在超长上下文跳过 Drafter 时（`gpu_model_runner.py:9188`）**：
  当 `not input_fits_in_drafter` 时，原代码直接跳过 Drafter，**共计调用 0 次**！
- **后果**：一旦一个 DP 副本有请求（2 次 `all_reduce`）而另一个 DP 副本空闲（4 次 `all_reduce`），两个 DP 副本在 `coordinate_batch_across_dp` 的 `dist.all_reduce` 上立即失步并死锁。

### 2.2 运行期隐患 2：`_synchronize_dp_ranks()` 跨 DP 强行降级 CUDA Graph 并 Padding Token 数

在 `vllm/v1/worker/dp_utils.py:130-162` 中，`_synchronize_dp_ranks()` 默认取所有 DP 副本的最小 `cudagraph_mode`（`synced_cudagraph_mode = min(dp0, dp1)`）并将所有 DP 副本的 Token 数 Pad 到最大值（`max(num_tokens_dp0, num_tokens_dp1)`）。
- 这一逻辑仅在 MoE 开启跨 DP 专家并行（`enable_expert_parallel=True`）或序列并行（`use_sequence_parallel_moe=True`）时才需要。
- 对于 `Qwen3.8-Flash-Next` 的副本内 `TP=4` MoE（`dp_size=1`），`DP0`（GPUs 0..3）与 `DP1`（GPUs 4..7）在 `model.forward` 和 `drafter.forward` 内部**没有任何跨 DP 的集合通信**。
- 若不关闭跨 DP Padding/Graph 降级：当 `DP1` 正在跑 `8192` token 的 Prefill（`CUDAGraphMode.NONE`）时，`DP0` 正在跑的 `5` token Decode 会被强行踢出 `CUDAGraphMode.FULL` 退化为 Eager 模式，甚至被错误 Pad 到 `DP1` 的 Token 数。

---

## 3. 代码修复内容（4 个文件，含完整 Unified Diff）

本分支已将以下 4 处修复全部合入并通过语法与单测校验。若您需要在自己的 `main` 分支上直接打补丁，只需修改以下 4 个文件（共约 40 行代码）：

### 3.1 修复 1：`vllm/model_executor/models/qwen3_next.py`
**作用**：当未开启 `--enable-expert-parallel` 且未开启 `use_sequence_parallel_moe` 时，显式向 `FusedMoE` 传入 `tp_size=self.tp_size, dp_size=1, pcp_size=1`，阻止 `FusedMoEParallelConfig` 将 `TP=4` 乘以 `DP=2` 展平为 `8`，同时保证 Target NVFP4 MoE 与 MTP4 FP8 MoE 均严格运行在各自的 4 卡 `TP` 组内。

```diff
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -256,6 +256,10 @@ class Qwen3NextSparseMoeBlock(nn.Module):
                     scope="local",
                 )
 
+        keep_replica_local_tp = not (
+            bool(getattr(parallel_config, "enable_expert_parallel", False))
+            or self.is_sequence_parallel
+        )
         self.experts = FusedMoE(
             shared_experts=self.shared_expert,
             gate=self.gate,
@@ -265,6 +269,9 @@ class Qwen3NextSparseMoeBlock(nn.Module):
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

### 3.2 修复 2：`vllm/v1/spec_decode/llm_base_proposer.py`
**作用**：将 `LLMBaseProposer.dummy_run()` 在非 Graph Capture 模式下的 `_determine_batch_execution_and_padding()` 调用次数对齐到 `propose()`（Step 0 调用 1 次，Step 1..K-1 共享第 2 次调用结果，共 2 次），消除 `DP0` Busy (`propose`) 与 `DP1` Idle (`dummy_run`) 时的 `dist.all_reduce` 次数不匹配死锁。

```diff
--- a/vllm/v1/spec_decode/llm_base_proposer.py
+++ b/vllm/v1/spec_decode/llm_base_proposer.py
@@ -2443,16 +2443,17 @@ class LLMBaseProposer(SpecDecodeBaseProposer):
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

### 3.3 修复 3：`vllm/v1/worker/gpu_model_runner.py`
**作用**：当某个 DP 副本因达到 `effective_drafter_max_model_len` 而跳过 `propose_draft_token_ids`（`not input_fits_in_drafter`）时，在 `data_parallel_size > 1` 下执行一次 `self.drafter.dummy_run(1)`，保持跨 DP 的 Drafter 同步次数一致。

```diff
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -9173,6 +9173,22 @@ class GPUModelRunner(
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

### 3.4 修复 4：`vllm/v1/worker/dp_utils.py`
**作用**：当未开启 `enable_expert_parallel`、`use_sequence_parallel_moe` 和 `should_ubatch` 时（即每个 DP 副本的 MoE 完全在副本自身的 TP 组内独立运行），保留各 DP 副本自身的 `cudagraph_mode` 与 `num_tokens_padded`，避免一个副本的 Prefill 迫使另一个副本的 Decode 退出 CUDA Graph 或发生跨副本 Token Padding。

```diff
--- a/vllm/v1/worker/dp_utils.py
+++ b/vllm/v1/worker/dp_utils.py
@@ -136,18 +136,26 @@ def _synchronize_dp_ranks(
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
+        # Synchronize cudagraph_mode across ranks first (take min).
+        # This is needed before DP padding decision since we use the synced
+        # cudagraph mode to determine whether DP padding is needed.
+        synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
+
+        # DP padding is needed when cudagraph is enabled (synced across ranks)
+        # or when ubatching/DBO is active (ubatching requires uniform batch
+        # sizes across DP ranks currently).
+        # Use the synced runtime cudagraph mode rather than the compilation config
+        # so we can avoid padding when cudagraph is not enabled for this step.
+        should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    else:
+        synced_cudagraph_mode = cudagraph_mode
+        should_dp_pad = False
 
    # Pad all DP ranks up to the maximum token count across ranks if
    # should_dp_pad is True
```

### 3.5 修复 5（回归单测）：`tests/models/qwen4_exp/test_weight_loading.py`
**作用**：新增单元测试 `test_qwen3_next_sparse_moe_keeps_replica_local_tp_under_dp`，验证在 `DP > 1` 且 `enable_expert_parallel=False` 时，`Qwen3NextSparseMoeBlock` 向 `FusedMoE` 传入 `tp_size=4, dp_size=1, pcp_size=1`，防止回退为 `flatten_tp_size = 8`。

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

### 3.6 各文件修改后的完整代码段与在 `main` 分支上一键打补丁命令

#### (A) 修改后的完整代码段对照

1. **`vllm/model_executor/models/qwen3_next.py`（第 257–284 行）**：
```python
        keep_replica_local_tp = not (
            bool(getattr(parallel_config, "enable_expert_parallel", False))
            or self.is_sequence_parallel
        )
        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            tp_size=self.tp_size if keep_replica_local_tp else None,
            dp_size=1 if keep_replica_local_tp else None,
            pcp_size=1 if keep_replica_local_tp else None,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )
```

2. **`vllm/v1/spec_decode/llm_base_proposer.py`（`dummy_run` 第 2445–2458 行）**：
```python
        for fwd_idx in fwd_indices:
            if fwd_idx <= 1 or is_graph_capturing or spec_step_idx is not None:
                (
                    cudagraph_runtime_mode,
                    num_input_tokens,
                    num_tokens_across_dp,
                    base_batch_descriptor,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens, use_cudagraphs=use_cudagraphs
                )
            batch_descriptor = self._batch_descriptor_for_spec_step(
                base_batch_descriptor, fwd_idx
            )
```

3. **`vllm/v1/worker/gpu_model_runner.py`（`sample_tokens` 第 9174–9193 行）**：
```python
            if not input_fits_in_drafter:
                if (
                    self.vllm_config.parallel_config.data_parallel_size > 1
                    and (
                        spec_config.use_eagle()
                        or spec_config.uses_draft_model()
                        or spec_config.uses_extract_hidden_states()
                    )
                ):
                    assert isinstance(
                        self.drafter,
                        EagleProposer
                        | DFlashProposer
                        | DraftModelProposer
                        | ExtractHiddenStatesProposer
                        | Gemma4Proposer,
                    )
                    self.drafter.dummy_run(1)
```

4. **`vllm/v1/worker/dp_utils.py`（`_synchronize_dp_ranks` 第 139–167 行）**：
```python
    # Check conditions for microbatching
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    requires_cross_dp_moe_sync = bool(
        getattr(parallel_config, "enable_expert_parallel", False)
        or getattr(parallel_config, "use_sequence_parallel_moe", False)
    )
    if requires_cross_dp_moe_sync or should_ubatch:
        # Synchronize cudagraph_mode across ranks first (take min).
        # This is needed before DP padding decision since we use the synced
        # cudagraph mode to determine whether DP padding is needed.
        synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

        # DP padding is needed when cudagraph is enabled (synced across ranks)
        # or when ubatching/DBO is active (ubatching requires uniform batch
        # sizes across DP ranks currently).
        # Use the synced runtime cudagraph mode rather than the compilation config
        # so we can avoid padding when cudagraph is not enabled for this step.
        should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
    else:
        synced_cudagraph_mode = cudagraph_mode
        should_dp_pad = False

    # Pad all DP ranks up to the maximum token count across ranks if
    # should_dp_pad is True
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )
```

#### (B) 在 `main` 分支上一键应用补丁命令

如果您在本地 `main` 分支下，既可以直接运行：
```bash
git fetch origin arena/5368f936-1cat-vllm && git cherry-pick c0b3ae0
```
也可以直接复制下方整段 `git apply` 命令到终端执行（不依赖任何其它提交，纯 Python 修改无需重新编译 C++/CUDA）：

```bash
cat << 'EOF' | git apply -
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
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
diff --git a/vllm/v1/spec_decode/llm_base_proposer.py b/vllm/v1/spec_decode/llm_base_proposer.py
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
 
             # Make sure to use EAGLE's own buffer during cudagraph capture.
diff --git a/vllm/v1/worker/dp_utils.py b/vllm/v1/worker/dp_utils.py
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
+        # Synchronize cudagraph_mode across ranks first (take min).
+        # This is needed before DP padding decision since we use the synced
+        # cudagraph mode to determine whether DP padding is needed.
+        synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
+
+        # DP padding is needed when cudagraph is enabled (synced across ranks)
+        # or when ubatching/DBO is active (ubatching requires uniform batch
+        # sizes across DP ranks currently).
+        # Use the synced runtime cudagraph mode rather than the compilation config
+        # so we can avoid padding when cudagraph is not enabled for this step.
+        should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch
+    else:
+        synced_cudagraph_mode = cudagraph_mode
+        should_dp_pad = False
 
     # Pad all DP ranks up to the maximum token count across ranks if
     # should_dp_pad is True
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
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
                 # make token id 0 look like a real speculative token to the
EOF
```

---

## 4. `8 × V100`（`256K + 多模态 + 工具调用 + MTP4`）完整启动方案

在 `8 × V100 (32GB)` 上部署 `Qwen3.8-Flash-Next`（`TP4 + DCP2 + MTP4 + 256K + 多模态 + 工具调用`），您有以下 **两种生产启动方案**：

---

### 方案一：单端口原生 `DP2 × TP4 + DCP2 + MTP4`（应用第 3 节代码修复后）

应用上述 4 个文件的修复后，可以直接通过 `--data-parallel-size 2 --tensor-parallel-size 4 --decode-context-parallel-size 2` 用单条命令启动全部 8 张 V100，由 vLLM 内置 `DPCoordinator` 在 `8000` 端口统一对外提供负载均衡服务：

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# 8 张 V100 全部可见：DP0 使用 GPU 0..3，DP1 使用 GPU 4..7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ==============================================================================
# 关键配置：消除默认 Disk-Mmap PLE 导致的 Prefill 暴跌，并为 DP2×TP4 固定 Host 预算
# 注意：DP2 × TP4 下共 8 个 GPU Worker 进程。
# - 若主机空闲内存 >= 110 GiB：设置 VLLM_QWEN4EXP_PLE_HOST_GIB=10（8 × 10 = 80 GiB Pinned RAM）
# - 若主机空闲内存较小（64~96 GiB）：改用下方注释的共享 CPU Offload 模式（仅占 ~14.3 GiB RAM）
# ==============================================================================
export VLLM_PLE_CPU_OFFLOAD=0
export VLLM_PLE_DISK_OFFLOAD=0
export VLLM_SM70_QWEN38_HYBRID_PLE=0
export VLLM_QWEN4EXP_PLE_HOST_GIB="${VLLM_QWEN4EXP_PLE_HOST_GIB:-10}"

# （可选：若主机 RAM < 100 GiB，请注释上面 4 行并启用下面 5 行，8 个 Worker 共享单份 14.3 GiB PLE 表）
# export VLLM_SM70_QWEN38_HYBRID_PLE=1
# export VLLM_PLE_CPU_OFFLOAD=1
# export VLLM_PLE_DISK_OFFLOAD=0
# export VLLM_PLE_OFFLOAD_PREFAULT=1
# export VLLM_QWEN4EXP_PLE_HOST_GIB=4

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
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --seed 0
```

> **注意**：在 `DP2 × TP4` 下启用共享 `PleOffloadWorker`（`VLLM_PLE_CPU_OFFLOAD=1`）时，**必须显式设置固定的 `VLLM_QWEN4EXP_PLE_HOST_GIB`**（如 `4`），避免 `DP0` 与 `DP1` 在加载权重时因实时探测 `/proc/meminfo` 的瞬时值微小差异触发 `PleOffloadWorker._bind_remote_placements` 的 `remote placements differ between data-parallel ranks` 校验异常。

---

### 方案二：免改代码的双独立实例并发部署（在未打补丁的 `main` 分支上立即可用，且零跨 DP 同步开销）

如果您当前处于未打补丁的原始 `main` 分支，或者希望两个 4 卡副本之间**完全物理隔离**（即使一个副本在跑 256K 超长 Prefill，另一个副本也无需参与任何 `DPEngineCoreProc` 周期性 Wave 同步或 Dummy Batch），可以直接在 `8 × V100` 上启动 **2 个独立的 `TP4 + DCP2 + MTP4` 实例**：
- **实例 0**：绑定 `CUDA_VISIBLE_DEVICES=0,1,2,3`，监听端口 `8001`
- **实例 1**：绑定 `CUDA_VISIBLE_DEVICES=4,5,6,7`，监听端口 `8002`

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 纯 Pinned-Host UVA 模式（禁用磁盘 Mmap 卸载）
export VLLM_PLE_CPU_OFFLOAD=0
export VLLM_PLE_DISK_OFFLOAD=0
export VLLM_SM70_QWEN38_HYBRID_PLE=0
export VLLM_QWEN4EXP_PLE_HOST_GIB="${VLLM_QWEN4EXP_PLE_HOST_GIB:-10}"

COMMON_ARGS=(
  "${MODEL_PATH}"
  --served-model-name Qwen3.8-Flash-Next
  --host 0.0.0.0
  --trust-remote-code
  --tensor-parallel-size 4
  --decode-context-parallel-size 2
  --dtype half
  --kv-cache-dtype auto
  --max-model-len 262144
  --max-num-batched-tokens 8192
  --max-num-seqs 4
  --gpu-memory-utilization 0.90
  --no-async-scheduling
  --mamba-cache-mode align
  --enable-prefix-caching
  --no-language-model-only
  --limit-mm-per-prompt '{"image":4,"video":0}'
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --kernel-config '{"ple_disk_cascade":false}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
  --seed 0
)

# 启动副本 0（GPUs 0,1,2,3 -> Port 8001）
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "${COMMON_ARGS[@]}" --port 8001 &
PID0=$!

# 启动副本 1（GPUs 4,5,6,7 -> Port 8002）
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "${COMMON_ARGS[@]}" --port 8002 &
PID1=$!

wait "${PID0}" "${PID1}"
```
随后使用 Nginx / HAProxy（`least_conn` 策略）在 `8000` 端口反向代理 `127.0.0.1:8001` 与 `127.0.0.1:8002` 即可获得完全无锁的 8 卡双副本最大吞吐。

---

### 方案三：单机 `4 × V100 (32GB)`（`TP4 + DCP2 + MTP4 + 256K + 多模态 + 工具调用`）启动命令

如果机器只有 **4 张 V100 (32GB)**，去掉 `--data-parallel-size 2`，直接使用 `--tensor-parallel-size 4 --decode-context-parallel-size 2` 即可刚好占满 4 张卡（`DCP=2` 复用 `TP=4` 的 4 张 GPU，将每张卡的 256K KV Cache 显存占用减半）：

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# 4 张 V100 32GB
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ==============================================================================
# 关键配置：关闭默认 Disk-Mmap PLE，消除 Prefill 掉到几千 tok/s 的瓶颈
# 4 张卡共 4 个 Worker 进程，每个进程固定分配 12 GiB Pinned Host RAM（共 ~48 GiB Host RAM）
# ==============================================================================
export VLLM_PLE_CPU_OFFLOAD=0
export VLLM_PLE_DISK_OFFLOAD=0
export VLLM_SM70_QWEN38_HYBRID_PLE=0
export VLLM_QWEN4EXP_PLE_HOST_GIB="${VLLM_QWEN4EXP_PLE_HOST_GIB:-12}"

# （可选：若主机空闲内存 < 64 GiB，请注释上面 4 行并启用下面 5 行，总计仅占 ~14.3 GiB RAM）
# export VLLM_SM70_QWEN38_HYBRID_PLE=1
# export VLLM_PLE_CPU_OFFLOAD=1
# export VLLM_PLE_DISK_OFFLOAD=0
# export VLLM_PLE_OFFLOAD_PREFAULT=1
# export VLLM_QWEN4EXP_PLE_HOST_GIB=4

exec vllm serve "${MODEL_PATH}" \
  --served-model-name Qwen3.8-Flash-Next \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --dtype half \
  --kv-cache-dtype auto \
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --seed 0
```

