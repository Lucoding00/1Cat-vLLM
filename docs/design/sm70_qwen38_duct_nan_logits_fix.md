# `Qwen3.8-Flash-Next` SM70/V100 推理输出坍缩 `"duct"` + `logits NaN` + `MTP 0% 接受率` 根因定性与代码修复手册（含最终 GPU 实证结论）

> **针对环境**：`/home/ma-user/work/1cat-vllm-latest`（`4 × V100`，`TP4 + DCP2 (+ MTP4)`，`Qwen3.8-Flash-Next-NVFP4`）  
> **最终实证根因（已通过 GPU 独立脚本与线上 Eager 推理 100% 闭环验证）**：
> 1. **主根因（`sm70_fp16_gemv.py` GDN 投影 Split Kernel 启动网格错配）**：在 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` 中，`_split_gdn_projection_tails` 与 `_split_gdn_projection_outputs` 的 Triton 启动网格被设为 `(m, triton.cdiv(1536, 256))`（仅 6 个 CTA，覆盖 `col ∈ [0, 1536)`），而 `_qwen38_gdn_projection_split_kernel` 内核写回 `z` 的掩码为 `(col >= QKV) & (col < QKV + Z)`（`QKV = 2560, Z = 1536`）。当 `col < 1536` 时 `col >= 2560` **恒为 `False`**，导致 `z = mixed_qkvz.new_empty((num_tokens, 12, 128))` **100% 永远不被写入**，直接将未初始化显存垃圾（含稀疏 FP16 `NaN`/`Inf`）送入 `RMSNormGated(core_attn_out, z)` 与 `out_proj`，从首个出现脏显存复用的 GDN 层扩散至全模型。将该文件两处 `triton.cdiv(1536, 256)` 恢复为 **`triton.cdiv(4096, 256)`（16 块）** 后，`NaN` 完全归零，推理输出恢复正常。
> 2. **次级防御加固（`qsa.py` 异步 DCP Query AllGather 顺序与 3D 形状校验）**：当开启 `sm70_qsa_prep` 时，确保 `_run_qsa` 先完成 `qsa_prep_sm70_out`（生成归一化 3D `prepared` Query）再发起 `_start_all_gather_dcp_query`，并在 `_finish_all_gather_dcp_query` 中保证 `rows > 1` 的跨卡维度转置。

---

## 0. 最终版根因复盘：`sm70_fp16_gemv.py` GDN 投影 Split Kernel 网格错配（为什么正是这个原因？）

### 0.1 为什么这个根因完美解释了三轮实验的全部现象？

1. **为什么 `z` 缓冲区 100% 未被写入，且呈现“稀疏 `NaN`（`96 / 36864 ≈ 0.26%`）”？**
   - 在 `qwen_gdn_linear_attn.py`（`L3090–3115` 与 `L5866–5879`）中，因 `D16` 将 `_split_gdn_projection_tails` 的触发条件从 `mixed_qkvz.shape[0] in (5, 20)` 放宽到了 **`mixed_qkvz.shape[0] > 1`**，所有 `m > 1` 的 Prefill（如 24 tokens）都会进入：
     ```python
     z = mixed_qkvz.new_empty((num_tokens, 12, 128))
     mixed_qkv, b, a = _split_gdn_projection_tails(mixed_qkvz, ba, z)
     ```
   - 在 `sm70_fp16_gemv.py` 中，`_qwen38_gdn_projection_split_kernel` 的列索引与 `z` 写回逻辑为：
     ```python
     col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
     value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
     if COPY_QKV:
         tl.store(qkv + row * QKV + col, value, col < QKV)
     tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
     ```
     其中 `QKV = 2560`、`Z = 1536`、`BLOCK = 256`。只有 **`program_id(1) ∈ [10, 15]`**（对应 `col ∈ [2560, 4096)`）的 6 个 CTA 才会满足 `(col >= 2560) & (col < 4096)` 并向 `z` 写入数据。
   - 当服务器上 `_split_gdn_projection_tails`（及 `_split_gdn_projection_outputs`）的启动 grid 被写成 `(m, triton.cdiv(1536, 256)) = (m, 6)`（可能由全局替换 `s/triton.cdiv(4096, 256)/triton.cdiv(1536, 256)/g` 或未同步内核分支导致）时：
     - 只启动了 `program_id(1) ∈ [0, 5]`，即 `col ∈ [0, 1536)`；
     - `col >= 2560` 对所有线程**恒为 `False`**，`tl.store(z + ...)` 被 100% 掩码屏蔽，**没有任何一个线程向 `z` 写入过哪怕 1 个字节**！
     - 因此 `z` 完全保留了 `torch.new_empty((24, 12, 128))` 从 PyTorch CUDA Caching Allocator 分配出来的**未初始化显存旧数据**（在 IEEE-754 FP16 中，指数位为 `0b11111` 的随机旧显存位模式恰好表现为 `~0.26%–0.7%` 的稀疏 `NaN`/`Inf`）。

2. **为什么“短 Prompt（1 token 即结束）偶发正常”，而正常多 Token Prompt 必现 `NaN`？**
   - 因为 `_can_fuse_gdn_projection_split`（`sm70_fp16_gemv.py:150`）和 `qwen_gdn_linear_attn.py:3108` 都显式要求 **`qkvz.shape[0] > 1`**！
   - 当 Prompt 只有 `1` 个 token（`m == 1`）时，不走 `_split_gdn_projection_tails`，而是走 `mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)` 视图切片，所以 `m == 1` 时 `z` 是正确的！
   - 一旦 Prompt 长度 `m > 1`（如 17、24、2500 tokens），Prefill 必定走 `_split_gdn_projection_tails`，`z` 立即变成未初始化显存垃圾。

3. **为什么在 Eager 模式下首现点是 `layers.2.linear_attn`，而在第二轮（`torch.compile` 编译模式）下 `gdn_out=0` 且只看到 `ops/hc.py`、`moe_in_x`、`qsa_in_hidden` 报 `NaN`？**
   - **为什么第二轮编译模式看不到 `gdn_out` 和 `layer{i}_after_attn`**：
     `nan_diag.py` 中写有 `if not _ENABLED or torch.compiler.is_compiling(): return`。在开启 `torch.compile` 时，`Qwen4ExpModel.forward`、`Qwen4ExpDecoderLayer.forward`、`QwenGatedDeltaNetAttention.forward`、`RMSNormGated` 均在 Dynamo 追踪期（`is_compiling() == True`）被内联编译，其中的 `nan_diag.check()` 在追踪期直接 `return` 被短路，根本没有进入编译后的 FX 图！运行时只有通过 `direct_register_custom_op` 注册的图外不透明算子（`ops/hc.py` 的 `_hc_combine_norm_impl` / `_hc_gate_mix`、`nvfp4_sm70_moe.py`、`qsa.py:_run_qsa`、`indexer_qsa.py`）以 `is_compiling() == False` 执行 Python 回调，因此第二轮只能看到下游扩散后的 `NaN`。
   - **为什么第三轮切换 `--compilation-config '{"mode":"NONE"}'`（Eager 模式）后一击锁定 `layers.2.linear_attn`**：
     Eager 模式下 `is_compiling()` 恒为 `False`，全模型探针全部生效。在 `layers.0` 和 `layers.1` 中，虽然 `z` 同样未被写入，但模型刚启动时 `new_empty` 拿到的是刚清零的显存页（全 `0` 有限值，虽不产生 `NaN` 但已破坏数值）；到了 **`layers.2.linear_attn`**，`new_empty((24, 12, 128))` 复用了前两层释放的脏显存块，其中恰好含有 `96` 个 `NaN` 和 `1` 个 `Inf`（`96 / 36864 ≈ 0.26%`），经 `RMSNormGated(core_attn_out, z)` 与 `out_proj`（`[24, 1536] @ [1536, 2560]` 稠密矩阵乘 + TP4 AllReduce）后，将含 `NaN` 的行放大为整行 `2560` 维全 `NaN`，并从 `layers.3` 起污染后续全部层的 `hidden_states`、KV Cache 与 Mamba 状态。

### 0.2 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` 修复代码

确保 `_qwen38_gdn_projection_split_kernel`、`_split_gdn_projection_outputs` 与 `_split_gdn_projection_tails` 统一使用覆盖完整 `QKV + Z = 4096` 列的 **`(m, triton.cdiv(4096, 256))`（16 块）** 网格：

```python
@triton.jit
def _qwen38_gdn_projection_split_kernel(
    qkvz,
    ba,
    qkv,
    z,
    b,
    a,
    QKV: tl.constexpr,
    Z: tl.constexpr,
    B: tl.constexpr,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
    COPY_QKV: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(qkvz + row * (QKV + Z) + col, col < QKV + Z, other=0)
    if COPY_QKV:
        tl.store(qkv + row * QKV + col, value, col < QKV)
    tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
    if tl.program_id(1) == 0:
        tl.static_assert(B + A <= BLOCK)
        gate_col = tl.arange(0, BLOCK)
        gate = tl.load(ba + row * (B + A) + gate_col, gate_col < B + A, other=0)
        tl.store(b + row * B + gate_col, gate, gate_col < B)
        tl.store(a + row * A + gate_col - B, gate, (gate_col >= B) & (gate_col < B + A))


def _split_gdn_projection_outputs(qkvz, ba):
    m = qkvz.shape[0]
    out = tuple(qkvz.new_empty((m, n)) for n in (2560, 1536, 12, 12))
    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
        qkvz,
        ba,
        *out,
        QKV=2560,
        Z=1536,
        B=12,
        A=12,
        BLOCK=256,
        COPY_QKV=True,
        num_warps=4,
        num_stages=1,
    )
    return out


def _split_gdn_projection_tails(qkvz, ba, z_out):
    """Copy z/b/a together while preserving the input QKV view and stride."""
    m = qkvz.shape[0]
    qkv = qkvz[:, :2560]
    b, a = (qkvz.new_empty((m, 12)) for _ in range(2))
    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
        qkvz,
        ba,
        qkv,
        z_out,
        b,
        a,
        QKV=2560,
        Z=1536,
        B=12,
        A=12,
        BLOCK=256,
        COPY_QKV=False,
        num_warps=4,
        num_stages=1,
    )
    return qkv, b, a
```

---

## 1. 次级加固：`vllm/models/qwen4_exp/nvidia/qsa.py` 异步 DCP Query AllGather 顺序与 3D 形状保护

### 1.1 背景说明：`vllm/models/qwen4_exp/nvidia/qsa.py` 中 `_run_qsa` 与 `_forward_qsa_dcp` 的防御校验

在服务器 `/home/ma-user/work/1cat-vllm-latest/vllm/models/qwen4_exp/nvidia/qsa.py` 中，当合入 `D14`（异步 DCP Query AllGather 与 Indexer 重叠）时，出现了以下 **3 处致命错配**：

#### 错配 ①：在 `qsa_prep_sm70_out` 之前对 **原始 2D 未归一化 `qkv`** 发起了异步 `all_gather`（行号 `1163–1211`）
1. 当切换为 `--kv-cache-dtype auto`（FP16）后，`self._sm70_qsa_prep` 首次变为 **`True`**（`qsa.py:893` 要求 `cache_dtype in ("auto", "float16")`）。
2. 在 `Qwen4ExpQSAAttention.forward()` 中，当 `self._sm70_qsa_prep` 生效时：
   ```python
   q_gate = qkv[:, : self.q_size * 2].view(
       num_tokens, self.num_heads, 2 * self.head_dim
   )
   gate = q_gate[..., self.head_dim :]
   query = qkv          # <-- 注意：此时 query 是 2D 原始未归一化 qkv [num_tokens, 3584]！
   key = value = qkv
   ```
   此时传入 `_run_qsa` 的 `query` **根本不是** 归一化后的 3D Query `[num_tokens, 6, 256]`，而是 **包含原始未归一化 Q + Gate + K + V 的 2D 张量 `[num_tokens, 3584]`**（`query.dim() == 2`）！
3. 然而在服务器 `_run_qsa`（行号 `1163–1211`）中，执行顺序写成了：
   ```python
   # [错误顺序] 1163-1173：此时 query 还是 2D 原始未归一化 qkv [T, 3584]！
   dcp_ag_state = impl._start_all_gather_dcp_query(query[:num_tokens])

   # 1174-1179：运行 indexer
   selected = self.indexer(...)

   # 1194-1211：此时才运行 qsa_prep_sm70_out 算出归一化+RoPE 的 3D prepared [T, 6, 256]
   torch.ops._C.qsa_prep_sm70_out(query[:num_tokens], ..., prepared, key_cache, value_cache, slot_mapping)
   query = key = value = prepared

   # 1220-1233：虽然传了 query=prepared，但同时传了装有【原始 2D qkv】的 dcp_ag_state！
   impl.forward_qsa(self, query, ..., dcp_ag_state=dcp_ag_state)
   ```
4. 随后进入 `_forward_qsa_dcp`（行号 `580`）：
   `_all_gather_dcp_query(prepared, dcp_ag_state)` 看到 `dcp_ag_state is not None`，**直接丢弃了归一化好的 `prepared`，返回了 `dcp_ag_state` 中跨卡 Gather 的原始未归一化 2D `qkv`**！

---

#### 错配 ②：服务器 `580` 行 `_all_gather_dcp_query` 返回了 `state[2] = recv[dcp_rank]`（或未对 `T > 1` 做 `movedim(0, 1)` 转置）
- 在 `TP4 + DCP2` 下，每个 Rank 持有 `6` 个本地 Query Head（`[T, 6, 256]`），而 `_forward_qsa_dcp` 必须沿 **`dim=1`（Head 维）** 拼接两个 DCP Rank 的 Query，得到 **`[T, 12, 256]`**（等价于 `group.all_gather(query.contiguous(), dim=1)`）。
- 服务器 `580` 行若返回 `state[2] = recv[dcp_rank]`，则只取了单卡切片（或原始 `qkv` 的 `3584` 维），根本不是两个 DCP Rank 合并后的 `[T, 12, 256]`！
- 同时，`torch.distributed.all_gather_into_tensor(recv_view, q_contig)` 在底层总是沿 **`dim=0`** 拼接，写入内存布局为 **`[dcp_world_size, T, 6, 256]`**（即先排 Rank 0 的全部 `T` 个 Token，再排 Rank 1 的全部 `T` 个 Token）：
  - 当 `T == 1` 时，`[2, 1, 6, 256]` 直接 `.view(1, 12, 256)` 恰好等价于沿 `dim=1` 拼接；
  - 当 **`T > 1`**（如 Prompt Prefill `T = 17` / `2500`，或 MTP4 Verify `T = 5`）时，**必须先执行 `.view(world_size, rows, heads, dim).movedim(0, 1)` 交换前两维并拷贝为连续的 `[T, 2, 6, 256]`，再 `.view(T, 12, 256)`**！否则 Token 0 的后 6 个 Head 会错读成同 Rank 的 Token 1，导致多 Token 批次下跨卡注意力彻底错乱。

---

#### 错配 ③：服务器 `615` 行 `partial_lse_buffer[:num_tokens, :gathered_query.shape[1]]` 越界与 `exp2` 溢出产生 `NaN`
- 当 `gathered_query` 是从原始 2D `qkv` Gather 出来的张量时：
  1. **数值上**：原始 `qkv` 未做 `q_norm`（RMSNorm）和 RoPE，幅值高达数十至上百（正常 RMSNorm 后约 `~1.0`）。在 `qsa_sparse_paged_attention` 中与 `k_cache` 做点积后，`scores * qk_scale_log2` 远超 FP32 `exp2` 上限（`> 128`），`tl.math.exp2` 直接溢出为 **`+inf`**，随后在在线 Softmax 归一化 `accumulator / normalizer`（`inf / inf`）以及 `dcp_a2a_lse_reduce` 的 `exp2(lse_val - global_lse)`（`inf - inf`）中立即产生 **`NaN`**！
  2. **形状上**：`partial_lse_buffer` 的预分配形状为 `[max_tokens, 12]`，而原始 `qkv` 的 `shape[1]` 为 `3584`（或 `7168`），导致 `partial_lse` 切片与内核写回严重越界。

---

### 1.2 为什么【2500+ Token 长上下文实验】也同样复现 `NaN` + `"duct"`？

报告中第 2 节提到：*“构造 ~2500+ token 长 prompt 绕过 D22，结果长上下文仍 NaN”*。原因非常直接：
1. 如果服务器上合入了 `P12`（去掉了 `if self._sm70_qsa_prep and num_tokens <= 32:` 中的 `and num_tokens <= 32`），那么 2500+ Token 的 Prefill 也会走 `query = qkv`（2D），在 Prefill 第 1 个 QSA 层（`layers.3`）就直接产出 `NaN`；
2. 退一步说，即便服务器上保留了 `if self._sm70_qsa_prep and num_tokens <= 32:`，使得 2500 Token 的 Prefill 走了 `else` 分支（`_project_qkv_gate`），**一旦进入生成阶段（`max_tokens=80`），每一步 Decode（`num_tokens = 1 <= 32`）或 MTP Verify（`num_tokens = 5 <= 32`）都必然满足 `num_tokens <= 32`**，从而在第 1 个生成步立即进入 `if self._sm70_qsa_prep:` $\rightarrow$ 触发 `_start_all_gather_dcp_query(原始 qkv)` $\rightarrow$ 产生 `NaN`！
3. 而对于 17 Token 的短 Prompt（`prompt_tokens = 17 <= 32`），甚至连 Prefill 本身也满足 `num_tokens <= 32`，从 Prefill 第 1 步起就是 `NaN`。

---

### 1.3 为什么第 3 层 QSA 一旦产出 `NaN`，会同时导致那 3 个外在症状？

1. **`logprobs` 请求报 HTTP 400 (`Out of range float values are not JSON compliant: nan`)**：
   - `layers.3`（首个 QSA 层）输出的 `NaN` 写入 HyperConnection 的 4 条残差流后，沿 `layers.4..47` 传播到最终 `lm_head`，使主模型输出的 `logits` 全为 `NaN`。开启 `logprobs` 时 API Server 序列化 `NaN` 浮点数直接抛 400 错误。
2. **所有生成 Token 恒为 `"duct" × 80`**：
   - 当 `logits` 全为 `NaN` 时，CUDA 端 `argmax` / 采样算子在进行浮点数比较（`val > max_val` 对 `NaN` 恒为 `False`）时，每一轮都会确定性地落回固定兜底下标（对应词表中的 `"duct"` Token），直到跑满 `max_tokens=80`（`finish_reason=length`）。
3. **MTP 草稿 `172` 个 Token 接受率 `0.0%`（`Accepted: 0 tokens`）**：
   - 在 `vllm/v1/sample/rejection_sampler.py:1446` 中，投机采样接受条件为：
     ```python
     accepted = (draft_prob > 0) & ((target_prob / draft_prob) >= uniform_sample)
     ```
   - 因为主模型 `logits` 为 `NaN`，`target_prob` 也是 `NaN`。根据 IEEE-754 浮点数规范，**`NaN >= uniform_sample` 恒为 `False`**！因此无论 MTP 草稿模型预测得有多准，172 个草稿 Token 会被 100% 拒绝，只保留每轮固定回退的 1 个 bonus Token（`"duct"`），所以 `Mean acceptance length: 1.00, Accepted: 0 tokens`。

---

## 2. 直接修复方案（修改服务器 `vllm/models/qwen4_exp/nvidia/qsa.py`）

需要且仅需修复 `/home/ma-user/work/1cat-vllm-latest/vllm/models/qwen4_exp/nvidia/qsa.py` 中的 **两处**：
1. **在 `_run_qsa`（服务器约 `1162–1233` 行）中**：
   - **必须先执行 `qsa_prep_sm70_out`（将 2D 原始 `qkv` 归一化/旋转为 3D `prepared` `[num_tokens, 6, 256]`，并赋值给 `query = key = value = prepared`）**；
   - **然后再调用 `dcp_ag_state = impl._start_all_gather_dcp_query(query[:num_tokens])`**，随后立即执行 `selected = self.indexer(...)`（从而让**归一化后的 3D `prepared` Query** 的跨卡 AllGather 与 Indexer 的 4 个算子并行重叠！）。
2. **在 `_start_all_gather_dcp_query` / `_finish_all_gather_dcp_query` / `_all_gather_dcp_query` / `_forward_qsa_dcp`（服务器约 `420–657` 行）中**：
   - 强制校验 `query.dim() == 3`（严禁对 2D 原始 `qkv` 发起 Gather）；
   - 在 `ag_state` 中记录源张量指纹 `(query.data_ptr(), tuple(query.shape))`，若 `_forward_qsa_dcp` 发现传入的 `dcp_ag_state` 与当前 `query`（`prepared`）不一致，自动丢弃旧 `dcp_ag_state` 并对 `prepared` 重新发起 Gather；
   - 修复 `_finish_all_gather_dcp_query`：当 `rows == 1` 时返回 `recv_view.view(1, world_size * heads, dim)`（`[1, 12, 256]`）；当 `rows > 1` 时通过 `recv_view.view(world_size, rows, heads, dim).movedim(0, 1)` 转置拷贝为 `[rows, 12, 256]`，**绝不能返回 `recv[dcp_rank]`**；
   - 在 `_forward_qsa_dcp` 中使用 `partial_output = partial_output_buffer[:rows]` 和 `partial_lse = partial_lse_buffer[:rows]`（形状固定为 `[rows, 12, 256]` 与 `[rows, 12]`），并校验 `partial_output.shape == gathered_query.shape`。

---

### 2.1 修复后的完整函数代码（可直接替换 `vllm/models/qwen4_exp/nvidia/qsa.py` 对应函数）

#### （A）`Qwen4ExpQSAFlashAttentionImpl` 中的 `_start_all_gather_dcp_query`、`_finish_all_gather_dcp_query`、`_all_gather_dcp_query` 与 `_forward_qsa_dcp`（约 `420–657` 行）

```python
    def _start_all_gather_dcp_query(
        self,
        query: torch.Tensor,
        group: Any | None = None,
    ) -> tuple[
        Any,
        torch.Tensor,
        torch.Tensor | None,
        int,
        int,
        int,
        int,
        tuple[int, ...],
    ]:
        if query.dim() != 3:
            raise RuntimeError(
                f"QSA DCP query all_gather requires 3D prepared query [T, H, D], "
                f"got shape {tuple(query.shape)} (ensure qsa_prep_sm70_out runs "
                f"before _start_all_gather_dcp_query)"
            )
        if group is None:
            from vllm.distributed import get_dcp_group

            group = get_dcp_group()
        source_ptr = query.data_ptr()
        source_shape = tuple(query.shape)
        q_contig = query if query.is_contiguous() else query.contiguous()
        device_group = getattr(group, "device_group", None)
        if device_group is None and hasattr(group, "device_communicator"):
            device_group = getattr(group.device_communicator, "device_group", None)
        if device_group is None:
            return (
                None,
                group.all_gather(q_contig, dim=1),
                None,
                0,
                0,
                0,
                source_ptr,
                source_shape,
            )
        rows, heads, dim = q_contig.shape
        world_size = self.dcp_world_size
        needed = world_size * rows * heads * dim
        if q_contig.is_cuda:
            device_index = (
                q_contig.device.index if q_contig.device.index is not None else -1
            )
            stream_id = int(torch.cuda.current_stream(q_contig.device).cuda_stream)
            ws_key = (device_index, stream_id, q_contig.dtype)
            ws_cache = workspace_cache("qsa_dcp_ag", _SM70_QSA_DCP_AG_WORKSPACES)
            workspace = ws_cache.get(ws_key)
            if workspace is None or workspace[0] < needed:
                cap = 1 << (max(needed, 1) - 1).bit_length()
                if workspace is not None:
                    cap = max(cap, workspace[0])
                recv_buf = torch.empty(
                    (cap,), dtype=q_contig.dtype, device=q_contig.device
                )
                out_buf = torch.empty(
                    (cap,), dtype=q_contig.dtype, device=q_contig.device
                )
                workspace = (cap, recv_buf, out_buf)
                ws_cache[ws_key] = workspace
            retain_for_capture(ws_cache, workspace, q_contig)
            _, recv_buf, out_buf = workspace
        else:
            recv_buf = torch.empty(
                (needed,), dtype=q_contig.dtype, device=q_contig.device
            )
            out_buf = (
                torch.empty((needed,), dtype=q_contig.dtype, device=q_contig.device)
                if rows > 1
                else None
            )
        recv_view = recv_buf[:needed].view(world_size * rows, heads, dim)
        work = torch.distributed.all_gather_into_tensor(
            recv_view, q_contig, group=device_group, async_op=True
        )
        return (
            work,
            recv_view,
            out_buf,
            rows,
            heads,
            dim,
            source_ptr,
            source_shape,
        )

    def _finish_all_gather_dcp_query(
        self,
        ag_state: tuple[Any, ...],
        expected_query: torch.Tensor | None = None,
        group: Any | None = None,
    ) -> torch.Tensor:
        if expected_query is not None and len(ag_state) >= 8:
            source_ptr = ag_state[6]
            source_shape = ag_state[7]
            if (
                source_ptr != expected_query.data_ptr()
                or source_shape != tuple(expected_query.shape)
            ):
                if ag_state[0] is not None:
                    ag_state[0].wait()
                ag_state = self._start_all_gather_dcp_query(expected_query, group)
        work, recv_view, out_buf, rows, heads, dim = ag_state[:6]
        if work is None:
            return recv_view
        work.wait()
        world_size = self.dcp_world_size
        if rows == 1:
            return recv_view.view(1, world_size * heads, dim)
        assert out_buf is not None
        needed = world_size * rows * heads * dim
        out_view = out_buf[:needed].view(rows, world_size, heads, dim)
        out_view.copy_(recv_view.view(world_size, rows, heads, dim).movedim(0, 1))
        return out_view.view(rows, world_size * heads, dim)

    def _all_gather_dcp_query(
        self,
        query: torch.Tensor,
        group: Any | None = None,
        dcp_ag_state: Any | None = None,
    ) -> torch.Tensor:
        if dcp_ag_state is None:
            dcp_ag_state = self._start_all_gather_dcp_query(query, group)
        return self._finish_all_gather_dcp_query(
            dcp_ag_state, expected_query=query, group=group
        )

    def _forward_qsa_dcp(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        logical_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor,
        dcp_ag_state: Any | None = None,
    ) -> None:
        from vllm.distributed import get_dcp_group
        from vllm.v1.attention.ops.dcp_alltoall import (
            _dcp_a2a_send_recv_buffers,
            dcp_a2a_lse_reduce,
        )

        from .ops.qsa import _qsa_output_gate, qsa_sparse_paged_attention
        from .ops.qsa_dcp import (
            qsa_dcp_local_selection_width,
            qsa_localize_dcp_indices,
        )

        local_indices_buffer = layer.dcp_local_indices_buffer
        partial_output_buffer = layer.dcp_partial_output_buffer
        partial_lse_buffer = layer.dcp_partial_lse_buffer
        if any(
            buffer is None
            for buffer in (
                local_indices_buffer,
                partial_output_buffer,
                partial_lse_buffer,
            )
        ):
            raise RuntimeError("QSA DCP target workspaces are not initialized")
        group = get_dcp_group()
        if dcp_ag_state is not None:
            if len(dcp_ag_state) < 8 or (
                dcp_ag_state[6] != query.data_ptr()
                or dcp_ag_state[7] != tuple(query.shape)
            ):
                if dcp_ag_state[0] is not None:
                    dcp_ag_state[0].wait()
                dcp_ag_state = None
        if dcp_ag_state is None:
            dcp_ag_state = self._start_all_gather_dcp_query(query, group)
        # Only this prefix of the compacted columns can hold an owned token.
        local_width = qsa_dcp_local_selection_width(
            layer.indexer.token_topk,
            layer.indexer.compress_ratio,
            self.dcp_world_size,
            layer.cp_kv_cache_interleave_size,
            logical_indices.shape[1],
        )
        resolve_physical = bool(key_cache.shape[0] * key_cache.shape[1] < 2**31)
        rows = query.shape[0]
        if logical_indices.is_cuda:
            device_index = (
                logical_indices.device.index
                if logical_indices.device.index is not None
                else -1
            )
            stream_id = int(
                torch.cuda.current_stream(logical_indices.device).cuda_stream
            )
            counts_key = (device_index, stream_id)
            counts_cache = workspace_cache(
                "qsa_dcp_valid_counts", _SM70_QSA_DCP_VALID_COUNTS_WORKSPACES
            )
            counts_ws = counts_cache.get(counts_key)
            if counts_ws is None or counts_ws[0] < rows:
                cap = 1 << (max(rows, 1) - 1).bit_length()
                if counts_ws is not None:
                    cap = max(cap, counts_ws[0])
                counts_buf = torch.empty(
                    (cap,), dtype=torch.int32, device=logical_indices.device
                )
                counts_ws = (cap, counts_buf)
                counts_cache[counts_key] = counts_ws
            retain_for_capture(counts_cache, counts_ws, logical_indices)
            local_valid_counts = counts_ws[1][:rows]
        else:
            local_valid_counts = torch.empty(
                (rows,), dtype=torch.int32, device=logical_indices.device
            )
        local_indices = qsa_localize_dcp_indices(
            logical_indices,
            local_indices_buffer[:rows, :local_width],
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            interleave_size=layer.cp_kv_cache_interleave_size,
            local_block_size=key_cache.shape[1],
            block_table=block_table if resolve_physical else None,
            token_to_req=token_to_req if resolve_physical else None,
            num_cache_blocks=key_cache.shape[0] if resolve_physical else None,
            valid_counts=local_valid_counts,
        )
        gathered_query = self._finish_all_gather_dcp_query(
            dcp_ag_state, expected_query=query, group=group
        )
        partial_output = partial_output_buffer[:rows]
        partial_lse = partial_lse_buffer[:rows]
        if partial_output.shape != gathered_query.shape:
            raise RuntimeError("QSA DCP partial output has wrong head geometry")
        use_fused_a2a_pack = bool(
            self.dcp_combine is dcp_a2a_lse_reduce
            and self.dcp_world_size > 1
            and partial_output.dtype == torch.float32
            and gathered_query.shape[1] % self.dcp_world_size == 0
        )
        send_buffer: torch.Tensor | None = None
        recv_buffer: torch.Tensor | None = None
        if use_fused_a2a_pack:
            send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
                (
                    self.dcp_world_size,
                    rows,
                    gathered_query.shape[1] // self.dcp_world_size,
                    gathered_query.shape[2] + 1,
                ),
                device=partial_output.device,
                dtype=partial_output.dtype,
            )
        qsa_sparse_paged_attention(
            gathered_query,
            key_cache,
            value_cache,
            local_indices,
            block_table,
            token_to_req,
            partial_output,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            lse=partial_lse,
            pre_resolved_indices=resolve_physical,
            dcp_packed_out=send_buffer,
            valid_counts=local_valid_counts,
        )
        gate_view = (
            output_gate
            if output_gate.shape == output.shape
            else output_gate.reshape(output.shape)
        )
        if self.dcp_combine is dcp_a2a_lse_reduce:
            dcp_a2a_lse_reduce(
                partial_output,
                partial_lse,
                group,
                is_lse_base_on_e=False,
                out=output,
                output_gate=gate_view,
                packed_send_buffer=send_buffer,
                recv_buffer=recv_buffer,
            )
            return
        merged = cast(
            torch.Tensor,
            self.dcp_combine(
                partial_output,
                partial_lse,
                group,
                is_lse_base_on_e=False,
            ),
        )
        output.copy_(merged)
        _qsa_output_gate(output, gate_view)
```

---

#### （B）`Qwen4ExpQSAAttention._run_qsa` 中的正确执行顺序（约 `1160–1285` 行）

注意：**`qsa_prep_sm70_out`（生成 3D `prepared` 并赋值给 `query`）必须在 `_start_all_gather_dcp_query(query[:num_tokens])` 和 `self.indexer(...)` 之前执行**：

```python
        dense = self._use_dense_short_context(num_tokens, side_metadata)
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if self.host_kv_enabled:
            self.host_kv.write(key, value, main_metadata.slot_mapping)
        elif query.dim() == 2:
            # 1. 先执行 SM70 prep：将原始 2D qkv 归一化+旋转为 3D prepared [num_tokens, 6, 256]，并写入 KV Cache
            kv_cache_key = (
                self.kv_cache.data_ptr(),
                self.kv_cache.shape,
                self.kv_cache.stride(),
            )
            if getattr(self, "_cached_raw_kv_unbind_key", None) == kv_cache_key:
                key_cache, value_cache = self._cached_raw_kv_unbind_pair
            else:
                key_cache, value_cache = self.kv_cache.unbind(1)
                self._cached_raw_kv_unbind_key = kv_cache_key
                self._cached_raw_kv_unbind_pair = (key_cache, value_cache)
            needed_q = num_tokens * self.num_heads * self.head_dim
            if query.is_cuda:
                device_index = (
                    query.device.index if query.device.index is not None else -1
                )
                stream_id = int(torch.cuda.current_stream(query.device).cuda_stream)
                prep_key = (device_index, stream_id, query.dtype)
                prep_cache = workspace_cache(
                    "qsa_prepared_q", _SM70_QSA_PREPARED_Q_WORKSPACES
                )
                prep_ws = prep_cache.get(prep_key)
                if prep_ws is None or prep_ws[0] < needed_q:
                    cap = 1 << (max(needed_q, 1) - 1).bit_length()
                    if prep_ws is not None:
                        cap = max(cap, prep_ws[0])
                    prep_buf = torch.empty(
                        (cap,), dtype=query.dtype, device=query.device
                    )
                    prep_ws = (cap, prep_buf)
                    prep_cache[prep_key] = prep_ws
                retain_for_capture(prep_cache, prep_ws, query)
                prepared = prep_ws[1][:needed_q].view(
                    num_tokens, self.num_heads, self.head_dim
                )
            else:
                prepared = query.new_empty(
                    (num_tokens, self.num_heads, self.head_dim)
                )
            text_positions = (positions[0] if positions.dim() == 2 else positions)[
                :num_tokens
            ]
            if text_positions.dtype != torch.int64:
                text_positions = text_positions.to(torch.int64)
            slot_mapping = main_metadata.slot_mapping[:num_tokens]
            if slot_mapping.dtype != torch.int64:
                slot_mapping = slot_mapping.to(torch.int64)
            torch.ops._C.qsa_prep_sm70_out(
                query[:num_tokens],
                text_positions,
                self.rotary_emb.cos_sin_cache,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
                prepared,
                key_cache,
                value_cache,
                slot_mapping,
            )
            query = key = value = prepared
        else:
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        # 2. 此时 query 必定是归一化后的 3D [num_tokens, 6, 256]，发起异步 DCP AllGather
        dcp_ag_state = None
        if (
            getattr(self, "qsa_dcp_sharded", False)
            and num_tokens > 0
            and not dense
            and query.dim() == 3
        ):
            dcp_ag_state = impl._start_all_gather_dcp_query(query[:num_tokens])
        # 3. 运行 Indexer（4 个内核与上面的异步 DCP AllGather 通信重叠）
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
            select=not dense,
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        selected = _sm70_dump_qwen_layer_tensor(
            "qsa_selected_indices",
            self.indexer.layer_id,
            "qsa",
            selected,
        )
        # 4. 进入 forward_qsa，消费同一个归一化后的 query 与 dcp_ag_state
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            output_gate=output_gate,
            query_positions=side_metadata.logical_positions,
            sequence_lengths=side_metadata.seq_lens,
            dense_short_context=dense,
            dcp_ag_state=dcp_ag_state,
        )
```

---

## 3. 服务器端最快验证方法（无需重编 C++，纯 Python 修改）

1. **按第 2.1 节调整 `/home/ma-user/work/1cat-vllm-latest/vllm/models/qwen4_exp/nvidia/qsa.py` 中 `_run_qsa` 的顺序**（将 `if self.host_kv_enabled: ... elif query.dim() == 2: ... torch.ops._C.qsa_prep_sm70_out(...) ... query = key = value = prepared` 移到 `dcp_ag_state = impl._start_all_gather_dcp_query(query[:num_tokens])` 和 `selected = self.indexer(...)` **之前**），并替换为上方 `_start_all_gather_dcp_query` / `_finish_all_gather_dcp_query` / `_all_gather_dcp_query` / `_forward_qsa_dcp`。
2. **清理编译缓存并重启服务**：
   ```bash
   rm -rf /home/ma-user/.cache/vllm/torch_compile_cache /tmp/torchinductor_ma-user /tmp/vllm-compile-cache*
   ```
3. **验证指标**：
   - 发送带 `"logprobs": true` 的请求：`200 OK`（不再报 `Out of range float values are not JSON compliant: nan`）；
   - 发送 `请用一句话介绍你自己。`：正常输出中文自我介绍（不再输出 `"duct"`）；
   - 查看日志 `SpecDecoding metrics`：`Mean acceptance length` 与 `Avg Draft acceptance rate` 恢复正常正值（不再是 `0.0%`）。
