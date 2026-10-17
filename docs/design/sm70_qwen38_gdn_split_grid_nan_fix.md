# Qwen3.8-Flash-Next SM70 推理 NaN 根因分析与修复报告（独立完整版）

> **关联提交**：`eb277f9` (`fix(sm70): restore 16-block grid in GDN projection split kernels to prevent uninitialized z NaN`)  
> **核心涉及文件**：`vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py`、`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`  
> **目标硬件与拓扑**：`4 × NVIDIA V100 32GB (SM70)`，`TP4 + DCP2 (+ MTP4)`，模型 `Qwen3.8-Flash-Next-NVFP4`

---

## 结论摘要

1. **根本原因**：`vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` 中 GDN（Gated Delta Net）输入投影拆分内核 `_qwen38_gdn_projection_split_kernel` 的 **Triton 启动网格（launch grid）配置错误**——`_split_gdn_projection_tails`（及 `_split_gdn_projection_outputs`）仅以 `(m, triton.cdiv(1536, 256))` 启动了 **6 个线程块**（仅覆盖列区间 `col ∈ [0, 1536)`），而内核中写入门控分支 `z` 的列偏移基于 `QKV = 2560`，写掩码为 `(col >= QKV) & (col < QKV + Z)`（即 `col ∈ [2560, 4096)`，对应第 11～16 个线程块 `program_id(1) ∈ [10, 15]`）。由于 `col < 1536` 时 `col >= 2560` **恒为 `False`**，由 `mixed_qkvz.new_empty((num_tokens, 12, 128))` 分配的 `z` 缓冲区**永远不会被任何线程写入**，直接残留未初始化显存垃圾（包含稀疏 FP16 `NaN` / `Inf` 位模式），进而通过 `RMSNormGated` 和 `out_proj` 稠密矩阵乘污染全模型激活、KV Cache 与 Mamba 状态，导致 `logprobs` 返回 HTTP 400 `NaN` 且贪心解码坍缩为 `"duct"`。
2. **修复方案**：将 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` 中两处启动网格从 `triton.cdiv(1536, 256)`（6 块）统一修正为 **`triton.cdiv(4096, 256)`（16 块）**，完整覆盖 `QKV + Z = 4096` 列。
3. **验证结论**：GPU 独立哨兵测试脚本（`-999` 覆盖率检测）与线上实跑推理全部通过，全链路 `[NAN_DIAG]` 计数归零，生成结果与 `logprobs` 完全恢复正常。

---

## 一、问题现象与复现环境

| 维度 | 详细配置 / 现象描述 |
|---|---|
| **硬件环境** | 单机 `4 × Tesla V100-SXM2-32GB (SM70)`（容器节点 `10.155.96.34:30965`） |
| **模型与量化** | `Qwen3.8-Flash-Next`（`compressed-tensors` NVFP4 权重，FP16 激活，`--kv-cache-dtype auto`） |
| **并行配置** | `--tensor-parallel-size 4 --decode-context-parallel-size 2`（`TP4 + DCP2`） |
| **故障现象 1（Logprobs 报错）** | 请求携带 `logprobs=True` 时，服务端返回 HTTP `400 Bad Request`：`{"error":{"message":"Out of range float values are not JSON compliant: nan"}}` |
| **故障现象 2（输出坍缩 `"duct"`）** | 不带 `logprobs` 时返回 HTTP `200`，但无论输入何种多 Token Prompt，输出恒为 `"ductductduct..."` 重复至 `max_tokens` 上限（`finish_reason: length`） |
| **故障现象 3（MTP 全拒）** | 开启 MTP4 时，主模型验证阶段的 `target_prob` 含 `NaN`，导致投机采样接受率恒为 `0.0%`（`Drafted: 172, Accepted: 0`） |
| **特殊边界现象** | 极短 Prompt（仅 `1` 个 Token，即 `m == 1`）偶发可正常返回，而所有多 Token Prompt（`m > 1`，如 `17`、`24`、`2500+` tokens）100% 触发 `NaN` |

---

## 二、三轮渐进式排查路径复盘

### 2.1 第一轮：控制变量黑盒排查（排除外围模块的干扰）

在第一轮排查中，通过三组严格的控制变量实验，先后排除了 KV 量化、MTP 草稿模型以及短上下文分支：
1. **排除 `fp8_e4m3` 无校准 Scale 问题**：将 `--kv-cache-dtype fp8_e4m3` 切换为 `--kv-cache-dtype auto`（纯 FP16 KV Cache），`NaN` 与 `"duct"` 依旧复现。
2. **排除 MTP4 投机解码路径**：完全去掉 `--speculative-config`，仅运行单模型 `TP4 + DCP2`，`NaN` 与 `"duct"` 依旧复现。
3. **排除短上下文 `D22`（`seq_lens <= 2048` 跳过 Indexer Top-K）**：构造 `2500+` tokens 的长上下文 Prompt 强行绕过 `D22`，`NaN` 与 `"duct"` 依旧复现。

**第一轮结论**：问题位于**主模型 Backbone 在 `m > 1` 时的基础前向计算链路**中。

---

### 2.2 第二轮：编译模式（`torch.compile`）下全模型 `nan_diag` 插桩与“编译短路”现象

在第二轮中，向模型 11 个关键文件植入了 `nan_diag.check()` 探针（受环境变量 `QWEN38_NAN_DIAG=1` 控制）。但在默认的 `torch.compile` 模式下运行后，出现了看似矛盾的统计结果：
- **主路径探针全部为 `0` 次触发**：`model.py`（`layer{i}_after_attn`、`layer{i}_after_mlp`）、`qwen_gdn_linear_attn.py`（`gdn_in_hidden`、`gdn_out`）、`layernorm.py`（`rmsnorm_*`）、`hyperconnection.py`（`hc_cap_*`）均为 **`0` 条日志**；
- **仅自定义算子（Custom Op）内部触发 `NaN`**：`ops/hc.py`（`cn_out=336`、`cn_res=332`、`cn_block=284`）、`nvfp4_sm70_moe.py`（`moe_in_x=168`）、`qsa.py`（`qsa_in_hidden=88`）、`indexer_qsa.py`（`indexer_in=44`）大量报错。

**原因剖析（为什么编译模式下主路径探针被跳过）**：
- `nan_diag.py` 的入口保护逻辑为：
  ```python
  if not _ENABLED or torch.compiler.is_compiling():
      return
  ```
- 在默认开启 `torch.compile` 时，`Qwen4ExpModel.forward`、`Qwen4ExpDecoderLayer.forward`、`QwenGatedDeltaNetAttention.forward`、`RMSNormGated.forward` 等常规 `nn.Module` 方法在 TorchDynamo 追踪构图阶段（此时 `torch.compiler.is_compiling() == True`）被内联追踪。`nan_diag.check()` 在追踪期直接 `return`，**根本没有被编译进生成的 FX 图与 Inductor 内核中**。
- 而在实际运行期（此时 `torch.compiler.is_compiling() == False`），只有通过 `direct_register_custom_op` 注册为图外不透明节点（Opaque Custom Op）的 Python 回调函数（即 `ops/hc.py`、`nvfp4_sm70_moe.py`、`qsa.py:_run_qsa`、`indexer_qsa.py`）会回到 Python 解释器执行。因此第二轮捕获到的全部是**下游扩散后的结果**，真正的源头被 `torch.compile` 隐藏了。

---

### 2.3 第三轮：Eager 模式（`--compilation-config '{"mode":"NONE"}'`）一击锁定第一现场

为彻底消除 `torch.compile` 对探针的短路效应，第三轮在启动参数中加入 `--compilation-config '{"mode":"NONE"}'` 强制以纯 Eager 模式运行，在 11 个关键文件布控：
- **主路径**：`model_embed`、`ple_in`、`layer_in_hidden`、`layer_attn_out`、`layer_out_hidden`、`layer_out_mlp`、`moe_router`、`moe_out`
- **HC 融合**：`combine_res/block/inj`、`cn_res/cn_block/cn_inj/cn_out/cn_y`、`gate_mix_x/gate_mix_g`
- **GDN 线性注意力**：`gdn_proj_in`（即 `core_attn_out`）、`gdn_proj_z`、`gdn_proj_out`
- **QSA 稀疏注意力**：`qsa_in_hidden/query/key`、`indexer_in`

在收集到的 35.5 万行 `[NAN_DIAG]` 日志中，按时间序列**全模型第一个触发的 `NaN` 探针**被精确捕获：

```text
[NAN_DIAG] layer=language_model.model.layers.2.linear_attn gdn_proj_z | nan=96 inf=1 | shape=[24, 12, 128]
```

（在 24 tokens 的 Prefill 阶段触发；4 个 TP Worker 捕获到的稀疏异常元素数分别为：`97`、`96 NaN + 1 Inf`、`103`、`78`）。

**三个决定性证据**：
1. **同层输入完全干净**：`layers.2.linear_attn` 的 `gdn_proj_in`（即 GDN 核心递推输出 `core_attn_out`）为 **0 条 `NaN`**，证明污染源不是 GDN 状态递推，而是门控分支张量 **`z`（`gdn_proj_z`）本身**！
2. **层间传播起点明确**：`layer_in_hidden` 在 `layer 0`、`layer 1`、`layer 2` 均为 **0 条 `NaN`**，从 **`layer 3` 起变为全 `NaN`**——说明 `layer 2` 的 `gdn_proj_z` 就是全模型第一个产生 `NaN` 的源头！
3. **稀疏未初始化特征明显**：`z` 的形状为 `[24, 12, 128]`（共 `36,864` 个 FP16 元素），其中仅有 `96` 个 `NaN` 和 `1` 个 `Inf`（占比仅 **`0.26%`**），这是典型的**未初始化显存垃圾（`torch.new_empty` 残留旧显存位模式）**，而非算术计算溢出。

### 2.4 完整 `NaN` 扩散链

```text
Prefill (m = 24 tokens):
  layers.2.linear_attn:
    z = mixed_qkvz.new_empty((24, 12, 128))  --> 未被内核写入，残留 96 NaN + 1 Inf 显存垃圾 (0.26%)
    └─> self.norm(core_attn_out, z)          --> RMSNormGated: norm(core_attn_out) * silu(z)，含 96 个稀疏 NaN
    └─> self.out_proj(...) [24, 1536] @ [1536, 2560] + TP4 AllReduce
                                             --> 稠密矩阵乘沿 1536 维求和，凡含任一 NaN 的行整行 2560 维全变 NaN！
    └─> gdn_proj_out [24, 2560] 全 NaN
    └─> layer_attn_out [24, 2560] 全 NaN
    └─> layers.2.mlp_hc (cn_block -> cn_out -> cn_y -> gate_mix_x / gate_mix_g [24, 10240] 全 NaN)
    └─> layers.2.mlp (moe_layer_in -> moe_in_x -> moe_out 全 NaN)
  layers.3 .. layers.47:
    └─> layer_in_hidden 全 NaN（同时污染各层写入的 QSA KV Cache 与 GDN Mamba 状态）
  lm_head:
    └─> 顶层 logits / logprobs 全 NaN
        ├─> 开启 logprobs 时：JSON 序列化抛 HTTP 400 ("Out of range float values are not JSON compliant: nan")
        ├─> 贪心采样时：NaN 比较恒 False，落回固定兜底 token ("duct" × 80)
        └─> 开启 MTP4 时：target_prob 为 NaN，草稿接受率恒为 0.0%
```

---

## 三、代码级根因剖析与 GPU 实证

### 3.1 第一现场代码定位

#### （1）调用方：`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`（L3090–3115 及 L5866–5879）

在合入 **D16** 优化后，Prefill 阶段（`m > 1`）的 GDN 输入投影拆分进入 `_split_gdn_projection_tails` 分支：

```python
            if (
                not self.gqa_interleaved_layout
                and not self.disable_tp_for_ba_proj
                and mixed_qkvz.shape[0] > 1
                and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
                and self.value_dim // self.tp_size == 1536
                and _can_fuse_gdn_projection_split(mixed_qkvz, ba)
            ):
                z = mixed_qkvz.new_empty(
                    (
                        mixed_qkvz.shape[0],
                        (self.value_dim // self.tp_size) // self.head_v_dim,
                        self.head_v_dim,
                    )
                )
                mixed_qkv, b, a = _split_gdn_projection_tails(mixed_qkvz, ba, z)
```

注意：此处 `z` 是通过 **`mixed_qkvz.new_empty((m, 12, 128))`** 分配的**未初始化显存缓冲区**，完全依赖 `_split_gdn_projection_tails` 将 `mixed_qkvz[:, 2560:4096]` 的 1536 列数据拷贝进 `z`。

#### （2）被调方与 Bug 现场：`vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py`（L72–136）

在服务器上，`_qwen38_gdn_projection_split_kernel` 的内核实现为：

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
```

然而，启动该内核的 `_split_gdn_projection_outputs`（L103）和 `_split_gdn_projection_tails`（L124）却将 grid 写成了：

```python
# ❌ 错误的启动网格：只启动了 cdiv(1536, 256) = 6 个线程块！
_qwen38_gdn_projection_split_kernel[(m, triton.cdiv(1536, 256))](
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
```

### 3.2 数学证明：为什么 6 个线程块会导致 `z` 100% 未被写入？

- 内核参数为 `QKV = 2560`、`Z = 1536`、`BLOCK = 256`，总列数为 `QKV + Z = 4096`。
- 当 grid 第二维为 `triton.cdiv(1536, 256) = 6` 时：
  - `tl.program_id(1)` 的取值范围仅为 **`{0, 1, 2, 3, 4, 5}`**；
  - `col = tl.program_id(1) * 256 + tl.arange(0, 256)` 的取值范围仅为 **`[0, 1536)`**。
- 内核中写入 `z` 的语句为：
  ```python
  tl.store(z + row * Z + col - QKV, value, (col >= QKV) & (col < QKV + Z))
  ```
  其掩码条件要求 **`col >= 2560` 且 `col < 4096`**（对应 `tl.program_id(1) ∈ {10, 11, 12, 13, 14, 15}`）。
- 因为对所有已启动的 6 个线程块都有 `col < 1536`，所以 **`(col >= 2560)` 恒为 `False`**！
- **后果**：
  1. 在 `_split_gdn_projection_tails`（`COPY_QKV=False`）中，`z_out` **没有任何一个元素被写入**（100% 保持 `new_empty` 的未初始化显存状态）；
  2. 在 `_split_gdn_projection_outputs`（`COPY_QKV=True`）中，`qkv` 仅写入了前 `1536` 列（后 `1024` 列未初始化），`z` 同样 **100% 未被写入**！

---

### 3.3 四个关键历史现象的完整解答

1. **为什么 `m == 1`（1-token 短 Prompt）偶发正常？**
   - `_can_fuse_gdn_projection_split`（L150）与 `qwen_gdn_linear_attn.py`（L3108）均包含条件 `qkvz.shape[0] > 1`。当 Prompt 仅有 1 个 Token（`m == 1`）时，不会调用 `_split_gdn_projection_tails`，而是回退到 PyTorch 原生切片 `mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)`，因此 `z` 数据正常。
2. **为什么 `layers.0` 和 `layers.1` 同样是 GDN 层，但首现 `NaN` 却在 `layers.2`？**
   - 在 `layers.0` 和 `layers.1` 中，`z` **同样没有被写入**！区别仅在于：前向传播刚开始执行 `layers.0` 和 `layers.1` 时，`mixed_qkvz.new_empty((24, 12, 128))` 从 PyTorch CUDA Caching Allocator 分配到的是刚向驱动申请并清零的干净显存页（内部全为 `0.0` 或有限浮点数——虽然不触发 `isnan` 报警，但已经使前两层 GDN 输出变为错误数值）；
   - 当执行到 `layers.2.linear_attn` 时，由于 `layer 0` / `layer 1` 的 MoE、PLE、HC 等模块已经申请并释放了大量临时工作区显存，`new_empty((24, 12, 128))` 复用了带有随机位模式的旧显存块。在 IEEE-754 FP16 格式（1 位符号 + 5 位指数 + 10 位尾数）中，5 位指数全为 `1`（`0b11111`）且尾数非零的旧显存字节即被硬件解释为 `NaN`（占 `36,864` 个元素的 `0.26%`，即 `96` 个 `NaN` + `1` 个 `Inf`）。
3. **为什么 `z` 中仅 `0.26%` 的稀疏 `NaN` 会导致 `gdn_proj_out` 整张量全 `NaN`？**
   - `z` 首先在 `self.norm(core_attn_out, z)`（`RMSNormGated`）中逐元素执行 `norm(core_attn_out) * silu(z)`，此时输出仍只有 `96` 个稀疏 `NaN`；
   - 紧接着执行 `self.out_proj`（`[24, 1536] @ [1536, 2560]` 的稠密矩阵乘法 + TP4 `all_reduce`）。由于矩阵乘法中每个输出元素都对全部 `1536` 个输入通道求和，只要某一行（某个 Token）的 `1536` 个通道中含有哪怕 `1` 个 `NaN`，该 Token 对应的全部 `2560` 维输出以及跨卡 `all_reduce` 结果就会**整行变为 `NaN`**！

---

### 3.4 GPU 独立脚本实证结果

编写独立 GPU 验证脚本直接调用 `_split_gdn_projection_tails` 与 `_split_gdn_projection_outputs`，并在调用前将输出缓冲区 `z` 填充为哨兵值 `-999.0`：

| 测试函数 | 修复前（`grid = cdiv(1536, 256) = 6`） | 修复后（`grid = cdiv(4096, 256) = 16`） |
|---|---|---|
| `_split_gdn_projection_tails(qkvz, ba, z)` | `z` 中 `-999.0` 哨兵值 **100% 原样保留（`36864 / 36864`）**，证明 `z` 从未被写入 | `z`、`qkv`、`b`、`a` 与参考切片 **100% 逐位一致（`allclose = True`）** |
| `_split_gdn_projection_outputs(qkvz, ba)` | `qkv` 仅前 `1536` 列被写入（后 `1024` 列未写），`z` **完全未写** | `qkv`（`2560` 列）、`z`（`1536` 列）、`b`、`a` **全部正确（`allclose = True`）** |

---

## 四、代码修复方案

### 4.1 服务器快捷修复命令

在服务器环境 `/home/ma-user/work/1cat-vllm-latest` 下执行：

```bash
sed -i 's/triton.cdiv(1536, 256)/triton.cdiv(4096, 256)/g' \
  /home/ma-user/work/1cat-vllm-latest/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py

python3 -m py_compile \
  /home/ma-user/work/1cat-vllm-latest/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
```

### 4.2 修复前后的完整代码对比（`vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py`）

```diff
--- a/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
+++ b/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
@@ -100,7 +100,7 @@ def _split_gdn_projection_outputs(qkvz, ba):
     m = qkvz.shape[0]
     out = tuple(qkvz.new_empty((m, n)) for n in (2560, 1536, 12, 12))
-    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(1536, 256))](
+    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
         qkvz,
         ba,
         *out,
@@ -121,7 +121,7 @@ def _split_gdn_projection_tails(qkvz, ba, z_out):
     m = qkvz.shape[0]
     qkv = qkvz[:, :2560]
     b, a = (qkvz.new_empty((m, 12)) for _ in range(2))
-    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(1536, 256))](
+    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
         qkvz,
         ba,
         qkv,
```

### 4.3 修复后的标准函数实现（`vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` L72–140）

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

## 五、线上实跑验证结果

修复 `sm70_fp16_gemv.py` 并重启 vLLM 服务后，在线发送短指令与开放式长生成请求进行端到端验证：

| 测试用例 | 修复前表现 | 修复后表现 | 全链路 `NAN_DIAG` 触发数 |
|---|---|---|---|
| `"Hi, please repeat exactly: apple banana"` | HTTP 400 (`Out of range float values are not JSON compliant: nan`) / `"duct"` | **HTTP 200**，`content = 'apple banana'`，logprobs 全部为正常有限浮点数 | **0 次** |
| `"Write a short poem about the ocean..."` | HTTP 400 `NaN` / `"duct" × 80` | **HTTP 200**，正常生成 3 行英文诗歌（26 tokens），语义连贯 | **0 次** |

服务端日志执行 `grep -c 'NAN_DIAG'` 统计结果为 **`0`**，确认全模型 48 层（含全部 GDN、QSA、HyperConnection、MoE、PLE 模块）无任何 `NaN` 或 `Inf` 产生。

---

## 六、容器环境运维与进程清理备注

在容器环境内重启 vLLM 服务时，切勿仅使用 `fuser -k 7866/tcp`：
- **原因**：`fuser -k 7866/tcp` 只能杀掉监听 HTTP 端口的前端 `APIServer` 进程，而底层的 **`VLLM::EngineCore`** 与 **4 个 `VLLM::Worker_TP*` 进程**会作为孤儿进程残留在后台，继续占满每张 V100 约 **~29 GB** 显存；此时直接重启服务会触发 `"Free memory on device ... is less than desired GPU memory utilization"` 报错。
- **注意容器 PID Namespace 差异**：`nvidia-smi` 输出的进程 PID 是**宿主机（Host）PID Namespace** 下的编号，在容器内直接 `kill` 该编号会提示进程不存在。
- **标准清理命令（在容器内执行）**：
  ```bash
  ps -ef | grep -E "VLLM::EngineCore|VLLM::Worker_TP|vllm.entrypoints" | grep -v grep | awk '{print $2}' | xargs -r kill -9
  ```
  执行后通过 `nvidia-smi` 确认 4 张 V100 显存已回落至 `0 MiB`，再启动新的 vLLM 实例。
