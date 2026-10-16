# `Qwen3.8-Flash-Next` QSA 草稿块表展开 `torch.mul(..., out=expanded_view)` 静默 Resize 崩溃修复方案（`D28`）

---

## 1. 错误原因（一句话版）

**`torch.mul(..., out=...)` 的 `out=` 参数不会将计算结果向 `out` 张量自动广播，而是将 `out`（`expanded_view`）静默 resize 成左操作数的形状 `[rows, columns, 1]`，进而导致后续基于 `expanded_view` 的子块偏移计算或广播加法在 `.view(1, 1, expansion)` / `add_` 时崩溃。**

---

## 2. 报错现象与日志特征

在 `TP4 + DCP2 + MTP4`（或主模型 `fp8_e4m3` + 草稿 `float16` 等 `expansion = 2` 的场景）下，构建 MTP 草稿层 QSA Attention Metadata 时会先打印一条 PyTorch 静默 resize 告警，随后立即抛出形状不匹配 `RuntimeError`：

```text
UserWarning: An output with one or more elements was resized since it had shape [4, 8211, 2], which does not match the required output shape [4, 8211, 1]. This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor by resizing it, inplace, to zero elements with t.resize_(0).
...
RuntimeError: shape '[1, 1, 2]' is invalid for input of size 1
```
（或在直接使用预构造 `[1, 1, 2]` 偏移张量时报错：`RuntimeError: output with shape [4, 8211, 1] doesn't match the broadcast shape [4, 8211, 2]`。）

---

## 3. 详细根因链路剖析

在 `vllm/models/qwen4_exp/nvidia/qsa.py` 的草稿块表展开逻辑（`expansion = 2`）中，问题代码为：

```python
expanded_view = expanded.view(rows, columns, expansion)        # 或 expanded.unflatten(1, (columns, expansion))，形状为 [4, 8211, 2]
torch.mul(table.unsqueeze(-1), expansion, out=expanded_view)   # ← 元凶
sub_block_offsets = self._draft_sub_block_offsets(expanded_view)  # 返回 size 1
expanded_view.add_(sub_block_offsets.view(1, 1, expansion))    # ← 崩溃
```

### 根因链条（4 步推导）：
1. **左操作数与标量相乘后的输出形状仍为 `[4, 8211, 1]`**：
   - `table` 的形状为 `[rows, columns]`（即 `[4, 8211]`），`table.unsqueeze(-1)` 的形状为 `[4, 8211, 1]`；
   - 将 `[4, 8211, 1]` 与 Python 标量 `expansion`（`2`）相乘，PyTorch 推导出的算子输出形状依然是 **`[4, 8211, 1]`**。
2. **`out=expanded_view` 只做 `resize_`、不做向 `out` 的反向广播**：
   - 传入的 `out=expanded_view` 形状是 `[4, 8211, 2]`；
   - PyTorch 的 `torch.mul(a, b, out=c)` 语义规定：算子输出形状仅由输入参数 `(a, b)` 的广播结果决定，当 `c.shape != broadcast(a, b).shape` 时，PyTorch **不会**把结果广播到 `c.shape`，而是触发 `UserWarning` 并**原地将 `expanded_view` 静默 `resize` 成 `[4, 8211, 1]`**！
3. **`expanded_view` 最后一维被篡改为 `1`**：
   - 被静默 resize 后，`expanded_view.shape[-1]` 从 `2` 变成了 `1`；
   - 若后续调用 `_draft_sub_block_offsets(expanded_view)` 内部用 `target.shape[-1]` 推导 `expansion`，就会错误得到 `torch.arange(1)`（元素个数为 `1`）。
4. **在 `.view(1, 1, expansion)` 或 `add_` 时崩溃**：
   - 外层局部变量 `expansion` 仍为 `2`，对大小为 `1` 的 `sub_block_offsets` 执行 `.view(1, 1, 2)` 需要 2 个元素，立即触发 `RuntimeError: shape '[1, 1, 2]' is invalid for input of size 1`。

---

## 4. 修复方案（`D28`）

### 4.1 设计意图与零拷贝修复原理
该段代码的原本数学意图是：
$$\text{expanded\_view}[r, c, k] = \text{table}[r, c] \times \text{expansion} + k \quad (k = 0, \dots, \text{expansion}-1)$$
即把每个调度页号乘上 `expansion` 倍，再加上子块偏移 `0..expansion-1`。

只需在调用 `torch.mul(..., out=expanded_view)` 前，先对 `table.unsqueeze(-1)` 调用零拷贝视图扩展 `.expand(rows, columns, expansion)`（只修改 stride=0，不分配任何显存、不发生数据拷贝），使左操作数形状直接等于 `[rows, columns, expansion]`（`[4, 8211, 2]`）：
- 此时 `torch.mul` 的计算结果形状即为 `[4, 8211, 2]`，与 `out=expanded_view` 的形状 `[4, 8211, 2]` 完全一致；
- 既不会触发 `out=` 的静默 resize 和 `UserWarning`，也无需额外启动一次 `copy_` 内核。

---

### 4.2 代码修改 Diff（`vllm/models/qwen4_exp/nvidia/qsa.py`）

#### 方案 A（针对本分支 `Qwen4ExpQSAMetadataBuilder.build` 的修改）：

```diff
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -165,7 +165,11 @@ class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
                     expansion, dtype=table.dtype, device=table.device
                 ).view(1, 1, expansion)
                 self._draft_sub_block_offsets = sub_block_offsets
-            torch.mul(table.unsqueeze(-1), expansion, out=expanded_view)
+            torch.mul(
+                table.unsqueeze(-1).expand(rows, columns, expansion),
+                expansion,
+                out=expanded_view,
+            )
             expanded_view.add_(sub_block_offsets)
             common_attn_metadata = common_attn_metadata.replace(
                 block_table_tensor=expanded
```

#### 方案 B（针对包含 `_build_draft_attn_metadata` / `_draft_sub_block_offsets` 辅助函数版本的修改）：

```diff
--- a/vllm/models/qwen4_exp/nvidia/qsa.py
+++ b/vllm/models/qwen4_exp/nvidia/qsa.py
@@ -176,7 +176,11 @@ class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
             expanded = self.draft_block_table_buffer[:rows, : columns * expansion]
             expanded_view = expanded.view(rows, columns, expansion)
-            torch.mul(table.unsqueeze(-1), expansion, out=expanded_view)
+            torch.mul(
+                table.unsqueeze(-1).expand(rows, columns, expansion),
+                expansion,
+                out=expanded_view,
+            )
             sub_block_offsets = self._draft_sub_block_offsets(expanded_view)
             expanded_view.add_(sub_block_offsets.view(1, 1, expansion))
```

---

### 4.3 修改后的完整代码段（`vllm/models/qwen4_exp/nvidia/qsa.py`）

```python
        expansion = draft_blocks_per_page // group_blocks_per_page
        if expansion > 1:
            table = common_attn_metadata.block_table_tensor
            rows, columns = table.shape
            if self.draft_block_table_buffer is None:
                self.draft_block_table_buffer = torch.empty(
                    (
                        self.vllm_config.scheduler_config.max_num_seqs,
                        columns * expansion,
                    ),
                    dtype=table.dtype,
                    device=table.device,
                )
            if (
                rows > self.draft_block_table_buffer.shape[0]
                or columns * expansion > self.draft_block_table_buffer.shape[1]
            ):
                raise RuntimeError("QSA draft block-table buffer is too small")
            expanded = self.draft_block_table_buffer[:rows, : columns * expansion]
            expanded_view = expanded.unflatten(1, (columns, expansion))
            sub_block_offsets = getattr(self, "_draft_sub_block_offsets", None)
            if (
                sub_block_offsets is None
                or sub_block_offsets.shape[-1] != expansion
                or sub_block_offsets.device != table.device
                or sub_block_offsets.dtype != table.dtype
            ):
                sub_block_offsets = torch.arange(
                    expansion, dtype=table.dtype, device=table.device
                ).view(1, 1, expansion)
                self._draft_sub_block_offsets = sub_block_offsets
            torch.mul(
                table.unsqueeze(-1).expand(rows, columns, expansion),
                expansion,
                out=expanded_view,
            )
            expanded_view.add_(sub_block_offsets)
            common_attn_metadata = common_attn_metadata.replace(
                block_table_tensor=expanded
            )
            metadata.block_table = expanded
```
