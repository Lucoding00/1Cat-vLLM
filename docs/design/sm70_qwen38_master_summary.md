# SM70 (Tesla V100) `Qwen3.8-Flash-Next` 全栈性能优化与多卡部署总览手册（三合一汇总版）

---

## 1. 三个现有设计文档的可用性说明与关系总览

### 1.1 之前三个文档还能继续使用吗？
**完全可以继续使用，且全部有效。**
本分支（`arena/5368f936-1cat-vllm`，基于 `main` 分支提交 `be3722bc095b9792bf9833068595f0b1ef70ad37`）上的三个专项文档分别记录了我们在三个阶段完成的优化与修复，它们与当前分支代码 **100% 对应且无任何冲突**：

| 专项文档路径 | 定位与涵盖内容 | 当前状态 |
| :--- | :--- | :--- |
| **`docs/design/sm70_qwen38_prefill_decode_optimizations.md`** | **第一阶段（Bit-Exact 算子融合、零拷贝与持久工作区优化）**：涵盖 **`P1`–`P10`**（Prefill & MTP4 10 项优化）与 **`D1`–`D16`**（`TP4 + DCP2` Decode 16 项优化）的详细设计说明与阶段一 Diff。 | **完全可用**（保留第一阶段全部细节） |
| **`docs/design/sm70_qwen38_capability_breakthroughs.md`** | **第二阶段（能力突破与 `Qwen3.8-27B` vs `Flash-Next` 架构对比）**：涵盖 **`P11`–`P13`** 与 **`D17`–`D26`** 的能力突破、27B 与 Flash-Next 速度差异根因分析，以及阶段二增量 Diff。 | **完全可用**（保留第二阶段全部细节） |
| **`docs/design/sm70_qwen38_dp2_tp4_dcp2_fix.md`** | **第三阶段（8 卡 `DP2 × TP4 + DCP2 + MTP4` 修复与 `fp8_e4m3` 生产启动指南）**：涵盖 `DP2 × TP4` 报错根因与修复（`D26`）、MTP 草稿 `kv_cache_dtype` 继承修复（`D27`）、一键补丁、4 卡/8 卡启动命令。 | **完全可用**（保留部署修复与启动脚本） |

### 1.2 为什么要将三个文档汇总成本文档（`docs/design/sm70_qwen38_master_summary.md`）？
由于 `vllm/models/qwen4_exp/nvidia/ops/qsa.py`、`vllm/models/qwen4_exp/nvidia/qsa.py`、`vllm/models/qwen4_exp/nvidia/mtp.py`、`vllm/models/qwen4_exp/nvidia/ple_layer.py`、`vllm/v1/spec_decode/llm_base_proposer.py` 等核心文件在第一、二、三阶段中被**连续迭代增强**，分看三个文档时每个文档展示的是对应阶段的增量 Diff。

因此，本文档将三个文档的全部成果**一次性整合到一个总手册中**：
- **第 2 节**：四大核心问题与架构根因总结（为什么 `main` 分支 Prefill 只有几千 tok/s、为什么 `27B` Decode 200+ tok/s 而 `Flash-Next` 100+ tok/s、为什么 `DP2 × TP4` 在 `main` 分支报错、以及 `fp8_e4m3` + `MTP4` 草稿 KV dtype 继承链路修复 `D27`）。
- **第 3 节**：全量 **40 项优化与修复（`P1`–`P13` + `D1`–`D27`）** 完整对照总表（与 `README.md` 及三个专项文档编号严格一致）。
- **第 4 节**：生产级最佳启动脚本汇总（**`4 × V100 32GB`** 与 **`8 × V100 32GB`**，支持 `256K + 多模态 + 工具调用 + MTP4 + fp8_e4m3 / auto`）。
- **第 5 节**：相对于 `main` 分支（`be3722bc09`）的 **31 个源码/测试文件最终合并版 Unified Diff**（每个文件只有一份完整合并 Diff，无需跨文档拼接）。
- **第 6 节**：从 `main` 分支一键导出/应用全量补丁命令。

---

## 2. 四大核心问题诊断与架构剖析汇总

### 2.1 为什么在 `main` 分支直接部署 `Qwen3.8-Flash-Next` 时 Prefill 只有几千 tok/s？
1. **根因一（最致命）：默认开启了 PLE Disk-Mmap 磁盘文件映射卸载（`ple_disk_cascade: true`）**
   - `Qwen3.8-Flash-Next` 含 4 层 PLE（每层含 4 个 HyperConnection 分支，词表 `159,744 × 10,240` FP16，单份整表约 **11.92 GiB**，4 个 TP 进程共 **47.68 GiB**）。
   - 在 `main` 分支中，`vllm/config/kernel.py:137` 默认 `ple_disk_cascade: True`；当未显式指定足够的 `VLLM_QWEN4EXP_PLE_HOST_GIB` 时，`ple_layer.py:1653` 会将溢出层落盘为 `/tmp` 下的 `mmap` 文件。
   - 长上下文 Prefill 一次访问数千个不同 N-gram 词表行，直接引发 Linux Page Cache 抖动与磁盘同步缺页中断，把 GPU 算子阻塞在磁盘 I/O 上，导致 Prefill 暴跌到几千 tok/s。
   - **解决方式**：设置 `VLLM_PLE_CPU_OFFLOAD=0 VLLM_PLE_DISK_OFFLOAD=0 VLLM_SM70_QWEN38_HYBRID_PLE=0 VLLM_QWEN4EXP_PLE_HOST_GIB=12`（4 卡）或 `10`（8 卡）并配合 `--kernel-config '{"ple_disk_cascade":false}'`，让全部 PLE 表常驻 Pinned Host RAM 走 UVA 零缺页直读（或内存受限时用 `VLLM_PLE_CPU_OFFLOAD=1` + `VLLM_PLE_OFFLOAD_PREFAULT=1` 共享单份 14.3 GiB 内存池）。
2. **根因二：Chunked Prefill 中间块在开启 `MTP4` 时白跑了 3 步自回归 Draft Decode（`P7` 已修复）**
   - 256K 上下文按 8192 分块 Prefill 时，前 `N-1` 个中间块并不产出最终采样 token，但 `main` 分支在每个中间块后依然完整执行了 MTP step 0 `lm_head` 投影 + 采样以及 step 1~3 的 3 步 Draft 自回归解码与 D2H 拷贝。`P7` 在 `first_forward` 更新完 KV/Mamba 状态后立即提前退出。
3. **根因三：GDN `RMSNormGated`（`rows > 192`）、`causal_conv1d` 跨步访存与 PLE Prefill 逐行循环（`P6`/`P9`/`P13` 已修复）**
   - 开启 MTP4 后，36 层 GDN 的 `RMSNormGated` 在 `rows > 192` 时退回 360 次 FP32 Eager ATen 算子启动（每 8K 块产生 `~27.1 GB` HBM 读写）；`causal_conv1d_fn` 因切片步幅导致 16 KB 非合并访存；PLE Prefill 对 `m > 32` 退回 104 次算子启动。本分支通过 `P6`、`P9`、`P13` 彻底消除了这些瓶颈。

---

### 2.2 为什么 `Qwen3.8-27B` Decode 能达到 200+ tok/s，而 `Qwen3.8-Flash-Next` 在 100+ tok/s？

| 对比维度 | `Qwen3.8-27B-NVFP4 + DFlash2` | `Qwen3.8-Flash-Next-NVFP4 + MTP4` | 架构级差距根因 |
| :--- | :--- | :--- | :--- |
| **投机解码机制** | **DFlash2 块并行投机**：单次前向同时生成 4 个草稿 token（1 次 Draft Forward） | **MTP4 串行自回归投机**：4 个草稿 token 需串行跑 4 步 `Qwen4ExpMultiTokenPredictor`（4 次 Draft Forward） | MTP4 草稿阶段存在 4 倍串行启动开销与跨卡 AllReduce/AllGather |
| **主干层结构** | 48 层标准 Dense GQA + SwiGLU | **48 层（36 GDN + 12 QSA）× 512 专家 MoE（Top-10 + Shared Expert）+ 4 路 HyperConnection + 4 层 PLE** | 每层算子数量多出 **3~4 倍**，包含大量小 M 形状 GEMV、路由、索引与跨 PCIe/NVLink 查表 |
| **注意力与索引开销** | 标准 Paged Attention | **12 层 QSA 需先跑 `QSAIndexer`（`index_qk_proj` + `q_norm/rope` + 分页打分 + `topk(2048)` + `DCP2` 局部化）** | 即使短上下文，每层 QSA 也有整套 Indexer 预处理流水线 |
| **PLE 词表与短卷积** | 无 PLE | **主干 4 层 + MTP 4 步 = 每轮 8 次 PLE N-gram 查表 + Dilated Short-Conv** | 涉及 Pinned-Host UVA 收集与 4-tap 膨胀短卷积 |

针对上述四项差距，本分支通过 **`D1`–`D25`** 重点突破了：
- **`D18` + `D22`（MTP 共享 Top-K 索引 + Key-Only 切片投影）**：在 MTP Draft 第 1–3 步直接复用第 0 步的 Top-K 物理块索引，跳过 `q_norm/q_rope` 并将 `index_qk_proj` 从 `[640, 2560]` 切片为 `[128, 2560]`（权重搬运与计算量直降 **80%**）；短上下文（`<= 2051`）下主模型 12 层 QSA 同样跳过 `wq` 投影与分页打分。
- **`D13` + `D14` + `D17`（`TP4 + DCP2` 异步重叠与动态裁剪）**：将 Query `all_gather` 提前并与 4 个 Indexer 算子异步重叠；将 Split-K 合并与 DCP 发送缓冲打包融合；利用 `valid_counts` 动态截断 DCP2 下约 50% 的无效 `-1` 尾部块。
- **`D20` + `D23` + `D24` + `D25`（MTP 算子深度融合）**：新增多请求 CUDA 投机短卷积核、补充 `shared_expert.down_proj` 专用 GEMV 计划、将 MTP `fc_embedding + fc_hidden + add` 融合为单次 Triton 内核，并用单行原地 Triton 核加速 `compact_topk_indices`。

---

### 2.3 为什么 `8 × V100` 启动 `DP2 × TP4 + DCP2 + MTP4` 会报错，如何修复（`D26`）？
在 `main` 分支启动 `--data-parallel-size 2 --tensor-parallel-size 4 --decode-context-parallel-size 2` 时会依次触发 3 个阻断问题：
1. **权重加载阶段崩溃**：`Qwen3NextSparseMoEBlock` 默认将 `DP2 × TP4` 展平为 `flatten_tp_size = 8`，触发 `NotImplementedError: SM70 TurboMind NVFP4 MoE currently supports tensor parallel sizes (1, 2, 4), plus GLM-5.3 TP8; got 8.`。
2. **MTP `dummy_run` 与 `propose` 的跨 DP `all_reduce` 次数不匹配**：当 DP0 有请求（调用 `propose()`，内含 2 次跨 DP `all_reduce`）而 DP1 空闲（调用 `dummy_run()`，原本无 `all_reduce`）时，引发跨 DP 死锁。
3. **跨 DP 同步循环吞没 `cudagraph_mode`**：`dp_utils.py` 原本把 `cudagraph_mode` 参与跨 DP 最小值归约并强制空闲副本循环陪跑，在未开启 EP 时完全多余且会破坏 CUDA Graph 步幅。
- **修复结论（`D26`）**：修改 `qwen3_next.py`、`llm_base_proposer.py`、`dp_utils.py`、`gpu_model_runner.py`，即可让 `8 × V100` 原生稳定运行 `DP2 × TP4 + DCP2 + MTP4`。

---

### 2.4 KV Cache 量化选型与 MTP 草稿 `kv_cache_dtype` 继承修复（`D27`）

| `--kv-cache-dtype` 选项 | 实际存储格式（36 层 GDN + 12 层 QSA） | 与 `MTP4` 兼容性 | 显存节省与速度特点 | 推荐场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`fp8_e4m3`**（**推荐**） | 36 层 GDN 状态保持紧凑页；**12 层 QSA 主 KV 压缩为 8-bit `fp8_e4m3`（1 byte/elem）** | **原生支持 `<主模型 E4M3 + MTP Draft FP16>` 混布** | **QSA KV 显存减半（省 50%），Decode 阶段 QSA 读 KV 的 HBM 带宽减半**；V100 上使用专用整型位运算反量化内核 | **4 卡 / 8 卡 256K 长上下文 + MTP4 首选配置** |
| **`auto`**（或 `float16`） | 36 层 GDN + 12 层 QSA 全部使用原生 **FP16（2 bytes/elem）** | 完全兼容 | 零量化/反量化指令开销，短上下文极速 | 对数值精度要求 100% 原生 FP16 的场景 |
| **`turboquant_k8v4`** | 12 层 QSA 压缩为 **Key INT8 + Value INT4（平均 ~6 bit/elem）** | **不支持与 MTP 同时开启**（`qsa.py:424` 硬性拦截） | 显存压缩率最高（比 FP16 省 ~62.5%），但仅限无 MTP 场景 | 不开 MTP、追求单机最大并发序列数的场景 |

#### 为什么在 `main` 分支指定了 `--speculative-config '{"kv_cache_dtype":"float16"}'` 仍会报 `Loaded 0/2`？（已由 `D27` 修复）
- **一句话总结**：MTP 草稿模型的 KV cache 数据类型（`dtype`）继承错了，被设成了主模型的 `fp8_e4m3`，而草稿层没有对应的 FP8 缩放因子（`scale`）。
- **细节链路**：
  1. 启动参数里，主模型 KV cache 用 `fp8_e4m3`，并明确指定了草稿模型 KV cache 用 `float16`（`--speculative-config '{"method":"mtp","num_speculative_tokens":4,"kv_cache_dtype":"float16"}'`）。
  2. 但 `vllm/models/qwen4_exp/nvidia/mtp.py`（及 `amd/mtp.py`）里 `_make_draft_vllm_config()` 函数在构造草稿模型的配置时，用 `replace()` 从主配置整体继承，漏掉了 `speculative_config.kv_cache_dtype="float16"` 这一项。
  3. 结果草稿模型拿到的 `cache_dtype` 仍是 `fp8_e4m3`。
  4. 因为草稿 cache 被当成 FP8，加载权重时触发了 `_finalize_qsa_e4m3_scale_load(..., require_calibrated_speculative_draft=True)` 的强制校验，要求草稿层必须有 K/V 的 E4M3 scale。
  5. 但 checkpoint（模型权重文件）里没有草稿层的 K/V scale → 报 `"Loaded 0/2"` 直接拒绝启动。
- **修复方式（`D27`）**：在 `_make_draft_vllm_config()` 中检测 `speculative_config.kv_cache_dtype`，若已指定则将其覆盖至 `draft_vllm_config.cache_config.cache_dtype`（详见第 5.8、5.13、5.30 节）。

---

## 3. 全量 40 项优化与修复总览表（`P1`–`P13` + `D1`–`D27`）

### 3.1 Prefill 与 `MTP4` 优化（`P1`–`P13`）

| 编号 | 阶段 | 涉及文件 | 优化内容与精确收益 |
| :--- | :--- | :--- | :--- |
| **`P1`** | 阶段一 | `qsa_cache.py`, `indexer_qsa.py`, `ops/qsa.py` | 在 `QSAForwardMetadata` 中缓存跨层 `visible_blocks_cache`，每个 Prefill 分块消除 11/12 次 `.item()` 主机同步与 `_qsa_visible_blocks_kernel` 启动 |
| **`P2`** | 阶段一 | `nvidia/qsa.py` | 支持 3D 跨步 `output_gate` 视图直传（`flatten_gate=False`），每步消除 12–16 次 `aten::copy_` 启动，每 8K Prefill 分块节省 `302 MB` HBM 流量 |
| **`P3`** | 阶段一 | `fp4_workspace.py`, `nvfp4_sm70_moe.py` | 在 `NvFp4MoEWorkspace._eager_buffers` 中按设备缓存只读 `int32` 序列索引缓冲，消除重复构造 |
| **`P4`** | 阶段一 | `fp4_workspace.py`, `nvfp4_sm70_moe.py` | 当 `expert_map is None` 时跳过冗余的 `output.zero_()`、`permuted_idx.fill_()` 与 `topk_ids_i32.copy_()`，每 8K 分块节省 `2.04 GB` HBM 流量与 240 次算子启动 |
| **`P5`** | 阶段一 | `sm70_fp16_gemv.py`, `qwen_gdn_linear_attn.py` | 将 `_qwen38_gdn_projection_split_kernel`（`COPY_QKV=False`）网格从 `(m, 16)` 缩减至 `(m, 6)`（**减少 62.5% CTA**），并在 `M > 1` 时启用 |
| **`P6`** | 阶段一 | `layernorm.py` | 在开启 MTP4 且 `rows > 192` 时，对不透明自定义算子内部的 `RMSNormGated` 启用 `@torch.compile(dynamic=True)` 融合内核，每 8K 分块消除 **360 次 FP32 ATen 启动与 `~27.1 GB` HBM 流量** |
| **`P7`** | 阶段一 | `llm_base_proposer.py`, `gpu_model_runner.py` | 当 `_is_all_reqs_chunked_prefill()` 为真时，MTP Proposer 在完成 `first_forward` 后立即提前退出，跳过中间 Prefill 分块上浪费的 step-0 采样与 step 1~3 自回归解码 |
| **`P8`** | 阶段一 | `sm70_mtp_fc.py`, `mtp.py` | 在 MTP Prefill 阶段启用 `maybe_combine_fc` 单次 AllGather 合并 `fc_embedding + fc_hidden`（每 8K 分块省 1 次 `40 MB` AllGather 与 `270 MB` HBM 流量），并预分配静态 `topk_indices_buffer` 与 `DCP2` 缓冲 |
| **`P9`** | 阶段一 | `causal_conv1d.py` | 从切片视图分配 `out` 时保留 channel-last `(1, dim)` 步幅，消除 36 层 GDN 在 `_causal_conv1d_fwd_kernel` 与 `_fused_post_conv_kernel` 中的 16 KB 跨步非合并访存 |
| **`P10`** | 阶段一 | `ops/hc.py`, `hyperconnection.py`, `model.py` | 在 `_hcx_final_mix` 路径中支持 `_hc_combine_norm_impl` 直接原地写入 `_mtp_hidden_buffer`，每 8K 分块省去 `160 MB` 临时分配与 `320 MB` HBM 拷贝 |
| **`P11`** | 阶段二 | `ops/qsa_dcp.py`, `ops/qsa.py`, `nvidia/qsa.py` | `TP4 + DCP2` 下统计并写回每个查询行在当前 DCP 域的实际有效 token 数 `valid_counts`（`int32 [T]`），动态裁剪 `_qsa_sparse_paged_gqa_splitk_kernel` 的循环上界，跳过约 50% 的 `-1` 尾部空块 |
| **`P12`** | 阶段二 | `model.py`, `mtp.py`, `qsa.py`, `qwen_gdn_linear_attn.py` | 在 `Qwen4ExpMTP.load_weights` 启用 `_maybe_enable_sm70_peer_paths`，解除 `qsa_prep_sm70_out` 的 `num_tokens <= 32` 限制，并通过 `COPY_QKV=True` 将 GDN 输入投影尾部拷贝融合进拆分内核 |
| **`P13`** | 阶段二 | `ple_layer.py`, `qwen38_ple_spec_sm70.cu` | 新增行主序 4-tap Triton Prefill 短卷积 + 状态更新内核（8K Prefill 省 `~5.36 GB` HBM 流量），并对 `m > 32` 启用 `sm70_ple_ngram_ids` CUDA 内核（从 104 次启动降至 4 次） |

---

### 3.2 `TP4 + DCP2` Decode、`MTP4` 与 `DP2 × TP4` 优化（`D1`–`D27`）

| 编号 | 阶段 | 涉及文件 | 优化内容与精确收益 |
| :--- | :--- | :--- | :--- |
| **`D1`** | 阶段一 | `dcp_alltoall.py`, `nvidia/qsa.py` | 将 DCP All-to-All LSE 解包/合并 + FP16 舍入 + Sigmoid 输出门控直接融合写入 `output` |
| **`D2`** | 阶段一 | `ops/qsa_dcp.py`, `ops/qsa.py`, `nvidia/qsa.py` | 将 `_qsa_localize_dcp_indices_kernel` 输出存储宽度减半（`1026` vs `2051`） |
| **`D3`** | 阶段一 | `ops/qsa.py` | 预分配并按 1D 连续视图复用 Split-K `partial_out` / `partial_lse` 工作区 |
| **`D4`** | 阶段一 | `nvidia/qsa.py` | 在 DCP QSA Decode 路径跳过冗余的 `output.zero_()` |
| **`D5`** | 阶段一 | `gdn_attn.py` | 向量化展开复制草稿块表（replicated-draft block-table expansion） |
| **`D6`** | 阶段一 | `gdn_attn.py` | 在 Decode 阶段跨 6 个 GDN 组延迟计算 `compute_num_computed_tokens()` |
| **`D7`** | 阶段一 | `ops/qsa.py`, `ops/qsa_shared_key.py`, `indexer_qsa.py` | 为 Selector、Shared-Key 与 Indexer 提供持久工作区，每步 Decode 消除 90+ 次动态 GPU 分配 |
| **`D8`** | 阶段一 | `nvidia/qsa.py` | 复用持久化的 DCP `all_gather` 查询接收工作区 |
| **`D9`** | 阶段一 | `qsa_cache.py`, `indexer_qsa.py`, `ops/qsa.py` | Decode 阶段按实际存活上下文长度动态收缩 `score_columns`（削减 80%~98% 空打分 CTA），并在 Split-K 内核中加入空块短路（`tl.max(logical_token, axis=0) >= 0`） |
| **`D10`** | 阶段一 | `ops/qsa_dcp.py`, `ops/qsa.py`, `nvidia/qsa.py` | 在 `_qsa_localize_dcp_indices_kernel` 内直接融合解析物理 KV 槽位（`RESOLVE_PHYSICAL=True`） |
| **`D11`** | 阶段一 | `ops/qsa.py` | 针对 `DCP2`（`group_size == 12`）启用 2-warp SM70 轻量级部分归约启动配置 |
| **`D12`** | 阶段一 | `gpu_model_runner.py`, `model_runner.py`, `cudagraph_utils.py` | 仅对 MLA 模型准备 `dcp_local_seq_lens`，跳过 QSA 模型上的冗余元数据构造 |
| **`D13`** | 阶段一 | `ops/qsa.py`, `dcp_alltoall.py`, `nvidia/qsa.py` | 将 Split-K 合并与 DCP All-to-All 发送缓冲打包融合（`dcp_packed_out`），每步省去 12 次 `_dcp_a2a_pack_send_kernel` 启动 |
| **`D14`** | 阶段一 | `nvidia/qsa.py` | 将 `qsa_prep_sm70_out` 与异步 `all_gather_into_tensor(query, async_op=True)` 提前，与全部 4 个 Indexer 算子及 `qsa_localize_dcp_indices` 并行重叠 |
| **`D15`** | 阶段一 | `ple_layer.py` | 将 `_qwen38_ple_m1_short_conv_kernel` 扩展为 2D 网格 `(cdiv(10240, 256), num_rows)`，原生支持 MTP4 验证阶段 `M = 5` 并消除 `zeros_like` |
| **`D16`** | 阶段一 | `model.py`, `qwen_gdn_linear_attn.py` | 消除 `output_projection` 关闭时的冗余 `attn_buffer` 分配/拷贝，缓存空元数据张量，使 `mixed_qkv_recurrence` 直接写入 `core_attn_out`，并启用 `qwen38_sm70_fp16_gdn_input` |
| **`D17`** | 阶段二 | `ops/qsa_dcp.py`, `ops/qsa.py`, `nvidia/qsa.py` | `TP4 + DCP2` 下利用 `valid_counts` 在 Decode 阶段跳过空 Owner 分片与 `-1` 尾部块 |
| **`D18`** | 阶段二 | `llm_base_proposer.py`, `indexer_qsa.py`, `ops/qsa_pre_indexer.py` | 在 V1 `SpecDecodeBaseProposer` 中接入 `index_share_for_mtp_iteration`（`set_skip_topk` + `compact_topk_indices`），并在 `skip_topk or not select` 时跳过 `q_norm` 与 `q_rope`（`num_q_work = 0`） |
| **`D19`** | 阶段二 | `model.py`, `mtp.py`, `qsa.py`, `qwen_gdn_linear_attn.py` | MTP 启用 SM70 Peer 路径、融合 GDN 投影尾部拷贝并扩展 `qsa_prep_sm70_out` |
| **`D20`** | 阶段二 | `ple_layer.py`, `qwen38_ple_spec_sm70.cu` | 支持多请求（`1 <= num_reqs <= 64`）MTP4 PLE 投机短卷积（每步从 112 次启动降至 8 次）并兼容 FP16/FP32 `conv_state` |
| **`D21`** | 阶段二 | `model.py` | 修复 `Qwen4ExpModel`（`"final"`）与 `Qwen4ExpMultiTokenPredictor`（`"mtp.final"`）在 `enable_sm70_hcx` 注册表中的命名冲突 |
| **`D22`** | 阶段二 | `indexer_qsa.py`, `ops/qsa.py`, `sm70_fp16_gemv.py` | 当 `skip_query_prep` 为真时对 `index_qk_proj` 执行 Key-only 切片投影（`weight[512:]`，`[128, 2560]` vs `[640, 2560]`，计算与权重带宽减 **80%**），并在 `max_seq_len <= 2051` 时启用 `IDENTITY_BLOCKS=True` 恒等块展开（每轮省 **26 次算子启动**） |
| **`D23`** | 阶段二 | `sm70_fp16_gemv.py` | 为 `.mlp.shared_expert.down_proj`（`(2560, 160)`）添加专用 `_GemvPlan(256, 2, 0)`，每轮 `M = 1` 消除 52 次通用 cuBLAS 回退 |
| **`D24`** | 阶段二 | `sm70_mtp_fc.py` | 将 MTP Draft 的 `fc_embedding`、`fc_hidden`（4 路 HC）与逐元素相加融合为单个 Triton 内核 `_qwen38_mtp_fc_fused_gemv_add_kernel`，严格保留两次 FP16 舍入边界，每轮省 **8 次算子启动** |
| **`D25`** | 阶段二 | `mtp.py` | 对 `compact_topk_indices` 共享缓冲去重，为 `B = 1` Decode 新增单行原地 Triton 内核 `_compact_topk_single_row_kernel`，并为 `B > 1` 缓存持久 `_compact_topk_scratch` |
| **`D26`** | 阶段三 | `qwen3_next.py`, `llm_base_proposer.py`, `gpu_model_runner.py`, `dp_utils.py`, `test_weight_loading.py` | 完整修复 `8 × V100` 下 `DP2 × TP4 + DCP2 + MTP4` 启动崩溃与跨 DP 同步挂起问题 |
| **`D27`** | 阶段三 | `nvidia/mtp.py`, `amd/mtp.py`, `test_e4m3_mtp.py` | 修复 `_make_draft_vllm_config()` 漏传 `speculative_config.kv_cache_dtype` 导致草稿层误继承主模型 `fp8_e4m3` 并触发 `_finalize_qsa_e4m3_scale_load` 报 `Loaded 0/2` 拒绝启动的问题 |

---

## 4. 生产级最佳启动脚本汇总（256K + 多模态 + 工具调用 + MTP4）

### 4.1 `4 × V100 32GB` 启动脚本（`TP4 + DCP2 + MTP4 + fp8_e4m3`）
适用于 4 张 V100 32GB，开启 **256K 上下文、多模态、工具调用、MTP4 投机解码、以及主模型 `fp8_e4m3` + MTP 草稿 `float16` KV Cache**：

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ==============================================================================
# 关键配置：关闭默认 Disk-Mmap PLE，使用 Pinned-Host UVA 内存直读
# 4 个 GPU Worker 进程各分配 12 GiB Pinned Host RAM（主机共占用 ~48 GiB RAM）
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

---

### 4.2 `8 × V100 32GB` 单入口 `DP2 × TP4 + DCP2 + MTP4 + fp8_e4m3` 启动脚本
适用于在本分支（或打上第 5 节补丁后）用 8 张 V100 启动单入口 `DP2 × TP4 + DCP2 + MTP4` 服务：

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 方案 A（主机内存 >= 96 GiB 推荐）：8 个进程各分配 10 GiB Pinned Host RAM（共 ~80 GiB）
export VLLM_PLE_CPU_OFFLOAD=0
export VLLM_PLE_DISK_OFFLOAD=0
export VLLM_SM70_QWEN38_HYBRID_PLE=0
export VLLM_QWEN4EXP_PLE_HOST_GIB="${VLLM_QWEN4EXP_PLE_HOST_GIB:-10}"

# 方案 B（主机内存 < 96 GiB 时替代方案 A）：8 卡共享单份 ~14.3 GiB PLE 内存池
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

---

## 5. 相对于 `main` 分支（`be3722bc09`）的全量代码修改合并 Diff（共 31 个文件）

以下为本分支所有源码与测试文件相对于 `main` 分支基线（`be3722bc095b9792bf9833068595f0b1ef70ad37`）的**最终合并版 Unified Diff**（已将阶段一 `P1`–`P10`/`D1`–`D16`、阶段二 `P11`–`P13`/`D17`–`D25`、阶段三 `D26`–`D27` 在同一文件上的修改全部合入一处）：

### 5.1 `csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu` — CUDA 内核：SM70 PLE 多请求 MTP4 投机 Dilated Short-Conv 融合内核（P13 / D20）

```diff
diff --git a/csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu b/csrc/sm70_turbomind/ops/qwen38_ple_spec_sm70.cu
index a678cd2..20caa3d 100644
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
   half w[4];
 #pragma unroll
   for (int k = 0; k < 4; ++k) w[k] = weight[c * 4 + k];
@@ -40,12 +46,23 @@ __global__ void ple_conv(const half* x, State* state, const half* weight,
     for (int k = 0; k < 4; ++k)
       v = __fmaf_rn(__half2float(h[r + k * 3]), __half2float(w[k]), v);
     v = __half2float(__float2half_rn(v));
-    out[r * hidden + c] = r < count
-                              ? __float2half_rn(v / __fadd_rn(1.f, expf(-v)))
-                              : __float2half_rn(0.f);
+    if (r < end - start && start + r < rows) {
+      out[(start + r) * hidden + c] =
+          r < count ? __float2half_rn(v / __fadd_rn(1.f, expf(-v)))
+                    : __float2half_rn(0.f);
+    }
   }
-  for (int r = 5; r < rows; ++r) out[r * hidden + c] = __float2half_rn(0.f);
-  if (sid) {
+  for (int r = start + 5; r < min(end, rows); ++r)
+    out[r * hidden + c] = __float2half_rn(0.f);
+  if (req == 0) {
+    for (int r = 0; r < min(start, rows); ++r)
+      out[r * hidden + c] = __float2half_rn(0.f);
+  }
+  if (req == num_reqs - 1) {
+    for (int r = max(0, end); r < rows; ++r)
+      out[r * hidden + c] = __float2half_rn(0.f);
+  }
+  if (sid > 0) {
 #pragma unroll
     for (int t = 0; t < 13; ++t)
       if (t < 8 + count)
@@ -56,9 +73,9 @@ __global__ void ple_conv(const half* x, State* state, const half* weight,
 void ple_spec_conv(torch::Tensor output, torch::Tensor state, torch::Tensor x,
                    torch::Tensor weight, torch::Tensor ids,
                    torch::Tensor starts, torch::Tensor accepted) {
-  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(1) == 10240 &&
-                  (x.size(0) == 5 || x.size(0) == 10),
-              "SM70 PLE MTP4 requires M5/M10 and H10240");
+  TORCH_CHECK(
+      x.is_cuda() && x.dim() == 2 && x.size(1) == 10240 && x.size(0) >= 5,
+      "SM70 PLE MTP4 requires M>=5 and H10240");
   const c10::cuda::CUDAGuard guard(x.device());
   const auto* props = at::cuda::getCurrentDeviceProperties();
   TORCH_CHECK(props->major == 7 && props->minor == 0, "SM70 required");
@@ -72,31 +89,36 @@ void ple_spec_conv(torch::Tensor output, torch::Tensor state, torch::Tensor x,
     TORCH_CHECK(
         t.scalar_type() == at::kInt && t.dim() == 1 && t.is_contiguous(),
         "PLE MTP4 metadata must be contiguous int32 vectors");
-  TORCH_CHECK(ids.numel() == 1 && starts.numel() >= 2 && accepted.numel() >= 1,
-              "PLE MTP4 fast path requires one speculative request");
+  const int64_t num_reqs = ids.numel();
+  TORCH_CHECK(num_reqs >= 1 && starts.numel() >= num_reqs + 1 &&
+                  accepted.numel() >= num_reqs,
+              "PLE MTP4 fast path requires valid speculative request metadata");
   TORCH_CHECK(state.dim() == 3 && state.size(0) > 0 && state.size(1) == 10240 &&
-                  state.size(2) == 13 &&
+                  state.size(2) >= 13 &&
                   (state.scalar_type() == at::kHalf ||
                    state.scalar_type() == at::kFloat),
-              "PLE MTP4 requires FP16/FP32 [states,10240,13] cache");
-  TORCH_CHECK((state.stride(2) == 1 && state.stride(1) >= 13) ||
+              "PLE MTP4 requires FP16/FP32 [states,10240,>=13] cache");
+  TORCH_CHECK((state.stride(2) == 1 && state.stride(1) >= state.size(2)) ||
                   (state.stride(1) == 1 && state.stride(2) >= 10240),
               "Unsupported PLE cache layout");
   TORCH_CHECK(
-      state.stride(0) >= 10239 * state.stride(1) + 12 * state.stride(2) + 1,
+      state.stride(0) >=
+          10239 * state.stride(1) + (state.size(2) - 1) * state.stride(2) + 1,
       "PLE cache rows must not overlap");
   TORCH_CHECK(weight.sizes() == at::IntArrayRef({10240, 4}) &&
                   output.sizes() == x.sizes(),
               "Invalid PLE MTP4 weight/output geometry");
   const auto stream = at::cuda::getCurrentCUDAStream();
+  const dim3 grid(80, static_cast<unsigned int>(num_reqs));
 #define LAUNCH(T)                                                            \
-  ple_conv<T><<<80, 128, 0, stream>>>(                                       \
+  ple_conv<T><<<grid, 128, 0, stream>>>(                                     \
       reinterpret_cast<const half*>(x.data_ptr()),                           \
       reinterpret_cast<T*>(state.data_ptr()),                                \
       reinterpret_cast<const half*>(weight.data_ptr()), ids.data_ptr<int>(), \
       starts.data_ptr<int>(), accepted.data_ptr<int>(),                      \
       reinterpret_cast<half*>(output.data_ptr()), x.size(0), 10240,          \
-      state.stride(0), state.stride(1), state.stride(2))
+      static_cast<int>(num_reqs), state.stride(0), state.stride(1),          \
+      state.stride(2))
   if (state.scalar_type() == at::kHalf) {
     LAUNCH(half);
   } else {
```

---

### 5.2 `vllm/model_executor/layers/fused_moe/sm70/fp4_workspace.py` — SM70 NVFP4 MoE 工作区缓存只读序列索引与跳过冗余清零（P3, P4）

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

### 5.3 `vllm/model_executor/layers/layernorm.py` — RMSNormGated 启用 MTP4 时大行数（rows > 192）@torch.compile 融合快路径（P6）

```diff
diff --git a/vllm/model_executor/layers/layernorm.py b/vllm/model_executor/layers/layernorm.py
index 8affdf0..bd79bf2 100644
--- a/vllm/model_executor/layers/layernorm.py
+++ b/vllm/model_executor/layers/layernorm.py
@@ -412,11 +412,24 @@ class RMSNormGated(CustomOp):
             and self.group_size is None
             and self.norm_before_gate
             and self.activation in ("sigmoid", "silu", "swish")
-            and _sm70_gated_norm_shape_supported(x, z, self.weight)
         ):
-            return torch.ops.vllm.sm70_rmsnorm_gated_exact(
-                x, z, self.weight, self.eps, self.activation != "sigmoid"
-            )
+            if _sm70_gated_norm_shape_supported(x, z, self.weight):
+                return torch.ops.vllm.sm70_rmsnorm_gated_exact(
+                    x, z, self.weight, self.eps, self.activation != "sigmoid"
+                )
+            if (
+                not torch.compiler.is_compiling()
+                and not torch.cuda.is_current_stream_capturing()
+            ):
+                return _compiled_rmsnorm_gated_forward_static(
+                    x,
+                    z,
+                    self.weight,
+                    self.eps,
+                    x.dtype,
+                    self.norm_before_gate,
+                    self.activation,
+                )
         return self.forward_static(
             x,
             z,
@@ -450,6 +463,28 @@ class RMSNormGated(CustomOp):
         return self.forward_cuda(x, z)
 
 
+@torch.compile(dynamic=True)
+def _compiled_rmsnorm_gated_forward_static(
+    x: torch.Tensor,
+    z: torch.Tensor | None,
+    weight: torch.Tensor,
+    epsilon: float,
+    orig_dtype: torch.dtype,
+    norm_before_gate: bool,
+    activation: str,
+) -> torch.Tensor:
+    return RMSNormGated.forward_static(
+        x,
+        z,
+        weight,
+        epsilon,
+        orig_dtype,
+        group_size=None,
+        norm_before_gate=norm_before_gate,
+        activation=activation,
+    )
+
+
 class LayerNorm(nn.Module):
     """
     Layer Normalization.
```

---

### 5.4 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — GDN 投影拆分网格缩减（P5）、尾部拷贝融合（P12 / D19）与 Decode 零拷贝（D16）

```diff
diff --git a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
index a82ba15..8438479 100644
--- a/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
@@ -789,28 +789,43 @@ def _qwen_gdn_metadata_tensors(
     return tensors
 
 
+_EMPTY_INT32_BY_DEVICE: dict[torch.device, torch.Tensor] = {}
+
+
+def _empty_int32_tensor(device: torch.device) -> torch.Tensor:
+    tensor = _EMPTY_INT32_BY_DEVICE.get(device)
+    if tensor is None:
+        tensor = torch.empty(0, dtype=torch.int32, device=device)
+        _EMPTY_INT32_BY_DEVICE[device] = tensor
+    return tensor
+
+
 def _qwen_gdn_non_spec_metadata_tensors(
     layer_name: LayerNameType,
     device: torch.device,
 ) -> tuple[torch.Tensor, torch.Tensor]:
-    empty = torch.empty(0, dtype=torch.int32, device=device)
     try:
         forward_context = get_forward_context()
     except AssertionError:
+        empty = _empty_int32_tensor(device)
         return empty, empty
     attn_metadata_raw = forward_context.attn_metadata
     if not isinstance(attn_metadata_raw, dict):
+        empty = _empty_int32_tensor(device)
         return empty, empty
     attn_metadata = attn_metadata_raw.get(_resolve_layer_name(layer_name))
     if not isinstance(attn_metadata, GDNAttentionMetadata):
+        empty = _empty_int32_tensor(device)
         return empty, empty
 
-    def _or_empty(tensor: torch.Tensor | None) -> torch.Tensor:
-        return tensor if tensor is not None else empty
-
+    qsl = attn_metadata.non_spec_query_start_loc
+    ssi = attn_metadata.non_spec_state_indices_tensor
+    if qsl is not None and ssi is not None:
+        return qsl, ssi
+    empty = _empty_int32_tensor(device)
     return (
-        _or_empty(attn_metadata.non_spec_query_start_loc),
-        _or_empty(attn_metadata.non_spec_state_indices_tensor),
+        qsl if qsl is not None else empty,
+        ssi if ssi is not None else empty,
     )
 
 
@@ -3090,7 +3105,7 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
             if (
                 not self.gqa_interleaved_layout
                 and not self.disable_tp_for_ba_proj
-                and mixed_qkvz.shape[0] in (5, 20)
+                and mixed_qkvz.shape[0] > 1
                 and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
                 and self.value_dim // self.tp_size == 1536
                 and _can_fuse_gdn_projection_split(mixed_qkvz, ba)
@@ -3750,7 +3765,15 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
         )
         if (
             self.enable_packed_recurrent_decode
-            and not mixed_qkv_decode_requested
+            and (
+                not mixed_qkv_decode_requested
+                or (
+                    mixed_qkv.is_contiguous()
+                    and not getattr(
+                        self, "compare_sm70_fused_sigmoid_mixed_qkv", False
+                    )
+                )
+            )
             and attn_metadata.spec_sequence_masks is None
             and attn_metadata.num_prefills == 0
             and attn_metadata.num_decodes > 0
@@ -4003,6 +4026,11 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
                 self.compare_sm70_fused_sigmoid_mixed_qkv
                 and not torch.cuda.is_current_stream_capturing()
             )
+            direct_core_out = (
+                core_attn_out[:num_non_spec_tokens].unsqueeze(1)
+                if not compare_mixed_qkv
+                else None
+            )
             core_attn_out_non_spec, mixed_final_state = mixed_qkv_recurrence(
                 self.gdn_heads,
                 A_log=self.A_log,
@@ -4016,6 +4044,7 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
                     : attn_metadata.num_decodes + 1
                 ],
                 state_indices=non_spec_state_indices_tensor[:num_non_spec_tokens],
+                out=direct_core_out,
                 schedule=self.gdn_policy.schedule,
             )
             if compare_mixed_qkv:
@@ -4081,8 +4110,7 @@ class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
                         "SM70 fused sigmoid mixed-QKV compare matched exactly: "
                         "out_max_diff=0.0 state_max_diff=0.0."
                     )
-                core_attn_out_non_spec = ref_out
-            core_attn_out[:num_non_spec_tokens] = core_attn_out_non_spec.squeeze(0)
+                core_attn_out[:num_non_spec_tokens] = ref_out.squeeze(0)
             return
 
         # The speculative rows are the whole batch, or a subset next to prefill
@@ -4953,7 +4981,10 @@ def qwen_gdn_attention_core_standard_spec(
     if (
         attn_metadata is not None
         and self.enable_packed_recurrent_decode
-        and not mixed_qkv_decode_requested
+        and (
+            not mixed_qkv_decode_requested
+            or not getattr(self, "compare_sm70_fused_sigmoid_mixed_qkv", False)
+        )
         and attn_metadata.spec_sequence_masks is None
         and attn_metadata.num_prefills == 0
         and attn_metadata.num_decodes > 0
@@ -5371,7 +5402,10 @@ def qwen_gdn_attention_core_standard(
     if (
         attn_metadata is not None
         and self.enable_packed_recurrent_decode
-        and not mixed_qkv_decode_requested
+        and (
+            not mixed_qkv_decode_requested
+            or not getattr(self, "compare_sm70_fused_sigmoid_mixed_qkv", False)
+        )
         and attn_metadata.spec_sequence_masks is None
         and attn_metadata.num_prefills == 0
         and attn_metadata.num_decodes > 0
@@ -5795,6 +5829,24 @@ def qwen_gdn_input_projection_core(
             "SM70 GDN QPN8 N4096 plus FP16 b/a N24 split route enabled."
         )
         z = z_out
+    elif (
+        getattr(self, "sm70_qwen38_fp16_fused_input", False)
+        and isinstance(self.in_proj_qkvz, LinearBase)
+        and isinstance(self.in_proj_ba, LinearBase)
+        and use_sm70_decode_graph_semantics(
+            getattr(self, "_execution_graph_policy", None)
+        )
+        and not _sm70_gdn_projection_dump_requested(layer_name)
+    ):
+        assert self.in_proj_ba is not None
+        mixed_qkv, z_flat, b, a = torch.ops.vllm.qwen38_sm70_fp16_gdn_input(
+            hidden_states,
+            self.in_proj_qkvz.weight,
+            self.in_proj_ba.weight,
+            getattr(self.in_proj_qkvz, "_sm70_qwen38_gdn_packed", None),
+            getattr(self.in_proj_ba, "_sm70_qwen38_gdn_packed", None),
+        )
+        z_out.copy_(z_flat.reshape_as(z_out))
     else:
         mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
         ba, _ = self.in_proj_ba(hidden_states)
@@ -5813,7 +5865,7 @@ def qwen_gdn_input_projection_core(
         copied_tails = (
             not self.gqa_interleaved_layout
             and not self.disable_tp_for_ba_proj
-            and mixed_qkvz.shape[0] in (5, 20)
+            and mixed_qkvz.shape[0] > 1
             and (self.key_dim * 2 + self.value_dim) // self.tp_size == 2560
             and self.value_dim // self.tp_size == 1536
             and z_out.shape == (mixed_qkvz.shape[0], 12, 128)
@@ -5951,6 +6003,46 @@ def qwen_gdn_input_projection(
     assert self.in_proj_ba is not None
     ba, _ = self.in_proj_ba(hidden_states)
 
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
+
     if self.gqa_interleaved_layout:
         query, key, value, z, b, a = self.fix_query_key_value_ordering(
             mixed_qkvz,
```

---

### 5.5 `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` — GDN causal_conv1d 保持切片视图 channel-last 连续步幅，消除 16KB 跨步非合并访存（P9）

```diff
diff --git a/vllm/model_executor/layers/mamba/ops/causal_conv1d.py b/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
index 4ea8e28..b82f4fd 100644
--- a/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
+++ b/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
@@ -540,7 +540,13 @@ def causal_conv1d_fn(
     # Store original dtype to cast back at the end
     original_x_dtype = x.dtype
     x = x.to(conv_states.dtype)
-    out = torch.empty_like(x)
+    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
+    dim, cu_seqlen = x.shape
+    out = (
+        x.new_empty((cu_seqlen, dim)).transpose(0, 1)
+        if is_channel_last
+        else torch.empty_like(x)
+    )
     if metadata is not None:
         nums_dict = metadata.nums_dict
         args = nums_dict
@@ -558,8 +564,6 @@ def causal_conv1d_fn(
             (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device
         )  # tracking BLOCK_M-based index in the sequence the Triton program is handling
 
-    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
-    dim, cu_seqlen = x.shape
     _, width = weight.shape
     state_len = width - 1
     np2_statelen = triton.next_power_of_2(state_len)
```

---

### 5.6 `vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py` — NVFP4 MoE expert_map=None 快路径跳过冗余清零与拷贝（P3, P4）

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

---

### 5.7 `vllm/model_executor/models/qwen3_next.py` — DP2 × TP4 下保持 FusedMoE 副本内局部 TP=4（不跨 DP 扁平化为 TP=8）（D26）

```diff
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
index 093f5ce..367a3ed 100644
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

---

### 5.8 `vllm/models/qwen4_exp/amd/mtp.py` — AMD MTP _make_draft_vllm_config 正确继承 speculative_config.kv_cache_dtype（D27）

```diff
diff --git a/vllm/models/qwen4_exp/amd/mtp.py b/vllm/models/qwen4_exp/amd/mtp.py
index 2c93808..a6aaa33 100644
--- a/vllm/models/qwen4_exp/amd/mtp.py
+++ b/vllm/models/qwen4_exp/amd/mtp.py
@@ -14,6 +14,9 @@ The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
 """
 
 from collections.abc import Iterable
+import copy
+from dataclasses import is_dataclass
+from typing import Any
 
 import regex as re
 import torch
@@ -171,9 +174,25 @@ def _make_draft_vllm_config(
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

### 5.9 `vllm/models/qwen4_exp/common/qsa_cache.py` — QSA 跨层 visible_blocks_cache（P1）与 Decode 动态有效 score_columns 边界（D9）

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

### 5.10 `vllm/models/qwen4_exp/nvidia/hyperconnection.py` — HyperConnection _hcx_final_mix 原地写入 _mtp_hidden_buffer（P10）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/hyperconnection.py b/vllm/models/qwen4_exp/nvidia/hyperconnection.py
index 6d059d3..3532748 100644
--- a/vllm/models/qwen4_exp/nvidia/hyperconnection.py
+++ b/vllm/models/qwen4_exp/nvidia/hyperconnection.py
@@ -240,6 +240,17 @@ class GatedResidual(nn.Module):
         prev_block_output: torch.Tensor,
         prev_injection: torch.Tensor,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
+        mtp_buf = getattr(self, "_mtp_hidden_buffer", None)
+        out_buf = None
+        if (
+            mtp_buf is not None
+            and not torch.compiler.is_compiling()
+            and mtp_buf.device == hidden_states.device
+            and mtp_buf.dtype == hidden_states.dtype
+            and mtp_buf.shape[0] >= hidden_states.shape[0]
+            and mtp_buf.shape[1:] == hidden_states.shape[1:]
+        ):
+            out_buf = mtp_buf[: hidden_states.shape[0]]
         hidden_states, xn = hc_combine_norm(
             hidden_states,
             prev_block_output,
@@ -247,6 +258,7 @@ class GatedResidual(nn.Module):
             self.hc_norm.weight,
             self.config.rms_norm_eps,
             self.hc_count,
+            out=out_buf,
         )
 
         block_input, injection = self._project(xn)
```

---

### 5.11 `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` — QSA Indexer 持久工作区（D7）、skip_topk 跳过 q_norm/q_rope（D18）与 Key-only 切片投影（D22）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/indexer_qsa.py b/vllm/models/qwen4_exp/nvidia/indexer_qsa.py
index 0369fac..8d31702 100644
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
@@ -220,7 +242,11 @@ class QSAIndexer(nn.Module):
                 return out
             return result
 
+        from vllm.config.compilation import CUDAGraphMode
+        from vllm.forward_context import is_forward_context_available
+
         from .ops.qsa import (
+            _use_sm70_qsa_lexicographic_topk,
             qsa_compress_groups_with_ratio,
             qsa_select_paged_tokens,
             qsa_store_cache_rows,
@@ -231,30 +257,84 @@ class QSAIndexer(nn.Module):
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
 
         if self.use_fused_pre_indexer:
-            q = projected_q.new_empty(
-                num_tokens,
-                self.index_n_heads,
-                self.index_head_dim,
+            q = self._get_q_buffer(num_tokens, raw_keys.dtype, raw_keys.device)
+            q_input = (
+                q.view(num_tokens, q_dim) if projected_q is None else projected_q
             )
             qsa_pre_indexer(
-                projected_q,
+                q_input,
                 raw_keys,
                 positions,
                 self.rotary_emb.cos_sin_cache,
@@ -277,18 +357,20 @@ class QSAIndexer(nn.Module):
                     if raw_key_state_cache.rope_position_cache is not None
                     else None
                 ),
+                skip_q=skip_query_prep,
             )
         else:
             # Unfused reference path
             from flashinfer.norm import gemma_rmsnorm
 
-            q = projected_q.reshape(-1, self.index_n_heads, self.index_head_dim)
-            q = gemma_rmsnorm(
-                q.reshape(-1, self.index_head_dim),
-                self.q_layernorm.weight,
-                self.q_layernorm.variance_epsilon,
-            ).reshape_as(q)
-            q = apply_qsa_rope(self.rotary_emb, positions, q)
+            if not skip_query_prep:
+                q = projected_q.reshape(-1, self.index_n_heads, self.index_head_dim)
+                q = gemma_rmsnorm(
+                    q.reshape(-1, self.index_head_dim),
+                    self.q_layernorm.weight,
+                    self.q_layernorm.variance_epsilon,
+                ).reshape_as(q)
+                q = apply_qsa_rope(self.rotary_emb, positions, q)
 
             raw_key_cache = raw_key_state_cache.key_cache
             rope_position_cache = raw_key_state_cache.rope_position_cache
@@ -361,6 +443,8 @@ class QSAIndexer(nn.Module):
             out,
             query_start_loc_cpu=compressed_metadata.query_start_loc_cpu,
             shared_key_scoring=self.shared_key_scoring,
+            visible_blocks_cache=compressed_metadata.visible_blocks_cache,
+            max_seq_len=compressed_metadata.max_seq_len,
         )
```

---

### 5.12 `vllm/models/qwen4_exp/nvidia/model.py` — HC 原地写回（P10）、GDN 免拷贝（D16）、MTP 启用 SM70 Peer 路径（P12 / D19）与修复 hcx final 命名冲突（D21）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/model.py b/vllm/models/qwen4_exp/nvidia/model.py
index 2a0ce19..de372a3 100644
--- a/vllm/models/qwen4_exp/nvidia/model.py
+++ b/vllm/models/qwen4_exp/nvidia/model.py
@@ -470,7 +470,11 @@ class Qwen4ExpDecoderLayer(nn.Module):
             hidden_states, block_input, injection = attn_hc.mix(hidden_states)
 
         if self.layer_type == "linear_attention":
+            gdn_policy = getattr(self.linear_attn, "gdn_policy", None)
             use_direct_attention_output = (
+                gdn_policy is not None
+                and not gdn_policy.projection.output_projection
+            ) or (
                 communication_policy().long_prefill_norm
                 and torch.compiler.is_compiling()
             )
@@ -617,8 +621,11 @@ def enable_sm70_hcx(
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
@@ -628,7 +635,12 @@ def _maybe_fuse_sm70_side_projections(model: nn.Module) -> None:
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
         if layer.layer_type == "linear_attention":
@@ -666,7 +678,12 @@ def _maybe_enable_sm70_peer_paths(vllm_config: VllmConfig, model: nn.Module) ->
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
 
 
 class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
@@ -841,6 +858,8 @@ class Qwen4ExpModel(nn.Module):
                 config.hc_count * config.hidden_size,
                 dtype=vllm_config.model_config.dtype,
             )
+            if self.hyper_connection_mixer is not None:
+                self.hyper_connection_mixer._mtp_hidden_buffer = self._mtp_hidden_buffer
         else:
             self._mtp_hidden_buffer = None
 
@@ -926,7 +945,11 @@ class Qwen4ExpModel(nn.Module):
         multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
             hidden_states, block_output, injection
         )
-        if self._mtp_hidden_buffer is not None:
+        if self._mtp_hidden_buffer is not None and not (
+            getattr(final_mixer, "_partial_inputs", False)
+            and getattr(final_mixer, "_mtp_hidden_buffer", None)
+            is self._mtp_hidden_buffer
+        ):
             # Capture the pre-final-mixer multi-stream hidden state
             # [T, hc_count*H] for the MTP drafter (zero extra compute:
             # this tensor is needed by the final mixer regardless).
```

---

### 5.13 `vllm/models/qwen4_exp/nvidia/mtp.py` — MTP 预分配 topk/DCP2 缓冲（P8）、启用 Peer 路径（P12 / D19）、compact_topk_indices 单行 Triton 核（D25）与修复草稿 kv_cache_dtype 继承（D27）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/mtp.py b/vllm/models/qwen4_exp/nvidia/mtp.py
index 05b7ce3..0eda2a9 100644
--- a/vllm/models/qwen4_exp/nvidia/mtp.py
+++ b/vllm/models/qwen4_exp/nvidia/mtp.py
@@ -14,6 +14,9 @@ The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
 """
 
 from collections.abc import Iterable
+import copy
+from dataclasses import is_dataclass
+from typing import Any
 
 import regex as re
 import torch
@@ -73,10 +76,28 @@ from .model import (
 )
 from .sm70_fp16_gemv import enable_qwen38_sm70_fp16_gemv
 from .sm70_fp16_hc import enable_qwen38_sm70_fp16_fused_hc
+from vllm.triton_utils import HAS_TRITON, tl, triton
 
 logger = init_logger(__name__)
 
 
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
+
+
 def _remap_ignored_layers(
     ignored_layers: list[str],
     mtp_start_layer_idx: int,
@@ -223,10 +244,26 @@ def _make_draft_vllm_config(
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
@@ -310,6 +347,50 @@ class Qwen4ExpMultiTokenPredictor(nn.Module):
         # Phase 2: the drafter shares the target cache_config; remember the main
         # KV cache dtype so load_weights can finalize the draft QSA E4M3 scales.
         self._kv_cache_dtype = draft_vllm_config.cache_config.cache_dtype
+        topk_indices_buffer: torch.Tensor | None = None
+        dcp_local_indices_buffer: torch.Tensor | None = None
+        dcp_partial_output_buffer: torch.Tensor | None = None
+        dcp_partial_lse_buffer: torch.Tensor | None = None
+        if getattr(config, "indexer_n_heads", None) is not None:
+            max_tokens = draft_vllm_config.scheduler_config.max_num_batched_tokens
+            width = (
+                int(config.indexer_budget) + int(config.indexer_compress_ratio) - 1
+            )
+            topk_indices_buffer = torch.empty(
+                max_tokens,
+                width,
+                dtype=torch.int32,
+            )
+            self.topk_indices_buffer = topk_indices_buffer
+            if draft_vllm_config.parallel_config.decode_context_parallel_size == 2:
+                tp_size = draft_vllm_config.parallel_config.tensor_parallel_size
+                local_q_heads = int(config.num_attention_heads) // max(tp_size, 1)
+                head_dim = int(
+                    getattr(
+                        config,
+                        "head_dim",
+                        config.hidden_size // config.num_attention_heads,
+                    )
+                )
+                dcp_local_indices_buffer = torch.empty(
+                    max_tokens,
+                    width,
+                    dtype=torch.int32,
+                )
+                dcp_partial_output_buffer = torch.empty(
+                    max_tokens,
+                    local_q_heads,
+                    head_dim,
+                    dtype=model_config.dtype,
+                )
+                dcp_partial_lse_buffer = torch.empty(
+                    max_tokens,
+                    local_q_heads,
+                    dtype=torch.float32,
+                )
+                self.dcp_local_indices_buffer = dcp_local_indices_buffer
+                self.dcp_partial_output_buffer = dcp_partial_output_buffer
+                self.dcp_partial_lse_buffer = dcp_partial_lse_buffer
         with set_current_vllm_config(draft_vllm_config, prefix=prefix):
             # residual_linear_shared fusion: fc_embedding projects the token
             # embedding, fc_hidden (shared across HC branches) projects the
@@ -338,6 +419,10 @@ class Qwen4ExpMultiTokenPredictor(nn.Module):
                     draft_vllm_config,
                     layer_type="full_attention",
                     prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
+                    topk_indices_buffer=topk_indices_buffer,
+                    dcp_local_indices_buffer=dcp_local_indices_buffer,
+                    dcp_partial_output_buffer=dcp_partial_output_buffer,
+                    dcp_partial_lse_buffer=dcp_partial_lse_buffer,
                 )
                 for idx in range(self.num_mtp_layers)
             )
@@ -387,10 +472,51 @@ class Qwen4ExpMultiTokenPredictor(nn.Module):
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
 
     def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
         return self.embed_tokens(input_ids)
@@ -713,7 +839,10 @@ class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):
         # Finalize its calibrated k/v scales strictly from the checkpoint (no
         # runtime fallback; missing names are listed). No-op for FP16/BF16 caches
         # and on the PLE offload process (is_offload_process()).
-        from .model import _finalize_qsa_e4m3_scale_load
+        from .model import (
+            _finalize_qsa_e4m3_scale_load,
+            _maybe_enable_sm70_peer_paths,
+        )
 
         _finalize_qsa_e4m3_scale_load(
             self,
@@ -721,6 +850,7 @@ class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):
             self.model._kv_cache_dtype,
             require_calibrated_speculative_draft=True,
         )
+        _maybe_enable_sm70_peer_paths(self.vllm_config, self.model)
         return loaded_weights
```

---

### 5.14 `vllm/models/qwen4_exp/nvidia/ops/hc.py` — HyperConnection _hc_combine_norm_impl 支持 out 缓冲原地写回（P10）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/hc.py b/vllm/models/qwen4_exp/nvidia/ops/hc.py
index c0a8a1d..8668eb1 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/hc.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/hc.py
@@ -335,13 +335,14 @@ def _hc_combine_norm_kernel(
     tl.store(y_ptr + row * stride_y + offs, y, mask_inner)
 
 
-def _hc_combine_norm(
+def _hc_combine_norm_impl(
     residual: torch.Tensor,
     block_output: torch.Tensor,
     injection_logits: torch.Tensor,
     norm_weight: torch.Tensor,
     eps: float,
     hc_count: int,
+    out: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     N, DIM = residual.shape
     assert DIM % hc_count == 0
@@ -354,7 +355,11 @@ def _hc_combine_norm(
     assert norm_weight.is_contiguous()
     assert norm_weight.numel() in (hc_dim, DIM)
 
-    out = residual.new_empty(residual.shape)
+    if out is None:
+        out = residual.new_empty(residual.shape)
+    else:
+        assert out.shape == residual.shape
+        assert out.stride(1) == 1
     y = residual.new_empty(residual.shape)
     # The M=1 SM70 path is register-bound with the generic 512-wide tile.
     # A 1024-wide tile keeps identical reduction/rounding results and is
@@ -391,6 +396,24 @@ def _hc_combine_norm(
     return out, y
 
 
+def _hc_combine_norm(
+    residual: torch.Tensor,
+    block_output: torch.Tensor,
+    injection_logits: torch.Tensor,
+    norm_weight: torch.Tensor,
+    eps: float,
+    hc_count: int,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    return _hc_combine_norm_impl(
+        residual,
+        block_output,
+        injection_logits,
+        norm_weight,
+        eps,
+        hc_count,
+    )
+
+
 def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
     return x.new_empty(x.shape)
 
@@ -483,7 +506,18 @@ def hc_combine_norm(
     norm_weight: torch.Tensor,
     eps: float,
     hc_count: int,
+    out: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
+    if out is not None and not torch.compiler.is_compiling():
+        return _hc_combine_norm_impl(
+            residual,
+            block_output,
+            injection_logits,
+            norm_weight,
+            eps,
+            hc_count,
+            out=out,
+        )
     return torch.ops.vllm.qwen4_exp_hc_combine_norm(
         residual,
         block_output,
```

---

### 5.15 `vllm/models/qwen4_exp/nvidia/ops/qsa.py` — QSA Split-K/Selector 持久缓冲（D3, D7）、空块短路与 2-warp 启动（D9, D11）、Split-K 融合 DCP 打包（D13）、DCP2 valid_counts 动态裁剪（P11 / D17）与短上下文 IDENTITY_BLOCKS 快路径（D22）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa.py b/vllm/models/qwen4_exp/nvidia/ops/qsa.py
index aa5e6a6..7d06033 100644
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
 
 
@@ -383,6 +391,7 @@ def _expand_qsa_indices_kernel(
     TOKEN_TOPK: tl.constexpr,
     OUTPUT_WIDTH: tl.constexpr,
     COLUMN_BLOCK: tl.constexpr,
+    IDENTITY_BLOCKS: tl.constexpr = False,
 ) -> None:
     row = tl.program_id(0)
     columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
@@ -409,11 +418,16 @@ def _expand_qsa_indices_kernel(
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
     tail_offset = columns - expanded_count
     is_tail = (
@@ -636,11 +650,14 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     HOST_INDICES=None,
     HOST_VALID_COUNTS=None,
     HOST_CACHE: tl.constexpr = False,
+    USE_VALID_COUNTS: tl.constexpr = False,
     HISTORY_SCALES=None,
     HISTORY_POSITIONS=None,
     HISTORY_LENGTHS=None,
     DEVICE_HISTORY: tl.constexpr = False,
     HISTORY_E4M3: tl.constexpr = False,
+    dcp_packed_ptr=None,
+    H_PER_RANK: tl.constexpr = 0,
 ) -> None:
     row = tl.program_id(0)
     kv_head = tl.program_id(1)
@@ -676,7 +693,7 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
     loop_start = split_tile_start
     loop_end = split_tile_end
-    if HOST_CACHE:
+    if HOST_CACHE or USE_VALID_COUNTS:
         selected_count = tl.load(HOST_VALID_COUNTS + row)
         # The selector compacts the open group's causal tail immediately after
         # the selected complete blocks, including at attention tile boundaries.
@@ -703,89 +720,92 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
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
@@ -795,44 +815,63 @@ def _qsa_sparse_paged_gqa_splitk_kernel(
     )
     output_mask = head_offsets[:, None] < GROUP_SIZE
     if NUM_SPLITS == 1:
-        if final_lse_ptr is not None:
+        if final_lse_ptr is not None or dcp_packed_ptr is not None:
             lse = tl.where(
                 has_values,
                 max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
                 -float("inf"),
             )
-            tl.store(
-                final_lse_ptr + row * NUM_QUERY_HEADS + first_head + head_offsets,
-                lse,
-                mask=head_offsets < GROUP_SIZE,
-            )
+            if dcp_packed_ptr is None:
+                tl.store(
+                    final_lse_ptr + row * NUM_QUERY_HEADS + first_head + head_offsets,
+                    lse,
+                    mask=head_offsets < GROUP_SIZE,
+                )
         if KV_E4M3:
             # V dequantization is linear, so apply its scalar after the
             # normalized FP32 accumulation instead of to every loaded value.
             normalized_output *= v_scale
-        if output_gate_ptr is not None:
-            # Preserve the compiled path's rounded attention output before
-            # evaluating the sigmoid gate and final product in FP32.
-            normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
-            output_gate = tl.load(
-                output_gate_ptr
-                + row * stride_output_gate_row
-                + (first_head + head_offsets[:, None]) * stride_output_gate_head
+        if dcp_packed_ptr is not None:
+            heads = first_head + head_offsets
+            rank_idx = heads // H_PER_RANK
+            local_head = heads % H_PER_RANK
+            packed_base = (
+                (rank_idx * num_rows + row) * H_PER_RANK + local_head
+            ) * (HEAD_DIM + 1)
+            tl.store(
+                dcp_packed_ptr + packed_base[:, None] + dim_offsets[None, :],
+                normalized_output,
+                mask=output_mask,
+            )
+            tl.store(
+                dcp_packed_ptr + packed_base + HEAD_DIM,
+                lse,
+                mask=head_offsets < GROUP_SIZE,
+            )
+        else:
+            if output_gate_ptr is not None:
+                # Preserve the compiled path's rounded attention output before
+                # evaluating the sigmoid gate and final product in FP32.
+                normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
+                output_gate = tl.load(
+                    output_gate_ptr
+                    + row * stride_output_gate_row
+                    + (first_head + head_offsets[:, None]) * stride_output_gate_head
+                    + dim_offsets[None, :],
+                    mask=output_mask,
+                    other=0.0,
+                ).to(tl.float32)
+                normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(
+                    output_gate
+                )
+            tl.store(
+                output_ptr
+                + row * stride_output_row
+                + (first_head + head_offsets[:, None]) * stride_output_head
                 + dim_offsets[None, :],
+                normalized_output,
                 mask=output_mask,
-                other=0.0,
-            ).to(tl.float32)
-            normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(
-                output_gate
             )
-        tl.store(
-            output_ptr
-            + row * stride_output_row
-            + (first_head + head_offsets[:, None]) * stride_output_head
-            + dim_offsets[None, :],
-            normalized_output,
-            mask=output_mask,
-        )
     else:
         partial_lse = tl.where(
             has_values,
@@ -879,6 +918,8 @@ def _qsa_merge_splitk_kernel(
     NUM_SPLITS: tl.constexpr,
     BLOCK_SPLITS: tl.constexpr,
     KV_E4M3: tl.constexpr,
+    dcp_packed_ptr=None,
+    H_PER_RANK: tl.constexpr = 0,
 ) -> None:
     row = tl.program_id(0)
     head = tl.program_id(1)
@@ -895,13 +936,14 @@ def _qsa_merge_splitk_kernel(
     shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
     weights = tl.math.exp2(shifted)
     denominator = tl.sum(weights, axis=0)
-    if final_lse_ptr is not None:
+    if final_lse_ptr is not None or dcp_packed_ptr is not None:
         merged_lse = tl.where(
             has_values,
             lse_max + tl.math.log2(tl.maximum(denominator, 1.0e-20)),
             -float("inf"),
         )
-        tl.store(final_lse_ptr + row * NUM_QUERY_HEADS + head, merged_lse)
+        if dcp_packed_ptr is None:
+            tl.store(final_lse_ptr + row * NUM_QUERY_HEADS + head, merged_lse)
     partial_output = tl.load(
         partial_output_ptr
         + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
@@ -916,19 +958,31 @@ def _qsa_merge_splitk_kernel(
         # Apply the V scale once after combining all independently normalized
         # splits. Scaling partials earlier would repeat this work per split.
         merged *= v_scale
-    if output_gate_ptr is not None:
-        merged = merged.to(output_ptr.dtype.element_ty)
-        output_gate = tl.load(
-            output_gate_ptr
-            + row * stride_output_gate_row
-            + head * stride_output_gate_head
-            + dim_offsets
-        ).to(tl.float32)
-        merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
-    tl.store(
-        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
-        merged,
-    )
+    if dcp_packed_ptr is not None:
+        rank_idx = head // H_PER_RANK
+        local_head = head % H_PER_RANK
+        packed_base = (
+            (rank_idx * num_rows + row) * H_PER_RANK + local_head
+        ) * (HEAD_DIM + 1)
+        tl.store(dcp_packed_ptr + packed_base + dim_offsets, merged)
+        tl.store(dcp_packed_ptr + packed_base + HEAD_DIM, merged_lse)
+    else:
+        if output_gate_ptr is not None:
+            merged = merged.to(output_ptr.dtype.element_ty)
+            output_gate = tl.load(
+                output_gate_ptr
+                + row * stride_output_gate_row
+                + head * stride_output_gate_head
+                + dim_offsets
+            ).to(tl.float32)
+            merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
+        tl.store(
+            output_ptr
+            + row * stride_output_row
+            + head * stride_output_head
+            + dim_offsets,
+            merged,
+        )
 
 
 @triton.jit
@@ -1174,6 +1228,8 @@ def qsa_mqa_paged(
     num_columns: int | None = None,
     score_scale: float | None = None,
     shared_key_scoring: bool = False,
+    out_logits: torch.Tensor | None = None,
+    out_visible_blocks: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     """Compute QSA scores directly from a paged compressed-key cache."""
 
@@ -1217,11 +1273,31 @@ def qsa_mqa_paged(
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
@@ -1523,6 +1599,8 @@ def _qsa_select_by_request(
     compress_ratio: int,
     out: torch.Tensor,
     query_start_loc_cpu: torch.Tensor,
+    visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]]
+    | None = None,
 ) -> bool:
     """Select a multi-request batch one long request at a time.
 
@@ -1565,6 +1643,8 @@ def _qsa_select_by_request(
                 token_topk,
                 compress_ratio,
                 out[start:end],
+                visible_blocks_cache=visible_blocks_cache,
+                _cache_row_range=(start, end),
             )
         else:
             qsa_select_paged_tokens(
@@ -1577,10 +1657,93 @@ def _qsa_select_by_request(
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
@@ -1593,6 +1756,10 @@ def qsa_select_paged_tokens(
     out: torch.Tensor | None = None,
     query_start_loc_cpu: torch.Tensor | None = None,
     shared_key_scoring: bool = False,
+    visible_blocks_cache: dict[tuple[int, int, int], tuple[torch.Tensor, int]]
+    | None = None,
+    _cache_row_range: tuple[int, int] | None = None,
+    max_seq_len: int | None = None,
 ) -> torch.Tensor:
     """Score, select, and expand QSA indices without host synchronization."""
 
@@ -1615,29 +1782,87 @@ def qsa_select_paged_tokens(
         compress_ratio,
         out,
         query_start_loc_cpu,
+        visible_blocks_cache=visible_blocks_cache,
     ):
         return out
 
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
@@ -1653,13 +1878,26 @@ def qsa_select_paged_tokens(
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
@@ -1669,6 +1907,7 @@ def qsa_select_paged_tokens(
     for row_start in range(0, rows, rows_per_chunk):
         row_end = min(row_start + rows_per_chunk, rows)
         row_slice = slice(row_start, row_end)
+        chunk_n = row_end - row_start
         if contiguous_keys is not None and key_valid is not None:
             assert all_visible is not None
             logits, visible_blocks = _qsa_mqa_cublas(
@@ -1678,6 +1917,16 @@ def qsa_select_paged_tokens(
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
@@ -1686,9 +1935,9 @@ def qsa_select_paged_tokens(
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
@@ -1976,6 +2225,40 @@ def _qsa_grouped_page4_workspace(
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
@@ -2349,6 +2632,9 @@ def qsa_sparse_paged_attention(
     k_scale: float = 1.0,
     v_scale: float = 1.0,
     lse: torch.Tensor | None = None,
+    pre_resolved_indices: bool = False,
+    dcp_packed_out: torch.Tensor | None = None,
+    valid_counts: torch.Tensor | None = None,
 ) -> torch.Tensor:
     """Run sparse GQA, optionally returning base-2 LSE for a cross-rank merge.
 
@@ -2414,6 +2700,25 @@ def qsa_sparse_paged_attention(
             raise ValueError(
                 "QSA DCP must apply its gate after merging partial outputs"
             )
+    if dcp_packed_out is not None:
+        if lse is None:
+            raise ValueError("QSA DCP packed output requires LSE")
+        if (
+            dcp_packed_out.ndim != 4
+            or dcp_packed_out.dtype != torch.float32
+            or dcp_packed_out.device != q.device
+            or not dcp_packed_out.is_contiguous()
+            or dcp_packed_out.shape[1] != q.shape[0]
+            or dcp_packed_out.shape[0] * dcp_packed_out.shape[2] != q.shape[1]
+            or dcp_packed_out.shape[3] != head_dim + 1
+        ):
+            raise ValueError(
+                "QSA DCP packed output must be contiguous FP32 "
+                "[world_size, rows, heads_per_rank, head_dim + 1]"
+            )
+    dcp_h_per_rank = (
+        dcp_packed_out.shape[2] if dcp_packed_out is not None else q.shape[1]
+    )
     output_gate_view = output_gate.view_as(q) if output_gate is not None else None
     if output_gate_view is not None:
         if output_gate_view.dtype != q.dtype or output_gate_view.device != q.device:
@@ -2453,10 +2758,10 @@ def qsa_sparse_paged_attention(
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
@@ -2500,16 +2805,18 @@ def qsa_sparse_paged_attention(
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
-
+        # splits. Reuse per-stream scratch buffers to avoid 24 dynamic tensor
+        # allocations per step across the 12 target QSA layers on TP4+DCP2.
+        partial_output, partial_lse = _qsa_splitk_workspace(q, num_splits)
+
+    use_valid_counts = bool(
+        valid_counts is not None
+        and valid_counts.ndim == 1
+        and valid_counts.shape[0] >= q.shape[0]
+        and valid_counts.dtype == torch.int32
+        and valid_counts.device == q.device
+        and valid_counts.is_contiguous()
+    )
     partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
     _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
         q,
@@ -2554,6 +2861,10 @@ def qsa_sparse_paged_attention(
         BLOCK_N=block_n,
         KV_E4M3=kv_e4m3,
         RESOLVED_INDICES=resolved_indices,
+        HOST_VALID_COUNTS=valid_counts if use_valid_counts else None,
+        USE_VALID_COUNTS=use_valid_counts,
+        dcp_packed_ptr=dcp_packed_out,
+        H_PER_RANK=dcp_h_per_rank,
         num_warps=partial_warps,
         num_stages=2,
     )
@@ -2577,6 +2888,8 @@ def qsa_sparse_paged_attention(
         NUM_SPLITS=num_splits,
         BLOCK_SPLITS=triton.next_power_of_2(num_splits),
         KV_E4M3=kv_e4m3,
+        dcp_packed_ptr=dcp_packed_out,
+        H_PER_RANK=dcp_h_per_rank,
         num_warps=2,
         num_stages=1,
     )
@@ -2624,7 +2937,7 @@ def _use_sm70_qsa_two_warp_partial(
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

### 5.16 `vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py` — DCP2 _qsa_localize_dcp_indices_kernel 半宽存储、物理 KV 槽位融合解析（D2, D10）与 valid_counts 统计（P11 / D17）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
index 050f072..e76f74c 100644
--- a/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
+++ b/vllm/models/qwen4_exp/nvidia/ops/qsa_dcp.py
@@ -13,28 +13,68 @@ import torch
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
+    counts_ptr=None,
+    WRITE_COUNTS: tl.constexpr = False,
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
+    if WRITE_COUNTS:
+        tl.store(counts_ptr + row, tl.minimum(count, OUT_WIDTH))
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
@@ -69,6 +109,10 @@ def qsa_localize_dcp_indices(
     dcp_rank: int,
     interleave_size: int,
     local_block_size: int,
+    block_table: torch.Tensor | None = None,
+    token_to_req: torch.Tensor | None = None,
+    num_cache_blocks: int | None = None,
+    valid_counts: torch.Tensor | None = None,
 ) -> torch.Tensor:
     """Compact this rank's selected IDs into an independent persistent buffer.
 
@@ -77,7 +121,14 @@ def qsa_localize_dcp_indices(
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
@@ -96,23 +147,51 @@ def qsa_localize_dcp_indices(
     ):
         raise ValueError("QSA DCP interleave must divide the positive local block size")
     if indices.numel() == 0:
+        if valid_counts is not None:
+            valid_counts.zero_()
         return out
     if indices.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
         raise ValueError("QSA DCP output must not alias the MTP selection buffer")
     if dcp_world_size == 1:
-        out.copy_(indices)
+        out.copy_(indices[:, : out.shape[1]])
+        if valid_counts is not None:
+            valid_counts.copy_((out >= 0).sum(dim=1, dtype=torch.int32))
         return out
     if not indices.is_cuda:
         raise ValueError("QSA DCP localization requires CUDA buffers")
+    resolve_physical = (
+        block_table is not None
+        and token_to_req is not None
+        and num_cache_blocks is not None
+    )
+    write_counts = bool(
+        valid_counts is not None
+        and valid_counts.ndim == 1
+        and valid_counts.shape[0] >= indices.shape[0]
+        and valid_counts.dtype == torch.int32
+        and valid_counts.device == indices.device
+        and valid_counts.is_contiguous()
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
+        counts_ptr=valid_counts,
+        WRITE_COUNTS=write_counts,
     )
     return out
```

---

### 5.17 `vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py` — QSA Pre-Indexer 当 num_q_work == 0 时跳过 q_norm + q_rope（D18）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py b/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py
index 3c16955..2aeb8d0 100644
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
         q,
         q.stride(0),
```

---

### 5.18 `vllm/models/qwen4_exp/nvidia/ops/qsa_shared_key.py` — QSA Shared Key Norm 持久输出缓冲复用（D7）

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

### 5.19 `vllm/models/qwen4_exp/nvidia/ple_layer.py` — PLE 2D 网格短卷积（D15）、行主序 4-tap Triton Prefill 短卷积 + 状态更新、全尺寸 m > 32 N-gram 哈希与多请求投机短卷积（P13 / D20）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/ple_layer.py b/vllm/models/qwen4_exp/nvidia/ple_layer.py
index f1afbd5..092ad09 100644
--- a/vllm/models/qwen4_exp/nvidia/ple_layer.py
+++ b/vllm/models/qwen4_exp/nvidia/ple_layer.py
@@ -317,61 +317,99 @@ def _qwen38_ple_m1_short_conv_kernel(
     HIDDEN_SIZE: tl.constexpr,
     BLOCK: tl.constexpr,
 ):
-    """Fuse exact Qwen3.8 M=1 dilated conv and state-cache update."""
+    """Fuse exact Qwen3.8 decode dilated conv and state-cache update."""
 
     hidden = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
+    row = tl.program_id(1).to(tl.int64)
     hidden_mask = hidden < HIDDEN_SIZE
-    state_index = tl.load(state_index_ptr).to(tl.int64)
+    state_index = tl.load(state_index_ptr + row).to(tl.int64)
     valid = state_index != NULL_STATE_ID
     safe_state_index = tl.where(valid, state_index, 0)
     use_initial = valid
     if HAS_INITIAL:
-        use_initial &= tl.load(has_initial_ptr).to(tl.int1)
+        use_initial &= tl.load(has_initial_ptr + row).to(tl.int1)
     state_base = safe_state_index * STATE_STRIDE_0 + hidden * STATE_STRIDE_1
     state_mask = hidden_mask & use_initial
 
-    s0 = tl.load(state_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)
-    s1 = tl.load(
-        state_ptr + state_base + STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s2 = tl.load(
-        state_ptr + state_base + 2 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s3 = tl.load(
-        state_ptr + state_base + 3 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s4 = tl.load(
-        state_ptr + state_base + 4 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s5 = tl.load(
-        state_ptr + state_base + 5 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s6 = tl.load(
-        state_ptr + state_base + 6 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s7 = tl.load(
-        state_ptr + state_base + 7 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    s8 = tl.load(
-        state_ptr + state_base + 8 * STATE_STRIDE_2,
-        mask=state_mask,
-        other=0.0,
-    ).to(tl.float32)
-    x = tl.load(x_ptr + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
+    s0 = (
+        tl.load(state_ptr + state_base, mask=state_mask, other=0.0)
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s1 = (
+        tl.load(
+            state_ptr + state_base + STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s2 = (
+        tl.load(
+            state_ptr + state_base + 2 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s3 = (
+        tl.load(
+            state_ptr + state_base + 3 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s4 = (
+        tl.load(
+            state_ptr + state_base + 4 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s5 = (
+        tl.load(
+            state_ptr + state_base + 5 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s6 = (
+        tl.load(
+            state_ptr + state_base + 6 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s7 = (
+        tl.load(
+            state_ptr + state_base + 7 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    s8 = (
+        tl.load(
+            state_ptr + state_base + 8 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    row_offset = row * HIDDEN_SIZE + hidden
+    x = tl.load(x_ptr + row_offset, mask=hidden_mask, other=0.0).to(tl.float32)
     w0 = tl.load(weight_ptr + hidden * 4).to(tl.float32)
     w1 = tl.load(weight_ptr + hidden * 4 + 1).to(tl.float32)
     w2 = tl.load(weight_ptr + hidden * 4 + 2).to(tl.float32)
@@ -384,7 +422,11 @@ def _qwen38_ple_m1_short_conv_kernel(
     # retains native F.silu because its SM70 rounding differs slightly from
     # Triton's sigmoid approximation.
     conv = conv.to(tl.float16)
-    tl.store(output_ptr + hidden, tl.where(valid, conv, 0.0), mask=hidden_mask)
+    tl.store(
+        output_ptr + row_offset,
+        tl.where(valid, conv, 0.0),
+        mask=hidden_mask,
+    )
 
     update_mask = hidden_mask & valid
     tl.store(state_ptr + state_base, s1, mask=update_mask)
@@ -398,6 +440,524 @@ def _qwen38_ple_m1_short_conv_kernel(
     tl.store(state_ptr + state_base + 8 * STATE_STRIDE_2, x, mask=update_mask)
 
 
+@triton.jit
+def _qwen38_ple_prefill_short_conv_kernel(
+    x_ptr,
+    state_ptr,
+    weight_ptr,
+    output_ptr,
+    q_starts_ptr,
+    state_index_ptr,
+    has_initial_ptr,
+    STATE_STRIDE_0: tl.constexpr,
+    STATE_STRIDE_1: tl.constexpr,
+    STATE_STRIDE_2: tl.constexpr,
+    NUM_REQS: tl.constexpr,
+    BLOCK_R: tl.constexpr,
+    NULL_STATE_ID: tl.constexpr,
+    HIDDEN_SIZE: tl.constexpr,
+    BLOCK: tl.constexpr,
+):
+    """Row-major 4-tap dilated short-conv for Qwen3.8 prefill without transpose."""
+
+    hidden = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
+    row = tl.program_id(1).to(tl.int64)
+    hidden_mask = hidden < HIDDEN_SIZE
+
+    if NUM_REQS == 1:
+        req = tl.zeros([], dtype=tl.int64)
+        req_start = tl.zeros([], dtype=tl.int64)
+    else:
+        req_offsets = tl.arange(0, BLOCK_R)
+        starts = tl.load(
+            q_starts_ptr + req_offsets,
+            mask=req_offsets <= NUM_REQS,
+            other=0x7FFFFFFF,
+        ).to(tl.int64)
+        req = tl.minimum(
+            tl.sum((starts <= row).to(tl.int32), axis=0) - 1,
+            NUM_REQS - 1,
+        ).to(tl.int64)
+        req_start = tl.load(q_starts_ptr + req).to(tl.int64)
+
+    col = row - req_start
+    state_index = tl.load(state_index_ptr + req).to(tl.int64)
+    valid = state_index != NULL_STATE_ID
+    safe_state_index = tl.where(valid, state_index, 0)
+    use_initial = valid & (tl.load(has_initial_ptr + req).to(tl.int32) != 0)
+    state_base = safe_state_index * STATE_STRIDE_0 + hidden * STATE_STRIDE_1
+
+    hist0 = col
+    t0_from_x = hist0 >= 9
+    t0_x = tl.load(
+        x_ptr + (req_start + tl.maximum(col - 9, 0)) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & t0_from_x,
+        other=0.0,
+    ).to(tl.float32)
+    t0_s = (
+        tl.load(
+            state_ptr + state_base + tl.minimum(hist0, 8) * STATE_STRIDE_2,
+            mask=hidden_mask & (~t0_from_x) & use_initial,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    t0 = tl.where(t0_from_x, t0_x, t0_s)
+
+    hist1 = col + 3
+    t1_from_x = hist1 >= 9
+    t1_x = tl.load(
+        x_ptr + (req_start + tl.maximum(col - 6, 0)) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & t1_from_x,
+        other=0.0,
+    ).to(tl.float32)
+    t1_s = (
+        tl.load(
+            state_ptr + state_base + tl.minimum(hist1, 8) * STATE_STRIDE_2,
+            mask=hidden_mask & (~t1_from_x) & use_initial,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    t1 = tl.where(t1_from_x, t1_x, t1_s)
+
+    hist2 = col + 6
+    t2_from_x = hist2 >= 9
+    t2_x = tl.load(
+        x_ptr + (req_start + tl.maximum(col - 3, 0)) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & t2_from_x,
+        other=0.0,
+    ).to(tl.float32)
+    t2_s = (
+        tl.load(
+            state_ptr + state_base + tl.minimum(hist2, 8) * STATE_STRIDE_2,
+            mask=hidden_mask & (~t2_from_x) & use_initial,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    t2 = tl.where(t2_from_x, t2_x, t2_s)
+
+    t3 = tl.load(
+        x_ptr + row * HIDDEN_SIZE + hidden,
+        mask=hidden_mask,
+        other=0.0,
+    ).to(tl.float32)
+
+    w0 = tl.load(weight_ptr + hidden * 4, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    w1 = tl.load(weight_ptr + hidden * 4 + 1, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    w2 = tl.load(weight_ptr + hidden * 4 + 2, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    w3 = tl.load(weight_ptr + hidden * 4 + 3, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    conv = t0 * w0
+    conv += t1 * w1
+    conv += t2 * w2
+    conv += t3 * w3
+    conv = conv.to(tl.float16)
+    tl.store(
+        output_ptr + row * HIDDEN_SIZE + hidden,
+        tl.where(valid, conv, 0.0),
+        mask=hidden_mask,
+    )
+
+
+@triton.jit
+def _qwen38_ple_prefill_state_update_kernel(
+    x_ptr,
+    state_ptr,
+    q_starts_ptr,
+    state_index_ptr,
+    has_initial_ptr,
+    STATE_STRIDE_0: tl.constexpr,
+    STATE_STRIDE_1: tl.constexpr,
+    STATE_STRIDE_2: tl.constexpr,
+    NULL_STATE_ID: tl.constexpr,
+    HIDDEN_SIZE: tl.constexpr,
+    BLOCK: tl.constexpr,
+):
+    """Commit the trailing 9 tokens per prefill request into conv_state."""
+
+    hidden = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
+    req = tl.program_id(1).to(tl.int64)
+    hidden_mask = hidden < HIDDEN_SIZE
+
+    req_start = tl.load(q_starts_ptr + req).to(tl.int64)
+    req_end = tl.load(q_starts_ptr + req + 1).to(tl.int64)
+    length = req_end - req_start
+    state_index = tl.load(state_index_ptr + req).to(tl.int64)
+    valid = (state_index != NULL_STATE_ID) & (length > 0)
+    if not valid:
+        return
+    use_initial = tl.load(has_initial_ptr + req).to(tl.int32) != 0
+    state_base = state_index * STATE_STRIDE_0 + hidden * STATE_STRIDE_1
+    state_mask = hidden_mask & use_initial
+
+    s0 = tl.load(state_ptr + state_base, mask=state_mask, other=0.0).to(tl.float16)
+    s1 = tl.load(
+        state_ptr + state_base + STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s2 = tl.load(
+        state_ptr + state_base + 2 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s3 = tl.load(
+        state_ptr + state_base + 3 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s4 = tl.load(
+        state_ptr + state_base + 4 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s5 = tl.load(
+        state_ptr + state_base + 5 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s6 = tl.load(
+        state_ptr + state_base + 6 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s7 = tl.load(
+        state_ptr + state_base + 7 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+    s8 = tl.load(
+        state_ptr + state_base + 8 * STATE_STRIDE_2, mask=state_mask, other=0.0
+    ).to(tl.float16)
+
+    for idx in tl.static_range(9):
+        hist_pos = length + idx
+        from_x = hist_pos >= 9
+        x_val = tl.load(
+            x_ptr
+            + (req_start + tl.maximum(hist_pos - 9, 0)) * HIDDEN_SIZE
+            + hidden,
+            mask=hidden_mask & from_x,
+            other=0.0,
+        ).to(tl.float16)
+        s_val = tl.where(
+            hist_pos == 0,
+            s0,
+            tl.where(
+                hist_pos == 1,
+                s1,
+                tl.where(
+                    hist_pos == 2,
+                    s2,
+                    tl.where(
+                        hist_pos == 3,
+                        s3,
+                        tl.where(
+                            hist_pos == 4,
+                            s4,
+                            tl.where(
+                                hist_pos == 5,
+                                s5,
+                                tl.where(
+                                    hist_pos == 6,
+                                    s6,
+                                    tl.where(hist_pos == 7, s7, s8),
+                                ),
+                            ),
+                        ),
+                    ),
+                ),
+            ),
+        )
+        next_val = tl.where(from_x, x_val, s_val)
+        tl.store(
+            state_ptr + state_base + idx * STATE_STRIDE_2,
+            next_val,
+            mask=hidden_mask,
+        )
+
+
+@triton.jit
+def _qwen38_ple_spec_short_conv_kernel(
+    x_ptr,
+    state_ptr,
+    weight_ptr,
+    output_ptr,
+    state_index_ptr,
+    q_starts_ptr,
+    accepted_ptr,
+    ROWS,
+     NUM_REQS: tl.constexpr,
+    STATE_STRIDE_0: tl.constexpr,
+    STATE_STRIDE_1: tl.constexpr,
+    STATE_STRIDE_2: tl.constexpr,
+    NULL_STATE_ID: tl.constexpr,
+    HIDDEN_SIZE: tl.constexpr,
+    BLOCK: tl.constexpr,
+):
+    """Fused multi-request MTP4 PLE rollback + 4-tap conv + state commit."""
+
+    hidden = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
+    req = tl.program_id(1).to(tl.int64)
+    hidden_mask = hidden < HIDDEN_SIZE
+
+    sid = tl.load(state_index_ptr + req).to(tl.int64)
+    valid_state = sid != NULL_STATE_ID
+    safe_sid = tl.where(valid_state, sid, 0)
+    start = tl.maximum(0, tl.load(q_starts_ptr + req).to(tl.int64))
+    end = tl.maximum(start, tl.load(q_starts_ptr + req + 1).to(tl.int64))
+    count = tl.minimum(5, end - start)
+    accepted = tl.load(accepted_ptr + req).to(tl.int64)
+    rollback = tl.where(valid_state, tl.maximum(0, tl.minimum(4, accepted - 1)), 0)
+    state_base = (
+        safe_sid * STATE_STRIDE_0
+        + hidden * STATE_STRIDE_1
+        + rollback * STATE_STRIDE_2
+    )
+    state_mask = hidden_mask & valid_state
+
+    h0 = (
+        tl.load(state_ptr + state_base, mask=state_mask, other=0.0)
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h1 = (
+        tl.load(
+            state_ptr + state_base + STATE_STRIDE_2, mask=state_mask, other=0.0
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h2 = (
+        tl.load(
+            state_ptr + state_base + 2 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h3 = (
+        tl.load(
+            state_ptr + state_base + 3 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h4 = (
+        tl.load(
+            state_ptr + state_base + 4 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h5 = (
+        tl.load(
+            state_ptr + state_base + 5 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h6 = (
+        tl.load(
+            state_ptr + state_base + 6 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h7 = (
+        tl.load(
+            state_ptr + state_base + 7 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+    h8 = (
+        tl.load(
+            state_ptr + state_base + 8 * STATE_STRIDE_2,
+            mask=state_mask,
+            other=0.0,
+        )
+        .to(tl.float16)
+        .to(tl.float32)
+    )
+
+    h9 = tl.load(
+        x_ptr + start * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & (0 < count) & (start < ROWS),
+        other=0.0,
+    ).to(tl.float32)
+    h10 = tl.load(
+        x_ptr + (start + 1) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & (1 < count) & (start + 1 < ROWS),
+        other=0.0,
+    ).to(tl.float32)
+    h11 = tl.load(
+        x_ptr + (start + 2) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & (2 < count) & (start + 2 < ROWS),
+        other=0.0,
+    ).to(tl.float32)
+    h12 = tl.load(
+        x_ptr + (start + 3) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & (3 < count) & (start + 3 < ROWS),
+        other=0.0,
+    ).to(tl.float32)
+    h13 = tl.load(
+        x_ptr + (start + 4) * HIDDEN_SIZE + hidden,
+        mask=hidden_mask & (4 < count) & (start + 4 < ROWS),
+        other=0.0,
+    ).to(tl.float32)
+
+    w0 = tl.load(weight_ptr + hidden * 4, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    w1 = tl.load(weight_ptr + hidden * 4 + 1, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    w2 = tl.load(weight_ptr + hidden * 4 + 2, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+    w3 = tl.load(weight_ptr + hidden * 4 + 3, mask=hidden_mask, other=0.0).to(
+        tl.float32
+    )
+
+    c0 = (h0 * w0 + h3 * w1 + h6 * w2 + h9 * w3).to(tl.float16)
+    c1 = (h1 * w0 + h4 * w1 + h7 * w2 + h10 * w3).to(tl.float16)
+    c2 = (h2 * w0 + h5 * w1 + h8 * w2 + h11 * w3).to(tl.float16)
+    c3 = (h3 * w0 + h6 * w1 + h9 * w2 + h12 * w3).to(tl.float16)
+    c4 = (h4 * w0 + h7 * w1 + h10 * w2 + h13 * w3).to(tl.float16)
+
+    tl.store(
+        output_ptr + start * HIDDEN_SIZE + hidden,
+        tl.where(0 < count, c0, 0.0),
+        mask=hidden_mask & (0 < end - start) & (start < ROWS),
+    )
+    tl.store(
+        output_ptr + (start + 1) * HIDDEN_SIZE + hidden,
+        tl.where(1 < count, c1, 0.0),
+        mask=hidden_mask & (1 < end - start) & (start + 1 < ROWS),
+    )
+    tl.store(
+        output_ptr + (start + 2) * HIDDEN_SIZE + hidden,
+        tl.where(2 < count, c2, 0.0),
+        mask=hidden_mask & (2 < end - start) & (start + 2 < ROWS),
+    )
+    tl.store(
+        output_ptr + (start + 3) * HIDDEN_SIZE + hidden,
+        tl.where(3 < count, c3, 0.0),
+        mask=hidden_mask & (3 < end - start) & (start + 3 < ROWS),
+    )
+    tl.store(
+        output_ptr + (start + 4) * HIDDEN_SIZE + hidden,
+        tl.where(4 < count, c4, 0.0),
+        mask=hidden_mask & (4 < end - start) & (start + 4 < ROWS),
+    )
+
+    r_pad = start + 5
+    r_pad_end = tl.minimum(end, ROWS)
+    while r_pad < r_pad_end:
+        tl.store(
+            output_ptr + r_pad * HIDDEN_SIZE + hidden,
+            0.0,
+            mask=hidden_mask,
+        )
+        r_pad += 1
+    if req == 0:
+        r_pre = tl.zeros([], dtype=tl.int64)
+        r_pre_end = tl.minimum(start, ROWS)
+        while r_pre < r_pre_end:
+            tl.store(
+                output_ptr + r_pre * HIDDEN_SIZE + hidden,
+                0.0,
+                mask=hidden_mask,
+            )
+            r_pre += 1
+    if req == NUM_REQS - 1:
+        r = end
+        while r < ROWS:
+            tl.store(
+                output_ptr + r * HIDDEN_SIZE + hidden,
+                0.0,
+                mask=hidden_mask,
+            )
+            r += 1
+
+    write_base = safe_sid * STATE_STRIDE_0 + hidden * STATE_STRIDE_1
+    tl.store(
+        state_ptr + write_base,
+        h1.to(tl.float16),
+        mask=state_mask & (0 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + STATE_STRIDE_2,
+        h2.to(tl.float16),
+        mask=state_mask & (1 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 2 * STATE_STRIDE_2,
+        h3.to(tl.float16),
+        mask=state_mask & (2 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 3 * STATE_STRIDE_2,
+        h4.to(tl.float16),
+        mask=state_mask & (3 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 4 * STATE_STRIDE_2,
+        h5.to(tl.float16),
+        mask=state_mask & (4 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 5 * STATE_STRIDE_2,
+        h6.to(tl.float16),
+        mask=state_mask & (5 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 6 * STATE_STRIDE_2,
+        h7.to(tl.float16),
+        mask=state_mask & (6 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 7 * STATE_STRIDE_2,
+        h8.to(tl.float16),
+        mask=state_mask & (7 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 8 * STATE_STRIDE_2,
+        h9.to(tl.float16),
+        mask=state_mask & (8 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 9 * STATE_STRIDE_2,
+        h10.to(tl.float16),
+        mask=state_mask & (9 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 10 * STATE_STRIDE_2,
+        h11.to(tl.float16),
+        mask=state_mask & (10 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 11 * STATE_STRIDE_2,
+        h12.to(tl.float16),
+        mask=state_mask & (11 < 8 + count),
+    )
+    tl.store(
+        state_ptr + write_base + 12 * STATE_STRIDE_2,
+        h13.to(tl.float16),
+        mask=state_mask & (12 < 8 + count),
+    )
+
+
 def _splitmix64(value: int) -> int:
     value = (value + _SPLITMIX_GAMMA) & _MASK64
     value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
@@ -1529,8 +2089,35 @@ class Qwen4ExpNGramEmbedding(PleOffloadLayer):
                 self.ngram_heads_vocab_sizes,
                 self.ngram_heads_offsets,
             )
-            if reason is None:
-                logger.info_once("SM70 fused small-batch PLE ngram-ID path enabled.")
+            can_use_large_batch_ngram = (
+                reason == "outside_small_batch_band"
+                and num_tokens > 32
+                and 1 <= num_reqs <= 64
+                and ngram_context.ndim == 2
+                and ngram_context.shape[0] >= num_reqs
+                and ngram_context.shape[1] == 2
+                and (
+                    self.layer_multipliers.numel(),
+                    self.ngram_heads_vocab_sizes.numel(),
+                    self.ngram_heads_offsets.numel(),
+                )
+                == (3, 16, 16)
+                and all(
+                    tensor.device == input_ids.device
+                    and tensor.is_contiguous()
+                    and tensor.dtype in (torch.int32, torch.int64)
+                    for tensor in (
+                        input_ids,
+                        query_start_loc,
+                        ngram_context,
+                        self.layer_multipliers,
+                        self.ngram_heads_vocab_sizes,
+                        self.ngram_heads_offsets,
+                    )
+                )
+            )
+            if reason is None or can_use_large_batch_ngram:
+                logger.info_once("SM70 fused PLE ngram-ID path enabled.")
                 return sm70_ple_ngram_ids(
                     input_ids,
                     query_start_loc,
@@ -2215,27 +2802,32 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
         state_indices_tensor_d: torch.Tensor,
         has_initial_states_d: torch.Tensor | None,
     ) -> torch.Tensor:
+        num_rows = x_d.shape[0] if x_d.ndim == 2 else 0
         has_initial_ok = has_initial_states_d is None or (
-            has_initial_states_d.numel() >= 1
+            has_initial_states_d.numel() >= num_rows
             and has_initial_states_d.is_cuda
             and has_initial_states_d.is_contiguous()
         )
         if (
             current_platform.is_device_capability((7, 0))
-            and x_d.shape == (1, 10240)
+            and x_d.ndim == 2
+            and num_rows >= 1
+            and x_d.shape[1] == 10240
             and x_d.dtype == torch.float16
             and x_d.is_cuda
             and x_d.is_contiguous()
+            and getattr(self, "conv_state_len", 9) == 9
+            and getattr(self, "short_conv_dilation", 3) == 3
             and conv_state.ndim == 3
             and conv_state.shape[1] == 10240
-            and conv_state.shape[2] == 9
-            and conv_state.dtype == torch.float16
+            and conv_state.shape[2] >= 9
+            and conv_state.dtype in (torch.float16, torch.float32)
             and conv_state.is_cuda
             and conv_weights.shape == (10240, 4)
             and conv_weights.dtype == torch.float16
             and conv_weights.is_cuda
             and conv_weights.is_contiguous()
-            and state_indices_tensor_d.numel() == 1
+            and state_indices_tensor_d.numel() >= num_rows
             and state_indices_tensor_d.dtype in (torch.int32, torch.int64)
             and state_indices_tensor_d.is_cuda
             and state_indices_tensor_d.is_contiguous()
@@ -2255,7 +2847,7 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
                 if has_initial_states_d is None
                 else has_initial_states_d
             )
-            _qwen38_ple_m1_short_conv_kernel[(triton.cdiv(10240, 256),)](
+            _qwen38_ple_m1_short_conv_kernel[(triton.cdiv(10240, 256), num_rows)](
                 x_d,
                 conv_state,
                 conv_weights,
@@ -2386,11 +2978,91 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
 
         hidden_size = x_p.shape[1]
         state_len = self.conv_state_len
-        positions = torch.arange(
-            num_prefill_tokens, device=x_p.device, dtype=torch.int64
+        if (
+            current_platform.is_device_capability((7, 0))
+            and x_p.ndim == 2
+            and num_prefill_tokens >= 1
+            and x_p.shape[0] == num_prefill_tokens
+            and hidden_size == 10240
+            and x_p.dtype == torch.float16
+            and x_p.is_cuda
+            and x_p.is_contiguous()
+            and state_len == 9
+            and getattr(self, "short_conv_dilation", 3) == 3
+            and 1 <= num_prefills <= 64
+            and conv_state.ndim == 3
+            and conv_state.shape[0] > 0
+            and conv_state.shape[1] == 10240
+            and conv_state.shape[2] >= 9
+            and conv_state.dtype in (torch.float16, torch.float32)
+            and conv_state.is_cuda
+            and conv_weights.shape == (10240, 4)
+            and conv_weights.dtype == torch.float16
+            and conv_weights.is_cuda
+            and conv_weights.is_contiguous()
+            and query_start_loc_p.is_cuda
+            and state_indices_tensor_p.is_cuda
+            and has_initial_states_p.is_cuda
+            and x_p.device
+            == conv_state.device
+            == conv_weights.device
+            == query_start_loc_p.device
+            == state_indices_tensor_p.device
+            == has_initial_states_p.device
+        ):
+            q_starts_c = query_start_loc_p[: num_prefills + 1].contiguous()
+            state_indices_c = state_indices_tensor_p[:num_prefills].contiguous()
+            has_initial_c = has_initial_states_p[:num_prefills].contiguous()
+            conv_output = torch.empty_like(x_p)
+            grid_conv = (triton.cdiv(10240, 256), num_prefill_tokens)
+            _qwen38_ple_prefill_short_conv_kernel[grid_conv](
+                x_p,
+                conv_state,
+                conv_weights,
+                conv_output,
+                q_starts_c,
+                state_indices_c,
+                has_initial_c,
+                STATE_STRIDE_0=conv_state.stride(0),
+                STATE_STRIDE_1=conv_state.stride(1),
+                STATE_STRIDE_2=conv_state.stride(2),
+                NUM_REQS=num_prefills,
+                BLOCK_R=triton.next_power_of_2(num_prefills + 1),
+                NULL_STATE_ID=NULL_BLOCK_ID,
+                HIDDEN_SIZE=10240,
+                BLOCK=256,
+                num_warps=4,
+            )
+            grid_state = (triton.cdiv(10240, 256), num_prefills)
+            _qwen38_ple_prefill_state_update_kernel[grid_state](
+                x_p,
+                conv_state,
+                q_starts_c,
+                state_indices_c,
+                has_initial_c,
+                STATE_STRIDE_0=conv_state.stride(0),
+                STATE_STRIDE_1=conv_state.stride(1),
+                STATE_STRIDE_2=conv_state.stride(2),
+                NULL_STATE_ID=NULL_BLOCK_ID,
+                HIDDEN_SIZE=10240,
+                BLOCK=256,
+                num_warps=4,
+            )
+            F.silu(conv_output, inplace=True)
+            logger.info_once(
+                "SM70 Qwen3.8 fused prefill PLE short-conv path enabled."
+            )
+            return conv_output
+
+        single_prefill_fastpath = (
+            num_prefills == 1 and x_p.shape[0] == max_len == num_prefill_tokens
         )
-        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
-        col_indices = positions - q_starts[req_indices]
+        if not single_prefill_fastpath:
+            positions = torch.arange(
+                num_prefill_tokens, device=x_p.device, dtype=torch.int64
+            )
+            req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
+            col_indices = positions - q_starts[req_indices]
 
         state_indices = state_indices_tensor_p[:num_prefills].to(
             device=conv_state.device, dtype=torch.int64
@@ -2409,8 +3081,14 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
         # the profiling run never takes. Keep at most two alive: pack the
         # tokens straight into the channels-first history, take the next state
         # before the convolution, and free each buffer as soon as it is spent.
-        history = x_p.new_zeros((num_prefills, hidden_size, state_len + max_len))
-        history[..., state_len:].transpose(1, 2)[req_indices, col_indices] = x_p
+        if single_prefill_fastpath:
+            history = x_p.new_empty((1, hidden_size, state_len + max_len))
+            if state_len > 0 and conv_state.shape[0] <= 0:
+                history[..., :state_len].zero_()
+            history[..., state_len:] = x_p.transpose(0, 1).unsqueeze(0)
+        else:
+            history = x_p.new_zeros((num_prefills, hidden_size, state_len + max_len))
+            history[..., state_len:].transpose(1, 2)[req_indices, col_indices] = x_p
         next_state = None
         if state_len > 0 and conv_state.shape[0] > 0:
             state = conv_state.index_select(0, state_indices)[..., :state_len].to(
@@ -2446,7 +3124,10 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
         conv_output.masked_fill_(
             ~valid_state.to(device=x_p.device).view(num_prefills, 1, 1), 0
         )
-        output = conv_output.transpose(1, 2)[req_indices, col_indices]
+        if single_prefill_fastpath:
+            output = conv_output.squeeze(0).transpose(0, 1).contiguous()
+        else:
+            output = conv_output.transpose(1, 2)[req_indices, col_indices]
         del conv_output
 
         if next_state is not None:
@@ -2492,18 +3173,23 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
         hidden_size = x_spec.size(-1)
         if (
             layer_policy().value("ple_spec_conv")
-            and num_reqs == 1
+            and 1 <= num_reqs <= 64
             and spec_query_len == 5
             and self.conv_state_len == 9
             and self.short_conv_dilation == 3
-            and x_spec.shape in ((5, 10240), (10, 10240))
+            and x_spec.ndim == 2
+            and x_spec.shape[0] >= 5
+            and x_spec.shape[1] == 10240
             and x_spec.is_cuda
             and x_spec.dtype == torch.float16
             and x_spec.is_contiguous()
             and conv_weights.shape == (10240, 4)
             and conv_weights.dtype == torch.float16
             and conv_weights.is_contiguous()
-            and conv_state.shape[1:] == (10240, 13)
+            and conv_state.ndim == 3
+            and conv_state.shape[0] > 0
+            and conv_state.shape[1] == 10240
+            and conv_state.shape[2] >= 13
             and conv_state.dtype in (torch.float16, torch.float32)
             and all(
                 t.dtype == torch.int32 and t.is_cuda and t.is_contiguous()
@@ -2516,16 +3202,45 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
             and current_platform.is_device_capability((7, 0))
         ):
             output = torch.empty_like(x_spec)
-            torch.ops._C.qwen38_ple_spec_sm70_out(
-                output,
-                conv_state,
+            if hasattr(torch.ops._C, "qwen38_ple_spec_sm70_out") and (
+                num_reqs == 1 and x_spec.shape in ((5, 10240), (10, 10240))
+                and conv_state.shape[1:] == (10240, 13)
+            ):
+                torch.ops._C.qwen38_ple_spec_sm70_out(
+                    output,
+                    conv_state,
+                    x_spec,
+                    conv_weights,
+                    spec_state_indices_tensor,
+                    spec_query_start_loc,
+                    num_accepted_tokens,
+                )
+                logger.info_once(
+                    "SM70 MTP4 PLE rollback/conv/SiLU/state fusion enabled."
+                )
+                return output
+            _qwen38_ple_spec_short_conv_kernel[(triton.cdiv(10240, 256), num_reqs)](
                 x_spec,
+                conv_state,
                 conv_weights,
+                output,
                 spec_state_indices_tensor,
                 spec_query_start_loc,
                 num_accepted_tokens,
+                x_spec.shape[0],
+                NUM_REQS=num_reqs,
+                STATE_STRIDE_0=conv_state.stride(0),
+                STATE_STRIDE_1=conv_state.stride(1),
+                STATE_STRIDE_2=conv_state.stride(2),
+                NULL_STATE_ID=NULL_BLOCK_ID,
+                HIDDEN_SIZE=10240,
+                BLOCK=256,
+                num_warps=4,
+            )
+            F.silu(output, inplace=True)
+            logger.info_once(
+                "SM70 MTP4 multi-request PLE short-conv Triton fusion enabled."
             )
-            logger.info_once("SM70 MTP4 PLE rollback/conv/SiLU/state fusion enabled.")
             return output
         # Use a fixed packing width instead of synchronizing on lengths.max().
         max_len = spec_query_len
@@ -2799,7 +3514,11 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
             inputs,
             layer_attn_metadata,
             conv_state,
-            conv_weights.to(dtype=inputs.dtype),
+            (
+                conv_weights
+                if conv_weights.dtype == inputs.dtype
+                else conv_weights.to(dtype=inputs.dtype)
+            ),
         )
 
     def forward(
@@ -2850,7 +3569,7 @@ class Qwen4ExpPLELayer(nn.Module, MambaBase):
         normalized = snapshot_ple_diagnostic(
             normalized, self.prefix + ":07_conv_input", diagnostic
         )
-        conv_output = torch.zeros_like(normalized)
+        conv_output = torch.empty_like(normalized)
         torch.ops.vllm.qwen4_exp_ple_short_conv(
             normalized,
             conv_output,
@@ -2869,7 +3588,11 @@ def qwen4_exp_ple_short_conv(
 ) -> None:
     layer = get_forward_context().no_compile_layers[layer_name]
     result = layer._short_conv(inputs)
-    output[: result.shape[0]].copy_(result)
+    active_rows = result.shape[0]
+    if active_rows:
+        output[:active_rows].copy_(result)
+    if active_rows < output.shape[0]:
+        output[active_rows:].zero_()
 
 
 def qwen4_exp_ple_short_conv_fake(
```

---

### 5.20 `vllm/models/qwen4_exp/nvidia/qsa.py` — QSA 3D 门控视图直传（P2）、DCP A2A 融合门控（D1）、跳过冗余清零（D4）、异步 AllGather 重叠 Indexer（D14）、解除 qsa_prep_sm70_out 32 行上限（P12 / D19）与 DCP2 valid_counts 透传（P11 / D17）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/qsa.py b/vllm/models/qwen4_exp/nvidia/qsa.py
index 7826024..c7d9acd 100644
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
@@ -258,6 +268,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         output_scale: torch.Tensor | None = None,
         output_block_scale: torch.Tensor | None = None,
         dense_short_context: bool = False,
+        dcp_ag_state: Any | None = None,
     ) -> torch.Tensor:
         del key, value
         if output_scale is not None or output_block_scale is not None:
@@ -268,10 +279,14 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
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
@@ -300,9 +315,15 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
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
@@ -348,6 +369,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
                 token_to_req,
                 output[:num_tokens],
                 output_gate[:num_tokens],
+                dcp_ag_state=dcp_ag_state,
             )
             return output
 
@@ -373,6 +395,92 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         )
         return output
 
+    def _start_all_gather_dcp_query(
+        self,
+        query: torch.Tensor,
+        group: Any | None = None,
+    ) -> tuple[Any, torch.Tensor, torch.Tensor | None, int, int, int]:
+        if group is None:
+            from vllm.distributed import get_dcp_group
+
+            group = get_dcp_group()
+        q_contig = query if query.is_contiguous() else query.contiguous()
+        device_group = getattr(group, "device_group", None)
+        if device_group is None and hasattr(group, "device_communicator"):
+            device_group = getattr(group.device_communicator, "device_group", None)
+        if device_group is None:
+            return (None, group.all_gather(q_contig, dim=1), None, 0, 0, 0)
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
+        out_buf: torch.Tensor | None = None
+        if rows > 1:
+            out_buf = getattr(self, "_dcp_ag_out_buffer", None)
+            if (
+                out_buf is None
+                or out_buf.numel() < needed
+                or out_buf.dtype != q_contig.dtype
+                or out_buf.device != q_contig.device
+            ):
+                cap = 1 << (max(needed, 1) - 1).bit_length()
+                if (
+                    out_buf is not None
+                    and out_buf.device == q_contig.device
+                    and out_buf.dtype == q_contig.dtype
+                ):
+                    cap = max(cap, out_buf.numel())
+                out_buf = torch.empty(
+                    (cap,), dtype=q_contig.dtype, device=q_contig.device
+                )
+                self._dcp_ag_out_buffer = out_buf
+        work = torch.distributed.all_gather_into_tensor(
+            recv_view, q_contig, group=device_group, async_op=True
+        )
+        return (work, recv_view, out_buf, rows, heads, dim)
+
+    def _finish_all_gather_dcp_query(
+        self,
+        ag_state: tuple[Any, torch.Tensor, torch.Tensor | None, int, int, int],
+    ) -> torch.Tensor:
+        work, recv_view, out_buf, rows, heads, dim = ag_state
+        if work is None:
+            return recv_view
+        work.wait()
+        world_size = self.dcp_world_size
+        if rows == 1:
+            return recv_view.view(1, world_size * heads, dim)
+        assert out_buf is not None
+        needed = world_size * rows * heads * dim
+        out_view = out_buf[:needed].view(rows, world_size, heads, dim)
+        out_view.copy_(recv_view.view(world_size, rows, heads, dim).movedim(0, 1))
+        return out_view.view(rows, world_size * heads, dim)
+
+    def _all_gather_dcp_query(
+        self,
+        query: torch.Tensor,
+        group: Any,
+    ) -> torch.Tensor:
+        return self._finish_all_gather_dcp_query(
+            self._start_all_gather_dcp_query(query, group)
+        )
+
     def _forward_qsa_dcp(
         self,
         layer: torch.nn.Module,
@@ -384,8 +492,13 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         token_to_req: torch.Tensor,
         output: torch.Tensor,
         output_gate: torch.Tensor,
+        dcp_ag_state: Any | None = None,
     ) -> None:
         from vllm.distributed import get_dcp_group
+        from vllm.v1.attention.ops.dcp_alltoall import (
+            _dcp_a2a_send_recv_buffers,
+            dcp_a2a_lse_reduce,
+        )
 
         from .ops.qsa import _qsa_output_gate, qsa_sparse_paged_attention
         from .ops.qsa_dcp import (
@@ -405,33 +518,73 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             )
         ):
             raise RuntimeError("QSA DCP target workspaces are not initialized")
+        group = get_dcp_group()
+        if dcp_ag_state is None:
+            dcp_ag_state = self._start_all_gather_dcp_query(query, group)
+        # Only this prefix of the compacted columns can hold an owned token.
+        local_width = qsa_dcp_local_selection_width(
+            layer.indexer.token_topk,
+            layer.indexer.compress_ratio,
+            self.dcp_world_size,
+            layer.cp_kv_cache_interleave_size,
+            logical_indices.shape[1],
+        )
+        resolve_physical = bool(key_cache.shape[0] * key_cache.shape[1] < 2**31)
+        rows = query.shape[0]
+        counts_buf = getattr(self, "_dcp_valid_counts_buffer", None)
+        if (
+            counts_buf is None
+            or counts_buf.numel() < rows
+            or counts_buf.device != logical_indices.device
+        ):
+            cap = 1 << (max(rows, 1) - 1).bit_length()
+            if counts_buf is not None and counts_buf.device == logical_indices.device:
+                cap = max(cap, counts_buf.numel())
+            counts_buf = torch.empty(
+                (cap,), dtype=torch.int32, device=logical_indices.device
+            )
+            self._dcp_valid_counts_buffer = counts_buf
+        local_valid_counts = counts_buf[:rows]
         local_indices = qsa_localize_dcp_indices(
             logical_indices,
-            local_indices_buffer[: query.shape[0]],
+            local_indices_buffer[:rows, :local_width],
             dcp_world_size=self.dcp_world_size,
             dcp_rank=self.dcp_rank,
             interleave_size=layer.cp_kv_cache_interleave_size,
             local_block_size=key_cache.shape[1],
+            block_table=block_table if resolve_physical else None,
+            token_to_req=token_to_req if resolve_physical else None,
+            num_cache_blocks=key_cache.shape[0] if resolve_physical else None,
+            valid_counts=local_valid_counts,
         )
-        group = get_dcp_group()
-        gathered_query = group.all_gather(query.contiguous(), dim=1)
-        partial_output = partial_output_buffer[: query.shape[0]]
-        partial_lse = partial_lse_buffer[: query.shape[0]]
+        partial_output = partial_output_buffer[:rows]
+        partial_lse = partial_lse_buffer[:rows]
+        use_fused_a2a_pack = bool(
+            self.dcp_combine is dcp_a2a_lse_reduce
+            and self.dcp_world_size > 1
+            and partial_output.dtype == torch.float32
+        )
+        send_buffer: torch.Tensor | None = None
+        recv_buffer: torch.Tensor | None = None
+        if use_fused_a2a_pack:
+            send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
+                (
+                    self.dcp_world_size,
+                    rows,
+                    query.shape[1],
+                    query.shape[2] + 1,
+                ),
+                device=partial_output.device,
+                dtype=partial_output.dtype,
+            )
+        gathered_query = self._finish_all_gather_dcp_query(dcp_ag_state)
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
@@ -439,7 +592,23 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
             k_scale=layer._k_scale_float,
             v_scale=layer._v_scale_float,
             lse=partial_lse,
+            pre_resolved_indices=resolve_physical,
+            dcp_packed_out=send_buffer,
+            valid_counts=local_valid_counts,
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
+                packed_send_buffer=send_buffer,
+                recv_buffer=recv_buffer,
+            )
+            return
         merged = cast(
             torch.Tensor,
             self.dcp_combine(
@@ -453,7 +622,7 @@ class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
         # apply the Triton gate DCP1's page4 route uses. This replaces a chain
         # of five elementwise kernels (12.2 us per decode layer on V100).
         output.copy_(merged)
-        _qsa_output_gate(output, output_gate.view_as(output))
+        _qsa_output_gate(output, gate_view)
 
 
 def _verify_e4m3_kv_requirements(
@@ -862,6 +1031,8 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
         self,
         qkv: torch.Tensor,
         positions: torch.Tensor,
+        *,
+        flatten_gate: bool = True,
     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
         """Split, normalize, and rotate Q/K using this tree's Qwen3 API."""
         q_gate, key, value = qkv.split(
@@ -875,7 +1046,9 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             key.view(*token_shape, self.num_kv_heads, self.head_dim)
         ).reshape(*token_shape, self.kv_size)
         query, key = self.rotary_emb(positions, query, key)
-        return query, key, value, gate.reshape(*token_shape, self.q_size)
+        if flatten_gate:
+            gate = gate.reshape(*token_shape, self.q_size)
+        return query, key, value, gate
 
     def _use_dense_short_context(
         self, num_tokens: int, side_metadata: QSAForwardMetadata
@@ -956,34 +1129,53 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
                 value[:num_tokens],
             )
         dense = self._use_dense_short_context(num_tokens, side_metadata)
-        selected = self.indexer(
-            hidden_states,
-            positions,
-            self.topk_indices_buffer[:num_tokens],
-            select=not dense,
-        )
-        if selected.shape != (
-            num_tokens,
-            self.indexer.output_width,
-        ):
-            raise RuntimeError("QSA indexer returned an invalid selection shape")
-        selected = _sm70_dump_qwen_layer_tensor(
-            "qsa_selected_indices",
-            self.indexer.layer_id,
-            "qsa",
-            selected,
-        )
         impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
         if self.host_kv_enabled:
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
@@ -991,7 +1183,7 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
                 prepared,
                 key_cache,
                 value_cache,
-                main_metadata.slot_mapping[:num_tokens].to(torch.int64),
+                slot_mapping,
             )
             query = key = value = prepared
         else:
@@ -1002,6 +1194,26 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
                 self.kv_cache,
                 main_metadata.slot_mapping,
             )
+        dcp_ag_state = None
+        if getattr(self, "qsa_dcp_sharded", False) and num_tokens > 0 and not dense:
+            dcp_ag_state = impl._start_all_gather_dcp_query(query[:num_tokens])
+        selected = self.indexer(
+            hidden_states,
+            positions,
+            self.topk_indices_buffer[:num_tokens],
+            select=not dense,
+        )
+        if selected.shape != (
+            num_tokens,
+            self.indexer.output_width,
+        ):
+            raise RuntimeError("QSA indexer returned an invalid selection shape")
+        selected = _sm70_dump_qwen_layer_tensor(
+            "qsa_selected_indices",
+            self.indexer.layer_id,
+            "qsa",
+            selected,
+        )
         impl.forward_qsa(
             self,
             query,
@@ -1015,6 +1227,7 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             query_positions=side_metadata.logical_positions,
             sequence_lengths=side_metadata.seq_lens,
             dense_short_context=dense,
+            dcp_ag_state=dcp_ag_state,
         )
         _sm70_dump_qwen_layer_tensor(
             "qsa_core_out",
@@ -1037,18 +1250,20 @@ class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
             qkv, indexer_input = side(hidden_states)
         else:
             qkv, _ = self.qkv_proj(hidden_states)
-        if self._sm70_qsa_prep and num_tokens <= 32:
+        if self._sm70_qsa_prep and num_tokens >= 1:
             # Raw rows go to the attention op, which normalizes, rotates and
             # caches them in one launch; the gate stays strided in q_gate.
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

### 5.21 `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` — GDN 投影拆分网格缩减（P5）、Key-only index_qk_proj 128×2560 计划（D22）与 shared_expert.down_proj 专用 GEMV 计划（D23）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py b/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
index 14b4750..b25d2fe 100644
--- a/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
+++ b/vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py
@@ -47,6 +47,7 @@ _QSA_OUT_SUFFIX = ".self_attn.o_proj"
 _QSA_INDEX_SUFFIX = ".self_attn.indexer.index_qk_proj"
 _ROUTER_SUFFIX = ".mlp.gate"
 _SHARED_UP_SUFFIX = ".mlp.shared_expert.gate_up_proj"
+_SHARED_DOWN_SUFFIX = ".mlp.shared_expert.down_proj"
 
 # Plans are cold-cache CUDA Graph winners on real checkpoint weights. Keep the
 # role in the key: GDN and QSA can share a physical shape while remaining
@@ -83,11 +84,15 @@ def _qwen38_gdn_projection_split_kernel(
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
@@ -120,7 +125,7 @@ def _split_gdn_projection_tails(qkvz, ba, z_out):
     m = qkvz.shape[0]
     qkv = qkvz[:, :2560]
     b, a = (qkvz.new_empty((m, 12)) for _ in range(2))
-    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(4096, 256))](
+    _qwen38_gdn_projection_split_kernel[(m, triton.cdiv(1536, 256))](
         qkvz,
         ba,
         qkv,
@@ -714,6 +719,10 @@ class Qwen38SM70FP16LinearMethod(UnquantizedLinearMethod):
 
 
 def _plan_for(prefix: str, shape: tuple[int, int]) -> _GemvPlan | None:
+    if prefix.endswith(_QSA_INDEX_SUFFIX) and shape == (128, 2560):
+        return _GemvPlan(512, 2, 0)
+    if prefix.endswith(_SHARED_DOWN_SUFFIX) and shape == (2560, 160):
+        return _GemvPlan(256, 2, 0)
     for suffix, expected_shape, plan in _ROLE_PLANS:
         if prefix.endswith(suffix) and shape == expected_shape:
             return plan
```

---

### 5.22 `vllm/models/qwen4_exp/nvidia/sm70_mtp_fc.py` — MTP Prefill 单次 AllGather 合并 fc（P8）与 Decode 单内核融合 _qwen38_mtp_fc_fused_gemv_add_kernel（D24）

```diff
diff --git a/vllm/models/qwen4_exp/nvidia/sm70_mtp_fc.py b/vllm/models/qwen4_exp/nvidia/sm70_mtp_fc.py
index 658f74f..f621be8 100644
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
+
+
+def _qwen38_sm70_mtp_fc_local_add(
+    e: torch.Tensor,
+    h: torch.Tensor,
+    w_e: torch.Tensor,
+    w_h: torch.Tensor,
+) -> torch.Tensor:
+    rows = e.shape[0]
+    if (
+        HAS_TRITON
+        and e.device.type == "cuda"
+        and 1 <= rows <= 4
+        and e.shape == (rows, 2560)
+        and h.shape == (rows, 4, 2560)
+        and w_e.shape == (640, 2560)
+        and w_h.shape == (640, 2560)
+        and e.dtype == h.dtype == w_e.dtype == w_h.dtype == torch.float16
+        and e.is_contiguous()
+        and h.is_contiguous()
+        and w_e.is_contiguous()
+        and w_h.is_contiguous()
+        and not getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
+    ):
+        out = torch.empty((rows, 4, 640), dtype=e.dtype, device=e.device)
+        _qwen38_mtp_fc_fused_gemv_add_kernel[(640, rows)](
+            e,
+            h,
+            w_e,
+            w_h,
+            out,
+            K=2560,
+            N=640,
+            BLOCK_K=512,
+            num_warps=2,
+        )
+        return out
+    el = torch.nn.functional.linear(e, w_e)
+    hl = torch.nn.functional.linear(h, w_h)
+    return el.unsqueeze(-2) + hl
+
+
+def _qwen38_sm70_mtp_fc_local_add_fake(
+    e: torch.Tensor,
+    h: torch.Tensor,
+    w_e: torch.Tensor,
+    w_h: torch.Tensor,
+) -> torch.Tensor:
+    del h, w_h
+    return e.new_empty((e.shape[0], 4, w_e.shape[0]))
+
+
+direct_register_custom_op(
+    op_name="qwen38_sm70_mtp_fc_local_add",
+    op_func=_qwen38_sm70_mtp_fc_local_add,
+    fake_impl=_qwen38_sm70_mtp_fc_local_add_fake,
+)
 
 
 @torch.compiler.assume_constant_result
@@ -22,7 +130,6 @@ def _sm70_device():
 def _can_combine_fc(model, embedding, hidden):
     if (
         envs.VLLM_BATCH_INVARIANT
-        or not is_sm70_decode_graph_compiling()
         or embedding.ndim != 2
         or hidden.ndim != 2
         or embedding.shape[-1] != 2560
@@ -35,7 +142,8 @@ def _can_combine_fc(model, embedding, hidden):
     ):
         return False
     return all(
-        type(layer) is ColumnParallelLinear
+        hasattr(model, attr)
+        and type(layer := getattr(model, attr)) is ColumnParallelLinear
         and type(layer.quant_method) is UnquantizedLinearMethod
         and layer.tp_size == 4
         and layer.gather_output
@@ -43,7 +151,7 @@ def _can_combine_fc(model, embedding, hidden):
         and layer.bias is None
         and layer.weight.shape == (640, 2560)
         and layer.weight.dtype == torch.float16
-        for layer in (model.fc_embedding, model.fc_hidden)
+        for attr in ("fc_embedding", "fc_hidden")
     )
 
 
@@ -53,8 +161,14 @@ def maybe_combine_fc(model, embedding, hidden):
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
     # Addition commutes with the feature gather, with exactly the same FP16
     # rounding per element. Norms/projections and TP ownership are unchanged.
-    return tensor_model_parallel_all_gather(el.unsqueeze(-2) + hl).flatten(-2)
+    return tensor_model_parallel_all_gather(local_sum).flatten(-2)
```

---

### 5.23 `vllm/v1/attention/backends/gdn_attn.py` — GDN Decode 向量化草稿块表展开与延迟计算 num_computed_tokens（D5, D6）

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

### 5.24 `vllm/v1/attention/ops/dcp_alltoall.py` — DCP2 All-to-All LSE 解包合并 + FP16 舍入 + Sigmoid 输出门控融合及打包直传（D1, D13）

```diff
diff --git a/vllm/v1/attention/ops/dcp_alltoall.py b/vllm/v1/attention/ops/dcp_alltoall.py
index 1469a5c..3e37f55 100644
--- a/vllm/v1/attention/ops/dcp_alltoall.py
+++ b/vllm/v1/attention/ops/dcp_alltoall.py
@@ -103,6 +103,11 @@ def _lse_weighted_combine(
     return result
 
 
+_DCP_A2A_BUFFERS: dict[
+    tuple[torch.device, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]
+] = {}
+
+
 def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
     bits = torch.finfo(output_dtype).bits
     if bits == 16:
@@ -124,6 +129,24 @@ def _dcp_a2a_send_recv_buffers(
         )
         return send_buffer, recv_buffer
 
+    needed = 1
+    for dim in shape:
+        needed *= dim
+    if device.type == "cuda" and not torch.cuda.is_current_stream_capturing():
+        stream_id = torch.cuda.current_stream(device).cuda_stream
+        key = (device, stream_id, dtype)
+        cached = _DCP_A2A_BUFFERS.get(key)
+        if cached is None or cached[0].numel() < needed:
+            cap = 1 << (max(needed, 1) - 1).bit_length()
+            if cached is not None:
+                cap = max(cap, cached[0].numel())
+            cached = (
+                torch.empty((cap,), device=device, dtype=dtype),
+                torch.empty((cap,), device=device, dtype=dtype),
+            )
+            _DCP_A2A_BUFFERS[key] = cached
+        return cached[0][:needed].view(shape), cached[1][:needed].view(shape)
+
     return (
         torch.empty(shape, device=device, dtype=dtype),
         torch.empty(shape, device=device, dtype=dtype),
@@ -198,6 +221,7 @@ def _dcp_a2a_unpack_combine_kernel(
     recv_ptr,
     out_ptr,
     out_lse_ptr,
+    gate_ptr,
     recv_stride_N,
     recv_stride_B,
     recv_stride_H,
@@ -207,11 +231,15 @@ def _dcp_a2a_unpack_combine_kernel(
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
@@ -310,6 +338,18 @@ def _dcp_a2a_unpack_combine_kernel(
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
@@ -353,23 +393,32 @@ def _dcp_a2a_unpack_combine(
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
@@ -377,15 +426,20 @@ def _dcp_a2a_unpack_combine(
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
 
@@ -397,6 +451,10 @@ def dcp_a2a_lse_reduce(
     ctx: CPTritonContext | None = None,
     return_lse: bool = False,
     is_lse_base_on_e: bool = True,
+    out: torch.Tensor | None = None,
+    output_gate: torch.Tensor | None = None,
+    packed_send_buffer: torch.Tensor | None = None,
+    recv_buffer: torch.Tensor | None = None,
 ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
     """
     Combine partial attention outputs across DCP ranks using All-to-All.
@@ -411,6 +469,13 @@ def dcp_a2a_lse_reduce(
         ctx: CPTritonContext (unused, for signature compatibility)
         return_lse: If True, also return the combined global LSE
         is_lse_base_on_e: If True, LSE is base e; if False, base 2
+        out: Optional preallocated [B, H/N, D] output buffer
+        output_gate: Optional [B, H/N, D] sigmoid gate applied after rounding to
+            out.dtype
+        packed_send_buffer: Optional pre-packed [N, B, H/N, D + lse_pack_dim]
+            send buffer written directly by the upstream attention merge kernel
+        recv_buffer: Optional preallocated [N, B, H/N, D + lse_pack_dim] receive
+            buffer paired with packed_send_buffer
 
     Returns:
         Combined output [B, H/N, D] (head-scattered)
@@ -419,9 +484,17 @@ def dcp_a2a_lse_reduce(
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
@@ -429,21 +502,23 @@ def dcp_a2a_lse_reduce(
     H_per_rank = H // world_size
     lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)
 
-    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
-        (world_size, B, H_per_rank, D + lse_pack_dim),
-        device=cp_attn_out.device,
-        dtype=cp_attn_out.dtype,
-    )
-
-    _dcp_a2a_pack_send(
-        cp_attn_out,
-        cp_attn_lse,
-        send_buffer,
-        world_size,
-        H_per_rank,
-        D,
-        lse_pack_dim,
-    )
+    if packed_send_buffer is not None and recv_buffer is not None:
+        send_buffer = packed_send_buffer
+    else:
+        send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
+            (world_size, B, H_per_rank, D + lse_pack_dim),
+            device=cp_attn_out.device,
+            dtype=cp_attn_out.dtype,
+        )
+        _dcp_a2a_pack_send(
+            cp_attn_out,
+            cp_attn_lse,
+            send_buffer,
+            world_size,
+            H_per_rank,
+            D,
+            lse_pack_dim,
+        )
 
     work = dist.all_to_all_single(
         recv_buffer.view(-1),
@@ -454,5 +529,11 @@ def dcp_a2a_lse_reduce(
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

### 5.25 `vllm/v1/spec_decode/llm_base_proposer.py` — Chunked Prefill 中间块提前退出（P7）、接入 index_share_for_mtp_iteration（D18）与 DP2×TP4 dummy_run AllReduce 对齐（D26）

```diff
diff --git a/vllm/v1/spec_decode/llm_base_proposer.py b/vllm/v1/spec_decode/llm_base_proposer.py
index f547596..60391c7 100644
--- a/vllm/v1/spec_decode/llm_base_proposer.py
+++ b/vllm/v1/spec_decode/llm_base_proposer.py
@@ -1167,6 +1167,8 @@ class SpecDecodeBaseProposer:
             pre_forward_stream_wait_ms = (time.perf_counter() - sync_t0) * 1000.0
         forward_enqueue_t0 = time.perf_counter() if ddtree_worker_profile else 0.0
         first_forward_start = self._step_profiler.start(profile_events)
+        if getattr(self, "share_mtp_topk_indices", False):
+            self.model.model.set_skip_topk(False)
         with set_forward_context(
             per_layer_attn_metadata,
             self.vllm_config,
@@ -1185,6 +1187,28 @@ class SpecDecodeBaseProposer:
             else:
                 last_hidden_states, hidden_states = ret_hidden_states
         self._step_profiler.finish(profile_events, "first_forward", first_forward_start)
+        is_all_chunked_prefill_fn = getattr(
+            self.runner, "_is_all_reqs_chunked_prefill", None
+        )
+        if (
+            is_all_chunked_prefill_fn is not None
+            and getattr(getattr(self.runner, "input_batch", None), "num_reqs", 0) > 0
+            and is_all_chunked_prefill_fn()
+        ):
+            self._step_profiler.finish(profile_events, "total_gpu", profile_total_start)
+            self._step_profiler.add_cpu_ms(
+                profile_cpu_ms, "total_wall_cpu", profile_wall_start
+            )
+            self._step_profiler.report_proposal(
+                profile_events, profile_cpu_ms, batch_size, num_tokens
+            )
+            if isinstance(self._static_draft_vocab, DynamicDraftVocabRuntime):
+                self._static_draft_vocab.end_proposal()
+            return torch.zeros(
+                (batch_size, self.num_speculative_tokens),
+                dtype=torch.int32,
+                device=self.device,
+            )
         if ddtree_worker_profile:
             forward_enqueue_ms = (time.perf_counter() - forward_enqueue_t0) * 1000.0
             sync_t0 = time.perf_counter()
@@ -1397,102 +1421,119 @@ class SpecDecodeBaseProposer:
 
         block_size = self.block_size
         assert block_size > 0, "block_size has not been initialized."
-        for token_index in range(self.num_speculative_tokens - 1):
-            spec_step_idx = token_index + 1
-            loop_cpu_start = time.perf_counter() if profile_events is not None else 0.0
-            # Update the inputs.
-            # cast to int32 is crucial when eagle model is compiled.
-            # tensor.argmax() returns int64 by default.
-            input_ids = draft_token_ids_list[-1].int()
-
-            if not self.constant_draft_positions:
-                positions = self._update_positions_dependent_metadata(
-                    positions,
-                    common_attn_metadata,
-                    batch_size,
-                    input_batch_size,
-                    block_size,
+        share_mtp_topk_indices = bool(
+            getattr(self, "share_mtp_topk_indices", False)
+            and self.num_speculative_tokens > 1
+        )
+        if share_mtp_topk_indices:
+            self.model.model.compact_topk_indices(token_indices_to_sample[:batch_size])
+            self.model.model.set_skip_topk(True)
+        try:
+            for token_index in range(self.num_speculative_tokens - 1):
+                spec_step_idx = token_index + 1
+                loop_cpu_start = (
+                    time.perf_counter() if profile_events is not None else 0.0
                 )
-
-            # Rebuild attention metadata. When draft positions are constant
-            # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
-            # loop iterations so we build once and reuse.
-            if not self.constant_draft_positions or token_index == 0:
-                _, per_layer_attn_metadata = (
-                    self.build_per_group_and_layer_attn_metadata(
-                        common_attn_metadata, draft_index=spec_step_idx
+                # Update the inputs.
+                # cast to int32 is crucial when eagle model is compiled.
+                # tensor.argmax() returns int64 by default.
+                input_ids = draft_token_ids_list[-1].int()
+
+                if not self.constant_draft_positions:
+                    positions = self._update_positions_dependent_metadata(
+                        positions,
+                        common_attn_metadata,
+                        batch_size,
+                        input_batch_size,
+                        block_size,
                     )
-                )
 
-            # copy inputs to buffer for cudagraph
-            self.input_ids[:batch_size] = input_ids
-            self.hidden_states[:batch_size] = hidden_states
-            if self.supports_mm_inputs:
-                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
+                # Rebuild attention metadata. When draft positions are constant
+                # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
+                # loop iterations so we build once and reuse.
+                if not self.constant_draft_positions or token_index == 0:
+                    _, per_layer_attn_metadata = (
+                        self.build_per_group_and_layer_attn_metadata(
+                            common_attn_metadata, draft_index=spec_step_idx
+                        )
+                    )
 
-                input_ids = None
-                inputs_embeds = self.inputs_embeds[:input_batch_size]
-            else:
-                input_ids = self.input_ids[:input_batch_size]
-                inputs_embeds = None
-
-            # Run the model.
-            model_kwargs = {
-                "input_ids": input_ids,
-                "positions": self._get_positions(input_batch_size),
-                "inputs_embeds": inputs_embeds,
-            }
-            if self.pass_hidden_states_to_model:
-                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
-            model_kwargs = self._add_spec_step_idx(model_kwargs, spec_step_idx)
-            model_kwargs = self._prepare_model_kwargs_for_aot(model_kwargs)
-
-            if profile_events is not None:
-                metadata_name = f"loop{token_index}_metadata_cpu"
-                self._step_profiler.add_cpu_ms(
-                    profile_cpu_ms, metadata_name, loop_cpu_start
-                )
-                profile_cpu_ms["loop_metadata_cpu"] = (
-                    profile_cpu_ms.get("loop_metadata_cpu", 0.0)
-                    + profile_cpu_ms[metadata_name]
-                )
+                # copy inputs to buffer for cudagraph
+                self.input_ids[:batch_size] = input_ids
+                self.hidden_states[:batch_size] = hidden_states
+                if self.supports_mm_inputs:
+                    self.inputs_embeds[:batch_size] = self.model.embed_input_ids(
+                        input_ids
+                    )
 
-            loop_forward_start = self._step_profiler.start(profile_events)
-            with set_forward_context(
-                per_layer_attn_metadata,
-                self.vllm_config,
-                num_tokens=input_batch_size,
-                num_tokens_across_dp=batch_size_across_dp,
-                cudagraph_runtime_mode=cudagraph_runtime_mode,
-                batch_descriptor=self._batch_descriptor_for_spec_step(
-                    batch_descriptor, spec_step_idx
-                ),
-                slot_mapping=self._get_slot_mapping(input_batch_size),
-            ):
-                ret_hidden_states = self.model(**model_kwargs)
-                if not self.model_returns_tuple():
-                    last_hidden_states = ret_hidden_states
-                    hidden_states = ret_hidden_states
+                    input_ids = None
+                    inputs_embeds = self.inputs_embeds[:input_batch_size]
                 else:
-                    last_hidden_states, hidden_states = ret_hidden_states
-            self._step_profiler.finish(
-                profile_events, f"loop{token_index}_forward", loop_forward_start
-            )
+                    input_ids = self.input_ids[:input_batch_size]
+                    inputs_embeds = None
 
-            hidden_states = hidden_states[:batch_size]
-            loop_sample_start = self._step_profiler.start(profile_events)
-            draft_token_ids, draft_probs = self._sample_draft_tokens(
-                last_hidden_states[:batch_size],
-                sampling_metadata,
-                spec_step_idx=spec_step_idx,
-            )
-            self._step_profiler.finish(
-                profile_events, f"loop{token_index}_sample", loop_sample_start
-            )
-            if draft_probs is not None:
-                assert draft_probs_list is not None
-                draft_probs_list.append(draft_probs)
-            draft_token_ids_list.append(draft_token_ids)
+                # Run the model.
+                model_kwargs = {
+                    "input_ids": input_ids,
+                    "positions": self._get_positions(input_batch_size),
+                    "inputs_embeds": inputs_embeds,
+                }
+                if self.pass_hidden_states_to_model:
+                    model_kwargs["hidden_states"] = self.hidden_states[
+                        :input_batch_size
+                    ]
+                model_kwargs = self._add_spec_step_idx(model_kwargs, spec_step_idx)
+                model_kwargs = self._prepare_model_kwargs_for_aot(model_kwargs)
+
+                if profile_events is not None:
+                    metadata_name = f"loop{token_index}_metadata_cpu"
+                    self._step_profiler.add_cpu_ms(
+                        profile_cpu_ms, metadata_name, loop_cpu_start
+                    )
+                    profile_cpu_ms["loop_metadata_cpu"] = (
+                        profile_cpu_ms.get("loop_metadata_cpu", 0.0)
+                        + profile_cpu_ms[metadata_name]
+                    )
+
+                loop_forward_start = self._step_profiler.start(profile_events)
+                with set_forward_context(
+                    per_layer_attn_metadata,
+                    self.vllm_config,
+                    num_tokens=input_batch_size,
+                    num_tokens_across_dp=batch_size_across_dp,
+                    cudagraph_runtime_mode=cudagraph_runtime_mode,
+                    batch_descriptor=self._batch_descriptor_for_spec_step(
+                        batch_descriptor, spec_step_idx
+                    ),
+                    slot_mapping=self._get_slot_mapping(input_batch_size),
+                ):
+                    ret_hidden_states = self.model(**model_kwargs)
+                    if not self.model_returns_tuple():
+                        last_hidden_states = ret_hidden_states
+                        hidden_states = ret_hidden_states
+                    else:
+                        last_hidden_states, hidden_states = ret_hidden_states
+                self._step_profiler.finish(
+                    profile_events, f"loop{token_index}_forward", loop_forward_start
+                )
+
+                hidden_states = hidden_states[:batch_size]
+                loop_sample_start = self._step_profiler.start(profile_events)
+                draft_token_ids, draft_probs = self._sample_draft_tokens(
+                    last_hidden_states[:batch_size],
+                    sampling_metadata,
+                    spec_step_idx=spec_step_idx,
+                )
+                self._step_profiler.finish(
+                    profile_events, f"loop{token_index}_sample", loop_sample_start
+                )
+                if draft_probs is not None:
+                    assert draft_probs_list is not None
+                    draft_probs_list.append(draft_probs)
+                draft_token_ids_list.append(draft_token_ids)
+        finally:
+            if share_mtp_topk_indices:
+                self.model.model.set_skip_topk(False)
 
         # [batch_size, num_speculative_tokens]
         draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
@@ -2158,6 +2199,17 @@ class SpecDecodeBaseProposer:
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
 
         if (
             self.parallel_drafting
@@ -2391,16 +2443,17 @@ class SpecDecodeBaseProposer:
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
```

---

### 5.26 `vllm/v1/worker/dp_utils.py` — DP2 × TP4 独立副本保留各自 cudagraph_mode 与 num_tokens_padded（D26）

```diff
diff --git a/vllm/v1/worker/dp_utils.py b/vllm/v1/worker/dp_utils.py
index e7c6d81..0ecc574 100644
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
```

---

### 5.27 `vllm/v1/worker/gpu/cudagraph_utils.py` — 仅对 MLA 模型准备 dcp_local_seq_lens（D12）

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

### 5.28 `vllm/v1/worker/gpu/model_runner.py` — 仅对 MLA 模型准备 dcp_local_seq_lens（D12）

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

### 5.29 `vllm/v1/worker/gpu_model_runner.py` — Chunked Prefill MTP 提前退出配合（P7）、跳过非 MLA dcp_local_seq_lens（D12）与 DP2×TP4 禁用跨 DP dummy 死循环（D26）

```diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 001204b..9c66cac 100644
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
@@ -8234,7 +8234,7 @@ class GPUModelRunner(
         request (e.g., for chunked prefill requests that are not the last
         prefill chunk)."""
         num_reqs = self.input_batch.num_reqs
-        return bool(self.discard_request_mask.np[:num_reqs].all())
+        return num_reqs > 0 and bool(self.discard_request_mask.np[:num_reqs].all())
 
     @torch.inference_mode()
     def execute_model(
@@ -9099,7 +9099,10 @@ class GPUModelRunner(
                 self._step_profiler.add_cpu_context(
                     mtp_profile_ctx, "draft_wall_cpu", mtp_draft_wall_start
                 )
-                self._copy_draft_token_ids_to_cpu(scheduler_output)
+                self._copy_draft_token_ids_to_cpu(
+                    scheduler_output,
+                    zeros_only=self._is_all_reqs_chunked_prefill(),
+                )
                 if trace_log:
                     trace_draft_ms += (time.perf_counter() - trace_draft_t0) * 1000.0
 
@@ -9169,6 +9172,23 @@ class GPUModelRunner(
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
```

---

### 5.30 `tests/models/qwen4_exp/test_e4m3_mtp.py` — 单测：验证 _make_draft_vllm_config 正确应用 speculative_config.kv_cache_dtype（D27）

```diff
diff --git a/tests/models/qwen4_exp/test_e4m3_mtp.py b/tests/models/qwen4_exp/test_e4m3_mtp.py
index 5d341e8..7a0daf5 100644
--- a/tests/models/qwen4_exp/test_e4m3_mtp.py
+++ b/tests/models/qwen4_exp/test_e4m3_mtp.py
@@ -247,3 +247,40 @@ def test_automatic_e4m3_refuses_unit_scale_fallback(monkeypatch):
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
+
```

---

### 5.31 `tests/models/qwen4_exp/test_weight_loading.py` — 单测：验证 Qwen3NextForCausalLM 在 DP2×TP4 下保持副本内局部 MoE TP=4（D26）

```diff
diff --git a/tests/models/qwen4_exp/test_weight_loading.py b/tests/models/qwen4_exp/test_weight_loading.py
index f8eb0ea..be6fcee 100644
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
+
```

---

## 6. 一键导出或应用全量补丁命令

如果您需要将本分支的全部 40 项优化与修复（`P1`–`P13` + `D1`–`D27`）一次性导出为单个 `.patch` 文件并应用到任何干净的 `main` 分支，可直接运行：

```bash
# 在当前分支导出仅包含源码与单测的完整补丁
git diff be3722bc095b9792bf9833068595f0b1ef70ad37 HEAD -- csrc/ vllm/ tests/ > sm70_qwen38_all_optimizations.patch

# 在目标机器的 main 分支上一键应用
git apply --check sm70_qwen38_all_optimizations.patch && git apply sm70_qwen38_all_optimizations.patch
```
