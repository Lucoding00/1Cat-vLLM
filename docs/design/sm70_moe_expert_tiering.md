# SM70 MoE 专家分层（GPU 热专家缓存 + 主机内存全量专家 + CPU 计算缺失）

日期：2026-10-16
状态：Phase 0（标定工具）已提交；运行时实现未开始。

## 1. 背景与目标

NInfer（单张 RTX 5090）与 Strata（12–24GB 游戏卡）都用同一思路在小显存上运行
Qwen3.8-Flash-Next：

- 显存：attention/线性注意力、router、共享专家、LM head、KV，以及“最常用专家”缓存；
- 主机内存：全部专家（pinned 并注册给 CUDA）；
- SSD：只按行读取的大查找表（Flash-Next 的 PLE）；
- decode（小 M）：缓存命中的专家由 GPU 计算，缺失专家由 CPU 在内存中原地计算，二者重叠后合并；
- prefill（大 M）：把当前层缺失的专家流式拷到显存窗口，同时预取下一层，用大 chunk 摊薄 PCIe。

本文评估把这一机制集成到 1Cat 的 SM70 NVFP4 MoE 路径（首个目标 GLM-5.3-Flash-NVFP4，
8×V100-32GB，512GB 内存，4TB SSD），并给出分阶段方案。

## 2. 结论先行

可以集成，而且 1Cat 现有结构有几个很合适的挂点（第 5 节）。但要明确收益边界：

| 维度 | 影响 |
|---|---|
| 显存 | 主要收益。GLM 专家在每张卡上约 19.9 GiB（TP4/PP2 与 TP8 相同）。缓存 50% 热专家约释放 10 GiB/卡，KV/工作区从约 3–4 GiB 增至约 13 GiB/卡 |
| 更大的模型 | 1Cat 已支持架构的更大 MoE 变体，权重超过 256GB 显存但小于 512GB 内存时可以运行 |
| 更少的卡 | 例如 4 卡 + 内存跑 GLM-5.3-Flash，腾出另外 4 卡 |
| decode | 下降。单请求估计 53 → 35–45 tok/s，取决于真实命中率（见 Phase 0） |
| prefill 吞吐 | 基本持平；不能提升算力上限。chunk ≥ 8K 时 PCIe 流式可被计算掩盖，更大的 chunk 反而受益于腾出的显存 |
| SSD | GLM-5.3 没有 PLE 类查找表。SSD 适合做 KV 前缀缓存的第三层，或给超过 512GB 的模型溢出专家（NVMe 3–7 GB/s，decode 会很慢） |

prefill “几十 K tok/s” 仍受 8×V100 FP16 算力约束（理论上限约 2.9 万 tok/s，实际 5K–10K），
分层不会改变这一点。

## 3. GLM-5.3-Flash 的关键数字

- 288 个路由专家，top-8，hidden 4096，专家 intermediate 2048，42 个 MoE 层（3–44）。
- 单专家参数 3×4096×2048 = 25.2M；NVFP4（E2M1 + 每 16 个一个 E4M3 scale）约 0.5625 B/参数，
  即完整专家 14.2 MB，TP4 分片 3.54 MB，TP8 分片 1.77 MB。
- TP4/PP2（24,21 切分）下两个 stage 各 21 个 MoE 层：每卡专家 21×288×3.54 MB ≈ 19.9 GiB。
- 全部专家约 159 GiB；单份 pinned 主机副本放得下 512GB 内存，双份（两种布局）约 318 GiB 也可行。

一阶代价模型（由 `benchmarks/sm70_moe_expert_tiering.py` 实现，参数全部可调）：

- decode：每个 TP rank 在 CPU 上计算自己那份 intermediate 分片。一个 stage 每 token 读取
  `tp × Σ_layer misses × shard` 字节；再加每层一次交接开销（D2H 路由与 hidden、CPU 唤醒、H2D 部分和）。
- prefill：每张卡每个 chunk 拷入 `Σ_layer 未缓存且被触及的专家 × shard`。chunk ≥ 1K 时 288 个专家
  几乎全被触及，所以每个 chunk 的拷贝量约等于全部未缓存专家，与 chunk 大小无关，chunk 越大摊得越薄。
- 默认值：主机内存带宽 150 GB/s（双路 DDR4 实际可用），PCIe 3.0 每卡 6 GB/s（两卡共用一个 switch），
  交接 40 µs/层，基线 TPOT 18.86 ms。

合成 Zipf(0.9) 路由下的示例输出（仅演示工具，不代表真实分布）：

| 缓存比例 | 释放 GiB/卡 | decode 命中率 | 估计 decode tok/s | 每 2K chunk 拷贝 s | 拷贝约束的 prefill 上限 |
|---:|---:|---:|---:|---:|---:|
| 50% | 9.97 | 84.3% | 39.2 | 1.78 | 约 1,150 tok/s（2K chunk） |
| 70% | 5.95 | 91.8% | 43.2 | 1.07 | 约 1,920 tok/s（2K chunk） |

chunk 为 16K 时，同样的拷贝量对应的上限是上表的 8 倍，因此大 chunk 必不可少。

## 4. 运行时设计

### 4.1 存储分层

| 层 | 内容 |
|---|---|
| GPU 常驻 | 非专家权重；每层一个紧凑专家 bank `[C_l, ...]`（TurboMind 布局）放热专家；`slot_map[l][e]` 给出槽号或 -1 |
| GPU 窗口 | 大 M 路径的两个流式窗口（双缓冲）。每个窗口需容纳 `max_l(E - C_l)` 个分片 |
| 主机 pinned | 每个 rank 自己分片的全部专家。CPU 计算读原始 ModelOpt 布局；流式窗口读 TurboMind 布局副本（内存充足时两份都保留），或 H2D 原始布局后在 GPU 上调用 `nvfp4_sm70_prepare` 重新打包 |
| SSD | 可选：KV 前缀缓存第三层；或超大模型专家溢出 |

放置在启动时静态确定，按 Phase 0 的频率排序。规划方式有两种：每层相同数量（uniform），或全局按频率分配（global）。
自适应换入放到 Phase 4。

### 4.2 小 M 路径（decode、DFlash 验证窗口）

1. router 在 GPU 上算出 `topk_ids/topk_weights`。
2. 把 hidden `[M,4096]` 和路由复制到 pinned 缓冲区，并通知 CPU（见 4.4）。
3. GPU 用 `slot_map` 把 `topk_ids` 重映射为本地槽号，未缓存的权重置 0、跳过，只计算命中的专家；共享专家照常计算。
4. CPU 线程池（按 NUMA 绑在该 GPU 所在的 socket）对缺失的 (token, expert) 计算
   `Σ w_k · W2·SwiGLU(W13·x)` 的本 rank 分片，FP32 累加，写回 pinned 缓冲区。
5. GPU 等待 CPU 完成后，把部分和加到路由输出上；**加在 TP all-reduce 之前**。这样每个 rank
   只贡献自己的分片，all-reduce 的语义不变，PP 传递的负载也不变。

### 4.3 大 M 路径（prefill chunk）

当 `M × top_k` 足以触及大部分专家时（阈值由标定决定）：

1. 在上传流上把本层未缓存专家紧凑拷入窗口 A，同时把下一层拷入窗口 B。
2. dense 路径本来就用每专家指针表（`awq_moe_build_strided_ptrs`）。把命中专家指向缓存槽、
   缺失专家指向窗口，就能组成一个完整的 indexed bank，不需要 D2D 拷贝。
3. 计算流等待本层上传事件；层结束后窗口 A 被下下层复用。

mixed batch（decode + prefill 混在一步）按总 M 选择路径；也可以拆开分别执行。

### 4.4 CUDA Graph

现有 decode 是 full CUDA Graph。每层与 CPU 交接有两种做法：

- 先用 `vllm/compilation/breakable_cudagraph.py` 在每个 MoE 层断开图（Phase 1，简单；42 次断点，预计约 1–2 ms/token 开销）；
- 再改为图内可捕获的握手（Phase 3）：GPU kernel 向映射的 pinned 内存写序号并 `__threadfence_system()`；
  CPU 线程轮询序号后开始计算；GPU 侧在合并前用单 CTA 自旋 kernel 等待完成序号。这样不需要断图，
  开销接近一次 D2H + H2D 的延迟。

## 5. 1Cat 中的集成点

| 位置 | 用途 |
|---|---|
| `vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py::process_weights_after_loading` | 现在把 `[288,...]` 全部 repack 成 `w13_tm_weight/w2_tm_weight` 并删除 checkpoint 张量。改为：保留主机分片（pinned），GPU 只 repack 热专家到紧凑 bank，指针表和 QPN 元数据按槽号构建 |
| 同文件 `apply()` | 已有 `layer.expert_map` 分支（EP 用的“全局 ID → 本地 ID / -1”），dense 路径会跳过 -1。小 M 的 QPN/grouped 快速路径直接按全局 ID 访问 bank，需要先映射到槽号并给缺失项置零权重 |
| `vllm/model_executor/layers/fused_moe/sm70/fp4_codec.py`、`fp4_stages.py` | `Fp4MoECodec` 借用 layer 上的 bank 与指针表；在这里加入缓存槽和窗口指针 |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` 和 `--enable-return-routed-experts` | 上游已有的逐 token 路由导出，Phase 0 直接复用，不改运行时 |
| `vllm/model_executor/offloader/prefetch.py` | 现有的逐层整块预取卸载。它按参数整体搬运，不懂“热专家子集 + CPU 计算”，可以借鉴其静态缓冲和事件分叉，但不能直接复用 |
| `csrc/`（新增） | AVX-512/AVX2 NVFP4 专家 GEMV：E2M1 查表、E4M3 scale 转 FP32、FP32 累加；按 rank 绑核、按 NUMA 绑内存 |

## 6. 分阶段计划

| 阶段 | 内容 | 退出条件 |
|---|---|---|
| **0（已完成工具）** | `benchmarks/sm70_moe_expert_tiering.py`：采集真实路由，留出集评估静态放置的命中率、释放显存、估计 decode/prefill 代价 | 在你的真实业务负载上得到命中率曲线，决定是否继续以及缓存比例 |
| 1 | 静态热专家 bank + CPU 计算缺失专家（参考实现可以先用 PyTorch CPU，再换 AVX-512），eager / 断图；TP8 与 TP4/PP2 | 与全常驻相比 greedy 输出质量一致（参照现有 GSM8K 60/128 审计），测得的 decode 速度与代价模型偏差 < 30% |
| 2 | 大 M 流式窗口（双缓冲 + 下一层预取），prefill chunk 16K–32K | 8K–32K prefill 吞吐不低于全常驻的 90% |
| 3 | 图内握手、NUMA/绑核、原生 AVX-512 kernel、DFlash 验证窗口 | 端到端 decode 达到 Phase 0 估计值 |
| 4（可选） | 自适应缓存（按会话换入）、SSD 溢出、KV 前缀缓存第三层 | 视需要 |

## 7. 风险

- **质量**：CPU 用 FP32 累加，GPU 用 FP16 舍入，结果不会逐位一致。需要沿用本仓库的数据集审计（GSM8K、低 margin token 检查），尤其是 DFlash 的接受率。
- **命中率未知**：GLM 只有 288 个专家，每个专家较大，路由分布可能比 Flash-Next（512 个小专家）更平。Phase 0 必须在真实负载上测，合成数据不能代替。
- **PCIe 拓扑**：V100 平台是 PCIe 3.0，两卡共享一个 switch 时每卡约 6 GB/s；跨 socket 访问还要经过 UPI。需要用 `nvidia-smi topo -m` 确认拓扑。
- **pinned 内存**：注册约 160–320 GiB 需要调整 `ulimit -l`，启动时触页耗时可能达分钟级。
- **PP 与采集**：使用 PP 时，路由采集可能只覆盖部分 stage。工具会自动识别没有写入的层；建议用 TP8/PP1 做标定。

## 8. Phase 0 使用方法

```bash
# 1) 启动服务端：开启路由导出，关闭前缀缓存和投机解码（避免统计偏差）
vllm serve LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --tensor-parallel-size 8 --kv-cache-dtype fp8_e4m3 \
  --enable-return-routed-experts --no-enable-prefix-caching

# 2) 用有代表性的真实请求采集（JSONL，每行 {"messages": [...]} 或 {"prompt": "..."}）
python benchmarks/sm70_moe_expert_tiering.py collect \
  --model LibertAIDAI/GLM-5.3-Flash-NVFP4 --prompts my_workload.jsonl \
  --max-tokens 1024 --output glm_routes.npz

# 3) 分析：按 TP4/PP2 的分片大小和你机器的带宽假设估算
python benchmarks/sm70_moe_expert_tiering.py analyze glm_routes.npz \
  --moe-layers 3-44 --tp 4 --pp 2 --cpu-bw-gbs 150 --pcie-gbs-per-gpu 6 \
  --chunk-tokens 2048,8192,16384 --json-out tiering_report.json
```

工具会把请求切成标定集和留出评估集（避免在同一批数据上既排序又评估），对每个缓存比例输出：
实际缓存比例、每卡释放显存、decode 命中率、每 token 每层缺失专家数、估计增加的毫秒数和 decode tok/s，
以及每个 prefill chunk 的拷贝时间和拷贝约束下的 prefill 上限。
