# MTP MicroBenchmark（唯一权威）

> ⛔ **2026-11-01 定稿**：**所有 MTP 性能数据以本文件为准**。
> 任何其他文档中与本文冲突的 MTP 数字（15.0/30.1 tok/s、+6.6%、1.3×、E=384 的 2.09 ms、
> CPU 871/593 µs、`moe4_smallm` 的“服务级无收益”等）**全部过时/不正确**，不得再引用。
> 旧文档只保留历史过程；其 MTP 性能结论一律失效。

## 1. 口径（必须先读）

| 项 | 值 |
|---|---|
| 模型 | DeepSeek-V4.1-Flash-1017 的 MTP（`mtp.0/1/2`） |
| 层数 | 3 |
| routed experts | **E=128**（`mtp.0.ffn.experts.0..127`；**不是 384**） |
| 形状 | `H=5120`, `I=2304`, `GK=32`, `topk=6` |
| 微基准 batch | `M=6`（decode 形状） |
| GPU 计算 | vLLM 上游 `fused_marlin_moe`（MXFP4 W4A16，A100 BF16 tensor core） |
| CPU 计算 | `MOE_MXFP4.cpu_prefill`（与 `cpu_decode` 同一 `forward_many`；不含 D2H/H2D，是 CPU **下界**） |
| 测量 | GPU 用 CUDA Event，CPU 用 `perf_counter`；30–40 次中位；GPU 先 burn |
| 生产 | 已按用户 2026-11-01 指令停止；测试只用 GPU0/1/2，未重启生产 |

⚠️ 这是 **MTP 专家层 microbench**，不含 attention / main_proj / Markov head，也不含服务级 KV/调度/接受率；
不能直接外推为整步 tok/s。3 层数字是按单层乘 3 的估算。

## 2. 单 GPU（GPU1，PCIe x16，真实 MTP 权重，E=128）

| 项 | 数值 |
|---|---:|
| GPU Marlin MXFP4 W4A16 | **588.8 µs/层**（min 580.6，p75 590.8） |
| CPU `cpu_prefill`（GPU1 同机） | **2759.3 µs/层**（min 1860.9） |
| CPU 历史两次（与 PCIe 无关） | 2379.9 / 2829.7 µs/层 |
| CPU/GPU | **4.04–4.81×**（保守按 CPU min 为 **3.18×**） |
| 3 层 | GPU **1.77 ms**；CPU **7.14–8.49 ms** |
| 数值 | GPU vs CPU abs p50 **2.36e-3**，归一化 p50 **4.02e-3**（bf16 级） |

**结论**：单卡全 E=128 时，Marlin 纯 GPU 比 CPU 快 **4–4.8×**。这里没有 TP all-reduce。
**CPU 计算不走 PCIe，x8/x16 与 CPU 数字无关**；GPU0 上测的 CPU 2379.9/2829.7 与 GPU1 上测的 2759.3 都是有效数据。
单卡 GPU 也没有跨卡通信，GPU1（x16）复测 588.8 µs 与 GPU0 一致；**x8 只作废 GPU0/1 的 TP=2 all-reduce，不作废任何 CPU 数据**。
早期 1.3× 来自 `moe4_smallm` 的 E=6 合成档，不是本口径。

## 3. TP=2（GPU1/2，生产同对卡，PCIe x16，真实分片）——按 rank 实测

> ⚠️ **2026-11-01 23:59 状态：以下 603.1/384.8/571.6 是 840/765 MHz 的 pre-boost 数据，不是最终口径**。  
> 两卡没有稳定在 ~1400 MHz；rank1 的 571.6 µs 可能是未 boost，不能据此判定 TP=2 比单卡慢。  
> 已准备 `scripts/bench_mtp_tp2_boost.py`（烧到 ≥1350 MHz、32 次、去 10% 极值、按 rank 记录 clock/active），  
> 等 GPU 执行器恢复后重跑；在拿到该结果前，**603.1 只作低 boost 参考，不作结论**。

每 rank 本地 **64** 个专家；Marlin 用 `expert_map` 映射全局 128；输出做 NCCL `all_reduce`。
以下不是“取一个 rank 的数”或估算，是 rank0/rank1 各自的 CUDA Event/wall-clock 中位（wall 含 host 提交）：

| 项 | rank0（GPU1） | rank1（GPU2） |
|---|---:|---:|
| SM clock 实测 | 840 MHz | 765 MHz |
| GPU Marlin 本地 wall | **384.8 µs** | **571.6 µs** |
| GPU Marlin 本地 event | 378.9 µs | 567.3 µs |
| NCCL all-reduce wall | **56.1 µs** | **57.1 µs** |
| NCCL all-reduce event | 54.3 µs | 55.3 µs |
| 本地 + AR wall（合并） | **603.1 µs** | **604.8 µs** |
| 本地 → AR 的 trace | local_host 350.2 µs / ar_host 68.0 µs / sync_tail 278.0 µs | local_host 352.3 µs / ar_host 75.1 µs / sync_tail 268.8 µs |

| 项 | 数值 |
|---|---:|
| 单卡 GPU（无 AR） | 588.8 µs |
| TP=2 GPU 合并（取慢 rank） | **603.1 µs** |
| TP=2 vs 单卡 | **慢 2.4%**（不是快 13%；旧的 511 µs 是**只打印了 rank0**，已作废） |
| GPU 侧 AR | **56 µs**（NCCL；count=30720 bf16） |
| rank0 等 rank1 的时间 | 603.1 − 384.8 = **218.3 µs** |
| rank1 本地→合并的额外 | 604.8 − 571.6 = **33.2 µs** |
| CPU 本地（每 rank 64，无 AR） | 701.6 µs |
| CPU 本地 + H2D + AR | ≈762.6 µs |
| GPU(603.1) / CPU(762.6) | **≈1.26×** |
| GPU(603.1) / 全 E=128 CPU(2759) | **≈4.58×** |
| 3 层 | GPU **1.81 ms**；CPU **≈2.29 ms** |

**能确认的事实**：
1. **TP=2 合并时间 603.1 µs，由 rank1 的 571.6 µs 本地计算 + ~33 µs AR 决定**；rank0 的 384.8 µs 早已算完，**等了 218.3 µs**。
2. **NCCL all-reduce 本身约 56 µs**（rank0 56.1、rank1 57.1），不是 216 µs；之前把“rank0 等 rank1”的 216 µs 误写成“AR 216 µs”，是错的。
3. **rank1 本地 571.6 vs rank0 384.8**，差 48%；当时 clock 840 vs 765（差 10%），**10% 的时钟差解释不了 48%**。  
   每个 rank 实际命中的 active 专家数/对数在这一次没有记录；balanced-routing 复测**没有完成**（卡住后终止）。  
   ⇒ **“为什么 rank1 慢”目前是未确认项，不能归因于 NCCL、PCIe 或 CPU。**
4. NCCL 后端已确认（`torch.distributed` backend=nccl，日志为 `AllReduce count=30720 datatype=9`），但**传输/ring/tree 细节没有抓到**；不能据此断言是 NCCL 的问题。

**旧的 511 µs / 1.49× 结论作废**：511 只是 rank0 的合并值，rank1 当时没打印；真实合并是 603.1 µs。

## 4. 旧数据（全部过时/不正确）

| 旧结论 | 出处 | 状态 |
|---|---|---|
| “GPU 常驻 draft 15.0 vs CPU 30.1 tok/s ⇒ GPU 更慢” | B840/B851/B873/B974 | **过时/错误，不合逻辑**；bug 路径产物 |
| ds41f 8.75 vs 9.33（+6.6%，n=4） | 2026-10-26 A/B | **过时**；旧 GPU 内核，不能代表 Marlin |
| `moe4_smallm` E=6 670 µs / E=384 2091 µs / 1.3× / 2.4× | B1177 | **过时/非 MTP**；E=384 是合成压力档，MTP E=128 |
| CPU 871 µs / 593 µs | B1096/B1110 | **过时**；是热池/小 M 口径，不是完整 MTP |
| “小 M 4-bit 内核服务级 −0.6%/+0.3%，不值得开” | `MOE4_SMALLM_STATUS.md` | **过时**；该文是 `moe4_smallm`，不是 Marlin MTP |
| “draft 落 CPU 是显存取舍 / GPU 常驻略快是噪声” | code comments / `BENCH_ds41f_2026-10-26.md` / `INCIDENT_2026-10-26_XID31.md` | **过时**；以本文第 2/3 节为准 |

> 以上旧数字保留在历史文档中只用于追溯，**不得再作为现状或判据**。

## 5. 复现

```bash
# 单 GPU（GPU0）
CUDA_VISIBLE_DEVICES=0 XIAOTU_MOE_THREADS=60 \
  python scripts/bench_mtp_marlin_vs_cpu.py

# 真实 TP=2（GPU1/2，生产同对卡）
CUDA_VISIBLE_DEVICES=1,2 XIAOTU_MOE_THREADS=60 \
  python scripts/bench_mtp_tp2_marlin_vs_cpu.py
```

脚本：
- `scripts/bench_mtp_marlin_vs_cpu.py`（单卡、真实 E=128 权重）
- `scripts/bench_mtp_tp2_marlin_vs_cpu.py`（TP=2、每 rank 64、NCCL all-reduce）

## 6. 边界

1. 只测 **MTP 专家层**；attention/dense/Markov 未包含，3 层是估算。
2. CPU 侧是 `cpu_prefill` 下界；生产 `cpu_decode` 含 D2H/H2D，CPU 侧只会更慢。
3. TP=2 的 CPU 侧按“每 rank 本地 64 专家 + AR”计算；若生产实际是每 rank 全 E=128，CPU 会更慢，
   GPU 倍率会更大——**以真实服务 A/B 为准**。
4. 未做服务级 A/B；未改生产 draft 路径；生产仍停止。

## 7. 2026-11-02 决定：GPU0 专职 MTP draft（target 仍 TP=2）

**架构决定**：GPU1/2 继续跑 target 的 TP=2；GPU0 不再做 target 热专家池，专职跑 MTP draft（TP=1）。
MTP 全部 3 层、E=128、7.388 GiB 放 GPU0；Marlin W4A16；独立 CUDA graph / workspace / KV / stream。

**数据流**：
1. target 每个 draft step 把 MTP 需要的 hidden states 从 rank0 送到 GPU0（TP 的 hidden 在 rank 间是复制的，
   不需要跨 rank 汇聚）；
2. GPU0 独立跑 MTP 3 层，产出 draft tokens / logits；
3. draft tokens 广播回 GPU1/2，target 继续 verify。

**为什么合适**：
- MTP 只有 3 层、qlen=6，独立图正好；
- 单卡 Marlin 588.8 µs/层，3 层约 1.77 ms；CPU draft 约 7.14–8.49 ms；
- target TP=2 的显存/KV/图完全不动；不再为 MTP 在 GPU1/2 留 3.7 GiB/rank（若曾常驻），
  当前 CPU 版则释放 6.7 GiB 主机内存和 CPU draft 计算；
- GPU0 x8 只影响每步 hidden/token 的小传输，不搬 7.4 GiB 权重。

**未决/必须验证**：
1. vLLM 当前 `_raise_if_draft_tp_mismatch` 强制 draft TP == target TP；要改成 draft TP=1 + target TP=2，
   或独立 draft worker；
2. draft KV 的归属和 graph 捕获边界（不能混进 target 图）；
3. hidden/token 跨卡交换的实际延迟（x8）；
4. TP=1 全专家 vs 当前 TP=2 分片的数值/接受率变化；
5. 服务级收益必须扣除交换和 target verify 变化，不能只用 1.77 ms 减 8.49 ms。

**状态**：设计已定；代码未接；未启动 A/B；生产仍停止。

## 8. 2026-11-02 60s post-boost 复测（推翻 603.1 的 pre-boost 口径）

GPU1/2，`scripts/bench_mtp_tp2_boost.py`，`DURATION_S=60 N=0 BOOST_MHZ=1350`；
每 rank 52518 个样本，去掉两端各 10% 后取均值（trim_mean），同时报中位。

| 项 | rank0（GPU1） | rank1（GPU2） |
|---|---:|---:|
| SM clock min/med/mean | 1245 / 1410 / **1409.9** | 1200 / 1410 / **1409.7** |
| local trim_mean | **366.9 µs** | **390.2 µs** |
| local median | 366.6 | 390.1 |
| AR trim_mean | **48.4 µs** | **48.0 µs** |
| both trim_mean | **412.5 µs** | **412.4 µs** |
| active pairs / experts | 18 / 15 | 18 / 17 |

**结论**：

- 两卡在采样期都稳定在 **~1410 MHz**；60s 内 **52518** 个样本，10% trim 后均值稳定。
- **TP=2 合并 ≈412.4 µs**；旧的 **603.1 µs 是 840/765 MHz 的 pre-boost 数据，作废**。
- rank1 本地从旧 **571.6 → 390.2 µs**，与 rank0 的差从 **48%** 降到 **6.3%**；
  rank1 命中 **17** 个专家 vs rank0 **15** 个，剩余差异与 active 专家数一致，**不再有 48% 的异常**。
- **NCCL AR 约 48 µs**，不是 216 µs；旧文“rank0 等 rank1 218 µs”是 pre-boost 的排队，不是 AR 成本。
- **注意**：单卡 GPU 588.8 µs 是 **1155 MHz** 口径，和本表 **1410 MHz** 不是同一时钟，不能直接算“TP=2 快 30%”；
  要同钟对比，需在 1410 MHz 重跑单卡。
- 这仍只是 **MTP 专家层**，不含 attention/dense/Markov head，不能直接外推整步 tok/s。

日志：`dev-docs/report/tuning/logs/bench_mtp_tp2_boost60_20261102.log`

## 9. 2026-11-02 单卡 60s 稳态（同 10% trim 口径）

`scripts/bench_mtp_single_boost.py`，GPU1，`DURATION_S=60 BOOST_MHZ=1350`，真实 E=128 全专家，
Marlin MXFP4 W4A16，`M=6, topk=6`，无 NCCL。采样期 **98200** 个样本，两端各去 10% 取均值。

| 项 | 单卡（GPU1） |
|---|---:|
| SM clock min/med/mean | 1110 / 1275 / **1270.9 MHz** |
| local median | 585.7 µs |
| local trim_mean | **585.5 µs** |
| local min / p75 / max | 572.4 / 587.8 / 1141.8 µs |

**与 TP=2 对比**：

- TP=2 合并 **412.4 µs @ ~1410 MHz**；
- 单卡 **585.5 µs @ ~1271 MHz**；
- 单卡完整 E=128 时 **251 W 触发功率墙**（`clocks_throttle_reasons=0x4`），到不了 1410；
  TP=2 每 rank 64 专家，单卡功耗约 185–209 W，能稳 1410。
- 因此 **412.4 vs 585.5 不是同钟对比**；按频率粗归一后 TP=2 仍约快 **15–20%**，但远小于“两倍 GPU 应有”的幅度。
- 3 层估算：TP=2 **≈1.24 ms**；GPU0 单卡 **≈1.76 ms**。

**对“GPU0 专职 draft”的含义**：

- GPU0 单卡 3 层 **~1.76 ms/step**，不是 0；它是否“近乎免费”取决于能否与 target verify **流水重叠**；
- 若串行，target 每步多等 **~1.76 ms**；
- 若 draft(n+1) 与 target verify(n) 并行，且 target verify >1.76 ms，则 draft 可被完全遮住；
- GPU0 的价值是 **target GPU1/2 不再承担 draft 的 SM/NCCL/显存**，不是“draft 不耗时间”。

## 附：2026-11-05 服务级实测补充（ds41f / k=5 / CED=1 / MAXSEQS=4）

> 本节补的是**服务级端到端**与**dspark 本体**两组数，与正文（微基准口径）不冲突。
> 正文说的"`moe4_smallm` 服务级无收益"在本轮得到了**直接解释**，见下。

| 臂 | 端到端 tok/s（890p·256out·N=7·先预热·掐头去尾） | dspark 本体 |
|---|---:|---|
| CPU 专家（默认） | 34.41 | ~2.6 ms/token（3 × 871 µs） |
| GPU 常驻（旧 Triton 路径） | 34.19 | **1.5 ms/token ≈ 650 tok/s**（eager + CUDA event 实测） |
| GPU 常驻 + 小 M 内核 | 33.80 | — |

三条臂差异 <0.5%（噪声内），接受率 2.7–3.0 未退化。

### 为什么"常驻更快却测不出"

- dspark 的 attention/norm/head **一直在 GPU**，常驻只改变其 3 层 **MoE 专家权重**的放置；
- 该步只占一步 **≈7%**（7.3 ms / 98.3 ms），省 ~1 ms/token ≈ **1%**，被 91 ms 的验证步淹没；
- 代价 +3.36 GiB/卡，并挤掉 KV 1.75 GiB ⇒ 默认仍为 0（落 CPU）。

### 为什么 `moe4_smallm` 在服务级看不到收益（直接原因，非"内核无效"）

1. 入口 `serve_ds41.sh:435` 原本把 `XIAOTU_PIN_CUDA_MOE` **写死为 0**（现已参数化为 `PIN_CUDA_MOE`，默认仍 0）；
2. 线上引擎 `.so` 里 `moe4_smallm_*` 符号数为 **0**（产物早于内核，运行时会打印
   `[xtu-moe4] 内核不在当前变体 .so 里（需要重编译引擎），已禁用`）；
3. 即便内核生效，它优化的是那 3 层 draft MoE（占一步 ~4%），上限也只有个位数百分比。

⇒ 结论：**内核本身有效（352 vs 1272 µs/层），但它在端到端上的天花板很低**；
要复现其收益必须先从本地源码重建引擎，再用同协议 A/B。
