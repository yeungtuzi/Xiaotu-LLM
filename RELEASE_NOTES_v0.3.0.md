# v0.3.0 —— 辅助 GPU（挎斗）：提升多并发解码性能

> 主题：**新增辅助 GPU（挎斗）功能，把闲置的 GPU0 用来分担 MoE 热专家，
> 提升多并发（C≥2）解码性能**。
> 挎斗**默认关闭**；不设置 `XTU_SIDECAR` 时行为与 v0.2.9 完全一致。

---

## 一、新增能力

### 1. 辅助 GPU（挎斗）`XTU_SIDECAR=gpu0`

- 主服务仍占 **GPU1+GPU2（TP=2）**，第三张卡（物理 GPU0）作为**挎斗**：
  常驻 200 / 1200 / 1900 个热专家（`expert_pin_sidecar_*.bin`），
  与 CPU 引擎、GPU1/2 上的热专家**并行**计算；
- 启动方式：`GPUS=1,2,0`（逻辑 0/1 = 物理 1/2，逻辑 2 = 物理 0），
  并设置 `XTU_SIDECAR=gpu0`、`XTU_SIDECAR_TABLE=<表路径>`；
- 已支持 ds41f（DeepSeek-V4.1-Flash 1017，FP8 稠密 + FP4 专家）；
- **默认关闭**：未设置 `XTU_SIDECAR` 或表不存在 ⇒ 自动回退标准路径，**零行为变化**。

### 2. 挎斗内核缓冲复用

`sidecar_nmajor` 内核现在支持外部预分配缓冲（`bufs=`），
每层省去 4 次 `torch.empty`，降低分配抖动。

---

## 二、实测数据（ds41f 1017，15K，统一官方口径）

口径：`docs/BENCH_PROTOCOL.md` §6（`dsh_15k_v2.jsonl`、`max_tokens=1024` 自然结束、
每档 16 条、每档 2×C 预热）+ `bench_decode_window.py --tokenizer`（按 token 计数）。
配置：`TP=2 · util 0.70 · MAXLEN 524288 · MBT 8192 · seqs 4 · DSpark k=5 · PIN off`。

| 并发 | 无挎斗 decode | 挎斗 1900 槽 decode | 无挎斗步时 | 挎斗步时 |
|---|---|---|---|---|
| 1 | 68.6 | **72.5** | 82.2 ms | **66.8 ms** |
| 2 | 70.8 | **106.8** | 115.8 ms | **85.4 ms** |
| 4 | 79.6 | **110.2** | 197.5 ms | **119.4 ms** |

- 步时三档均下降（82.2→66.8 / 115.8→85.4 / 197.4→119.4 ms）；
- **tok/s 随 DSpark 接受率浮动**，表中取热机窗口口径；
- prefill ≈1,250 tok/s（冷启动 TTFT 折算），与无挎斗 1,190 同量级。

---

## 三、适用与不适用

| 场景 | 结论 |
|---|---|
| 多并发 C≥2，CPU MoE 是瓶颈，且有空闲 GPU0 | ✅ 推荐（步时显著下降） |
| C=1 单流 | ⚠️ 中性，收益取决于每层固定开销 |
| qfn（Qwen3.8-Flash-Next） | ❌ 不建议：每层跨设备开销 ~8.6ms，净负（见 `docs/QFN_NVFP4_SIDECAR_A_PLAN.md`） |
| 显存 | 1900 槽约占 GPU0 **36 GB**；1200/200 槽可按需选择 |

---

## 四、已知问题

1. **挎斗会降低 DSpark 接受率**（实测 83% → 65% → 40%），
   因此 tok/s 的提升不稳定；步时收益真实，但需要数值对齐后才能稳定转化为吞吐；
2. 挎斗要求 `GPUS` 显式包含物理 GPU0（`GPUS=1,2,0`），否则初始化会回退标准路径并打印原因；
3. 挎斗与 PIN（`XIAOTU_PIN`）**不要同时开**，两者都会占用 GPU1/2 显存。

---

## 五、口径与文档

- 基准口径：[`docs/BENCH_PROTOCOL.md`](docs/BENCH_PROTOCOL.md) §6 / §6.1 / §6.2
- 挎斗复盘（目标、设计、实现、测试、分析、建议）：
  [`docs/SIDECAR_GPU0_POSTMORTEM.md`](docs/SIDECAR_GPU0_POSTMORTEM.md)
- 挎斗计划与逐日记录：
  [`docs/AUXILIARY_GPU_SIDECAR_PLAN_2026-10-22.md`](docs/AUXILIARY_GPU_SIDECAR_PLAN_2026-10-22.md)
- qfn 挎斗 A 方案（负结果）：
  [`docs/QFN_NVFP4_SIDECAR_A_PLAN.md`](docs/QFN_NVFP4_SIDECAR_A_PLAN.md)

---

## 六、升级

```bash
pip install -U vllm-xtu-moe==0.3.0
```

- 默认关闭，**升级本身不改变任何行为**；
- 启用挎斗：`GPUS=1,2,0 XTU_SIDECAR=gpu0 XTU_SIDECAR_TABLE=<table.bin> bash scripts/serve_v41.sh`
- 回滚：去掉 `XTU_SIDECAR*` 即可回到 v0.2.9 行为。

---

## 七、可变 K（DSpark adaptive verification）——已实现并评估，**暂时暂停**

本次一并完成了 DSpark「可变长度验证」的实现与评估，结论是**本机收益不大，暂停**：

- **固定 k 扫描**（15K，官方口径）：C=1 最优 k=5（68.6 tok/s），C=4 最优 k=3（83.7 tok/s）⇒ 最优 k 随并发变化；
- **完整 adaptive verification**：A100 上 `DeepseekV41IndexerBackend` 不支持设备侧裁剪，需移植 ROCm device-lens 路径，**评估为不划算**；
- **置信度驱动 per-request 变长**：已实现且与固定 k=5 **逐 token 一致**，但各请求置信度曲线同质、verify 固定成本不随 k 缩短 ⇒ 无差异化收益；
- **内建 batch-size 动态 k**（零代码，`num_speculative_tokens_per_batch_size = [[1,2,5],[3,4,3]]`）：
  C=1 69.3 / C=2 74.1~75.3 / C=4 **83.2**（固定 k=5 为 68.6 / 70.8 / 79.6，C=4 **+4.5%**）。

**处置**：暂停进一步投入；**代码默认关**（不设 `XTU_VARLEN_K` 即原行为）；
重新启用条件 = **硬件变化（sm90+/sm100）**或**其他挖潜手段穷尽**后再评估。
实现细节、数据与门限见 [`docs/DYNAMIC_K_NOTES.md`](docs/DYNAMIC_K_NOTES.md)。

---

## 八、已知问题（2026-10-24 发现，重要）

**挎斗目前只在单请求（qlen=1）下输出正确**；batch>1 时会出现空回复/截断/内容异常。
实测（temp=0，SPEC=0，相同提示词并发对比）：

| 配置 | 单请求 | batch=4 |
|---|---|---|
| 无挎斗 | 119 字 | 121 / 129 / 125 / 119（均完整）|
| 有挎斗 | 118 字 | **0 / 153 / 0 / 33** |

因此本页 §一 中 C≥2 的挎斗吞吐数据（106.8 / 110.2 tok/s）**是在输出已损坏的前提下测得的**，
不能作为生产特性；C≥2 请勿启用挎斗。另：挎斗与 DSpark 投机解码叠加时质量同样不可接受。
定位与复现见 [`docs/QUALITY_SIDECAR_SPEC_CONFLICT.md`](docs/QUALITY_SIDECAR_SPEC_CONFLICT.md)。
