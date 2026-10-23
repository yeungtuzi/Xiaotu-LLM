# v0.3.0 —— 辅助 GPU（侧车）：提升多并发解码性能

> 主题：**新增辅助 GPU（侧车）功能，把闲置的 GPU0 用来分担 MoE 热专家，
> 提升多并发（C≥2）解码性能**。
> 侧车**默认关闭**；不设置 `XTU_SIDECAR` 时行为与 v0.2.9 完全一致。

---

## 一、新增能力

### 1. 辅助 GPU（侧车）`XTU_SIDECAR=gpu0`

- 主服务仍占 **GPU1+GPU2（TP=2）**，第三张卡（物理 GPU0）作为**侧车**：
  常驻 200 / 1200 / 1900 个热专家（`expert_pin_sidecar_*.bin`），
  与 CPU 引擎、GPU1/2 上的热专家**并行**计算；
- 启动方式：`GPUS=1,2,0`（逻辑 0/1 = 物理 1/2，逻辑 2 = 物理 0），
  并设置 `XTU_SIDECAR=gpu0`、`XTU_SIDECAR_TABLE=<表路径>`；
- 已支持 ds41f（DeepSeek-V4.1-Flash 1017，FP8 稠密 + FP4 专家）；
- **默认关闭**：未设置 `XTU_SIDECAR` 或表不存在 ⇒ 自动回退标准路径，**零行为变化**。

### 2. 侧车内核缓冲复用

`sidecar_nmajor` 内核现在支持外部预分配缓冲（`bufs=`），
每层省去 4 次 `torch.empty`，降低分配抖动。

---

## 二、实测数据（ds41f 1017，15K，统一官方口径）

口径：`docs/BENCH_PROTOCOL.md` §6（`dsh_15k_v2.jsonl`、`max_tokens=1024` 自然结束、
每档 16 条、每档 2×C 预热）+ `bench_decode_window.py --tokenizer`（按 token 计数）。
配置：`TP=2 · util 0.70 · MAXLEN 524288 · MBT 8192 · seqs 4 · DSpark k=5 · PIN off`。

| 并发 | 无侧车 decode | 侧车 1900 槽 decode | 无侧车步时 | 侧车步时 |
|---|---|---|---|---|
| 1 | 68.6 | 43.1（fixed 口径）/ 样本不足（窗口） | 82.2 ms | 63.0 ms |
| 2 | 70.8 | 76.8 | 115.8 ms | **80.2 ms** |
| 4 | 79.6 | **106.8 / 108.5 / 81.1**（三次） | 197.5 ms | **115.5 ms** |

- **C=4 步时 −41%（115.5ms vs 197.5ms，三次一致）** —— 这是稳定的引擎收益；
- **tok/s 随 DSpark 接受率浮动**（三次 106.8 / 108.5 / 81.1），不作为稳定收益宣传；
- C=1 中性（fixed 口径 43.1 vs 43.7）。

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

1. **侧车会降低 DSpark 接受率**（实测 83% → 65% → 40%），
   因此 tok/s 的提升不稳定；步时收益真实，但需要数值对齐后才能稳定转化为吞吐；
2. 侧车要求 `GPUS` 显式包含物理 GPU0（`GPUS=1,2,0`），否则初始化会回退标准路径并打印原因；
3. 侧车与 PIN（`XIAOTU_PIN`）**不要同时开**，两者都会占用 GPU1/2 显存。

---

## 五、口径与文档

- 基准口径：[`docs/BENCH_PROTOCOL.md`](docs/BENCH_PROTOCOL.md) §6 / §6.1 / §6.2
- 侧车复盘（目标、设计、实现、测试、分析、建议）：
  [`docs/SIDECAR_GPU0_POSTMORTEM.md`](docs/SIDECAR_GPU0_POSTMORTEM.md)
- 侧车计划与逐日记录：
  [`docs/AUXILIARY_GPU_SIDECAR_PLAN_2026-10-22.md`](docs/AUXILIARY_GPU_SIDECAR_PLAN_2026-10-22.md)
- qfn 侧车 A 方案（负结果）：
  [`docs/QFN_NVFP4_SIDECAR_A_PLAN.md`](docs/QFN_NVFP4_SIDECAR_A_PLAN.md)

---

## 六、升级

```bash
pip install -U vllm-xtu-moe==0.3.0
```

- 默认关闭，**升级本身不改变任何行为**；
- 启用侧车：`GPUS=1,2,0 XTU_SIDECAR=gpu0 XTU_SIDECAR_TABLE=<table.bin> bash scripts/serve_v41.sh`
- 回滚：去掉 `XTU_SIDECAR*` 即可回到 v0.2.9 行为。
