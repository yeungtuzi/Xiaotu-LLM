# 基准留档：Qwen3.8-Flash-Next-MXFP4-FP8-MTP-MXFP4 @ MUSA S5000 —— 2026-11-17

> 口径来自 `dev-docs/BENCH_ARCHIVE_TEMPLATE.md`：
> **先预热（同实例连发 3 次、丢第 1 次）→ 每格 3 次取中位 → 每请求带 salt 避开前缀缓存 → 同协议同顺序。**
> 本文所有数字均为**新提示（salt 后无前缀缓存命中）**，预热已做过。

## 1. 配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3.8-Flash-Next-MXFP4-FP8-MTP-MXFP4（`lovedheart/…`，231 文件 / 125,072,914,749 B，**已按 ModelScope 清单逐文件核对 100%**）|
| 引擎 / 提交 | `lovedheart/sglang` 分支 `qsa-on-main`，HEAD `799ed33` + **11 个 MUSA 移植补丁 P1–P11**（见主报告 §2–§6）|
| 节点 / 容器 | S5000 `worker32093`，容器 `sgltest`（`--runtime=mthreads --ipc=host --network=host`）|
| GPU | 2× MUSA S5000（`MUSA_VISIBLE_DEVICES=1,2`）|
| TP / PP | **2** / 1 |
| 量化 | `modelopt MIXED_PRECISION`：`W4A8_MXFP4_FP8`（49 层 experts）+ `FP8_PB_WO`（168 层线性）|
| MoE 后端 | `--moe-runner-backend triton` + **P4 打包权重直读（e2m1 LUT + e8m0 位转 float）** |
| KV / 状态 | `kv_cache_dtype=auto`；mamba state 102 槽（ssm 5.43 GB + conv 0.11 GB）|
| 图模式 | **`--disable-cuda-graph`**（MUSA 图捕获与 eager 兜底不兼容，见主报告 §7.2）|
| 上下文 | `context_len=262144`，`max_total_num_tokens=506,816` |
| 并发上限 | `--max-running-requests 16` |
| 其他 | `--language-only`（vision 塔依赖 CUDA FA），`--mem-fraction-static 0.90` |
| 两臂 | `nomtp` = 基线；`mtp` = `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |

## 2. 性能（提示 1370 token，输出 128 token，贪心）

| C | 臂 | tok/s（聚合）| 步时 TPOT 中位 | TTFT 中位 | prefill tok/s |
|---|---|---|---|---|---|
| 1 | nomtp | **5.75** | 170.5 ms | **423.3 ms** | **3239** |
| 2 | nomtp | **10.94** | 178.4 ms | 554.5 ms | 2131 |
| 4 | nomtp | **20.44** | 178.0 ms | 2254.4 ms | 608 |
| 1 | mtp | 4.55 | 204.8 ms | 511.6 ms | 2682 |
| 2 | mtp | 8.70 | 209.3 ms | 597.8 ms | 1954 |
| 4 | mtp | 11.85 | 209.1 ms | 4611.1 ms | 424 |

- 每格 = 3 次独立 cell 的中位；`raw` 三次数据见原始日志 `JSON` 行；
- **结论：当前（无图 + eager 兜底）配置下 MTP 是负收益**——步时从 170 ms 涨到 205 ms，聚合 tok/s 全面低于基线。
  原因：draft 层的每步 host 开销没有图捕获摊薄，而贪心合成提示的接受率不足以抵消。

## 3. 正确性

| 项 | 结果 |
|---|---|
| 语义抽检（5 问，greedy）| **4/5 字符串匹配** —— 其中"水的化学式"模型答 `H₂O`（下标写法），匹配串写的是 `H2O` ⇒ **实为 5/5** |
| 输出通顺性 | 中文连贯、无乱码（`中国的首都是` → `北京，北京位于华北平原的北部…`）|
| greedy 确定性（同 prompt 两次逐 token 一致）| 未测（本轮只做单口径性能）|
| needle（长文捞针）| 未测 |

## 4. 资源

| 项 | 值 |
|---|---|
| 权重/卡 | 打包 MXFP4 直读，2 卡共 ~117 GiB（约 58.5 GiB/卡）|
| KV 池 | `max_total_num_tokens=506,816` |
| mamba state | 102 槽 × 55.05 MB ≈ 5.54 GB/卡 |
| 启动后可用显存 | 7.90 GB/卡 |
| 冷启动 → ready | ~90 s（23:36:14 启动，23:37:41 ready）|

## 5. 原始数据

- 启动日志：`/home/mccuseradmin/s5000/probe/sgl_real.log`（nomtp）、`sgl_real_mtp.log`（mtp）
- 基准日志：`/home/mccuseradmin/s5000/probe/proto_nomtp.log`、`proto_mtp.log`（后者被后续误覆盖，MTP 数字以主报告 §10 记录为准）
- 基准脚本：`/home/mccuseradmin/s5000/probe/bench_protocol.py`（本地源 `notes/bench_protocol.py`）
- 权重校验：`/home/mccuseradmin/s5000/verify_model.log` → `SUMMARY ok=231 bad=0 … (100.0%)`

---

## 6. 勘误（2026-11-18 00:10）：§2 的 TTFT / prefill 列作废，以本节为准

**原因**：§2 所用脚本把 salt 加在提示**末尾**，而长提示主体是重复文本 ⇒ **前缀缓存仍然命中**，
TTFT 与 prefill 被严重低估。decode / TPOT / 聚合 tok/s **不受影响，仍然有效**。

### 6.1 真·fresh（unique 前缀）长短 prefill

| 档 | prompt_tokens | TTFT 中位 | decode 中位 | TPOT 中位 |
|---|---|---|---|---|
| 短 | ~40 | **441.3 ms** | **5.72 tok/s** | 171.2 ms |
| 长 | ~8000 | **4379.3 ms** | **4.84 tok/s** | 171.6 ms |

warmup=2、每档 3 次取中位、`salt` 在提示开头、radix cache 开启（miss）。

### 6.2 prefill 吞吐（非流式 e2e 斜率，出 1 token）

| prompt_tokens | e2e 中位 |
|---|---|
| 40 | 235.3 ms |
| 1621 | 789.7 ms |
| 8023 | 6009.1 ms |
| 16025 | 10487.6 ms |
| 32024 | 19526.5 ms |

⇒ 稳态 **prefill ≈ 1.5–1.8K tok/s**；固定开销 ≈ 235 ms。
