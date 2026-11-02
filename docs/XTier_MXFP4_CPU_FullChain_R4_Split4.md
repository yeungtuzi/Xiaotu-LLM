# XTier MXFP4 CPU Full-Chain R=4 / split4：实验全过程与结论

**状态：未采纳（opt-in，默认 off，非生产）。**

**归档 tag：**

```text
archive/xtier-mxfp4-cpu-fullchain-r4-split4
```

正文中的数字来自单层 A/B 与真实 ds41f vLLM 测试实例；生产未重启、未改默认。

---

## 1. 背景

目标是在真实 ds41f 的 CPU 专家解码上，验证一条“行主序 MXFP4 全链 bf16”路径能否
替代现有 fp32_old：

- 权重：MXFP4（E8M0 scale，group K=32）；
- 计算：`vpmultishiftqb + ternarylogic + vpermt2b` 解码，scaled bf16 table，`vdpbf16ps` 累加；
- 输出：一次处理 R=4 个输出行，行内 hsum，bf16 下转；
- C=8：`split4`（两组 M=4，权重解码两次）。

开关全部 opt-in：

```text
XIAOTU_MOE_DPBF16=1
XIAOTU_MOE_XTIER_CHAIN=1
XIAOTU_MOE_XTIER_SPLIT4=1   # 仅 C=8 需要
```

默认关闭，不改变 fp32_old。

---

## 2. 单层门禁

硬件：2×A100-PCIE-40GB、2×EPYC 9654；CPU 侧单核 `taskset -c 0`。
真实 ds41f 权重，`H=5120 I=2304 E=384 topk=6`。

### C=4

| arm | ms/layer（3 次） | 均值 |
|---|---:|---:|
| fp32_old | 26.62 / 27.92 / 26.49 | 27.01 |
| xtier R=4 | 17.81 / 17.69 / 17.55 | **17.68** |

每 token 成本：fp32 6.75 ms，xtier 4.42 ms；**PASS**。

数值：`max_abs=2.56e-05`，`mean_rel=9.15e-07`，`max_rel=0.294`。

### C=8

| arm | ms/layer（3 次） | 均值 |
|---|---:|---:|
| fp32_old | 51.30 / 48.36 / 51.15 | 50.27 |
| xtier split4 | 29.95 / 29.76 / 29.92 | **29.88** |

每 token 成本：fp32 6.28 ms，xtier 3.74 ms；**PASS**。

数值：`max_abs=2.75e-05`，`mean_rel=4.41e-07`，`max_rel=0.046`。

### C=1（真实 layer9，单核）

| arm | ms/layer |
|---|---:|
| fp32_old | 4.71 |
| xtier R=4 | **3.46**（−26.5%） |

结论：bf16 全链在 C=1 下也更快，不是“只有 C=4 才快”。

---

## 3. 真实服务接入

测试实例：

```bash
bash scripts/serve_ds41.sh start TAG=e2e_xtier PORT=8701 PC=0 WARMUP=1 THREADS=60
```

两个接入问题：

1. `NO_CUDA=1` 构建的 `.so` 缺服务 decode 所需的 `cpu_decode`，必须用默认 CUDA 构建；
2. vLLM EngineCore/Worker 会丢 `XIAOTU_*`，通过项目既有 `/tmp/xiaotu_env` 文件桥补齐。

服务日志确认：

```text
[vllm-xtu-moe/env] XIAOTU_MOE_XTIER_CHAIN=1
[xtier] full-chain branch taken
```

即 CPU 路径确实命中。

---

## 4. 端到端结果

`scripts/e2e_token_probe.py`，同 nonce、贪心，三个短 prompt + 5.2k 长上下文。

### 正确性

- 所有 case 文本完全一致；
- 注意：ds41f 的 `message.content` 多为空（内容在 reasoning），文本长度不能当质量判据。

### 热态延迟（秒）

| case | baseline fp32 | xtier | 变化 |
|---|---:|---:|---:|
| short1 | 1.371 | 1.532 | +12% |
| short2 | 1.309 | 1.388 | +6% |
| short3 | 1.451 | 1.427 | −2% |
| long_context | 1.036 | 1.178 | +14% |

首次请求另有冷启动：xtier 33–37 s vs baseline 6.6 s。

### 结论

- 接入/数值/输出一致性：**PASS**；
- 端到端吞吐：**未获得收益**，热态约慢 **0–14%**；
- 单层 C=4/C=8 的收益没有转化为真实服务单请求解码收益。

---

## 5. 为什么没有端到端收益

不是 bf16 dot 慢；单层 C=1 仍快 26.5%。问题在：

1. **CPU 占比只有约 15%**  
   C=1 单层 4.71 ms × 43 ≈ 0.20 s；端到端约 1.31 s。  
   理论最多省 54 ms ≈ 4%，小于噪声/冷热波动。

2. **服务 TP=2 的 per-rank 形状**  
   单层门禁是 TP=1 全形状；服务每 rank 只算一半，固定开销（hsum、scale 表、解码、写回）占比变大。

3. **CPU↔GPU 交接**  
   xtier 端到端比基线慢约 0.1–0.2 s，说明除了 CPU 计算，还有 D2H/H2D、信箱、线程唤醒、同步等成本；单层 A/B 不包含这些。

4. **GPU/spec decode 主导**  
   端到端大部分时间是 GPU attention、draft、verify、采样；CPU 只占小头。

因此：**单层门禁需要通过，但不足以证明端到端收益。**

---

## 6. 决策与回滚

- **不接默认**；
- **不写生产**；
- 测试服务已停止，GPU 已释放；
- `/tmp/xiaotu_env` 已恢复，测试专用 `sitecustomize.py` 已删除；
- 所有 xtier 开关保持 opt-in、默认 off；
- 源码归档到 tag，patch 保存在：

```text
dev-docs/e3_scale_lut/xtier_r4/xtier_mxfp4_cpu_fullchain_r4_split4.patch
dev-docs/e3_scale_lut/xtier_r4/xtier_mxfp4_cpu_fullchain_r4_split4.patch.sha256
```

---

## 7. 如何找回

```bash
git show archive/xtier-mxfp4-cpu-fullchain-r4-split4

git checkout archive/xtier-mxfp4-cpu-fullchain-r4-split4 -- \
  xiaotu_moe/csrc/moe/moe_v2_packed4_base.inc scripts/bench_engine_ab.py
```

或：

```bash
git apply dev-docs/e3_scale_lut/xtier_r4/xtier_mxfp4_cpu_fullchain_r4_split4.patch
```

重新构建时注意用默认 CUDA 构建，不要用 `NO_CUDA=1`。

---

## 8. 一句话结论

> **XTier MXFP4 CPU full-chain 在单层 C=1/C=4/C=8 上确实比 fp32_old 快；但在真实 ds41f vLLM 服务中，CPU 只占约 15%，TP=2 分片与 CPU↔GPU 交接吃掉了收益，端到端没有提升。因此保留为 opt-in 研究线，不接默认。**
