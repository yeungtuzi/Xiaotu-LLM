# 可变 K（DSpark adaptive verification）评估记录

> 结论先行：**本机（3×A100-PCIE-40GB，SM80）收益不大，暂时暂停**。
> 代码与文档均已保留；该能力**默认关闭**（不设 `XTU_VARLEN_K` 即为原行为）。
> 重新启用条件：**硬件变化（sm90+/sm100）**，或**其他挖潜手段穷尽**后再评估。

## 1. 背景

- 1017 checkpoint 自带 `mtp.*.confidence_head`；
- vLLM 上游已有 `enable_adaptive_verification`（confidence 调度验证）；
- 目标：按每请求每位置置信度决定验证长度，替代固定 k=5，降低 verify 的 CPU 专家计算。

## 2. 固定 k 扫描（ds41f，15K，官方口径，热机）

| k | C=1 tok/s | C=1 步时 | C=4 tok/s | C=4 步时 |
|---|---|---|---|---|
| 5 | **68.6** | 82.2 ms | 79.6 | 197.4 ms |
| 4 | 61.9 | 77.5 ms | 79.1 | 174.1 ms |
| 3 | 54.9 | 70.6 ms | **83.7** | 149.9 ms |

⇒ 最优 k 随并发变化：C=1 要 k=5，C=4 要 k=3。

## 3. 实现与实测

1. **完整 adaptive verification（设备侧裁剪）**：A100 上
   `DeepseekV41IndexerBackend` 不支持，启动即报错；补丁能启动但首请求
   CUDA illegal access ⇒ 需移植 ROCm 的 device-lens 路径，评估为不划算。
2. **置信度驱动 per-request 变长（CPU 口径）**：已实现，8/8 探针与固定 k=5
   **逐 token 完全一致**；但各请求置信度曲线同质（生存率
   `[0.511, 0.323, 0.185, 0.110, 0.081]`），top-B 预算退化为统一 k；
   且 verify 的固定成本不随 k 缩短 ⇒ 无差异化收益。
3. **内建 batch-size 动态 k**（零代码，`num_speculative_tokens_per_batch_size`）：
   `[[1,2,5],[3,4,3]]` →

| 档 | 动态 k | 固定 k=5 |
|---|---|---|
| C=1 | 69.3 | 68.6 |
| C=2 | 74.1 / 75.3 | 70.8 |
| C=4 | **83.2** | 79.6 |

## 4. 结论与门限

- 收益上限：**C=4 约 +4~5%**，低于挎斗在 C≥2 的步时收益（−41%）；
- **暂停**进一步投入（不移植 device-lens、不做 confidence 校准）；
- 代码默认关：不设 `XTU_VARLEN_K` ⇒ 行为与 v0.3.0 原版一致；
- 重新启用条件：硬件变化（sm90+/sm100）或其他优化穷尽后再评估。
