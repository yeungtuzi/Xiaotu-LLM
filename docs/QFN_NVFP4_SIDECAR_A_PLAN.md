# qfn NVFP4 侧车 A 方案 —— 实施计划（2026-10-24）

> 目标：每层把 40–60% 专家放 GPU0（原始 NVFP4 布局），其余留在 GPU1/2 走 Marlin；
> 三卡同时计算；结果按路由权重并回；去掉 1GB CPU offload；GPU0 撤除回退纯 GPU1/2。
> 默认关（`QFN_SIDECAR_A=1` 才启用），可回滚。

## 1. 数据面

- GPU0（逻辑侧车卡）持有：每层一个**专家子集**的原始 NVFP4 权重
  `w13[S,2I,H/2]` U8 + `s13[S,2I,H/32]` E4M3、`w2[S,H,I/2]` + `s2[S,H,I/32]`
  （由已验证内核直接消费，**不能**用 Marlin 重排后的权重）；
- GPU1/2：其余专家，native Marlin（TP=2）；
- 每层耗时 = max(GPU0 子集, GPU1/2 子集) + 合并（PCIe ~26µs/层）。

## 2. 关键机制

1. **权重捕获**：在 `ModelOptNvFp4FusedMoE.process_weights_after_loading` 调用 native
   重排**之前**，把本 rank 的专家里属于 GPU0 子集的那些快照到 GPU0；
2. **路由排除**：让 Marlin 不计算 GPU0 子集的专家（`expert_map=-1` 或把 topk 权重置 0；
   前者省算力，后者只保证数值）；
3. **GPU0 计算**：`apply` 里把 hidden（每 rank 相同）+ topk ids/weights 发给 GPU0，
   用 `nvfp4_gemv` 只算子集命中专家，按路由权重求和；
4. **合并**：在 all-reduce **之前**把 GPU0 结果加到其中一个 rank 的 MoE 输出上
   （避免两 rank 重复加），再走原有 all-reduce；
5. **显存**：P2 阶段把 GPU0 子集的权重从 GPU1/2 **减掉**（缩小 num_experts 与权重张量），
   腾出 KV；P1 阶段先不动权重、只验证数值与并发。

## 3. 阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | TP=1 + 大 CPU offload，dump 单层 hidden/ids/weights/out；离线用内核复算 | 与 Marlin 输出相对误差 <1e-2 |
| P1 | 引擎内仅对 **1 层** 做 GPU0 子集 + Marlin 其余，权重仍全在 GPU1/2 | 同请求 greedy 与基线一致；单层耗时 |
| P2 | 48 层全量拆分；GPU1/2 权重缩到 60% | KV 从 0.41GB → ~15GB；C=1 速度下降 <15% |
| P3 | GPU0 撤除（`QFN_SIDECAR_A=0`）回退纯 GPU1/2；C=1/2/4 + needle 留档 | 两种形态 needle 一致 |

## 4. 风险

- TP=2 下 expert_map/路由的语义要逐层确认（vLLM 的 MoE TP 是否用 global ids）；
- all-reduce 前加 GPU0 结果只能加一次，需确认 rank 同步；
- Marlin 的 expert 数与权重形状绑定，P2 的缩减要改 create_weights/loading（最重）；
- 回滚：env 关闭 + 代码 revert；vLLM 补丁已有 `docs/patches/`。
