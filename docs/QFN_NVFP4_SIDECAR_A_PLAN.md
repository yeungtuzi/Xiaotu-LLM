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

## 5. 2026-10-24 08:45 P0 状态

### 5.1 已完成
- vLLM 侧 dump 钩子（`docs/patches/qfn_dump_hooks_2026-10-24.patch`）：
  `ModelOptNvFp4FusedMoE.process_weights_after_loading` 开头 dump 原始权重；
- **layer 0 原始 NVFP4 权重已落盘**（1.4GB：w13/s13/w2/s2，512 专家）；
- 离线真实权重对拍已通过（`dev-docs/nvfp4_real_check.py`：0 误差）。

### 5.2 阻塞点
- `apply`（modelopt 686/1153）与 modular/monolithic 两条路径都加了 dump，
  但**运行时都没触发**（无 `[qfn-apply]`/`[qfn-dump] io` 日志）；
- backend 已确认是 **MARLIN**（`nvfp4.py:312`），但实际 MoE 执行路径不是
  `ModelOptNvFp4FusedMoE.apply` / `forward_modular`，需在 `FusedMoE.forward` 或
  runner 的 kernel dispatch 层再定位；
- vLLM 源码已从 `modelopt.py.bak_qfn` 恢复干净（探针不留在生产路径）。

### 5.3 下一步
1. 在 `RoutedExperts.forward_modular` 入口加探针，确认实际调用栈；
2. 找到后 dump 单层 `x/ids/weights/out`，完成引擎内对拍；
3. 再进入 P1（单层拆分）。

### 5.4 P0 **通过**（2026-10-24 09:00）

- 真实请求的 layer 0：`x/ids/weights/out` + 原始权重（512 专家）已 dump；
- 离线复算（含 `weight_scale_2` 全局缩放）与 Marlin 输出相对误差 **0.3–0.5%**（bf16 量级）；
- 结论：GPU0 内核对 qfn 的 NVFP4 专家计算在**数值上可直接替代** Marlin 的命中部分；
- 补丁归档：`docs/patches/qfn_dump_weights.patch`、`qfn_trace.patch`；
- 下一步 P1：单层交给 GPU0（全部命中），验证整模型 greedy 一致与耗时。

### 5.5 P1 机制定稿（2026-10-24 09:10）

**关键发现**：qfn 每层是 `shared_expert（bf16，I=640）+ 10 个路由专家（NVFP4）`，
且还有 `shared_expert_gate`（sigmoid 门控）。所以"整层替换"必须连 bf16 共享专家一起算，
不划算。

**改用"权重置零 + 并回"（与 P2 同一机制，单层先行）**：

1. pwal 时把目标层的**原始路由权重**（含 `w13/w2/scale/scale_2`）复制到 GPU0；
2. `forward_modular` 里对目标层：
   - 复制 `topk_weights`，把"归 GPU0 的专家"权重置 0；
   - 调用原生 `quant_method.apply`（Marlin 仍算全部 + 共享专家，被置零的贡献为 0）；
   - 用 GPU0 内核算这些被置零的命中，按**原始** `topk_weights` 加权；
   - 返回 `marlin_out + gpu0_out`（TP=1 直接在 rank 上相加；TP=2 需在 all-reduce 前只加一次）。
3. 共享专家完全不动 ⇒ 不需要复刻 bf16 MLP/sigmoid 门控。

**验收**：同请求 greedy 文本一致（数值差 ≤0.5%）；该层耗时；
**显存**：P1 不释放 GPU1/2（权重还在），P2 再缩容。

### 5.6 P1 进展（2026-10-24 09:15）

- **GPU 可见性打通**：`GPUS=1,2,0` ⇒ 逻辑 0/1 = 物理 1/2（TP=2），逻辑 2 = 物理 0（侧车）；
- **权重复制已验证**：pwal 时把 layer0 的原始路由权重复制到 `cuda:2`，
  每 rank 0.708GB（两 rank 合计 ~1.4GB），耗时 1.3s；GPU0 占用 2.3GB；
- `CPU_OFFLOAD_GB=3 / GPU_UTIL=0.92` 时启动成功，KV 0.72GB；
- **forward 分流补丁已写好**（`routed_experts.py`）：目标层把 40% 专家的 topk 权重置 0
  交给 Marlin，GPU0 用原始权重算这 40% 并按原权重加回；
- 下一步：实跑对比 greedy 文本与耗时，确认数值一致后推到 48 层。

### 5.7 P1 实跑：分流已接通，但输出异常（2026-10-24 09:35）

**已打通**：
- `GPUS=1,2,0` 三设备可见，`cuda:2` = 物理 GPU0；
- 两个 rank 都成功把 layer0 原始权重（各 0.708GB）复制到 GPU0（合计 ~1.4GB）；
- `EAGER=1` 启动成功（KV 0.88GB），`forward_modular` 分流被调用。

**异常**：请求输出退化为 `ductduct...`（96 token 全是同一 token），
"1+1" 也乱码；说明**分流后的数值/映射不对**。

**最可能原因**：
1. TP=2 下 `topk_ids` 可能是**本地 id**（0–255）而非全局（0–511），
   当前 `eg // ne == local` 的映射会错；
2. `topk_weights` 在 Marlin 内部可能被**归一化/routed_scaling**，
   置零个别专家会改变其余专家的缩放；
3. 两个 rank 的置零/并回语义需要逐 rank 对拍确认。

**下一步**：先用 **TP=1**（单 rank、无 all-reduce）验证"置零 + 并回"的端到端正确性，
确认数值后再移植到 TP=2（正确 id 映射 + 每专家只算一次）。

**同时确认**：跨设备 CUDA Graph capture 冲突 ⇒ P1 暂用 `EAGER=1`；
后续要么禁用该层的 graph，要么把侧车调用移到 capture 之外。

### 5.8 P1 深挖（2026-10-24 10:05，TP=1 隔离）

- 侧车设备改为"最后一个可见设备"（TP=1: cuda:1；TP=2: cuda:2），
  TP=1 单 rank 成功复制 **1.416GB**（512 专家）；
- `forward_modular` 分流触发，`topk_ids` 是**全局 id**（0–505），`w13_weight_scale_2`
  形状 `[512,2]` fp32；
- **发现 NaN**：prefill（x 最大 7.5、560 行）时 `acc=nan`；decode（x 2.3、10 行）时
  `acc=0.0497` 正常 ⇒ NaN 只在大 batch/较大激活时出现；
- 加 `nan_to_num` 兜底后文本**部分恢复**（"1+1 equals 2" 出现），但仍有乱码 token，
  且 debug 显示 decode 的 `acc` 被兜底成 0 ⇒ 贡献丢失；
- 下一步：定位 NaN 来源（怀疑 `silu*up` 在 bf16 溢出，或个专家 `scale_2` 异常），
  改为 fp32 激活、逐专家上限检查后再 bf16。

**当前结论**：P1 的"置零 + 并回"机制可跑通，但数值稳定性未解决；
vLLM 源码已恢复，补丁归档 `docs/patches/qfn_a_forward_v2.patch`。

### 5.9 P1 再测（2026-10-24 10:25）：_os 作用域修复，但数值仍不对

- 修 `UnboundLocalError` 后侧车真正执行；诊断输出：
  - `w13_weight_scale_2`：`[512,2]` fp32，**min=max=4.476e-4**（vLLM 里是**统一值**，
    不是 checkpoint 的逐专家/逐投影值）；
  - prefill 一步：`acc=2279.87` vs Marlin `r=0.0283`（差 ~8e4）；
  - decode 一步：`acc=0`（被 nan_to_num 兜底，说明仍有 NaN）；
- 结论：
  1. vLLM 的 `w13_weight_scale_2` 语义与 checkpoint 不同（统一 4.476e-4），
     不能直接沿用 P0 的逐专家 scale_2；
  2. 卷积核的缩放链（block E4M3 × scale_2 × down scale_2）在引擎里需要重新对齐；
  3. NaN 仍未定位（decode 行触发）。
- 补丁归档 `docs/patches/qfn_a_forward_v3.patch`；vLLM 源码已恢复。
- 下一步：以 `layer.w13_weight_scale_2`/`w2_weight_scale_2` 的**实际值**为准重算，
  并在单层再做一次离线对拍（用引擎 dump 的 x/ids/out + 引擎里的 scale_2）。

### 5.10 P1 诊断收敛（2026-10-24 10:35）

- 引擎实际值：`w13_scale_2` 统一 4.476e-4；`w2_scale_2` 逐专家 3.97e-5–4.53e-4；
- 文本**恢复连贯**（"用户要求用一句话解释快速排序…"），但 debug 显示 `acc=0`，
  且没有 `[qfn-a-nan]`/`[qfn-a-nan2]` 触发 ⇒ `acc` 的 NaN 不是来自 act/o 的溢出；
- 更可能的解释：**置零没有影响 Marlin**（`apply_router_weight_on_input` 或权重在输入端生效），
  于是输出 ≈ 原始 Marlin，侧车贡献被丢弃/覆盖 ⇒ 文本正常但机制没生效；
- 下一步必须先确认：
  1. `layer.apply_router_weight_on_input` 是否为 True；
  2. 置零后的 `wts2` 是否真的让被置零专家的贡献变为 0（可对比 r 与 baseline）；
  3. `acc` 的 NaN 究竟在哪一步（打印 `wt`、`S2_2[e]`、`o` 的有限性）。
- 补丁归档 `docs/patches/qfn_a_forward_v4.patch`；vLLM 源码已恢复。
