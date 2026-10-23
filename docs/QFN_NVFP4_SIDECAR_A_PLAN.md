# qfn NVFP4 挎斗 A 方案 —— 实施计划（2026-10-24）

> 目标：每层把 40–60% 专家放 GPU0（原始 NVFP4 布局），其余留在 GPU1/2 走 Marlin；
> 三卡同时计算；结果按路由权重并回；去掉 1GB CPU offload；GPU0 撤除回退纯 GPU1/2。
> 默认关（`QFN_SIDECAR_A=1` 才启用），可回滚。

## 1. 数据面

- GPU0（逻辑挎斗卡）持有：每层一个**专家子集**的原始 NVFP4 权重
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

- **GPU 可见性打通**：`GPUS=1,2,0` ⇒ 逻辑 0/1 = 物理 1/2（TP=2），逻辑 2 = 物理 0（挎斗）；
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
后续要么禁用该层的 graph，要么把挎斗调用移到 capture 之外。

### 5.8 P1 深挖（2026-10-24 10:05，TP=1 隔离）

- 挎斗设备改为"最后一个可见设备"（TP=1: cuda:1；TP=2: cuda:2），
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

- 修 `UnboundLocalError` 后挎斗真正执行；诊断输出：
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
  于是输出 ≈ 原始 Marlin，挎斗贡献被丢弃/覆盖 ⇒ 文本正常但机制没生效；
- 下一步必须先确认：
  1. `layer.apply_router_weight_on_input` 是否为 True；
  2. 置零后的 `wts2` 是否真的让被置零专家的贡献变为 0（可对比 r 与 baseline）；
  3. `acc` 的 NaN 究竟在哪一步（打印 `wt`、`S2_2[e]`、`o` 的有限性）。
- 补丁归档 `docs/patches/qfn_a_forward_v4.patch`；vLLM 源码已恢复。

### 5.11 P1 **通过**（TP=1，2026-10-24 10:50）

**根因**：挎斗 kernel 在 `cuda:1` 上执行，但 Triton 按 **当前设备**（cuda:0）launch，
指针却指向 cuda:1 ⇒ 读到 0/NaN。修法是计算前 `torch.cuda.set_device(side)`，算完恢复。

**结果**（layer0，40% 专家置零 + GPU0 并回，与"全量 Marlin"对比）：

| 步 | `masked + sidecar` vs `full` 相对误差 | acc |
|---|---|---|
| decode 1 | **0.004791** | 0.00895 |
| decode 2 | **0.003738** | 0.02780 |

- 文本连贯（"用户要求用一句话解释快速排序…"）；
- 与 P0 离线对拍的 0.3–0.5% 完全一致 ⇒ **单层"置零 + 并回"机制数值正确**；
- 补丁归档 `docs/patches/qfn_a_forward_v5.patch`。

**下一步**：移植到 TP=2（本地/全局 id、每专家只算一次、all-reduce 前只加一次），
再推到 48 层并做显存缩容（P2）。

### 5.12 TP=2 移植（2026-10-24 11:00）：单层机制在 TP=1 通过，TP=2 仍偏大

- TP=2（GPUS=1,2,0，每 rank 256 专家）下：`masked_r=0.238/0.160`、`full_r=0.258/0.459`，
  `acc=222.1/277.2`，`rel_err=860/604`；
- `acc` 量级约是期望的 **1/4.5e-4 ≈ 2222 倍**，与 `w13_scale_2` 同量级
  ⇒ 高度怀疑 TP=2 下 **scale_2 没乘上/取到了 1.0**（或索引到错误列）；
- 诊断打印因在 `S13_2` 赋值前引用而崩溃（已回滚），下一轮先把打印移到赋值后，
  确认 TP=2 的 `w13_weight_scale_2` 形状/数值，再修正索引；
- vLLM 源码已恢复；补丁归档 `docs/patches/qfn_a_forward_v6.patch`。

### 5.13 TP=2 诊断受阻（2026-10-24 11:10）

- 连续两次诊断打印都落在 `local` 赋值之前，导致 `UnboundLocalError`；
  说明用字符串替换往长函数里插探针不可靠（顺序不可控）；
- 现状：TP=1 已通过（0.37–0.48%），TP=2 的 `acc` 偏大约 `1/w13_scale_2`（≈2222×），
  仍然高度怀疑 `w13_scale_2` 在 TP=2 下未生效；
- 下一步改为**行号定位插入**（或用 `sys.settrace`/单独脚本读层参数），
  不再用文本替换插探针；
- vLLM 源码已恢复；GPU 全 0 MiB。

### 5.14 TP=2 关键结论（2026-10-24 11:15）

- 用 `QFN_A_MAP=interleave` 与默认 `block` 两种专家映射跑 TP=2，
  **结果完全相同**（`rel_err=860/604`，`acc=222.13/277.23`）
  ⇒ 映射不是根因；更可能 TP=2 下 **专家是复制的（E=512）、intermediate 被切分**，
  所以 `multi` 分支没走到；
- 结合复制数据：w13 形状约 `[512, 640, 1280]`、w2 `[512, 2560, 320]`（I=320/rank），
  与 0.708GB/rank 的体积一致；
- `acc` 仍偏大约 222×（≈1/4.5e-3），说明**缩放链在 TP=2 仍不对**，
  不是专家映射问题；
- 补丁归档 `docs/patches/qfn_a_forward_v7.patch`；vLLM 源码已恢复；GPU 全空。

**下一步**：在 TP=2 下把 `w13/w2` 的实际形状、`S13/S2` 的字节范围、`S13_2/S2_2` 的值
用**行号插入**的方式一次性打印出来，再决定缩放怎么乘；不要再靠文本替换插探针。

### 5.15 TP=2 真实形状确认（2026-10-24 11:20）

用 pwal 里落盘的 `qfn_a_shapes.txt`（不经 forward，避免探针顺序问题）：

| 张量 | 形状 | dtype | 值域 |
|---|---|---|---|
| `w13_weight` | **(512, 640, 1280)** | uint8 | 0–255 |
| `w13_weight_scale` | (512, 640, 160) | E4M3 | 2–224 |
| `w13_weight_scale_2` | (512, 2) | fp32 | 4.4759e-4（两列相同）|
| `w2_weight` | (512, 2560, **160**) | uint8 | 0–255 |
| `w2_weight_scale` | (512, 2560, **20**) | E4M3 | 2.75–384 |
| `w2_weight_scale_2` | (512,) | fp32 | 3.97e-5–4.53e-4 |

**含义**：TP=2 下 `w13` 本地只有 **640 行**（= I，而非 2I），`w2` 的 K=320（= I/2），
即 **intermediate 被切半**（每 rank 320 gate + 320 up，down 吃 320）。

**这解释了 acc 偏大**：我们的内核按 `N13=640 → I=320` 解释，
若本地 640 行的实际排布是 `[gate(640)]` 或 `[up(640)]`（而非 `[gate_half; up_half]`），
`g[:I]`/`g[I:]` 的切法就完全错误，激活与 down 都不成立。

**下一步**：确认 vLLM NVFP4 MoE 在 TP 下 fused w13 的**行序**：
- 若是 `[gate_half; up_half]`：现有切法正确，问题只在缩放；
- 若是整块 gate / 整块 up：挎斗必须先 all-gather 另一半，或在本地改为只算 gate/up 之一，
  由 all-reduce 合并（与 Marlin 的 TP 语义一致）。

### 5.16 下一步的判定依据（2026-10-24 11:25）

TP=2 的 `w13` 本地 640 行只有两种可能排布：

1. **`[gate[0:320]; up[0:320]]`**（MergedColumnParallel 常规切法）
   ⇒ 现有 `g[:I]`/`g[I:]`（I=320）切法正确，问题只在缩放；
2. **整块 `gate(640)` 或 `up(640)`**（按列连续切）
   ⇒ 本地无法算激活，必须 all-gather 另一半或改由 all-reduce 合并。

判定方法（不改引擎、不需要重启）：
- 从 checkpoint 取 layer0 某专家的 `gate_proj`（[640,1280] U8）与 `up_proj`；
- 用 pwal 落盘的 `qfn_a_shapes.txt` 只能看形状，不能看行序；
  所以需要**再落一次权重字节**（pwal 里把 `w13_weight[0, :8, :4]` 的原始字节写文件），
  与 checkpoint 的 gate/up 对应位置比对即可确定行序；
- 确定后再决定是修缩放还是加 all-gather。

当前状态：TP=1 验证通过；TP=2 卡在“行序 + 缩放”两点。

### 5.17 行序**已判定**（2026-10-24 11:30）

用 pwal 落盘的 `w13_weight[0, :8, :4]`、`[316:324]`、`[636:640]` 与 checkpoint 对比：

| 引擎行 | 匹配 |
|---|---|
| 0–7 | `gate_proj[0:8]` ✅ |
| 316–319 | `gate_proj[316:320]` ✅ |
| 320–323 | **`up_proj[0:4]`** ✅ |

⇒ TP=2 本地 `w13` 排布 = **`[gate[0:320]; up[0:320]]`**，
我们现有的 `g[:I]` / `g[I:]`（I=320）切法**正确**，行序不是问题。

**剩余嫌疑**（按可能性）：
1. **down 的 K 半区**：本地 `w2` K=320 是否对应同一 rank 的 intermediate 半区
   （若错位，激活与 down 不匹配）；
2. 激活函数细节（`swiglu_limit`/alpha/beta 是否在 TP=2 下不同）；
3. `acc` 与 `r` 的 all-reduce 语义（r 是否已含部分和）。

下一步：用同一份 dump 打印 `w2_weight[0, :8, :4]` 与 checkpoint
`down_proj[0:8, 0:4]` / `[0:8, 320:324]` 比对，确定 K 半区。

### 5.18 2026-10-24 11:35 停点（需要方向）

- TP=1 单层"置零+并回"已验证（0.37–0.48%）；
- TP=2 已确认：本地 `w13=[gate[0:320];up[0:320]]`（字节级比对）、
  `w13_scale_2=4.4759e-4`、`w2_scale_2=3.97e-5..4.53e-4`；
- 但 TP=2 的 `acc` 仍比本地全量 Marlin 大 ~2222×（≈1/w13_scale_2），
  逐专家中间量落盘未触发（挎斗 branch/调试条件未生效），
  需要一次专门的"逐专家 g/act/o vs checkpoint 参考"审计才能定位；
- 这不是再猜一次能解决的，估计 1–2 天；
- 用户指定的 ds41f C=2/C=4 挎斗基准尚未开始（可先做，1–2 小时有结果）。

**建议**：先做 ds41f C=2/C=4 基准（复用现有插件，不需要改 A 的代码），
把 qfn A 的 TP=2 审计作为下一阶段。

### 5.19 TP=2 **通过**（2026-10-24 13:50）——根因是 BK 不整除 K

- TP=2 下本地 `w2` 的 K=**320**，而内核默认 `BK=128`：`320 = 2×128 + 64`，
  最后一个 chunk 仍按 `BK//16=8` 读 scale，**越界读到 24 > 20**，
  垃圾尺度导致某些专家 `o` 爆到 **7e8**（这正是之前 2222× 的来源）。
- 修法：`BK=64`（320/64=5、2560/64=40、640/64=10 都整除）。
- 结果（layer0，40% 专家置零 + GPU0 并回 vs 本地全量 Marlin）：

| 步 | 相对误差 |
|---|---|
| decode 1 | **0.004303** |
| decode 2 | **0.005157** |

- 文本连贯；与 TP=1 的 0.37–0.48% 同级 ⇒ **TP=2 单层机制数值正确**。
- 补丁归档：`docs/patches/qfn_a_forward_v8.patch`、`qfn_a_copy_v8.patch`。

**下一步**：把这套机制推到 **48 层**（`QFN_A_LAYER` → 全层），再缩容 GPU1/2 拿 KV，
最后做 C=1/2/4 基准。

### 5.20 全 48 层挎斗**跑通**（2026-10-24 14:00）

- `QFN_A_LAYER=all`：rank0 把每层 40%（206/512）专家搬到 GPU0，共 **13.9GB**；
- GPU1/2 各 36.1/36.6GB，KV 1.35GB（仍带 CPU_OFFLOAD_GB=3）；
- 文本连贯；**needle 正确**：`ORANGE-7731`；
- 但 **C=1 只有 4.9 tok/s**（基线 68 tok/s）——慢 14×。

**性能瓶颈**：挎斗是"逐 token、逐专家"的 Python 循环，
每层 10 命中 × 2 次 kernel = 20 次 launch，48 层 ≈ 960 次 launch/token，
再加 `set_device` 与跨设备复制；eager 模式更放大。

**下一步**：把每层所有命中专家批成一个 kernel（gate/up 一次、down 一次），
launch 数从 ~960 降到 ~96/token；再评估是否值得继续。

补丁归档：`docs/patches/qfn_a_all_v9.patch`、`qfn_a_copy_v9.patch`。

### 5.21 性能结论（2026-10-24 14:10）：A 在 A100 上**净负**

同配置（TP=2、`CPU_OFFLOAD_GB=3`、`EAGER=1`、MAXLEN=512、260+64）对比：

| 配置 | C=1 |
|---|---|
| 无挎斗（基线） | **12.3 tok/s** |
| 全 48 层挎斗（40% 专家） | **5.9 tok/s** |

⇒ 挎斗把步时从 81ms 拉到 169ms，**每层多 ~1.8ms**，而被替代的 Marlin MoE 只有 ~0.15ms/层。

- 批量化 kernel 只把 4.9 → 5.9 tok/s（launch 不是主因）；
- 主因：**每层 `set_device` + 3 次跨设备复制 + 同步**，以及 eager；
- 即便把这些优化掉，A 的收益也只是“省 3GB CPU offload 换 KV”，
  但代价是每层固定开销 ≥0.4ms（跨设备），在 A100 上**不可能**接近 68 tok/s。

**结论：qfn 不建议用 GPU0 挎斗。**
- 推荐形态：现有 **TP=2 + 1GB CPU offload，C=1 68 tok/s**；
- 若要长上下文：加大 CPU offload（2GB→KV 1.66GB）或换更大显存的卡；
- A 的唯一潜在价值在 sm120/96GB 级挎斗（内核快 + 显存大），A100 上不成立。

补丁与文档保留，供以后换卡时复用。

### 5.22 挎斗耗时实测（2026-10-24 14:25）：**8.6–11.3 ms/层**

在 `_qfn_a_forward` 内部计时（rank0，全 48 层，批量化 + eager）：

```
[qfn-a-t] n=600 avg_ms=11.295
[qfn-a-t] n=700 avg_ms=9.767
[qfn-a-t] n=800 avg_ms=8.621
```

- 目标本来是 **0.3–0.5ms/层**，实际 **8.6–11.3ms**，差 **20–30×**；
- 48 层 ⇒ 步时 +0.4–0.5s，与实测 5.9 tok/s 一致；
- 主要嫌疑：**每层 `set_device` + 跨设备 `.to()` 的隐式同步**（每层 3 次复制），
  以及 eager 下无法用图消除；
- 批量化只把 4.9→5.9 说明 kernel 本身不是瓶颈，**跨设备边界**才是。

**结论**：A 在 A100 上无法达到 40–60 tok/s；即便把 kernel 优化到 0，
每层跨设备开销仍 ~8ms 量级。要长上下文只能换硬件（96GB/sm120），或接受 512 token。
