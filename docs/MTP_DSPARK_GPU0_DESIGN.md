# MTP/DSpark 完全放 GPU0：设计与实施计划

**状态：设计阶段；target 基线测试实例启动中；未改默认、未接生产。**

## 1. 目标

- target：TP=2，固定 GPU1/2；
- draft：MTP/DSpark 全部放 **GPU0 单卡**；
- draft 权重、KV、CUDA graph/workspace、采样全部在 GPU0；
- draft 不占 GPU1/2 显存/SM，不调用 CPU 专家；
- 完成后做服务级基准：正确性、接受率、step/tok/s、冷/热、显存与主机内存；
- 保留回滚，不接生产默认。

## 2. 当前实现的关键事实

- `serve_ds41.sh` 的 `--speculative-config` 是：

  ```json
  {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}
  ```

- vLLM 的 DSpark speculator 在 **target worker 内**创建：

  ```text
  model_runner.init_speculator(vllm_config, self.device)
  DSparkSpeculator(vllm_config, device)
  ```

- `DSparkSpeculator` 的 `self.device` 就是 target worker 的 device；
- 它的 draft buffers、draft logits、采样、CUDA graph 都建在这个 device；
- `set_attn()` 直接引用 target 的 `block_tables`、`slot_mappings`、`kv_cache_config`；
- draft 的注意力层使用 target 的 KV cache 布局，`precompute_and_store_context_kv()` 也写 target 的 slot；
- vLLM 的 `DraftModelProposer` 明确拒绝 `draft_tp != target_tp`：
  `_raise_if_draft_tp_mismatch()`，原因是 TP>1 时 draft TP=1 会发生 rank 冲突/编译缓存覆盖；
- `_verify_and_get_draft_tp()` 允许显式 `draft_tensor_parallel_size=1`，但 DSpark 仍没有“独立 device / 独立 KV”的实现。

结论：

> **“draft_tp=1 + CUDA_VISIBLE_DEVICES 调换”不能把 draft 放到 GPU0。**
> draft 与 target 共享 KV/slot/device，必须拆执行域。

## 3. 两条可行路线

### A. 独立 draft 服务（推荐先做）

- GPU0 跑独立 draft 进程/服务；
- 自带 MTP 3 层权重、Marlin、KV、CUDA graph、采样；
- target worker 的 speculator 改为 **client**：
  - 把 target hidden states、positions、seq/block 信息发到 GPU0；
  - 收到 draft tokens，返回给 target verify；
- 通信：UDS/ZMQ/共享内存；同机，延迟可控；
- 优点：
  - 完全不占 GPU1/2；
  - draft 与 target 解耦，便于独立重启/基准；
  - 不需要改 vLLM 的 target KV 分配；
- 缺点：
  - 需要自己实现 draft 的 KV 管理与 hidden 交换；
  - 多一次 RPC/同步；
  - draft 的 CUDA graph 要独立捕获。

### B. target worker 内 cuda:2 draft

- 仍在 rank0 worker 内，但把 `DSparkSpeculator` 的 device 改为 GPU0；
- 为 draft 单独分配 KV/block table；
- target hidden 从 GPU1 copy 到 GPU0，draft tokens copy 回；
- 优点：少一次进程/RPC；
- 缺点：
  - 改 vLLM KV 分配、attention backend、CUDA graph；
  - `set_attn()` 目前完全依赖 target 的 block tables，改动面大；
  - 容易把 target 图/缓存搞坏。

**建议：先 A 做通，再评估 B 是否值得。**

## 4. 分阶段

### Phase 0：基线

- 当前 `SPEC=1`（CPU draft）target TP=2；
- 记录 TTFT、step、tok/s、接受率、GPU1/2 显存、CPU 占用；
- 同一组 prompt 固定 nonce。

### Phase 1：GPU0 draft 独立 microbench

- 已有 `scripts/bench_mtp_single_boost.py`：GPU1 单卡 60s 稳态 **585.5 µs/层**，3 层 **1.76 ms**；
- 在 GPU0 上复测，确认权重、Marlin、KV、graph 都在 GPU0；
- 记录 clock/功率；GPU0 全 E=128 会撞 250W 功率墙，约 1110–1290 MHz。

### Phase 2：draft 服务接口

- 定义请求/响应：

  ```text
  request:  hidden_states, positions, seq_lens, block_table/slot, temperature, seeds
  response: draft_tokens [num_reqs, k], draft_logits/confidence
  ```

- 先做 **单请求、单步** 对拍：
  - target 同 hidden；
  - GPU0 draft 输出与当前 DSpark 完全一致；
  - 再扩到多请求/CUDA graph。

### Phase 3：接 target speculator

- target 侧 `DSparkSpeculator` 改成 client；
- rank0 发请求，rank1 通过 TP group 同步 draft tokens；
- 保留 fallback：GPU0 不可用时回退当前 CPU/内置 draft。

### Phase 4：服务级基准

- 同一 target、同一 prompt、同 nonce；
- 对比：
  - 当前 CPU draft；
  - GPU0 draft；
  - 可选：TP=2 draft；
- 指标：接受率、每 verify step、tok/s、TTFT、GPU1/2 显存、GPU0 显存、CPU 使用；
- 判据：GPU0 draft 不占 GPU1/2、不调用 CPU；端到端不劣化才考虑保留。

## 5. 主要风险

1. **DSpark draft 依赖 target KV/slot**：独立 KV 是最大改动；
2. **CUDA graph**：draft 图必须在 GPU0 捕获，且不能混 target 图；
3. **TP 同步**：draft tokens 要广播到 rank1，不能引入新的 all-reduce 热点；
4. **hidden 交换**：GPU1→GPU0 的 hidden 是每步关键路径，必须测实际延迟；
5. **功率墙**：GPU0 单卡全 E=128 只有 ~1.1–1.3 GHz，不能假设 1410 MHz；
6. **回退**：任何失败都能回当前 CPU draft，不影响 target。

## 6. 回滚

- 所有新开关默认 off；
- target 启动参数不变；
- GPU0 draft 失败 → `SPEC` 回退当前路径；
- 不接生产默认，不 push。

## 2026-11-02 补充：外挂 MTP（method=mtp）

vLLM 有 `method=mtp` 的独立 draft 路径：

- `MTPSpeculator.load_draft_model()` 调 `load_eagle_model()`；
- draft 权重是**独立模型**，不是 target 内联 MTP；
- 它有自己的 `self.model`、draft logits/采样、draft KV；
- 比 DSpark 更接近“外挂”。

但它**仍然**：

- 在 target worker 内由 `init_speculator(vllm_config, self.device)` 创建；
- `self.device` 仍是 target GPU；
- `MTPSpeculator` 继承 `AutoRegressiveSpeculator`，draft 与 target 同进程、同 TP 组；
- 没有 `draft_device`，也没有独立 worker。

因此：

> **外挂 MTP 解决了“权重独立”，没有解决“设备独立”。**
> **它比 DSpark 更适合作为 GPU0 独立 draft 的基座**，因为不依赖 target 的 slot/block table；但仍需要把 draft 执行域拆出去，或给 worker 增加独立 draft device + 独立 KV。

对 GPU0 方案的优先级：

1. **外部 MTP 独立服务**（权重/KV/采样全 GPU0）；
2. **DSpark 独立服务**（需要额外处理共享 target KV）；
3. 直接在 target worker 内改 device（改动 vLLM KV/attention，风险最高）。

## 2026-11-02 第 1 轮实测基线

- target TP=2 + 当前 DSpark（`TAG=mtp_gpu0_base PORT=8705 SPEC=1`）已就绪；
- GPU0 单卡 MTP（E=128，60s，10% trim）**587.2 µs/层**，与 GPU1 的 **585.5 µs/层** 一致；
  GPU0 全专家同样撞功率墙，采样期平均 **1239 MHz**；
- 当前服务 e2e（同 nonce `gpu0_base_1102`）：

| case | 当前 CPU/draft 路径 |
|---|---:|
| short1 | 1.517 s |
| short2 | 1.386 s |
| short3 | 1.673 s |
| long_context | 6.163 s |

- 该基线仅作对照；目标版本尚未实现。
- 关键阻塞：DSpark 的 draft 与 target 共享 `block_tables`/`slot_mappings`/`kv_cache_config`，
  不能只靠配置把 draft 搬到 GPU0；外部 MTP 权重独立，但仍在 target worker 内、同 device。

## 2026-11-02 关键结构差异：MTP 可以独立，DSpark 不行

读 `vllm/model_executor/models/deepseek_mtp.py` 确认：

- `DeepSeekMultiTokenPredictor` 自带 **`embed_tokens`**；
- 每层 MTP 自带 **`shared_head` / `lm_head`**；
- 前向只需要：
  - `input_ids`
  - `positions`
  - `previous_hidden_states`（target 的 hidden）
- 不需要 target 的 KV、block table、slot mapping。

因此：

> **MTP 具备“独立 draft 服务”的模型条件：权重、embed、head、KV 都可以在 GPU0。**
> **只缺 target hidden 的传递和 target 侧 proxy。**

而 DSpark：

- draft 注意力直接使用 target 的 `block_tables` / `slot_mappings` / `kv_cache_config`；
- 不能只搬权重。

⇒ **P1 应以 MTP 为第一实现，而不是 DSpark。**

## 2026-11-02 勘误：ds41f 没有 classic MTP，只有 DSpark

前面的“MTP 可独立”结论**不适用于 ds41f**。实测 `method=mtp` 被 vLLM 直接拒绝：

```text
DeepSeek V4.1 has no classic-MTP draft:
its checkpoints ship DSpark stages under mtp.*
(main_proj/markov_head/confidence_head)
and carry no e_proj/h_proj/enorm/hnorm/hc_head weights.
Use speculative method 'dspark' instead of 'mtp'.
```

因此：

- `method=mtp` ❌ 不可用；
- 唯一路径是 `method=dspark`；
- DSpark 的 checkpoint 仍叫 `mtp.*`，但结构是 DSpark，不是经典 MTP；
- DSpark 仍然复用 target 的 KV/slot/block table；
- 独立 GPU0 必须给 DSpark 做独立 KV 和 target hidden 交换。

**结论：外部 MTP 路线在 ds41f 上关闭；P1 只能围绕 DSpark。**

## 2026-11-02 第 5 轮：DSpark 可在 GPU0 独立加载

实测脚本 `scripts/probe_dspark_gpu0_load.py`：

```text
CUDA_VISIBLE_DEVICES=0
method=dspark
draft_tensor_parallel_size=1
XIAOTU_OOT_OVERRIDE=0   # 避免 CPU MoE 覆盖污染独立 draft
```

结果：

```text
draft_arch       = DSparkV41DraftModel
loaded_params    = 3,752,660,578
cuda_alloc_gib   = 10.22
cuda_reserved_gib= 13.20
LOAD_OK
```

结论：

1. **DSpark 可以在 GPU0 单卡独立加载**；
2. 它自带 `embed_tokens` 和 draft 主体，不依赖 target 权重才能构造；
3. 但 `load_weights()` 会**跳过 target 的 embed/lm_head**，这部分目前是随机初始化；
4. 独立服务还必须补：
   - target embed/lm_head 权重；
   - 自己的 KV cache；
   - target hidden 输入；
   - draft token 输出/proxy。

这比“DSpark 必须和 target 共卡”前进了一步：**模型层已经能独立上 GPU0**。

## 2026-11-02 第 6 轮：GPU0 独立 DSpark 加载了 embed/head，logits 通过

在 `scripts/probe_dspark_gpu0_load.py` 中补：

- 从 target checkpoint 读 `embed.weight` 和 `head.weight`；
- 直接拷贝到独立 DSpark 的 `embed_tokens` / `lm_head`；
- 验证 shape、logits。

结果：

```text
loaded embed.weight -> (129280, 5120)
loaded head.weight  -> (129280, 5120)
loaded_params       = 3,752,660,578
cuda_alloc_gib      = 10.22
cuda_reserved_gib   = 13.20
logits_shape        = (2, 129280)
logits_finite       = True
LOGIT_OK
```

结论：

1. **DSpark 的 draft 主体 + target embed/head 已经能在 GPU0 独立加载**；
2. `compute_logits()` 在 GPU0 正常出 logits；
3. 还缺 KV cache 和 attention metadata，暂未跑完整 forward；
4. 下一步是给 3 个 draft 层分配独立 KV，跑 anchor/mask 单步。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_load_20261102.log`

## 2026-11-02 第 7 轮：DSpark 在 GPU0 跑通完整 forward（无 attention 路径）

KV 规格：

```text
layer 40/41/42
block_size = 32
dtype      = uint8
spec       = SlidingWindowMLASpec
page       = 19008 bytes
```

forward 探针（不注入 attention metadata，走 profile/无 KV 路径）：

```text
forward_only_shape  = (1, 5120)
forward_only_finite = True
FORWARD_ONLY_OK
```

这证明：

- DSpark 的完整 `DSparkDeepseekV4Model.forward()` 可以在 GPU0 上执行；
- 3 层 decoder、MoE、hc/mhc、RMSNorm 都能在 GPU0 跑；
- 只差 attention metadata + 真正的 KV 读写。

### 未解决

给 `swa_cache_layer` 手工分配 `4 × 32 × 19008` 的 uint8 KV 后，探针在打印
`kv[2]` 规格后静默退出，未进入 `alloc begin`，也没有 traceback。下一轮先单独
最小化复现这一步，再注入真实 attention metadata。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_forward_20261102.log`

## 2026-11-02 第 8 轮：GPU0 上 DSpark 带 KV 的 forward 跑通

完成了三件事：

1. 给 3 个 draft 层分配并绑定 KV：

   ```text
   num_blocks=4, block_size=32, page=19008, dtype=uint8
   kv[0..2] alloc done / bind done
   ```

2. 手工构造单 token decode 的 `DeepseekSparseSWAMetadata`：

   - `block_table=[[0]]`
   - `slot_mapping=[0]`
   - `seq_lens=[1]`
   - `query_start_loc=[0,1]`
   - `decode_swa_indices=[0]*window`
   - `decode_swa_lens=[1]`

3. 在 GPU0 上执行带 attention 的 forward：

   ```text
   kv_forward_shape  = (1, 5120)
   kv_forward_finite = True
   KV_FORWARD_OK
   ```

结论：

- DSpark 不再只是“无 attention 的 profile forward”；
- GPU0 上已经能完成 **KV 绑定 + attention metadata + 真实 attention 调用**；
- 仍缺：target hidden 的 context KV、真正的 anchor/mask 多 token draft、采样。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_kv_forward_20261102.log`

## 2026-11-02 第 9 轮：GPU0 上跑通 5-token DSpark draft forward

修正 KV cache 的 row 宽度：

- `page_size_bytes=19008` 是 block 级；
- `bind_kv_cache` 需要的 row 宽度是 **584**（`state_content_bytes`，fp8_ds_mla 的 SWA record）；
- 用 `num_blocks=4, block_size=32, row=584, dtype=uint8` 绑定。

流程：

1. 手工写 5 个 context token 的 KV；
2. 构造 5 个 query token 的 metadata；
3. 在 GPU0 上执行 DSpark 完整 forward；
4. 计算 logits。

结果：

```text
context_kv_done
draft_forward_shape = (5, 5120)
draft_forward_finite = True
draft_logits_shape  = (5, 129280)
draft_logits_finite = True
DRAFT_FORWARD_OK
```

结论：

- DSpark 已能在 GPU0 上完成 **context KV + 5-token draft forward + logits**；
- context hidden 目前是随机的，不是 target 的真实 hidden；
- 还没有 sampling；
- 还没有 target 侧 proxy。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_draft_forward_20261102.log`

## 2026-11-02 第 10 轮：GPU0 采样 + target hidden dump 接入

### 采样

在 5-token draft forward 后加 greedy Markov-biased 采样：

```text
sampled_tokens = [15416, 15, 3600, 63, 855]
SAMPLE_OK
```

这是 synthetic context 下的基础设施验证，不代表真实接受率。

### target hidden dump

为接真实 target hidden，在 vLLM 的 `DFlashSpeculator.propose()` 中加了 opt-in dump：

```text
XIAOTU_DSPARK_HIDDEN_DUMP=/tmp/dspark_hidden.pt
```

保存：

- `main_x`：DSpark 的 projected target hidden；
- `last`：target last hidden；
- `input_ids` / `positions`。

测试实例已重启，等待 dump。

### 下一步

1. e2e 一次，拿到 `/tmp/dspark_hidden.pt`；
2. GPU0 用真实 `main_x` 预填 context KV；
3. 跑 5-token draft + 采样；
4. 与 target 当前 DSpark 的 draft 对拍。


## 2026-11-02 第 11 轮：GPU0 5-token draft 步时

在 GPU0 上对 5-token DSpark forward 做 30 次计时（无 CUDA graph，synthetic context）：

```text
draft_step_us_median = 5256.2
draft_step_us_min    = 5191.7
draft_step_us_max    = 6320.1
```

这是“完整 3 层 DSpark + 5 个 query token + 当前 eager 路径”的步时。

关键判断：

- 5.26 ms 已经明显高于 target 单步的 draft 预算；
- 如果 GPU0 draft 不能和 target verify 重叠，这个步时会直接变成净损失；
- 必须做 CUDA graph / 更小 batch / 重叠，才能进入“接近免费”的区间；
- 当前数据也说明：**“独立 GPU”本身不等于免费**，draft 的计算同样要压到 target verify 的影子下。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_draft_bench_20261102.log`

## 2026-11-02 第 11 轮补充：dump 补丁的 `os` 漏导入

首次重启后 target worker 失败：

```text
Worker failed with error 'name 'os' is not defined'
```

原因：在 `dflash/speculator.py` 里加了 `os.environ` dump 逻辑，但文件没有 `import os`。

已补：

```python
import copy
import os
```

并用 `py_compile` 验证通过。测试实例正在重新加载。

教训：vLLM 侧的 opt-in 探针补丁也必须先做 `py_compile`，否则会在 worker 启动/首次 propose 时才炸。

## 2026-11-02 第 13 轮：真实 target hidden 在 GPU0 跑出 5 个 draft tokens

target 侧 dump 已生成：

```text
/tmp/dspark_hidden.pt
main_x shape = (63, 5120)
```

GPU0 独立 DSpark：

```text
real_hidden       = (63, 5120)
context_kv_done
draft_forward_shape = (5, 5120)
draft_logits_shape  = (5, 129280)
sampled_tokens      = [1, 7117, 16, 21017, 304]
SAMPLE_OK
```

这是第一次用**真实 target hidden** 在 GPU0 上完成：

- context KV 预填；
- 5-token DSpark forward；
- 采样出 5 个 draft token。

仍缺：

1. 与 target 当前 DSpark 的 draft tokens 对拍；
2. target 侧 proxy：把 hidden 发给 GPU0，把 tokens 收回来；
3. 多请求、CUDA graph、服务级 A/B。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_real_draft_20261102.log`

## 2026-11-02 第 14 轮：target draft tokens 对拍（首次）

同一份真实 hidden（61 tokens）：

```text
target_draft = [[1309, 3287, 16, 21017, 23917]]
gpu0_draft   = [1, 7117, 16, 21017, 3287]
```

位置 3/4 完全一致，位置 1/2/5 不一致。

当前**不能**据此判定对错，因为：

- target 使用 `draft_sample_method=probabilistic`；
- GPU0 当前只是 greedy Markov-biased；
- 两边采样器不同；
- GPU0 的 `decode_swa_indices` 是简化的全可见索引，target 的 SWA 索引由真实 block table 生成。

这组数据说明：

- hidden 传递链路正确；
- draft forward 数值在部分位置一致；
- **采样和 attention metadata 仍需对齐**，否则不能做逐 token 正确性判定。

下一轮优先：
1. 对齐采样器（至少同一 temperature/seed）；
2. 对齐 SWA index 构造；
3. 再对拍。

日志：`dev-docs/report/tuning/logs/probe_dspark_gpu0_compare_20261102.log`

## 2026-11-02 第 15 轮：第一次 logits 对拍暴露了 profile 污染

第一次 dump 到的 target logits：

```text
shape = (4, 129280)
target_argmax = [30925, 30925, 30925, 30925]
```

四行完全相同，明显是 **profile / dummy** 路径，不是真实 propose。

GPU0 的 5 行 logits：

```text
gpu0_argmax = [1, 35542, 36602, 48543, 3132]
```

因此这组 target logits **不可用于对拍**。

已把 dump 条件改成：

```python
logits.shape[0] > 1 and not (logits == logits[0]).all()
```

只 dump 行间有差异的真实 logits。测试实例正在再次重启。

教训：在 vLLM 里 dump 中间张量，必须同时跳过 `dummy_run` / `is_profile`，并检查张量是否退化为常量。

## 2026-11-02 第 15 轮补充：logits dump 在 CUDA graph capture 中炸掉

第二次重启失败：

```text
CUDA error: operation failed due to a previous error during capture
cudaErrorStreamCaptureInvalidated
```

原因：`_sample_logits` 在 CUDA graph capture 阶段也执行了，`torch.save` / D2H
把 capture 流打断。

已加：

```python
not torch.cuda.is_available() or not torch.cuda.is_current_stream_capturing()
```

并在 `py_compile` 后重启。教训：

> 在 vLLM 的 CUDA graph capture 路径里，不能做 `.cpu()` / `torch.save` / 同步；
> 任何 dump 都必须显式跳过 `is_current_stream_capturing()`。

## 2026-11-02 第 15 轮结果：logits 仍不可比，token 仍不一致

修复 capture 后，target 侧只成功 dump 到：

- `main_x` 62 个 token；
- `draft_tokens = [[1309, 3287, 16, 21017, 23917]]`；
- logits dump 没有触发（真实采样是逐步 1 行，`shape[0] > 1` 条件把它跳过了）。

GPU0 用同一份 hidden（62 token）得到的 argmax：

```text
gpu0_argmax = [1, 35542, 36602, 48543, 3132]
```

仍与 target tokens 不一致。

当前最可能的差异源不是权重，而是：

1. **SWA 索引构造**：GPU0 用的是简化的“全可见”索引；
2. **target 的 sliding-window / bounded replay** 会裁掉左侧历史；
3. **采样器**：target 是 probabilistic，GPU0 是 greedy；
4. **query 布局**：DSpark 的 anchor/mask 位置可能不是 GPU0 当前的 5 个连续位置。

结论：

> GPU0 的 DSpark 能算，但**还没有复现 target 的 draft 逻辑**；
> 在 SWA/query 布局对齐前，token 对拍没有意义。

## 2026-11-02 用户观察：GPU0 只有 ~500MB、无负载

这是对的，而且直接说明当前状态：

- 运行中的 `serve_ds41` 仍然是 **target worker 内 DSpark**；
- draft 权重/KV/计算仍占 **GPU1/2**；
- GPU0 上的 DSpark 只是 `probe_dspark_gpu0_load.py` 进程，**加载约 10–13GB，跑完即退出**；
- 退出后 GPU0 只剩驱动/上下文残留（~500MB），所以看起来“没用”；
- 离线 probe 的真实 hidden / draft token 结果**不等于服务已经使用 GPU0**。

⇒ **目标远未完成。**
下一步必须是：

1. GPU0 上常驻一个 DSpark draft server，持有权重和 KV；
2. target 侧 proxy 把 hidden 发过去、把 tokens 收回来；
3. 移除 target 内部 DSpark 的 draft KV 分配（否则 GPU1/2 仍在为 draft 付显存）；
4. 完成后 GPU0 应稳定显示 **~13GB 权重 + KV**，并且有持续 SM 负载。

仅靠“离线 probe 能跑”不能算完成。

## 2026-11-02 GPU1/2 显存细分（基于 12:03 的实测日志）

来源：`/home/user/lvllm/run/ds41_mtp_gpu0_base.log`（TAG=mtp_gpu0_base，TP=2，util=0.85，PC=0，EXPANDABLE=0）。

### 驱动侧

```text
GPU1/2 各约 38.6 GiB
```

### vLLM 内部

```text
torch allocated = 23.163 GiB
torch reserved  = 37.160 GiB
max_alloc       = 30.395 GiB
fragmentation   = 13.997 GiB
free            = 1.234 GiB
total           = 39.493 GiB
```

### 已知分项

| 项 | 每 GPU |
|---|---:|
| KV cache | **3.61 GiB** |
| 持久 staging/buffer | **2.058 GiB** |
| torch allocated 余量（权重等） | **约 17.5 GiB** |
| reserved 中的碎片/未用 | **约 14.0 GiB** |

持久 buffer 细分：

```text
moe_reuse 0.562 GiB / 2 块
dma       0.234 GiB / 4 块
tile_ring 0.840 GiB / 8 块
landing   0.420 GiB / 1 块
```

### 对“削减”的直接含义

1. **最大的一块不是 draft，是 allocator 碎片**：14.0 GiB。
   - 当前 `EXPANDABLE=0`；
   - 这部分是下一步最值得动的。
2. **KV 只有 3.61 GiB**，不是主要矛盾。
3. **draft 独立加载实测 10.22 GiB（3.75B 参数）**。
   - 如果 target 内的 DSpark draft 权重能被排除，GPU1/2 理论上可省下接近 10 GiB；
   - 但前提是 target 不再为 draft 分配 KV/权重。
4. **持久 buffer 2.058 GiB** 是第二块可优化项，但比碎片小一个量级。
5. **GPU0 常驻 draft server 已达 14.0 GiB**（含权重 + 8 个 KV block + workspace），说明“draft 完全放 GPU0”在显存上是成立的。

优先级判断：

```text
allocator 碎片 14 GiB  >  draft 权重 ~10 GiB  >  持久 buffer 2 GiB  >  KV 3.6 GiB
```

下一步削减应先做 allocator/`expandable_segments` 的 A/B；否则即使 draft 搬走，GPU1/2 仍会被 14 GiB 碎片占着。

## 2026-11-02 第 17 轮：GPU0 常驻 DSpark server 已跑通

新增：

- `scripts/dspark_gpu0_server.py`
- `scripts/dspark_gpu0_client.py`

server 在 GPU0 常驻：

```text
GPU0 used = ~14.0 GiB
socket    = /tmp/dspark_gpu0.sock
```

client 发送真实 hidden 后收到：

```text
{'ok': True, 'tokens': [73165, 342, 1309, 304, 3132]}
```

这是第一次 GPU0 上**常驻**一个可被外部调用的 DSpark draft 服务，而不是跑完即退的 probe。

仍未完成：

- target 还没接到这个 server；
- target 内部 DSpark 还在 GPU1/2；
- target 内部 draft KV 还没移除；
- 还没有服务级 A/B。

但这解决了“GPU0 看起来只有 500MB、没负载”的问题：只要 server 在跑，GPU0 应稳定 ~14GB。

## 2026-11-02 更正：production 的 draft 权重不是全在 GPU

前一节把“standalone DSpark 10.22 GiB”误当成 production 可省的 GPU 显存，这是错的。

事实：

- production 走 `XIAOTU_OOT_OVERRIDE=1`，draft 的 MoE 专家是 **CPU 版**；
- 10.22 GiB 是 **`XIAOTU_OOT_OVERRIDE=0` 的原生 GPU Marlin** 路径；
- 因此“draft 搬走可省 10 GiB”不成立；
- production 的 draft GPU 部分（attention/norm/heads 等）需要单独测，不能按 3.75B 全参数算。

### torch reserved 为什么还占着

最新一次 steady 数据：

```text
allocated = 23.163 GiB
reserved  = 37.160 GiB
max_alloc = 30.395 GiB
frag      = 13.997 GiB
free      = 1.234 GiB
```

原因不是“泄漏”，而是：

1. caching allocator 保留已申请段；
2. `EXPANDABLE=0`，段不能重映射，碎片无法归还；
3. CUDA graph / workspace 使用独立池；
4. 峰值 `max_alloc=30.4`，reserved 按峰值扩到 37.2。

第一批可做的 A/B：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（当前 `PC=0`，无 connector 冲突）；
- 缩小 CUDA graph capture sizes。

### CUDA graph 形状

当前：

```text
[1, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48]
```

`MAXSEQS=4`，`6 = 1 + 5 spec`。因此 **8/12/16/24/32/40/48 这 7 个形状不会用到**。

第一批建议只保留：

```text
[1, 2, 4, 6]
```

预计收益必须实测，不能按单个 graph 大小拍脑袋；下一轮做 A/B：

| 臂 | 配置 |
|---|---|
| A | 当前 |
| B | capture sizes 只留 [1,2,4,6] |
| C | B + `expandable_segments:True` |

指标：`nvidia-smi used`、`torch allocated/reserved/frag`、KV、持久 buffer、TTFT/步时。

## 2026-11-02 第 20 轮：server 128 KV blocks + target remote 重启

发现 server 第一次 illegal access 的根因：

- server 最初只分配 `KV_BLOCKS=8`（256 slots）；
- target 的 e2e 首包 `num_target_tokens` 约 544；
- `precompute_and_store_context_kv` 写到第 256 个 slot 之后越界 ⇒ `cudaErrorIllegalAddress`。

已改为：

```bash
DSPARK_GPU0_KV_BLOCKS=128
```

覆盖 4096 token（128×32），server 重新监听，client 正常返回：

```text
{'ok': True, 'tokens': [35166, 1613, 455, 3967, 14]}
```

这是“离线 probe 能跑 ≠ 服务请求能跑”的又一个具体例子：probe 只有 62 token，服务首包 544 token。

## 2026-11-02 第 20 轮：target 已接 GPU0 server（remote proxy 首次连续工作）

服务日志出现连续：

```text
[dspark-remote] tokens=[223, 1002, 16, 4480, 734]
[dspark-remote] tokens=[223, 19, 16, 223, 20]
[dspark-remote] tokens=[16, 23, 16, 4480, 734]
[dspark-remote] tokens=[23917, 16, 342, 5922, 23917]
[dspark-remote] tokens=[76871, 14, 295, 270, 582]
```

说明 target 的 `DFlash.propose()` 已经在真实服务路径里调用 GPU0 server，而不是只跑离线 probe。

当前 live：

```text
GPU0 server   ≈ 11.5 GiB
GPU1/2 target ≈ 34.3 GiB / 卡
```

对比之前 target 本地的 **38.6 GiB / 卡**，确实下降了约 4 GiB；但这还不是最终形态：

- target 内部 DSpark 权重仍在；
- target 内部 draft KV 是否完全没分配，还需要单独验证；
- 仍有一次 `fallback local: Ran out of input`；
- e2e 还在跑。

下一步：

1. 让 remote proxy 不再 fallback；
2. 用 `[xtu-vram]` 对比 remote 前后的 allocated/reserved/frag；
3. 再移除 target 内部 DSpark 的权重/KV；
4. 做 CUDA graph `[1,2,4,6]` + `expandable_segments` A/B。

## 2026-11-02 第 21 轮：remote proxy 让 target 的 torch reserved 从 37.2 → 22.7 GiB

同一 target，remote draft 生效后的 steady：

```text
[xtu-vram] steady cuda:1 allocated=22.020 reserved=22.736 max_alloc=29.479 frag=0.716 free=15.666 total=39.493 GiB
[xtu-vram] steady cuda:0 allocated=22.020 reserved=22.736 max_alloc=29.479 frag=0.716 free=15.666 total=39.493 GiB
```

持久缓冲：

```text
1.546 GiB (moe_reuse 0.281 + tile_ring 0.840 + landing 0.420 + dma 0.004)
```

对比之前 target 本地 DSpark：

| 指标 | 本地 DSpark | remote DSpark |
|---|---:|---:|
| allocated | 23.163 | **22.020** |
| reserved | 37.160 | **22.736** |
| fragmentation | 13.997 | **0.716** |
| free | 1.234 | **15.666** |
| 持久 buffer | 2.058 | **1.546** |

结论：

- **reserved 降了 14.4 GiB**；
- **碎片从 14.0 GiB 降到 0.7 GiB**；
- 这不是“省 draft 权重”，而是 **target 不再为本地的 draft KV/workspace/graph 付预留**。

但 nvidia-smi 仍是：

```text
GPU1/2 34.3 GiB / 卡
```

torch reserved 只有 22.7 GiB ⇒ 还有约 **11.6 GiB 不在 torch 的 reserved 里**（CUDA context、模块/权重、其它分配）。这部分下一步单独查，不能把它算成“draft”。

当前：

```text
GPU0 server   ≈ 11.5 GiB
GPU1/2 target ≈ 34.3 GiB / 卡
```

注意：GPU0 也有一份 draft server 的权重/KV；总显存并没有变成“免费”，只是把 draft 的 KV/workspace 从 GPU1/2 挪走了。

## 2026-11-02 第 22 轮：必须收回“remote 省 14 GiB”的结论

上一轮看到：

```text
allocated=22.020 reserved=22.736 frag=0.716 free=15.666
```

那是 **startup-before-kv**，draft KV 还没分配。e2e 跑过之后，同一 target 变成：

```text
[xtu-vram] steady cuda:1 allocated=22.936 reserved=36.924 max_alloc=30.397 frag=13.988 free=1.470
[xtu-vram] steady cuda:0 allocated=22.936 reserved=36.924 max_alloc=30.397 frag=13.988 free=1.470
```

持久 buffer 也回到 **2.058 GiB**。

⇒ **remote proxy 本身没有减少 target 的 steady 显存**：
- target 仍然为 draft 分配 **KV/workspace/graph**；
- 22.7 只是“KV 还没分配”的瞬间；
- steady 36.9 / frag 14.0 与本地 DSpark 基本一致；
- GPU1/2 nvidia-smi 仍是 **~38.9 GiB/卡**。

这是本轮最重要的负结果。**remote 只改了计算位置，没有改 target 的显存分配路径。**

要真正做到“draft 不占 GPU1/2”，必须动：

1. `DFlashSpeculator` 的 draft KV cache group / `get_kv_cache_spec`；
2. `bind_kv_cache` 对 draft 层的绑定；
3. target 的 draft CUDA graph/workspace 捕获。

否则 remote 只是“在 GPU0 上又算了一遍”，GPU1/2 的预留一点没少。

## 2026-11-02 第 22 轮：remote proxy 的服务级 e2e 已跑通（但不是正确性验收）

`nonce=remote6_1102`：

| case | prompt | completion | elapsed |
|---|---:|---:|---:|
| short1 | 62 | 64 | **20.235 s** |
| short2 | 61 | 64 | **36.235 s** |
| short3 | 56 | 64 | **5.323 s** |
| long_context | 5222 | 16 | **7.58 s** |

`e2e_rc=0`，没有连接错误、没有 server 越界。

但：

- `short1/short2` 是明显的冷启动；
- `short2` 36 s 很可能是 compile/capture 或 fallback 造成；
- `short3` 5.32 s / 64 token ≈ **12 tok/s**；
- 没有测接受率，也没有和本地 DSpark 同 prompt A/B；
- `text` 对 short2/short3/long 为 null（reasoning 字段），不能当正确性判据。

结论：

> remote proxy **已经能在真实服务里完成请求**；  
> 但**性能与正确性都还没验收**，不能把 e2e_rc=0 当成目标完成。

同时 steady 显存仍是：

```text
allocated=22.936 reserved=36.924 frag=13.988
```

⇒ remote 没有减少 target 的 draft KV/workspace 预留；下一步必须动 `DFlashSpeculator` 的 draft KV 分配路径。

## 2026-11-02 第 23 轮：尝试从 target 里移除 draft KV（进行中）

为让 remote draft 真正不占 GPU1/2，已加两条实验性 patch：

1. `model_runner.get_kv_cache_spec()`：remote 模式下把 `speculator.draft_attn_layer_names`
   从 KV spec 中删掉；
2. `DFlashSpeculator.set_attn()`：remote 模式下不建立 draft KV groups，
   并把 `_context_slot_mappings` 设为 0 行、`_layer_group_idx=None`。

第一次启动报：

```text
'DSparkSpeculator' object has no attribute 'attn_cg_support'
```

第二次报：

```text
'DSparkSpeculator' object has no attribute '_context_slot_mappings'
```

已补字段，正在第三次启动验证。

这不是免费改动：它绕过 vLLM 的 draft KV group 假设，必须确认：
- target 自己能正常分配/绑定 KV；
- remote proxy 不依赖被删掉的 draft metadata；
- 输出与之前 remote（未删 KV）版本一致。

若失败，回滚这两条 patch，保留 remote + 本地 draft KV 的中间态。

## 2026-11-02 第 23 轮结果：删除 target draft KV 的三次尝试均失败，已回滚

尝试路径：

1. `model_runner.get_kv_cache_spec()` 去掉 draft layer；
2. `DFlashSpeculator.set_attn()` 跳过 draft KV group；
3. 补 `_context_slot_mappings` 空张量、`_layer_group_idx=None`。

三次启动错误：

```text
'DSparkSpeculator' object has no attribute 'attn_cg_support'
'DSparkSpeculator' object has no attribute '_context_slot_mappings'
index 0 is out of bounds for dimension 0 with size 0
```

结论：

- vLLM 的 draft KV group 不是“只分配、不用”；
- `set_attn` 之后还有 capture/metadata 路径会读 `_context_slot_mappings[0]`；
- 仅靠“删 spec + 跳过 set_attn”无法安全绕过；
- 已把 `model_runner.py` 和 `dflash/speculator.py` 的这两处实验 patch **全部回滚**，恢复 remote + 本地 draft KV 的中间态。

这轮没有降低 GPU1/2 steady 显存；但确认了“删 draft KV”不是两行 patch 能解决的，必须改 `bind_kv_cache` / metadata / capture 的完整契约。

## 2026-11-02 第 24 轮：回滚后 target 已恢复，steady 仍为 36.2 / frag 13.4

回滚第 23 轮的两处 draft-KV 实验 patch 后，target 重新 ready：

```text
[xtu-vram] steady cuda:1 allocated=22.707 reserved=36.152 max_alloc=30.386 frag=13.445 free=2.242 total=39.493 GiB
```

GPU1/2：

```text
38.1 GiB / 卡
```

KV：

```text
Available KV cache memory: 3.61 GiB
kv cache group sizes [32, 32, 32, 32, 128, 8]
```

⇒ 回滚成功，remote 中间态恢复；但 128/8 两个 draft KV group 仍在，steady 与本地 DSpark 基本一致。

本轮没有实现“draft 不占 GPU1/2”，只是把第 23 轮的失败改动清干净，避免把不稳定的服务留在机器上。

## 2026-11-02 第 25 轮：remote draft 的接受率只有 0.46%

`/metrics`：

```text
spec_decode_num_drafts_total         = 653
spec_decode_num_draft_tokens_total   = 3265  (653 × 5)
spec_decode_num_accepted_tokens_total = 15
```

总接受率：

```text
15 / 3265 = 0.46%
```

按位置：

```text
position 0: 11 / 653 = 1.68%
position 1:  3 / 653 = 0.46%
position 2:  1 / 653 = 0.15%
position 3:  0
position 4:  0
```

结论：

- remote 能出 token，但 **draft 基本不可用**；
- 不是“性能差一点”，而是 **第一个位置就 98% 不匹配**；
- 说明 GPU0 的 metadata / SWA / query 布局与 target 不一致，采样不是主因；
- 在这种接受率下讨论 step/tok/s 没有意义。

这也解释了为什么 e2e 的 generation throughput 只有 **3–10 tok/s**：每轮几乎都在拒绝 draft。

下一步必须先做 **同 hidden、同 metadata、同采样** 的 token 对拍，把 position 0 的接受率提上去，否则 remote 只是“形式上接通”。

## 2026-11-02 第 26 轮：发现 GPU0 server 用错了 query input_ids

server 之前把 query 位的 `input_ids` 全填 0，只把第 0 位设成 anchor：

```python
ids = torch.zeros((q_len,), ...)
ids[0] = anchor
```

但 target 的 DSpark 查询位是真实的最后 `q_len` 个 token，DSpark 的 `input_ids` 会参与
query embedding / 位置语义。zeros 会让所有 query 位看到错误的 token。

已改：

- target remote 请求里增加：
  ```python
  "input_ids": input_batch.input_ids[
      num_target_tokens - self.num_query_per_req : num_target_tokens
  ].cpu()
  ```
- server 优先使用请求里的 `input_ids`，不再 zeros。

这条与“SWA/metadata 不一致”是独立的错误源，可能是 position 0 只有 1.68% 接受率的主因之一。
正在重启验证。

## 2026-11-02 第 27 轮：修 query input_ids 后接受率仍为 0.45%

修复后跑 e2e，指标增量：

```text
drafts      54 → 454      (+400)
draft tokens 270 → 2270   (+2000)
accepted     1 → 10       (+9)
```

总接受率：

```text
9 / 2000 = 0.45%
```

按位置：

```text
position 0: 7 / 400 = 1.75%
position 1: 2 / 400 = 0.50%
position 2: 1 / 400 = 0.25%
position 3: 0
position 4: 0
```

⇒ **query input_ids 不是主因**；修完没有提升。

e2e 时延：

```text
short1 18.56 s
short2  6.86 s
short3 12.89 s
long    2.12 s
```

时延有改善，但接受率仍不可用。

结论：GPU0 的 DSpark 与 target 的 **attention metadata / SWA / query 布局 / 采样** 仍不一致。
下一步必须先做“同 hidden、同 metadata”的单步 logits 对拍，而不是继续调时延或显存。

## 2026-11-02 第 28 轮：同 hidden 的 token 对拍，target/server 完全不同

同一份 `/tmp/dspark_hidden.pt`（63 token，最后 5 个 id=[5234,20,21858,128804,128821]）：

```text
target draft = [1309, 3287, 16, 21017, 23917]
server draft = [5013, 62, 80, 12904, 21000]
```

**5 个位置全不一致。**

最可能的根因：**draft MoE 后端不同**。

- target 走 `XIAOTU_OOT_OVERRIDE=1`，draft experts 是 **CPU XiaotuMoE**；
- server 走 `XIAOTU_OOT_OVERRIDE=0`，draft experts 是 **GPU native Marlin**；
- 两者不是同一套数值路径，top-1 完全可能不同。

尝试让 server 也用 `XIAOTU_OOT_OVERRIDE=1` 复现 target：

```text
KeyError: 'model.layers.0.ffn.experts.routed_experts.w13_weight_scale'
```

⇒ 当前 server 的加载路径无法直接套 CPU override。

结论：

1. 0.45% 接受率不是单纯的 metadata/采样问题；
2. **draft 权重后端不一致** 是首要嫌疑；
3. 在 server 与 target 使用同一 draft MoE 后端之前，token 对拍没有意义。

下一步必须先做一个“同一后端”的最小对拍：
- 要么让 server 正确加载 CPU XiaotuMoE；
- 要么让 target 的 draft 用 native GPU 后端；
- 之后再比较 SWA/采样。

## 2026-11-02 第 32 轮：发现 remote 开关的 "0" 真值 bug

`/tmp/xiaotu_env` 已改成：

```text
XIAOTU_DSPARK_REMOTE=0
```

worker environ 也确认是 `0`，但服务日志仍出现：

```text
[dspark-remote] tokens=...
```

根因：

```python
if os.environ.get("XIAOTU_DSPARK_REMOTE"):
```

字符串 `"0"` 在 Python 里是 **truthy**，所以 remote 一直没关。

已改成：

```python
if os.environ.get("XIAOTU_DSPARK_REMOTE", "") not in ("", "0", "false", "False"):
```

这也解释了为什么上一轮“切回本地 DSpark”完全没生效。现在重新启动本地 target，准备 dump step0–4 logits。

## 2026-11-02 第 32 轮结果：remote 真关掉了，但 local 也不对

修复 `"0"` 真值 bug 后，worker 环境为：

```text
XIAOTU_DSPARK_REMOTE=0
```

服务日志不再出现 `[dspark-remote]`，local DSpark 路径确实生效。

e2e：

```text
short1 21.661 s
short2 23.487 s
short3  2.044 s
long    7.297 s
e2e_rc=0
```

但本地 DSpark 的指标：

```text
num_drafts_total       = 85
num_draft_tokens_total = 425
num_accepted_tokens    = 缺失（即 0）
```

⇒ **local 也没有接受任何 draft token**。

这推翻了“remote 是唯一问题”的假设。现在至少有三层问题：

1. remote/native 后端差异；
2. target local DSpark 自身也不接受；
3. logits dump 仍未触发（`/tmp/dspark_logits.pt.step*` 不存在）。

下一步不能再假设“local 一定正确”。必须先用 local 自己的 draft 对拍 target 的 `draft_tokens`，
确认 local DSpark 在本服务里是否真的工作。

## 2026-11-02 第 33 轮：干净 local 基线恢复，接受率 45.6%

清空 `/tmp/xiaotu_env`（不 dump、不 remote）后，target 本地 DSpark：

```text
num_drafts_total       = 71
num_draft_tokens_total = 355
num_accepted_tokens    = 162
```

接受率：

```text
162 / 355 = 45.6%
```

e2e：

```text
short1 1.555 s
short2 1.252 s
short3 1.595 s
long_context 11.558 s
e2e_rc=0
```

⇒ **target 本地 DSpark 是好的**，45.6% 是正常量级。

这直接证明：

- 之前的 0 接受不是 target 本身的问题；
- **remote 路径的 GPU0 draft tokens 与 target 不一致**；
- 0.45% 是 remote server 的 forward/权重/metadata 问题，不是 DSpark 模型本身。

当前：

```text
GPU0 server   ≈ 13.8 GiB（空闲）
GPU1/2 target ≈ 37.2 GiB / 卡
local DSpark  : 45.6% acceptance
remote DSpark : 0.45% acceptance
```

下一步：只给 target 开 hidden/draft dump（不开 remote），拿本地 hidden + draft，
再和 GPU0 server 的同 hidden 输出逐 token 对拍。

## 2026-11-02 第 34 轮：local hidden 544 + server 对拍仍不一致

干净 local baseline：

```text
target local acceptance = 162 / 355 = 45.6%
```

只开 hidden/draft dump（不开 remote/logits），拿到：

```text
hidden = (544, 5120)
target draft = [1309, 3287, 3967, 1353, 16]
```

GPU0 server 用同一 hidden + input_ids：

```text
server draft（全可见）= [4139, 1610, 14, 790, 1936]
server draft（SWA=128）= [271, 2581, 1309, 304, 7722]
```

仍然不一致。

唯一值得注意的偶然：

```text
server[2] = 1309 = target[0]
```

但这不足以说明对齐；更像 query 位置/SWA 布局不同导致的错位。

结论：

- local 45.6% 是可靠基线；
- server 的 forward 不等价于 target；
- 可能同时有 **SWA window** 和 **query 位置/query token 对齐** 两个问题；
- 仅把 `decode_swa_width` 从 544 改成 128 不够。

下一步：固定 `hidden`，先只对 **第一行 logits** 做 target/server 对比，并打印 `sample_pos` / query positions；
在 logits 对齐前不再跑整段 e2e。

## 2026-11-02 第 36 轮：发现 query 布局是 1+N，不是 N

target：

```text
num_query_per_req = 1 + num_speculative_steps = 6
```

server 之前一直用 `q_len=5`，少了一个 anchor query。

改成 6 后，同一 hidden 的结果：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server draft = [73165, 3132, 6922, 16, 21017, 3132]
```

仍然不一致。

说明 query 长度不是唯一问题；但这是必须修的布局错误：

- target 查询是 **anchor + N 个 mask**；
- server 之前只有 N 个；
- 只改长度还不够，anchor/mask 的 logits 取法、SWA、positions 仍需对齐。

当前 local 45.6%、remote 0.45% 的差距仍未解释。

## 2026-11-02 第 37 轮：补上 anchor 丢弃后仍不一致

target remote 请求已经是：

```text
q_len = 1 + N = 6
input_ids = 最后 6 个
```

server 也改成返回 `tokens[1:]`（丢弃 anchor 位）：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server draft = [3132, 6922, 16, 21017, 3132]
```

仍不一致。

这轮至少把 1+N 的 anchor 对齐做了；但 server 的 token 序列与 target 仍有明显偏移：
server 第 3 个 token 是 `16`，target 最后一个是 `16`，但其余位置不对。

结论：query 长度和 anchor 丢弃只是必要条件，不是充分条件；SWA/positions/权重后端仍有差异。

## 2026-11-02 第 38 轮：query/logits dump 打错了类

在 `DFlashSpeculator._generate_draft` 里加 query dump，结果：

```text
/tmp/dspark_query.pt 不存在
[dspark-query-dump] 0 次
```

原因：ds41f 的 DSpark 子类 **覆盖了 `_generate_draft`**，实际走的是
`DSparkSpeculator._generate_draft`，不是 DFlash 的基类实现。

这解释了：

- query dump 不触发；
- logits dump 也可能因为 `_sample_logits` 的 `_xtu_real_propose`/路径打在错误类上而不触发；
- 之前的 hidden/draft dump 能工作，是因为它们打在 `DFlash.propose` 的公共入口。

下一步必须直接改 `dspark/speculator.py` 的 `_generate_draft` / `_sample_logits`，
而不是 DFlash 基类。

## 2026-11-02 第 39 轮：DSpark 侧 query dump 也失败，已回滚

在 `DSparkSpeculator._generate_draft` 里加 query dump 后，服务启动/首请求直接：

```text
RuntimeError: Executor failed.
```

说明 `input_buffers.input_ids[:num_tokens_padded]` 在该路径下不是安全的可 dump 张量
（可能为空/未初始化，或与 capture 交互）。

已把 `/tmp/xiaotu_env` 恢复为只带 hidden/draft dump，去掉 `XIAOTU_DSPARK_QUERY_DUMP`，
重新启动干净 local target。

⇒ 第 39 轮没有拿到 query；也没有新接受率；只确认了 DSpark 侧同样不能随意 dump。

## 2026-11-02 第 40 轮：干净 local 基线再次确认，48.95% 接受率

清空实验 env 后，target 本地 DSpark：

```text
num_draft_tokens_total = 715
num_accepted_tokens    = 350
acceptance             = 350 / 715 = 48.95%
```

e2e：

```text
short1       1.692 s
short2       2.050 s
short3      19.156 s
long_context 31.139 s
e2e_rc=0
```

⇒ local 基线稳定在 **~49%**，与上一轮 45.6% 同量级。

这再次确认：

- local DSpark 可用；
- remote GPU0 路径的 0.45% 是 server 侧 forward/query/SWA/权重问题；
- 不是 target 本身不会 draft。

当前：

```text
GPU0 server   ≈ 11.4 GiB
GPU1/2 target ≈ 38.5 GiB / 卡
local         : 48.95%
remote        : 0.45%
```

下一步仍应先用同一 hidden 做 server/local 的 token 对拍；
local 基线已经足够稳定，可以作为验收对照。

## 2026-11-02 第 41 轮：找到 mask token 128799，但 server 仍不一致

从 config：

```json
"dspark_noise_token_id": 128799
```

target 的并行 draft mask 是 **128799**，不是最后 6 个 prompt token。

server 改成：

```text
ids[0] = anchor
ids[1..5] = 128799
q_len = 6
```

同一 hidden（544）的结果：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server draft = [28, 23917, 223, 1002, 982]
```

仍不一致。

这排除了“query mask 用错 token”这个简单原因。
剩下更可能的：

- draft 权重/后端；
- SWA window；
- query positions；
- target 的 sample_pos 取法。

local 48.95% / remote ~0.45% 的差距仍未解释。

## 2026-11-02 第 43 轮：DFlash 侧 query dump 仍然崩

在 `DFlashSpeculator.propose` 里、`prepare_dflash_inputs` 之后 dump
`sample_indices/sample_pos/input_buffers`，服务首请求直接：

```text
RuntimeError: Executor failed.
```

说明 `input_buffers` 在这个时机仍然不是安全的 CPU 可读张量（可能绑定在
capture 的持久缓冲上，`.cpu()` 会打断 graph/capture，或索引未初始化）。

已再次去掉 `XIAOTU_DSPARK_QUERY_DUMP`，恢复只带 hidden/draft dump 的干净 local。

结论：连续两次 query dump 失败，不再用这种方式拿 query；
要拿只能从 `prepare_dflash_inputs` 的 kernel 参数侧旁路记录，不能直接 `.cpu()`。

## 2026-11-02 第 44 轮：draft block_size 不是 128

怀疑 target 的 draft KV block size 与 server（32）不同，尝试把 server 改成 128：

```text
RuntimeError: fused_deepseek_v4_kv_rope_insert ...
k_cache must be a paged cache on the KV device
```

说明 128 与当前 cache layer 的 paged cache 契约不匹配。

已回滚为 `cache_layer.block_size`（32）。

⇒ block_size 不是简单的 32/128 二选一；必须与 target 的 `kv_cache_group` 真实 spec 一致。
没有新的接受率数据。

## 2026-11-02 第 45 轮：找到 draft KV group 的真实 block_size=8

target 启动日志：

```text
[xtu-kv-group] gid=0 layers=11 block_size=32
[xtu-kv-group] gid=1 layers=11 block_size=32
[xtu-kv-group] gid=2 layers=11 block_size=32
[xtu-kv-group] gid=3 layers=10 block_size=32
[xtu-kv-group] gid=4 layers=8  block_size=128
[xtu-kv-group] gid=5 layers=3  block_size=8
```

DSpark 的 3 个 draft 层属于 **gid=5，block_size=8**，不是 32。

server 改成：

```python
cache_layer.block_size = 8
kv = torch.zeros(KV_BLOCKS, 1, 8, 584, ...)
```

同一 hidden 的结果：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server draft = [1309, 23917, 223, 1002, 16]
```

**首尾两个 token 已经对上**，中间三个仍不一致。

这是本轮最大的进展：

- 之前全错；
- 现在 block_size=8 后首尾对齐；
- 说明 cache 的 paged block 契约修对了一大块；
- 剩余问题是中间 query 的 sample_indices / Markov 采样 / SWA。

SWA 宽度 544/256/64/32 都得到同一结果，说明当前不是 SWA 宽度问题。

## 2026-11-02 第 46 轮：sample_pos +1 不是答案

DFlash kernel：

```python
sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
```

DSpark `SAMPLE_FROM_ANCHOR=True`，所以 target 采样位置是 `query_pos + 1`。

把 server 的 query positions 改成 `query_base + 1` 后：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server draft = [16, 60919, 223, 1002, 982]
```

比原来的 `[1309, 23917, 223, 1002, 16]` 更差（首尾也丢了）。

已回滚到 `query_base`。

结论：server 的 `positions` 与 target 的 `query_pos` 对应关系仍不对；
简单 +1 不可行。block_size=8 仍是唯一确认有效的修正。

## 2026-11-02 第 47 轮：从 anchor 开始采样更差

根据 `SAMPLE_FROM_ANCHOR=True`，尝试让 server 从 logits[0:5] 采样、返回 tokens[0:5]：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server draft = [2581, 1309, 23917, 223, 1002]
```

更差：target 第一个 1309 变成 server 第二个。

已回滚到原来的 `tokens[1:]` 采样，保留 block_size=8。

当前最好的 server 结果仍是：

```text
server draft = [1309, 23917, 223, 1002, 16]
```

首尾对上，中间不对。

结论：仅靠 sample offset 无法解释中间 3 个 token 的差异。
下一步必须拿到 sample_indices/sample_pos 或直接对 logits，而不是继续猜 offset。

## 2026-11-02 第 48 轮：query dump patch 把 DFlash 的 propose 弄丢了，已从备份恢复

启动报：

```text
TypeError: Can't instantiate abstract class DSparkSpeculator
without an implementation for abstract method 'propose'
```

原因：连续的 query dump 编辑把 `DFlashSpeculator.propose` 定义弄丢了。

已：

```bash
cp speculator.py speculator.py.broken_querydump
cp speculator.py.bak_hidden_dump speculator.py
py_compile -> OK
```

恢复后 target 重新 ready，local 指标：

```text
draft tokens = 435
accepted     = 213
acceptance   = 213 / 435 = 48.97%
```

⇒ local 基线完全恢复。

当前 server 仍保留第 45 轮的 **block_size=8**，其最好 token 结果仍是：

```text
target [1309, 3287, 3967, 1353, 16]
server [1309, 23917, 223, 1002, 16]
```

首尾对齐，中间不对。

## 2026-11-02 第 49 轮：把 remote proxy 重接回 restored DFlash 后首请求失败

在从 `bak_hidden_dump` 恢复 `DFlash.propose` 后，重新应用 remote proxy patch，
`remote=1` 启动，首请求：

```text
RuntimeError: Executor failed.
```

说明 remote proxy patch 与恢复后的 DFlash `propose` 版本不兼容（或 proxy 路径与
DSpark 的 query 布局冲突）。

已切回：

```text
XIAOTU_DSPARK_REMOTE=0
```

并恢复干净 local target。

⇒ 第 49 轮没有新的 remote 接受率；local 48.97% 基线保留。

## 2026-11-02 第 50 轮：DSpark 的 query 长度其实可以是 5，且 server 返回 token 数必须匹配

读 `DSparkSpeculator.__init__`：

```python
self.sample_from_anchor = getattr(hf_config, "sample_from_anchor", True)
if self.sample_from_anchor:
    self.num_query_per_req = self.num_speculative_steps   # 5
else:
    self.num_query_per_req = 1 + self.num_speculative_steps  # 6
```

⇒ 之前把 DSpark 固定当成 `1+N=6` 是错的；要读 config 的 `sample_from_anchor`。

更关键的 bug：server 在 `q_len=6` 时返回 `tokens[1:]`（5 个），在 `q_len=5` 时也返回 `tokens[1:]`（4 个）。
target remote 期望 **5 个** draft token；q_len=5 时只返回 4 个 ⇒ 远端路径必然失败。

已改为：

```python
if q_len >= 6:
    return tokens[1:], logits[1:]
return tokens, logits
```

当前同一 hidden：

```text
target draft = [1309, 3287, 3967, 1353, 16]
server q6    = [1309, 23917, 223, 1002, 16]   # 首尾对齐
server q5    = [2581, 1309, 304, 23917, 223]
```

target 形态更像 q6 后取 5 个（首尾对齐）。

⇒ server 现在 q6 返回 5、q5 返回 5，remote 不再因长度失败；
下一步测 remote 接受率，并继续查中间三个 token。

## 2026-11-02 第 50 轮补充：中间 token 不一致也可能是采样方式不同

target 的 config：

```text
draft_sample_method = probabilistic
```

而 GPU0 server 现在用 **argmax（贪心）** 采样 draft token。

即使 forward/KV 完全一致，贪心 draft 与 probabilistic draft 也会在中段分叉；
这可以解释：

```text
target [1309, 3287, 3967, 1353, 16]
server [1309, 23917, 223, 1002, 16]
```

首尾偶然相同、中间随机不同。

⇒ 在确认 forward 一致前，必须先把 server 的采样方式对齐到 target；
否则 token 级对拍本身不成立。

下一步：remote 跑通后，若接受率仍极低，优先做
`draft_sample_method=greedy` 的对照臂（仅实验，不改生产默认），
用同一 hidden 比较 server greedy 与 target greedy。

## 2026-11-02 第 50 轮关键修复：恢复 DFlash 备份时丢了 `import os`

remote proxy 首请求报：

```text
RuntimeError: Worker failed with error 'name 'os' is not defined'
```

根因：从 `speculator.py.bak_hidden_dump` 恢复时，备份里没有 `import os`；
而 remote proxy 用 `os.environ.get("XIAOTU_DSPARK_REMOTE")`，于是直接 NameError。

已在 DFlash `speculator.py` 顶部补回：

```python
import copy
import os
```

并 `py_compile` 通过。

⇒ 之前第 49 轮“remote proxy 不兼容”的结论是错的，真正原因是缺 `import os`。
现在 remote=1 的 target 重新启动，等待 e2e 接受率。

## 2026-11-02 第 50 轮：remote 启动 OOM，暴露 target 的 draft 可能本来就在 GPU0

remote=1 的 target 启动到后半段报：

```text
CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 ... 
Process 333962 has 18.96 GiB memory in use.
```

当时 GPU0 上还跑着 standalone server（约 11–14 GiB）。

这说明 target 的 DSpark draft 模型/工作区在启动时**也会占用 GPU0**，
与 standalone server 叠加后 OOM。

⇒ 之前“target 本地 DSpark 一定在 GPU1/2”假设可能不成立；
需要做一次 **无 server 的 target local** 实验：

```bash
XIAOTU_DSPARK_REMOTE=0
# 不启动 dspark_gpu0_server.py
bash scripts/serve_ds41.sh start ...
```

观察：

- GPU0 是否出现 target draft 的显存；
- GPU1/2 是否只承载 target TP；
- local 接受率是否仍是 ~49%。

若 GPU0 已承载 draft，则“draft 放 GPU0”这个目标的核心部分可能已经由 vLLM
自身的 DSpark 放置实现完成，而 standalone server 是重复的一份。

## 2026-11-02 第 50 轮：无 server 的 local 基线 = 53.08%，但 GPU0 仍是 0

停掉 GPU0 server 后，只跑 local target：

```text
GPU0 = 0 MiB
GPU1 = 37933 MiB
GPU2 = 37933 MiB

num_draft_tokens_total = 925
num_accepted_tokens    = 491
acceptance             = 491 / 925 = 53.08%
```

e2e：

```text
short1       1.941 s
short2       3.175 s
short3       8.820 s
long_context 7.818 s
e2e_rc=0
```

⇒ 结论修正：

- target local DSpark 的 draft 实际在 **GPU1/2**，不在 GPU0；
- GPU0 只有 standalone server 时才被占用；
- 停掉 server 后 GPU0 完全空闲；
- local 基线提升到 **53.08%**，比之前 48.97% 更高（本次 e2e 更长）。

因此目标仍未完成：draft 还占着 GPU1/2。
下一步必须让 target **不加载/不常驻本地 DSpark 权重与 draft KV**，只保留 remote 路径；
这是真正把 draft 从 GPU1/2 释放的关键。

## 2026-11-02 第 52 轮：尝试 remote 模式不加载本地 DSpark，失败

改动：

1. `DSparkSpeculator.load_draft_model` 在 remote 时返回 `torch.nn.Identity()`；
2. `DSparkSpeculator.set_attn` 在 remote 时不建本地 draft KV groups；
3. `DFlash.capture` / dummy profile 在 remote 时跳过。

启动日志仍显示 target 侧有 draft KV groups：

```text
[xtu-kv-group] gid=4 layers=8 block_size=128
[xtu-kv-group] gid=5 layers=3 block_size=8
```

说明 KV groups 由 **target 的 spec** 生成，不依赖本地模型是否加载。

随后 worker 报：

```text
AssertionError: No draft attention groups found.
```

因为 DFlash `set_attn` 里仍断言 `draft_kv_cache_group_ids` 非空。

⇒ 仅跳过本地模型不够；必须同时：

- 从 `model_runner.get_kv_cache_spec()` 的 spec 中删除 draft groups；
- 让 `set_attn` 在 remote 下不再依赖 `attn_groups`，但仍提供 `attn_cg_support`。

本轮失败后已停止 server，恢复 `XIAOTU_DSPARK_REMOTE=0` 的 local target。

local 53.08% 基线保留。

## 2026-11-02 第 53 轮：成功把 target 的 draft KV group 过滤掉，但 target 起不来

改动：

1. `model_runner.get_kv_cache_spec()` 在 remote 下过滤 `block_size != 32` 的 spec；
2. DFlash `set_attn` 在 remote 下只调用基类 `DraftModelSpeculator.set_attn` 取得
   `attn_cg_support`，不依赖 `attn_groups`；
3. `load_draft_model` 返回带 `combine_hidden_states` 的 dummy。

效果：

```text
[xtu-kv-group] gid=0 layers=40 block_size=32
```

**draft KV groups（gid4/5）确实不见了**，GPU1/2 也不再为 draft 分配 KV。

但 target 仍起不来：

```text
KeyError: 'language_model.model.layers.2.attn.compressor.state_cache'
```

原因：按 `block_size != 32` 过滤时，把 target 自己的 **state_cache**（也是非 32）一并删了。

⇒ 不能按 block_size 粗过滤；必须只删 **DSpark draft 层** 的 spec，
保留 target 的 state_cache（layer2 等）。

本轮结束后已停 server，恢复 `XIAOTU_DSPARK_REMOTE=0` local target。

## 2026-11-02 第 54 轮：remote 路径终于跑通，但接受率 0/2130

修复链：

1. dummy 模型补 `combine_hidden_states`（原 `Identity` 缺该接口）；
2. remote 时 target 发送的是 **15360 维** aux hidden；
3. GPU0 server 增加：

```python
if main_x.shape[-1] == 15360:
    main_x = model.combine_hidden_states(main_x)
```

之后 target 不再报错，日志出现：

```text
[dspark-remote] tokens=[361, 565, 35, 201, 15]
```

target READY，但指标：

```text
num_draft_tokens_total = 2130
num_accepted_tokens    = 0
acceptance             = 0.0%
```

⇒ remote 链路已经打通，但 draft token 全错；不是“没接上”，是“接上了但输出不对”。

同时 target 仍有 draft KV groups：

```text
gid=4 layers=8 block_size=128
gid=5 layers=3 block_size=8
```

⇒ 名称过滤没有删掉 draft KV；GPU1/2 仍被 draft KV 占用。

当前已停 remote 与 server，恢复 `REMOTE=0` local target。
local 53.08% 基线保留。

## 2026-11-02 第 56 轮：清掉残留进程，local 恢复 49.88%

清理 371281/371707/371708 等残留后，local target 重新 ready：

```text
GPU0 = 0 MiB
GPU1 = 38331 MiB
GPU2 = 38331 MiB

draft tokens = 405
accepted     = 202
acceptance   = 49.88%
```

e2e：

```text
short1       1.427 s
short2       1.501 s
short3       1.584 s
long_context 6.626 s
e2e_rc=0
```

服务已恢复可用，local 基线 49.88%（与此前 53.08% 同量级）。

remote 仍为 0%，draft 仍在 GPU1/2，目标未完成。

## 2026-11-02 第 57 轮：server 改用 compute_draft_logits/map 后 token 不变

尝试把 server 从 target head 改为 draft head：

```python
logits = model.compute_draft_logits(h)
tok = model.map_draft_to_target(tok)
```

同一 local hidden（62）：

```text
target [[1309, 3287, 16, 21017, 23917]]
server q5 [2581, 1309, 304, 23917, 223]
server q6 [1309, 23917, 223, 1002, 16]
```

与改之前完全一样，说明：

- `compute_draft_logits` 在该 checkpoint 上没有产生可区分的 reduced-vocab 行为；
- 或 `map_draft_to_target` 是恒等；
- server 的第二 token 仍与 target 不同。

⇒ 不是 draft head/remap 的简单问题；剩余差异在 Markov / KV / query 对齐。

server 已停止，保留 local target（49.88%）可用。

## 2026-11-02 第 58 轮：local 冷/热 e2e 基线

local target（REMOTE=0）当前：

```text
GPU0 = 0 MiB
GPU1/2 = 38333 MiB / 卡
```

同 prompt 第二轮（热，前缀缓存命中）：

```text
short1       1.385 s
short2       1.489 s
short3       1.583 s
long_context 0.967 s   # 冷启动同项 6.626 s
```

spec 指标：

```text
draft tokens = 720
accepted     = 344
acceptance   = 47.78%
```

⇒ local 服务级冷/热基线可复现：

- 热 long_context 从 6.626 s 降到 0.967 s；
- 短请求稳定在 1.4–1.6 s；
- 接受率 ~48–50%。

remote 仍 0%，draft 仍占 GPU1/2，目标未完成。

## 2026-11-02 第 60 轮：local 再次确认 49.42%

当前 local target READY：

```text
GPU0 = 0 MiB
GPU1/2 = 39139 MiB / 卡
draft tokens = 520
accepted     = 257
acceptance   = 49.42%
```

e2e 的 JSON 日志因进程提前结束没有落盘，但 spec 指标完整。

⇒ local 基线稳定在 **49–50%**；
remote 仍 0%，draft 仍占 GPU1/2，目标未完成。

## 2026-11-02 第 61 轮：关掉 Markov bias 后 server 完全跑偏

同一 local hidden（62）：

```text
target    [[1309, 3287, 16, 21017, 23917]]
server q5 (no bias) [15537, 15537, 44480, 89127, 1002]
server q6 (no bias) [54204, 44480, 71863, 1002, 1002]
```

对比有 bias 时：

```text
server q6 [1309, 23917, 223, 1002, 16]
```

⇒ Markov bias 是必需的，server 的第一个 token 依赖它；
关闭后连 1309 都丢了。

说明：

- 不是“bias 多余”，而是 server 的 bias/prev/logits 组合仍与 target 不同；
- 下一步应对比 target 与 server 的 `markov_embed(prev)` / `markov_bias` 数值，
  而不是继续猜 query。

server 已停止，local target 保留。

## 2026-11-02 第 63 轮：target/server 的 Markov 权重完全一致

对 target 与 GPU0 server 分别 dump：

```text
model.main_proj.weight
model.main_proj.weight_scale
model.markov_head.markov_w1.weight
model.markov_head.markov_w2.weight
model.confidence_head.proj.weight
```

逐项比较：

```text
main_proj.weight            maxdiff = 0.0
main_proj.weight_scale      maxdiff = 0.0
markov_w1.weight            maxdiff = 0.0
markov_w2.weight            maxdiff = 0.0
confidence_head.proj.weight maxdiff = 0.0
```

⇒ **不是 Markov 权重不一致**。

因此 server q6 第二个 token（23917 vs target 3287）的差异只能来自：

- KV/attention；
- query positions / slot mapping；
- 或 server 的 backbone hidden 与 target 不同。

这排除了一个长期怀疑点。

## 2026-11-02 第 64 轮：server 按 8×128 + 3×8 分块，token 仍不变

按 target 的 KV group 结构，把 server 改成：

```text
前 8 层 block_size=128
后 3 层 block_size=8
```

并给两类 cache 各自构造 `DeepseekSparseSWAMetadata`。

同一 local hidden（62）：

```text
target [[1309, 3287, 16, 21017, 23917]]
server q5 [2581, 1309, 304, 23917, 223]
server q6 [1309, 23917, 223, 1002, 16]
```

与全 8 时**完全一样**。

⇒ block_size 不是当前 token 分叉的原因（至少不是唯一原因）。

已停 server；local target 保持可用。

## 2026-11-02 第 69 轮：清理后 local 恢复 48.43%

实验失败并回滚后，清理所有残留 target，重新启动 local：

```text
GPU0 = 0 MiB
GPU1/2 = 38275 MiB / 卡

draft tokens = 415
accepted     = 201
acceptance   = 48.43%
```

e2e：

```text
short1       5.758 s   # 首请求冷启动
short2       1.431 s
short3       1.696 s
long_context 6.502 s
```

⇒ local 服务恢复可用，基线 ~48%。

remote 仍 0%，draft 仍占 GPU1/2；目标未完成。

## 2026-11-03 第 79 轮：target 的 6 个 KV group 全部是 target 自己的

关键日志：

```text
[xtu-group-names] gid=4 n=8 head=['language_model.model.layers.2.attn', ...]
[xtu-group-names] gid=5 n=3 head=['language_model.model.layers.2.attn.compressor.state_cache', ...]
[xtu-tensor-names] tid=4 head=['language_model.model.layers.2.attn', ...]
[xtu-tensor-names] tid=8 head=['language_model.model.layers.2.attn.compressor.state_cache', ...]
```

6 个 group、9 个 tensor 全部是 `language_model.*` target 层，**没有** `model.layers.40/41/42`。

⇒ draft KV 已经不在 target 的 KV 集合里；
target 仍能启动到 GPU prefill ACTIVE。

当前显存/e2e：

```text
GPU0 = 11415 MiB（server）
GPU1/2 = 38081 MiB / 卡（target）
draft tokens = 125
accepted     = 0
acceptance   = 0.00%
```

结论：

- target 侧 draft KV 已移除；
- GPU0 server 有 draft 权重；
- remote 接受率仍为 0；
- 正确性问题仍在 server token / query / 采样对齐。

## 2026-11-03 第 80 轮：hidden 是 5120，不是 15360；block8 仍不改 token

target hidden dump：

```text
main_x: (62, 5120)
last:   (62, 20480)
```

server 收到的是已经 combine 过的 5120，不需要再 combine。

server 改成全 block8 后：

```text
target [[1309, 3287, 16, 21017, 23917]]
server q6 [1309, 23917, 223, 1002, 16]
```

与之前完全一样，block_size 仍不是 token 分叉原因。

当前：

```text
GPU0    ≈ 11.4 GiB（server）
GPU1/2  ≈ 38.1 GiB / 卡（target）
remote  0 / 125
```

target 的 6 个 KV group 全是 language_model.*，draft KV 不在 target；
剩余问题是 server 的 backbone / context slot / query 对齐。

## 2026-11-03 第 90 轮：恢复快照后 local 回到 58%

从 `/home/user/lvllm/rollback_20261102_r65/` 恢复四个 speculator/worker 文件后重启 local：

```text
GPU0 = 0 MiB
GPU1/2 = 37275 MiB / 卡
draft tokens = 145
accepted     = 84
acceptance   = 57.9%（e2e 进行中）
```

对比实验污染时的 18–22%，说明低接受率确实来自本地树被 remote 实验补丁污染。

当前 remote 已停止；local 恢复可用。

结论：

- local 回滚成功；
- remote base logits 对齐；
- remote 仍 0%；
- 目标未完成。

## 2026-11-03 第 91 轮后：gumbel 采样协议已验证，remote 仍受 context/query 限制

target 本地 dump 的关键参数：

```text
seeds       = [0]
temperature = [0.0]
sample_pos  = [2477..2481]
idx_map     = [2,2,2,2,2]
anchor      = 12529
target draft = [305, 4609, 16, 21017, 15490]
```

把 target 的 base_logits、seeds、temperature、sample_pos、idx_map、anchor 原样送入 GPU0 server 的 `probe_gumbel`：

```text
server = [305, 4609, 16, 21017, 15490]
target = [305, 4609, 16, 21017, 15490]
```

⇒ gumbel 采样逻辑已逐位复现。

remote target 改传完整 seeds/temperature + `req_idx`，server 按 `req_idx` 取 seed/temperature；target 先起 server 再起 target，未再 fallback local。

remote e2e 仍：

```text
draft tokens = 1225
accepted     = 0
```

⇒ 剩余差异在 remote 的 context/query/base_logits，不在采样函数。

local 已恢复：

```text
draft = 380
accepted = 196
acceptance = 51.6%
GPU0 = 0
GPU1/2 = 37255 MiB
```

## 2026-11-03 第 94 轮：server base_logits argmax 与 target 一致，但 bf16 差异使 gumbel 分叉

短 prompt 直接从 target dump 的 `main_x[:61]` 构造 server 请求，context positions 必须用 `arange(61)`，不能用 `input_buffers.positions`（该缓冲前 5 个是 query/stale 数据）。

结果：

```text
target base argmax = [1309, 3287, 23917, 101579, 3132]
server base argmax = [1309, 3287, 23917, 101579, 3132]
base maxdiff = 0.296875
head maxdiff = 0.25

target draft = [1309, 3287, 16, 21017, 23917]
server draft = [1309, 52087, 23917, 60862, 129266]
```

⇒ context/query 的 argmax 完全一致；gumbel 对 bf16 微小差异敏感，第二 token 起分叉。

target 采样参数：

```text
seeds=[0], temperature=[0.0]
idx_map=2
sample_pos=[62..66]
anchor=2581
```

结论：remote 0% 的剩余原因是 server 与 target 的 base_logits 数值未逐位一致（context KV/slot/block 细节），不是采样协议。

local normal 已恢复：

```text
draft=435 accepted=195 acceptance=44.8%
GPU0=0
GPU1/2=37767 MiB
```

## 2026-11-03 第 95 轮：SWA / block 扫描，base 仍有 0.297 数值差

target 短 context（61）直接送 server，扫描 `swa_width`：

```text
w=16  maxdiff=6.890625
w=32  maxdiff=2.421875
w=64  maxdiff=0.375
w=128 maxdiff=0.296875
```

`swa_width=128` 最优；`DSPARK_GPU0_BLOCK=128` 与 block8 结果相同（maxdiff 0.296875）。

argmax：

```text
target = [1309, 3287, 23917, 101579, 3132]
server = [1309, 3287, 23917, 101579, 3132]
```

但 gumbel：

```text
target draft = [1309, 3287, 16, 21017, 23917]
server draft = [1309, 52087, 23917, 60862, 129266]
```

⇒ SWA/block 不是根因；剩余是 target/server query/context hidden 的 bf16 数值差。

target `input_buffers.positions` / `input_ids` 前 5 个是 query 数据，不是 context；用 `arange(ctx_len)` 才是 context positions。

## 2026-11-03 第 97 轮：remote 0% 新根因——target 请求可能是多请求 batch，server 按单请求处理

target `propose()` 的真实 dump：

```text
main_x  = (66, 5120)
positions = [422,423,424,425,426,427, 0..59]
num_target_tokens = 66
target base/draft = shape (2,5)
draft[1] = [1309, 3287, 16, 21017, 23917]
sample_pos[1] = [61..65]
anchor[1] = 2581
```

即：**同一个 main_x 里包含两个请求段**（6 tokens + 60 tokens），positions 每段重新从 0 开始。

server 当前 `serve_one` 把整个 `main_x` 当成**一个** context：

```text
ctx_len = main_x.shape[0]   # 66
query_base = positions[-1]+1
```

于是两段 context 被错误地拼成一个序列，base_logits 必然错，remote draft 0 接受。

### 影响

- `target input_batch` 的隐藏状态是 **batch 拼接**；
- `positions` 每请求重置；
- server 必须按 `query_start_loc` / `seq_lens` 拆分；
- 单请求 e2e 也可能因调度出现 batch>1。

### 下一步

target remote 请求必须携带：

```text
num_reqs
query_start_loc[:num_reqs+1]
seq_lens[:num_reqs]
anchors[:num_reqs]
```

server `serve_one` 必须：

1. 逐请求切 `main_x` / `positions`；
2. 每请求独立 context KV slot + block table；
3. 每请求独立 SWA indices / query slots；
4. 返回 per-request tokens。

这是 remote correctness 的必要条件。

## 2026-11-03 第 99 轮：batch 路径打通，remote 首次非零（15/1120）

修复三处：

1. server gumbel 的 seed/temperature 必须是完整数组，不能用长度 1 的张量配 `req_idx=2`，否则 gumbel kernel 索引越界。
2. server batch 的 `decode_swa_indices` 是请求内位置，不能用全局 KV slot offset。
3. target `req_idx_per_req` 改为 `input_batch.idx_mapping[:num_reqs]`，不能用 `query_start_loc` 索引 per-request 张量（会 GPU 越界 assert）。

结果：

```text
remote e2e:
draft tokens = 1120
accepted     = 15
acceptance   = 1.34%
```

不再出现 HTTP 500 / fallback local / CUDA assert。

但 remote 1.34% 仍远低于 local 45.8%。

local 恢复：

```text
draft=415 accepted=190 acceptance=45.8%
GPU0=0
GPU1/2=37767 MiB
```

剩余问题：target/server base_logits 的数值差导致 gumbel draft 大量不被 target 接受。

## 2026-11-03 第 100 轮：逐请求串行 remote 也只有 1.12%，batch 不是主因

将 target remote 改为逐请求串行调用 server 单请求路径，避免 batch context 拼接：

```text
remote:
draft tokens = 1160
accepted     = 13
acceptance   = 1.12%
```

对比 batch 路径：

```text
15 / 1120 = 1.34%
```

两者都远低于 local 50.8%，说明 batch 不是主因。

已确认：

- target/server base logits argmax 一致；
- gumbel 用 target base_logits + target 参数可逐位复现；
- 但 server 实际 base/head hidden 与 target 有 0.25–0.297 的 bf16 数值差；
- gumbel 对这一点差异敏感，draft 大量不被接受。

local 恢复：

```text
draft=400 accepted=203 acceptance=50.8%
GPU0=0
GPU1/2=37767 MiB
```

剩余根因：target/server query/context hidden 数值不一致，不是 batch/采样协议。

## 2026-11-03 第 102 轮：draft KV cache dtype 线索

target DSpark 的 draft cache dtype 来自：

```python
speculative_config.kv_cache_dtype
```

见 `dspark/utils.py` 的 `cache_config=replace(..., cache_dtype=speculative_config.kv_cache_dtype)`。

server 原来固定 `uint8` 分配 584-byte cache；target attention 的 `_insert_context_kv`
会在 `cache.dtype == torch.float8_e4m3fn` 时使用 `_flashinfer_fp8_kv_scale`。
⇒ server 的 uint8 cache **不会走 FP8 scale**，这是 base/head hidden 0.25–0.297 数值差的
高概率来源之一。

尝试把 server cache 改成 `float8_e4m3fn` 后：

- `precompute_and_store_context_kv` 报
  `AttributeError: 'DeepseekV4FlashMLAAttention' object has no attribute '_flashinfer_fp8_kv_scale'`；
- 给 server attention 补 `_flashinfer_fp8_kv_scale = 1.0` 后，server 又因 `29531` TCPStore
  端口被占用而启动失败，未取得有效对比数据。

当前决定：

- server 默认 cache dtype 回退到 `uint8`（已知 remote 1.1–1.3% 的可用路径）；
- `DSPARK_GPU0_CACHE_DTYPE=float8_e4m3fn` 保留为实验开关；
- 下一步需要：先释放 `29531`、单独起 server，再用同一 target context 对比
  float8 + 真实 scale 与 uint8 的 base logits。

## 2026-11-03 第 103 轮：float8 cache 实验被 kernel 行宽拒绝

隔离起 server（无 target）后，float8 cache + `_flashinfer_fp8_kv_scale = tensor(1.0)`：

```text
RuntimeError: fused_deepseek_v4_kv_rope_insert ...
cache row width does not match its dtype
```

说明 584 元素/行的 float8 布局与 kernel 期望不一致；target 的 draft cache 仍是
`uint8`（target 侧 `cache.dtype == float8_e4m3fn` 分支未触发）的概率更高。

结论：

- float8 cache 线索不成立；
- server 默认回退 `uint8` 是正确的；
- 仍需要从 target `swa_cache_layer.kv_cache` 直接读 dtype/shape 才能确认；
- remote 仍 1.1–1.3%，local 基线维持。

当前 local 正在恢复启动。

## 2026-11-03 第 105 轮：server 改用 target draft VllmConfig

target 的 draft 模型不是用 target 的 attention/cache 配置加载的，而是
`dspark/utils.py:load_dspark_model()` 里：

```python
draft_attention_backend = _resolve_dspark_attention_backend(...)
draft_vllm_config = replace(
    vllm_config,
    attention_config=replace(..., use_non_causal=..., backend=draft_attention_backend),
    cache_config=replace(cache_config, cache_dtype=speculative_config.kv_cache_dtype)
                 if speculative_config.kv_cache_dtype is not None else cache_config,
)
draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)
```

server 之前用 target `cfg` 直接 `get_model`，attention backend/cache 配置与 target draft 不一致。

本轮改动：

- server 用 `replace` 构造 `draft_cfg`；
- 调用 `_resolve_dspark_attention_backend`；
- 设置 `use_non_causal` / `cache_dtype` / `get_draft_quant_config`；
- `set_current_vllm_config(draft_cfg)` 并 `__enter__()`；
- server 分布式端口改为 `DSPARK_GPU0_MASTER_PORT`（默认 29541），可与 target 的 29531 共存。

目的：让 server 和 target 用同一套 draft attention/cache 配置，消除 base/head 数值差。

## 2026-11-03 第 106 轮：target draft cache = fp8_ds_mla（uint8）

启动日志确认 target：

```text
dtype=torch.bfloat16
kv_cache_dtype=fp8_ds_mla
speculative_config=SpeculativeConfig(method='dspark', num_spec_tokens=5)
```

`SpeculativeConfig.kv_cache_dtype` 缺省 `None` ⇒ target DSpark 的 draft cache 继承
target `cache_config.cache_dtype = fp8_ds_mla`，不是 `float8_e4m3fn`。

因此：

- server 的 584-byte `uint8` draft cache 布局是正确的；
- float8 cache 路线确认排除；
- draft attention backend 已对齐后 base maxdiff 仍 0.296875；
- 剩余差异来自 bf16 / attention kernel 数值路径，不是 cache dtype 或配置。

local 正在重新初始化 EngineCore（权重加载阶段），尚未 READY。

## 2026-11-03 第 110 轮：新增 server argmax draft 选项

base maxdiff 0.296875 无法快速消除时，gumbel 会放大差异。server 新增：

```text
DSPARK_GPU0_ARGMAX=1
```

单请求和 batch 路径都支持：draft token 用 argmax，而不是 gumbel；target 侧仍按自己的
gumbel 采样校验。

直接 target context 的 argmax draft：

```text
server argmax = [1309, 3287, 23917, 101579, 3132]
target draft  = [1309, 3287, 16, 21017, 23917]
```

前两 token 与 target draft 一致，后三 token 是 target 的 argmax，不是 target 的 gumbel 采样值。
下一轮验证该选项能否把 remote 接受率从 1% 提升到接近 local。

## 2026-11-03 第 110 轮关键结果：argmax draft 与 target 完全一致

在 target 同一 context 上：

```text
server_argmax = [1309, 3287, 16, 21017, 23917]
target_draft  = [1309, 3287, 16, 21017, 23917]
```

5 个 token **逐位一致**。

此前 server gumbel 得到：

```text
[1309, 52087, 23917, 60862, 129266]
```

⇒ target 的 draft（尽管配置是 `draft_sample_method=probabilistic`）在这个 context 上
等价于 argmax；gumbel 的 bf16 敏感性才是 remote 1% 的直接原因。

server 已支持：

```text
DSPARK_GPU0_ARGMAX=1
```

下一步：remote target + argmax server 跑 e2e，验证接受率是否回到 local 量级。

## 2026-11-04 最终归档：放弃 GPU0 独立 draft，恢复生产

用户决定：**停止 GPU0 独立 DSpark draft 路线，归档相关改动，恢复生产环境并启动。**

### 结论

- 远程 argmax draft 在单 context 上与 target 逐位一致；
- 但服务级 remote 接受率只有 1.1–1.3%，远低于 local 50.8%；
- batch / gumbel / cache dtype / attention backend / SWA 均已排查，剩余 bf16 hidden 数值差无法在独立 server 上稳定复现 target；
- 该路线不再继续，不作为生产默认。

### 归档位置

- 远端：`/home/user/lvllm/archive_gpu0_draft_20261104/`
- 远端打包：`/home/user/lvllm/archive_gpu0_draft_20261104.tar.gz`
- 远端 MANIFEST：`archive_gpu0_draft_20261104/MANIFEST.md`
- 本地：`docs/archive/GPU0_DRAFT_20261104/`

### 恢复内容

已从 `/home/user/lvllm/rollback_20261102_r65/` 恢复：

- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`

`model_runner.py` 恢复前已与 rollback 一致，未覆盖。

### 生产

- 端口：8700
- TP=2（GPU1+GPU2）
- 入口：`scripts/serve_ds41.sh start TAG=prod PORT=8700`
- GPU0 不再承载 draft model；测试实例已全部停止。

## 2026-11-04 回滚事务与 MUSA 保留

- 恢复前/后两套 target 文件已保存为受校验事务：
  - `/home/user/lvllm/rollback_gpu0_draft_20261104/pre`
  - `/home/user/lvllm/rollback_gpu0_draft_20261104/post`
- 回滚入口：`bash /home/user/lvllm/rollback_gpu0_draft_20261104/rollback.sh check|apply pre|post`
- 每次 apply 前自动把当前文件快照到 `backup/<timestamp>/` 并写 SHA256SUMS。
- 当前状态：`STATE=post`（生产 rollback 文件）。
- MUSA 检查：三个被恢复文件中没有 `musa` / `MUSA` / `GPU_DEVICE` / `torch_musa` 标记；
  target 树中其他修改文件未被触碰，MUSA 线修改保留。
- 生产启动：`serve_ds41.sh start TAG=prod PORT=8700`，pid 108250，GPU0 不承载 draft。

## 2026-11-04 生产恢复故障与临时启动

整文件回滚 rollback 后生产两次失败：

1. `KV offloading found no eligible cache groups.`
   - 原因：rollback 的 `gpu_worker.py` 仍带旧的“丢最后 2 个 DSpark KV group”块；
   - 已用补丁只删除该块，`gpu_worker.py` 回到恢复前 md5 `5883faa…`。

2. `dflash propose: self.hidden_states(5120).copy_(hidden_states(15360))`
   - 发生在 PC=1 / KV offload 路径；
   - 旧 dflash 本地 fallback 没处理 15360 → 5120 的 `combine_hidden_states`；
   - 该修复未在本地 patch 中，不能继续远端直接改。

临时处置：

- 用 `PC=0` 启动生产，避开 KV offload 路径；
- 8700 正在加载，pid 113701；
- PC=1 的 DSpark 15360 兼容修复必须在本地 patch 完成后，再单向部署。

## 2026-11-04 8700 恢复成功

根因有两层：

1. `/tmp/xiaotu_env` 默认 bridge 设置了 `XIAOTU_DSPARK_REMOTE=1`；若只 `unset`，bridge 会重新生效，DSpark 走 `_IdentityDraftModel`，本地权重不加载。
2. 真 DSpark + 旧 dflash 在 KV offload 路径下，`hidden_states` 是 15360，而 `self.hidden_states` 是 5120，fallback 缺 combine。

最终启动（显式覆盖 bridge）：

```bash
env XIAOTU_DSPARK_REMOTE=0 \
  bash scripts/serve_ds41.sh start TAG=prod PORT=8700
```

修复文件（本地生成并单向部署）：

```text
docs/archive/PC1_DSPARK_15360_20261104/dflash_speculator.py.fixed
md5 df3909e87ee5643059d460d156d25261
```

效果：

- 生产默认 PC=1（`--kv-offloading-size 64`）
- `/health` HTTP 200
- `/v1/chat/completions` HTTP 200（max_tokens=8）
- GPU1/2 各 38443 MiB
- 日志无 15360 / dflash / ERROR

注意：直接 `serve_ds41.sh start` 会再次加载 `/tmp/xiaotu_env` 的 remote=1；重启生产必须显式 `XIAOTU_DSPARK_REMOTE=0`，或在启动脚本中固化。
