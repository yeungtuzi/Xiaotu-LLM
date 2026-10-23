# qfn NVFP4 全驻留 —— 加载链路进展（2026-10-23）

> 检查点：`lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4`（129GB，208 分片）
> 目标：3×A100 全驻留、同挎斗机制、GPU0 可撤除。

## 1. 已解决（按出现顺序）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `AssertionError` 于 `load_merged_column_weight`，param `(16,1,20,1)` vs loaded `(16,20)` | modelopt 的 156 个 FP8 投影是 **2D 128×128 block scale**，vLLM 用 4D packed scale | `parameter.py` 加 `_qfn_2d_to_4d()`，在 narrow 前 reshape（补丁见 `docs/patches/qfn_nvfp4_vllm_2026-10-23.patch`） |
| 2 | `'MergedColumnParallelLinear' object has no attribute 'orig_dtype'` | vLLM `marlin_utils_fp8.prepare_fp8_layer_for_marlin` 假定 `layer.orig_dtype` 存在 | 三处改为 `getattr(layer, "orig_dtype", torch.bfloat16)`（同一补丁） |
| 3 | `gptq_marlin_repack` on CPU | 插件把 `ModelOptNvFp4FusedMoE` 的 pwal 走 CPU 路径 | 新增 env **`XTU_NVFP4_GPU=1`**：跳过 NVFP4 的 CPU shim，native pwal（GPU Marlin）后再把权重搬上卡、重建 kernel（`mainline_shims.py`） |
| 4 | `16 is not divisible by 3` | qfn 16 个 head，TP=3 不支持 | 用 TP=2 |

## 2. 当前状态

- **208/208 分片全部加载成功**；`Initializing model took 6.5s`（TP=2，GPU1+2）；
- 已进入推理，但在 `moe_wna16_marlin_gemm` 报：
  `b_q_weight is not on GPU`；
- 诊断打印显示 pwal 之后 `w13_weight/w2_weight/scale` 都在 `cuda:0/1`，但 kernel 拿到的
  `b_q_weight` 仍是 CPU ⇒ **不是 pwal 的 device，而是 experts/kernel 缓存或 apply 路径**。

## 2.1 2026-10-23 21:51 已定位：运行时 w1/w2/scale 都是 CPU

在 `_fused_marlin_moe` 入口打印（临时探针，已回滚）：

```
[qfn-marlin] w1=cpu w2=cpu s1=cpu hs=cuda:0
[qfn-marlin] w1=cpu w2=cpu s1=cpu hs=cuda:1
```

- pwal 之后 `layer.w13_weight` 是 cuda，但 **kernel 实际收到的 w1 是 CPU**；
- 说明 w1 来自另一处缓存（FusedMoE 外层引用/插件 CPU staging），不是 pwal 改的那个属性；
- 在 capture 里做 `.to(cuda)` 会触发 `cudaErrorStreamCaptureInvalidated`，不能靠 kernel 内搬。

## 2.2 2026-10-23 21:57 排除 `device_loading_context`

给 `device_loading_context` 加 `XTU_NVFP4_GPU=1` 直通（不做 CPU 往返）后重跑 TP=2，
仍然 `b_q_weight is not on GPU` ⇒ **不是加载期的 CPU 往返**，是 pwal 之后另有代码
把 `layer.w13_weight` 换成 CPU 张量（或模块级 `w13_weight` 与 pwal 的不是同一个）。

## 2.3 PP=3 不可行（PLE 限制）

尝试 `PP=3, TP=1` 让 3 卡各放 16 层专家（每卡 ~23GB）以避开 CPU 权重，vLLM 直接拒绝：

```
NotImplementedError: Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1
because non-first pipeline ranks do not receive the raw input_ids it needs.
```

⇒ qfn 的 PLE 嵌入把 PP 锁死为 1；**不能用 PP 做 3 卡全驻留**。

## 2.4 结论：TP=2 装不下全部 NVFP4 专家

- 每 rank 专家 ~34GB + 非专家 ~6.5GB + KV ⇒ 超过 40GB；
- vLLM/插件因此把专家权重退到 CPU（这就是 `w1=cpu` 的来源），
  但 native Marlin 只接受 GPU 权重；
- ⇒ **必须实现 NVFP4 的 CPU 专家引擎**（W4A16、group 16、E4M3 scale），
  它同时就是"撤 GPU0 → CPU 兜底"所需要的那块。

## 2.5 2026-10-23 23:00 定位到 create_weights：专家天生就在 CPU

vLLM `ModelOptNvFp4FusedMoE.create_weights` 里是：

```python
data=torch.empty(num_experts, ..., dtype=torch.uint8)   # 没有 device ⇒ CPU
```

loader 的 `device_loading_context` 在加载结束后按设计把参数搬回 CPU；
pwal 期间看到 cuda，但模型加载统计只有 **4.92 GiB/卡**，GPU 峰值 10.7GB
⇒ **34GB 专家从未真正驻留 GPU**。

尝试在 `base_loader.load_model` 之后统一搬卡（`_install_qfn_experts_gpu_shim`），
但该 shim 没有出现在引擎进程的 `mainline shims applied` 列表里（只看到 pwal 的 gpu_fix），
说明 hook 落点/安装时机还不对；且 34+5=39GB 对 39.49GB 只剩 ~0.5GB KV，本身也极紧。

## 2.6 2026-10-23 23:05 搬卡 hook 仍没装上

- 已把 hook 挂到 `base_loader.load_model` 与 `Worker.load_model` 两处；
- 但引擎进程的 `mainline shims applied` 列表里**两条都没有**，且无 `skip/failed` 日志
  ⇒ 函数在 `if os.environ.get("XTU_NVFP4_GPU") != "1"` 或 `orig is None` 处**静默返回 []**；
- 也就是说：**引擎子进程没有看到 `XTU_NVFP4_GPU=1`**（或 base_loader 没有 `load_model` 属性）；
- 这个 env 只在 APIServer/父进程生效（之前 pwal 的 gpu_fix 列表来自父进程），
  worker 是 `spawn` 的，环境可能被 serve 脚本的 envfile **过滤**。

## 4. 2026-10-23 23:15 **qfn NVFP4 已跑通（TP=2 + 2GB CPU offload）**

### 4.1 可用启动命令

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
D=/home/user/.cache/modelscope/models/lovedheart--Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4/snapshots/master
setsid env ALLOW_PROD_GPU=1 XTU_NVFP4_GPU=1 VLLM_EXPERTS_LOAD_DEVICE=gpu \
  CPU_OFFLOAD_GB=2 GPUS=1,2 TP=2 PORT=8142 LOAD=auto CKPT=$D \
  TAG=qfn_nv_cpu2 MAXLEN=512 MBT=512 GPU_UTIL=0.95 \
  XIAOTU_MOE_GPU_RESIDENT_LAYERS=0 bash scripts/serve_qwen38.sh
```

### 4.2 实测

- `Application startup complete`；`Available KV cache memory: 1.66 GiB`；
- GPU1/2 各 **37.78 GiB**，**GPU0 全空（0 MiB）**；
- 语义冒烟：`1+1 等于 2`、`What is the capital of France?` 正常起句；
- **C=1：260 prompt + 64 decode，wall 1.36s ⇒ 47.1 tok/s**
  （官方 FP8 qfn C=1 基线 46.3 tok/s，基本持平）。

### 4.3 关键改动

- vLLM：2D block scale → 4D、`orig_dtype` 回落（见 `docs/patches/`）；
- 插件：`XTU_NVFP4_GPU=1` 跳过 NVFP4 CPU shim + 搬卡；
- serve：`CPU_OFFLOAD_GB=2`（差 ~1.7GB，用 CPU offload 补）；
- **GPU0 未使用** ⇒ "GPU0 可撤除"天然成立；当前形态就是"部分驻留 + CPU 兜底"。

### 5. 2026-10-23 23:30 最佳运行点：1GB offload，C=1 68 tok/s

| 配置 | KV | C=1 | needle | 备注 |
|---|---|---|---|---|
| CPU_OFFLOAD_GB=0 | — | — | — | **OOM**（38.52GB 已用，仅 179MB 空闲；EAGER=1 也不行）|
| **CPU_OFFLOAD_GB=1** | 0.41 GiB | **68.0 tok/s** | ✅ ORANGE-7731 | **最佳** |
| CPU_OFFLOAD_GB=2 | 1.66 GiB | 47.1 tok/s | ✅ ORANGE-7731 | 更慢，但 KV 更大 |

- 1GB vs 2GB：needle **最终答案一致**（ORANGE-7731）；reasoning 文本长度不同（55 vs 63 tokens），
  **不是 bit-identical**，符合计划里“greedy 或 needle 语义一致”的验收口径；
- GPU0 两种配置都**完全空闲** ⇒ “GPU0 可撤除”天然成立；
- 0GB 不可行 ⇒ **至少需要 1GB CPU 兜底**；TP=3 被 16 head 卡死、PP=3 被 PLE 卡死，
  所以“纯 GPU 全驻留”在 2 卡上做不到。

## 6. 2026-10-23 23:40 并发基准（1GB offload，MAXLEN=512，短 prompt 32 tokens）

| 并发 | wall | 单请求 | 聚合 tok/s | 单请求 tok/s |
|---|---|---|---|---|
| C=1 | 0.599 s | 0.599 s | **53.5** | 53.5 |
| C=2 | 0.690 s | 0.345 s | **92.7** | 46.4 |
| C=4 | 1.215 s | 0.304 s | **105.3** | 26.3 |

- 长 prompt（260）C=1 为 **68.0 tok/s**；短 prompt 含 prefill，数字偏低；
- C=4 聚合 105 tok/s，扩展性正常；
- ⚠️ `MAXLEN=4096` 时 KV 只剩 **0.05 GiB**（activation 随 MAXLEN 增长）⇒ 长上下文需更多 offload。

## 7. 2026-10-23 23:55 NVFP4 GEMV 原型跑通（E2M1 + E4M3 g16）

脚本：`dev-docs/nvfp4_gemv_prototype.py`（N-major，1 token/专家）。

| 形状 | 耗时/专家 | 相对误差 |
|---|---|---|
| gate/up (N=1280,K=2560) | **0.030 ms** | 0.0 |
| down (N=2560,K=640) | **0.012 ms** | 0.0 |

- 6 专家全层 ≈ 6×0.030 + 6×0.012 + act ≈ **0.27 ms**（< 0.3ms）；
- 反量化 = e2m1 bitcast + E4M3 算术（无 LUT/常量内存）；
- ⚠️ 这是**单层内核原型**，还不是挎斗集成；但说明 A 方案在性能上可行。

## 8. 2026-10-24 00:05 NVFP4 内核在**真实权重**上验证通过

脚本：`dev-docs/nvfp4_real_check.py`（layer0, experts 0–5，真实 checkpoint）。

| 项 | 结果 |
|---|---|
| w13 形状 / 尺度 | [1280,1280] U8 / [1280,160] F8_E4M3 |
| w2 形状 / 尺度 | [2560,320] U8 / [2560,40] F8_E4M3 |
| gate/up 相对误差 | **0.0**（6/6 专家） |
| down 相对误差 | **0.0**（6/6 专家） |
| **6 专家全层耗时** | **0.2496 ms**（gate/up+down，无 act） |

⇒ 布局/尺度语义（E2M1 + E4M3 g16）**与真实 checkpoint 完全一致**，
且 6 专家 < 0.3ms ⇒ A 方案的**计算内核已就绪**；剩下是模型层集成。

## 4.4 下一步

1. needle / greedy 一致性留档；
2. 尝试去掉 2GB offload（缩小 graph/eager 或把少量专家放 GPU0 挎斗）；
3. C=2/4 与长上下文基准。

## 3. 下一步

0. 把 `XTU_NVFP4_GPU` 写进 serve 的 envfile（不走“父进程 env”），或改用**进程内**
   `pwal` 里注册的 forward-pre-hook（pwal 已证明在 worker 内执行）；
1. **实现 NVFP4 CPU 专家引擎**（复用 e2m1 codebook，scale 从 e8m0 换成 E4M3 g16）——
   这是 TP=2 部分驻留与 GPU0 撤除兜底的共同前提；
   `XTU_NVFP4_GPU=1` 时让 NVFP4 层**完全不进 CPU staging**（权重常驻 GPU）；
2. 或给 NVFP4 单独实现 full-residency 的 `FusedMoE` 路径（不经插件 CPU engine）；
3. 再跑 TP=2 起步，随后做 GPU0+1+2 的挎斗全驻留与“撤 GPU0 → 部分表”验证。

## 4. 复现

```bash
cd /home/user/lvllm/vllm-xiaotu-moe
D=/home/user/.cache/modelscope/models/lovedheart--Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4/snapshots/master
setsid env ALLOW_PROD_GPU=1 XTU_NVFP4_GPU=1 GPUS=1,2 TP=2 PORT=8142 \
  LOAD=auto CKPT=$D TAG=qfn_nv_tp2 MAXLEN=8192 GPU_UTIL=0.88 \
  XIAOTU_MOE_GPU_RESIDENT_LAYERS=0 bash scripts/serve_qwen38.sh
```

- vLLM 侧补丁：`docs/patches/qfn_nvfp4_vllm_2026-10-23.patch`（可 `patch -R` 回滚）；
- 插件侧：`mainline_shims.py` 的 `XTU_NVFP4_GPU` 分支（默认不设 ⇒ 零行为变化）。

## 9. 2026-10-24 09:00 P0 单层对拍**通过**（0.3–0.5%）

- 用 vLLM 钩子 dump 出 **layer 0** 的原始 NVFP4 权重（512 专家，1.4GB）与一次真实请求的
  `x/ids/weights/out`（`topk_ids` 是全局 id，10 个命中）；
- 离线用我们的内核复算：gate/up → silu*up → down，按 `weight_scale_2`（每专家/投影的
  per-tensor 全局缩放，~1e-4）还原，再按路由权重求和；
- **相对误差：0.00488 / 0.005236 / 0.003163（3 个 token）**，量级 0.3–0.5%，
  与 bf16 舍入一致 ⇒ **内核与 Marlin 数值一致**；
- 关键坑：`weight_scale_2` 必须乘（之前只乘 per-16 E4M3 尺度，结果放大 1e4 倍）；
- 钩子补丁已归档：`docs/patches/qfn_dump_weights.patch`、`docs/patches/qfn_trace.patch`；
- vLLM 源码已恢复干净。

⇒ P0 完成；P1 可以开始做"单层交给 GPU0 计算"的引擎内替换。
