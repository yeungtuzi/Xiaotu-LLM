# qfn NVFP4 全驻留 —— 加载链路进展（2026-10-23）

> 检查点：`lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4`（129GB，208 分片）
> 目标：3×A100 全驻留、同侧车机制、GPU0 可撤除。

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

## 3. 下一步

1. 在 `fused_moe_modular_method.py` 的 `w1=layer.w13_weight` 处打印 device/id，
   与 pwal 后的 device/id 对比，定位“被谁换掉”；
   `XTU_NVFP4_GPU=1` 时让 NVFP4 层**完全不进 CPU staging**（权重常驻 GPU）；
2. 或给 NVFP4 单独实现 full-residency 的 `FusedMoE` 路径（不经插件 CPU engine）；
3. 再跑 TP=2 起步，随后做 GPU0+1+2 的侧车全驻留与“撤 GPU0 → 部分表”验证。

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
