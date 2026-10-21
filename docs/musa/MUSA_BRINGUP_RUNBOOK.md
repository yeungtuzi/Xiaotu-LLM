# MUSA / S4000 上线 Runbook

> 目标：官方 FP8 qfn 正确性；不追性能；不使用厂商专用镜像作为运行路径。

## 0. 设备确认

```bash
bash scripts/musa_preflight.sh
python3 -m vllm_musa.collect_env   # 安装后
```

记录：S4000 数量、显存、MUSA Toolkit、驱动、torch_musa、vLLM、MUSA 拓扑。

## 1. 代码基线

```bash
git switch feature/musa-correctness
git pull --ff-only
```

目标：

- vLLM **v0.32.0**（qwen4_exp 在此版本）；
- vllm-musa **v0.32.0-dev** @ `93183864`；
- 不用 `qwen38-flash-next` 专用镜像作为运行路径。

## 1.5 检查 CPU 引擎 .so

```bash
bash scripts/musa_preflight.sh | sed -n '/xiaotu CPU engine/,/python packages/p'
```

如果看到：

```text
libcudart.so.13 => not found
libcuda.so.1 => not found
```

说明当前 `.so` 是 CUDA 构建，**MUSA 不能直接用**。先构建 CPU-only 引擎：

```bash
NO_CUDA=1 PYTHON=/path/to/venv/bin/python \
  PYBIND11_INC=/path/to/torch/include \
  bash scripts/build_engine_variants.sh
```

构建后应看到 `.so` **不再依赖 libcudart/libcuda**。

## 2. 应用 qwen4_exp MUSA 候选补丁

先检查：

```bash
bash scripts/musa/apply_qwen4exp_patches.sh /path/to/vllm --check
```

再应用：

```bash
bash scripts/musa/apply_qwen4exp_patches.sh /path/to/vllm
```

补丁：

```text
0001  MUSA 走 amd/Triton 分派
0002  hc.py 关闭 PDL/GDC
0003  PLE 关闭 PDL/GDC
0004  QSA cache 关闭 PDL/GDC
```

## 2.5 CPU-only FP8 qfn-shape 金标准

在真实 MUSA 之前，先验证 CPU 引擎在 qfn-like 路由形状（top_k=10）上的数值：

```bash
python scripts/musa/cpu_fp8_qfn_golden.py
```

期望：

```text
PASS
rel < 2e-2
```

## 3. qwen4_exp import 冒烟

```bash
python scripts/musa/smoke_qwen4exp.py
```

期望：

```text
is_musa: True
qwen4_exp import: ok
model module: vllm.models.qwen4_exp.amd.model
PASS
```

若仍出现 `nvidia.model`，说明 0001 未生效。

## 4. 小 dense 模型

先用小模型验证 vllm-musa 基础链路（attention/KV/sampling），不要直接上 qfn。

记录：

- 是否能起 OpenAI server；
- 是否能完成一次 greedy 请求；
- 是否出现 MUSA kernel/JIT 错误。

## 5. 官方 FP8 qfn，8K，batch 1

启动前设置：

```bash
export XTU_MUSA_CORRECTNESS=1
```

它会强制：`VLLM_EXPERTS_LOAD_DEVICE=cpu`、CPU 全专家、无 GPU 常驻层、无 GPU prefill、上游 `EngramConfig.cpu_offload=True` 负责 PLE host、MTP/投机。
显式设置的环境变量优先。

⚠️ 仍建议在启动 vLLM **之前**显式导出：

```bash
export VLLM_EXPERTS_LOAD_DEVICE=cpu
export XTU_MUSA_CORRECTNESS=1
```

因为 vLLM 可能在插件加载前已缓存 `envs`。

启动（已内置 MUSA correctness profile）：

```bash
CKPT=/path/to/Qwen3.8-Flash-Next-FP8 bash scripts/serve_qwen38.sh
```

该 profile 会强制 CPU 专家、无 GPU 常驻、无 GPU prefill、eager、8K。

⚠️ 启动脚本会显式写 `XIAOTU_ENV_FILE`，插件在 `musa_correctness.apply()` **之前**
读取它；因此不要手工往这个文件里塞旧的 `XIAOTU_PLE_CPU=1` 或
`XIAOTU_MOE_GPU_RESIDENT_LAYERS=...`，否则会覆盖 MUSA correctness profile。

条件：

```text
官方 Qwen3.8-Flash-Next-FP8
CPU 全部专家
GPU attention/KV
上游 `EngramConfig.cpu_offload=True` 负责 PLE host
MTP off
eager
max_model_len = 8192
```

先生成 8K prompt：

```bash
python scripts/musa/make_long_prompts.py --tokenizer Qwen/Qwen3.8-Flash-Next-FP8 \
  --target-tokens 8192 --out /tmp/prompts-8k.jsonl
```

记录：

- 是否 OOM；
- 首 token 是否正常；
- 是否 NaN / 乱码；
- PLE 是否越界。

## 6. 正确性对拍

```bash
python scripts/musa/compare_openai.py \
  --ref http://cuda-ref:8000 --ref-model qwen38-fp8 \
  --cand http://musa:8000 --cand-model qwen38-fp8 \
  --prompts docs/musa/fixtures/prompts.jsonl \
  --max-tokens 256 --out /tmp/musa-correctness.json
```

验收见 [MUSA_CORRECTNESS_PROTOCOL.md](MUSA_CORRECTNESS_PROTOCOL.md)。

## 7. 之后

- 8K exact match → 32K；
- 通过后立即 NVFP4；
- 永不 BF16。
