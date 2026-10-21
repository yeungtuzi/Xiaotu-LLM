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

条件：

```text
官方 Qwen3.8-Flash-Next-FP8
CPU 全部专家
GPU attention/KV
PLE host
MTP off
eager
max_model_len = 8192
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
