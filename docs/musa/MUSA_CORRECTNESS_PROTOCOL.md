# MUSA / S4000 正确性对拍协议

> 目标：官方 FP8 qfn 在 MUSA 与 CUDA 参考上输出一致；不追求性能。

## 1. 固定条件

| 项 | 值 |
|---|---|
| 权重 | `Qwen/Qwen3.8-Flash-Next-FP8` |
| 采样 | greedy：`temperature=0`, `top_p=1` |
| batch | 1 |
| MTP / 投机 | 关 |
| CUDA Graph | 关（eager） |
| GPU prefill | 关 |
| 上下文 | 先 8K，再 32K |
| 输出 | 固定 `max_tokens=256`（首轮） |
| 专家 | CPU 全部专家 |
| PLE | host |

## 2. 参考端

- CUDA qfn官方 FP8 现有可运行配置；
- 记录：模型 revision、vLLM 版本、量化配置、启动参数。

## 3. 候选端

- `vllm-musa v0.32.0-dev` + qwen4_exp MUSA 补丁；
- 记录：MUSA Toolkit、torch_musa、驱动、S4000 拓扑。

## 4. 对拍

```bash
python scripts/musa/compare_openai.py \
  --ref http://cuda-ref:8000 --ref-model qwen38-fp8 \
  --cand http://musa:8000 --cand-model qwen38-fp8 \
  --prompts docs/musa/fixtures/prompts.jsonl \
  --max-tokens 256 --out /tmp/musa-correctness.json
```

判定：

- 首轮要求 **exact text match**（greedy，batch 1）；
- 若不一致，记录首个差异位置、两边 token 数、finish_reason；
- 若仅末尾截断差异，检查 `max_tokens` / EOS；
- 若数值差异导致 token 分叉，保留两边 logprobs（后续加）并缩到单层。

## 5. 8K 扩展

- 用同一组长 prompt（8K 左右）重复上述流程；
- 不压缩 MBT，不指定 KV 字节，只指定上下文长度；
- 记录 prefill 是否 OOM、KV 峰值、首 token 时间（仅记录，不作性能结论）。

## 6. 验收

- [ ] 4 条固定 prompt 全部 exact match；
- [ ] 8K 至少 2 条 exact match；
- [ ] 无 NaN / 无异常 token；
- [ ] PLE 路径参与且无越界；
- [ ] MTP 关闭时结果稳定。
