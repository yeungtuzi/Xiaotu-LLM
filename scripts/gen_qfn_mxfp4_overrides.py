#!/usr/bin/env python3
"""生成 QFN(MXFP4-FP8 Qwen3.8-Flash-Next)的 `--hf-overrides` JSON。

为什么需要它(证据:docs/EXPERIMENTS.md B465 / B479):
    检查点 `config.json` 的 `compressed-tensors` 配置里,`group_2`(8-bit float **block**
    [128,128] 权重 + dynamic group-128 激活)的 targets 只覆盖【未融合】名:
        re:.*\\.linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$
    而本版 vLLM 的 Qwen3.8-Flash-Next 把 GDN 线性注意力融合成 `in_proj_qkvz`(以及
    `in_proj_b` / `in_proj_a`)⇒ 没有任何 rule 命中 ⇒
        "No compressed-tensors compatible scheme was found for ... in_proj_qkvz"
    ⇒ 必须用 `--hf-overrides` 扩展 targets。

⛔ 关键教训(2026-10-15,B479):**override 里只写 `targets` 是错的** ✗
    `--hf-overrides` 对 `quantization_config` 是**整体替换该键** ⇒ 只写
        {"quantization_config": {"config_groups": {"group_2": {"targets": [...]}}}}
    会把 **group_0 / group_1 一起抹掉**,并把 group_2 的 `weights` / `input_activations`
    也抹掉 ⇒ 后果是两个症状:
      ① 专家退化成 **unquantized**(group_0 = MXFP4 没了);
      ② attention 层 scale 参数名变成 `weight_scale` 而检查点给的是
         `weight_scale_inv`(block 才有)⇒
         `ValueError: There is no module or parameter named '...in_proj_qkvz.weight_scale_inv'`
    ⇒ 正确做法:**完整继承检查点的 `config_groups`**,只**扩展 `group_2.targets`** ✓

用法:
    python scripts/gen_qfn_mxfp4_overrides.py \\
        --config /path/to/Qwen3.8-Flash-Next-MXFP4-FP8/config.json \\
        --out /tmp/ov/qfn_mxfp4_hf_overrides.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 需要补进 group_2 的【融合名】(vLLM 侧把未融合名折成这些)
FUSED_EXTRA_TARGETS = [
    r"re:.*\.linear_attn\.in_proj_qkvz$",
    r"re:.*\.linear_attn\.in_proj_b$",
    r"re:.*\.linear_attn\.in_proj_a$",
]

GROUP = "group_2"


def build(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    qc = cfg["quantization_config"]
    groups = qc["config_groups"]
    if GROUP not in groups:
        raise SystemExit(f"[qfn-ov] 检查点里没有 {GROUP};现有: {sorted(groups)}")

    # ⭐ 深拷贝整组(保留 weights / input_activations 等全部字段)
    new_groups = json.loads(json.dumps(groups))
    targets = list(new_groups[GROUP].get("targets") or [])
    for t in FUSED_EXTRA_TARGETS:
        if t not in targets:
            targets.append(t)
    new_groups[GROUP]["targets"] = targets

    # ⭐⭐ 审计发现(2026-10-15,独立评审 1faf305e 离线实证)⛔:
    # CT 会在组内缺 format 时用【全局 format】回填(compressed_tensors.py:983-984)⇒
    # 全局 format = `mxfp4-pack-quantized`,而我们的白名单只加了 :716/:819 两处
    # (**channel/group** 策略)⇒ 权重为 **block[128,128]** 的 group_1/group_2 全部落空 ⇒
    # `get_scheme` 在 :902 抛 NotImplementedError ⇒ 加载必然失败。
    # ⇒ 修法:给 group_1 / group_2 **显式**写入 `format: float-quantized`(group_0 是 MXFP4,不写)。
    for _g in ("group_1", "group_2"):
        if _g in new_groups:
            new_groups[_g]["format"] = "float-quantized"

    # ⛔⛔ 关键(2026-10-15,B486):**必须把整个 `quantization_config` 深拷贝回来** ✗
    #     若只写 {"quantization_config": {"config_groups": ...}},则 `format` / `quant_method`
    #     / `quantization_status` 等**同级键会被整体替换掉** ⇒ 后果:
    #       ① 专家侧认不出 MXFP4 ⇒ 退回 **unquantized**(日志实测 ✓);
    #       ② attention 侧 group_2 的 scale 参数**根本没建** ⇒
    #          `weight_scale_inv` 无法解析 ⇒ 加载崩(EXPERIMENTS B479/B484/B485)✓
    qc_new = json.loads(json.dumps(qc))          # 深拷贝全部同级键(format/quant_method/...)
    qc_new["config_groups"] = new_groups          # 只替换其中的 config_groups
    return {"quantization_config": qc_new}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    out = build(a.config)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False))
    got = out["quantization_config"]["config_groups"]
    print(f"[qfn-ov] 写出 {a.out} ({a.out.stat().st_size} B);组: {sorted(got)}")
    print(f"[qfn-ov] {GROUP}.targets 末 3 条: {got[GROUP]['targets'][-3:]}")
    if "weights" not in got[GROUP]:
        raise SystemExit("[qfn-ov] ⛔ group_2.weights 丢失 —— override 写法退化,拒绝写出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
