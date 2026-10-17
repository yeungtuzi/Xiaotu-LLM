#!/usr/bin/env python3
"""QFN(MXFP4-FP8)点火前【离线静态闸】—— 不加载模型权重、不占 GPU,秒级返回。

闸定义(与 docs/EXPERIMENTS.md B480/B482 一致):
  G2 override 完整性 : 生成的 hf-overrides 必须仍含 group_0/1/2,且 group_2 保留 weights
                       (8-bit float **block** [128,128])与 input_activations。
                       ⇒ 只写 targets 会把整组替换掉 ⇒ 专家退化成 unquantized + scale 名错(B479)。
  G3 名字覆盖        : 检查点 index 里的每个权重名都必须能归位;无法归位者直接红灯。
  G4 scheme 归属     : `language_model.model.layers.0.linear_attn.in_proj_qkvz` 解析到的
                       CT scheme 必须是 **块** FP8 那支(`CompressedTensorsW8A16Fp8`,
                       它在 process_weights_after_loading 里注册/改名为 `weight_scale_inv`),
                       而不是 `CompressedTensorsW8A8Fp8`(用 `weight_scale`)。
  G5 MXFP4 分支完整性 : `CompressedTensorsW4A4Mxfp4MoEMethod` 在 `__init__` 里选中的**每个**
                       `Mxfp4MoeBackend`,都必须在 `process_weights_after_loading` 的 if 链里
                       被**显式接住**。⭐ 判据来源(**2026-10-15 实证**):`xtu-9` 只在构造器里
                       选了 CPU 后端、忘了加载链 ⇒ CPU 掉进 `else` ⇒
                       `NotImplementedError: '_C::gptq_marlin_repack' ... 'CPU' backend` ⇒
                       `qfnFIX1`/`qfnFIX2` **两次点火全废**(worker 起 13:13:19 → 13:15:14 炸)。
                       ⇒ 这类错误**纯静态可查**,不该靠起服务撞(IRON_RULES **R55**)。
                       实现:`scripts/check_mxfp4_backend_branches.py` ✓

用法:
    python scripts/preflight_qfn.py \
        --vllm-tree /path/to/vllm-mainline \
        --config /path/to/Qwen3.8-Flash-Next-MXFP4-FP8/config.json \
        --override /tmp/ov/qfn_ov_full.json \
        [--index /path/to/model.safetensors.index.json]

退出码:0 = 全绿(可点火);1 = 有红灯(禁止点火);2 = 闸无法判定(禁止点火)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ⛔ 审计纠正(2026-10-15):模型的**真实请求路径**是【融合名】且前缀已被
# qwen4_exp/nvidia/model.py:627-629 从 `model.language_model.` 改写成 `model.` ✓
# (检查点里的 `in_proj_qkv`/`in_proj_z` 是**权重名**,不是 get_scheme 的入参 ✗)
LAYER = "model.layers.0.linear_attn.in_proj_qkvz"

# 检查点里"本应能归位"的权重名后缀(精确集合)
OK_SUFFIX = {
    "weight", "bias", "weight_scale", "weight_scale_inv", "input_scale",
    "weight_scale_2", "input_scale_2", "weight_scale_inv_2",
    # GDN / 线性注意力的常驻参数
    "A_log", "dt_bias",
}


def _ok_name(base: str) -> bool:
    """名字是否属于"模型自己会创建/上游约定"的合法参数名。

    ⚠️ 注意:本函数只回答"名字形态是否合法",**不回答"模型是否真的建了它"** ✗
    (后者只能在加载期才能确证;见 EXPERIMENTS B483)。
    """
    if base in OK_SUFFIX or base in UNFUSED:
        return True
    # MXFP4 打包命名:down_proj_packed / down_proj_scale / gate_up_proj_packed …
    if base.endswith("_packed") or base.endswith("_scale"):
        return True
    # ⭐ PLE(逐层嵌入)元数据:由模型/PLE 通路(插件 ple_offload.py)自行处理,
    # 不是普通线性层的参数(2026-10-15 实测只剩这 3 个,见 EXPERIMENTS B484)。
    if base in {"layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"}:
        return True
    return False
# 未融合名(由 packed_modules_mapping 折进融合参数;linear.py 的 xtu-6 白名单负责跳过)
UNFUSED = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "q_proj", "k_proj", "v_proj"}


def gate_g2(override: Path, cfg: Path) -> bool:
    print("── G2 override 完整性 ──")
    ov = json.loads(override.read_text())
    groups = ov.get("quantization_config", {}).get("config_groups", {})
    ok = True
    for need in ("group_0", "group_1", "group_2"):
        if need not in groups:
            print(f"  ⛔ 缺 {need}(override 把整组替换掉了?)")
            ok = False
    g2 = groups.get("group_2", {})
    w = g2.get("weights") or {}
    if not w:
        print("  ⛔ group_2.weights 丢失 ⇒ 会退化成逐张量(scale 名变 weight_scale)")
        ok = False
    else:
        bs = w.get("block_structure")
        print(f"  ✓ group_2.weights: num_bits={w.get('num_bits')} strategy={w.get('strategy')} block={bs}")
        if w.get("strategy") != "block" or bs != [128, 128]:
            print("  ⛔ 期望 strategy=block + block_structure=[128,128]")
            ok = False
    if not g2.get("input_activations"):
        print("  ⛔ group_2.input_activations 丢失")
        ok = False
    # 与检查点原配置比:targets 必须只增不减(超集)
    orig = json.loads(cfg.read_text())["quantization_config"]["config_groups"]["group_2"]["targets"]
    new = g2.get("targets") or []
    missing = [t for t in orig if t not in new]
    if missing:
        print(f"  ⛔ targets 丢条目: {missing}")
        ok = False
    else:
        print(f"  ✓ targets 为原超集(原 {len(orig)} 条 + 融合名 {len(new) - len(orig)} 条)")
    # ⭐⭐ B486:override 必须【保留 quantization_config 的全部同级键】✗
    # vLLM 的 --hf-overrides 对 `quantization_config` 是**整体替换该键** ⇒
    # 只写 {"config_groups": ...} 会把 format / quant_method / quantization_status 一起抹掉
    # ⇒ 专家退回 unquantized + attention 侧不建 scale 参数 ⇒ 加载崩(B479/B484/B486)。
    orig_qc = json.loads(cfg.read_text())["quantization_config"]
    ov_qc = ov["quantization_config"]
    lost = [k for k in orig_qc if k not in ov_qc]
    if lost:
        print(f"  ⛔ quantization_config 丢了同级键: {lost} ⇒ 必然退化(专家 unquantized / 无 scale 参数)")
        ok = False
    else:
        print(f"  ✓ quantization_config 同级键齐: {sorted(ov_qc)}")
    for k in ("format", "quant_method"):
        if orig_qc.get(k) != ov_qc.get(k):
            print(f"  ⛔ {k} 被改/丢失: 原={orig_qc.get(k)!r} 现={ov_qc.get(k)!r}")
            ok = False
    # ⭐⭐ 审计发现(1faf305e 离线实证):CT 会用【全局 format】回填组内缺失的 format
    # (compressed_tensors.py:983-984)⇒ 全局是 `mxfp4-pack-quantized`(:716/:819 只白名单
    # channel/group 策略)⇒ 权重为 block[128,128] 的 group_1/group_2 必然落空 ⇒
    # get_scheme 在 :902 抛 NotImplementedError ⇒ 加载必崩。
    # ⇒ 机械要求:**group_1 / group_2 必须各自带显式 format**(group_0 是 MXFP4,不要求)。
    for _g in ("group_1", "group_2"):
        _gf = ov_qc.get("config_groups", {}).get(_g, {}).get("format")
        if not _gf:
            print(f"  ⛔ {_g} 缺显式 format ⇒ CT 会用全局 {ov_qc.get('format')!r} 回填 ⇒ block 策略落空(必崩)")
            ok = False
        else:
            print(f"  ✓ {_g}.format = {_gf!r}(不再被全局 format 回填)")
    print(f"  ⇒ G2 {'绿 ✅' if ok else '红 ⛔'}")
    return ok


def gate_g3(index: Path) -> bool:
    print("── G3 名字覆盖 ──")
    names = list(json.loads(index.read_text())["weight_map"])
    bad = []
    for n in names:
        base = n.rsplit(".", 1)[-1]
        if _ok_name(base):
            continue
        bad.append(n)
    print(f"  名字总数 {len(names)};可按规则归位 {len(names) - len(bad)};无法归位 {len(bad)}")
    if bad:
        for n in bad[:10]:
            print(f"  ⛔ 无法归位: {n}")
        print("  ⇒ G3 红 ⛔(这些名字会让加载器兜底成模块 ⇒ 崩)")
        return False
    print("  ⇒ G3 绿 ✅")
    return True


def gate_g4(tree: Path, override: Path, index: Path | None = None) -> bool:
    print("── G4 scheme 归属 ──")
    # ⚠️ 层级前缀必须与检查点【逐字一致】:index 里是 `model.language_model...`,
    # 而我上一轮探针写的是 `language_model...` ⇒ get_scheme 返回 None(假红,见 B483)。
    global LAYER
    if index and index.is_file():
        names = list(json.loads(index.read_text())["weight_map"])
        cands = [n for n in names if ".linear_attn.in_proj_qkvz." in n]
        if cands:
            LAYER = cands[0].rsplit(".", 1)[0]
            print(f"  (layer_name 取自检查点 index:{LAYER})")
        else:
            print("  ⚠️ index 里找不到 .linear_attn.in_proj_qkvz.* ⇒ 沿用内置名")
    sys.path.insert(0, str(tree))
    import torch  # noqa: PLC0415
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: PLC0415
        CompressedTensorsConfig,
    )

    cfg_path = override
    try:
        # ⛔ 审计纠正:`from_config` 期望的入参**就是 QC 本体**
        # (compressed_tensors.py:229-250 直接 config.get("config_groups"/"format"/"ignore");
        #  真实调用点 weight_utils.py:264/379/386 也不包外层)。
        # 我先前多包一层 {"quantization_config": qc} ⇒ target_scheme_map=0 ⇒ **对任何名字都返回 None**
        # ⇒ 那是探针假红,不是被测对象有问题 ✗(见 QFN_LAUNCH_AUDIT.md)
        # ⭐ 且必须用【override 生效后】的 QC,而非检查点原值(两者 group format 可能不同)。
        ct = CompressedTensorsConfig.from_config(json.loads(override.read_text())["quantization_config"])  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ from_config 失败({type(e).__name__}: {e});可用属性: "
              f"{[a for a in dir(CompressedTensorsConfig) if not a.startswith('_')][:12]}")
        print("  ⇒ G4 无法判定 ⛔")
        return False

    class _Stub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # in_proj_qkvz 的输出维 = qkv(z) 的拼接;用足够大的形状即可触发 scheme 选择
            self.weight = torch.nn.Parameter(torch.empty(512, 256))
            self.prefix = LAYER
            self.logical_widths = [256, 256]

    try:
        sch = ct.get_scheme(_Stub(), LAYER)
    except Exception as e:  # noqa: BLE001
        print(f"  ⛔ get_scheme 抛错: {type(e).__name__}: {e}")
        return False
    if sch is None:
        print("  ⛔ get_scheme 返回 None(没有任何 rule 命中 ⇒ 正是 B307/B465 的症状)")
        return False
    cls = type(sch).__name__
    strat = getattr(sch, "strategy", None)
    print(f"  ✓ scheme 类 = {cls};strategy = {strat}")
    good = "W8A16Fp8" in cls and str(strat).upper().endswith("BLOCK")
    if not good:
        print(f"  ⛔ 期望 CompressedTensorsW8A16Fp8 + BLOCK(它才用 weight_scale_inv);"
              f"实际 {cls} + {strat}")
        print("     ⇒ 若为 W8A8Fp8(用 weight_scale),检查点给的 weight_scale_inv 就对不上")
    print(f"  ⇒ G4 {'绿 ✅' if good else '红 ⛔'}")
    return good


def gate_g5(vllm_tree: Path) -> bool:
    """G5 —— MXFP4 MoE 后端分支完整性(IRON_RULES R55:「半截补丁」静态判据)。

    为什么要有它(2026-10-15 实证,代价 = 两次点火全废):
      `xtu-9` 在 `__init__` 里把 `mxfp4_backend` 选成 `Mxfp4MoeBackend.CPU` ✓,
      **但没改 `process_weights_after_loading` 的 if 链** ✗ ⇒ CPU 掉进最后的 `else`
      ⇒ `prepare_moe_fp4_layer_for_marlin(layer)` ⇒ `_C::gptq_marlin_repack` on 'CPU' ✗

    这是**纯静态**可查的 ⇒ 必须在**点火前**挡住,⛔ 不许靠"起服务撞一下"发现 ✗
    检查器本身是**双向验证过**的:未修的树 rc=1 且点名缺哪个后端;已修的树 rc=0 ✓
    """
    print("── G5 MXFP4 后端分支完整性(R55:半截补丁静态判据)──")
    script = Path(__file__).resolve().parent / "check_mxfp4_backend_branches.py"
    if not script.is_file():
        print(f"  ⛔ 找不到检查器:{script}(⇒ 无法判定 ⇒ 禁止点火)")
        return False
    cp = subprocess.run([sys.executable, str(script), str(vllm_tree)],
                        capture_output=True, text=True)
    for line in (cp.stdout or "").rstrip().splitlines():
        print(f"  {line}")
    if cp.returncode != 0:
        for line in (cp.stderr or "").rstrip().splitlines():
            print(f"  {line}")
        print(f"  ⇒ G5 红 ⛔(rc={cp.returncode};见上,它点名了缺哪个后端)")
        return False
    print("  ⇒ G5 绿 ✅")
    return True


def _run(name: str, fn, *args, **kw):
    """⭐ 每个闸**独立**运行:任何一个抛异常 ⇒ 它判红,**但不许拖死其它闸** ✓

    为什么必须这样(2026-10-15 实测):`--override /dev/null` 时 `gate_g2` 抛
    `json.JSONDecodeError` ⇒ 旧版 `main()` 直接崩 ⇒ **G5 根本没跑** ✗
    ⇒ 后果:操作者以为"看了点火闸",其实**只看到第一个闸就没了** ✗
    ⚠️ 判红 = 禁止点火 ⇒ 方向仍是 fail-closed ✓,但**可见性**完全不同:一次看全 ✓
    """
    try:
        return fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⛔ {name} 抛异常 ⇒ 判红(闸无法判定):{type(exc).__name__}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-tree", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--override", required=True, type=Path)
    ap.add_argument("--index", type=Path)
    a = ap.parse_args()

    res = {"G2": _run("G2", gate_g2, a.override, a.config)}
    if a.index and a.index.is_file():
        res["G3"] = _run("G3", gate_g3, a.index)
    else:
        print("── G3 名字覆盖 ── 跳过(未提供 --index)⚠️")
    res["G4"] = _run("G4", gate_g4, a.vllm_tree, a.override, a.index)
    # ⭐ G5(R55):不需要权重/override/index,只要 vllm 树 ⇒ 永远可判 ✓
    res["G5"] = _run("G5", gate_g5, a.vllm_tree)

    print("\n===== 点火前闸汇总 =====")
    for k, v in res.items():
        print(f"  {k}: {'绿 ✅' if v else '红/未判定 ⛔'}")
    allgreen = all(res.values()) and "G3" in res
    print(f"\n⇒ {'✅ 允许点火' if allgreen else '⛔ 禁止点火'}")
    return 0 if allgreen else 1


if __name__ == "__main__":
    raise SystemExit(main())
