#!/usr/bin/env python3
"""检查一次 NVFP4 bench 日志是否**真的**验收通过（机械判据，别靠肉眼）。

用法::

    python3 scripts/musa/check_nvfp4_bench.py /root/lvllm/tmp/bench/bench_nvfp4cpu.log

退出码：0 = 全部通过；1 = 有硬失败；2 = 通过但缺证据（例如没开 QFN_PROBE）。

⭐ 为什么要有这个脚本
------------------
本项目最贵的失败模式是 **"能跑但数字全错"**（NVFP4 的组尺度被数值转换毁掉、
GPU 路径的内核开关没接上……）。只看 `tok/s` 根本分不出来 ⇒ 把判据写成代码，
每次跑完先过一遍，避免"看起来跑完了"被当成"验证通过"✗
"""

from __future__ import annotations

import argparse
import re
import sys

# 必须出现的正向标记（缺 = 没走到那条路）
NEED_ENGINE = "引擎=MOE_NVFP4"
NEED_FOLD = "scale_2 已装载"

# 任何一条出现 = 硬失败
FAIL_PATTERNS = (
    "Engine core initialization failed",
    "There is no module or parameter named",
    "未处理的 checkpoint 键",
    "shape mismatch",
    "weight_scale_2 全为 0",
    "CUDA out of memory",
    "illegal memory access",
    "EngineDeadError",
    "RuntimeError: [xtu-nvfp4]",
    "MoE 被跳过",          # A78-8：引擎未构造 ⇒ forward 退化成 identity（数值全错）
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--expect-experts", type=int, default=256,
                    help="每 rank 的本地专家数（EP 分片后），用于报信息")
    args = ap.parse_args()

    try:
        text = open(args.log, "r", errors="replace").read()
    except OSError as exc:
        print(f"[check] 读不到日志: {exc}", file=sys.stderr)
        return 1

    n_engine = text.count(NEED_ENGINE)
    n_fold = text.count(NEED_FOLD)
    fails = [(p, text.count(p)) for p in FAIL_PATTERNS if p in text]
    probes = re.findall(r"^\[probe\] Q=(.*?) -> (.*)$", text, flags=re.M)

    print(f"[check] {args.log}")
    print(f"[check] {NEED_ENGINE}: {n_engine} 次（期望 ≥2 = 两个 worker）")
    print(f"[check] {NEED_FOLD}: {n_fold} 次（期望 = 2×层数，例：48 层 ⇒ 96）")
    if fails:
        print("[check] ⛔ 硬失败标记:")
        for p, c in fails:
            print(f"        ×{c}  {p}")

    bad_probe = False
    if probes:
        print(f"[check] 文本探针 {len(probes)} 条:")
        for q, a in probes:
            print(f"        Q={q} -> {a}")
        # 2+2 必须出现 4（贪心、temperature=0）
        for q, a in probes:
            if "2 + 2" in q and "4" not in a:
                bad_probe = True
                print("        ⛔ 2+2 的答案里没有 4 ⇒ 数值/接线有问题")
    else:
        print("[check] ⚠️ 没有文本探针（没设 QFN_PROBE=1）⇒ **不能**判定数值正确")

    ok = (not fails) and n_engine >= 2 and n_fold >= 2 and not bad_probe
    if ok and not probes:
        print("[check] 结论: 结构 OK，但缺文本证据 ⇒ 需要重跑并带 QFN_PROBE=1")
        return 2
    print("[check] 结论:", "✅ 通过" if ok else "⛔ 不通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
