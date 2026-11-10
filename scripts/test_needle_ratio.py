#!/usr/bin/env python3
"""test_needle_ratio.py —— 校验 C 探针的**标定与反算**（台账 #256）。

背景：`--tokens 142000` 曾实际生成 181,394 tokens（+27.7%），因为"单字母+空格"不是 1 token/项。
修法：先发一条小请求标定 ratio，再按 `items = tokens / ratio` 反算。
本测试用假 ask() 覆盖：标定成功 / 标定失败兜底 / ratio 边界，断言生成项数与口径。
退出码：0 = 全过；1 = 有偏差。
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("needle", os.path.join(HERE, "long_uptime_needle.py"))
needle = importlib.util.module_from_spec(spec)
sys.modules["needle"] = needle
spec.loader.exec_module(needle)

fails = 0


def chk(name: str, cond: bool, detail: str = "") -> None:
    global fails
    if cond:
        print(f"✅ {name}")
    else:
        print(f"❌ {name}  {detail}")
        fails += 1


# ① 反算：ratio=1.279、目标 142000 ⇒ 项数 ≈ 111,024（±1）
prompt = needle.make_prompt(142000, 123456, needle.random.Random(7), 1.279)
items = prompt.count(" ") + 1 - 8          # 粗略项数（后段固定文案也会贡献空格，故只做区间检查）
chk("反算项数落在合理区间（≈111k，而不是 142k）", 105000 <= items <= 118000, f"items≈{items}")

# ② 标定成功：假 ask 返回 prompt_tokens = items*1.5 ⇒ ratio 应为 1.5
def fake_ask_ok(port, text, max_tokens):
    n_items = text.count(" ") + 1 - 8
    return "x", 0.1, {"prompt_tokens": int(n_items * 1.5)}


needle.ask = fake_ask_ok
r = needle.measure_ratio(8700)
chk("标定成功时 ratio≈1.5", r is not None and abs(r - 1.5) < 0.05, f"r={r}")

# ③ 标定失败（usage 里没有 prompt_tokens）⇒ 返回 None，主流程用兜底
needle.ask = lambda port, text, max_tokens: ("x", 0.1, {})
chk("标定失败返回 None（主流程会打印警告并用兜底）", needle.measure_ratio(8700) is None)

# ④ 兜底 ratio 必须 >1（否则又会生成 142k 项 ⇒ 偏大）
chk("兜底 ratio > 1.0", needle.FILLER_RATIO_FALLBACK > 1.0, f"{needle.FILLER_RATIO_FALLBACK}")

print()
if fails == 0:
    print("✅ 4/4 用例符合预期")
    sys.exit(0)
print(f"❌ {fails} 个用例不符合预期")
sys.exit(1)
