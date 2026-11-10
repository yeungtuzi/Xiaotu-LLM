#!/usr/bin/env python3
"""test_arms_accuracy.py —— 四臂探针**必须把命中率并入退出码**（台账 #259）。

原实现只校验 cached_tokens 的形态；命中数只打印 ⇒ "unique 答错一半"也 rc=0，
于是"交错 shared/unique 四臂 unique 命中率 100%"这条验收形同虚设。
本测试用假 ask() 覆盖三种情形（不通服务、不占 GPU）：
    ① unique 全对 ⇒ rc=0
    ② unique 有错 ⇒ rc=4（ACC-FAIL）
    ③ 全部无 cached 证据 ⇒ rc=3（ARM-BROKEN，防止 #223 式假通过）
退出码：0 = 全过；1 = 有偏差。
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "probe_degen_arms.py")


def load():
    spec = importlib.util.spec_from_file_location("arms", PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["arms"] = m
    spec.loader.exec_module(m)
    return m


def run_case(name, unique_ok, cached):
    m = load()
    counter = {"i": 0}

    def fake_ask(prompt):
        counter["i"] += 1
        # 用**形态**区分臂（不依赖调用顺序，因为 probe 每轮会 shuffle）：
        #   shared = BASE + doc + Q        ⇒ BASE 之后紧接 "Entry 0:"
        #   unique = BASE + uuid + doc + Q ⇒ BASE 之后是 uuid（十六进制）
        is_unique = not prompt[m.BASE_LEN:].startswith("Entry 0:")
        return {"ok": (unique_ok if is_unique else True),
                "cached": 5 if is_unique else (cached if isinstance(cached, int) else None),
                "prompt_tokens": 100, "sec": 0.01, "out": "x"}

    m.ask = fake_ask
    m.BASE_LEN = len(m.BASE)
    os.environ.update({"PORT": "1", "ROUNDS": "1", "N": "100", "ARM_FILTER": "shared,unique"})
    rc = m.main()
    return rc, name


fails = 0
for name, unique_ok, cached, want in (
        ("unique 全对 ⇒ rc=0", True, 5000, 0),
        ("unique 有错 ⇒ rc=4 ACC-FAIL", False, 5000, 4),
        ("无 cached 证据 ⇒ rc=3 ARM-BROKEN", True, None, 3),
):
    rc, _ = run_case(name, unique_ok, cached)
    if rc == want:
        print(f"✅ {name}（rc={rc}）")
    else:
        print(f"❌ {name}：期望 rc={want} 实得 {rc}")
        fails += 1

print()
if fails == 0:
    print("✅ 3/3 用例符合预期")
    sys.exit(0)
print(f"❌ {fails} 个用例不符合预期")
sys.exit(1)
