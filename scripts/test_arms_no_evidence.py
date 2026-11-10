#!/usr/bin/env python3
"""test_arms_no_evidence.py —— 四臂探针在"没有 cached 证据"时必须作废（不能被 100% 骗过）。

背景（台账 #225）：`probe_degen_arms.py` 原来的四个 cached 检查都写成 `... and c`，
当 cached_tokens 全为 None（既没回 usage、/metrics 也取不到）时检查全部静默跳过，
于是四个臂"命中率 100%"照常打印、退出码 0 ⇒ 与 #223 是同一个假通过通道。

用法： python3 scripts/test_arms_no_evidence.py
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("probe_degen_arms", os.path.join(HERE, "probe_degen_arms.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(cached_value):
    """让所有臂都"命中"，cached 由参数决定（None = 无证据）。"""
    def fake_ask(prompt):
        return {"ok": True, "cached": cached_value, "prompt_tokens": 1200,
                "sec": 1.0, "out": "ok"}

    mod.ask = fake_ask
    mod.ROUNDS = 1
    mod.N = 400          # N//4 = 100，用于 cached 的高/低判定
    mod.ARM_FILTER = []
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main()
    return rc, buf.getvalue()


cases = []

rc, out = run(None)                       # 完全无证据
cases.append(("cached 全 None（无证据）⇒ 作废 rc=3", rc == 3 and "缺少 cached_tokens 证据" in out, rc, out))

rc, out = run(1000)                       # shared/tail 高、unique 也高 ⇒ unique 应当被判 broken
cases.append(("cached 全部很高 ⇒ unique 臂被判 ARM-BROKEN rc=3", rc == 3 and "ARM-BROKEN" in out, rc, out))


def run_healthy():
    """按臂区分 cached：shared/tail 高、unique 低、suffix 高 ⇒ 应当 rc=0（证明门槛不会过度开火）。"""
    per_arm = {"shared": 900, "tail": 900, "unique": 10, "suffix": 900}
    mod.build = lambda arm, rnd, u: arm          # 用臂名当 prompt，便于映射
    # warm 阶段的 ask 收到的是真实 prompt（不是臂名）⇒ 用 per_arm.get(prompt, 900) 兜底
    mod.ask = lambda prompt: {"ok": True, "cached": per_arm.get(prompt, 900),
                              "prompt_tokens": 1200, "sec": 1.0, "out": "ok"}
    mod.ROUNDS = 1
    mod.N = 400
    mod.ARM_FILTER = []
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main()
    return rc, buf.getvalue()


rc, out = run_healthy()
cases.append(("按臂健康分布（shared/tail 高、unique 低）⇒ rc=0（门槛不过度开火）", rc == 0, rc, out))

fails = 0
for name, good, rc, out in cases:
    print(("✅ " if good else "❌ ") + name)
    if not good:
        fails += 1
        print("   rc=%s" % rc)
        print("   " + out.strip().replace("\n", "\n   "))
print()
if fails:
    print(f"❌ {fails} 个用例不符合预期")
    sys.exit(1)
print("✅ 3/3 用例符合预期（其中 1 条是历史漏洞：无证据时原本会 rc=0）")
