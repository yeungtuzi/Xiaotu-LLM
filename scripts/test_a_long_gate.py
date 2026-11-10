#!/usr/bin/env python3
"""test_a_long_gate.py —— A 臂的长上下文门槛与 ptok 自证。

背景（台账 #224）：验收 A 的默认 `--len 4000`（词）≈ 5K tokens；而退化只在
**长上下文（触发回放/pad）**出现 ⇒ 5K 上的"输出唯一"通过，不能代表 131K 的情形。
本测试用 monkeypatch 替换 ask()，验证门槛逻辑：

  1) ptok=131,000 且 --require-long-tokens=120000  ⇒ 通过
  2) ptok=5,000   且 --require-long-tokens=120000  ⇒ 拒绝（口径不足）
  3) 服务不回 ptok 且给了门槛                      ⇒ 拒绝（无法自证）
  4) 门槛=0（默认，向后兼容）                       ⇒ 通过
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("accept_abc", os.path.join(HERE, "accept_abc.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def run(ptok, require):
    def fake(port, prompt, max_tokens):
        usage = {} if ptok is None else {"prompt_tokens": ptok}
        return "SAME-ANSWER", usage

    mod.ask = fake
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = mod.test_a(port=0, rounds=3, n_words=100, max_tokens=8,
                        require_long_tokens=require)
    return ok, buf.getvalue()


cases = [
    ("ptok=131,000 + 门槛 120,000 ⇒ 通过", run(131000, 120000), True, None),
    ("ptok=5,000 + 门槛 120,000 ⇒ 拒绝（口径不足）", run(5000, 120000), False, "口径不足"),
    ("不回 ptok + 门槛 ⇒ 拒绝（无法自证）", run(None, 120000), False, "无法自证"),
    ("门槛=0（默认）⇒ 通过", run(5000, 0), True, None),
]

fails = 0
for name, (ok, out), want_ok, needle in cases:
    good = (ok is want_ok) and (needle is None or needle in out)
    print(("✅ " if good else "❌ ") + name)
    if not good:
        fails += 1
        print("   实际 ok=%s 输出：%s" % (ok, out.strip().replace("\n", " | ")))
print()
if fails:
    print(f"❌ {fails} 个用例不符合预期")
    sys.exit(1)
print("✅ 4/4 用例符合预期")
