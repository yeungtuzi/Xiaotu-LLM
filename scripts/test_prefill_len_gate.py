#!/usr/bin/env python3
"""test_prefill_len_gate.py —— prefill 对照的"长度一致性"门槛（台账 #227）。

背景：`prefill_perf.py --compare` 比较两条臂的**纯预填充 tok/s 中位数**。但 tok/s 与 prompt 长度
强相关：同样的实现，64K 提示天然比 131K 快。原实现不检查长度 ⇒ 用短提示跑"修后"臂会得到
更高的 tok/s，从而把"换了口径"误判成"性能更好"，95% 判据形同虚设。

四个用例：
  1) 两臂同为 ≈131,773 tokens、比约 97%  ⇒ rc=0（正常通过）
  2) 两臂长度差 51%（131K vs 64K）        ⇒ rc=2（拒绝比较，不给结论）
  3) 两臂同长度、比约 75%                 ⇒ rc=1（真的不达标）
  4) 长度不一致 + --allow-len-mismatch    ⇒ rc=0（显式承担风险）
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "prefill_perf.py")


def write(dir_, name, rows):
    p = os.path.join(dir_, name)
    with open(p, "w", encoding="utf-8") as f:
        for ptok, sec in rows:
            f.write(f"cached=0 ptok={ptok} {sec}s 命中=✗\n")
    return p


def run(before, after, extra=()):
    r = subprocess.run([sys.executable, SCRIPT, "--compare", before, after, *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


with tempfile.TemporaryDirectory() as d:
    b131 = write(d, "b131.txt", [(131774, 120.0), (131773, 122.0)])
    a131 = write(d, "a131.txt", [(131774, 124.0), (131773, 125.0)])   # ≈97%
    s131 = write(d, "s131.txt", [(131774, 160.0), (131773, 162.0)])   # ≈75%
    a64 = write(d, "a64.txt", [(64000, 50.0), (64000, 52.0)])         # 长度差 51%

    cases = [
        ("同长度 ≈97% ⇒ rc=0", run(b131, a131), 0),
        ("长度差 51% ⇒ rc=2（拒绝比较）", run(b131, a64), 2),
        ("同长度 ≈75% ⇒ rc=1（真不达标）", run(b131, s131), 1),
        ("长度不一致 + 显式允许 ⇒ rc=0", run(b131, a64, ["--allow-len-mismatch"]), 0),
    ]

fails = 0
for name, (rc, out), want in cases:
    good = rc == want
    print(("✅ " if good else "❌ ") + f"{name}（rc={rc}）")
    if not good:
        fails += 1
        print("   " + out.strip().replace("\n", "\n   "))
print()
if fails:
    print(f"❌ {fails} 个用例不符合预期")
    sys.exit(1)
print("✅ 4/4 用例符合预期")
