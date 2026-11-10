#!/usr/bin/env python3
"""test_needle_aggregate.py —— 验收 C 的聚合结论**必须进退出码**（台账 #260）。

原实现无论命中率多少都 `return 0` ⇒ "C 未通过"在自动化里等于通过（同 #259 的形态）。
覆盖 5 种情形（纯合成 jsonl，不通服务）：
  ① 5 小时、>4h 段 3/3 命中、ptok≈142K ⇒ 0
  ② >4h 段 1/2 命中 ⇒ 1
  ③ 命中率达标但 ptok=90K ⇒ 2（口径存疑，不许算通过）
  ④ 只有 2 小时数据 ⇒ 4（证据不足）
  ⑤ 记录里没有 ptok 字段 ⇒ 2（无法自证口径）
退出码：0 = 全过；1 = 有偏差。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("needle", os.path.join(HERE, "long_uptime_needle.py"))
needle = importlib.util.module_from_spec(spec)
sys.modules["needle"] = needle
spec.loader.exec_module(needle)

T0 = time.time() - 6 * 3600


def write(rows) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


def case5h(ptok=142000, hits=(True, True, True)):
    rows = []
    # 前 4 小时：3 条（不计入 >4h 判据）
    for i in range(3):
        rows.append({"t": T0 + i * 3600, "needle": 1, "hit": True, "ptok": ptok, "dt": 1.0})
    # >4h 段：按 hits 给
    for i, h in enumerate(hits):
        rows.append({"t": T0 + 4.5 * 3600 + i * 60, "needle": 2, "hit": h, "ptok": ptok, "dt": 1.0})
    return rows


CASES = [
    ("5h + >4h 全中 + 142K ⇒ 0", case5h(), 0),
    ("5h + >4h 一半 ⇒ 1", case5h(hits=(True, False)), 1),
    ("5h + 命中达标但 ptok=90K ⇒ 2", case5h(ptok=90000), 2),
    ("只有 2 小时 ⇒ 4", [{"t": T0 + i * 1800, "needle": 1, "hit": True, "ptok": 142000} for i in range(4)], 4),
    ("无 ptok 字段 ⇒ 2", [{k: v for k, v in r.items() if k != "ptok"} for r in case5h()], 2),
]

fails = 0
for name, rows, want in CASES:
    path = write(rows)
    try:
        rc = needle.aggregate(path)
    finally:
        os.unlink(path)
    if rc == want:
        print(f"✅ {name}（rc={rc}）")
    else:
        print(f"❌ {name}：期望 rc={want} 实得 {rc}")
        fails += 1

print()
if fails == 0:
    print(f"✅ {len(CASES)}/{len(CASES)} 用例符合预期")
    sys.exit(0)
print(f"❌ {fails} 个用例不符合预期")
sys.exit(1)
