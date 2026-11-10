#!/usr/bin/env python3
"""test_verdict_branches.py —— 校验 `analyze_replay_instrument.py --verdict` 的**每个分支**。

判据（台账 #158/#176）：
  * `computed == replay_start`  ⇒ 重放步，**必须** `pad_added == 128`
  * `computed  > replay_start`  ⇒ 后续 chunk，**必须** `pad_added == 0`（有 pad 才是异常），且不参与判定
  * `computed  < replay_start`  ⇒ 异常（调度重绕）

本测试用**合成日志**覆盖 9 种形态，断言退出码与结论文字。无需 GPU / 无需服务。
退出码：0 = 9 个分支都符合预期；1 = 有偏差。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "analyze_replay_instrument.py")

LINE = ("[xtu-swa-bounded-replay] reqs={reqs} prefilling={pf} qlen={ql} "
        "computed={comp} replay_start={st} hit={hit} pad_added={pad} "
        "pads_total=100 groups=[4]")

CASES = [
    # (名字, 行内容, 期望 rc, 期望结论关键词)
    ("健康：首个 chunk + pad 128",
     LINE.format(reqs=1, pf="[1]", ql="[212]", comp="[9216]", st="[9216]", hit=1, pad=128), 0,
     "三条判据全满足"),
    ("后续 chunk（pad=0）应被跳过",
     LINE.format(reqs=1, pf="[1]", ql="[8178]", comp="[16242]", st="[8064]", hit=1, pad=0), 0,
     "三条判据全满足"),
    ("首个 chunk 却没有 pad ⇒ 异常",
     LINE.format(reqs=1, pf="[1]", ql="[212]", comp="[9216]", st="[9216]", hit=1, pad=0), 1,
     "pad 语义"),
    ("后续 chunk 却发生 pad ⇒ 异常",
     LINE.format(reqs=1, pf="[1]", ql="[8178]", comp="[16242]", st="[8064]", hit=1, pad=128), 1,
     "非首个 chunk 却发生 pad"),
    ("短命中：replay_start=64 ⇒ pad 期望 64",
     LINE.format(reqs=1, pf="[1]", ql="[128]", comp="[64]", st="[64]", hit=1, pad=64), 0,
     "三条判据全满足"),
    ("短命中却按 128 pad ⇒ 异常",
     LINE.format(reqs=1, pf="[1]", ql="[128]", comp="[64]", st="[64]", hit=1, pad=128), 1,
     "pad 语义"),
    ("computed < replay_start ⇒ 调度重绕",
     LINE.format(reqs=1, pf="[1]", ql="[212]", comp="[1024]", st="[9216]", hit=1, pad=128), 1,
     "调度侧重绕"),
    # —— 2026-11-10 新增两类分支（台账 #238）——
    ("多请求同行：pad 无法归因 ⇒ 跳过（不判异常）",
     LINE.format(reqs=2, pf="[1, 1]", ql="[212, 64]",
                 comp="[9216, 9216]", st="[9216, 0]", hit=1, pad=128), 0,
     "多请求步"),
    ("pad_added=-1（仪器自测失败）⇒ 异常",
     LINE.format(reqs=1, pf="[1]", ql="[212]", comp="[9216]", st="[9216]", hit=1, pad=-1), 1,
     "仪器自身测失败"),
]


def run_one(line: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(line + "\n")
        path = fh.name
    try:
        p = subprocess.run([sys.executable, TOOL, path, "--verdict"],
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr
    finally:
        os.unlink(path)


fail = 0
for name, line, want_rc, want_kw in CASES:
    rc, out = run_one(line)
    ok = (rc == want_rc) and (want_kw in out)
    print(f"{'✅' if ok else '❌'} {name}  rc={rc}（期望 {want_rc}）  关键词={want_kw!r}")
    if not ok:
        print("   ---- 实际输出 ----")
        print("   " + out.strip().replace("\n", "\n   "))
        fail += 1

print()
if fail:
    print(f"❌ {fail} 个分支不符合预期")
    raise SystemExit(1)
print(f"✅ {len(CASES)} 个分支全部符合预期（健康/跳过后继 chunk/缺 pad/多余 pad/调度重绕/多请求跳过/仪器失败）")
