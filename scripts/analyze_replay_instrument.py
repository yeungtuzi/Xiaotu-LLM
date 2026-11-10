#!/usr/bin/env python3
"""解析 `[xtu-swa-bounded-replay]` 仪器输出，按"第 N 个带重放的 prefill 步"列表。

对应仪器补丁 `patches/diag/decoder_swa_bounded_replay_trigger_dbg.patch`（v3），
它每步打印：

    [xtu-swa-bounded-replay] reqs=R prefilling=[...] qlen=[...] replay_start=[...] \
        hit=0|1 pad_added=N pads_total=M groups=[...]

E1 的判据就是拿**第 1 条**与**第 8 条**长请求的这一行做对比（`hit` / `replay_start` /
`pad_added` 是否变了）。本脚本只读日志，不碰服务。

用法：
    python scripts/analyze_replay_instrument.py /home/user/lvllm/run/ds41_prod.log
    python scripts/analyze_replay_instrument.py <log> --only-hit   # 只看 hit=1 的 prefill 步
"""

from __future__ import annotations

import argparse
import re
import sys

LINE = re.compile(
    r"\[xtu-swa-bounded-replay\] reqs=(?P<reqs>\d+) "
    r"prefilling=\[(?P<prefill>[^\]]*)\] "
    r"qlen=\[(?P<qlen>[^\]]*)\] "
    r"computed=\[(?P<computed>[^\]]*)\] "
    r"replay_start=\[(?P<start>[^\]]*)\] "
    r"hit=(?P<hit>\d+) "
    r"pad_added=(?P<pad>-?\d+) "
    r"pads_total=(?P<total>\d+) "
    r"groups=\[(?P<groups>[^\]]*)\]"
)


def verdict(path: str) -> int:
    """按 E1 三条判据给结论（只读日志）。

    判据：① `hit == 1`；② `computed == replay_start`；③ `pad_added == 128`。
    """
    steps = []
    with open(path, errors="replace") as f:
        for raw in f:
            m = LINE.search(raw)
            if not m:
                continue
            g = m.groupdict()
            if g["hit"] != "1" or "1" not in g["prefill"]:
                continue
            steps.append(g)
    if not steps:
        print("⛔ 没有 hit=1 的 prefill 步 ⇒ 要么没有命中，要么仪器没开")
        return 1
    print(f"共 {len(steps)} 个重放步；逐条按三条判据检查：")
    bad = []
    skipped_chunks = 0
    skipped_multi = 0
    for i, g in enumerate(steps, 1):
        comp = [x.strip() for x in g["computed"].split(",") if x.strip()]
        start = [x.strip() for x in g["start"].split(",") if x.strip()]
        pad = int(g["pad"])
        idxs = [j for j, p in enumerate(g["prefill"].split(",")) if p.strip() == "1"]
        # ⚠️ 同一行里有 ≥2 个请求在 prefill 时，`pad_added` 是**多请求之和**，
        #    无法归因到某一个 replay_start ⇒ 这一行不能用于 pad 语义判定，必须跳过（否则假告警）。
        if len(idxs) > 1:
            skipped_multi += 1
            continue
        for j in idxs:
            c = comp[j] if j < len(comp) else "?"
            st = start[j] if j < len(start) else "?"
            try:
                ci, si = int(c), int(st)
            except ValueError:
                bad.append((i, j, f"字段不可解析 computed={c} replay_start={st}"))
                continue
            # ⚠️ 关键：长 prompt 会被**分块 prefill** ⇒ 同一请求出现在多个步里。
            #   只有"首个 chunk"带重放与 pad：那一步 computed == replay_start 且 pad_added == 128。
            #   后续 chunk 的 computed > replay_start、pad_added == 0 属于**正常**，必须跳过，
            #   否则会在真权重上产生假告警（本脚本的 dummy 演练实测踩到，见台账 #158）。
            if pad < 0:
                bad.append((i, j, "pad_added=-1 ⇒ 仪器自身测失败（dbg-failed 或计数异常）"))
                continue
            if ci == si:
                # 期望 pad 数 = min(窗口, 命中长度)：命中不足一个窗口时，重放行数就是命中长度。
                # 固定写 128 会在"短命中"（生产并发下很常见）时产生假告警。
                want = min(128, si)
                if pad != want:
                    bad.append((i, j, f"重放步 pad_added={pad} != min(128, replay_start)={want} ⇒ pad 语义"))
            elif ci > si:
                skipped_chunks += 1
                if pad != 0:
                    bad.append((i, j, f"非首个 chunk 却发生 pad（pad_added={pad}）⇒ pad 语义"))
            else:
                bad.append((i, j, f"computed({ci}) < replay_start({si}) ⇒ 调度侧重绕"))
    if not bad:
        print("✅ 三条判据全满足：hit=1 / computed == replay_start / pad_added=min(128, replay_start)"
              f"（另外跳过 {skipped_chunks} 个'后续 chunk'，属正常分块 prefill"
              + (f"；跳过 {skipped_multi} 个多请求步，pad 无法归因" if skipped_multi else "") + "）")
        return 0
    for i, j, why in bad[:12]:
        print(f"❌ 第 {i} 个重放步（批内第 {j} 个请求）：{why}")
    print(f"\n合计 {len(bad)} 处异常 ⇒ 按上面的指向继续查（见 PLAN §22）")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--only-hit", action="store_true", help="只看 hit=1 的 prefill 步")
    ap.add_argument("--tail", type=int, default=0, help="只显示最后 N 条（0=全部）")
    ap.add_argument("--verdict", action="store_true",
                    help="只对 hit=1 的 prefill 步按三条判据给结论（健康/异常+指向）")
    a = ap.parse_args()

    rows = []
    with open(a.log, errors="replace") as f:
        for raw in f:
            m = LINE.search(raw)
            if not m:
                continue
            prefill = [x.strip() for x in m.group("prefill").split(",") if x.strip()]
            if a.only_hit and (m.group("hit") != "1" or "1" not in prefill):
                continue
            rows.append(m.groupdict())

    if a.verdict:
        return verdict(a.log)

    if not rows:
        print(f"⛔ 没在 {a.log} 里找到仪器输出（触发文件是否存在？补丁是否应用？）")
        return 1

    if a.tail and len(rows) > a.tail:
        rows = rows[-a.tail :]

    print(f"{'#':>4}  {'reqs':>4}  {'prefilling':<12} {'qlen':<10} {'computed':<14} "
          f"{'replay_start':<14} {'hit':>3} {'pad_added':>9} {'pads_total':>10}")
    for i, r in enumerate(rows, 1):
        prefill = r["prefill"].replace(" ", "")
        print(f"{i:>4}  {r['reqs']:>4}  {prefill:<12} {r['qlen']:<10} "
              f"{r.get('computed', '?'):<14} {r['start']:<14} "
              f"{r['hit']:>3} {r['pad']:>9} {r['total']:>10}")
    print(f"\n合计 {len(rows)} 条仪器步；判据：第 1 条与第 8 条长请求的 "
          f"hit / replay_start / pad_added 若不一致，就是 E1 要找的差异。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
