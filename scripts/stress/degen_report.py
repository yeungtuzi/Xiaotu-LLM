#!/usr/bin/env python3
"""degen_report.py —— 从 replay.jsonl 里抽出所有"退化故障点"，生成一页可交付的 markdown。

用法:
    python3 degen_report.py [replay.jsonl] [-o 输出.md]

判定（任一命中即算一个退化点）:
    repeat_ratio > 0.6          # 高重复
    garbage_ratio > 0           # 乱码
    reasoning_only 且 finish_reason == "length"   # 只有思考、撞长度上限（最常见）

输出: 概览（按上下文规模分桶） + 明细表（时间/轮次/prompt tokens/重复率/正文与思考摘录）。
"""
import json
import sys
import os
from collections import Counter

THRESH = 0.6


def load(path):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "repeat_ratio" in d:
                recs.append(d)
    return recs


def is_degen(d):
    if (d.get("repeat_ratio") or 0) > THRESH:
        return True
    if (d.get("garbage_ratio") or 0) > 0:
        return True
    return bool(d.get("reasoning_only")) and d.get("finish_reason") == "length"


def short(s, n):
    s = " ".join((s or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.expanduser(
        "~/lvllm/stress-run4/replay.jsonl")
    out = os.path.expanduser("~/lvllm/stress-run4/degeneration.md")
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]

    recs = load(src)
    degens = [d for d in recs if is_degen(d)]

    lines = []
    lines.append("# 长上下文退化故障点汇总\n")
    lines.append(f"- 来源: `{src}`")
    lines.append(f"- 有效应答: **{len(recs)}** 条；退化点: **{len(degens)}** 条"
                 f"（{100.0 * len(degens) / len(recs):.1f}%）" if recs else "- 无数据")
    lines.append("")

    # 形态分布
    forms = Counter()
    for d in degens:
        if (d.get("garbage_ratio") or 0) > 0:
            forms["乱码"] += 1
        elif d.get("reasoning_only") and d.get("finish_reason") == "length":
            forms["只有思考 + 撞长度上限"] += 1
        elif (d.get("repeat_ratio") or 0) > THRESH:
            forms["高重复（有正文）"] += 1
        else:
            forms["其它"] += 1
    if forms:
        lines.append("## 形态分布\n")
        lines.append("| 形态 | 条数 |")
        lines.append("|---|---|")
        for k, v in forms.most_common():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # 按上下文规模分桶
    if degens:
        buckets = Counter()
        for d in degens:
            pt = d.get("prompt_tokens") or 0
            buckets[(pt // 50000) * 50] += 1
        lines.append("## 按 prompt 规模（k tokens）\n")
        lines.append("| prompt 规模 | 退化条数 |")
        lines.append("|---|---|")
        for k in sorted(buckets):
            lines.append(f"| {k}k–{k + 50}k | {buckets[k]} |")
        lines.append("")

    # 明细
    lines.append("## 明细（最近 40 条）\n")
    lines.append("| # | 时间(UTC) | 轮次 | prompt | 完成 | 形态 | 重复率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, d in enumerate(degens[-40:], 1):
        kind = ("乱码" if (d.get("garbage_ratio") or 0) > 0
                else "思考撞上限" if (d.get("reasoning_only") and d.get("finish_reason") == "length")
                else "高重复")
        lines.append(f"| {i} | {d.get('ts')} | {d.get('turn')} | {d.get('prompt_tokens')} | "
                     f"{d.get('completion_tokens')} | {kind} | {d.get('repeat_ratio')} |")
    lines.append("")

    # 摘录（只给最后 6 条，避免文件过长）
    tail = [d for d in degens if d.get("reasoning_tail") or d.get("content_head")][-6:]
    if tail:
        lines.append("## 摘录（思考尾部，判断是否在循环）\n")
        for d in tail:
            lines.append(f"**{d.get('ts')} / turn {d.get('turn')} / prompt {d.get('prompt_tokens')}**\n")
            if d.get("reasoning_tail"):
                lines.append("> " + short(d["reasoning_tail"], 320) + "\n")
            elif d.get("content_head"):
                lines.append("> " + short(d["content_head"], 320) + "\n")
            lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"写出 {out}（退化点 {len(degens)} / 有效 {len(recs)}）")


if __name__ == "__main__":
    main()
