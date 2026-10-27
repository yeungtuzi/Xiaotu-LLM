#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比对两份 PyTorch 显存快照：按**调用栈**聚合存活块，指出"哪个栈在长"。

输入是 `torch.cuda.memory._dump_snapshot()` 落下的 pickle（由插件在
`XIAOTU_MEM_TRACE=1` 时定时产生）。输出两类信息：

    A. 现存活块 Top-N（按字节）—— 现在谁最占显存
    B. 两份快照的**差值** Top-N —— 从 T1 到 T2 **谁在长**

用法：
    python ops/a100-stress/mem_snapshot_diff.py snapA.pickle snapB.pickle
    python ops/a100-stress/mem_snapshot_diff.py snapA.pickle            # 只看 Top-N
"""
from __future__ import annotations

import collections
import pickle
import sys


def _frame_key(frames, depth: int = 6) -> str:
    """把调用栈压成一个可聚合的键：取最近的 `depth` 帧（filename:line name）。"""
    if not frames:
        return "<no-stack>"
    # PyTorch 的 frames 顺序：最近的调用在前
    parts = []
    for fr in frames[:depth]:
        if isinstance(fr, dict):
            fn = fr.get("filename", "?")
            ln = fr.get("line", 0)
            nm = fr.get("name", "?")
        else:                                     # 兼容 (filename, line, name) 元组
            fn, ln, nm = (list(fr) + ["?", 0, "?"])[:3]
        parts.append(f"{fn.split('/')[-1]}:{ln} {nm}")
    return " <- ".join(parts)


def aggregate(path: str) -> tuple[collections.Counter, int]:
    with open(path, "rb") as f:
        snap = pickle.load(f)
    by_stack: collections.Counter = collections.Counter()
    total = 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = int(blk.get("size", 0))
            total += size
            by_stack[_frame_key(blk.get("frames"))] += size
    return by_stack, total


def show(title: str, counter: collections.Counter, total: int, topn: int = 15) -> None:
    print(f"\n=== {title} (存活合计 {total / 2**30:.3f} GiB) ===")
    for stack, nb in counter.most_common(topn):
        print(f"{nb / 2**30:9.3f} GiB  {stack}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    a, ta = aggregate(sys.argv[1])
    show(f"快照 A: {sys.argv[1]}", a, ta)
    if len(sys.argv) >= 3:
        b, tb = aggregate(sys.argv[2])
        show(f"快照 B: {sys.argv[2]}", b, tb)
        diff = collections.Counter()
        for k in set(a) | set(b):
            diff[k] = b.get(k, 0) - a.get(k, 0)
        grow = collections.Counter({k: v for k, v in diff.items() if v > 0})
        shrink = collections.Counter({k: -v for k, v in diff.items() if v < 0})
        print(f"\n=== 增量 B−A：总变化 {((tb - ta) / 2**30):+.3f} GiB ===")
        print("--- 增长 Top ---")
        for stack, nb in grow.most_common(15):
            print(f"{nb / 2**30:+9.3f} GiB  {stack}")
        print("--- 减少 Top ---")
        for stack, nb in shrink.most_common(8):
            print(f"{-nb / 2**30:+9.3f} GiB  {stack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
