#!/usr/bin/env python3
"""SWA bounded replay：**CPU 侧 gather_lens 与 GPU 侧公式必须一致**（纯计算，不需要 GPU/服务）。

背景（见 `dev-docs/BUG_replay_start_zero_sentinel.md` §6）：

* GPU（Triton `ComputePrefillMetadataKernel`，`sparse_swa.py:370-372`）：

      visible_prefix = max(prefix_len - replay_start, 0)
      gather_len     = query_len + min(visible_prefix, window - 1)

* CPU（`_build_deepseek_v4_metadata`，`sparse_swa.py:938-951`）**当前实现**：

      gather_len_cpu = query_len + min(prefix_len, window - 1)      ← 少了 replay_start

`flashmla.py:934` 会强制使用该 CPU 张量，所以两边不一致就是生产路径上的真实缺陷。

⚠️ **2026-11-12 更正**：按调度器真实不变量
（`replay_start = max(0, num_hit - window)` 且 `prefix_len >= num_hit`），
两个公式**恒等**：`num_hit <= window` 时 `replay_start = 0`；`num_hit > window` 时
`prefix_len - replay_start >= window`，两侧都被 `window-1` 截断。
⇒ 本脚本现在的用途是**守住这个恒等关系**（回归测试），而不是复现某个缺陷。

* `--expect equal`（默认）：断言“在真实不变量下两侧全部一致”；
* `--expect diverge`：用**违反不变量**的输入（`prefix_len < num_hit`）演示“形状不同会分叉”，
  仅用于说明为什么必须代入约束验算，**不代表生产状态**。

用法：
    python3 scripts/test_swa_gather_lens_parity.py                  # 回归：必须一致
    python3 scripts/test_swa_gather_lens_parity.py --expect diverge # 演示反例
退出码：0 = 符合预期；1 = 不符合。
"""
from __future__ import annotations

import argparse
import sys

WINDOW = 128


def gpu_formula(prefix_len: int, query_len: int, replay_start: int,
                window: int = WINDOW) -> int:
    """GPU 侧语义（应当的样子）：`replay_start` 先夹到 >=0（-1 = 无重放），再算可见前缀。"""
    visible = max(prefix_len - max(replay_start, 0), 0)
    return query_len + min(visible, window - 1)


def cpu_formula_broken(prefix_len: int, query_len: int, replay_start: int,
                       window: int = WINDOW) -> int:
    """当前生产实现：**没有减 replay_start**。"""
    del replay_start          # 故意的：这就是缺陷所在
    return query_len + min(prefix_len, window - 1)


def cpu_formula_fixed(prefix_len: int, query_len: int, replay_start: int,
                      window: int = WINDOW) -> int:
    """修好之后应当的样子：先减（并夹到 0），再夹上界。"""
    visible = max(prefix_len - max(replay_start, 0), 0)
    return query_len + min(visible, window - 1)


# 【不变量一致】用例：(命中 token 数, 前缀里额外已算 token 数, query 长度)
#   prefix_len = num_hit + extra ;  replay_start = max(0, num_hit - WINDOW)
CASES = [
    (0, 0, 8192),          # 无命中
    (0, 4096, 8192),       # 无命中但有已算前缀
    (1, 0, 8192),          # 极小命中
    (126, 0, 8192),
    (127, 0, 8192),
    (128, 0, 8192),        # 恰好一个窗口
    (129, 0, 8192),        # 刚过窗口
    (4096, 0, 8192),
    (131072, 0, 8192),     # 长命中（本项目的主力场景）
    (131072, 4096, 8192),
    (131072, 100000, 8192),
]

# 【反例】违反不变量的输入（prefix_len < num_hit）：仅用于演示“必须代入约束验算”
DIVERGE_CASES = [
    (131072, 8192, 131072),   # prefix_len == replay_start，真实不变量下不可能出现
    (5000, 8192, 5000),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", choices=("equal", "diverge"), default="equal")
    a = ap.parse_args()

    if a.expect == "diverge":
        cases = list(DIVERGE_CASES)          # 已是 (prefix_len, query_len, replay_start)
    else:
        # (num_hit, extra, query) → (prefix_len, query_len, replay_start)
        cases = [(nh + extra, q, max(0, nh - WINDOW)) for nh, extra, q in CASES]
    mism = 0
    print(f"{'prefix':>8} {'query':>7} {'rstart':>8} | {'GPU':>7} {'CPU(cur)':>9} {'CPU(fix)':>9} | 状态")
    for prefix, query, rs in cases:
        g = gpu_formula(prefix, query, rs)
        c = cpu_formula_broken(prefix, query, rs)
        f = cpu_formula_fixed(prefix, query, rs)
        same = g == c
        fixed_ok = g == f
        if not same:
            mism += 1
        flag = "一致" if same else "❌ 不一致"
        print(f"{prefix:>8} {query:>7} {rs:>8} | {g:>7} {c:>9} {f:>9} | {flag}"
              f"{'' if fixed_ok else ' / 修法本身也不对!'}")
        if not fixed_ok:
            print("  ⛔ 修复公式与 GPU 语义不等价，实现有误")
            return 1

    print()
    print(f"不一致的组数：{mism}/{len(cases)}")
    if a.expect == "equal":
        ok = mism == 0
        print("== 结论：" + ("真实不变量下两侧恒等 ✅" if ok else
                             f"仍有 {mism} 组不一致 ❌ —— 不变量被破坏或公式被改动"))
    else:
        ok = mism > 0
        print("== 结论：" + ("已演示：违反不变量时两侧会分叉 ✅（仅作反例）" if ok else
                             "未演示出分叉 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
