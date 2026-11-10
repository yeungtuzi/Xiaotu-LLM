#!/usr/bin/env python3
"""prove_padding_mismatch.py —— 用**纯 Python** 演示"捕获时算一次"的映射为何在真步上错位。

不需要 torch / GPU：这里只复现 `compute_token_to_req_indices` 的**语义**
（按 `query_start_loc` 把前 N 个 token 映射到请求号；越界部分映射到 0）。

场景（与 decoder-side SWA bounded replay 的实际情况同构）：
  * **捕获时**：dummy batch = 1 个请求、8 个 token ⇒ 映射全 0；
  * **真实步**：2 个请求各 3 个 token，另有 2 行为 padding ⇒ 正确映射应为
    [0,0,0,1,1,1,·,·]（padding 行必须是"无效"，而不是"属于请求 0"）。

若仍用捕获时的旧映射，req1 的三行会被当成 req0 的行 ⇒ SWA 读端按 **req0 的 slot** 取数
⇒ 读到别人的 KV。这就是"同一进程里越跑越错"的机制。

退出码：0 = 演示成功（确实能构造出差异）；1 = 未构造出差异（说明前提写错了）
"""
from __future__ import annotations

import sys


def map_tokens(query_start_loc: list[int], num_tokens: int, num_mapped: int) -> list[int]:
    """按 vLLM `_token_request_mapping_kernel` 的语义做映射。"""
    out = [0] * num_tokens
    for req in range(len(query_start_loc) - 1):
        start, end = query_start_loc[req], query_start_loc[req + 1]
        for tok in range(start, min(end, num_mapped)):
            out[tok] = req
    return out


def main() -> int:
    # ① 捕获时（dummy batch：1 请求 × 8 token）
    cap_qsl = [0, 8]
    cap_map = map_tokens(cap_qsl, num_tokens=8, num_mapped=8)

    # ② 真实步（2 请求 × 3 token，另有 2 行 padding；buffers 按 8 行分配）
    real_qsl = [0, 3, 6]
    real_map = map_tokens(real_qsl, num_tokens=8, num_mapped=6)

    # ③ 用了旧映射的后果：buffer 里仍是捕获时的内容
    stale = cap_map + [0] * 0

    print("捕获时映射      :", cap_map)
    print("真实步应有映射  :", real_map)
    print("实际用的旧映射  :", stale[:8])
    diff = [i for i in range(6) if real_map[i] != stale[i]]
    print("前 6 行中错位的行:", diff)
    if not diff:
        print("❌ 未构造出差异：说明这组参数不触发错位（换个形状再试）")
        return 1
    print()
    print("⇒ 结论：req1 的 token（第 %s 行）在旧映射下被当作 **req0** 的行，"
          "SWA 读端于是按 req0 的 slot 取 KV。" % diff)
    print("⇒ 这正是 `slot_mapping` 每步重算（D1）要解决的问题：")
    print("   `torch.ge(slot_mapping, 0, out=is_valid_token)` 让 padding 行变成无效，")
    print("   `compute_token_to_req_indices(...)` 让行→请求的映射与本步一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
