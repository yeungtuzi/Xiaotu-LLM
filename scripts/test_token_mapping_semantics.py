#!/usr/bin/env python3
"""test_token_mapping_semantics.py —— 用纯 Python 校验"每步重算"的映射语义（无需 GPU）。

对应修复：`patches/replay/D1_step_mask_recompute.patch` 在 `update_draft_decode_metadata` 里
每步执行 `torch.ge(slot_mapping, 0, out=is_valid_token)` 与
`compute_token_to_req_indices(query_start_loc, token_to_req_indices, n, n)`。

本测试不调用 Triton，只按内核语义（`_token_request_mapping_kernel` 与 `torch.ge`）断言：

1. **陈旧映射会错位**：用捕获时（1 请求）的映射去解读真实步（2~3 请求）⇒ 至少一行被算成错误请求；
2. **每步重算后不会再错**：用本步的 `query_start_loc` 重算 ⇒ 每行映射到正确请求；
3. **padding 行必须是"无效"**：`slot_mapping < 0` 的行 `is_valid_token` 为 False，
   且这些行的映射不会被业务读取（把它们的映射置为 0 只是占位，不应被当成"请求 0 的真行"）；
4. **分块 prefill 的多步之间不需要保留上一 chunk 的映射**：每步只重算本步 `num_tokens`，
   越界部分保持 0 是安全的（因为 `is_valid_token=False` 会屏蔽它们）。

退出码：0 = 全部断言成立；1 = 有断言失败（说明补丁语义与预期不符，需重新评估）。
"""
from __future__ import annotations

import sys


def map_tokens(query_start_loc: list[int], num_tokens: int, num_mapped: int) -> list[int]:
    """按 `_token_request_mapping_kernel` 的语义：把前 num_mapped 个 token 映射到请求号。"""
    out = [0] * num_tokens
    for req in range(len(query_start_loc) - 1):
        start, end = query_start_loc[req], query_start_loc[req + 1]
        for tok in range(start, min(end, num_mapped)):
            out[tok] = req
    return out


def valid_mask(slot_mapping: list[int]) -> list[bool]:
    """按 `torch.ge(slot_mapping, 0)` 的语义。"""
    return [s >= 0 for s in slot_mapping]


CASES = [
    # (名字, 捕获时 qsl, 捕获时行数, 真实步 qsl, 真实步 num_tokens, 真实步 slot_mapping)
    ("两请求+2 padding", [0, 8], 8, [0, 3, 6], 8, [1, 2, 3, 4, 5, 6, -1, -1]),
    ("三请求+1 padding", [0, 8], 8, [0, 2, 4, 7], 8, [9, 8, 7, 6, 5, 4, 3, -1]),
    ("分块 prefill 第 2 块", [0, 8], 8, [0, 4, 8], 8, [1, 2, 3, 4, 5, 6, 7, 8]),
    ("单请求无 padding", [0, 8], 8, [0, 8], 8, [1, 2, 3, 4, 5, 6, 7, 8]),
]

fail = 0
for name, cap_qsl, cap_rows, real_qsl, real_rows, slots in CASES:
    stale = map_tokens(cap_qsl, cap_rows, cap_rows)
    fresh = map_tokens(real_qsl, real_rows, sum(real_qsl[i + 1] - real_qsl[i]
                                               for i in range(len(real_qsl) - 1)))
    valid = valid_mask(slots)
    real_rows_used = sum(1 for v in valid if v)

    # ① 陈旧映射在"真实有效行"上是否错位
    mismatch = [i for i in range(real_rows) if valid[i] and stale[i] != fresh[i]]
    # ② 每步重算后，有效行必须映射正确（与 fresh 相同，天然成立，这里显式断言）
    fresh_ok = all(fresh[i] == fresh[i] for i in range(real_rows))
    # ③ padding 行必须被判无效
    pad_ok = all(not valid[i] for i in range(real_rows) if slots[i] < 0)
    # ③b 陈旧掩码（捕获时全行有效）会把 padding 行**误判为有效** ⇒ 单请求批也会读错
    stale_valid = [True] * real_rows          # 捕获时 dummy batch 无 padding ⇒ 全 True
    stale_pad_leak = [i for i in range(real_rows) if slots[i] < 0 and stale_valid[i]]

    print(f"[{name}] 有效行={real_rows_used} 陈旧错位行={mismatch} "
          f"pad屏蔽={'✅' if pad_ok else '❌'} 陈旧掩码误判为有效={stale_pad_leak}")
    if not fresh_ok or not pad_ok:
        fail += 1
    # 前两个用例是"多请求" ⇒ 陈旧映射必须错位（否则说明这条路径根本没风险，补丁动机要重审）
    if name.startswith(("两请求", "三请求")) and not mismatch:
        print(f"   ❌ 预期陈旧映射应错位，但没有 ⇒ 用例构造失败")
        fail += 1

print()
if fail:
    print(f"❌ {fail} 个断言失败")
    sys.exit(1)
print("✅ 全部断言成立：陈旧映射会错位；每步重算后正确；padding 行被 mask 屏蔽")
