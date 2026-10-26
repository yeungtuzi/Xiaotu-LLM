#!/usr/bin/env python3
"""`select_local_topk` 的机械单测（审计 m7：截断/EP 分支原先零覆盖）。

跑法：python3 scripts/test_pin_cap.py      （纯 torch，无 GPU）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from vllm_xiaotu_moe.expert_pool import (  # noqa: E402
    _pin_cap_per_layer, select_local_topk)

FAIL = []


def ck(name, got, want):
    if got != want:
        FAIL.append(f"{name}: got {got} want {want}")
        print(f"  ✗ {name}: got {got} want {want}")
    else:
        print(f"  ✓ {name}")


# ---- 1) 无 EP：cap 按本地(=全局)槽生效，且保持表序 ----
ck("no-ep cap3", select_local_topk([7, 2, 9, 2, 5], None, 64, 3), [7, 2, 9])
ck("no-ep cap0(不限)", select_local_topk([7, 2, 9, 2, 5], None, 64, 0), [7, 2, 9, 5])
ck("no-ep 越界过滤", select_local_topk([100, 3], None, 64, 0), [3])

# ---- 2) EP/TP=2：em 把全局号映射到本地；cap 必须数**本地槽** ----
# 全局 0..3 -> 本地 0..3（rank0），全局 4..7 -> -1（属于 rank1）
em0 = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
# 表序里前 4 条全属 rank1 ⇒ 旧实现(全局截断)会得到空池；新实现应继续往下找
ck("ep 跳过他 rank 继续取", select_local_topk([4, 5, 6, 7, 2, 0], em0, 4, 2), [2, 0])
ck("ep cap 数的是本地槽", select_local_topk([0, 4, 1, 5, 2], em0, 4, 2), [0, 1])
ck("ep 全属他 rank ⇒ 空", select_local_topk([4, 5, 6], em0, 4, 2), [])

# ---- 3) EP 下 cap 不应被"全局条目总数"限制 ----
big = list(range(100))
em1 = torch.tensor([i % 8 for i in range(100)], dtype=torch.int32)
ck("ep cap 取满本 rank", select_local_topk(big, em1, 8, 5), [0, 1, 2, 3, 4])

# ---- 4) cap 环境变量边界 ----
def cap_with(v):
    if v is None:
        os.environ.pop("XIAOTU_PIN_MAX_PER_LAYER", None)
    else:
        os.environ["XIAOTU_PIN_MAX_PER_LAYER"] = v
    return _pin_cap_per_layer()


ck("env 缺省=0", cap_with(None), 0)
ck("env 空=0", cap_with(""), 0)
ck("env 9", cap_with("9"), 9)
ck("env '09' 不误判", cap_with("09"), 9)
ck("env '+9' 不误判", cap_with("+9"), 9)
ck("env 非数字=0", cap_with("abc"), 0)
ck("env 负数=0", cap_with("-3"), 0)
os.environ.pop("XIAOTU_PIN_MAX_PER_LAYER", None)

print()
if FAIL:
    print(f"FAIL ({len(FAIL)} 项)")
    sys.exit(1)
print("ALL PASS — pin cap 单测通过")
