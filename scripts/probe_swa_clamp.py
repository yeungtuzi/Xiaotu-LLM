#!/usr/bin/env python3
"""SWA 窗口 clamp 的局部 probe（不启动服务、不加载权重）。

目的：回答一个可判定的问题 —— **我们部署树里的 Triton SWA indices kernel，
在 `replay_start > 0`（前缀命中 + decoder-side SWA bounded replay）时，
是否真的把 `start_pos` clamp 到 `replay_start`，从原理上不读窗口以下的 KV？**

做法：直接调用 `vllm/v1/attention/backends/mla/sparse_swa.py::_compute_swa_indices_and_lens_kernel`，
用手工构造的 block_table / seq_lens / query_start_loc / replay_start，检查输出的
`swa_indices`（slot id）与 `swa_lens`：

* 断言 1（有 clamp）：任一 token 的 `start_pos >= replay_start`，即
  `swa_lens[i] == min(pos+1, ...) - max(pos+1-window, replay_start)`；
* 断言 2（slot 合法）：所有 `>= 0` 的 slot id 反解出的 position 都 `>= replay_start`；
* 断言 3（无 clamp 对照）：把 `replay_start = 0` 跑一遍，slot 会覆盖到 `replay_start` 以下
  —— 证明"若 clamp 失效则确实会读到窗口以下"，也就是上游 SM8x 注释所说的 fault 源。

用法（远端 GPU）：
    /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python scripts/probe_swa_clamp.py
退出码：0 = 三条断言全部成立；1 = 有不成立项（打印细节）。
"""

from __future__ import annotations

import sys

import torch

WINDOW = 128
BLOCK_SIZE = 32


def build_case(num_reqs: int = 2, seq_len: int = 4096, query_len: int = 1):
    """构造 decode 态的输入：每请求 1 个查询 token，位置 = seq_len - 1。"""
    device = torch.device("cuda")
    # 每请求的块表：连续映射，块号从 req * 1000 开始，便于反解归属
    blocks_per_req = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.zeros((num_reqs, blocks_per_req), dtype=torch.int32, device=device)
    for r in range(num_reqs):
        block_table[r] = torch.arange(
            r * 1000, r * 1000 + blocks_per_req, dtype=torch.int32, device=device
        )
    query_start_loc = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
    token_to_req = torch.arange(0, num_reqs, dtype=torch.int32, device=device)
    is_valid = torch.ones((num_reqs,), dtype=torch.bool, device=device)
    return block_table, query_start_loc, seq_lens, token_to_req, is_valid


def run_kernel(replay_start_pos: int, num_reqs: int = 2, seq_len: int = 4096):
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_swa_indices_and_lens_kernel as kernel,
    )

    device = torch.device("cuda")
    block_table, query_start_loc, seq_lens, token_to_req, is_valid = build_case(
        num_reqs, seq_len
    )
    index_width = WINDOW + 8  # 略大于窗口，方便发现越界
    swa_indices = torch.full(
        (num_reqs, index_width), -2, dtype=torch.int32, device=device
    )
    swa_lens = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
    replay_start = torch.full(
        (num_reqs,), replay_start_pos, dtype=torch.int32, device=device
    )
    kernel[(num_reqs,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        WINDOW,
        index_width,
        swa_lens,  # left_visible（HAS_IMAGE=False 时未用）
        swa_lens,  # right_visible（未用）
        query_start_loc,
        seq_lens,
        token_to_req,
        is_valid,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        replay_start,
        0,  # token_offset
        HAS_IMAGE=False,
        TRITON_BLOCK_SIZE=128,
    )
    return swa_indices, swa_lens, block_table


def slots_to_positions(slots: torch.Tensor, block_table: torch.Tensor, num_reqs: int):
    """把 slot id 反解为 (req, position)；slot = block_number * block_size + offset。"""
    out = []
    for r in range(num_reqs):
        row = slots[r].tolist()
        blocks = block_table[r].tolist()
        base = r * 1000
        for s in row:
            if s < 0:
                continue
            blk = s // BLOCK_SIZE
            off = s % BLOCK_SIZE
            if blk < base or blk >= base + len(blocks):
                out.append((r, None, s))  # 跨请求/越界
            else:
                out.append((r, (blk - base) * BLOCK_SIZE + off, s))
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("NO_CUDA：需要 GPU 才能跑本 probe")
        return 1

    seq_len = 4096
    pos = seq_len - 1
    replay_start = seq_len - 8  # 命中后只剩 8 行窗口（比自然窗口短，才能看出 clamp）
    failures = []

    # ---- 断言 1/2：replay_start > 0 时，clamp 生效、slot 不越过窗口下界
    ind, lens, bt = run_kernel(replay_start, seq_len=seq_len)
    got_lens = lens.tolist()
    expect_len = pos + 1 - max(pos + 1 - WINDOW, replay_start)
    for r, ln in enumerate(got_lens):
        if ln != expect_len:
            failures.append(f"[A1] req{r} swa_len={ln} 期望 {expect_len}")
    positions = slots_to_positions(ind, bt, num_reqs=2)
    below = [(r, p) for (r, p, _) in positions if p is not None and p < replay_start]
    unknown = [(r, p, s) for (r, p, s) in positions if p is None]
    if below:
        failures.append(f"[A2] 有 {len(below)} 个 slot 落在 replay_start 以下：{below[:3]}")
    if unknown:
        failures.append(f"[A2] 有 {len(unknown)} 个 slot 无法反解（跨请求/越界）：{unknown[:3]}")
    print(f"[A] replay_start={replay_start} swa_len={got_lens} 期望={expect_len} "
          f"读出 slot 数={len(positions)} 下界违例={len(below)} 无法反解={len(unknown)}")

    # ---- 断言 3：对照（replay_start=0）应覆盖窗口下界以下，证明 clamp 是有效保护
    ind0, lens0, bt0 = run_kernel(0, seq_len=seq_len)
    positions0 = slots_to_positions(ind0, bt0, num_reqs=2)
    below0 = [(r, p) for (r, p, _) in positions0 if p is not None and p < replay_start]
    print(f"[B] 无 clamp(replay_start=0) 时 swa_len={lens0.tolist()}（应为 {WINDOW}）")
    print(f"[B] replay_start=0 时读出 {len(positions0)} 个 slot，其中 {len(below0)} 个在窗口下界以下"
          f"（>0 说明 clamp 才是唯一保护）")
    if not below0:
        failures.append("[B] replay_start=0 时也没读到下界以下，probe 构造有问题")

    if failures:
        print("\n❌ 断言失败：")
        for f in failures:
            print("  -", f)
        return 1
    print("\n✅ 三条断言成立：clamp 存在且有效；无 clamp 时会读到窗口以下")
    return 0


if __name__ == "__main__":
    sys.exit(main())
