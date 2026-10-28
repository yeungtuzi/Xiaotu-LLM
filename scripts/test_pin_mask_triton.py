#!/usr/bin/env python
"""Triton 融合掩码的自测：与 torch 参考实现**逐位**对拍。

不加载模型、不起服务、不碰任何 vLLM 进程；只用到一张卡的极小张量
（M*K ≤ 512 个元素），适合在共享机器上随时运行。

用法：
    CUDA_VISIBLE_DEVICES=0 python scripts/test_pin_mask_triton.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_xiaotu_moe.pin_mask_triton import (  # noqa: E402
    PinMaskFused, torch_pin_mask, _HAS_TRITON,
)


def one_case(M: int, K: int, E: int, n_local: int, S: int, seed: int,
             device: torch.device) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    # ids: 含 -1 哨兵（profile/warmup 期）与越界上界的邻值
    ids = torch.randint(-1, E, (M, K), generator=g, dtype=torch.int32).to(device)
    wts = torch.rand((M, K), generator=g, dtype=torch.float32).to(device)
    # slotmap: -1 = 不在池；[n_local, n_local+S) = 在池的第 s 个槽
    sm = torch.full((E,), -1, dtype=torch.int32)
    slots = torch.randint(0, S, (max(1, E // 8),), generator=g)
    idx = torch.randperm(E, generator=g)[: slots.numel()]
    sm[idx] = (n_local + slots).to(torch.int32)
    sm = sm.to(device)

    ref_ids, ref_w, ref_part, ref_res = torch_pin_mask(ids, wts, sm, n_local)
    fused = PinMaskFused(device)
    fused._enabled = True                       # 强制走 Triton（忽略 env）
    got_ids, got_w, got_part = fused.step(ids, wts, sm, n_local)

    ok = (torch.equal(got_ids, ref_ids)
          and torch.equal(got_w.view(torch.int32), ref_w.view(torch.int32))
          and torch.equal(got_part, ref_part))
    got_hits = int(fused.hits().item())
    exp_hits = int(ref_res.sum().item())
    ok = ok and (got_hits == exp_hits)
    tag = "OK " if ok else "BAD"
    print(f"[{tag}] M={M:4d} K={K} E={E:4d} n_local={n_local:4d} S={S:2d} "
          f"hits={got_hits}/{exp_hits}", flush=True)
    if not ok:
        for name, a, b in (("cpu_ids", got_ids, ref_ids), ("cpu_w", got_w, ref_w),
                           ("part", got_part, ref_part)):
            if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                bad = (a.view(torch.int32) != b.view(torch.int32)).nonzero()[:5]
                print(f"    {name} 不一致，前 5 处 idx={bad.tolist()} "
                      f"got={a.flatten()[bad.flatten()].tolist()} "
                      f"ref={b.flatten()[bad.flatten()].tolist()}")
        raise SystemExit(1)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: 没有 CUDA 设备（本测试需要 GPU 才能跑 Triton）")
        return 0
    if not _HAS_TRITON:
        print("SKIP: 环境里没有 triton")
        return 0
    dev = torch.device("cuda")
    print(f"[env] {torch.cuda.get_device_name(0)} triton={_HAS_TRITON} "
          f"dev={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    # 覆盖：真实 decode 形状、全 -1、全驻留、边界（slot == n_local）
    for M, K in ((1, 6), (6, 6), (6, 8), (32, 8)):
        for seed in range(3):
            one_case(M, K, E=384, n_local=192, S=50, seed=seed, device=dev)
    # 全 -1 哨兵
    ids = torch.full((6, 6), -1, dtype=torch.int32, device=dev)
    wts = torch.rand((6, 6), dtype=torch.float32, device=dev)
    sm = torch.full((384,), -1, dtype=torch.int32, device=dev)
    ref = torch_pin_mask(ids, wts, sm, 192)
    f = PinMaskFused(dev)
    f._enabled = True
    got = f.step(ids, wts, sm, 192)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]) \
        and torch.equal(got[2], ref[2]) and int(f.hits().item()) == 0
    print("[OK ] 全 -1 哨兵：cpu_ids 全 0、cpu_w 全 0、part 全 -1、hits=0", flush=True)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
