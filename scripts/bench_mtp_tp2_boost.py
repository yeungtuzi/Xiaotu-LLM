#!/usr/bin/env python
"""TP=2 MTP 稳定性复测：先烧到 1400 MHz，再跑 32 次，去极值取中位/均值。

只用于 GPU1/2（PCIe x16）。每 rank 记录 clock、active pairs/experts、本地 Marlin、
NCCL all-reduce、合并时间；不做因果猜测。
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, I, GK, E, K, M = 5120, 2304, 32, 128, 6, 6
CKPT = os.environ.get(
    "XIAOTU_MTP_CKPT",
    "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master",
)
LAYER = int(os.environ.get("MTP_LAYER", "0"))
TP = 2
N = int(os.environ.get("N", "0"))
DURATION_S = float(os.environ.get("DURATION_S", "60"))
BOOST_MHZ = int(os.environ.get("BOOST_MHZ", "1350"))


def load_local(ckpt: str, layer: int, start: int, end: int):
    from safetensors import safe_open

    prefix = f"mtp.{layer}.ffn.experts."
    nloc = end - start
    w13 = np.empty((nloc, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((nloc, H, I // 2), dtype=np.uint8)
    s13 = np.empty((nloc, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((nloc, H, I // GK), dtype=np.uint8)
    shard_of = {}
    for p in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        for k in hdr:
            if k.startswith(prefix) and k.endswith(".w1.weight"):
                e = int(k[len(prefix):].split(".")[0])
                if start <= e < end:
                    shard_of[e] = p
    assert len(shard_of) == nloc, f"local experts {len(shard_of)}/{nloc}"
    by_shard = {}
    for e, p in shard_of.items():
        by_shard.setdefault(p, []).append(e)
    for p, es in by_shard.items():
        with safe_open(p, "pt") as f:
            for e in es:
                i = e - start

                def g(sfx):
                    return f.get_tensor(prefix + f"{e}.{sfx}").cpu()

                w13[i, 0:I] = g("w1.weight").contiguous().numpy().copy()
                w13[i, I:2 * I] = g("w3.weight").contiguous().numpy().copy()
                w2[i] = g("w2.weight").contiguous().numpy().copy()
                s13[i, 0:I] = g("w1.scale").contiguous().view(torch.uint8).numpy().copy()
                s13[i, I:2 * I] = g("w3.scale").contiguous().view(torch.uint8).numpy().copy()
                s2[i] = g("w2.scale").contiguous().view(torch.uint8).numpy().copy()
    return w13, w2, s13, s2


def rank_clock(rank: int):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(rank + 1)
        return int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:
        return -1


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    cut = max(1, int(round(n * 0.10)))
    tr = ts[cut:n - cut]
    return {
        "n": n,
        "min": ts[0],
        "p25": ts[n // 4],
        "median": statistics.median(ts),
        "p75": ts[3 * n // 4],
        "max": ts[-1],
        "trim_mean": statistics.mean(tr),
        "trim_n": len(tr),
    }


def worker(rank: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29579", rank=rank, world_size=TP,
        device_id=torch.device("cuda", rank),
    )
    dev = torch.device("cuda", rank)
    start, end = rank * (E // TP), (rank + 1) * (E // TP)
    w13, w2, s13, s2 = load_local(CKPT, LAYER, start, end)
    print(f"[rank{rank}] weights ready", flush=True)

    rng = np.random.default_rng(7)
    half = K // 2
    rows = []
    for _ in range(M):
        a = rng.choice(E // TP, size=half, replace=False)
        b = rng.choice(E // TP, size=K - half, replace=False) + (E // TP)
        row = np.concatenate([a, b])
        rng.shuffle(row)
        rows.append(row)
    ids = np.stack(rows).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    xf = rng.standard_normal((M, H)).astype(np.float32) * 0.1
    x_bf = torch.from_numpy(xf).to(torch.bfloat16)

    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig, MoEActivation,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    class _Stub:
        params_dtype = torch.bfloat16

    t_w1 = torch.from_numpy(w13).to(dev)
    t_w2 = torch.from_numpy(w2).to(dev)
    t_s1 = torch.from_numpy(s13).to(dev)
    t_s2 = torch.from_numpy(s2).to(dev)
    t_w1, t_w2, t_s1, t_s2, _b1, _b2 = prepare_moe_mxfp4_layer_for_marlin(
        _Stub(), t_w1, t_w2, t_s1, t_s2, None, None
    )
    expert_map = torch.tensor(
        [e - start if start <= e < end else -1 for e in range(E)],
        dtype=torch.int32, device=dev,
    )
    hs = x_bf.to(dev)
    t_ids = torch.from_numpy(ids).to(dev)
    t_wts = torch.from_numpy(wts).to(dev)
    N2 = 2 * I
    ws13 = torch.empty(((end - start) * M, max(H, N2)), dtype=torch.bfloat16, device=dev)
    ws2 = torch.empty(((end - start) * M, N2), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    def local():
        fused_marlin_moe(
            hs, t_w1, t_w2, None, None, t_s1, t_s2, t_wts, t_ids,
            scalar_types.float4_e2m1f.id, global_num_experts=E,
            activation=MoEActivation.SILU,
            activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
            expert_map=expert_map,
            intermediate_cache13=ws2, intermediate_cache2=ws13,
            output=out, input_dtype=torch.bfloat16,
        )

    def ar():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

    def both():
        local()
        ar()

    active_pairs = int(((ids >= start) & (ids < end)).sum())
    active_experts = int(np.unique(ids[(ids >= start) & (ids < end)]).size)
    print(f"[rank{rank}] active_pairs={active_pairs} active_experts={active_experts} "
          f"clock_start={rank_clock(rank)}", flush=True)

    # Burn each rank long enough to reach boost, and require both ranks >= BOOST_MHZ.
    t_burn = time.perf_counter()
    for chunk in range(30):
        for _ in range(2000):
            local()
        torch.cuda.synchronize()
        c = rank_clock(rank)
        ct = torch.tensor([c], dtype=torch.int64, device=dev)
        dist.all_reduce(ct, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(f"[burn] chunk={chunk} min_clock={int(ct.item())} MHz "
                  f"elapsed={time.perf_counter()-t_burn:.1f}s", flush=True)
        if int(ct.item()) >= BOOST_MHZ:
            break
    print(f"[rank{rank}] burn_done clock={rank_clock(rank)} after "
          f"{time.perf_counter()-t_burn:.1f}s", flush=True)

    # 32 samples; each phase separately; barrier outside timing.
    local_ts, ar_ts, both_ts, clocks = [], [], [], []
    t_sample = time.perf_counter()
    i = 0
    while (N > 0 and i < N) or (N == 0 and time.perf_counter() - t_sample < DURATION_S):
        i += 1
        dist.barrier()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); local(); b.record(); b.synchronize()
        local_ts.append(a.elapsed_time(b) * 1000.0)

        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns(); ar(); torch.cuda.synchronize()
        ar_ts.append((time.perf_counter_ns() - t0) / 1000.0)

        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns(); both(); torch.cuda.synchronize()
        both_ts.append((time.perf_counter_ns() - t0) / 1000.0)
        clocks.append(rank_clock(rank))

    ls, ars, bs = stats(local_ts), stats(ar_ts), stats(both_ts)
    print(f"[rank{rank}] sample_s={time.perf_counter()-t_sample:.1f} n={len(local_ts)}", flush=True)
    print(f"[rank{rank}] clocks min/med/max/mean = {min(clocks)}/{int(statistics.median(clocks))}/{max(clocks)}/{statistics.mean(clocks):.1f}", flush=True)
    print(f"[rank{rank}] local n={ls['n']} med={ls['median']:.1f} trim_mean={ls['trim_mean']:.1f} "
          f"min={ls['min']:.1f} p25={ls['p25']:.1f} p75={ls['p75']:.1f} max={ls['max']:.1f}", flush=True)
    print(f"[rank{rank}] AR    n={ars['n']} med={ars['median']:.1f} trim_mean={ars['trim_mean']:.1f} "
          f"min={ars['min']:.1f} p25={ars['p25']:.1f} p75={ars['p75']:.1f} max={ars['max']:.1f}", flush=True)
    print(f"[rank{rank}] both  n={bs['n']} med={bs['median']:.1f} trim_mean={bs['trim_mean']:.1f} "
          f"min={bs['min']:.1f} p25={bs['p25']:.1f} p75={bs['p75']:.1f} max={bs['max']:.1f}", flush=True)
    print(f"[rank{rank}] TRIMMED_MEAN local={ls['trim_mean']:.1f}us AR={ars['trim_mean']:.1f}us both={bs['trim_mean']:.1f}us", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.start_processes(worker, nprocs=TP, start_method="fork")
