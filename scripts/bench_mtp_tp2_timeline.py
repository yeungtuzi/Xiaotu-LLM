#!/usr/bin/env python
"""TP=2 MTP 时间线：每个 rank 的本地 Marlin、NCCL all-reduce、合并路径分别计时。

只在 GPU1/2（PCIe x16）跑；GPU0 是 x8，不能用于 TP=2。
输出每个 rank 的 wall/event 时间、时钟、NCCL 传输信息；不做猜测。
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


def clock(rank: int):
    try:
        import pynvml
        pynvml.nvmlInit()
        # physical GPU1/2 when CUDA_VISIBLE_DEVICES=1,2
        h = pynvml.nvmlDeviceGetHandleByIndex(rank + 1)
        return pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return -1


def worker(rank: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29578", rank=rank, world_size=TP,
        device_id=torch.device("cuda", rank),
    )
    dev = torch.device("cuda", rank)
    start, end = rank * (E // TP), (rank + 1) * (E // TP)
    w13, w2, s13, s2 = load_local(CKPT, LAYER, start, end)

    rng = np.random.default_rng(7)
    # Balanced routing: each token sends exactly K/2 experts to each rank.
    # This removes the rank-workload imbalance from the measurement.
    half = K // 2
    if os.environ.get("BALANCED", "1") == "1":
        rows = []
        for _ in range(M):
            a = rng.choice(E // TP, size=half, replace=False)
            b = rng.choice(E // TP, size=K - half, replace=False) + (E // TP)
            row = np.concatenate([a, b])
            rng.shuffle(row)
            rows.append(row)
        ids = np.stack(rows).astype(np.int32)
    else:
        ids = np.stack([rng.choice(E, size=K, replace=False) for _ in range(M)]).astype(np.int32)
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
    N = 2 * I
    ws13 = torch.empty(((end - start) * M, max(H, N)), dtype=torch.bfloat16, device=dev)
    ws2 = torch.empty(((end - start) * M, N), dtype=torch.bfloat16, device=dev)
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

    # warmup (outside timing)
    for _ in range(100):
        both()
    torch.cuda.synchronize(); dist.barrier()

    def wall(fn, n=40):
        ts = []
        for _ in range(n):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter_ns() - t0) / 1000.0)
        ts.sort()
        return statistics.median(ts), ts[0], ts[-1]

    def event(fn, n=40):
        ts = []
        for _ in range(n):
            dist.barrier()
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        ts.sort()
        return statistics.median(ts), ts[0], ts[-1]

    # one explicit timestamp trace (ns) for this rank
    dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    local()
    t1 = time.perf_counter_ns()
    ar()
    t2 = time.perf_counter_ns()
    torch.cuda.synchronize()
    t3 = time.perf_counter_ns()

    lw = wall(local)
    aw = wall(ar)
    bw = wall(both)
    le = event(local)
    ae = event(ar)

    if rank == 0:
        print(f"[trace] GPU name rank0={torch.cuda.get_device_name(0)} rank1={torch.cuda.get_device_name(1)}", flush=True)
        print(f"[trace] torch={torch.__version__} cuda={torch.version.cuda} backend=nccl", flush=True)
    mask = (ids >= start) & (ids < end)
    print(f"[rank{rank}] local_E={end-start} clock={clock(rank)} MHz "
          f"active_pairs={int(mask.sum())} active_experts={int(np.unique(ids[mask]).size)}", flush=True)
    print(f"[rank{rank}] local wall median={lw[0]:.1f} us min={lw[1]:.1f} max={lw[2]:.1f}", flush=True)
    print(f"[rank{rank}] local event median={le[0]:.1f} us min={le[1]:.1f} max={le[2]:.1f}", flush=True)
    print(f"[rank{rank}] AR wall median={aw[0]:.1f} us min={aw[1]:.1f} max={aw[2]:.1f}", flush=True)
    print(f"[rank{rank}] AR event median={ae[0]:.1f} us min={ae[1]:.1f} max={ae[2]:.1f}", flush=True)
    print(f"[rank{rank}] local+AR wall median={bw[0]:.1f} us min={bw[1]:.1f} max={bw[2]:.1f}", flush=True)
    print(f"[rank{rank}] trace ns: local_host={t1-t0} ar_host={t2-t1} sync_tail={t3-t2}", flush=True)
    print(f"[rank{rank}] isolated sum local+AR={lw[0]+aw[0]:.1f} us, combined={bw[0]:.1f} us, gap={bw[0]-lw[0]-aw[0]:.1f} us", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.start_processes(worker, nprocs=TP, start_method="fork")
