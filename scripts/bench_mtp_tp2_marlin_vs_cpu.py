#!/usr/bin/env python
"""MTP 单层 **TP=2 真实分片** microbench：每 rank 64 专家 + NCCL all-reduce。

GPU 路径：vLLM `fused_marlin_moe`（MXFP4 W4A16，expert_map 映射全局 128 专家）；
CPU 路径：每 rank `MOE_MXFP4`（本地 64 专家）+ NCCL all-reduce；
M=6, topk=6；真实 `mtp.0.ffn.experts` 权重；只用 GPU0/1。
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
    # locate each expert's shard
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


def worker(rank: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29577", rank=rank, world_size=TP
    )
    dev = torch.device("cuda", rank)
    start, end = rank * (E // TP), (rank + 1) * (E // TP)
    w13, w2, s13, s2 = load_local(CKPT, LAYER, start, end)

    rng = np.random.default_rng(7)
    ids = np.stack([rng.choice(E, size=K, replace=False) for _ in range(M)]).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    xf = rng.standard_normal((M, H)).astype(np.float32) * 0.1
    x_bf = torch.from_numpy(xf).to(torch.bfloat16)
    x_u16 = x_bf.view(torch.uint16).numpy().copy()
    out_cpu = np.zeros((M, H), dtype=np.float32)

    import xiaotu_moe
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = rank
    cfg.has_gate_proj = True; cfg.expert_num = end - start; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 8; cfg.max_num_seqs = 64
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 64
    cfg.groupN = 1; cfg.groupK = GK; cfg.activation_type = 0
    cfg.swiglu_limit = 10.0
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)

    # global ids -> local ids; non-local gets -1 (CPU engine: use 0 + weight 0)
    mask = (ids >= start) & (ids < end)
    ids_l = np.where(mask, ids - start, 0).astype(np.uint32)
    wts_l = np.where(mask, wts, 0.0).astype(np.float32)

    eng.cpu_prefill(M, K, ids_l, wts_l, x_u16, out_cpu)
    tc = []
    for _ in range(30):
        t = time.perf_counter()
        eng.cpu_prefill(M, K, ids_l, wts_l, x_u16, out_cpu)
        tc.append(time.perf_counter() - t)
    tc.sort()
    t_cpu = statistics.median(tc) * 1e6

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

    def gpu_local():
        fused_marlin_moe(
            hs, t_w1, t_w2, None, None, t_s1, t_s2, t_wts, t_ids,
            scalar_types.float4_e2m1f.id, global_num_experts=E,
            activation=MoEActivation.SILU,
            activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
            expert_map=expert_map,
            intermediate_cache13=ws2, intermediate_cache2=ws13,
            output=out, input_dtype=torch.bfloat16,
        )

    def gpu_full():
        gpu_local()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

    cpu_dev = torch.zeros((M, H), dtype=torch.float32, device=dev)

    def cpu_full():
        eng.cpu_prefill(M, K, ids_l, wts_l, x_u16, out_cpu)
        cpu_dev.copy_(torch.from_numpy(out_cpu))
        dist.all_reduce(cpu_dev, op=dist.ReduceOp.SUM)

    def time_gpu(fn, n=40):
        fn(); torch.cuda.synchronize(); dist.barrier()
        for _ in range(300):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        ts = []
        for _ in range(n):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        ts.sort()
        return statistics.median(ts), ts[0]

    t_gpu, t_gpu_min = time_gpu(gpu_full)
    t_gpu_local, t_gpu_local_min = time_gpu(gpu_local)
    # all-reduce only
    def ar_only():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    t_ar, t_ar_min = time_gpu(ar_only)
    # H2D only
    def h2d_only():
        cpu_dev.copy_(torch.from_numpy(out_cpu))
    t_h2d, t_h2d_min = time_gpu(h2d_only)

    # CPU full (local compute + H2D copy + all-reduce)
    cpu_full(); torch.cuda.synchronize(); dist.barrier()
    for _ in range(30):
        cpu_full()
    torch.cuda.synchronize(); dist.barrier()
    tcf = []
    for _ in range(30):
        t = time.perf_counter(); cpu_full(); torch.cuda.synchronize(); tcf.append(time.perf_counter() - t)
    tcf.sort(); t_cpu_full = statistics.median(tcf) * 1e6

    if rank == 0:
        print(f"[tp2] per-rank local experts={end-start}; M={M} K={K}", flush=True)
        print(f"[tp2] CPU local compute: {t_cpu:.1f} us (median)", flush=True)
        print(f"[tp2] GPU Marlin local (no AR): {t_gpu_local:.1f} us (median, min {t_gpu_local_min:.1f})", flush=True)
        print(f"[tp2] GPU Marlin local + AR: {t_gpu:.1f} us (median, min {t_gpu_min:.1f})", flush=True)
        print(f"[tp2] all-reduce only: {t_ar:.1f} us (median, min {t_ar_min:.1f})", flush=True)
        print(f"[tp2] H2D only: {t_h2d:.1f} us (median, min {t_h2d_min:.1f})", flush=True)
        print(f"[tp2] CPU local + H2D + AR: {t_cpu_full:.1f} us (median)", flush=True)
        print(f"[tp2] CPU/GPU (both with AR) = {t_cpu_full/t_gpu:.2f}x; 3 layers GPU {t_gpu*3/1000:.2f} ms", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.start_processes(worker, nprocs=TP, start_method="fork")
