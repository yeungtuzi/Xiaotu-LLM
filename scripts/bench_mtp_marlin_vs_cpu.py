#!/usr/bin/env python
"""MTP 单层 microbench：真实 MXFP4 权重，**上游 Marlin W4A16 GPU** vs **CPU 引擎**。

用途：MTP 纯 GPU 计算（M=6, topk=6）到底比 CPU 快多少。
口径：
  * 权重：checkpoint 的 `mtp.0.ffn.experts`（E=384, H=5120, I=2304）；
  * GPU：vLLM `fused_marlin_moe`（MXFP4 W4A16，A100 BF16 tensor core）；
  * CPU：`MOE_MXFP4.cpu_prefill`（与 `cpu_decode` 同一 forward_many 计算路径；不含 D2H/H2D，
    因此是 CPU 的**下界**；GPU 时间也不含每步 D2H）；
  * M=6, K=6；ids 每行 6 个互异专家；wts 随机；x 随机 bf16；
  * 计时：GPU 用 CUDA Event，CPU 用 perf_counter；取 30 次中位；GPU 先 burn 到高时钟。

用法：
  CUDA_VISIBLE_DEVICES=0 XIAOTU_MOE_THREADS=60 \
    /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python - < scripts/bench_mtp_marlin_vs_cpu.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import time

import numpy as np
import torch

H, I, GK, E, K, M = 5120, 2304, 32, 128, 6, 6
CKPT = os.environ.get(
    "XIAOTU_MTP_CKPT",
    "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master",
)
LAYER = int(os.environ.get("MTP_LAYER", "0"))


def load_mtp_layer(ckpt: str, layer: int):
    from safetensors import safe_open

    prefix = f"mtp.{layer}.ffn.experts."
    # MTP experts are NOT all in one shard (mtp.0.ffn.experts.0..383 are split
    # across model-000*.safetensors). Build expert -> shard index first.
    by_shard: dict[str, list[int]] = {}
    for p in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        for k in hdr:
            if k.startswith(prefix) and k.endswith(".w1.weight"):
                e = int(k[len(prefix):].split(".")[0])
                by_shard.setdefault(p, []).append(e)
    found = sum(len(v) for v in by_shard.values())
    assert found == E, f"{prefix}: found {found}/{E} experts"
    w13 = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.empty((E, H, I // 2), dtype=np.uint8)
    s13 = np.empty((E, 2 * I, H // GK), dtype=np.uint8)
    s2 = np.empty((E, H, I // GK), dtype=np.uint8)
    t0 = time.time()
    for p, experts in by_shard.items():
        with safe_open(p, "pt") as f:
            for e in experts:
                def g(sfx):
                    return f.get_tensor(prefix + f"{e}.{sfx}").cpu()

                w13[e, 0:I] = g("w1.weight").contiguous().numpy().copy()
                w13[e, I:2 * I] = g("w3.weight").contiguous().numpy().copy()
                w2[e] = g("w2.weight").contiguous().numpy().copy()
                s13[e, 0:I] = g("w1.scale").contiguous().view(torch.uint8).numpy().copy()
                s13[e, I:2 * I] = g("w3.scale").contiguous().view(torch.uint8).numpy().copy()
                s2[e] = g("w2.scale").contiguous().view(torch.uint8).numpy().copy()
    print(f"[mtp] loaded mtp.{layer} ({found} experts, {len(by_shard)} shards) "
          f"in {time.time()-t0:.1f}s", flush=True)
    return w13, w2, s13, s2


def main() -> int:
    w13, w2, s13, s2 = load_mtp_layer(CKPT, LAYER)
    rng = np.random.default_rng(7)
    ids = np.stack([rng.choice(E, size=K, replace=False) for _ in range(M)]).astype(np.uint32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    xf = (rng.standard_normal((M, H)).astype(np.float32) * 0.1)
    x_bf = torch.from_numpy(xf).to(torch.bfloat16)
    x_u16 = x_bf.view(torch.uint16).numpy().copy()
    out_cpu = np.zeros((M, H), dtype=np.float32)

    import xiaotu_moe
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1
    cfg.process_id = 0
    cfg.gpu_id = 0
    cfg.has_gate_proj = True
    cfg.expert_num = E
    cfg.top_k = K
    cfg.hidden_size = H
    cfg.intermediate_size = I
    cfg.max_batch_size = 8
    cfg.max_num_seqs = 64
    cfg.stride = 32
    cfg.group_min_len = 10
    cfg.group_max_len = 64
    cfg.groupN = 1
    cfg.groupK = GK
    cfg.activation_type = 0
    cfg.swiglu_limit = 10.0
    t0 = time.time()
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    print(f"[mtp] CPU engine built in {time.time()-t0:.1f}s", flush=True)

    eng.cpu_prefill(M, K, ids, wts, x_u16, out_cpu)
    ts = []
    for _ in range(30):
        t = time.perf_counter()
        eng.cpu_prefill(M, K, ids, wts, x_u16, out_cpu)
        ts.append(time.perf_counter() - t)
    ts.sort()
    t_cpu = statistics.median(ts) * 1e6
    print(f"[mtp] CPU cpu_prefill: median {t_cpu:.1f} us  min {ts[0]*1e6:.1f} us", flush=True)

    dev = torch.device("cuda:0")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
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
    t0 = time.time()
    t_w1, t_w2, t_s1, t_s2, _b1, _b2 = prepare_moe_mxfp4_layer_for_marlin(
        _Stub(), t_w1, t_w2, t_s1, t_s2, None, None
    )
    print(f"[mtp] Marlin repack in {time.time()-t0:.1f}s", flush=True)

    hs = x_bf.to(dev)
    t_ids = torch.from_numpy(ids.astype(np.int32)).to(dev)
    t_wts = torch.from_numpy(wts).to(dev)
    N = 2 * I
    ws13 = torch.empty((E * M, max(H, N)), dtype=torch.bfloat16, device=dev)
    ws2 = torch.empty((E * M, N), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    def gpu_once():
        fused_marlin_moe(
            hs, t_w1, t_w2, None, None, t_s1, t_s2, t_wts, t_ids,
            scalar_types.float4_e2m1f.id, global_num_experts=E,
            activation=MoEActivation.SILU,
            intermediate_cache13=ws2, intermediate_cache2=ws13,
            output=out, input_dtype=torch.bfloat16,
        )

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        clock = lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
    except Exception:
        clock = lambda: -1

    gpu_once(); torch.cuda.synchronize()
    print("[mtp] gpu clock before burn:", clock(), flush=True)
    for _ in range(1500):
        gpu_once()
    torch.cuda.synchronize()
    print("[mtp] gpu clock after burn:", clock(), flush=True)

    ev = []
    for _ in range(40):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); gpu_once(); b.record(); b.synchronize()
        ev.append(a.elapsed_time(b) * 1000.0)
    ev.sort()
    t_gpu = statistics.median(ev)
    print(f"[mtp] GPU Marlin: median {t_gpu:.1f} us  min {ev[0]:.1f} p75 {ev[30]:.1f}  clock {clock()}", flush=True)

    # correctness (CPU fp32 vs GPU bf16-marlin); not the timing criterion
    g = out.float().cpu().numpy()
    scale = float(np.abs(out_cpu).mean()) or 1.0
    d = np.abs(g - out_cpu)
    print(f"[mtp] GPU vs CPU: abs p50 {np.percentile(d,50):.2e} p99 {np.percentile(d,99):.2e} "
          f"max {d.max():.2e}  norm p50 {np.percentile(d,50)/scale:.2e}", flush=True)
    print(f"[mtp] ratio CPU/GPU = {t_cpu/t_gpu:.2f}x; 3 MTP layers ~ {t_gpu*3/1000:.2f} ms GPU, "
          f"{t_cpu*3/1000:.2f} ms CPU", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
