#!/usr/bin/env python
"""Single-GPU MTP Marlin 60s steady-state (same shape as TP=2 per-rank, no NCCL).

Burn to >= BOOST_MHZ, then sample for DURATION_S seconds; 10% two-sided trim mean
plus median/percentiles. Reports SM clock per sample.
"""
from __future__ import annotations
import glob, json, os, statistics, time
import numpy as np
import torch

H, I, GK, E, K, M = 5120, 2304, 32, 128, 6, 6
CKPT = os.environ.get("XIAOTU_MTP_CKPT", "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master")
LAYER = int(os.environ.get("MTP_LAYER", "0"))
DURATION_S = float(os.environ.get("DURATION_S", "60"))
BOOST_MHZ = int(os.environ.get("BOOST_MHZ", "1350"))
GPU_INDEX = int(os.environ.get("GPU_INDEX", "1"))

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



def stats(ts):
    ts = sorted(ts); n = len(ts); cut = max(1, int(round(n * 0.10)))
    tr = ts[cut:n-cut]
    return {"n": n, "min": ts[0], "p25": ts[n//4], "median": statistics.median(ts),
            "p75": ts[3*n//4], "max": ts[-1], "trim_mean": statistics.mean(tr)}


def main():
    w13, w2, s13, s2 = load_mtp_layer(CKPT, LAYER)
    rng = np.random.default_rng(7)
    ids = np.stack([rng.choice(E, size=K, replace=False) for _ in range(M)]).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    xf = rng.standard_normal((M, H)).astype(np.float32) * 0.1
    x_bf = torch.from_numpy(xf).to(torch.bfloat16)
    dev = torch.device("cuda:0")

    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
    from vllm.scalar_type import scalar_types

    class _Stub: params_dtype = torch.bfloat16

    t_w1 = torch.from_numpy(w13).to(dev); t_w2 = torch.from_numpy(w2).to(dev)
    t_s1 = torch.from_numpy(s13).to(dev); t_s2 = torch.from_numpy(s2).to(dev)
    t_w1, t_w2, t_s1, t_s2, _b1, _b2 = prepare_moe_mxfp4_layer_for_marlin(_Stub(), t_w1, t_w2, t_s1, t_s2, None, None)
    hs = x_bf.to(dev)
    t_ids = torch.from_numpy(ids).to(dev)
    t_wts = torch.from_numpy(wts).to(dev)
    N2 = 2 * I
    ws13 = torch.empty((E * M, max(H, N2)), dtype=torch.bfloat16, device=dev)
    ws2 = torch.empty((E * M, N2), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    def gpu_once():
        fused_marlin_moe(hs, t_w1, t_w2, None, None, t_s1, t_s2, t_wts, t_ids,
            scalar_types.float4_e2m1f.id, global_num_experts=E,
            activation=MoEActivation.SILU,
            activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
            intermediate_cache13=ws2, intermediate_cache2=ws13, output=out,
            input_dtype=torch.bfloat16)

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
        clock = lambda: int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:
        clock = lambda: -1

    print(f"[single] active_experts={np.unique(ids).size} active_pairs={int((ids>=0).sum())} clock_start={clock()}", flush=True)
    gpu_once(); torch.cuda.synchronize()
    t_burn = time.perf_counter()
    for chunk in range(30):
        for _ in range(2000):
            gpu_once()
        torch.cuda.synchronize()
        c = clock()
        print(f"[single] burn chunk={chunk} clock={c} MHz elapsed={time.perf_counter()-t_burn:.1f}s", flush=True)
        if c >= BOOST_MHZ:
            break
    print(f"[single] burn_done clock={clock()} after {time.perf_counter()-t_burn:.1f}s", flush=True)

    ev, clocks = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < DURATION_S:
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); gpu_once(); b.record(); b.synchronize()
        ev.append(a.elapsed_time(b) * 1000.0); clocks.append(clock())
    s = stats(ev)
    print(f"[single] sample_s={time.perf_counter()-t0:.1f} n={s['n']}", flush=True)
    print(f"[single] clocks min/med/max/mean = {min(clocks)}/{int(statistics.median(clocks))}/{max(clocks)}/{statistics.mean(clocks):.1f}", flush=True)
    print(f"[single] local n={s['n']} med={s['median']:.1f} trim_mean={s['trim_mean']:.1f} "
          f"min={s['min']:.1f} p25={s['p25']:.1f} p75={s['p75']:.1f} max={s['max']:.1f}", flush=True)
    print(f"[single] TRIMMED_MEAN local={s['trim_mean']:.1f}us", flush=True)

if __name__ == "__main__":
    main()
