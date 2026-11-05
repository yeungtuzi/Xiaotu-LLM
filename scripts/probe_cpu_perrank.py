"""per-rank CPU 引擎探针：I=1152 / E=15 / topk=3 / B=5，合成权重，用于与 GPU 内核同口径对照。"""
import os, sys, time, statistics
import numpy as np
import xiaotu_moe

def f32_to_bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16).view(np.uint16)

def main():
    H = int(os.environ.get("H", 5120))
    I = int(os.environ.get("I", 1152))
    E = int(os.environ.get("E", 15))
    K = int(os.environ.get("TOPK", 3))
    B = int(os.environ.get("BS", 5))
    REP = int(os.environ.get("REP", 12))
    rng = np.random.default_rng(7)
    w13 = rng.integers(0, 256, size=(E, 2*I, H//2), dtype=np.uint8)
    s13 = rng.integers(120, 134, size=(E, 2*I, H//32), dtype=np.uint8)
    w2  = rng.integers(0, 256, size=(E, H, I//2), dtype=np.uint8)
    s2  = rng.integers(120, 134, size=(E, H, I//32), dtype=np.uint8)
    m = xiaotu_moe.load()
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 8192; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    eng = m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0)
    ids = rng.integers(0, E, size=(B, K)).astype(np.int32)
    wts = rng.uniform(-1, 1, size=(B, K)).astype(np.float32)
    x = f32_to_bf16_bits(rng.standard_normal((B, H)).astype(np.float32))
    out = np.zeros((B, H), dtype=np.float32)
    for _ in range(3):
        eng.cpu_prefill(B, K, ids, wts, x, out)
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        eng.cpu_prefill(B, K, ids, wts, x, out)
        ts.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(ts)
    by = (E * (H//2) * 2*I + E * (I//2) * H) / 2**20
    fl = E * (H * 2*I + I * H) * 2
    print(f"[cpu-perrank] H={H} I={I} E={E} K={K} B={B}  中位 {med:.3f} ms  "
          f"{by:.0f} MiB ⇒ {by/1024/(med/1000):.0f} GB/s   {fl/(med/1000)/1e12:.2f} TFLOP/s", flush=True)

if __name__ == "__main__":
    main()
