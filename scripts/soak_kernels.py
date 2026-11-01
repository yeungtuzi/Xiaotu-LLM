"""组件级 soak：随机形状反复调用改过的内核，观测显存/RSS/缓存是否平台化。

目的：在没有完整服务的情况下，先验证"内核层面不会随形状累积资源"。
判据：预热后 allocated / reserved / RSS / Triton 缓存变体数**不单调增长**。
"""
import os, sys, time, random, resource, torch

T = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
vcount = lambda: len([d for d in os.listdir(T) if os.path.isdir(os.path.join(T, d))])
rss_mb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import fp8_mqa_logits_triton as mqa
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import deepseek_v4_sm12x_fp8_einsum as ein

dev = "cuda:0"
H, D = 16, 128
G, HID, OUT = 8, 512, 128
b = torch.randint(0, 255, (G, OUT, HID), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
b_s = torch.ones(G, OUT, HID // 128, device=dev, dtype=torch.float32)

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
print(f"起始: 变体 {vcount()}  RSS {rss_mb():.0f} MB")
samples = []
t0 = time.time()
for i in range(ITERS):
    nq = random.choice([1, 2, 7, 16, 31, 64, 127, 128, 200, 333, 512, 777, 1024, 2048])
    nkv = random.choice([64, 128, 256, 512, 1024, 2048, 4096])
    q = torch.randn(nq, H, D, device=dev, dtype=torch.bfloat16).view(torch.uint8).contiguous()
    k = torch.randn(nkv, D, device=dev, dtype=torch.bfloat16).view(torch.uint8).contiguous()
    sc = torch.ones(nkv, 1, device=dev, dtype=torch.float32)
    w = torch.randn(nq, H, device=dev, dtype=torch.float32)
    ks = torch.zeros(nq, device=dev, dtype=torch.int32)
    ke = torch.full((nq,), nkv, device=dev, dtype=torch.int32)
    mqa(q, (k, sc), w, ks, ke)
    # einsum 的宿主断言要求 float8_e4m3fn；A100 支持该 dtype 作为存储格式
    a = torch.randint(0, 255, (nq, G, HID), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    a_s = torch.ones(nq, G, HID // 128, device=dev, dtype=torch.float32)
    o = torch.empty(nq, G, OUT, device=dev, dtype=torch.bfloat16)
    ein(a, a_s, b, b_s, o)
    torch.cuda.synchronize()
    del q, k, sc, w, ks, ke, a, a_s, o
    if (i + 1) % 40 == 0:
        alloc = torch.cuda.memory_allocated() / 2**20
        resv = torch.cuda.memory_reserved() / 2**20
        free, _ = torch.cuda.mem_get_info()
        samples.append((i + 1, alloc, resv, free / 2**20, rss_mb(), vcount()))
        print(f"  it {i+1:>3}: alloc {alloc:7.1f}  reserved {resv:7.1f}  "
              f"free {free/2**20:7.1f}  rss {rss_mb():6.0f} MB  variants {vcount()}")
dur = time.time() - t0
print(f"\n耗时 {dur:.1f}s")
if len(samples) >= 2:
    first, last = samples[0], samples[-1]
    print(f"reserved 变化 {last[2]-first[2]:+.1f} MiB  |  free 变化 {last[3]-first[3]:+.1f} MiB"
          f"  |  RSS 变化 {last[4]-first[4]:+.0f} MB  |  变体 +{last[5]-first[5]}")
    ok = (last[2] - first[2] < 512) and (last[4] - first[4] < 512) and (last[5] - first[5] <= 6)
    print("判定:", "✅ 平台化（无单调增长）" if ok else "❌ 仍在增长")
