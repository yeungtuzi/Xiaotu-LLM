"""生产标量内核 moe4_smallm 的基线探针：per-rank H=5120/I=1152、15 专家、topk=3、5 token。"""
import os, time, statistics
import numpy as np, torch
import xiaotu_moe
import vllm_xiaotu_moe.moe4_smallm as m4

H = int(os.environ.get("H", 5120)); I = int(os.environ.get("I", 1152))
E = int(os.environ.get("E", 15)); K = int(os.environ.get("TOPK", 3)); B = int(os.environ.get("BS", 5))
REP = int(os.environ.get("REP", 100))
torch.manual_seed(7)
dev = "cuda:0"
w13 = torch.randint(0, 256, (E, H//2, 2*I), dtype=torch.uint8, device=dev)
s13 = torch.randint(120, 134, (E, H//32, 2*I), dtype=torch.uint8, device=dev)
w2  = torch.randint(0, 256, (E, I//2, H), dtype=torch.uint8, device=dev)
s2  = torch.randint(120, 134, (E, I//32, H), dtype=torch.uint8, device=dev)
x = torch.randn(B, H, dtype=torch.bfloat16, device=dev)
# 15 个槽位正好覆盖 15 个专家（与 GPU 内核工作集一致）
ids = torch.arange(E, dtype=torch.int32, device=dev).reshape(B, K)
wts = torch.ones(B, K, dtype=torch.float32, device=dev)
if not m4.available():
    print("[scalar] moe4_smallm 不可用"); raise SystemExit(1)
planes = (w13, s13, w2, s2)
for _ in range(5):
    out = m4.moe_layer(planes, x, ids, wts, H, I, num_experts=E)
torch.cuda.synchronize()
ts = []
for _ in range(REP):
    t0 = time.perf_counter()
    out = m4.moe_layer(planes, x, ids, wts, H, I, num_experts=E)
    torch.cuda.synchronize()
    ts.append((time.perf_counter()-t0)*1000)
med = statistics.median(ts)
by = (E*(H//2)*2*I + E*(I//2)*H)/2**20
print(f"[scalar] H={H} I={I} E={E} K={K} B={B}  中位 {med:.3f} ms  "
      f"{by:.0f} MiB => {by/1024/(med/1000):.0f} GB/s", flush=True)
