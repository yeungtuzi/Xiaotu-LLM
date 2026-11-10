"""独立数值 probe：C++ host K-major 直接写连续 ring == 原 K-major ring 三块切片。

不启动 vLLM、不加载权重。构造合成 shard，按 qfn 真实几何（H=2560 I=640 ns=2）
分别用「三块切片」和「单次连续写」两条路填 ring，逐字节比对。

判据：torch.equal(...) == True（逐字节相等，不是 allclose）
"""
import torch

dev = torch.device("musa")
H, I, NS = 2560, 640, 2
cr13 = I // NS          # 320
Et = 48                 # experts in this tile
b = 128                 # fp8 block

torch.manual_seed(0)

# ── 引擎 host shard 布局（w13）：每 node 每 expert = [gate cbytes][up cbytes]
#    cbytes = cr13 * H，即 [2, cr13, H]
shard = [torch.randint(0, 256, (Et, 2, cr13, H), dtype=torch.uint8) for _ in range(NS)]

# ── 目标 ring：K-major [Et, H, 2I] 连续
ring_ref = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)
ring_new = torch.zeros((Et, H, 2 * I), dtype=torch.uint8, device=dev)

# ── 路径 A：今天 Python 的三块切片（逐 node 写列区间）
host_tr = []
for n in range(NS):
    t = shard[n].permute(0, 3, 1, 2).contiguous().pin_memory()   # [Et, H, 2, cr13]
    host_tr.append(t)
    c0 = n * cr13
    ring_ref[:, :, c0:c0 + cr13].copy_(t[:, :, 0, :])
    ring_ref[:, :, I + c0:I + c0 + cr13].copy_(t[:, :, 1, :])
torch.musa.synchronize()

# ── 路径 B：先在 host 组装整块连续 K-major，再一次性 H2D
host_full = torch.empty((Et, H, 2 * I), dtype=torch.uint8)
for n in range(NS):
    t = host_tr[n].clone()
    c0 = n * cr13
    host_full[:, :, c0:c0 + cr13].copy_(t[:, :, 0, :])
    host_full[:, :, I + c0:I + c0 + cr13].copy_(t[:, :, 1, :])
host_full = host_full.pin_memory()
ring_new[:, :].copy_(host_full)
torch.musa.synchronize()

ok = torch.equal(ring_ref, ring_new)
print(f"w13 逐字节相等: {ok}")
print(f"  ring_ref nz={int((ring_ref != 0).sum())} ring_new nz={int((ring_new != 0).sum())}")
if not ok:
    d = (ring_ref != ring_new)
    print(f"  MISMATCH count={int(d.sum())} of {ring_ref.numel()}")
    idx = d.nonzero()[:5]
    for z in idx:
        print(f"   at {tuple(z.tolist())}: ref={int(ring_ref[tuple(z.tolist())])} new={int(ring_new[tuple(z.tolist())])}")

# ── w2：ring [Et, I, H]，host shard 每 node = [cr2, I]，cr2 = H/NS = 1280
cr2 = H // NS
shard2 = [torch.randint(0, 256, (Et, cr2, I), dtype=torch.uint8) for _ in range(NS)]
ring2_ref = torch.zeros((Et, I, H), dtype=torch.uint8, device=dev)
ring2_new = torch.zeros((Et, I, H), dtype=torch.uint8, device=dev)

host2 = torch.empty((Et, I, H), dtype=torch.uint8)
for n in range(NS):
    c0 = n * cr2
    tr = shard2[n].transpose(1, 2).contiguous().pin_memory()   # [Et, I, cr2]
    ring2_ref[:, :, c0:c0 + cr2].copy_(tr)
    host2[:, :, c0:c0 + cr2].copy_(tr)
host2 = host2.pin_memory()
ring2_new[:, :].copy_(host2)
torch.musa.synchronize()

ok2 = torch.equal(ring2_ref, ring2_new)
print(f"w2  逐字节相等: {ok2}")

print("RESULT:", "PASS" if (ok and ok2) else "FAIL")
