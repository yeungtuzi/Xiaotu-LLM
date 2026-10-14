# scripts/test_pin_numeric.py —— XIAOTU_PIN 分流数值微测试(引擎级,GPU0,无需服务)
# 判据:分流(池命中走 GPU partial + CPU 只算 miss)vs 基线(全 CPU)= maxrel 小 + cos≈1。
import sys, os, numpy as np, torch
sys.path.insert(0, os.environ.get("XTU_DEVTREE", "/home/user/lvllm-tmp/devtree"))
os.environ.setdefault("XIAOTU_PIN", "1")
from vllm_xiaotu_moe import expert_pool as xp
import xiaotu_moe

m = xiaotu_moe.load()
cfg = m.MOEConfigV2()
E, H, I, gk, T, K = 8, 256, 128, 32, 4, 2
for k, v in dict(expert_num=E, hidden_size=H, intermediate_size=I, top_k=K,
                 has_gate_proj=True, num_processes=1, process_id=0, max_batch_size=256,
                 max_num_seqs=8, stride=gk, groupN=1, groupK=gk, activation_type=0).items():
    setattr(cfg, k, v)
rng = np.random.default_rng(7)
rb = lambda *s: rng.integers(0, 255, s, dtype=np.uint8)
rs = lambda *s: rng.integers(118, 138, s, dtype=np.uint8)  # E8M0 有效指数域(0xFF=NaN 编码,禁用)
e = m.MOE_MXFP4(cfg, rb(E, 2*I, H//2), rb(E, H, I//2), rs(E, 2*I, H//gk), rs(E, H, I//gk), 0, 0)
dev = torch.device("cuda:0")
torch.manual_seed(3)
h = torch.randn(T, H, device=dev).to(torch.bfloat16)
ids = torch.tensor([[2, 0], [5, 1], [2, 3], [7, 5]], dtype=torch.int32, device=dev)
w = torch.rand(T, K, device=dev)
st = torch.cuda.current_stream().cuda_stream

out_ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
e.cpu_decode(st, T, K, h.data_ptr(), ids.data_ptr(), w.data_ptr(), out_ref.data_ptr())
torch.cuda.synchronize()

lp = xp.LayerPool(e, dev, H, I, gk, E, [2, 5], 0)
sl = lp.slotmap[ids.long()]
res = sl >= E
cpu_ids = torch.where(res, torch.zeros_like(ids), ids)
cpu_w = torch.where(res, torch.zeros_like(w), w)
part_ids = torch.where(res, (sl - E).to(torch.int32), torch.full_like(ids, -1))

from vllm_xiaotu_moe import gpu_prefill as gp
out_pin = torch.zeros(T, H, dtype=torch.float32, device=dev)
e.cpu_decode(st, T, K, h.data_ptr(), cpu_ids.data_ptr(), cpu_w.data_ptr(), out_pin.data_ptr())
partial = gp.gpu_moe_layer(h, part_ids, w, *lp.planes, H, I, K, device=dev, slot=lp.as_slot())
out_pin += partial.to(out_pin.dtype)
torch.cuda.synchronize()
d = (out_pin - out_ref).abs()
rel = float(d.max() / out_ref.abs().max().clamp_min(1e-6))
cos = torch.nn.functional.cosine_similarity(out_pin.flatten(), out_ref.flatten(), dim=0)
print(f"maxrel={rel:.4f} cos={float(cos):.6f} partial_nz={bool((partial != 0).any())}")
