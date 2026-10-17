# scripts/test_pin_stage.py —— 阶段打印版:定位"池路径挂在哪一步"(引擎级,GPU0,无需服务)
# 判据:每步都 print(flush=True);【某步之后不再出现下一行】 ⇒ 挂点就在该步【之内】。
# 与 test_pin_numeric.py 的差别(只加打印 + 可调几何,不改算法):
#   ① 每步 flush 打印 ② XTU_NP 可把 cfg.num_processes 设成与生产一致的 2(默认 1)
# 用法: CUDA_VISIBLE_DEVICES=0 XTU_DEVTREE=<tree> XIAOTU_MOE_THREADS=8 XTU_NP=1 python -u scripts/test_pin_stage.py
import sys, os, numpy as np, torch
sys.path.insert(0, os.environ.get("XTU_DEVTREE", "/home/user/lvllm-tmp/devtree"))
os.environ.setdefault("XIAOTU_PIN", "1")
NP = int(os.environ.get("XTU_NP", "1"))     # 引擎 num_processes:生产 TP=2 ⇒ 2


def p(tag):
    print(f"[STAGE] {tag}", flush=True)


p(f"start XTU_NP={NP} pid={os.getpid()}")
from vllm_xiaotu_moe import expert_pool as xp   # noqa: E402
import xiaotu_moe                               # noqa: E402
p("import 完成")

m = xiaotu_moe.load()
p("xiaotu_moe.load() 完成")
cfg = m.MOEConfigV2()
E, H, I, gk, T, K = 8, 256, 128, 32, int(os.environ.get("XTU_T", "4")), 2
GMIN = int(os.environ.get("XTU_GMIN", "10"))   # 真实配置是 10(mixed_experts.py:1184);B838 假设已验伪
MB = int(os.environ.get("XTU_MB", "256"))      # 真实生产是 max_num_batched_tokens=8192
p(f"params: E={E} H={H} I={I} gk={gk} T={T} K={K} group_min_len={GMIN} max_batch_size={MB}")
for k, v in dict(expert_num=E, hidden_size=H, intermediate_size=I, top_k=K,
                 has_gate_proj=True, num_processes=NP, process_id=0, max_batch_size=MB,
                 max_num_seqs=8, stride=gk, groupN=1, groupK=gk, activation_type=0,
                 group_min_len=GMIN).items():
    setattr(cfg, k, v)
rng = np.random.default_rng(7)
rb = lambda *s: rng.integers(0, 255, s, dtype=np.uint8)          # noqa: E731
rs = lambda *s: rng.integers(118, 138, s, dtype=np.uint8)        # noqa: E731  E8M0 有效指数域
p("cfg 填好,开始构造 engine")
e = m.MOE_MXFP4(cfg, rb(E, 2 * I, H // 2), rb(E, H, I // 2),
                rs(E, 2 * I, H // gk), rs(E, H, I // gk), 0, 0)
p("engine 构造完成")

dev = torch.device("cuda:0")
torch.manual_seed(3)
h = torch.randn(T, H, device=dev).to(torch.bfloat16)
ids = torch.tensor([[2, 0], [5, 1], [2, 3], [7, 5]], dtype=torch.int32, device=dev)
w = torch.rand(T, K, device=dev)
st = torch.cuda.current_stream().cuda_stream

out_ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
e.cpu_decode(st, T, K, h.data_ptr(), ids.data_ptr(), w.data_ptr(), out_ref.data_ptr())
torch.cuda.synchronize()
p(f"① baseline cpu_decode ok, out_ref|max|={float(out_ref.abs().max()):.4f}")

p("② LayerPool 前")
lp = xp.LayerPool(e, dev, H, I, gk, E, [2, 5], 0)
p(f"② LayerPool ok, pinned={len(lp.pinned)} planes={len(lp.planes)}")

sl = lp.slotmap[ids.long()]
res = sl >= E
cpu_ids = torch.where(res, torch.zeros_like(ids), ids)
cpu_w = torch.where(res, torch.zeros_like(w), w)
part_ids = torch.where(res, (sl - E).to(torch.int32), torch.full_like(ids, -1))
p(f"③ 索引算好, hits={int(res.sum())}/{int(res.numel())}")

from vllm_xiaotu_moe import gpu_prefill as gp   # noqa: E402
out_pin = torch.zeros(T, H, dtype=torch.float32, device=dev)
e.cpu_decode(st, T, K, h.data_ptr(), cpu_ids.data_ptr(), cpu_w.data_ptr(), out_pin.data_ptr())
torch.cuda.synchronize()
p(f"④ miss 侧 cpu_decode ok, |max|={float(out_pin.abs().max()):.4f}")

p("⑤ gpu_moe_layer(池支路)前")
partial = gp.gpu_moe_layer(h, part_ids, w, *lp.planes, H, I, K, device=dev, slot=lp.as_slot())
torch.cuda.synchronize()
p(f"⑤ gpu_moe_layer ok, partial|max|={float(partial.abs().max()):.4f} "
  f"nz={bool((partial != 0).any())}")

out_pin += partial.to(out_pin.dtype)
torch.cuda.synchronize()
d = (out_pin - out_ref).abs()
rel = float(d.max() / out_ref.abs().max().clamp_min(1e-6))
cos = float(torch.nn.functional.cosine_similarity(
    out_pin.flatten(), out_ref.flatten(), dim=0))
print(f"[RESULT] maxrel={rel:.4f} cos={cos:.6f} partial_nz={bool((partial != 0).any())}",
      flush=True)
