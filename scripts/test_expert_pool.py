# scripts/test_expert_pool.py —— 池槽装配逐字节对拍(micro-engine,GPU0)
# 判据:逐槽 noraw 装配 == 整层装配切片的对应槽位(4 plane 逐字节)+ slotmap 语义。
import sys, os, numpy as np, torch
sys.path.insert(0, os.environ.get("XTU_DEVTREE", "/home/user/lvllm-tmp/devtree"))
os.environ.setdefault("XIAOTU_PIN", "1")
from vllm_xiaotu_moe import expert_pool as xp, gpu_prefill as gp
import xiaotu_moe

m = xiaotu_moe.load()
cfg = m.MOEConfigV2()
E, H, I, gk = 8, 256, 128, 32
for k, v in dict(expert_num=E, hidden_size=H, intermediate_size=I, top_k=2,
                 has_gate_proj=True, num_processes=1, process_id=0, max_batch_size=256,
                 max_num_seqs=8, stride=gk, groupN=1, groupK=gk, activation_type=0).items():
    setattr(cfg, k, v)
rng = np.random.default_rng(7)
rb = lambda *s: rng.integers(0, 255, s, dtype=np.uint8)
e = m.MOE_MXFP4(cfg, rb(E, 2*I, H//2), rb(E, H, I//2), rb(E, 2*I, H//gk), rb(E, H, I//gk), 0, 0)
dev = torch.device("cuda:0")
full = (torch.empty((E, H//2, 2*I), dtype=torch.uint8, device=dev),
        torch.empty((E, H//gk, 2*I), dtype=torch.uint8, device=dev),
        torch.empty((E, I//2, H), dtype=torch.uint8, device=dev),
        torch.empty((E, I//gk, H), dtype=torch.uint8, device=dev))
gp.kmajor_from_engine_shards_noraw(e, dev, H, I, E, gk, dst=full)
torch.cuda.synchronize()
lp = xp.LayerPool(e, dev, H, I, gk, E, [2, 5], 0)
ok = all(torch.equal(lp.planes[i][j], full[i][x]) for i in range(4) for j, x in enumerate([2, 5]))
sm_ok = bool((lp.slotmap[[2, 5]] == E + torch.tensor([0, 1], device=dev)).all())
untouched = bool((lp.slotmap[[0, 1, 3, 4, 6, 7]] == torch.tensor([0, 1, 3, 4, 6, 7], device=dev)).all())
print(f"BYTE-EQUAL={ok} slotmap-pinned={sm_ok} unpinned-identity={untouched}")
