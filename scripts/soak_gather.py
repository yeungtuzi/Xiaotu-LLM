"""组件级 soak（第二组）：变量块表宽度，检验 _dequantize_and_gather_k_kernel 是否累积。"""
import os, sys, time, random, resource, torch
T = os.environ.get("TRITON_CACHE_DIR", "/tmp/soak_g_cache")
vc = lambda: len([d for d in os.listdir(T) if os.path.isdir(os.path.join(T, d))])
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

from vllm.models.deepseek_v41.common.ops.cache_utils import dequantize_and_gather_k_cache

dev = "cuda:0"
BS = 64                      # cache_block_size
TOK = 576                    # fp8_ds_mla 行宽（512 fp8 + 16 scale + 48 其余）
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

print(f"起始: 变体 {vc()}  RSS {rss():.0f} MB")
samples = []
t0 = time.time()
for i in range(ITERS):
    nreq = random.choice([1, 2, 3, 4])
    mblk = random.choice([8, 16, 32, 64, 128, 256, 512, 1024])   # 块表宽度（疑似形状轴）
    nblk = max(4, mblk)
    k_cache = torch.zeros(nblk, BS, TOK, dtype=torch.uint8, device=dev)
    seq_lens = torch.full((nreq,), min(mblk * BS, 4096), dtype=torch.int32, device=dev)
    gather = torch.full((nreq,), min(mblk * BS, 4096), dtype=torch.int32, device=dev)
    block_table = torch.arange(mblk, dtype=torch.int32, device=dev).repeat(nreq, 1)
    out = torch.empty(nreq, min(mblk * BS, 4096), 512, dtype=torch.bfloat16, device=dev)
    try:
        dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather, block_table, BS, 0)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  ⚠️ 调用失败: {type(e).__name__}: {str(e)[:110]}"); break
    del k_cache, seq_lens, gather, block_table, out
    if (i + 1) % max(1, ITERS // 5) == 0:
        a = torch.cuda.memory_allocated() / 2**20
        r = torch.cuda.memory_reserved() / 2**20
        f, _ = torch.cuda.mem_get_info()
        samples.append((a, r, f / 2**20, rss(), vc()))
        print(f"  it {i+1:>4}: alloc {a:7.1f} reserved {r:7.1f} free {f/2**20:7.1f} "
              f"rss {rss():6.0f} MB variants {vc()}")
print(f"\n耗时 {time.time()-t0:.1f}s")
if len(samples) >= 2:
    f, l = samples[0], samples[-1]
    print(f"reserved {l[1]-f[1]:+.1f} MiB | free {l[2]-f[2]:+.1f} MiB | RSS {l[3]-f[3]:+.0f} MB | 变体 +{l[4]-f[4]}")
