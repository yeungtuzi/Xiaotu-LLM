"""负向测试：验证 v41 gather 入口新增的三条上界/跨距断言**真的会拦住违规输入**。

背景：这些内核在 out / k_cache 上无 mask 读写，违规输入过去会变成非法访存（Xid31）。
本测试不追求"跑对"，只追求"违规时给出明确 AssertionError 而不是 IMA"。
"""
import torch, sys
from vllm.models.deepseek_v41.common.ops.cache_utils import dequantize_and_gather_k_cache

dev = "cuda:0"
BS, TOK = 64, 584          # V4 plain 记录宽度（576 会被默认分支正确拒绝）
NB = 8

def build(nreq=2, mblk=8, seq=128, gather=128, stride0=None, slot=0):
    k_cache = torch.zeros(NB, BS, TOK, dtype=torch.uint8, device=dev)
    seq_lens = torch.full((nreq,), seq, dtype=torch.int32, device=dev)
    gl = torch.full((nreq,), gather, dtype=torch.int32, device=dev)
    bt = torch.arange(mblk, dtype=torch.int32, device=dev).repeat(nreq, 1)
    if stride0 is not None:                     # 造一个行内有 padding 的块表
        big = torch.zeros(nreq, stride0, dtype=torch.int32, device=dev)
        big[:, :mblk] = bt
        bt = big[:, :mblk]                      # stride(0)=stride0 ≠ shape[-1]=mblk
    out = torch.empty(nreq, seq, 512, dtype=torch.bfloat16, device=dev)
    return out, k_cache, seq_lens, gl, bt

def run(tag, **kw):
    try:
        out, k_cache, seq_lens, gl, bt = build(**kw)
        dequantize_and_gather_k_cache(out, k_cache, seq_lens, gl, bt, BS, kw.pop("offset", 0))
        torch.cuda.synchronize()
        print(f"  {tag}: 通过（无异常）")
        return "pass"
    except AssertionError as e:
        msg = str(e).split("\n")[0][:90]
        print(f"  {tag}: ✅ 被断言拦住 → {msg}")
        return "assert"
    except Exception as e:
        print(f"  {tag}: ⚠️ 其它异常 {type(e).__name__}: {str(e)[:90]}")
        return "other"

print("=== 正向：合法输入应当通过 ===")
run("合法输入")
print("\n=== 负向：三类违规都应被拦住 ===")
run("块表行内有 padding（stride(0)≠width）", stride0=16)
run("gather_lens 超出 out 容量", gather=9999)
run("seq_len 超出块表覆盖范围", seq=99999)
