#!/usr/bin/env python3
"""自检：给门禁喂"已知违规"的样本，验证它四类都能抓到（防假绿/假阳性）。

用法：python3 scripts/test_kernel_alloc_rules.py
退出码：0 = 四类全部被正确识别；1 = 有类别漏检或误报
"""
from __future__ import annotations
import os, subprocess, sys, tempfile, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "check_kernel_alloc_rules.py")

# 每份样本都应当触发**恰好**对应的类别
# ⚠️ 已知缺口（不放入样本，避免误报"通过"）：
#    C3 不做可达性分析 ⇒ 同函数内 if/else 不可达分支里的 synchronize
#    仍会给 empty_cache 背书（第7轮审计实测）。修它需要控制流可达性分析。
SAMPLES = {
    "C2_expr": ("""
import triton, triton.language as tl
@triton.jit
def bad_kernel(x_ptr, seq_len_kv: tl.constexpr):
    tl.store(x_ptr, seq_len_kv)
""", "C2"),

    "C1_expr": ("""
_SHAPES: dict = {}
def record(n):
    _SHAPES[n] = 1
""", "C1"),

    "C3_expr": ("""
import torch
def f():
    torch.cuda.empty_cache()
""", "C3"),

    "C4_expr": ("""
def host(out, block_table):
    assert out.shape[0] > 0
    _kernel[(1,)](out, block_table)
""", "C4"),

    "C2_positional": ("""
import triton, triton.language as tl
@triton.jit
def k2(x_ptr, stride_q_t: tl.constexpr):
    tl.store(x_ptr, stride_q_t)
def host(q, x):
    k2[(1,)](x, q.stride(0))
""", "C2"),

    "C2_dns_on_constexpr": ("""
import triton, triton.language as tl
@triton.jit(do_not_specialize=["seq_len_kv"])
def k3(x_ptr, seq_len_kv: tl.constexpr):
    tl.store(x_ptr, seq_len_kv)
""", "C2"),

    "C4_tautology": ("""
def host(block_table, batch_size):
    assert block_table.shape[1] >= 0
    _kernel[(1,)](block_table)
""", "C4"),

    "C1_defaultdict": ("""
import collections
_X = collections.defaultdict(list)
def f(n):
    _X[n].append(n)
""", "C1"),

    "OK_expr": ("""
import triton, triton.language as tl
@triton.jit
def good_kernel(x_ptr, n):
    tl.store(x_ptr, n)
def host(out, block_table, batch_size):
    assert block_table.dim() == 2
    assert block_table.shape[0] >= batch_size
    _kernel[(1,)](out, block_table)
""", None),
}


def run_gate(path: str) -> tuple[int, str]:
    # 变异测试用合成样本 ⇒ 必须跳过"白名单名字必须存在"的检查
    r = subprocess.run([sys.executable, GATE, "--scope-file", path, "--no-whitelist-check"],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="gate_self_")
    problems = []
    for name, (code, expect) in SAMPLES.items():
        path = os.path.join(tmp, "sample.py")
        open(path, "w").write(textwrap.dedent(code))
        scope = os.path.join(tmp, "scope.txt")
        open(scope, "w").write(path + "\n")
        rc, out = run_gate(scope)
        hit = [c for c in ("C1", "C2", "C3", "C4") if f"[{c}]" in out]
        if expect is None:
            ok = rc == 0 and not hit
            print(f"  {name}: {'✅ 零告警' if ok else f'❌ 误报 {hit}'}")
            if not ok:
                problems.append(f"{name} 误报 {hit}\n{out}")
        else:
            ok = expect in hit
            print(f"  {name}: {'✅ 命中 ' + expect if ok else f'❌ 未命中 {expect}（实得 {hit}）'}")
            if not ok:
                problems.append(f"{name} 未命中 {expect}（实得 {hit}）\n{out}")
    if problems:
        print("\n失败详情：")
        for p in problems:
            print(p[:600])
        return 1
    print("\n✅ 门禁自检通过：四类均可命中，合法样本零误报")
    return 0


if __name__ == "__main__":
    sys.exit(main())
