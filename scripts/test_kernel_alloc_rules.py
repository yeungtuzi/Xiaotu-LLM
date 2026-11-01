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
# ⛔ 第10轮 N11：旧自检**零回归覆盖** —— 第 9 轮修复前的门禁跑当时 9 个样本 9/9 全过。
#    现在每个样本都对准**一条历史 bug**（括号里是发现它的轮次），
#    并同时给出"合法版本"作为反例，防止判据收紧后变成纯误报机器。
SAMPLES = {
    # ── C2：编译期特化 ──────────────────────────────────────────────
    "C2_basic": ("""
import triton, triton.language as tl
@triton.jit
def bad_kernel(x_ptr, seq_len_kv: tl.constexpr):
    tl.store(x_ptr, seq_len_kv)
""", "C2"),

    # 第8轮阻断项A：`compressed_slot_ids` 前缀 ⇒ 词边界匹配失明
    "C2_prefixed_tensor": ("""
import triton, triton.language as tl
@triton.jit
def k(x_ptr, stride_slot_t: tl.constexpr):
    tl.store(x_ptr, stride_slot_t)
def host(compressed_slot_ids, x):
    k[(1,)](x, compressed_slot_ids.stride(0))
""", "C2"),

    # 第9轮：12 个张量名之外全失明（`dst` 不在名单里）
    "C2_unknown_tensor_name": ("""
import triton, triton.language as tl
@triton.jit
def k(x_ptr, stride_row: tl.constexpr):
    tl.store(x_ptr, stride_row)
def host(dst, x):
    k[(1,)](x, dst.stride(0))
""", "C2"),

    # 第10轮 N2：keyword-only 形参永久盲区
    "C2_keyword_only": ("""
import triton, triton.language as tl
@triton.jit
def k(x_ptr, *, stride_slot_row: tl.constexpr):
    tl.store(x_ptr, stride_slot_row)
def host(slot_ids, x):
    k[(1,)](x, stride_slot_row=slot_ids.stride(0))
""", "C2"),

    # 第6轮 B1：constexpr 形参出现在 do_not_specialize 里 = 空操作
    "C2_dns_on_constexpr": ("""
import triton, triton.language as tl
@triton.jit(do_not_specialize=["seq_len_kv"])
def k3(x_ptr, seq_len_kv: tl.constexpr):
    tl.store(x_ptr, seq_len_kv)
""", "C2"),

    # ── C1：模块级容器无界增长 ──────────────────────────────────────
    "C1_plain_dict": ("""
_SHAPES: dict = {}
def record(n):
    _SHAPES[n] = 1
""", "C1"),

    # 第6轮绕过①：非空字面量初始化
    "C1_nonempty_literal": ("""
_SEEN = {0: None}
def f(n):
    _SEEN[n] = 1
""", "C1"),

    # 第6轮绕过③：AugAssign
    "C1_augassign": ("""
_CNT = {}
def f(k):
    _CNT[k] += 1
""", "C1"),

    # 第7/10轮：defaultdict + 下标目标方法调用
    "C1_defaultdict_subscript": ("""
import collections
_X = collections.defaultdict(list)
def f(n):
    _X[n].append(n)
""", "C1"),

    # ── C3：empty_cache 前必须有同函数内的同步 ──────────────────────
    "C3_no_sync": ("""
import torch
def f():
    torch.cuda.empty_cache()
""", "C3"),

    # ── C4：块表/slot 裸地址取页必须有非恒真上界断言 ────────────────
    "C4_no_assert": ("""
def host(out, block_table):
    _kernel[(1,)](out, block_table)
""", "C4"),

    # 第9轮假绿①：诱饵断言（与块表无关）
    "C4_decoy_assert": ("""
def host(num_reqs, max_num_reqs, block_table):
    assert num_reqs <= max_num_reqs
    out = [1]
    _kernel[(1,)](out, block_table)
""", "C4"),

    # 第7轮 N5：恒真下界
    "C4_tautology": ("""
def host(block_table):
    assert block_table.shape[1] >= 0
    _kernel[(1,)](block_table)
""", "C4"),

    # 第9轮：`delegated` 只看名字出现即豁免
    "C4_delegated_by_name": ("""
def host(block_table, kv_cache):
    _kernel[(1,)](block_table, num_blocks=7)
""", "C4"),

    # 第10轮 N9：一条 slot 断言不能担保同一次启动里的块表
    "C4_partial_coverage": ("""
def host(slot_ids, block_table, cap):
    assert int(slot_ids.max()) < cap
    _kernel[(1,)](slot_ids, block_table)
""", "C4"),


    # ── 第12轮新增机制的反向样本（此前 0 覆盖）────────────────────
    # 白标必须给「内核体内的、引用该实参的」钳位证据；无关 `if pid < n:` 不算
    "C4_whitelabel_unrelated_clamp": ("""
import triton, triton.language as tl
@triton.jit
def _my_kernel(block_table_ptr, n):
    pid = tl.program_id(0)
    if pid < n:
        tl.store(block_table_ptr + pid, 1)
def host(block_table, n):
    # alloc-audit: masked-in-kernel sample.py:5
    _my_kernel[(1,)](block_table, n)
""", "C4"),

    # 白标必须给出 文件:行号
    "C4_whitelabel_no_evidence": ("""
import triton, triton.language as tl
@triton.jit
def _my_kernel(block_table_ptr, n):
    tl.store(block_table_ptr, n)
def host(block_table, n):
    # alloc-audit: masked-in-kernel
    _my_kernel[(1,)](block_table, n)
""", "C4"),


    # ── 合法样本：必须零告警（防判据收紧后变误报机器）─────────────
    "OK_clean": ("""
import triton, triton.language as tl
@triton.jit
def good_kernel(x_ptr, n):
    tl.store(x_ptr, n)
def host(out, block_table, batch_size):
    assert block_table.dim() == 2
    assert block_table.shape[0] >= batch_size
    _kernel[(1,)](out, block_table)
""", None),

    # 第10轮 N6：字面量容量上界 `<= 4096` **不是**恒真，不应被判红
    "OK_literal_cap": ("""
def host(block_table):
    assert block_table.shape[0] <= 4096
    _kernel[(1,)](block_table)
""", None),
}


# ⛔ 第10轮 N11（关键）：**源码级反回归**。
#    行为样本只能证明"当前门禁能抓"，不能证明"旧 bug 不会复活"。
#    这里直接禁止已知 bug 的**代码形态**再出现 —— 一旦有人把它们写回来，
#    自检立刻变红，而不用等到下一轮审计。
import re as _re

GATE_SRC = open(GATE, encoding="utf-8").read()

BUGGY_SHAPES = [
    # 第9轮假绿①：C4 `_is_bound` 提前 return ⇒ "必须引用块表"永不可达
    ("return not _tauto\n            refs =", "C4 _is_bound 提前 return（诱饵断言可过）"),
    # 第9轮假绿②：C2 按"12 个张量名"过滤
    ("for _t in _VOL_TENSOR_NAMES:", "C2 按张量名过滤（dst/lse 等失明）"),
    # 第10轮 N6：把对字面量的比较当恒真
    (r'[<>]=?\s*\d+\s*$', "字面量上界被判恒真（反向误报）"),
    # 第10轮 N2：只取位置形参
    ("_pnames = [x.arg for x in node.args.args]\n", "只取位置形参（keyword-only 盲区）"),
    # 第12轮：白标证据行必须引用被保护实参
    # 第12轮：页跨距证据必须同时含名字
    # 第12轮：委派必须只看函数体（不含签名）
    # 第12轮：白标不能是纯子串（必须行首锚定）
]

KERNEL_ANTI_PATTERNS = [
    # 第9轮 §4-3：双层嵌套 gate ⇒ 实际周期 65536
    ("cache_utils.py", '_bound_check_skip(("gather"', '_bound_check_skip(("entry"', "双层 _bound_check_skip 嵌套"),
]


# ⚠️ 与 BUGGY_SHAPES 相反：这些是**必须存在**的机制（防止被"简化"掉）
REQUIRED_SHAPES = [
    ("_refs_addr = any(", "白标证据行必须引用被保护实参"),
    ('.stride(0)" in _txt and', "页跨距证据必须同时含名字"),
    ("for x in _kfn.body", "委派判据只看函数体语句（不含签名）"),
    ("alloc-audit", "白标机制仍存在（行首锚定）"),
    ("_is_launch = isinstance(f, ast.Subscript)", "内核启动按**下标形态**识别（不靠名字含 kernel）"),
    ("^\\s*#\\s*alloc-audit:\\s*init-time", "C3 init-time 白标必须行首锚定"),
    ("^\\s*#\\s*alloc-audit:\\s*bounded", "C1 bounded 白标必须行首锚定"),
    ("elif isinstance(_v, ast.Attribute)", "C2 支持 `mod.kernel[...]` 属性限定调用"),
]


def check_no_regression() -> int:
    bad = 0
    for pat, why in BUGGY_SHAPES:
        if pat in GATE_SRC:
            print(f"  ❌ 门禁里仍有历史 bug 形态：{why}")
            bad += 1
        else:
            print(f"  ✅ 历史 bug 形态已不存在：{why}")
    for pat, why in REQUIRED_SHAPES:
        if pat in GATE_SRC:
            print(f"  ✅ 必需机制仍在：{why}")
        else:
            print(f"  ❌ 必需机制**被删掉了**：{why}")
            bad += 1
    for fname, pat_a, pat_b, why in KERNEL_ANTI_PATTERNS:
        # ⛔ 第10轮：**读不到文件必须算失败**，不许静默跳过（这正是"跳过即绿"）
        cands = [
            os.environ.get("XTU_VLLM_ROOT", ""),
            "/home/user/lvllm/process_data/ref/repos/vllm-mainline",
            "/root/lvllm/data/repos/vllm-mainline",
        ]
        rel = "vllm/models/deepseek_v41/common/ops/cache_utils.py"
        base = next((os.path.join(c, rel) for c in cands if c and os.path.exists(os.path.join(c, rel))), None)
        if base is None:
            print(f"  ❌ 读不到 {rel}（候选路径均不存在）⇒ 反回归检查**无法执行**，按失败处理")
            bad += 1
            continue
        src = open(base, encoding="utf-8").read()
        # 若 a 与 b 同时出现在同一个代码块（粗略：a 之后 400 字符内出现 b）
        i = src.find(pat_a)
        hit = False
        while i != -1:
            if src.find(pat_b, i, i + 400) != -1:
                hit = True
                break
            i = src.find(pat_a, i + 1)
        if hit:
            print(f"  ❌ 仍有 {why}")
            bad += 1
        else:
            print(f"  ✅ 已无 {why}")
    return bad


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

    # ⛔ 第10轮 N11：行为样本 + **源码级反回归**双保险
    print("\n反回归检查（历史 bug 的代码形态不许复活）：")
    if check_no_regression() != 0:
        return 1

    print("\n✅ 门禁自检通过：四类均可命中，合法样本零误报，历史 bug 形态均已消除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
