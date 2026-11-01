#!/usr/bin/env python3
"""内核资源治理的机械门禁（把四轮第三方审计的判据固化）。

用法：
    python3 scripts/check_kernel_alloc_rules.py [--root <vllm 树>] [--quiet]

检查四类（与审计口径一致）：
  C1 常驻且随形状累积：模块级 set/list/dict 被 shape/长度 派生值喂入
  C2 编译期特化：长度类形参仍是 tl.constexpr，或未进 do_not_specialize
  C3 释放时序：empty_cache() 前无 synchronize/record_stream
  C4 契约：内核无 mask 读写但宿主无上界断言（启发式：见下）

设计原则：**宁可误报，不可漏报**；每条告警给出 文件:行号 + 修法。
退出码：0 = 无告警；1 = 有告警。
"""
from __future__ import annotations
import argparse, ast, os, re, sys

# 名字像"随请求变化"的（长度/行数/跨距）
# ⚠️ 必须**精确**匹配：`num_query_heads`/`num_queries_per_kv`/`num_candidates`
#    都是模型或 init 期常量，用前缀匹配会大量误报（实测首版误报 40 条）。
VOLATILE_NAMES = {
    "num_q", "num_rows", "num_tokens", "num_reqs",
    "seq_len", "seq_len_kv", "max_blocks_per_seq",
    "next_n", "logits_width", "token_start", "token_count", "offset",
    "stride_lm", "stride_clb", "stride_qb", "stride_outs", "stride_sqs",
    "stride_group", "stride_hidden",
}
VOLATILE = re.compile(r"$^")   # 保留占位，实际用集合判断
MODULE_LEVEL_CTORS = ("set(", "list(", "dict(", "OrderedDict(")

# 已由第三方审计核定"有界"的模块级容器 → 理由（未登记的新容器仍会被告警）
KNOWN_BOUNDED = {
    "_BUF_CACHE": "2 幂桶化，key 只含列维/常量",
    "_BUF_MOE": "2 幂桶化",
    "_SEG_BUF": "2 幂桶化",
    "_RETIRED_BUF": "显式封顶 16 条",
    "_PIN_CACHE": "key 含 storage 地址 ⇒ 按参数而非按形状",
    "_BMM_BUF": "key 的 ne 只有 {tile, 余数} 两种",
    "_BT_WIDTHS": "显式 _BT_WIDTHS_MAX=8",
    "_PROF_PIN_RESERVE": "profile 期占位，用后即 clear",
    "_PROF_RESERVE": "同上",
    "_CAP_FREE": "显存抽样游标，长度固定",
    "_LAST_USE": "按层号索引（层数固定）",
    "_RING": "按层号/固定槽位索引",
    "_FP8_BUF": "行维 2 幂桶化（本轮修复项）",
    "_FP8_RETIRED": "同 _SEG_RETIRED：2 幂扩容驱动，释放会 UAF",
    "_D2H_STREAM": "按 device 索引",
    "_BND_PIN": "按 (device, 层号) 索引",
    "_ENGRAM_LAST_PENDING": "按层号累积（≤层数），用后 clear()",
    "_CED_GROUP_NAMES": "固定名字集合（常量）",
    "_GP_BUF2": "key = _gp_dev_key(dev) ⇒ 按设备",
    "_GP_BUF2_BYTES": "同上",
    "_GP_BUF2_DENIED": "同上（按设备记录拒绝标记）",
    "_LUT_CACHE": "key 是常量查表，条目数 = LUT 种类",
    "_SEG_RETIRED": "由 2 幂扩容驱动，上界 ~17；释放会引发 UAF，故不 free（见审计 §19 反驳）",
    "_PIN_BND_RETIRED": "同 _SEG_RETIRED：锁页缓冲，释放会 UAF",
    "flashinfer_sparse_index_cache": "每 build 前 clear",
    # 以下均以 **设备号 / 层号 / 固定槽位** 为 key（与请求长度无关）
    "_SLOTS": "按 device 索引（≤GPU 数）",
    "_PREFETCH_STREAMS": "按 device 索引",
    "_LANDING": "按 device 索引",
    "_LANDING_BUF": "按 device 索引",
    "_TILE_RINGS": "按 (device, tile 号) 索引，tile 数固定",
    "_GP_SIDE": "按 device 索引",
    "_GP_LAYERS": "按层号索引",
    "_GP_PROFILE_LAYERS": "按层号索引",
    "_PIN_LOCKS": "按 device 索引",
    "_PIN_BND": "按 (device, 层号) 索引",
    "_BOUND_BUF_LOGGED": "去重标记，key 是档位常量",
    "_DBG": "调试计数器，key 是固定字符串",
    "_RING": "按层号/固定槽位索引",
}


def iter_py(root: str):
    for dirpath, _dirs, files in os.walk(root):
        if any(x in dirpath for x in ("/.git", "__pycache__", "/tests/", "/benchmarks/")):
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def check_file(path: str):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except SyntaxError:
        return []            # 语法错误由别的检查负责
    lines = src.splitlines()
    out = []

    # ---------- C2: 仍为 constexpr 的"易变形参" ----------
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decs = " ".join(ast.unparse(d) for d in node.decorator_list)
        if "triton.jit" not in decs:
            continue
        # do_not_specialize 名单
        dns = set(re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']', decs))
        args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
        for kw in node.args.kwonlyargs:
            if kw.arg in args and kw.arg in dns:
                dns.add(kw.arg)
        # 只挑"声明为 constexpr 且在末尾常量区"的：Triton 习惯用注释界，改为看默认值/注解
        for a in node.args.args + node.args.kwonlyargs:
            ann = ast.unparse(a.annotation) if a.annotation else ""
            if "constexpr" not in ann:
                continue
            if a.arg in VOLATILE_NAMES and a.arg not in dns:
                out.append((path, a.lineno, "C2",
                            f"形参 `{a.arg}` 是 tl.constexpr 且名字像随请求变化，"
                            f"但不在 do_not_specialize 名单里（该内核 {node.name}）"))

    # ---------- C3: empty_cache 前 12 行内无同步（AST 精确识别，避开文档字符串） ----------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "empty_cache"):
            continue
        i = node.lineno - 1
        # 显式白标：`# alloc-audit: init-time`（仅允许用于**服务开始前**的路径）
        window_with_self = "\n".join(lines[max(0, i - 12):i + 1])
        if "alloc-audit: init-time" in window_with_self:
            continue
        if ("synchronize" not in window_with_self and "record_stream" not in window_with_self
                and "wait_event" not in window_with_self and "stream.wait" not in window_with_self):
            out.append((path, node.lineno, "C3",
                        "empty_cache() 前 12 行内没有 synchronize/record_stream/wait_event"
                        "（若在服务开始前执行，加 `# alloc-audit: init-time` 标注）"))

    # ---------- C1: 模块级可变容器被函数变更且无上限 ----------
    #    ⚠️ 首版只看"实参里是否有 .shape"⇒ **别名逃逸**（w = shape[-1]; container.add(w) 漏检）。
    #    现改为：只要是模块级容器、且被任何函数变更 ⇒ 要求该函数有上限判据或显式白标。
    mod_names = set()
    for node in tree.body:
        # ⚠️ `X: dict = {}` 是 AnnAssign（不是 Assign）——首版漏了它，自检立刻抓到
        if isinstance(node, ast.Assign):
            tgts = [t for t in node.targets if isinstance(t, ast.Name)]
            v = ast.unparse(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            tgts = [node.target]
            v = ast.unparse(node.value)
        else:
            continue
        if any(c in v for c in MODULE_LEVEL_CTORS) or v in ("{}", "[]", "set()"):
            mod_names |= {t.id for t in tgts}
    if mod_names:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            targets = []
            if isinstance(fn, ast.Attribute) and fn.attr in ("add", "append", "update", "insert"):
                if isinstance(fn.value, ast.Name) and fn.value.id in mod_names:
                    targets.append(fn.value.id)
            elif isinstance(fn, ast.Attribute) and fn.attr == "setdefault":
                if isinstance(fn.value, ast.Name) and fn.value.id in mod_names:
                    targets.append(fn.value.id)
            for name in targets:
                # 找所属函数
                owner = None
                for anc in ast.walk(tree):
                    if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if anc.lineno <= node.lineno <= (anc.end_lineno or anc.lineno):
                            if owner is None or anc.lineno > owner.lineno:
                                owner = anc
                scope_src = ast.unparse(owner) if owner else ""
                bounded = bool(re.search(r"MAX\w*|len\([^)]*\)\s*(>=|>)\s*\w", scope_src))
                labeled = ("alloc-audit: bounded" in scope_src) or (name in KNOWN_BOUNDED)
                if not (bounded or labeled):
                    out.append((path, node.lineno, "C1",
                                f"模块级容器 `{name}` 被变更（{fn.attr}）但所属函数无上限判据/白标"
                                f"（有界请加 `# alloc-audit: bounded` 并写明理由）"))
        # 下标赋值：_CACHE[key] = value（dict 的典型用法，同样会无界增长）
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in mod_names:
                    owner = None
                    for anc in ast.walk(tree):
                        if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if anc.lineno <= node.lineno <= (anc.end_lineno or anc.lineno):
                                if owner is None or anc.lineno > owner.lineno:
                                    owner = anc
                    scope_src = ast.unparse(owner) if owner else ast.unparse(tree)
                    bounded = bool(re.search(r"MAX\w*|len\([^)]*\)\s*(>=|>)\s*\w", scope_src))
                    labeled = ("alloc-audit: bounded" in scope_src) or (t.value.id in KNOWN_BOUNDED)
                    if not (bounded or labeled):
                        out.append((path, node.lineno, "C1",
                                    f"模块级容器 `{t.value.id}` 被下标赋值但无上限判据/白标"
                                    f"（有界请加 `# alloc-audit: bounded` 并写明理由）"))
                    break

    # ---------- C4: 把 `block_table`/`slot_ids` 交给内核的宿主函数必须有上界断言 ----------
    #    判据（与第 4 轮审计一致）：这类内核按块表/槽位直接算裸地址且**无 mask**，
    #    宿主若不校验 `stride(0)`/`max(seq_lens)`/`max(gather_lens)`/`max(slot_ids)`，
    #    越界输入会变成非法访存（Xid31）。要求：函数体内至少 2 条 assert。
    ADDR_ARGS = {"block_table", "block_tables", "slot_ids"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body_src = ast.unparse(node)
        # 是否把地址类参数传给了某个 triton kernel 调用
        passes_addr = any(re.search(rf"\b{a}\b", body_src) for a in ADDR_ARGS)
        if not passes_addr:
            continue
        has_kernel_call = bool(re.search(r"_kernel\s*\[", body_src))
        if not has_kernel_call:
            continue
        n_assert = len([x for x in ast.walk(node) if isinstance(x, ast.Assert)])
        if n_assert < 2:
            out.append((path, node.lineno, "C4",
                        f"宿主 `{node.name}` 把 block_table/slot_ids 交给内核（无 mask 寻址），"
                        f"但只有 {n_assert} 条 assert；至少需要上界 + 跨距两类校验"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/lvllm/process_data/ref/repos/vllm-mainline")
    ap.add_argument("--plugin", default="/home/user/lvllm/vllm-xiaotu-moe")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--scope-file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "kernel_audit_scope.txt"),
        help="只检查该清单里的文件（默认 scripts/kernel_audit_scope.txt）")
    a = ap.parse_args()

    roots = [r for r in (a.root, a.plugin) if os.path.isdir(r)]
    if not roots:
        print("⚠️ 找不到待查目录（本机没有 vLLM 树？请用 --root/--plugin 指定）")
        return 2

    targets = []
    if a.scope_file and os.path.isfile(a.scope_file):
        for raw in open(a.scope_file, encoding="utf-8"):
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("plugin:"):
                targets.append(os.path.join(a.plugin, ln[len("plugin:"):]))
            else:
                targets.append(os.path.join(a.root, ln))
    else:
        for r in roots:
            targets.extend(iter_py(r))
    targets = [t for t in targets if os.path.isfile(t)]
    print(f"扫描范围：{len(targets)} 个自有文件" if a.scope_file else f"扫描范围：全树 {len(targets)} 文件")
    # ⛔ 防"假绿"：一个文件都没扫到 ⇒ 必须报错，绝不返回 0
    if not targets:
        print("❌ 扫描范围为空（路径拼接错误或清单失效）—— 拒绝给出'通过'结论")
        return 3

    all_findings = []
    for p in targets:
        all_findings.extend(check_file(p))

    if not all_findings:
        print("✅ 未发现违规")
        return 0

    by_class = {}
    for f in all_findings:
        by_class.setdefault(f[2], []).append(f)
    print(f"共 {len(all_findings)} 条：")
    for cls in sorted(by_class):
        print(f"\n[{cls}] {len(by_class[cls])} 条")
        for path, line, _c, msg in by_class[cls]:
            rel = path.split("/vllm-mainline/")[-1].split("/vllm_xiaotu_moe/")[-1]
            print(f"  {rel}:{line}  {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
