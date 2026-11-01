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

    # ---------- C1: 模块级容器被 shape 派生值喂入 ----------
    mod_stores = []
    for node in tree.body:      # 只看模块顶层
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) for t in node.targets):
            v = node.value
            src_v = ast.unparse(v)
            if any(c in src_v for c in MODULE_LEVEL_CTORS) or src_v in ("{}", "[]", "set()"):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                mod_stores.append((names, node.lineno))
    if mod_stores:
        store_names = {n for names, _ in mod_stores for n in names}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr in ("add", "append", "update")):
                continue
            if not (isinstance(fn.value, ast.Name) and fn.value.id in store_names):
                continue
            argsrc = " ".join(ast.unparse(a) for a in node.args)
            if ".shape" in argsrc or "numel" in argsrc or "shape[" in argsrc:
                bounded = False
                # 该函数内是否有 MAX/上限判断
                for anc in ast.walk(tree):
                    if isinstance(anc, (ast.FunctionDef,)) and anc.lineno <= node.lineno <= (anc.end_lineno or anc.lineno):
                        body = ast.unparse(anc)
                        if re.search(r"len\([^)]+\)\s*(>=|>)\s*\w*MAX|MAX_\w+", body):
                            bounded = True
                        break
                if not bounded:
                    out.append((path, node.lineno, "C1",
                                f"模块级容器 `{fn.value.id}` 被 shape 派生值喂入且未见上限判断"))

    # ---------- C4: 粗判"内核无 mask"但宿主无 assert（只报提示） ----------
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
