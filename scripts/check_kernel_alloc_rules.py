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
# 已核"有界"的 constexpr 形参（值会变，但取值集合有限）→ 理由
KNOWN_BOUNDED_CONSTEXPR = {
    "PADDED_TOP_K": "2 的幂桶化（审计第 5 轮已核：combine_topk 的 CompileKey 字段均有界）",
}

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
    "_FP8_BUF": "行维 2 幂桶化（本轮修复项）",
    "_FP8_RETIRED": "同 _SEG_RETIRED：2 幂扩容驱动，释放会 UAF",
    "_D2H_STREAM": "按 device 索引",
    "_ENGRAM_LAST_PENDING": "按层号累积（≤层数），用后 clear()",
    "_CED_GROUP_NAMES": "每 KV 配置的层名集合，会 clear()+update（按配置数，有限）",
    "_GP_BUF2": "key = _gp_dev_key(dev) ⇒ 按设备",
    "_GP_BUF2_BYTES": "同上",
    "_GP_BUF2_DENIED": "同上（按设备记录拒绝标记）",
    # ── mixed_experts：审计第 5 轮逐个读过，均按 device / 层号索引（原靠 MAX 子串蒙过，
    #    现显式登记，杜绝"巧合式放行"）──
    "_GPF_INFO": "key = _dev.index（mixed_experts.py:2249/2252）",
    "_GP_EVENTS": "按 device 索引（:358/2649）",
    "_GP_PREFETCH_ON": "按 device 索引（:494）",
    "_GP_READY": "key = (device_key, 下一槽位序号)（:495/2594/2673）",
    "_GP_SLOT": "key = (device, slot)，slot ∈ {0,1}（:496/2654）",
    "_GP_SLOT_FREE": "同 _GP_SLOT：key = device（:497，经 setdefault 写入）",
    "_PROFILE_PREALLOC_DONE": "集合，元素 = device key（:509/2226）",
    "_GP_PF_WARNED": "告警去重，key = device（:512/2366）",
    "_PF_HB": "心跳计数，key 是字符串字面量 \"t\"/\"n\"（:514/2475-2476）",

    "_LUT_CACHE": "key 是 device（gpu_prefill.py:57）",
    "_SEG_RETIRED": "由 2 幂扩容驱动，上界 ~17；释放会引发 UAF，故不 free（见审计 §19 反驳）",
    "_PIN_BND_RETIRED": "同 _SEG_RETIRED：锁页缓冲，释放会 UAF",
    # 以下均以 **设备号 / 层号 / 固定槽位** 为 key（与请求长度无关）
    "_SLOTS": "key = (dev.index,) + shapes（限量 MAXSEQS≤4 × 设备数）",
    "_PREFETCH_STREAMS": "按 device 索引",
    "_LANDING": "key = (device, which∈{0..3})",
    "_LANDING_BUF": "按 device 索引",
    "_TILE_RINGS": "key = (device, 全 shapes, nslots) ⇒ tile 数固定、nslots 固定",
    "_GP_SIDE": "按 device 索引",
    "_GP_LAYERS": "按层号索引",
    "_GP_PROFILE_LAYERS": "按层号索引",
    "_PIN_LOCKS": "key = (storage_ptr, offset, shape, dtype, tag) ⇒ 按参数而非按请求长度",
    "_PIN_BND": "按 (device, 层号) 索引",
    "_BOUND_BUF_LOGGED": "去重标记，key 是缓冲区完整 key（只记录已打印过）",
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


def _slice_of(node) -> str:
    """取节点对应的**源码原文**（含注释），用于匹配 `# alloc-audit: ...` 白标。"""
    if node is None:
        return ""
    lo = getattr(node, "lineno", 1)
    hi = getattr(node, "end_lineno", lo)
    return "\n".join(_LINES[lo - 1:hi])


_LINES: list = []


def check_file(path: str):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except SyntaxError:
        return []            # 语法错误由别的检查负责
    lines = src.splitlines()
    global _LINES
    _LINES = lines
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
            # ⛔ 审计 #8：只认写死名字表 ⇒ `width_from_shape: tl.constexpr` 一律漏。
            #    增加第二判据：**调用点实参表达式**里出现 `.shape` / `.stride(`
            #    ⇒ 该 constexpr 事实上随形状变化。
            _shapey = False
            if a.arg not in dns:
                _pat = re.compile(rf"{re.escape(node.name)}\s*\[")
                for _m in _pat.finditer(src):
                    _seg = src[_m.end():_m.end() + 1200]
                    _km = re.search(rf"\b{re.escape(a.arg)}\s*=\s*([^,\n]{{0,160}})", _seg)
                    if not _km:
                        continue
                    _v = _km.group(1)
                    # ⚠️ 只认"**随请求变化的张量**上的 `.shape[`"：
                    #    `k_cache.stride(0)` / `out.stride(1)` 这类跨距在配置固定时是常量，
                    #    用 `.stride(` 当证据会 24/24 全误报（实测）。
                    _VOL_TENSORS = ("block_table", "block_tables", "out", "topk_indices",
                                    "slot_ids", "acc", "max_score", "cu_seqlens",
                                    "logits", "q", "kv")
                    if any(re.search(rf"\b{t}\b", _v) for t in _VOL_TENSORS) and ".shape[" in _v:
                        _shapey = True
                        break
            if (a.arg in VOLATILE_NAMES or _shapey) and a.arg not in dns \
                    and a.arg not in KNOWN_BOUNDED_CONSTEXPR:
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
        # ⛔ 审计 #3：旧判据是**文本**匹配 ⇒ 注释里写一句 "synchronize" 就算通过。
        #    改为只看 AST 里的真实调用节点（注释天然不在 AST 中）。
        real_sync = False
        for anc in ast.walk(tree):
            if not isinstance(anc, ast.Call) or anc.lineno > node.lineno or anc.lineno < node.lineno - 12:
                continue
            f = anc.func
            nm = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if nm in ("synchronize", "record_stream", "wait_event", "wait_stream"):
                real_sync = True
                break
        if not real_sync:
            out.append((path, node.lineno, "C3",
                        "empty_cache() 前 12 行内没有 synchronize/record_stream/wait_event"
                        "（若在服务开始前执行，加 `# alloc-audit: init-time` 标注）"))

    # ---------- C1: 模块级可变容器被函数变更且无上限 ----------
    #    ⚠️ 首版只看"实参里是否有 .shape"⇒ **别名逃逸**（w = shape[-1]; container.add(w) 漏检）。
    #    现改为：只要是模块级容器、且被任何函数变更 ⇒ 要求该函数有上限判据或显式白标。
    # ⛔ 审计 #5：旧版只看 `tree.body` 顶层 ⇒ 写在 `try:` 块里的模块级容器完全不可见。
    _top = []
    def _collect(nodes):
        for n in nodes:
            # ⛔ 递归只进 Try/If/With 这类**语句块**；绝不进 FunctionDef/ClassDef 的 body
            #    （否则函数内局部变量会被误当成模块级容器：实测 98 条误报）
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            _top.append(n)
            for attr in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(n, attr, None)
                if isinstance(sub, list):
                    _collect([x for x in sub if isinstance(x, ast.stmt)])
    _collect(tree.body)
    mod_names = set()
    for node in _top:
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
                # ⛔ 审计 #6：`MAX\w*` 子串会被任意含 MAX 的标识符"蒙过"
                #    （实据：mixed_experts 巨型函数里的 _DEFER_MAXQ 给 9 个容器背书）。
                #    ⇒ 取消启发式，只认：①KNOWN_BOUNDED 白名单 ②源码切片里的显式白标。
                scope_src = _slice_of(owner) if owner else ""
                labeled = ("alloc-audit: bounded" in scope_src) or (name in KNOWN_BOUNDED)
                if not labeled:
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
                    scope_src = _slice_of(owner) if owner else ""
                    labeled = ("alloc-audit: bounded" in scope_src) or (t.value.id in KNOWN_BOUNDED)
                    if not labeled:
                        out.append((path, node.lineno, "C1",
                                    f"模块级容器 `{t.value.id}` 被下标赋值但无上限判据/白标"
                                    f"（有界请加 `# alloc-audit: bounded` 并写明理由）"))
                    break

    # ---------- C4: kernel launch 的实参里出现块表/slot，且宿主无上界断言 ----------
    #    ⚠️ 审计 #5 + 第27轮实测：旧版按"函数体内出现过 kernel 调用 + 出现过块表"匹配
    #    ⇒ 把"把块表交给**已带断言的辅助函数**"也算漏检（indexer.build 就是这种）。
    #    现改为：只认 **kernel 调用的实参里真的出现块表表达式**。
    ADDR_ARGS = ("block_table", "block_tables", "slot_ids", "indexer_block_table")
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        launches = []
        for c in ast.walk(node):
            if not isinstance(c, ast.Call):
                continue
            f = c.func
            # 形如 name[...](...) 或 name(...) 且名字像内核
            fname = ""
            if isinstance(f, ast.Subscript):
                fname = ast.unparse(f.value)
            elif isinstance(f, ast.Name):
                fname = f.id
            elif isinstance(f, ast.Attribute):
                fname = f.attr
            if not fname:
                continue
            if not (fname.endswith("_kernel") or fname.endswith("_kernel_") or "kernel" in fname):
                continue
            argtxt = " ".join(ast.unparse(a) for a in c.args) + " " + " ".join(
                f"{k.arg}={ast.unparse(k.value)}" for k in c.keywords if k.arg)
            if any(re.search(rf"\b{a}\b", argtxt) for a in ADDR_ARGS):
                launches.append((c.lineno, argtxt))
        if not launches:
            continue
        body_src = ast.unparse(node)

        def _is_bound(a: ast.Assert) -> bool:
            t = ast.unparse(a.test)
            return any(k in t for k in ("max(", "min(", ".shape[", ".stride(", "numel", ">=", "<"))

        n_assert = len([x for x in ast.walk(node) if isinstance(x, ast.Assert) and _is_bound(x)])
        # 边界可能由内核内校验：宿主把行数/块数显式传给内核 ⇒ 不算漏
        delegated = any(re.search(r"num_table_rows|max_blocks|num_blocks|max_blocks_per_seq|"
                                  r"n_blocks|num_pages|table_rows|BLOCK_COLS", t) for _l, t in launches)
        # 已人工复核：内核内部对块表/slot 的访问自带 mask ⇒ 用源码切片白标
        _slice = "\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
        if "alloc-audit: masked-in-kernel" in _slice:
            continue
        if n_assert >= 2 or delegated:
            continue
        out.append((path, launches[0][0], "C4",
                    f"宿主 `{node.name}` 把块表/slot 作为**内核实参**传入（无 mask 寻址），"
                    f"但只有 {n_assert} 条上界/跨距断言"))
    return out


def main():
    ap = argparse.ArgumentParser()
    # ⛔ 审计 #2：默认根曾写死远端路径 ⇒ 在其它机器上 exit 2，仓库里没有可复现性。
    #    现按「环境变量 → 常见候选」顺序探测；都没有则**报错并给出如何指定**。
    def _pick(env: str, cands: list) -> str:
        v = os.environ.get(env)
        if v and os.path.isdir(v):
            return v
        for c in cands:
            if os.path.isdir(c):
                return c
        return cands[0]

    _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=_pick("XTU_VLLM_ROOT", [
        "/home/user/lvllm/process_data/ref/repos/vllm-mainline",
        os.path.join(_here, "data/repos/vllm-mainline"),
        "/root/lvllm/data/repos/vllm-mainline",
    ]))
    ap.add_argument("--plugin", default=_pick("XTU_PLUGIN_ROOT", [
        "/home/user/lvllm/vllm-xiaotu-moe",
        _here,
    ]))
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
    # ⛔ 审计 #1：缺失项曾被静默丢弃（清单 32 条只扫 31 条，且"31"被当成全量）
    missing = [t for t in targets if not os.path.isfile(t)]
    targets = [t for t in targets if os.path.isfile(t)]
    print(f"扫描范围：{len(targets)} 个自有文件" if a.scope_file else f"扫描范围：全树 {len(targets)} 文件")
    # ⛔ 防"假绿"：一个文件都没扫到 ⇒ 必须报错，绝不返回 0
    if not targets:
        print("❌ 扫描范围为空（路径拼接错误或清单失效）—— 拒绝给出'通过'结论")
        return 3
    if missing:
        print(f"❌ 清单中有 {len(missing)} 条路径不存在（不得静默跳过）：")
        for m in missing:
            print(f"   {m}")
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
