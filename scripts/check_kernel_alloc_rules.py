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
    "_BOUND_CHECKED": "显式 _BOUND_CHECKED_MAX=16（第6轮审计 N4 引入的 capture/热路径 gate）",
    # ── 第6轮绕过①③（非空字面量初始化 / AugAssign）生效后浮出的一批 ──
    #    全部逐条读过定义与写入点：均为**固定字符串 key 的统计/标志**，键集合由字面量决定。
    "_PH": "固定字符串 key 的统计字典（gpu_prefill_fp8.py:996，键为 'decode'/'gather'/'dma_k' 等字面量）",
    "_EV": "固定 key；acc 的二级键来自 _evt_add 的字符串字面量（:965/693 等）",
    "_SC": "单键 {'bounds': ...}（:917）",
    "_CED_STATE": "固定 key 的状态字典（mainline_shims.py:1444）",
    "_PWAL_DIAG": "固定 key {'n','src'}（mainline_shims.py:66）",
    "_LT_ACC": "固定 key 的计数器（mixed_experts.py:75）",
    "_PF_CALL": "单键 {'n'}（mixed_experts.py:515）",
    "_STAGE_ACC": "固定 key 的统计（gpu_prefill.py:2088；k 来自阶段名字面量）",
    "_PIN_DONE": "单键 {'n': -1}（gpu_prefill.py:2253）",
    "_IN_WARMUP": "单元素标志列表（gpu_prefill.py:1227，只写 [0]）",
    "_IN_STARTUP": "单元素标志列表（gpu_prefill.py:1222，只写 [0]）",
    "_T_ATTN": "固定 key {'t','n'}（deepseek_v4/attention.py:133）",
    "_XTU_MQA_SEEN": "固定字符串 key {'triton','torch'}（sm12x_deep_gemm_fallbacks.py:458）",
    "_CAP_WARNED": "单元素标志列表（expert_pool.py:52，只写 [0]）",
    "_STEP": "单元素计数器（gpu_prefill.py:506，只写 [0]）",
    "_LAST_REPORT": "单元素时间戳（gpu_prefill.py:615）",
    "_CLEANER": "固定 key（含 'lock'）的线程协调字典（gpu_prefill.py:583）",
    "_STAGE_EV": "固定 key {'pend'}（gpu_prefill.py:2127）",
    "_DDB": "单键 {'n'}（gpu_prefill_fp8.py:1007，位于模块级块内）",
    "_UPSTREAM_SEG": "单键 {'n'}（mainline_shims.py:466）",
    "_PROF_PIN_RESERVE": "profile 期占位，用后即 clear",
    "_PROF_RESERVE": "同上",
    "_CAP_FREE": "是**栈**（push/pop 配对），不是定长——配对已修（#14）",
    "_LAST_USE": "key = 缓冲 key（无层号）——与缓冲 key 同域，有限",
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
    "_GP_PREFETCH_ON": "⚠️ 死容器：全文件只有定义、无读写（:494）⇒ 可删",
    "_GP_READY": "key = (device_key, 层号)（第二项是层号，不是槽位序号）",
    "_GP_SLOT": "key 只有 device，slot 是 **value**（:2596/2654）",
    "_GP_SLOT_FREE": "同 _GP_SLOT：key = device（:497，经 setdefault 写入）",
    "_PROFILE_PREALLOC_DONE": "集合，元素是 (E, I)（不是 device）——按模型几何，有限",
    "_GP_PF_WARNED": "告警去重，key = device（:512/2366）",
    "_PF_HB": "有 4 个字符串 key（\"t\"/\"n\" 等，:514/2475-2476）",

    "_LUT_CACHE": "key 是 device（gpu_prefill.py:57）",
    "_SEG_RETIRED": "由 2 幂扩容驱动，上界 ~17；释放会引发 UAF，故不 free（见审计 §19 反驳）",
    "_PIN_BND_RETIRED": "同 _SEG_RETIRED：锁页缓冲，释放会 UAF",
    # 以下均以 **设备号 / 层号 / 固定槽位** 为 key（与请求长度无关）
    "_SLOTS": "key = 权重 shapes（与 MAXSEQS 无关）——按模型几何，有限",
    "_PREFETCH_STREAMS": "按 device 索引",
    "_LANDING": "key = (device, which∈{0..3})",
    "_LANDING_BUF": "key 含 device 与 shapes（不是纯 device）",
    "_TILE_RINGS": "key = (device, 全 shapes, nslots) ⇒ tile 数固定、nslots 固定",
    "_GP_SIDE": "按 device 索引",
    "_GP_LAYERS": "按层号索引",
    "_GP_PROFILE_LAYERS": "按层号索引",
    "_PIN_LOCKS": "key = (storage_ptr, offset, shape, dtype, tag) ⇒ 按参数而非按请求长度",
    "_PIN_BND": "key 只含 dtype（不是 device+层号）——当前有界，但理由需更正",
    "_BOUND_BUF_LOGGED": "去重标记，key 是缓冲区完整 key（只记录已打印过）",
    "_DBG": "调试计数器，key 是固定字符串",
    "_RING": "key = 权重 shapes（不是层号）——当前有界，但理由需更正",
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



_VOL_TENSOR_NAMES = ("block_table", "block_tables", "out", "topk_indices", "slot_ids",
                     "acc", "max_score", "cu_seqlens", "logits", "q", "kv", "slot_mapping")


def _check_call_sites(tree, fname: str, pnames: list, target_param: str) -> bool:
    """该内核的**任一调用点**把 target_param 传成了"随请求维张量的 .shape[ / .stride(0)"吗？

    同时覆盖**关键字**与**位置**实参（第7轮审计阻断项①：旧版只认关键字）。
    """
    try:
        idx = pnames.index(target_param)
    except ValueError:
        return False
    for c in ast.walk(tree):
        if not isinstance(c, ast.Call):
            continue
        f = c.func
        if not (isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name) and f.value.id == fname):
            continue
        expr = None
        # 关键字优先
        for kw in c.keywords:
            if kw.arg == target_param:
                expr = ast.unparse(kw.value)
                break
        if expr is None and idx < len(c.args):
            expr = ast.unparse(c.args[idx])
        if expr is None:
            continue
        if not any(re.search(rf"\b{t}\b", expr) for t in _VOL_TENSOR_NAMES):
            continue
        if ".shape[" in expr or ".stride(0)" in expr:
            return True
    return False


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
                # ⛔ 第7轮阻断项①：上一版只在**关键字实参**里找 `arg=expr`，
                #    而这些内核全部**位置传参** ⇒ 结构性失明（把 constexpr 改回去也照样绿）。
                #    这里用 AST 取调用点，把位置实参按**形参顺序**映射到名字。
                _pnames = [x.arg for x in node.args.args]
                _shapey = _check_call_sites(tree, node.name, _pnames, a.arg)
                _v = ""
                if not _shapey:
                    _pat = None
                else:
                    _v = ""
            if False:
                    # ⚠️ 只认"**随请求变化的张量**上的 `.shape[`"：
                    #    `k_cache.stride(0)` / `out.stride(1)` 这类跨距在配置固定时是常量，
                    #    用 `.stride(` 当证据会 24/24 全误报（实测）。
                    _VOL_TENSORS = ("block_table", "block_tables", "out", "topk_indices",
                                    "slot_ids", "acc", "max_score", "cu_seqlens",
                                    "logits", "q", "kv")

            # ⛔ 第7轮 N4：`arg in dns` 曾当**豁免**用 —— 但 constexpr 形参进 do_not_specialize
            #    是**空操作**（Triton 只对非 constexpr 生效）⇒ 这恰恰是 B1 的 bug 特征，
            #    应当**额外报警**，而不是放行。
            # ⚠️ 条件注解（`X: tl.constexpr if COND else None`）静态无法求值 ⇒ 跳过，不误报。
            #    （engram.py:650 就是这种写法，且 dns 列表已按条件构造，本来就是对的）
            if " if " in ann and " else " in ann:
                continue
            if a.arg in dns:
                out.append((path, a.lineno, "C2",
                            f"形参 `{a.arg}` 是 tl.constexpr 却出现在 do_not_specialize 里 —— "
                            f"对 constexpr 是**空操作**（B1 根因），应降级为运行时标量"
                            f"（该内核 {node.name}）"))
                continue
            if (a.arg in VOLATILE_NAMES or _shapey) \
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
        # ⛔ 第6轮 #3：旧判据按**行号窗口**（可跨函数）⇒ 前一个函数里的 synchronize
        #    会给后一个函数的 empty_cache 背书。改为只认**同一函数体内**、且在该调用之前。
        real_sync = False
        _owner = None
        for _f in ast.walk(tree):
            if isinstance(_f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _f.lineno <= node.lineno <= (_f.end_lineno or _f.lineno):
                    if _owner is None or _f.lineno > _owner.lineno:
                        _owner = _f
        _lo = _owner.lineno if _owner is not None else node.lineno - 12
        for anc in ast.walk(tree):
            if not isinstance(anc, ast.Call) or anc.lineno > node.lineno or anc.lineno < _lo:
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
        # ⛔ 第6轮绕过①：`_SEEN = {0: None}`（非空字面量）曾不被识别
        _v0 = v.lstrip()
        _is_lit = _v0.startswith("{") or _v0.startswith("[") or _v0.startswith("set(") or _v0.startswith("dict(")
        if any(c in v for c in MODULE_LEVEL_CTORS) or _is_lit:
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
                # ⛔ 第6轮绕过②：曾"函数里出现一句 # alloc-audit: bounded ⇒ 该函数内所有容器放行"。
                #    改为白标必须**指名**该容器：`# alloc-audit: bounded <name>`
                _lab = bool(re.search(rf"alloc-audit:\s*bounded[^\n]*\b{re.escape(name)}\b", scope_src))
                labeled = _lab or (name in KNOWN_BOUNDED)
                if not labeled:
                    out.append((path, node.lineno, "C1",
                                f"模块级容器 `{name}` 被变更（{fn.attr}）但所属函数无上限判据/白标"
                                f"（有界请加 `# alloc-audit: bounded` 并写明理由）"))
        # 下标赋值：_CACHE[key] = value；以及 AugAssign（`_CNT[k] += 1`）
        # ⛔ 第6轮绕过③：AugAssign 曾在检查之外
        for node in ast.walk(tree):
            _tgts = []
            if isinstance(node, ast.Assign):
                _tgts = list(node.targets)
            elif isinstance(node, ast.AugAssign):
                _tgts = [node.target]
            if not _tgts:
                continue
            for t in _tgts:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in mod_names:
                    owner = None
                    for anc in ast.walk(tree):
                        if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if anc.lineno <= node.lineno <= (anc.end_lineno or anc.lineno):
                                if owner is None or anc.lineno > owner.lineno:
                                    owner = anc
                    scope_src = _slice_of(owner) if owner else ""
                    _nm = t.value.id
                    _lab = bool(re.search(rf"alloc-audit:\s*bounded[^\n]*\b{re.escape(_nm)}\b", scope_src))
                    labeled = _lab or (_nm in KNOWN_BOUNDED)
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
            # ⛔ 第6轮 #7：上一版只数 assert 个数且 `<`/`>=` 也算 ⇒ 两条无关断言即可放行。
            #    要求：①含上界/跨距语义 ②**引用了块表/slot 本身或其 shape/stride**。
            t = ast.unparse(a.test)
            # ⛔ 第7轮阻断项⑤：旧版只要出现 `.shape[` + 变量名即计数 ⇒ 两条"下界/恒真"断言即可放行。
            #    要求必须有**值域上界**（max( 或 .min( 或 与 numel 比较），或至少形如
            #    `...max() <= ...` / `... <= ...shape[...] * ...` 的乘积上界。
            sem = ("max(" in t or "min(" in t) or (
                ".shape[" in t and ("<=" in t or "<" in t) and "*" in t)
            refs = any(re.search(rf"\b{x}\b", t) for x in ADDR_ARGS)
            return sem and refs

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
        # ⛔ 审计 N8：默认候选里可能有一个**过期镜像**（实测 1/24 文件比部署树旧 61 行）
        #    ⇒ 选"存在且最新"的那个，并在运行时打印 root + mtime，让"审的是哪个对象"可见。
        v = os.environ.get(env)
        if v and os.path.isdir(v):
            return v
        alive = [c for c in cands if os.path.isdir(c)]
        if not alive:
            return cands[0]
        def _newest(d: str) -> float:
            try:
                return max(os.path.getmtime(os.path.join(d, "vllm")) if os.path.isdir(os.path.join(d, "vllm")) else os.path.getmtime(d), 0)
            except OSError:
                return 0.0
        return max(alive, key=_newest)

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
    ap.add_argument("--no-whitelist-check", action="store_true",
                    help="跳过'白名单名字必须能在被扫文件里找到定义'的检查（供变异测试用）")
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

    # ⛔ 审计 N8：必须让人看见"这次审的是哪个 root、审的这批文件多旧"。
    #    ⚠️ 目录 mtime 不能反映文件新旧（实测 root 目录 mtime 停在 10-05，
    #    而内部文件一直在改）⇒ 改为取**被扫描文件**的最新 mtime。
    import datetime as _dt
    def _fmt(ts: float) -> str:
        try:
            return _dt.datetime.fromtimestamp(ts).strftime("%F %T")
        except Exception:  # noqa: BLE001
            return "N/A"
    for _name, _d in (("root", a.root), ("plugin", a.plugin)):
        _ft = []
        for _t in targets:
            if _t.startswith(_d):
                try:
                    _ft.append(os.path.getmtime(_t))
                except OSError:
                    pass
        print(f"  {_name}: {_d}  (该 root 下被扫文件 {len(_ft)} 个，最新 mtime "
              f"{_fmt(max(_ft)) if _ft else 'N/A'})")

    all_findings = []
    for p in targets:
        all_findings.extend(check_file(p))

    # ⛔ 第6轮 #9：白名单里出现过**不存在的名字**（`_BND_PIN`）、重复项、类别错。
    #    改为：每个白名单名字必须在被扫文件的源码里**出现过**，否则报错（退出码 3）。
    import re as _re
    _all_src = []
    for p in targets:
        try:
            _all_src.append(open(p, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    _blob = "\n".join(_all_src)
    # ⚠️ 逐名判断（用集合会很脆弱：`X = {}` 与 `X: T = {}` 混在一起时容易漏）
    def _has_def(n: str) -> bool:
        return _re.search(rf"^\s*{_re.escape(n)}\s*(?::|=)", _blob, _re.M) is not None
    _missing = [] if a.no_whitelist_check else [
        n for n in list(KNOWN_BOUNDED) + list(KNOWN_BOUNDED_CONSTEXPR) if not _has_def(n)]
    if _missing:
        print(f"❌ 白名单里有 {len(_missing)} 个名字在被扫文件中找不到定义：")
        for n in _missing:
            print(f"   {n}")
        return 3

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
