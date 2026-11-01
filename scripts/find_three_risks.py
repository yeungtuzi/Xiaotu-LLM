#!/usr/bin/env python3
"""只找会导致三类后果的缺陷（用户 2026-11-01 收窄）：

  ① 显存膨胀 GROWTH —— 随请求形状增长的编译变体 / 无界容器 / 按形状缓存
  ② 不当释放 RELEASE —— expandable_segments 下 empty_cache（页 unmap→Xid31）、缺同步
  ③ 可疑地址 ADDR   —— 内核实参 block_table/slot 无上界断言

其余（命名、死代码、浮点比较、`.item()` 性能…）**一律不报**。
"""
from __future__ import annotations

import ast
import os
import re
import sys


def _fx(*c):
    for x in c:
        try:
            if x and os.path.exists(x):
                return x
        except OSError:
            continue
    return c[0]


ROOT = _fx(os.environ.get("XTU_VLLM_ROOT", ""), "/root/lvllm/data/repos/vllm-mainline",
           "/home/user/lvllm/process_data/ref/repos/vllm-mainline")
PLUGIN = _fx(os.environ.get("XTU_PLUGIN_ROOT", ""), "/root/lvllm/repos/xiaotu-llm/vllm_xiaotu_moe",
             "/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe")
SCOPE = _fx(os.environ.get("XTU_SCOPE", ""),
            "/root/lvllm/repos/xiaotu-llm/scripts/kernel_audit_scope.txt",
            "/home/user/lvllm/kernel_audit_scope.txt")

ADDR_ARGS = ("block_table", "block_tables", "block_table_tensor", "slot_mapping",
             "slot_ids", "compressed_slot_ids", "kv_slot_mapping")
# 已知有界：page stride / 页几何 / 配置开关（清单已逐条核验）
BOUNDED_CX = {"RAW_STRIDE", "CACHE_STRIDE", "KV_BLOCK_STRIDE", "TAIL_BLOCK_ELEMS",
              "BLOCK_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K", "TOP_K", "PADDED_TOP_K",
              "COMPRESS_RATIO", "WINDOW_SIZE", "GROUP_SIZE_M", "quant_block"}


def files():
    out = []
    for line in open(SCOPE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = (os.path.join(PLUGIN, line[len("plugin:"):].split("/", 1)[-1])
             if line.startswith("plugin:") else os.path.join(ROOT, line))
        if os.path.exists(p):
            out.append(p)
    return out


def scan(p):
    src = open(p, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    base = os.path.basename(p)

    # ── ② 不当释放：empty_cache 必须在**同函数**里有 synchronize，且不能在 capture 期可达
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls = [c for c in ast.walk(n) if isinstance(c, ast.Call)]
            has_ec = [c for c in calls if "empty_cache" in ast.unparse(c.func)]
            if not has_ec:
                continue
            ftext = ast.unparse(n)
            has_sync = any("synchronize" in ast.unparse(c.func) for c in calls)
            # 门控要按**整个函数**判（原先只看 empty_cache 那一行，会漏）
            gated = ("capturing" in ftext or "EMPTY_CACHE" in ftext)
            # 平台门控（ROCm/MUSA 等）⇒ 我们的 CUDA 路径永不执行 ⇒ 不算缺陷
            if re.search(r"is_rocm|is_musa|current_platform", ftext):
                continue
            if not has_sync:
                out.append(("RELEASE", has_ec[0].lineno,
                            f"`{n.name}` 调 empty_cache 但**同函数无 synchronize**"))
            elif not gated:
                out.append(("RELEASE", has_ec[0].lineno,
                            f"`{n.name}` 的 empty_cache 未做 capture/env 门控（服务期可达）"))

    # ── ① 显存膨胀：模块级容器 + 动态键；按形状键的缓存
    for n in tree.body:
        tgt = None
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            tgt = n.targets[0]
        elif isinstance(n, ast.AnnAssign):
            tgt = n.target
        if isinstance(tgt, ast.Name) and isinstance(n.value, (ast.Dict, ast.List, ast.Set)):
            name = tgt.id
            # 只有**值可能是张量**的容器才会占显存；存 int/str 的统计字典不算
            tensorish = False
            for m in ast.walk(tree):
                if isinstance(m, (ast.Assign, ast.AugAssign)):
                    for t in (m.targets if isinstance(m, ast.Assign) else [m.target]):
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                                and t.value.id == name:
                            rhs = ast.unparse(m.value) if isinstance(m, ast.Assign) else ""
                            if re.search(r"torch\.|\bcuda\b|\.to\(|empty\(|zeros\(|ones\("
                                         r"|tensor\(|Tensor|buffer|\.data", rhs):
                                tensorish = True
            if not tensorish:
                continue
            # 是否有长度上限（`if len(_C) < N` / `<= N`）
            capped = False
            for m in ast.walk(tree):
                txt = ast.unparse(m) if isinstance(m, (ast.If, ast.While)) else ""
                if txt and f"len({name})" in txt and re.search(r"[<>]=?\s*\d+", txt):
                    capped = True
            for m in ast.walk(tree):
                if isinstance(m, (ast.Assign, ast.AugAssign)):
                    for t in (m.targets if isinstance(m, ast.Assign) else [m.target]):
                        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                                and t.value.id == name and not isinstance(t.slice, ast.Constant):
                            out.append(("GROWTH", m.lineno,
                                        f"模块级 `{name}` 动态键写入**张量**"
                                        f"（{'有' if capped else '**无**'}长度上限）"))

    # ── ③ 可疑地址：内核启动把 addr 实参传入，但函数内无上界断言
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        launches = []
        for c in ast.walk(n):
            if not isinstance(c, ast.Call) or not isinstance(c.func, ast.Subscript):
                continue
            txt = ast.unparse(c)
            hits = [a for a in ADDR_ARGS if re.search(rf"\b{a}\b", txt)]
            if hits:
                launches.append((c.lineno, hits, txt))
        if not launches:
            continue
        asserts = [a for a in ast.walk(n) if isinstance(a, ast.Assert)]
        atxt = " ".join(ast.unparse(a.test) for a in asserts)
        for ln, hits, txt in launches:
            missing = [h for h in hits if h not in atxt]
            if missing:
                out.append(("ADDR", ln, f"`{n.name}` 把 {missing} 作内核实参，但**无涉及它的边界断言**"))
    return [(base, c, ln, m) for c, ln, m in out]


def main() -> int:
    allf = files()
    rows = []
    for p in allf:
        rows.extend(scan(p))
    from collections import Counter
    cnt = Counter(r[1] for r in rows)
    print(f"扫描 {len(allf)} 个文件；发现 {len(rows)} 条 —— {dict(cnt)}\n")
    name = {"GROWTH": "① 显存膨胀", "RELEASE": "② 不当释放", "ADDR": "③ 可疑地址"}
    for cat in ("GROWTH", "RELEASE", "ADDR"):
        sub = [r for r in rows if r[1] == cat]
        if not sub:
            print(f"=== {name[cat]}：0 条 ===\n")
            continue
        print(f"=== {name[cat]}（{len(sub)}）===")
        for b, _c, ln, m in sub:
            print(f"  {b}:{ln}  {m}")
        print()
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
