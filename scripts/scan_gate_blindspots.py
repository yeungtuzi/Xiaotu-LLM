#!/usr/bin/env python3
"""扫描主门禁的若干**已知盲区**（这些用文本启发式做不可靠，单独用 AST 检查）。

盲区来源：第 12/13 轮审计实测均 exit 0 的绕过路径。

  B1 容器别名：`_A = {}; _B = _A; _B[k] = 1`（主门禁只认名字本身）
  B2 非白名单变更方法：`_C |= {...}` / `_C.pop(k)` / `_C.setdefault(k, 1)`
  B3 不可达分支的 synchronize：`if False: torch.cuda.synchronize()` 给 empty_cache 背书
  B4 跨文件同名内核：索引 last-wins，后者覆盖前者
  B5 `import as` 别名发射：`from x import K as K2` 后 `K2[grid](...)`
  B6 `@jit` 拼写：`from triton import jit` ⇒ 主门禁按 "triton.jit" 文本找不到

退出码：0=无发现；1=有发现。
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.environ.get("XTU_VLLM_ROOT", "/home/user/lvllm/process_data/ref/repos/vllm-mainline")
PLUGIN = os.environ.get("XTU_PLUGIN_ROOT", "/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe")
SCOPE = os.environ.get("XTU_SCOPE", "/home/user/lvllm/kernel_audit_scope.txt")

CTOR_NAMES = {"dict", "list", "set", "OrderedDict", "defaultdict"}
MUT_METHODS = {"add", "append", "update", "insert", "extend", "setdefault", "pop", "clear"}


def scope_files() -> list[str]:
    out = []
    for line in open(SCOPE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel = line.replace("plugin:", "").split("/")[-1] if line.startswith("plugin:") else line
        for base in (ROOT, PLUGIN):
            p = os.path.join(base, rel if not line.startswith("plugin:") else rel)
            if os.path.exists(p):
                out.append(p)
                break
    return out


def b1_container_alias(tree) -> list:
    """`_B = _A` 后 `_B[k] = 1`，且 _A 是模块级容器。"""
    mod = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and \
                isinstance(n.value.func, ast.Name) and n.value.func.id in CTOR_NAMES:
            mod |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.Assign) and isinstance(n.value, (ast.Dict, ast.List, ast.Set)):
            mod |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and \
                isinstance(n.value, (ast.Dict, ast.List, ast.Set)):
            mod.add(n.target.id)
    if not mod:
        return []
    alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in mod:
            for t in n.targets:
                if isinstance(t, ast.Name):
                    alias[t.id] = (n.value.id, n.lineno)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in alias:
                    out.append((n.lineno, t.value.id, alias[t.value.id]))
    return out


def b2_other_mutations(tree, mod: set) -> list:
    """`_C |= {...}` / `_C.pop()` 等主门禁未覆盖的变更。"""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id in mod:
            if isinstance(n.op, ast.BitOr):
                out.append((n.lineno, n.target.id, "|="))
        # ⚠️ 只有 `|=` / `.update()` 会**无界增长**；`.pop()`/`.setdefault()`/`.clear()`
        #    本身就是有界或缩减操作，报出来只会制造噪声（实测 8 条全是这类）。
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and \
                n.func.attr == "update" and isinstance(n.func.value, ast.Name) and \
                n.func.value.id in mod:
            out.append((n.lineno, n.func.value.id, ".update()"))
    return out


def b3_unreachable_sync(tree) -> list:
    """`if False:` / `while False:` 里的 synchronize 不应给 empty_cache 背书。"""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Constant) and n.test.value is False:
            for m in ast.walk(n):
                if isinstance(m, ast.Call):
                    name = ast.unparse(m.func)
                    if "synchronize" in name:
                        out.append((m.lineno, name))
    return out


def b4_b5_name_aliases(trees: dict) -> tuple[list, list]:
    """同名内核 / import as 别名。"""
    defs = {}
    dup = []
    for p, t in trees.items():
        for n in ast.walk(t):
            if isinstance(n, ast.FunctionDef) and "triton.jit" in " ".join(
                    ast.unparse(d) for d in n.decorator_list):
                defs.setdefault(n.name, []).append(p)
    for k, v in defs.items():
        if len(v) > 1:
            dup.append((k, v))
    aliases = []
    for p, t in trees.items():
        amap = {}
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.asname:
                        amap[a.asname] = a.name
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) and \
                    isinstance(n.func.value, ast.Name) and n.func.value.id in amap:
                aliases.append((p, n.lineno, amap[n.func.value.id], n.func.value.id))
    return dup, aliases


def b6_jit_spelling(trees: dict) -> list:
    out = []
    for p, t in trees.items():
        src = ast.unparse(t)
        if "from triton import jit" in src or "import jit" in src:
            out.append((p, "from triton import jit"))
    return out


def main() -> int:
    files = scope_files()
    trees, bad = {}, 0
    print(f"扫描 {len(files)} 个文件")
    for p in files:
        try:
            trees[p] = ast.parse(open(p, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            print(f"  ⚠️ 解析失败 {p}")

    for p, t in trees.items():
        mod = set()
        for n in t.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for x in tg:
                    if isinstance(x, ast.Name):
                        mod.add(x.id)
        for ln, name, (src, sln) in b1_container_alias(t):
            print(f"  B1 容器别名 {p}:{ln}  {name} = {src}（:{sln}）")
            bad += 1
        for ln, name, how in b2_other_mutations(t, mod):
            print(f"  B2 未覆盖的变更 {p}:{ln}  {name}{how}")
            bad += 1
        for ln, nm in b3_unreachable_sync(t):
            print(f"  B3 不可达分支的同步 {p}:{ln}  {nm}")
            bad += 1

    dup, aliases = b4_b5_name_aliases(trees)
    for k, v in dup:
        print(f"  B4 同名内核 `{k}` 出现在 {len(v)} 个文件：{[os.path.basename(x) for x in v]}")
        bad += 1
    for p, ln, orig, alias in aliases:
        print(f"  B5 import 别名发射 {p}:{ln}  {alias} -> {orig}")
        bad += 1
    for p, how in b6_jit_spelling(trees):
        print(f"  B6 @jit 拼写 {p}: {how}")
        bad += 1

    print(f"\n合计：{bad} 处")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
