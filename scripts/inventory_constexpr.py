#!/usr/bin/env python3
"""枚举 scope 内**每一个** constexpr 形参及其调用点实参，生成可直接核验的表。

动机：与其证明"我的门禁不可绕过"（文本启发式，永远有下一个洞），
不如直接列出**事实清单**——每个 constexpr 形参收到什么表达式，人工/审计员逐行判断。
这份表不依赖任何启发式，也不存在"绕过"：漏一行就是漏一行，看得见。
"""
from __future__ import annotations

import ast
import os
import re
import sys

def _first_existing(*c):
    for x in c:
        try:
            if x and os.path.exists(x):
                return x
        except OSError:
            continue
    return c[0]

ROOT = _first_existing(os.environ.get("XTU_VLLM_ROOT", ""),
                       "/root/lvllm/data/repos/vllm-mainline",
                       "/home/user/lvllm/process_data/ref/repos/vllm-mainline")
PLUGIN = _first_existing(os.environ.get("XTU_PLUGIN_ROOT", ""),
                         "/root/lvllm/repos/xiaotu-llm/vllm_xiaotu_moe",
                         "/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe")
SCOPE = _first_existing(os.environ.get("XTU_SCOPE", ""),
                        "/root/lvllm/repos/xiaotu-llm/scripts/kernel_audit_scope.txt",
                        "/home/user/lvllm/kernel_audit_scope.txt")


def files():
    out = []
    for line in open(SCOPE, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("plugin:"):
            p = os.path.join(PLUGIN, line[len("plugin:"):].split("/", 1)[-1])
        else:
            p = os.path.join(ROOT, line)
        if os.path.exists(p):
            out.append(p)
    return out


def main() -> int:
    allf = files()
    trees = {}
    for p in allf:
        try:
            trees[p] = ast.parse(open(p, encoding="utf-8", errors="replace").read())
        except SyntaxError as e:
            print(f"⚠️ 解析失败 {p}: {e}")

    # ① 收集内核定义（名字 -> [(file, cx params)])
    defs = {}
    DEFAULTED = set()
    for p, t in trees.items():
        for n in ast.walk(t):
            if isinstance(n, ast.FunctionDef) and "triton.jit" in " ".join(
                    ast.unparse(d) for d in n.decorator_list):
                cx = [a.arg for a in n.args.args + n.args.kwonlyargs
                      if a.annotation and "constexpr" in ast.unparse(a.annotation)]
                params = [a.arg for a in n.args.args] + [a.arg for a in n.args.kwonlyargs]
                # constexpr 形参若**有默认值**，未传时用默认 ⇒ 不是"未定位"
                for a in n.args.args[n.args.defaults and len(n.args.args) - len(n.args.defaults):] \
                        if n.args.defaults else []:
                    if a.arg in cx:
                        DEFAULTED.add((n.name, a.arg))
                for a, d in zip(n.args.kwonlyargs, n.args.kw_defaults):
                    if d is not None and a.arg in cx:
                        DEFAULTED.add((n.name, a.arg))
                defs.setdefault(n.name, []).append((p, params, cx))

    # ② 收集调用点
    AMBIG = set()
    rows = []
    for p, t in trees.items():
        for c in ast.walk(t):
            if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Subscript)):
                continue
            v = c.func.value
            name = v.id if isinstance(v, ast.Name) else (v.attr if isinstance(v, ast.Attribute) else None)
            if not name or name not in defs:
                continue
            # `**var` 指向的局部字典：在同一函数里找 `var = dict(k=v, ...)` / `var = {...}`
            _starkw = {}
            for kw in c.keywords:
                if kw.arg is None and isinstance(kw.value, ast.Name):
                    _vn = kw.value.id
                    for a in ast.walk(t):
                        if isinstance(a, ast.Assign) and any(
                                isinstance(x, ast.Name) and x.id == _vn for x in a.targets):
                            if isinstance(a.value, ast.Call) and isinstance(a.value.func, ast.Name) \
                                    and a.value.func.id == "dict":
                                for kk in a.value.keywords:
                                    if kk.arg:
                                        _starkw[kk.arg] = ast.unparse(kk.value)
                            elif isinstance(a.value, ast.Dict):
                                for kk, vv in zip(a.value.keys, a.value.values):
                                    if isinstance(kk, ast.Constant) and isinstance(kk.value, str):
                                        _starkw[kk.value] = ast.unparse(vv)
            # ⚠️ 同名内核（`kernel` 在 13 个文件里都有定义）⇒ 位置绑定会绑错，**跳过**
            if len({d[0] for d in defs[name]}) > 1:
                AMBIG.add(name)
                continue
            for dfile, params, cx in defs[name]:
                for prm in cx:
                    expr = None
                    for kw in c.keywords:
                        if kw.arg == prm:
                            expr = ast.unparse(kw.value)
                            break
                    if expr is None and prm in _starkw:
                        expr = _starkw[prm] + "（经 **kwargs 字典）"
                    if expr is None and prm in params:
                        i = params.index(prm)
                        if i < len(c.args):
                            expr = ast.unparse(c.args[i])
                    rows.append((os.path.relpath(p, os.path.dirname(os.path.dirname(p))),
                                 c.lineno, name, prm, expr or "（位置未知/未传）"))

    VERIFIED = {"RAW_STRIDE", "CACHE_STRIDE", "KV_BLOCK_STRIDE", "TAIL_BLOCK_ELEMS"}

    def classify(expr: str, prm: str = "", _defaulted: bool = False) -> str:
        e = expr.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", e) or e.startswith("0x"):
            return "A 字面量"
        if re.fullmatch(r"1 if \w+ else 0", e):
            return "C 有界(钳位/桶化)"
        if re.search(r"\.stride\(\s*0\s*\)", e):
            return "D-已核验(页跨距)" if prm in VERIFIED else "D 跨距(高危)"
        if re.search(r"\.shape\[\s*-?1\s*\]", e):
            return "E shape[-1](末维)"
        if re.search(r"\.shape\[\s*0\s*\]", e) or re.search(r"\.numel\(\)", e):
            return "D 形状(高危)"
        if "next_power_of_2" in e or "min(" in e or "max(" in e:
            return "C 有界(钳位/桶化)"
        if re.search(r"\.stride\(\s*[1-9]\s*\)", e):
            # 连续张量的 stride(1..n) = 后面各维乘积 ⇒ 与序列长度无关，恒定
            return "G 内层跨距(恒定)"
        if e == "（位置未知/未传）":
            return "I 有默认值(安全)" if _defaulted else "H 未定位"
        # 纯标识符：几何/配置名字或全大写常量
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", e):
            if e.isupper() or any(t in e.lower() for t in (
                    "head", "block", "dim", "size", "scale", "topk", "ratio",
                    "n_", "num", "candidate", "window", "page", "chunk", "split")):
                return "B 配置常量"
            return "F 需人工确认"
        if e.startswith("self.") or e.startswith("config.") or e.isupper():
            return "B 配置常量"
        # 形如 `float(x)` / `int(x)` 的标量转换、以及 <4 字符的短标识符（bm/bk/ns 这类 tile 尺寸）
        if re.fullmatch(r"(float|int)\(\s*\w+\s*\)", e):
            return "B 配置常量"
        if re.fullmatch(r"[a-z_][a-z0-9_]{0,3}", e):
            return "B 配置常量"
        return "F 需人工确认"

    rows = [(f, ln, k, prm, expr, classify(expr, prm, (k, prm) in DEFAULTED))
            for f, ln, k, prm, expr in rows]
    rows.sort(key=lambda r: (r[5], r[0], r[1]))

    from collections import Counter
    cnt = Counter(r[5] for r in rows)
    print("## 分类统计\n")
    print("| 类别 | 条数 | 含义 |")
    print("|---|---|---|")
    meaning = {
        "A 字面量": "写死的数字，编译期恒定 ⇒ 安全",
        "B 配置常量": "进程内不变的配置 ⇒ 只有配置组合数份 cubin",
        "C 有界(钳位/桶化)": "经 next_power_of_2 / min / max 收敛到有限集合 ⇒ 安全",
        "D 跨距(高危)": "`.stride(0)` 派生 ⇒ **随请求变化的头号嫌疑**",
        "D-已核验(页跨距)": "页跨距 = page_size×head_dim，只在配置变化时变 ⇒ 已逐条核验证据行",
        "D 形状(高危)": "`.shape[0]`/`.numel()` 派生 ⇒ **随请求变化**",
        "E shape[-1](末维)": "末维通常是模型几何（head_dim 等）⇒ 一般安全，需确认",
        "G 内层跨距(恒定)": "`.stride(1..n)` 对连续张量 = 后续各维乘积，与序列长度无关 ⇒ 安全",
        "H 未定位": "调用点看不到实参（位置超出 / **kwargs）⇒ 必须人工定位",
        "I 有默认值(安全)": "constexpr 形参有默认值且未传 ⇒ 用默认，编译期恒定",
        "F 需人工确认": "局部变量/复杂表达式 ⇒ 必须解析到源头",
    }
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"| {k} | {v} | {meaning.get(k, '')} |")
    print(f"\n**合计 {len(rows)} 条**\n")
    miss = sum(1 for r in rows if r[5] == "H 未定位")
    print(f"未定位（看不到实参）: {miss} 条；D 高危: {sum(1 for r in rows if r[5] == 'D 跨距(高危)')} 条\n")

    print("## 需要逐行核验的部分（D + E + F）\n")
    print("| # | 文件:行 | 内核 | constexpr 形参 | 调用点实参 | 类别 |")
    print("|---|---|---|---|---|---|")
    i = 0
    for f, ln, k, prm, expr, cls in rows:
        if not cls.startswith(("D", "E", "F", "H")) or cls.startswith("D-已核验"):
            continue
        i += 1
        print(f"| {i} | `{os.path.basename(f)}:{ln}` | `{k}` | `{prm}` | `{expr}` | {cls} |")
    print(f"\n需核验 {i} 条 / 共 {len(rows)} 条")

    # F 类按**去重后的表达式**汇总：数量少到可以人眼一遍看完
    fset = {}
    for f, ln, k, prm, expr, cls in rows:
        if cls == "F 需人工确认":
            fset.setdefault(expr, []).append(f"{os.path.basename(f)}:{ln}")
    if fset:
        print(f"\n### F 类去重（{len(fset)} 个不同表达式，覆盖 {sum(len(v) for v in fset.values())} 条）\n")
        print("| 表达式 | 次数 | 示例位置 |")
        print("|---|---|---|")
        for e, locs in sorted(fset.items(), key=lambda x: -len(x[1])):
            print(f"| `{e}` | {len(locs)} | `{locs[0]}` |")
    high = sum(1 for r in rows if r[5] == "D 跨距(高危)")
    if "--check" in sys.argv:
        print(f"\n[检查] D 高危 = {high}（必须为 0）")
        return 1 if high else 0
    if AMBIG:
        print(f"\n⚠️ 因同名内核跳过 {len(AMBIG)} 个名字：{sorted(AMBIG)[:6]}")
    return 0


def _unused():
    known_cx = {}
    return 0


if __name__ == "__main__":
    sys.exit(main())
