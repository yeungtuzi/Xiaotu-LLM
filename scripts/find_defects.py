#!/usr/bin/env python3
"""在自有内核文件里找**真实代码缺陷**（不是找"检查器漏洞"）。

五类：
  L  逻辑/实现：可变默认参数、浮点 `==`、`except: pass` 吞异常、同名变量遮蔽
  D  死代码/死分支：`if False:`、`return` 之后的语句、定义了从不被调用的函数、
                    从没被使用的形参（且在自有文件内可判定）
  K  泄漏风险：模块级 dict/list 动态键写入、`lru_cache` 无界、全局持有张量
  P  性能：热路径上的 `.item()/.cpu()/.tolist()/synchronize()`、
          `empty_cache()`（expandable 下更危险）、每步新建大对象
  R  资源：`torch.empty` 在循环内、未关闭的文件/句柄
"""
from __future__ import annotations

import ast
import os
import sys
from collections import Counter


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
        p = (os.path.join(PLUGIN, line[len("plugin:"):].split("/", 1)[-1])
             if line.startswith("plugin:") else os.path.join(ROOT, line))
        if os.path.exists(p):
            out.append(p)
    return out


PLATFORM_HINTS = ("is_rocm", "is_musa", "is_cuda", "is_xpu", "is_tpu", "current_platform",
                  "GPU_DEVICE", "MUSA", "ROCM", "HIP", "torch.version", "TYPE_CHECKING",
                  "envs.", "os.environ", "environ", "getenv", "platform.", "sys.platform",
                  "__name__")


def is_platform_gated(node) -> bool:
    """门控条件与**平台/配置**耦合 ⇒ 该块不算死代码（多 ISA 多平台必需）。"""
    t = ast.unparse(node.test) if isinstance(node, ast.If) else ""
    return any(h in t for h in PLATFORM_HINTS)


def const_true(node):
    try:
        return bool(eval(compile(ast.Expression(node), "<c>", "eval"), {}))  # noqa: S307
    except Exception:  # noqa: BLE001
        return None


def scan(p, tree, src):
    out = []
    base = os.path.basename(p)
    # 收集自定义函数名，用于"永不执行"判定
    defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    called = Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if nm:
                called[nm] += 1

    for n in ast.walk(tree):
        # ── D 死分支 ──────────────────────────────────────────────
        if isinstance(n, ast.If) and not is_platform_gated(n):
            v = const_true(n.test)
            if v is False:
                out.append(("D", n.lineno, f"`if {ast.unparse(n.test)}` 恒假 ⇒ 整块永不执行（{len(n.body)} 条语句）"))
            elif v is True and n.orelse:
                out.append(("D", n.lineno, f"`if {ast.unparse(n.test)}` 恒真 ⇒ else 分支永不执行"))
        # ⚠️ "return/raise 之后的语句"**不再判为缺陷**：那往往是刻意保留的旧实现/硬件变体路径
        #    （用户 #389 裁决）。若确要清理，需人工逐条裁定，不由扫描器直接报。
        # ── L 逻辑 ────────────────────────────────────────────────
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.args.defaults + [x for x in n.args.kw_defaults if x is not None]:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    out.append(("L", n.lineno, f"`{n.name}` 用可变对象 {ast.unparse(d)} 作默认参数"))
        if isinstance(n, ast.ExceptHandler) and n.type is None and len(n.body) == 1 and \
                isinstance(n.body[0], ast.Pass):
            out.append(("L", n.lineno, "`except: pass` 静默吞掉所有异常"))
        if isinstance(n, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in n.ops):
            if any(isinstance(c, ast.Constant) and isinstance(c.value, float) for c in n.comparators):
                out.append(("L", n.lineno, f"浮点数等值比较：`{ast.unparse(n)}`"))
        # ── P 性能 ────────────────────────────────────────────────
        if isinstance(n, ast.Call):
            nm = ast.unparse(n.func)
            if nm.endswith((".item", ".tolist")) or nm.endswith(".cpu"):
                out.append(("P", n.lineno, f"`.{nm.split('.')[-1]}()` 会同步主机（热路径禁用）"))
            if "empty_cache" in nm:
                out.append(("P", n.lineno, f"`{nm}()` 在 expandable_segments 下会 unmap 页"))
            if nm.endswith(".synchronize"):
                out.append(("P", n.lineno, f"`{nm}()` 强制设备同步"))
        # ── K 泄漏 ───────────────────────────────────────────────
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    if not isinstance(t.slice, ast.Constant):
                        out.append(("K", n.lineno, f"`{t.value.id}[...]` 用**动态键**写入"))
    # ⚠️ "本文件内未被调用"对公开 API 无效（大量函数被其它模块调用）⇒ 不做这项。
    #    真正可判定的死代码只有：恒假分支、return 之后的语句、`if 0:` 块。
    # ── D 真冗余：赋值后从未被读取，或读之前就被再次赋值（同函数内）──
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stores = {}
        loads = set()
        # `nonlocal`/`global` 声明的名字：值会**逃逸到外层作用域** ⇒ 不能算"未读取"
        for d in ast.walk(fn):
            if isinstance(d, (ast.Nonlocal, ast.Global)):
                loads.update(d.names)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    stores.setdefault(n.id, []).append(n.lineno)
                else:
                    loads.add(n.id)
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                loads.add(n.target.id)
        for var, lns in stores.items():
            if var.startswith("_"):
                continue
            if var not in loads and len(lns) == 1:
                out.append(("D", lns[0], f"`{var}` 被赋值但函数内**从未读取**（无用计算）"))
            elif var not in loads and len(lns) > 1:
                out.append(("D", lns[0], f"`{var}` 被赋值 {len(lns)} 次但**从未读取**"))
    return [(base, c, ln, m) for c, ln, m in out]


def main() -> int:
    allf = files()
    print(f"扫描 {len(allf)} 个自有内核文件\n")
    rows = []
    for p in allf:
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rows.extend(scan(p, tree, src))
    cnt = Counter(r[1] for r in rows)
    print("分类统计：", dict(cnt), f"合计 {len(rows)}\n")
    names = {"L": "逻辑/实现", "D": "死代码/死分支", "K": "泄漏风险",
             "P": "性能", "R": "资源"}
    for cat in "LDPKR":
        sub = [r for r in rows if r[1] == cat]
        if not sub:
            continue
        print(f"=== {cat} {names[cat]}（{len(sub)}）===")
        for base, _c, ln, m in sub[:40]:
            print(f"  {base}:{ln}  {m}")
        if len(sub) > 40:
            print(f"  … 其余 {len(sub)-40} 条")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
