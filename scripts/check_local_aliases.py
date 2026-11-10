"""静态检查：函数内使用了 `_os` / `_t` / `_time` / `_np` 等**局部别名**，但该作用域没绑定它。

为什么需要（2026-11-05 事故）：
    在 `_gmlet_impl` 里写了 `if _os.environ.get("XIAOTU_GPF_BOUNDS_EV") == "1":`，
    但该函数**没有** `import os as _os` ⇒ 不论开关是否开启，**这一行总会执行**
    ⇒ `NameError: name '_os' is not defined` ⇒ 引擎在第一次 prefill 时死掉。
    代价：一次 13 分钟的启动 + 一次基准，全部白费。

    ⚠️ 关键教训：**"默认关闭"的开关，它的判断表达式本身是无条件执行的** ——
    "默认关"不能保护任何东西。

判据：对每个函数，若用到下列名字但函数内未绑定、模块级也未绑定 ⇒ 报错，退出码 1。
误报处理：确实需要时，在函数里显式 `import os as _os`（这也是正确写法）。
"""
import ast
import sys

ALIASES = {"_os", "_t", "_time", "_np", "_gt", "_dt", "_wt", "_kt"}
TARGETS = sys.argv[1:] or ["vllm_xiaotu_moe/gpu_prefill_fp8.py"]


def module_bound(tree):
    """模块级已绑定的名字（import / 赋值 / 函数定义 / 类定义）。"""
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def fn_bound(fn):
    """函数内绑定的名字（参数 / import / 赋值 / except as / 推导式 / with as）。"""
    out = set()
    for a in list(fn.args.args) + list(fn.args.kwonlyargs) + list(fn.args.posonlyargs):
        out.add(a.arg)
    if fn.args.vararg:
        out.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        out.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.asname or a.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for n in ast.walk(node.optional_vars):
                if isinstance(n, ast.Name):
                    out.add(n.id)
        elif isinstance(node, ast.comprehension):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    out.add(n.id)
    return out


def check(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    mbound = module_bound(tree)
    problems = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = fn_bound(fn) | mbound
        # ⚠️ 只检查**本层**作用域：嵌套函数/闭包自成作用域，交给它自己那条 walk 检查，
        #   否则会把「子函数里自己 import 的 _t」误判成父函数未绑定（假阳性）。
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue                      # 不下降
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in ALIASES and node.id not in bound:
                    problems.append((node.lineno, fn.name, node.id))
            stack.extend(ast.iter_child_nodes(node))
    return sorted(set(problems))


bad = 0
for path in TARGETS:
    probs = check(path)
    if probs:
        bad = 1
        print(f"⛔ {path}: {len(probs)} 处使用了未绑定的局部别名")
        for line, fnname, name in probs:
            print(f"   {path}:{line}  在 `{fnname}()` 里用了 `{name}`，但该作用域未绑定")
        print("   ⇒ 修法：改用模块级 `os`/`torch`，或在函数内显式 "
              "`import os as _os` / `import torch as _t`")
    else:
        print(f"✅ {path}: 无非法的局部别名引用")

sys.exit(bad)
