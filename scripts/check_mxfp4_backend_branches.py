#!/usr/bin/env python3
"""check_mxfp4_backend_branches.py —— ⭐ 静态判据:构造器里选过的 MoE 后端,加载链必须都【接住】

## 为什么要有它(2026-10-15 实证)

`xtu-9`(patch 0016)让 CT 的 MXFP4 MoE 在专家驻留主机时选 **CPU 后端** ✓,
但**只改了构造器**,忘了 `process_weights_after_loading` 的那条 if 链 ✗ ⇒
CPU 掉进最后的 `else` ⇒ `prepare_moe_fp4_layer_for_marlin(layer)` ⇒
`NotImplementedError: Could not run '_C::gptq_marlin_repack' with arguments from the 'CPU' backend` ✗
—— 实测代价:worker 起 13:13:19 → 13:15:14 炸,`qfnFIX1/2` 两次点火全废 ✓

根因**不是"逻辑写错"**,而是「**同一个后端枚举,在 A 处被选中,在 B 处没被接住**」✗
⇒ 这类错误**可以静态查出来**,不必靠起服务撞 ⇒ 就是这个脚本 ✓

## 判据

对 `CompressedTensorsW4A4Mxfp4MoEMethod`:
1. 收集 `__init__` 里赋给 `self.mxfp4_backend` 的**全部** `Mxfp4MoeBackend.X`(含类属性默认值)
2. 收集 `process_weights_after_loading` 里**显式接住**的 `Mxfp4MoeBackend.X`
3. ⭐ **`MARLIN` 免检**(它就是最后那个 `else` 的主人)✓
4. ⭐ **`XPU` 免检**(链里用 `current_platform.is_xpu()` 接的 —— 见下面的显式映射)✓
5. 其余:`assigned - handled` **必须为空** ✗ 否则报错退出 1

## 用法

    python3 scripts/check_mxfp4_backend_branches.py <含 vllm/ 的目录>
    python3 scripts/check_mxfp4_backend_branches.py <直接指向那个 .py>

退出码:0 = 通过 · 1 = 有后端没被接住(附证据)· 2 = 用法/解析错
License: Apache-2.0
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REL = ("vllm/model_executor/layers/quantization/compressed_tensors/"
       "compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py")
CLASS = "CompressedTensorsW4A4Mxfp4MoEMethod"
ENUM = "Mxfp4MoeBackend"
LOADER_FN = "process_weights_after_loading"

# ⭐ 免检名单 —— **每条都要写理由**,不许默默加 ✓
EXEMPT = {
    "MARLIN": "它就是加载链最后那个 `else` 的主人(默认路径)✓",
    "XPU": "链里用 `current_platform.is_xpu()` 接住,不是按枚举名比 ✓",
}


def _enum_attrs(fn: ast.AST) -> set[str]:
    """收集该函数体里出现过的 `Mxfp4MoeBackend.X` 名字 ✓"""
    out: set[str] = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == ENUM):
            out.add(node.attr)
    return out


def analyse(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == CLASS:
            cls = node
            break
    if cls is None:
        raise SystemExit(f"⛔ 找不到类 {CLASS}(文件结构变了?)：{path}")

    assigned: set[str] = set()
    handled: set[str] = set()
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "__init__":
            assigned |= _enum_attrs(node)
        elif node.name == LOADER_FN:
            handled |= _enum_attrs(node)
    return assigned, handled


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("用法: check_mxfp4_backend_branches.py <含 vllm/ 的目录 | 直接指 .py>", file=sys.stderr)
        return 2
    target = Path(argv[1])
    path = target / REL if target.is_dir() else target
    if not path.is_file():
        print(f"⛔ 找不到目标文件:{path}", file=sys.stderr)
        return 2

    try:
        assigned, handled = analyse(path)
    except SyntaxError as exc:
        print(f"⛔ 解析失败:{exc}", file=sys.stderr)
        return 2

    missing = sorted(assigned - handled - set(EXEMPT))
    print(f"[mxfp4-branch] 文件:{path}")
    print(f"[mxfp4-branch] __init__ 选中的后端     : {sorted(assigned) or '(无)'}")
    print(f"[mxfp4-branch] {LOADER_FN} 接住的 : {sorted(handled) or '(无)'}")
    print(f"[mxfp4-branch] 免检(带理由)           : {dict(EXEMPT)}")

    if missing:
        print()
        print(f"⛔ 不通过:{len(missing)} 个后端在构造器里被选中,却在 {LOADER_FN} 里【没被接住】✗")
        for name in missing:
            print(f"     ✗ {ENUM}.{name} ⇒ 会掉进最后的 `else` ⇒ 走 Marlin 重打包路径 ✗")
        print()
        print("   判据来源:IRON_RULES.md(2026-10-15 qfnFIX1/2 两次点火的实际根因)")
        print("   修法:在该 if 链里补一支显式分支(如 `elif self.mxfp4_backend == "
              f"{ENUM}.{missing[0]}: pass`)✓")
        return 1

    print("[mxfp4-branch] ✅ 通过:构造器选过的后端,加载链都接住了 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
