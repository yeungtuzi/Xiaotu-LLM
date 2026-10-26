#!/usr/bin/env python3
"""C 扩展「点名」检查：源码里声明的算子/函数，编译产物里必须真的能调到。

来由（2026-10-22，见 docs/musa/MUSA_BUILD_PITFALLS.md P1）：
  `csrc/musa_torch_bindings.cpp` 写了 `mcv.impl("get_cuda_view_from_cpu_tensor", ...)`，
  但那个编译单元从来没被编进装到 site-packages 的 `_C.so` ⇒ 运行时
  `NotImplementedError: Could not run '_C::...' with arguments from the 'CPU' backend`
  —— **schema 有、kernel 没有**。它伪装成 GPU 非法访存，浪费了一整天。

检查三类：
  A. Python 里 `torch.ops.<ns>.<name>` 的引用  → 运行时是否真的能解析到；
  B. C++ 源码里的注册（`*.impl("name"`、`TORCH_LIBRARY*`）→ 二进制里是否有该名字；
  C. 两边求差集：MISSING（引用了但没有）/ ORPHAN（源码注册了但二进制里没有）。

用法：
  python scripts/musa/verify_c_extensions.py --vllm-root /root/vllm-musa/third_party/vllm
  python scripts/musa/verify_c_extensions.py --vllm-root ... --json   # 机读

退出码：0 = 无 MISSING；3 = 有 MISSING（fail-closed）
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

OPS_REF_RE = re.compile(r"torch\.ops\.([A-Za-z_][\w]*)\.([A-Za-z_]\w*)")
IMPL_RE = re.compile(r"""\.impl\(\s*["']([A-Za-z_][\w.]*)["']""")
LIBRARY_RE = re.compile(r"TORCH_LIBRARY(?:_FRAGMENT)?\(\s*([A-Za-z_][\w]*)")
SO_NAMES_RE = re.compile(r"^[A-Za-z_][\w.]*$")


def scan_python_refs(vllm_root: Path) -> dict[tuple[str, str], list[str]]:
    """A: Python 代码里引用的 torch.ops.<ns>.<name>。"""
    refs: dict[tuple[str, str], list[str]] = {}
    for py in vllm_root.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in OPS_REF_RE.finditer(txt):
            key = (m.group(1), m.group(2))
            refs.setdefault(key, []).append(
                f"{py.relative_to(vllm_root)}:{txt[:m.start()].count(chr(10)) + 1}"
            )
    return refs


def scan_csrc_registrations(csrc: Path) -> dict[str, list[str]]:
    """B: C++ 源码里注册的算子名（不含命名空间前缀时归到 ""）。"""
    regs: dict[str, list[str]] = {}
    for src in list(csrc.rglob("*.cpp")) + list(csrc.rglob("*.cu")):
        try:
            txt = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ns = None
        for m in LIBRARY_RE.finditer(txt):
            ns = m.group(1)
        for m in IMPL_RE.finditer(txt):
            name = m.group(1)
            key = f"{ns or ''}::{name}" if ns else name
            regs.setdefault(key, []).append(
                f"{src.relative_to(csrc.parent)}:{txt[:m.start()].count(chr(10)) + 1}"
            )
    return regs


_SO_TEXT_CACHE: dict[str, str] = {}


def so_text(so: Path) -> str:
    """每个 .so 只跑一次 strings（36MB 的库逐个算子跑一次会慢到不可用）。"""
    key = str(so)
    if key not in _SO_TEXT_CACHE:
        try:
            _SO_TEXT_CACHE[key] = subprocess.run(
                ["strings", "-a", key],
                capture_output=True, text=True, timeout=300, check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            _SO_TEXT_CACHE[key] = ""
    return _SO_TEXT_CACHE[key]


def so_strings(so: Path, needle: str) -> int:
    text = so_text(so)
    return sum(1 for line in text.splitlines() if needle in line) if text else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default="/root/vllm-musa/third_party/vllm")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip-runtime", action="store_true",
                    help="只做源码侧检查（不 import vllm）")
    args = ap.parse_args()

    vllm_root = Path(args.vllm_root)
    refs = scan_python_refs(vllm_root / "vllm")
    regs = scan_csrc_registrations(vllm_root / "csrc")

    report: dict = {
        "python_refs": len(refs),
        "csrc_registrations": len(regs),
        "missing": [],     # 引用了、运行时调不到
        "orphan": [],      # 源码注册了、二进制里没找到
        "ok": [],
    }

    # ---- A: 运行时点名
    if not args.skip_runtime:
        import torch  # noqa: F401
        import vllm  # noqa: F401  （注册 _C 的 schema）

        for (ns, name), where in sorted(refs.items()):
            try:
                obj = getattr(getattr(torch.ops, ns), name)
                report["ok"].append(f"{ns}::{name}")
                del obj
            except AttributeError:
                report["missing"].append(
                    {"op": f"{ns}::{name}", "where": where[:3]}
                )
            except Exception as exc:  # noqa: BLE001
                report["missing"].append(
                    {"op": f"{ns}::{name}", "where": where[:3],
                     "note": f"{type(exc).__name__}: {exc}"}
                )

    # ---- B/C: 源码注册 vs 二进制字符串
    sos = sorted((vllm_root / "vllm").glob("*.so"))
    for key, where in sorted(regs.items()):
        name = key.split("::")[-1]
        hits = sum(so_strings(so, name) for so in sos)
        if hits == 0:
            report["orphan"].append({"op": key, "where": where[:3]})

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"python refs: {report['python_refs']}   "
              f"csrc registrations: {report['csrc_registrations']}   "
              f".so scanned: {len(sos)}")
        if report["missing"]:
            print(f"\n❌ MISSING（Python 引用了，运行时没有）: {len(report['missing'])}")
            for m in report["missing"]:
                print(f"   {m['op']}   <- {', '.join(m['where'])}"
                      + (f"  [{m['note']}]" if m.get("note") else ""))
        if report["orphan"]:
            print(f"\n⚠️  ORPHAN（源码注册了，二进制里没找到）: {len(report['orphan'])}")
            for o in report["orphan"]:
                print(f"   {o['op']}   <- {', '.join(o['where'])}")
        if not report["missing"]:
            print("\n✅ 点名通过：没有「引用了却调不到」的算子")

    return 3 if report["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
