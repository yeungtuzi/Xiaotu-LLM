#!/usr/bin/env python3
"""Static audit of a vLLM qwen4_exp checkout for MUSA portability.

Usage: python scripts/musa/check_qwen4exp_musa.py /path/to/vllm
No torch / no GPU required.
"""
from __future__ import annotations
import argparse
import pathlib
import re


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_root", type=pathlib.Path)
    args = ap.parse_args()
    root = args.vllm_root
    if (root / "vllm/models/qwen4_exp").is_dir():
        root = root / "vllm/models/qwen4_exp"
    files = sorted(root.rglob("*.py"))
    if not files:
        print("no qwen4_exp python files under", root)
        return 2
    print(f"qwen4_exp files: {len(files)}")
    checks = {
        "is_rocm_only_dispatch": re.compile(r"if current_platform\.is_rocm\(\):"),
        "is_musa_branch": re.compile(r"is_musa"),
        "tl_extra_cuda": re.compile(r"tl\.extra\.cuda"),
        "pdl": re.compile(r"is_arch_support_pdl|launch_pdl"),
    }
    hits = {k: [] for k in checks}
    for f in files:
        text = f.read_text(errors="replace")
        for k, rx in checks.items():
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits[k].append((f.relative_to(root), i, line.strip()))
    for k in ("is_rocm_only_dispatch", "is_musa_branch", "tl_extra_cuda", "pdl"):
        print(f"\n## {k}: {len(hits[k])}")
        for f, i, line in hits[k][:20]:
            print(f"{f}:{i}: {line}")
        if len(hits[k]) > 20:
            print(f"... {len(hits[k]) - 20} more")
    print("\nExpected after MUSA port:")
    print("- is_musa_branch > 0")
    print("- tl_extra_cuda / pdl either guarded or launch_pdl forced false on MUSA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
