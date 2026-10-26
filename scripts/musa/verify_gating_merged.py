"""Prove the merged single-code ``GPU_DEVICE`` gate on the target host.

Run with the *deployed* tree on ``PYTHONPATH`` (not a cwd-relative import) so the
check exercises exactly what the engine will import::

    GPU_DEVICE=MUSA PYTHONPATH=/root/musa_postmerge_20261027 \
        /root/venvs/vllm310/bin/python scripts/musa/verify_gating_merged.py

Matrix required by ``MERGE_MUSA_TO_MAINLINE_PLAN.md`` §9.1 / §9.6:

* unset / ``CUDA``            -> CUDA path (must not depend on the hardware)
* ``MUSA`` / ``MTGPU`` / ``S4000`` (any case) -> MUSA path
* unknown non-empty value     -> platform auto-detection (never a silent wrong path)

The script also counts branch decisions through the *runtime* helper the engine
actually calls (``mixed_experts._musa_active``), and prints the defaults that
``musa_correctness.apply()`` installs for the MUSA correctness profile.
"""
from __future__ import annotations

import json
import os
import sys

CLEAR_KEYS = ("GPU_DEVICE", "XTU_MUSA_CORRECTNESS")

CASES = [
    ("unset", None),
    ("empty", ""),
    ("CUDA", "CUDA"),
    ("cuda", "cuda"),
    ("MUSA", "MUSA"),
    ("musa", "musa"),
    ("MTGPU", "MTGPU"),
    ("S4000", "S4000"),
    ("s4000", "s4000"),
    ("unknown:FOO", "FOO"),
]


def _set(value):
    for key in CLEAR_KEYS:
        os.environ.pop(key, None)
    if value is not None:
        os.environ["GPU_DEVICE"] = value


def main() -> int:
    from vllm.platforms import current_platform

    from vllm_xiaotu_moe import mixed_experts as me
    from vllm_xiaotu_moe import musa_correctness as mc

    platform_is_musa = bool(getattr(current_platform, "is_musa", lambda: False)())
    print(f"[gating] platform={type(current_platform).__name__} "
          f"is_musa={platform_is_musa}")
    print(f"[gating] module={mc.__file__}")
    print(f"[gating] runtime_helper={me.__file__}")

    rows = []
    failures = []
    for name, value in CASES:
        _set(value)
        active = mc.musa_path_active()
        # Branch counter through the helper the engine calls at every MoE site.
        hits = sum(1 for _ in range(100) if me._musa_active())
        expected = (name in ("MUSA", "musa", "MTGPU", "S4000", "s4000")
                    or (name.startswith("unknown") and platform_is_musa))
        ok = (active is expected) and (hits == (100 if expected else 0))
        rows.append({
            "case": name,
            "GPU_DEVICE": value,
            "requested_device": mc.requested_device(),
            "enabled": mc.enabled(),
            "musa_path_active": active,
            "helper_hits_per_100": hits,
            "expected_musa": expected,
            "ok": ok,
        })
        if not ok:
            failures.append(name)
        print(f"[gating] {name:14s} requested={mc.requested_device():5s} "
              f"enabled={mc.enabled()!s:5s} musa_path_active={active!s:5s} "
              f"helper={hits:3d}/100 expected_musa={expected!s:5s} "
              f"{'OK' if ok else 'FAIL'}")

    # Defaults that the MUSA correctness profile installs (must stay unchanged).
    for key in ("VLLM_EXPERTS_LOAD_DEVICE", "XIAOTU_MOE_GPU_RESIDENT_LAYERS",
                "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS", "XIAOTU_PLE_CPU",
                "XIAOTU_SPEC_DECODE"):
        os.environ.pop(key, None)
    os.environ["GPU_DEVICE"] = "MUSA"
    applied = mc.apply()
    defaults = {k: os.environ.get(k) for k in
                ("XTU_MUSA_CORRECTNESS", "VLLM_EXPERTS_LOAD_DEVICE",
                 "XIAOTU_MOE_GPU_RESIDENT_LAYERS",
                 "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS", "XIAOTU_PLE_CPU",
                 "XIAOTU_SPEC_DECODE")}
    print(f"[apply] applied={applied}")
    print(f"[apply] defaults={json.dumps(defaults, ensure_ascii=False)}")
    expect_defaults = {
        "XTU_MUSA_CORRECTNESS": "1",
        "VLLM_EXPERTS_LOAD_DEVICE": "cpu",
        "XIAOTU_MOE_GPU_RESIDENT_LAYERS": "",
        "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS": "0",
        "XIAOTU_PLE_CPU": "0",
        "XIAOTU_SPEC_DECODE": "0",
    }
    for key, want in expect_defaults.items():
        if defaults.get(key) != want:
            failures.append(f"apply:{key}={defaults.get(key)!r}!={want!r}")

    print(f"[gating] RESULT={'PASS' if not failures else 'FAIL ' + ','.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
