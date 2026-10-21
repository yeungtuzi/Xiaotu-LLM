#!/usr/bin/env python3
"""On-device qwen4_exp/MUSA smoke test.

Run on S4000 after installing vllm-musa v0.32 + the candidate patches.
No model weights needed.
"""
from __future__ import annotations
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-device", default="musa")
    args = ap.parse_args()
    import torch
    print("torch:", torch.__version__)
    try:
        import torch_musa  # noqa: F401
        print("torch_musa import: ok")
    except Exception as e:
        print("torch_musa import: FAIL", repr(e))
        return 2
    from vllm.platforms import current_platform
    print("platform:", type(current_platform).__name__, "device_name:", getattr(current_platform, "device_name", "?"))
    print("is_musa:", getattr(current_platform, "is_musa", lambda: False)())
    if args.expect_device == "musa" and not getattr(current_platform, "is_musa", lambda: False)():
        print("FAIL: current_platform is not MUSA")
        return 3
    from vllm.models.qwen4_exp import (  # noqa: F401
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpMTP,
    )
    print("qwen4_exp import: ok")
    print("classes:", Qwen4ExpForCausalLM.__name__, Qwen4ExpForConditionalGeneration.__name__, Qwen4ExpMTP.__name__)
    # Verify dispatch selected the AMD/Triton implementation on MUSA.
    mod = sys.modules.get(Qwen4ExpForCausalLM.__module__)
    print("model module:", getattr(mod, "__name__", "?"))
    if getattr(mod, "__name__", "").endswith(".nvidia.model"):
        print("FAIL: MUSA still dispatched to nvidia/model.py; apply 0001")
        return 4
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
