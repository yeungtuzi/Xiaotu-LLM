# SPDX-License-Identifier: Apache-2.0
"""Opt-in MUSA correctness mode.

Set ``XTU_MUSA_CORRECTNESS=1`` before starting vLLM.  The mode chooses the
simplest, easiest-to-debug path: CPU experts, no GPU-resident layers, no GPU
prefill, no speculative decoding, PLE on host.  It is deliberately conservative
and is not a performance mode.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("XTU_MUSA_CORRECTNESS", "0") == "1"


def apply() -> list[str]:
    """Set defaults; explicit user env always wins (setdefault)."""
    if not enabled():
        return []
    defaults = {
        # CPU experts for every layer.
        "XIAOTU_MOE_GPU_RESIDENT_LAYERS": "",
        # Do not run the Triton GPU prefill path.
        "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS": "0",
        # PLE n-gram table stays in host memory.
        "XIAOTU_PLE_CPU": "1",
        # No MTP/speculator in the first correctness pass.
        "XIAOTU_SPEC_DECODE": "0",
    }
    applied = []
    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v
            applied.append(f"{k}={v or '<empty>'}")
    if applied:
        print("[vllm-xtu-moe/musa] correctness mode: " + ", ".join(applied), flush=True)
    return applied
