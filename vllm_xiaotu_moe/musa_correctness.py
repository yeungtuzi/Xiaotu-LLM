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
        # vLLM's native switch: routed experts live on the host from construction.
        # This is the prerequisite for the CPU expert backend.
        "VLLM_EXPERTS_LOAD_DEVICE": "cpu",
        # CPU experts for every layer.
        "XIAOTU_MOE_GPU_RESIDENT_LAYERS": "",
        # Do not run the Triton GPU prefill path.
        "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS": "0",
        # PLE n-gram table stays in host memory.
        "XIAOTU_PLE_CPU": "1",
        # No MTP/speculator in the first correctness pass.
        "XIAOTU_SPEC_DECODE": "0",
    }
    # vLLM may have cached envs at import time; mirror the switch into vllm.envs
    # as well so mainline_shims sees it.  Explicit user settings still win.
    try:
        import vllm.envs as _vllm_envs

        if getattr(_vllm_envs, "VLLM_EXPERTS_LOAD_DEVICE", None) != "cpu":
            _vllm_envs.VLLM_EXPERTS_LOAD_DEVICE = "cpu"
    except Exception:  # noqa: BLE001 - vLLM may not be importable yet
        pass

    applied = []
    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v
            applied.append(f"{k}={v or '<empty>'}")
    if applied:
        print("[vllm-xtu-moe/musa] correctness mode: " + ", ".join(applied), flush=True)
    return applied
