# SPDX-License-Identifier: Apache-2.0
"""Opt-in MUSA correctness mode.

Set ``XTU_MUSA_CORRECTNESS=1`` before starting vLLM.  The mode chooses the
simplest, easiest-to-debug path: CPU experts, no GPU-resident layers, no GPU
prefill, no speculative decoding, PLE on host.  It is deliberately conservative
and is not a performance mode.
"""
from __future__ import annotations

import os


def requested_device() -> str:
    """Explicit compute-path selector.

    ``GPU_DEVICE=MUSA`` selects the MUSA line; unset/CUDA keeps the default
    CUDA line.  This is intentionally independent of vLLM's auto-detected
    platform so a user can pin the path.
    """
    return (os.environ.get("GPU_DEVICE") or "CUDA").strip().upper()


def musa_requested() -> bool:
    return requested_device() in ("MUSA", "MTGPU", "S4000")


def enabled() -> bool:
    return (
        os.environ.get("XTU_MUSA_CORRECTNESS", "0") == "1"
        or musa_requested()
    )


def apply() -> list[str]:
    """Set defaults; explicit user env always wins (setdefault)."""
    if not enabled():
        return []
    if musa_requested() and "XTU_MUSA_CORRECTNESS" not in os.environ:
        os.environ["XTU_MUSA_CORRECTNESS"] = "1"
    defaults = {
        # vLLM's native switch: routed experts live on the host from construction.
        # This is the prerequisite for the CPU expert backend.
        "VLLM_EXPERTS_LOAD_DEVICE": "cpu",
        # CPU experts for every layer.
        "XIAOTU_MOE_GPU_RESIDENT_LAYERS": "",
        # Do not run the Triton GPU prefill path.
        "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS": "0",
        # vLLM 0.32's own EngramConfig.cpu_offload handles the qwen4 PLE table;
        # do not layer our older UVA hook on top for the first MUSA pass.
        "XIAOTU_PLE_CPU": "0",
        # No MTP/speculator in the first correctness pass.
        "XIAOTU_SPEC_DECODE": "0",
    }
    # vLLM may have cached envs at import time; mirror the switch into vllm.envs
    # as well so mainline_shims sees it.  Explicit user settings still win.
    try:
        import vllm.envs as _vllm_envs

        # Honor an explicit user override; only mirror the CPU choice.
        if os.environ.get("VLLM_EXPERTS_LOAD_DEVICE") == "cpu":
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
