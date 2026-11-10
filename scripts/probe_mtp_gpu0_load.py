#!/usr/bin/env python3
"""Load the ds41f MTP head standalone on GPU0, without the target model.

This is a probe, not a service entry point. It verifies that the MTP draft
model can be instantiated with its own weights/embed/head on a single GPU,
which is a prerequisite for the GPU0 draft-server work.
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.model_loader import get_model  # noqa: E402

MODEL = os.environ.get(
    "MODEL_DIR",
    "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master",
)
SPEC = {
    "method": "mtp",
    "model": MODEL,
    "num_speculative_tokens": 1,
    "draft_tensor_parallel_size": 1,
}

args = EngineArgs(
    model=MODEL,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=1,
    gpu_memory_utilization=0.15,
    enforce_eager=True,
    trust_remote_code=True,
    disable_log_stats=True,
    speculative_config=SPEC,
)
cfg = args.create_engine_config()
print("target_tp =", cfg.parallel_config.tensor_parallel_size, flush=True)
print("spec =", cfg.speculative_config, flush=True)
draft_cfg = cfg.speculative_config.draft_model_config
print("draft_arch =", draft_cfg.architectures, flush=True)
print("draft_model =", draft_cfg.model, flush=True)

model = get_model(vllm_config=cfg, model_config=draft_cfg)
model.eval()
n = sum(p.numel() for p in model.parameters())
print("loaded_params =", n, flush=True)
print("cuda_alloc_gib =", torch.cuda.memory_allocated() / 2**30, flush=True)
print("cuda_reserved_gib =", torch.cuda.memory_reserved() / 2**30, flush=True)
print("LOAD_OK", flush=True)
