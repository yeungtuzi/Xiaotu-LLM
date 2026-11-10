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
from vllm.distributed import (  # noqa: E402
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.model_loader import get_model  # noqa: E402

MODEL = os.environ.get(
    "MODEL_DIR",
    "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master",
)
SPEC = {
    "method": "dspark",
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

_cfg_ctx = set_current_vllm_config(cfg)
_cfg_ctx.__enter__()
init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method="tcp://127.0.0.1:29511",
    local_rank=0,
    backend="gloo",
)
initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)
model = get_model(vllm_config=cfg, model_config=draft_cfg)

# DSpark's load_weights() deliberately skips the target's embed/head. For a
# standalone draft these must be loaded explicitly from the target checkpoint.
if os.environ.get("LOAD_EMBED_HEAD", "1") == "1":
    import glob
    import json

    from safetensors import safe_open

    index_path = glob.glob(os.path.join(MODEL, "*.index.json"))[0]
    weight_map = json.load(open(index_path))["weight_map"]
    for ckpt_key, param in (
        ("embed.weight", model.model.embed_tokens.weight),
        ("head.weight", model.lm_head.weight),
    ):
        shard = weight_map[ckpt_key]
        with safe_open(os.path.join(MODEL, shard), framework="pt", device="cpu") as f:
            w = f.get_tensor(ckpt_key)
        if tuple(w.shape) != tuple(param.shape):
            raise RuntimeError(f"{ckpt_key}: ckpt {tuple(w.shape)} vs param {tuple(param.shape)}")
        param.data.copy_(w.to(device=param.device, dtype=param.dtype))
        print(f"loaded {ckpt_key} -> {tuple(param.shape)}", flush=True)

model.eval()
n = sum(p.numel() for p in model.parameters())
print("loaded_params =", n, flush=True)
print("cuda_alloc_gib =", torch.cuda.memory_allocated() / 2**30, flush=True)
print("cuda_reserved_gib =", torch.cuda.memory_reserved() / 2**30, flush=True)
print("LOAD_OK", flush=True)

if os.environ.get("LOGIT_PROBE", "1") == "1":
    dev = next(model.parameters()).device
    h = torch.randn(2, cfg.speculative_config.draft_model_config.hf_config.hidden_size,
                    dtype=torch.bfloat16, device=dev)
    logits = model.compute_logits(h)
    print("logits_shape =", tuple(logits.shape), flush=True)
    print("logits_finite =", bool(torch.isfinite(logits).all()), flush=True)
    print("logits_mean =", float(logits.float().mean()), flush=True)
    print("LOGIT_OK", flush=True)

if os.environ.get("KV_PROBE", "1") == "1":
    for i, layer in enumerate(model.model.layers):
        cache_layer = layer.attn.swa_cache_layer
        spec = cache_layer.get_kv_cache_spec(cfg)
        page = getattr(spec, "page_size_bytes", None)
        print(
            f"kv[{i}] prefix={cache_layer.prefix} block_size={cache_layer.block_size} "
            f"dtype={cache_layer.dtype} spec={type(spec).__name__} page={page}",
            flush=True,
        )
        num_blocks = 4
        print(f"kv[{i}] alloc begin", flush=True)
        # fp8_ds_mla SWA record width is state_content_bytes (584), not the
        # block-level page size.
        tokens_per_block = 584
        kv = torch.zeros(
            num_blocks, 1, cache_layer.block_size, tokens_per_block,
            dtype=torch.uint8, device=dev,
        )
        print(f"kv[{i}] alloc done", flush=True)
        cache_layer.bind_kv_cache(kv)
        print(f"kv[{i}] bind done", flush=True)


if os.environ.get("FORWARD_ONLY", "0") == "1":
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager
    dev = next(model.parameters()).device
    init_workspace_manager(dev)
    ids = torch.tensor([0], dtype=torch.long, device=dev)
    pos = torch.tensor([0], dtype=torch.long, device=dev)
    emb = model.embed_input_ids(ids)
    with set_forward_context(None, cfg):
        h = model(input_ids=ids, positions=pos, inputs_embeds=emb)
    print("forward_only_shape =", tuple(h.shape), flush=True)
    print("forward_only_finite =", bool(torch.isfinite(h).all()), flush=True)
    print("FORWARD_ONLY_OK", flush=True)


if os.environ.get("KV_FORWARD", "0") == "1":
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    dev = next(model.parameters()).device
    init_workspace_manager(dev)
    cache0 = model.model.layers[0].attn.swa_cache_layer
    n = 1
    block_table = torch.zeros((n, 1), dtype=torch.int32, device=dev)
    slot_mapping = torch.zeros((n,), dtype=torch.int64, device=dev)
    seq_lens = torch.ones((n,), dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    qsl_cpu = torch.tensor([0, 1], dtype=torch.int32)
    swa_width = int(cache0.window_size)
    swa_idx = torch.zeros((n, swa_width), dtype=torch.int32, device=dev)
    swa_lens = torch.ones((n,), dtype=torch.int32, device=dev)
    meta = DeepseekSparseSWAMetadata(
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=cache0.block_size,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl_cpu,
        decode_swa_indices=swa_idx,
        decode_swa_lens=swa_lens,
        decode_swa_width=swa_width,
        num_decodes=1,
        num_decode_tokens=1,
        max_decode_query_len=1,
    )
    attn = {layer.attn.swa_cache_layer.prefix: meta for layer in model.model.layers}
    ids = torch.tensor([0], dtype=torch.long, device=dev)
    pos = torch.tensor([0], dtype=torch.long, device=dev)
    emb = model.embed_input_ids(ids)
    with set_forward_context(attn, cfg):
        h = model(input_ids=ids, positions=pos, inputs_embeds=emb)
    print("kv_forward_shape =", tuple(h.shape), flush=True)
    print("kv_forward_finite =", bool(torch.isfinite(h).all()), flush=True)
    print("KV_FORWARD_OK", flush=True)


if os.environ.get("DRAFT_FORWARD", "0") == "1":
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    dev = next(model.parameters()).device
    init_workspace_manager(dev)
    cache0 = model.model.layers[0].attn.swa_cache_layer
    q_len = 5
    if os.environ.get("REAL_HIDDEN"):
        d = torch.load(os.environ["REAL_HIDDEN"], map_location="cpu")
        main_x = d["main_x"].to(device=dev, dtype=torch.bfloat16)
        ctx_len = int(main_x.shape[0])
        ctx_pos = d["positions"][:ctx_len].to(device=dev, dtype=torch.int64)
        print("real_hidden =", tuple(main_x.shape), flush=True)
    else:
        ctx_len = 5
        # Synthetic context hidden: infrastructure test, not a semantic draft.
        main_x = torch.randn(ctx_len, 5120, dtype=torch.bfloat16, device=dev)
        ctx_pos = torch.arange(ctx_len, dtype=torch.int64, device=dev)
    total = ctx_len + q_len
    ctx_slots = [torch.arange(ctx_len, dtype=torch.int64, device=dev) for _ in model.model.layers]
    model.precompute_and_store_context_kv(main_x, ctx_pos, ctx_slots)
    print("context_kv_done", flush=True)

    block_table = torch.arange(4, dtype=torch.int32, device=dev).unsqueeze(0)
    slot_mapping = torch.arange(ctx_len, total, dtype=torch.int64, device=dev)
    seq_lens = torch.full((1,), total, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, q_len], dtype=torch.int32, device=dev)
    qsl_cpu = torch.tensor([0, q_len], dtype=torch.int32)
    swa_idx = torch.arange(total, dtype=torch.int32, device=dev).unsqueeze(0).repeat(q_len, 1)
    swa_lens = torch.full((q_len,), total, dtype=torch.int32, device=dev)
    meta = DeepseekSparseSWAMetadata(
        block_table=block_table,
        slot_mapping=slot_mapping,
        block_size=cache0.block_size,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl_cpu,
        decode_swa_indices=swa_idx,
        decode_swa_lens=swa_lens,
        decode_swa_width=total,
        num_decodes=1,
        num_decode_tokens=q_len,
        max_decode_query_len=q_len,
    )
    attn = {layer.attn.swa_cache_layer.prefix: meta for layer in model.model.layers}
    if os.environ.get("REAL_HIDDEN"):
        anchor = int(d["input_ids"][-1].item())
    else:
        anchor = 0
    ids = torch.zeros((q_len,), dtype=torch.long, device=dev)
    ids[0] = anchor
    pos = torch.arange(ctx_len, total, dtype=torch.long, device=dev)
    emb = model.embed_input_ids(ids)
    with set_forward_context(attn, cfg):
        h = model(input_ids=ids, positions=pos, inputs_embeds=emb)
    logits = model.compute_logits(h)
    if os.environ.get("LOGITS_OUT"):
        torch.save(logits.detach().cpu(), os.environ["LOGITS_OUT"])
        print("saved_logits =", os.environ["LOGITS_OUT"], flush=True)
    print("draft_forward_shape =", tuple(h.shape), flush=True)
    print("draft_forward_finite =", bool(torch.isfinite(h).all()), flush=True)
    print("draft_logits_shape =", tuple(logits.shape), flush=True)
    print("draft_logits_finite =", bool(torch.isfinite(logits).all()), flush=True)
    print("DRAFT_FORWARD_OK", flush=True)

    if os.environ.get("BENCH", "0") == "1":
        for _ in range(5):
            with set_forward_context(attn, cfg):
                model(input_ids=ids, positions=pos, inputs_embeds=emb)
        torch.cuda.synchronize()
        times = []
        for _ in range(30):
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            with set_forward_context(attn, cfg):
                model(input_ids=ids, positions=pos, inputs_embeds=emb)
            e_ev.record()
            e_ev.synchronize()
            times.append(s_ev.elapsed_time(e_ev) * 1000.0)
        times.sort()
        print("draft_step_us_median =", round(times[len(times)//2], 1), flush=True)
        print("draft_step_us_min =", round(times[0], 1), flush=True)
        print("draft_step_us_max =", round(times[-1], 1), flush=True)

    if os.environ.get("SAMPLE", "1") == "1":
        # Greedy Markov-biased sample, infrastructure probe only.
        prev = torch.zeros((1,), dtype=torch.long, device=dev)
        sampled = []
        for i in range(logits.shape[0]):
            base = logits[i : i + 1]
            emb_i = model.markov_embed(prev)
            bias = model.markov_bias(emb_i)
            tok = (base + bias).argmax(dim=-1, keepdim=True)
            sampled.append(int(tok.item()))
            prev = tok.squeeze(1)
        print("sampled_tokens =", sampled, flush=True)
        print("SAMPLE_OK", flush=True)
