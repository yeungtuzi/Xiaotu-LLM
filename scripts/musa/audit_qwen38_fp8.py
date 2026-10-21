#!/usr/bin/env python3
"""Offline audit of the official Qwen3.8-Flash-Next-FP8 checkpoint layout.

No torch / no GPU required. Reads config.json + model.safetensors.index.json.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path


def classify(name: str) -> str:
    n = name.lower()
    if "ple" in n:
        return "ple"
    if name.startswith("mtp."):
        return "mtp"
    if ".mlp.experts." in name:
        return "experts"
    if "vision" in n or "visual" in n:
        return "vision"
    if "embed" in n or "lm_head" in n:
        return "embed/lm_head"
    if "attn" in n or "linear_attn" in n:
        return "attention"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path, help="directory containing config.json and model.safetensors.index.json")
    args = ap.parse_args()
    cfg = json.loads((args.model_dir / "config.json").read_text())
    idx = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    tc = cfg.get("text_config", cfg)
    print("arch:", cfg.get("architectures"))
    for key in ("num_hidden_layers", "num_experts", "num_experts_per_tok",
                "moe_intermediate_size", "hidden_size", "max_position_embeddings",
                "mtp_num_hidden_layers", "ple_embedding_dtype"):
        print(f"{key}: {tc.get(key, cfg.get(key))}")
    q = cfg.get("quantization_config") or {}
    print("quant_method:", q.get("quant_method"), "quant_algo:", q.get("quant_algo"),
          "quantized_layers:", len(q.get("quantized_layers") or {}))
    total = idx.get("metadata", {}).get("total_size")
    print("index total_size GB:", round(total / 1e9, 3) if isinstance(total, int) else total)
    counts = collections.Counter(classify(n) for n in idx["weight_map"])
    print("tensor counts:", dict(counts))
    samples = collections.defaultdict(list)
    for n in idx["weight_map"]:
        k = classify(n)
        if len(samples[k]) < 6:
            samples[k].append(n)
    for k in ("ple", "mtp", "experts", "attention", "embed/lm_head"):
        print(f"--- {k} sample ---")
        for n in samples.get(k, []):
            print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
