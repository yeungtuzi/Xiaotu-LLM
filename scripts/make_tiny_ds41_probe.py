#!/usr/bin/env python3
"""生成一个「结构同形、尺寸极小」的 DeepSeek-V4.1 探针模型目录（不含权重）。

目的：在没有 510G 真权重的情况下，**在真机上跑通 decoder-side SWA bounded replay +
prefix cache 的代码路径**，用于快速差分验证（同 prompt：shared 前缀 vs unique 前缀，
两者输出必须逐 token 相同）。

做法：读一份真实 config.json，按给定比例缩小 hidden / experts / topk，
但**保留**所有影响 replay 的结构字段：`compress_ratios`、`candidate_source_layer_id`、
`kv_source_layer_ids`、`index_source_layer_ids`、`sliding_window`、`num_hidden_layers`。
再拷真实 tokenizer 文件（→ 与真模型同词表）。

用法（远端）：
    python scripts/make_tiny_ds41_probe.py \
        --src /home/user/lvllm/models/ds41f-topk1024 \
        --dst /home/user/lvllm/models/tiny41-probe \
        --hidden 512 --heads 8 --experts 16 --topk-experts 2 --index-topk 32
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def build(src: Path, dst: Path, hidden: int, heads: int, experts: int,
          topk_experts: int, index_topk: int, layers: int, max_pos: int,
          keep_quant: bool = False) -> None:
    cfg = json.loads((src / "config.json").read_text())
    text = cfg.get("text_config", cfg)

    text["hidden_size"] = hidden
    text["num_attention_heads"] = heads
    # DSv4.1 硬编码 MLA latent head_dim=512、rope 64（compressor.py:210 有 assert）
    text["head_dim"] = 512
    text["qk_rope_head_dim"] = 64
    text["moe_intermediate_size"] = 256
    text["n_routed_experts"] = experts
    text["n_shared_experts"] = 1
    text["num_experts_per_tok"] = topk_experts
    text["index_topk"] = index_topk
    text["index_n_heads"] = max(1, heads // 4)
    text["index_head_dim"] = 128  # indexer 硬性要求 128（k_pre.shape[1] == 128 assert）
    text["num_hidden_layers"] = layers
    text["max_position_embeddings"] = max_pos
    text["num_nextn_predict_layers"] = 0
    # 默认去量化（bf16 dummy 权重最省事）；--keep-quant 则保留生产 fp8/fp4 量化配置，
    # 让 MARLIN / PIN_CUDA_MOE 这些**生产 kernel 路径**也能被 dummy 权重走到。
    if not keep_quant:
        cfg.pop("quantization_config", None)
        text.pop("quantization_config", None)
        cfg["dtype"] = "bfloat16"
        text["dtype"] = "bfloat16"
    # vision 子模块关掉；dspark/engram 的键**必须保留**（模型代码会直接读，缺了会 AttributeError）
    cfg.pop("vision_config", None)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in TOKENIZER_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    print(f"wrote {dst}/config.json "
          f"(layers={layers} hidden={hidden} heads={heads} experts={experts} "
          f"topk={topk_experts} index_topk={index_topk} max_pos={max_pos})")
    print("replay-relevant fields kept:",
          {k: text.get(k) for k in ("compress_ratios", "candidate_source_layer_id",
                                    "kv_source_layer_ids", "index_source_layer_ids",
                                    "sliding_window")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk-experts", type=int, default=2)
    ap.add_argument("--index-topk", type=int, default=32)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--max-pos", type=int, default=32768)
    ap.add_argument("--keep-quant", action="store_true")
    a = ap.parse_args()
    build(a.src, a.dst, a.hidden, a.heads, a.experts, a.topk_experts,
          a.index_topk, a.layers, a.max_pos, a.keep_quant)


if __name__ == "__main__":
    main()
