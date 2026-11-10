#!/usr/bin/env python3
"""_weights_probe.py —— 读 config.json + index.json，打印模型口径事实（不加载权重）。

输出（供 shell 解析，空格分隔）：
  F <num_hidden_layers> <num_nextn_predict_layers> <sliding_window> <index_topk> <max_position_embeddings>
  M <mtp层号逗号表> <最大layers层号> <index声明分片数>
不依赖 transformers；纯 json。
"""
import json
import os
import re

d = os.environ.get("WPROBE_DIR", ".")
c = json.load(open(os.path.join(d, "config.json")))
tc = c.get("text_config") or c.get("llm_config") or {}
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
wm = idx["weight_map"]

mtps = sorted({int(m.group(1)) for k in wm for m in [re.search(r"(?:^|\.)mtp\.(\d+)\.", k)] if m})
lays = [int(m.group(1)) for k in wm for m in [re.search(r"layers\.(\d+)\.", k)] if m]
print("F", tc.get("num_hidden_layers"), tc.get("num_nextn_predict_layers"),
      tc.get("sliding_window"), tc.get("index_topk"), tc.get("max_position_embeddings"))
print("M", ",".join(map(str, mtps)), max(lays) if lays else -1, len(set(wm.values())))
