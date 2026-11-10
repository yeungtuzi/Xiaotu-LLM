#!/usr/bin/env python3
"""Compare target DSpark base logits against GPU0 DSpark logits."""
import sys
import torch

target = torch.load(sys.argv[1], map_location="cpu").float()
gpu0 = torch.load(sys.argv[2], map_location="cpu").float()
print("target", tuple(target.shape), "gpu0", tuple(gpu0.shape))
n = min(target.shape[0], gpu0.shape[0])
t = target[:n].reshape(n, -1)
g = gpu0[:n].reshape(n, -1)
print("max_abs", float((t - g).abs().max()))
print("mean_abs", float((t - g).abs().mean()))
print("cosine", float(torch.nn.functional.cosine_similarity(t, g, dim=-1).mean()))
print("target_argmax", t.argmax(-1).tolist())
print("gpu0_argmax", g.argmax(-1).tolist())
