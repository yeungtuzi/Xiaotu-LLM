#!/usr/bin/env python3
"""Send one hidden-state request to the persistent GPU0 DSpark server."""
import pickle
import socket
import sys

import torch

hidden = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dspark_hidden.pt"
d = torch.load(hidden, map_location="cpu")
req = {
    "main_x": d["main_x"],
    "positions": d["positions"][: d["main_x"].shape[0]],
    "anchor": int(d["input_ids"][-1].item()),
    "q_len": 5,
    "input_ids": d["input_ids"][-5:],
}
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/tmp/dspark_gpu0.sock")
f = s.makefile("rwb")
pickle.dump(req, f)
f.flush()
resp = pickle.load(f)
print(resp)
