#!/usr/bin/env python3
"""把派生 ckpt 里**仍然是 FP8** 的 attention 张量真正反量化成 BF16，并重写索引。

背景（A78-15）：`/root/models/Qwen3.8-Flash-Next-NVFP4-attnBF16` 的
`model-bf16-*.safetensors` 其实是**指向原始 121 GB 模型的软链**，里面仍有 156 个
`F8_E4M3` 的 attention 张量。上一轮只把这 156 个键反量化写进了 `attn_bf16.safetensors`，
**没有把旧的 FP8 张量从可读路径上移除** ⇒ vLLM 的加载器按文件读，会先读到 BF16 版本、
再读到 FP8 版本，最后一笔把 FP8 **直接 cast 成 BF16（不乘 `weight_scale_inv`）**
⇒ attention 数值放大 ~1/scale（数百倍）；被每层 RMSNorm 归一化后"量级正常、语义全毁"。

本脚本做的事（幂等、可重跑）：
  1. 读 4 个 `model-bf16-*.safetensors`（软链到原始目录）
  2. FP8 张量 × 其 `*_scale_inv`（block 128）⇒ BF16；`*_scale_inv` 不再写出
  3. 其余张量原样搬运
  4. 写到本目录的 `model-attnbf16-<tag>.safetensors`（**实体文件**）
  5. 重写 `model.safetensors.index.json`：这些键指向新文件；删掉已不存在的 `_scale_inv` 键
  6. 删除那 4 个软链（改名成 `.disabled`），确保 FP8 版本**不可达**

用法：
    /root/venvs/vllm310/bin/python scripts/musa/dequant_derived_attn_to_bf16.py --dry-run
    /root/venvs/vllm310/bin/python scripts/musa/dequant_derived_attn_to_bf16.py --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

DEFAULT_DIR = "/root/models/Qwen3.8-Flash-Next-NVFP4-attnBF16"
BLOCK = 128


def _dequant(w, sc, block: int):
    import torch

    s = sc.float()
    if s.dim() == 4:
        s = s.squeeze(1).squeeze(-1)
    up = s.repeat_interleave(block, 0).repeat_interleave(block, 1)
    up = up[: w.shape[0], : w.shape[1]]
    return (w.float() * up).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tag", default="00001")
    args = ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    D = args.dir
    idx_path = os.path.join(D, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]

    srcs = sorted(glob.glob(os.path.join(D, "model-bf16-*.safetensors")))
    if not srcs:
        print("[dequant] ⛔ 找不到 model-bf16-*.safetensors")
        return 2
    print(f"[dequant] 待处理分片: {[os.path.basename(s) for s in srcs]}")

    out_name = f"model-attnbf16-{args.tag}.safetensors"
    out_path = os.path.join(D, out_name)
    new_map: dict[str, str] = {}
    n_fp8 = 0
    total_bytes = 0
    tensors: dict[str, "torch.Tensor"] = {}

    for src in srcs:
        with safe_open(src, framework="pt") as f:
            keys = list(f.keys())
            for k in keys:
                dt = str(f.get_slice(k).get_dtype())
                if dt.startswith("F8"):
                    sc_key = k + "_scale_inv"
                    if sc_key not in keys:
                        print(f"[dequant] ⛔ {k} 是 FP8 但没有 {sc_key}，跳过该张量")
                        continue
                    w = f.get_tensor(k)
                    sc = f.get_tensor(sc_key)
                    t = _dequant(w, sc, BLOCK)
                    n_fp8 += 1
                else:
                    if k.endswith("_scale_inv"):
                        continue  # 尺度已折进 BF16，不再需要
                    t = f.get_tensor(k)
                tensors[k] = t
                total_bytes += t.numel() * t.element_size()
        print(f"[dequant] 读完 {os.path.basename(src)}：累计 {len(tensors)} 张量 "
              f"{total_bytes / 2**30:.2f} GiB", flush=True)

    for k in tensors:
        new_map[k] = out_name
    print(f"[dequant] 其中 FP8→BF16: {n_fp8} 个；输出 {out_name} "
          f"{total_bytes / 2**30:.2f} GiB")

    if not args.apply:
        print("[dequant] 体检模式（加 --apply 才写盘）")
        return 0

    if os.path.exists(out_path):
        print(f"[dequant] ⛔ 目标已存在，拒绝覆盖: {out_path}")
        return 2
    save_file(tensors, out_path, metadata={"format": "pt"})
    print(f"[dequant] ✅ 已写 {out_path}")

    # 重写索引：这些键 → 新文件；丢掉不存在的 _scale_inv 键
    old_names = {os.path.basename(s) for s in srcs}
    new_wm = {}
    dropped = 0
    for k, v in wm.items():
        if k in new_map:
            new_wm[k] = out_name
        elif os.path.basename(v) in old_names:
            dropped += 1          # 该键原本在旧分片里、现在已不存在（如 *_scale_inv）
        else:
            new_wm[k] = v
    idx["weight_map"] = new_wm
    json.dump(idx, open(idx_path, "w"))
    print(f"[dequant] 索引已重写：{len(new_wm)} 键（丢弃 {dropped} 条失效键）")

    for s in srcs:
        os.rename(s, s + ".disabled")
    print("[dequant] ✅ 旧 FP8 分片已改名为 *.disabled（不可达）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
