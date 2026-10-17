#!/usr/bin/env python3
"""恢复 QFN(MXFP4-FP8 Qwen3.8-Flash-Next)所需的 vLLM 侧改动(devpatch #6:未融合名过滤)。

背景(证据见 docs/EXPERIMENTS.md B477):
    vllm/model_executor/layers/linear.py 的 `load_weights`(上游新写法)在权重名解析不到
    真实 Parameter 时,用 `getattr(..., self)` **静默兜底成模块**;
    随后 `weight_loader` 里 `param.data` 抛
        AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
    直接崩掉引擎。

    触发场景:检查点里存在【未融合】的名字(如 Qwen3.8-Flash-Next GDN 的
    `in_proj_qkv` / `in_proj_z`),而模型侧已被 `packed_modules_mapping` 融合为
    `in_proj_qkvz`;这些名字本来"无处安放",其权重也已折进融合参数。

本脚本的改动(**只跳过白名单内的未融合名**):
    `param is self` 时:
      * 末段名在 `_XTU_UNFUSED_NAMES` 内 -> 跳过并 `logger.warning_once` 计数;
      * 否则 -> **抛错**(拒绝盲目跳过,避免把 scale 之类的真实张量也跳掉,
        这一教训见 B478:曾把 `weight_scale_inv` 一并跳过,导致
        `ValueError: There is no module or parameter named '...weight_scale_inv'`)。

用法:
    python scripts/patch_vllm_qfn_recover.py --vllm-tree /path/to/vllm-mainline [--revert]
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

REL = "vllm/model_executor/layers/linear.py"

UNFUSED_BLOCK = '''_XTU_UNFUSED_NAMES = frozenset({
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",  # Qwen3.8-Flash-Next GDN 未融合名
    "q_proj", "k_proj", "v_proj",                          # 经典 QKV 未融合名
})


'''

ANCHOR = "class MergedColumnParallelLinear("

OLD = """            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)"""

NEW = """            if param is None and name == "bias":
                continue
            if param is self:
                _base = name.rsplit(".", 1)[-1]
                # [xtu-2] CT 的块量化 scheme(CompressedTensorsW8A16Fp8)在 create_weights
                # 时把 scale 注册为 "weight_scale",而检查点给的名字是 "weight_scale_inv"
                # (该改名要到 process_weights_after_loading 才发生,
                #  见 compressed_tensors_w8a16_fp8.py:121-135)。
                # ⇒ 加载期把它【重定向】到已存在的 weight_scale:
                #   不新增参数 ⇒ 不会内存×2;且其后 process_weights_after_loading
                #   仍会把它改名成 weight_scale_inv ⇒ 最终名字与权威约定一致。
                if _base == "weight_scale_inv" and hasattr(self, "weight_scale"):
                    param = self.weight_scale
                # [xtu-6] 检查点里的【未融合】名(其权重已由 packed_modules_mapping 折进
                # 融合参数)。上游原实现 getattr(..., self) 会静默兜底成【模块】,随后
                # param.data 抛 AttributeError 直接崩掉引擎(EXPERIMENTS B477)。
                elif _base not in _XTU_UNFUSED_NAMES:
                    raise ValueError(
                        "[xtu-6] unresolvable weight name=%r on %r; neither a "
                        "weight_scale_inv redirect nor a known unfused name, so refusing "
                        "to skip blindly (upstream would fall back to the module). "
                        "See EXPERIMENTS B477/B478/B481." % (name, getattr(self, "prefix", "?"))
                    )
                else:
                    _n = getattr(self, "_xtu_unresolved_names", 0) + 1
                    self._xtu_unresolved_names = _n
                    logger.warning_once(
                        "[xtu-6] skip unfused weight name=%s on %s (count=%d)",
                        name, getattr(self, "prefix", "?"), _n,
                    )
                    continue
            param.weight_loader(param, loaded_weight, shard_id)"""


def patch(tree: Path, revert: bool) -> int:
    f = tree / REL
    if not f.is_file():
        print(f"[xtu6] 找不到 {f}", file=sys.stderr)
        return 2
    src = f.read_text()

    if revert:
        if NEW not in src:
            print("[xtu6] 未发现本补丁痕迹,无需回退")
            return 0
        src = src.replace(NEW, OLD).replace(UNFUSED_BLOCK, "")
        ast.parse(src)
        f.write_text(src)
        print("[xtu6] 已回退 ✓")
        return 0

    if NEW in src:
        print("[xtu6] 已打过(幂等)✓")
        return 0

    n_anchor = src.count(ANCHOR)
    if n_anchor != 1:
        print(f"[xtu6] 锚点 {ANCHOR!r} 计数={n_anchor},期望 1", file=sys.stderr)
        return 3
    n_old = src.count(OLD)
    if n_old != 2:
        print(f"[xtu6] 目标块计数={n_old},期望 2(MergedColumnParallelLinear + QKVParallelLinear)",
              file=sys.stderr)
        return 4

    backup = f.with_suffix(".py.xtu6bak")
    if not backup.exists():
        shutil.copy2(f, backup)
        print(f"[xtu6] 备份 -> {backup}")

    out = src.replace(ANCHOR, UNFUSED_BLOCK + ANCHOR, 1).replace(OLD, NEW)
    ast.parse(out)  # 语法不过就不写盘
    f.write_text(out)
    print("[xtu6] linear.py 两处过滤已收窄为白名单 ✓")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-tree", required=True, type=Path)
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    return patch(a.vllm_tree, a.revert)


if __name__ == "__main__":
    raise SystemExit(main())
