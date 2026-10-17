#!/usr/bin/env python3
"""xtu-10:在 vLLM 的权重名解析层给 CT 的 FP8 block scale 加别名。

根因(EXPERIMENTS B492/B493):
  `compressed_tensors_w8a16_fp8.py:102` 在**加载期**把 scale 注册为 `weight_scale` ✓,
  而 `:128-130` 是**加载之后**才把它改名为 `weight_scale_inv` ✓。
  但第三方 qfn 检查点里存的是 **`weight_scale_inv`** ✗ ⇒
  `model_executor/models/utils.py` 的 `_load_module` 是**按模块树逐级找名字**的,
  在 `:425-440` 找不到就**直接 raise ValueError** ✗ ⇒
  **叶子层(LinearBase 等)的 load_weights 根本还没被调用** ⇒
  在 `linear.py` 里加重定向(我先前的 xtu-2)**永远不会生效** ✗。

修法:在名字解析层,当 `child_prefix == "weight_scale_inv"` 而该模块**只有 `weight_scale`** 时,
把名字临时改写成 `weight_scale` ✓ —— 一处覆盖所有层型(普通/packed/merged/QKV)✓。

用法:python3 patch_vllm_utils_scale_alias.py --tree <vllm tree> [--revert]
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

ANCHOR = "        for child_prefix, child_weights in self._groupby_prefix(weights):"
MARK = "[xtu-10]"
PATCH = '''            # [xtu-10] CT 的 FP8 block scale:加载期参数名是 `weight_scale`,
            # 而检查点给的是 `weight_scale_inv`(_inv 是 process_weights_after_loading
            # 里【加载后】才改的名)⇒ 本层按模块树找名字会直接 raise(见 B492/B493)⇒
            # 在此做别名,等价于检查点名字与模块参数名对齐 ✓
            if child_prefix == "weight_scale_inv" and "weight_scale" in child_params:
                child_prefix = "weight_scale"
                prefix = self._get_qualname(base_prefix, child_prefix)
'''

# ⭐⭐ xtu-11(B494/B495):检查点里**多出的**量化项(模块本身未量化 / 不需要它)⇒
#    必须**跳过并告警**,而不是 raise ✗ —— 否则就是"每撞一个名字崩一次"的打地鼠 ✗。
#    历史 devpatch #6「filter 36 unresolvable names」正是这一类 ✓(B320 ✓)。
#    插在【确已判定既非子模块、也非子参数】之后、`can_ignore_*` 判定之前 ✓,
#    因此【只影响本来到不了任何目标的名字】✓ ⇒ 对能正常加载的权重零影响 ✓。
ANCHOR2 = '                can_ignore_module = self._can_ignore_unexpected(prefix + ".")'
MARK2 = "[xtu-11]"
PATCH2 = '''                # [xtu-11] 检查点/映射带来、但本模块【确实没有】的参数名(典型:未量化模块的
                # `*_scale` / `weight_scale` / `input_scale` 残留)⇒ 跳过并告警 ✓
                # 判据:名字以量化 scale 形态结尾 ⇒ 不是权重本体 ⇒ 跳过安全(可逆:见 --revert)。
                _xtu_base = child_prefix.rsplit(".", 1)[-1]
                if _xtu_base.endswith((
                    "weight_scale", "weight_scale_inv", "input_scale",
                    "weight_scale_2", "input_scale_2", "weight_scale_inv_2",
                    "_scale_inv", "_scale", "scale",
                )):
                    import logging as _xtu_logging
                    _xtu_logging.getLogger(__name__).warning(
                        "[xtu-11] 跳过无法归位的量化项: %s(模块 %s 无此参数)",
                        prefix, module._get_name(),
                    )
                    continue
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    target = Path(a.tree) / "vllm/model_executor/models/utils.py"
    if not target.is_file():
        print(f"⛔ 找不到 {target}")
        return 2
    text = target.read_text()
    bak = target.with_suffix(".py.xtu10bak")

    if a.revert:
        if not bak.is_file():
            print("⛔ 没有备份可回退")
            return 2
        shutil.copy2(bak, target)
        print(f"✓ 已回退 {target}")
        return 0

    shutil.copy2(target, bak)
    new = text
    done: list[str] = []
    for mark, anchor, patch in ((MARK, ANCHOR, PATCH), (MARK2, ANCHOR2, PATCH2)):
        if mark in new:
            print(f"✓ {mark} 已在(幂等)")
            continue
        n = new.count(anchor)
        if n != 1:
            print(f"⛔ {mark} 锚点出现 {n} 次(要求恰好 1 次)⇒ 拒绝改")
            return 2
        new = new.replace(anchor, anchor + "\n" + patch, 1)
        done.append(mark)
    if not done:
        print("✓ 两处补丁都已在(幂等)")
        return 0
    try:
        ast.parse(new)                      # ⭐ 先证语法,再落盘(用户明令)
    except SyntaxError as e:
        print(f"⛔ 改写后语法错误,已放弃: {e}")
        return 2
    target.write_text(new)
    print(f"✓ 已打 {', '.join(done)}(备份 {bak.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
