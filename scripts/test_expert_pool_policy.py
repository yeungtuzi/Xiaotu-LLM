#!/usr/bin/env python
"""expert_pool 的【运行中自适应】验收:① 策略正确性(纯 CPU)② 三条纪律门禁(源码级机械检查)。

为什么要有这个文件:`expert_pool.swap_in/swap_out` 属于"运行中改服务状态"的代码,
它的风险不在算法而在**违反门禁**(释放 plane / 顺序反了 / 在 staging 突发里动手)。
⇒ 门禁必须**可机械检查**,不能靠"我看着像对的" ✗

用法(CPU 即可,CUDA_VISIBLE_DEVICES=""):
  python scripts/test_expert_pool_policy.py
"""
from __future__ import annotations

import ast
import os
import sys
import types

os.environ.pop("XIAOTU_PIN_WINDOW", None)      # 默认必须关

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vllm_xiaotu_moe import expert_pool as EP  # noqa: E402
from vllm_xiaotu_moe.expert_pool import LayerPool, plan_swap  # noqa: E402

# ⚠️ 插件 import 时可能从 env 桥(XIAOTU_ENV_FILE,默认 /tmp/xiaotu_env)回填环境变量
# ⇒ 必须在 import **之后**再清一次,否则"默认关"的行为验证会被外部 envfile 污染 ✗
os.environ.pop("XIAOTU_PIN_WINDOW", None)

FAIL = []


def ck(cond, msg):
    print(f"[{'OK ' if cond else 'BAD'}] {msg}")
    if not cond:
        FAIL.append(msg)


def main() -> int:
    # ---------------- ① 策略正确性(纯 CPU,无 torch 张量) ----------------
    ck(plan_swap([], {1: 5}, 0) == ([], []), "空池/零槽 ⇒ 不换")
    ck(plan_swap([], {1: 5}, 4) == ([], []), "无驻留者 ⇒ 不换")
    ck(plan_swap([0, 1], {}, 2) == ([], []), "窗口内无命中 ⇒ 不换(不抖)")
    ev, ld = plan_swap([0, 1, 2, 3], {4: 100, 5: 90, 6: 80, 7: 70}, 4,
                       hyst=0.0, maxswap=2)
    ck(len(ev) == 2 and len(ld) == 2, f"限流:maxswap=2 只换 2 个(得 {ev}/{ld})")
    ck(ev == [0, 1], f"冷者先出:驻留计数全 0 ⇒ 换出最冷的 0,1(得 {ev})")
    ck(sorted(ld) == [4, 5], f"换入最热的两名(得 {ld})")
    ev2, ld2 = plan_swap([0], {0: 100, 1: 110}, 1, hyst=0.25, maxswap=1)
    ck(ev2 == [] and ld2 == [], f"滞回:110 未超过 100×1.25 ⇒ 不动(得 {ev2}/{ld2})")
    ev3, ld3 = plan_swap([0], {0: 100, 1: 130}, 1, hyst=0.25, maxswap=1)
    ck(ev3 == [0] and ld3 == [1], f"滞回:130 > 125 ⇒ 换(得 {ev3}/{ld3})")
    ev4, ld4 = plan_swap([0, 1, 2, 3], {4: 1, 5: 1, 6: 1, 7: 1}, 4,
                         hyst=0.0, maxswap=0)
    ck(len(ev4) <= 1, f"maxswap<=0 ⇒ 默认 max(1,nslots//4)=1(得 {len(ev4)})")
    a = plan_swap([0, 1, 2, 3], {0: 9, 1: 1, 4: 50}, 4, hyst=0.0, maxswap=4)
    ck(4 not in a[0], "已驻留者不会出现在换出名单里")
    ck(0 not in a[1], "已驻留者不会出现在换入名单里")

    # ---------------- ② 三维门禁(源码级,机械) ----------------
    src = open(EP.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "LayerPool")
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for need in ("swap_in", "swap_out", "maybe_adapt", "note_routing"):
        ck(need in fns, f"LayerPool 提供 {need}()")

    def code_only(fn) -> str:
        """只取**可执行代码**:剔除 docstring 与 `#` 注释。

        ⛔ 来由:本测试第一次跑就**假阳性**——`swap_out` 的 docstring 里写着
        "不 del / 不 empty_cache" 这几个词,被整段源码匹配命中 ✗
        ⇒ 门禁必须只看**代码**,否则"写在注释里的承诺"会被当成"代码里的违规",
          反过来也意味着:注释里写什么都不能证明代码守规矩 ✓
        """
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        txt = "\n".join(ast.get_source_segment(src, st) or "" for st in body)
        return "\n".join(ln.split("#", 1)[0] for ln in txt.splitlines())

    so = code_only(fns["swap_out"])
    bad = [t for t in ("del ", ".resize", ".empty", "empty_cache") if t in so]
    ck(not bad, f"§4.1 红线:swap_out **不得**释放 plane(命中 {bad})")

    si = code_only(fns["swap_in"])
    i_wait, i_tbl = si.find("wait_event"), si.find("self.slotmap[new_expert] =")
    ck(0 < i_wait < i_tbl, f"§4.2 顺序:(wait_event@{i_wait}) 必须早于 (翻表@{i_tbl})")

    ma = ast.get_source_segment(src, fns["maybe_adapt"])
    ck("staging_active: bool = True" in ma,
       "§4.3 fail-closed:默认 staging_active=True(⇒ 默认什么都不做)")

    # ---------------- ③ 默认关闭 / fail-closed 的**行为**验证 ----------------
    # ⭐ 2026-10-17:窗口计数已由 Python int `_win` 改为 device 标量 `_win_t`
    #    (SPEC §9.1 图安全)⇒ 桩对象必须同步,否则行为验证会因为 **AttributeError** 假失败 ✗
    import torch

    def _stub(**kw):
        base = dict(nslots=0, _hits=None, _win_t=torch.zeros((), dtype=torch.int64))
        base.update(kw)
        return types.SimpleNamespace(**base)

    stub = _stub(nslots=0, _hits=None)
    ck(LayerPool.maybe_adapt(stub, None) == 0, "零槽 ⇒ 0")
    stub2 = _stub(nslots=4, _hits=None)
    ck(LayerPool.maybe_adapt(stub2, None) == 0, "还没统计过 ⇒ 0")
    stub3 = _stub(nslots=4, _hits=object(), _win_t=torch.full((), 999, dtype=torch.int64))
    ck(LayerPool.maybe_adapt(stub3, None) == 0,
       "⛔ 默认 staging_active(无法判定安全性)⇒ 0(fail-closed)")
    stub4 = _stub(nslots=4, _hits=object(), _win_t=torch.full((), 999, dtype=torch.int64))
    ck(LayerPool.maybe_adapt(stub4, None, staging_active=False) == 0,
       "窗口开关未设(XIAOTU_PIN_WINDOW 缺省)⇒ 0(默认关)")

    print()
    if FAIL:
        print(f"⛔ {len(FAIL)} 项未过:")
        for m in FAIL:
            print("   -", m)
        return 1
    print("✅ 全部通过(策略 9 项 + 门禁 6 项 + 行为 4 项)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
