#!/usr/bin/env python
"""item④(profile 期占位)验收 —— 纯 CPU。

核心命题只有一句:**"引擎还没建"时算出来的占位数,必须与"引擎已建"时逐字节相同。**
    ⛔ 旧代码 `if eng is None: return 0` ⇒ profile 期占位 = 0 ⇒ KV 池按"没有 staging"
       定容(handoff:池少 7.59 GiB ⇒ 3 路 256K 丢失)✗
    ✅ 新代码回落到**权重形状**,且与 `apply()` 用**同一个表达式 + 同一个真源函数** ✓

用法:CUDA_VISIBLE_DEVICES="" python scripts/test_profile_reserve.py
"""
from __future__ import annotations

import ast
import os
import re
import sys
import types

os.environ["XIAOTU_GPF_STAGE_TILE_E"] = "0"     # ⭐ 测试隔离门(见 test_gpu_prefill_fp8_assembly.py)
os.environ.pop("XIAOTU_PIN_WINDOW", None)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from vllm_xiaotu_moe import gpu_prefill as G  # noqa: E402
from vllm_xiaotu_moe import mixed_experts as MX  # noqa: E402

FAIL = []


def ck(cond, msg):
    print(f"[{'OK ' if cond else 'BAD'}] {msg}")
    if not cond:
        FAIL.append(msg)


def code_only(src, name):
    """函数体里**可执行代码**(剔 docstring + 剔 `#` 注释)—— 注释里的承诺不算数 ✓"""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == name)
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    txt = "\n".join(ast.get_source_segment(src, st) or "" for st in body)
    return "\n".join(ln.split("#", 1)[0] for ln in txt.splitlines())


def main() -> int:
    cls = MX.XiaotuCPUExpertsFp8          # ⭐ 用**真类** ⇒ _engine_attr/_group_k 是真默认值
    gp_fp8 = MX._gpu_prefill_mod("MOE_FP8")
    ck(gp_fp8 is not None, "MOE_FP8 模块可解析")
    ck(cls._engine_attr == "MOE_FP8", f"真类默认 _engine_attr={cls._engine_attr}")
    ck(int(cls._group_k) == 128, f"真类默认 _group_k={cls._group_k}")

    # ---- 桩:继承真类 ⇒ 继承真默认值,但不跑 vLLM 的 __init__(无需 GPU/权重) ----
    E, H, I = 288, 4096, 1024
    GK = int(cls._group_k)
    layer = types.SimpleNamespace(w2_weight=types.SimpleNamespace(shape=(E, H, I)))

    class _Stub(cls):
        def __init__(self, **kw):
            self._gpu_resident = kw.pop("_gpu_resident", False)
            self._layer_ref = kw.pop("_layer_ref", layer)
            self._xiaotu_engine = kw.pop("_xiaotu_engine", None)
            self._E_eng = kw.pop("_E_eng", None)
            self._H_eng = kw.pop("_H_eng", None)
            self._I_eng = kw.pop("_I_eng", None)
            for k, v in kw.items():
                setattr(self, k, v)

    stub = _Stub

    # ---- ① 引擎未建 ⇒ 必须**非 0**(旧代码这里是 0) ----
    need_noeng = int(cls.staging_need_bytes(_Stub()))
    ck(need_noeng > 0, f"⭐ 引擎未建时占位 > 0(得 {need_noeng / 2**30:.3f} GiB)")

    # ---- ② ⭐⭐ 两条几何来源必须**逐字节同数** ----
    need_eng = int(cls.staging_need_bytes(_Stub(
        _xiaotu_engine=object(), _E_eng=E, _H_eng=H, _I_eng=I)))
    ck(need_noeng == need_eng,
       f"⭐⭐ 权重路径 == 引擎路径({need_noeng} vs {need_eng})")

    # ---- ③ 必须等于【唯一真源】的直接调用 ----
    direct = int(MX.layer_staging_need_bytes(gp_fp8, None, E, H, I, GK))
    ck(need_noeng == direct,
       f"⭐ 等于 layer_staging_need_bytes 直接调用({need_noeng} vs {direct})")

    # ---- ④ pack 规则:FP8 的 w2 是 [E,H,I] ⇒ I 不得被 ×2 ----
    ck(need_noeng != int(MX.layer_staging_need_bytes(gp_fp8, None, E, H, 2 * I, GK)),
       "pack:FP8 未被当成 4-bit(I 没有 ×2)")

    # ---- ⑤ 权重还没到 ⇒ 不许猜(返回 0) ----
    nw = _Stub(_layer_ref=None)
    ck(int(cls.staging_need_bytes(nw)) == 0, "权重未到手 ⇒ 0(不猜)")
    ck(int(cls.staging_need_bytes(_Stub(_gpu_resident=True))) == 0, "常驻层 ⇒ 0")

    # ---- ⑥ 可观测:占位信息(含 ns)被记下 ----
    s = _Stub()
    cls.staging_need_bytes(s)
    info = getattr(s, "_prof_info", None)
    ck(isinstance(info, dict) and int(info.get("ns", 0)) > 0 and info.get("E") == E,
       f"⭐ _prof_info 记录了几何与 ns(得 {info})")

    # ---- ⑦ ⭐ 验收条件 3:tile 开启时占位数必须**变小**(走 tile 分支) ----
    os.environ["XIAOTU_GPF_STAGE_TILE_E"] = "96"
    fp8 = MX._gpu_prefill_mod("MOE_FP8")
    tile_need = int(fp8.fp8_staging_bytes(E, H, I, 4))
    os.environ["XIAOTU_GPF_STAGE_TILE_E"] = "0"
    whole_need = int(fp8.fp8_staging_bytes(E, H, I, 4))
    ck(tile_need < whole_need,
       f"⭐ tile 分支生效:tile=96 的占位({tile_need / 2**30:.3f} GiB) < 整层"
       f"({whole_need / 2**30:.3f} GiB) —— 这正是 handoff 要求 ②→④ 的理由")

    # ---- ⑧ 源码门禁 ----
    mxs = open(MX.__file__, encoding="utf-8").read()
    snb = code_only(mxs, "staging_need_bytes")
    ck("_live_w2_shape" in snb, "staging_need_bytes 走权重形状回落")
    ck(not re.search(r"eng\s+is\s+None:\s*\n\s*return\s+0", snb),
       "⛔ 旧缺陷(eng is None ⇒ return 0)已移除")
    lsb = code_only(mxs, "layer_staging_need_bytes")
    ck("info" in lsb and "ns" in lsb, "唯一真源把实际用的 ns 暴露出来")

    gs = open(G.__file__, encoding="utf-8").read()
    rsm = code_only(gs, "_reserve_staging_from_model")
    ck("_GP_PROFILE_LAYERS" in rsm,
       "⭐ 占位候选源含**加载期**注册表(否则 profile 期扫到 0 个)")
    ck("reserve_staging_for_profile" in rsm, "仍走唯一的占位分配入口")
    # ⛔ SIGSEGV 约束:占位路径**不得**出现任何"真跑预填内核"的调用
    banned = [t for t in ("gpu_moe_layer", "stage_tiles", "kmajor_from_engine_shards",
                          "prealloc_") if t in rsm]
    ck(not banned, f"⛔ 占位路径不得真跑预填/分配内核(命中 {banned})")

    print()
    if FAIL:
        print(f"⛔ {len(FAIL)} 项未过:")
        for m in FAIL:
            print("   -", m)
        return 1
    print("✅ 全部通过(行为 9 项 + 源码门禁 6 项)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
