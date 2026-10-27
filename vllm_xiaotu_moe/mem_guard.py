# -*- coding: utf-8 -*-
"""显存守卫：软上限触发**选择性清理**，硬上限**报详细日志后优雅退出**。

三条要求（用户 2026-10-27 定）
------------------------------
1. **清理不能是全清**，要清"本次用不到的"。⇒ 插件里每个缓存条目都记 `last_use_step`，
   只淘汰**空闲 ≥ `XIAOTU_MEM_IDLE_STEPS` 步**的条目；清前 `cuda.synchronize()`，
   避免侧流仍在写导致的 use-after-free（历史教训见 `_tile_ring` 的注释）。
2. **软上限**：`reserved` 超过 `XIAOTU_MEM_SOFT_GIB` ⇒ **立刻**执行第 1 条；
   若仍超 ⇒ 再退一步做 `empty_cache()`（全清空闲块，最后手段）。
3. **硬上限**：`reserved` 超过 `XIAOTU_MEM_HARD_GIB` ⇒ 打印遥测 + 落内存快照 + 全线程栈，
   `flush` 后 `os._exit(EXIT_HARD)`，由外部看门狗重启（不再等到楔死）。

**默认开启**（`XIAOTU_MEM_GUARD=0` 可关）；只在到阈值时动作，不改变任何计算路径。
默认阈值：soft=33 GiB / hard=38 GiB（总量 39.49 GiB，hard 留 1.5 GiB 缓冲）。
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time

import torch

EXIT_HARD = 13
_LAST_TICK = [0.0]


def enabled() -> bool:
    return os.environ.get("XIAOTU_MEM_GUARD", "1") == "1"


def _gib(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _reserved_gib(dev: torch.device) -> float:
    return torch.cuda.memory_reserved(dev) / 2**30


def _hard_exit(gp, dev: torch.device, reserved: float, hard: float) -> None:
    """硬上限：先把能拿到的现场全部落盘，再退出。"""
    # ⛔【2026-10-27 实测更正】硬限**默认不再退出**：实测（r88）单侧 `os._exit(13)`
    #   只是把对端留在 collective 上挂满 900 s 再 cancelled ⇒ 故障形态更差、诊断更少
    #   （对端日志只有 "cancelled"，没有真正死因）。让 CUDA/vLLM 自己报 OOM 更有信息量。
    #   需要旧行为时显式 `XIAOTU_MEM_HARD_EXIT=1`。
    if os.environ.get("XIAOTU_MEM_HARD_EXIT", "0") != "1":
        print(f"[xtu-guard] ⚠️ 硬上限**仅告警不退出**: free {free:.3f} < hard {hard:.3f} GiB "
              f"(reserved={reserved:.3f}) ⇒ 继续运行，让真实 OOM 自己报错"
              f"（`XIAOTU_MEM_HARD_EXIT=1` 可恢复旧行为）", flush=True)
        try:
            gp.vram_report("guard-hard-warn")
        except Exception:                             # noqa: BLE001
            pass
        return
    print(f"[xtu-guard] ⛔ 硬上限触发: free 低于 {hard:.3f} GiB (reserved={reserved:.3f} GiB)",
          flush=True)
    try:
        gp.vram_report("guard-hard")                  # 遥测（含 allocated/reserved/frag）
    except Exception as e:                            # noqa: BLE001
        print(f"[xtu-guard] vram_report 失败: {e}", flush=True)
    try:
        from . import mem_trace
        mem_trace.dump("guard-hard")                  # 逐次分配快照（按栈可定位）
    except Exception as e:                            # noqa: BLE001
        print(f"[xtu-guard] 快照失败: {e}", flush=True)
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:                                 # noqa: BLE001
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_HARD)


def tick(gp, tag: str = "", min_interval_s: float = 5.0) -> None:
    """检查一次；由 60 s 心跳和关键调用点共同驱动（限频避免开销）。"""
    if not enabled():
        return
    now = time.monotonic()
    if now - _LAST_TICK[0] < min_interval_s:
        return
    _LAST_TICK[0] = now

    # ⭐【2026-10-27】判据用 **free**（`mem_get_info`），不用 reserved 拍数字：
    #   free = total − reserved − 非torch开销(实测 ≈1.70 GiB)
    #   · 软限 free<6.0 GiB：实测清理后 free 回到 5.8–13.6 ⇒ 清了就有救
    #   · 硬限 free<1.5 GiB：历史死亡都在 free 547–1297 MiB，且单次最大分配 0.51 GiB
    #   ⇒ 与显存总量无关，换卡不用重算
    # ⭐【2026-10-27 · 用户要求】阈值**按比例**给，不写死 GiB：
    #   软限 = 可用总量的 15%，硬限 = 4%（在 39.49 GiB 卡上正好 ≈5.9 / 1.6 GiB，
    #   与实测"清理后 free 回到 5.8–13.6""死亡发生在 0.55–1.3 GiB"吻合）。
    #   显式给 `XIAOTU_MEM_SOFT_FREE_GIB`/`..._HARD_...` 时以绝对值为准（调试用）。
    dev = torch.device("cuda", torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(dev)
    free_b = int(free)
    total_gib = int(total) / 2**30
    free = free_b / 2**30
    # ⭐⭐【2026-10-27 · 用户指出】阈值**不能相对 total**：`--gpu-memory-utilization`
    #   的语义就是"让进程占到 total×util"，它自身会把 free 压到 (1−util)×total。
    #   ⇒ 按 total 的 15% 设软限，在 util=0.85 时**永远踩线**（每 tick 触发）。
    #   正确做法：相对 vLLM 计划**本身留的 slack** 来判断"碎片是否吃掉太多余量"。
    util = _gib("GPU_UTIL", 0.0) or _gib("VLLM_GPU_MEMORY_UTILIZATION", 0.0) or 0.85
    util = min(max(util, 0.5), 0.99)
    slack = max(0.5, (1.0 - util) * total_gib)
    soft = (_gib("XIAOTU_MEM_SOFT_FREE_GIB", 0.0)
            or total_gib * _gib("XIAOTU_MEM_SOFT_FREE_FRAC", 0.0)
            or max(1.5, 0.40 * slack))
    hard = (_gib("XIAOTU_MEM_HARD_FREE_GIB", 0.0)
            or total_gib * _gib("XIAOTU_MEM_HARD_FREE_FRAC", 0.0)
            or max(0.8, 0.15 * slack))
    if soft <= 0 and hard <= 0:
        return
    reserved = _reserved_gib(dev)

    if hard > 0 and free < hard:
        _hard_exit(gp, dev, reserved, hard)

    # ⭐【2026-10-27 · 实测教训】只看 free 会**空转**：大 prefill 期间 free 天然低到 2.1 GiB，
    # 但那时 reserved≈allocated（内存真在用），淘汰 0 MiB、empty_cache 也释放不出东西
    # ⇒ 只是刷日志。真正可回收的是 **reserved − allocated（碎片/缓存）**，
    # 所以软限触发还要求 frag 足够大。
    frag = reserved - free_b / 2**30 + 0.0
    frag = reserved - (torch.cuda.memory_allocated(dev) / 2**30)   # reserved − allocated
    min_frag = _gib("XIAOTU_MEM_MIN_FRAG_GIB", 2.0)
    if soft > 0 and free < soft and frag >= min_frag:
        idle_steps = int(_gib("XIAOTU_MEM_IDLE_STEPS", 32))
        # ⭐【2026-10-27 · 用户反馈①②】日志必须能自证：
        #   · ours_bytes = **我们自己的**缓存总量（evict_idle 只覆盖这些）⇒ 与 frag 对比即知"是否覆盖"
        #   · before/after 成对打印 ⇒ 证明读数发生在清理**之后**，且能看出到底释放没释放
        try:
            ours = gp.persistent_bytes()
        except Exception:                             # noqa: BLE001
            ours = -1
        freed = 0
        try:
            freed = gp.evict_idle(max_idle_steps=idle_steps)
        except Exception as e:                        # noqa: BLE001
            print(f"[xtu-guard] evict_idle 失败: {type(e).__name__} {e}", flush=True)
        free_after_evict, _t = torch.cuda.mem_get_info(dev)
        reserved_after_evict = _reserved_gib(dev)
        print(f"[xtu-guard] 软上限触发({tag}): free={free:.3f} < soft={soft:.3f} | "
              f"ours={ours / 2**30:.3f} GiB 可淘汰范围, 整卡 frag={frag:.3f} GiB | "
              f"淘汰空闲条目 {freed / 2**20:.1f} MiB",
              flush=True)
        if int(free_after_evict) / 2**30 < soft:
            import time as _t_mod
            _t0 = _t_mod.perf_counter()
            _err = None
            try:
                # ⚠️ 本 PyTorch 版本 `empty_cache()` **不接受参数**（传 device 会 TypeError，
                #    而异常被吞 ⇒ 清理静默失效，2026-10-27 实测踩到）。切设备用上下文。
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()
            except Exception as e:                    # noqa: BLE001
                _err = f"{type(e).__name__}: {str(e)[:120]}"
            _ms = (_t_mod.perf_counter() - _t0) * 1000.0   # ⭐ 异常也计时，不写 -1.0
            free_final, _t2 = torch.cuda.mem_get_info(dev)
            reserved_final = _reserved_gib(dev)
            print(f"[xtu-guard] empty_cache 兜底（耗 {_ms:.1f} ms{'' if _err is None else ', 异常=' + _err}）: "
                  f"free {int(free_after_evict) / 2**30:.3f} → {int(free_final) / 2**30:.3f} GiB, "
                  f"reserved {reserved_after_evict:.3f} → {reserved_final:.3f} GiB",
                  flush=True)


def on_oom(gp, exc: BaseException, where: str = "") -> None:
    """OOM 现场取证：打印遥测 + 最大空闲块 + 逐次分配快照，然后**原样返回**（由调用方决定抛/降级）。

    目的（用户 2026-10-27）：「OOM 的时候能明确说明错误所在」——
    这里不改变控制流，只保证错误现场被完整记下：
      · allocated/reserved/frag/free 四项（区分"真用满"与"碎片拿不到连续块"）
      · `memory_summary()` 里按尺寸统计的空闲块（直接看有没有够大的连续块）
      · 逐次分配快照（按调用栈定位是谁占着）
    """
    try:
        print(f"[xtu-guard] ⛔ OOM 现场({where or 'unknown'}): {type(exc).__name__}: {str(exc)[:300]}",
              flush=True)
        gp.vram_report("oom")
    except Exception:                                 # noqa: BLE001
        pass
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        print(torch.cuda.memory_summary(dev, abbreviated=True), flush=True)
    except Exception:                                 # noqa: BLE001
        pass
    try:
        from . import mem_trace
        mem_trace.dump("oom")
    except Exception:                                 # noqa: BLE001
        pass
