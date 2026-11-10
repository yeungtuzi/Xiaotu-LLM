"""S5000 长稳验证（10+ 分钟）：GPU 计算满载同时持续压 H2D。

⚠️ 上一版的 bug：worker 里用 `torch.musa.synchronize()`（**设备级**）节流，
   它会把另一条流一起阻塞 ⇒ compute 被“饿死”到 0.2%，测出假的“不能重叠”。
   本版改用**流级同步**（`Stream.synchronize()`，缺失时用 Event 兜底）。

结构（总计约 13 分钟）：
  solo copy 45s → solo compute 45s → concurrent 600s → post solo copy 45s → post solo compute 45s
  * 前后各测一次 solo，用于识别热降频（post/solo 比值）
  * concurrent 每 30s 打印一次中间速率
  * 全程采样 mthreads-gmi %GPU 作旁证

判据：
  compute 并发保留率 >= 60% ⇒ 本轮 H2D 数字可信
  overlap 效率 = copy_retained + comp_retained（2.0 完美 / 1.0 串行）
"""
import json
import subprocess
import threading
import time

import torch

DEV = torch.device("musa")
COPY_BYTES = 512 * 1024 * 1024
SOLO_SEC = 45
CONC_SEC = 600

h = torch.empty(COPY_BYTES, dtype=torch.uint8, pin_memory=True)
d = torch.empty(COPY_BYTES, dtype=torch.uint8, device=DEV)
A = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
B = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)

s_copy = torch.musa.Stream()
s_comp = torch.musa.Stream()

_stop = threading.Event()
_counts = {}


def _ssync(stream):
    """流级同步；MUSA 若无 Stream.synchronize 则用 Event 兜底（**不是**设备级）。"""
    try:
        stream.synchronize()
        return
    except AttributeError:
        pass
    e = torch.musa.Event()
    with torch.musa.stream(stream):
        e.record()
    e.synchronize()


def copy_worker():
    n = 0
    with torch.musa.stream(s_copy):
        while not _stop.is_set():
            d.copy_(h, non_blocking=True)
            n += 1
            if n % 32 == 0:
                _ssync(s_copy)          # 只等本流，不碰 compute
    _ssync(s_copy)
    _counts["copy"] = n


def comp_worker():
    n = 0
    with torch.musa.stream(s_comp):
        while not _stop.is_set():
            A @ B
            n += 1
            if n % 16 == 0:
                _ssync(s_comp)          # 只等本流，不碰 copy
    _ssync(s_comp)
    _counts["comp"] = n


def gpu_util_sampler(tag, sink):
    """旁证：采 %GPU。找不到工具就静默跳过。"""
    while not _stop.is_set():
        try:
            out = subprocess.run(
                ["mthreads-gmi", "-q", "-i", "0"],
                capture_output=True, text=True, timeout=5).stdout
            val = None
            for line in out.splitlines():
                ls = line.strip()
                if ls.startswith("Gpu") and ":" in ls:
                    tail = ls.split(":")[-1].strip().rstrip("%")
                    if tail.isdigit():
                        val = int(tail)
                        break
            if val is not None:
                sink.append(val)
        except Exception:
            pass
        time.sleep(3)


def run_phase(name, copies, compute, seconds, progress=False):
    _stop.clear()
    _counts.clear()
    util = []
    ths = []
    if copies:
        ths.append(threading.Thread(target=copy_worker, daemon=True))
    if compute:
        ths.append(threading.Thread(target=comp_worker, daemon=True))
    if copies and compute:
        ths.append(threading.Thread(target=gpu_util_sampler, args=("u", util), daemon=True))
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    if progress:
        next_at = 30.0
        while time.perf_counter() - t0 < seconds:
            time.sleep(min(5.0, seconds - (time.perf_counter() - t0)))
            el = time.perf_counter() - t0
            if el >= next_at:
                n_c = _counts.get("copy") or 0
                n_m = _counts.get("comp") or 0
                # 中间速率用锁外近似（worker 只在结束写 _counts）⇒ 用全局进度探针
                print(f"    t={el:5.0f}s   util(最近)={util[-1] if util else -1}%"
                      f"  copies_so_far={n_c} gemm_so_far={n_m}", flush=True)
                next_at += 30.0
    else:
        time.sleep(seconds)
    _stop.set()
    for t in ths:
        t.join(timeout=30)
    dt = time.perf_counter() - t0
    n_copy = _counts.get("copy", 0)
    n_comp = _counts.get("comp", 0)
    gib = n_copy * COPY_BYTES / 2**30
    cps = gib / dt if n_copy else 0.0
    gps = n_comp / dt if n_comp else 0.0
    um = (sum(util) / len(util)) if util else -1
    print(f"  {name:<20} {dt:6.1f}s copies={n_copy:<6} gemm={n_comp:<7} "
          f"copy={cps:6.2f} GiB/s  comp={gps:8.1f} gemm/s  gpu%={um:.0f}", flush=True)
    return cps, gps, um, util


def main():
    print("== S5000 长稳 overlap（流级同步版）==", flush=True)
    res = {}
    res["solo_copy1"], _, _, _ = run_phase("solo copy (pre)", True, False, SOLO_SEC)
    _, res["solo_comp1"], _, _ = run_phase("solo compute (pre)", False, True, SOLO_SEC)
    res["conc_copy"], res["conc_comp"], um, util = run_phase(
        "concurrent 600s", True, True, CONC_SEC, progress=True)
    res["solo_copy2"], _, _, _ = run_phase("solo copy (post)", True, False, SOLO_SEC)
    _, res["solo_comp2"], _, _ = run_phase("solo compute (post)", False, True, SOLO_SEC)

    # 用前后平均抵御热降频
    sc = 0.5 * (res["solo_copy1"] + res["solo_copy2"])
    sm = 0.5 * (res["solo_comp1"] + res["solo_comp2"])
    cr = res["conc_copy"] / sc if sc else 0
    mr = res["conc_comp"] / sm if sm else 0
    print("\n=== 结论 ===")
    print(f"H2D  retained = {cr*100:5.1f}%  ({sc:.2f} -> {res['conc_copy']:.2f} GiB/s)")
    print(f"GEMM retained = {mr*100:5.1f}%  ({sm:.1f} -> {res['conc_comp']:.1f} gemm/s)")
    print(f"平均 %GPU (concurrent) = {um:.0f}%")
    if util:
        util_s = sorted(util)
        p10 = util_s[len(util_s)//10]
        p50 = util_s[len(util_s)//2]
        print(f"  %GPU p10/p50 = {p10}% / {p50}%   (n={len(util)})")
    print(f"overlap 效率 = {cr + mr:.2f}   (2.0 完美 / 1.0 串行)")
    if mr < 0.6:
        print("⚠️ compute 未被压满 ⇒ H2D 数字不可信")
    else:
        print("✅ compute 确实在跑")
    print("RESULT " + json.dumps({
        "solo_copy_gibs": round(sc, 2), "conc_copy_gibs": round(res["conc_copy"], 2),
        "copy_retained": round(cr, 3),
        "solo_comp_ips": round(sm, 1), "conc_comp_ips": round(res["conc_comp"], 1),
        "comp_retained": round(mr, 3),
        "gpu_util_mean": round(um, 1) if um >= 0 else None,
        "overlap_eff": round(cr + mr, 3),
    }))


main()
