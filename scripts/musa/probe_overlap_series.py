"""S5000 长稳验证（10+ 分钟）：计算满载 + 持续 H2D，**逐秒时间序列**。

为什么必须长时间 + 逐秒采样：
  短测试的聚合均值会把**周期性 DMA 回压**抹平 —— 看起来 53 GiB/s 稳定，
  实际是「一阵快、一阵卡」。需要时间序列才能看到周期与回压深度。

关键修正（相对前两版）：
  1. ⛔ 不用 `torch.musa.synchronize()` 节流（设备级，会互相阻塞）⇒ 改**流级**同步。
  2. 计数器**增量更新**（带锁），采样线程每秒抓一次快照 ⇒ 得到真实时间序列。
  3. 全程采样 mthreads-gmi %GPU。

输出：
  * 每秒 sample：t, copy_MiB_delta, gemm_delta, gpu%
  * 汇总：p10/p50/p90 速率、回压率（速率<50% 中位的秒数占比）、
    以及用简单自相关粗测周期（报告前 5 个峰的滞后秒数）
  * 机器可读的 RESULT json
"""
import json
import statistics
import subprocess
import threading
import time

import torch

DEV = torch.device("musa")
COPY_BYTES = 512 * 1024 * 1024
SOLO_SEC = 45
CONC_SEC = 660          # 11 分钟并发，覆盖任何 <10min 的周期

h = torch.empty(COPY_BYTES, dtype=torch.uint8, pin_memory=True)
d = torch.empty(COPY_BYTES, dtype=torch.uint8, device=DEV)
A = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
B = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)

s_copy = torch.musa.Stream()
s_comp = torch.musa.Stream()

_stop = threading.Event()
_lock = threading.Lock()
_n = {"copy": 0, "comp": 0}


def _bump(key):
    with _lock:
        _n[key] += 1


def _ssync(stream):
    """流级同步（不是设备级）。"""
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
    with torch.musa.stream(s_copy):
        i = 0
        while not _stop.is_set():
            d.copy_(h, non_blocking=True)
            _bump("copy")
            i += 1
            if i % 32 == 0:
                _ssync(s_copy)
    _ssync(s_copy)


def comp_worker():
    with torch.musa.stream(s_comp):
        i = 0
        while not _stop.is_set():
            A @ B
            _bump("comp")
            i += 1
            if i % 16 == 0:
                _ssync(s_comp)
    _ssync(s_comp)


def _gpu_pct():
    try:
        out = subprocess.run(["mthreads-gmi", "-q", "-i", "0"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            ls = line.strip()
            if ls.startswith("Gpu") and ":" in ls:
                t = ls.split(":")[-1].strip().rstrip("%")
                if t.isdigit():
                    return int(t)
    except Exception:
        pass
    return -1


def run_phase(name, copies, compute, seconds, series):
    _stop.clear()
    with _lock:
        _n["copy"] = 0
        _n["comp"] = 0
    ths = []
    if copies:
        ths.append(threading.Thread(target=copy_worker, daemon=True))
    if compute:
        ths.append(threading.Thread(target=comp_worker, daemon=True))
    torch.musa.synchronize()
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    prev_c = prev_m = 0
    prev_t = t0
    next_report = 60.0
    while True:
        time.sleep(1.0)
        now = time.perf_counter()
        if now - t0 >= seconds:
            break
        with _lock:
            c, m = _n["copy"], _n["comp"]
        dc, dm = c - prev_c, m - prev_m
        dt = now - prev_t
        prev_c, prev_m, prev_t = c, m, now
        if series is not None:
            series.append({
                "t": round(now - t0, 2),
                "copy_gibs": round(dc * COPY_BYTES / 2**30 / dt, 3),
                "gemm_ips": round(dm / dt, 2),
                "gpu": _gpu_pct(),
            })
        el = now - t0
        if el >= next_report:
            print(f"    t={el:6.0f}s  copy={series[-1]['copy_gibs']:6.2f} GiB/s  "
                  f"gemm={series[-1]['gemm_ips']:8.1f}/s  gpu={series[-1]['gpu']}%",
                  flush=True)
            next_report += 60.0
    _stop.set()
    for t in ths:
        t.join(timeout=30)
    dt = time.perf_counter() - t0
    with _lock:
        c, m = _n["copy"], _n["comp"]
    gib = c * COPY_BYTES / 2**30
    print(f"  {name:<20} {dt:6.1f}s copies={c:<6} gemm={m:<7} "
          f"copy={gib/dt if c else 0:6.2f} GiB/s  comp={m/dt if m else 0:8.1f} gemm/s",
          flush=True)
    return (gib / dt if c else 0.0), (m / dt if m else 0.0)


def autocorr_peaks(xs, topk=5):
    """粗测周期：自相关在 lag>=3 上的局部极大。"""
    n = len(xs)
    if n < 20:
        return []
    mu = statistics.fmean(xs)
    var = statistics.fmean([(v - mu) ** 2 for v in xs]) or 1e-9
    ac = []
    for lag in range(3, min(n // 2, 300)):
        s = sum((xs[i] - mu) * (xs[i + lag] - mu) for i in range(n - lag))
        ac.append((s / (n - lag) / var, lag))
    peaks = [(v, l) for v, l in ac if v > 0.15]
    peaks.sort(reverse=True)
    out, used = [], []
    for v, l in peaks:
        if all(abs(l - u) > 3 for u in used):
            out.append((round(v, 3), l))
            used.append(l)
        if len(out) >= topk:
            break
    return out


def main():
    print("== S5000 长稳 overlap（逐秒时间序列，流级同步）==", flush=True)
    s1, _ = run_phase("solo copy (pre)", True, False, SOLO_SEC, None)
    _, m1 = run_phase("solo compute (pre)", False, True, SOLO_SEC, None)

    series = []
    cc, cm = run_phase("concurrent 660s", True, True, CONC_SEC, series)

    s2, _ = run_phase("solo copy (post)", True, False, SOLO_SEC, None)
    _, m2 = run_phase("solo compute (post)", False, True, SOLO_SEC, None)

    sc = 0.5 * (s1 + s2)
    sm = 0.5 * (m1 + m2)
    cr = cc / sc if sc else 0
    mr = cm / sm if sm else 0

    copies = [s["copy_gibs"] for s in series]
    gemms = [s["gemm_ips"] for s in series]
    gpus = [s["gpu"] for s in series if s["gpu"] >= 0]

    print("\n=== 时间序列统计（concurrent）===")
    def stats(name, xs, unit):
        if not xs:
            print(f"  {name}: 无数据")
            return {}
        ys = sorted(xs)
        def q(p):
            return ys[min(len(ys) - 1, int(p * (len(ys) - 1)))]
        med = statistics.median(xs)
        low = sum(1 for v in xs if v < 0.5 * med)
        print(f"  {name}: p10={q(.1):7.2f} p50={med:7.2f} p90={q(.9):7.2f} {unit}"
              f"  | min={ys[0]:.2f} max={ys[-1]:.2f}"
              f"  | 回压秒(<50%中位)={low}/{len(xs)} ({low/len(xs)*100:.0f}%)")
        return {"p10": q(.1), "p50": med, "p90": q(.9), "min": ys[0],
                "max": ys[-1], "stall_frac": round(low / len(xs), 3)}

    cs = stats("copy GiB/s", copies, "GiB/s")
    consts = stats("gemm /s  ", gemms, "gemm/s")
    if gpus:
        print(f"  %GPU: mean={statistics.fmean(gpus):.0f}%  "
              f"min={min(gpus)}%  p50={statistics.median(gpus):.0f}%  n={len(gpus)}")

    print("\n=== 周期性（自相关峰）===")
    pc = autocorr_peaks(copies)
    pg = autocorr_peaks(gemms)
    print(f"  copy 周期峰 (corr, lag秒): {pc if pc else '无明显周期'}")
    print(f"  gemm 周期峰 (corr, lag秒): {pg if pg else '无明显周期'}")

    print("\n=== 结论 ===")
    print(f"H2D  retained = {cr*100:5.1f}%  ({sc:.2f} -> {cc:.2f} GiB/s)")
    print(f"GEMM retained = {mr*100:5.1f}%  ({sm:.1f} -> {cm:.1f} gemm/s)")
    print(f"overlap 效率 = {cr + mr:.2f}   (2.0 完美 / 1.0 串行)")
    print("判定：" + ("可重叠" if (cr + mr) > 1.3 else "不能重叠"))
    if mr < 0.6:
        print("⚠️ compute 未被压满 ⇒ H2D 数字不可信")

    print("RESULT " + json.dumps({
        "solo_copy_gibs": round(sc, 2), "conc_copy_gibs": round(cc, 2),
        "copy_retained": round(cr, 3),
        "solo_comp_ips": round(sm, 1), "conc_comp_ips": round(cm, 1),
        "comp_retained": round(mr, 3),
        "overlap_eff": round(cr + mr, 3),
        "copy_stats": cs, "gemm_stats": consts,
        "copy_periodic": pc, "gemm_periodic": pg,
        "n_samples": len(series),
    }))

    with open("/tmp/overlap_series.json", "w") as f:
        json.dump(series, f)
    print("series -> /tmp/overlap_series.json")


main()
