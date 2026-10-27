#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A100 生产服务长稳监控（与 replay_driver.py 配套，只读）

每 <interval> 秒采样一次，全部写入 CSV + 事件日志：
  1) API 层：/health 是否 200、响应耗时；/metrics 的 num_requests_running /
     num_requests_waiting / KV 使用率 / 累计 token 与完成请求数；
  2) 进程层：vLLM 服务进程的 RSS、线程数、fd 数（fd 单调增长是典型泄漏信号）；
  3) 系统层：MemAvailable、PSI memory、loadavg、/proc/vmstat 的 oom_kill 计数；
  4) GPU 层：nvidia-smi 的显存占用 / 利用率 / 温度 / 功耗。

判定（写 alerts.log，并有事件行）：
  * /health 连续 3 次失败 ⇒ SERVICE_DOWN
  * 服务 PID 变化 ⇒ SERVICE_RESTARTED（重启会重置 RSS/KV，指标要分段看）
  * oom_kill 计数增加 ⇒ OOM_KILL
  * RSS / GPU 显存 / KV 使用率在滑动窗口内持续单调上升且斜率超阈值 ⇒ LEAK_SUSPECT
  * 无请求（running+waiting=0）时 KV 使用率仍 > <idle-kv> ⇒ KV_NOT_RELEASED

只读：不 kill、不写被测进程的任何文件，只写自己的日志目录。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ⛔ 只用 loopback：显式禁用 proxy 解析
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
from collections import deque

METRIC_KEYS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:gpu_cache_usage_perc": "kv_usage",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "gen_tokens_total",
    "vllm:request_success_total": "requests_ok_total",
    "vllm:request_failure_total": "requests_fail_total",
    "vllm:num_preemptions_total": "preemptions_total",
}


def now() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts if ts is not None else now()))


def read_metrics(port: int, timeout: float = 4.0) -> dict:
    out = {}
    try:
        with _OPENER.open(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                key = parts[0].split("{", 1)[0]
                name = METRIC_KEYS.get(key)
                if not name:
                    continue
                try:
                    out[name] = out.get(name, 0.0) + float(parts[1])
                except ValueError:
                    pass
    except Exception:
        pass
    return out


def health(port: int, timeout: float = 4.0):
    t0 = now()
    try:
        with _OPENER.open(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            resp.read(32)
            return resp.status == 200, (now() - t0) * 1000
    except Exception:
        return False, (now() - t0) * 1000


def read_meminfo() -> tuple[float, float]:
    total = avail = 0
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k == "MemTotal":
                total = int(v.split()[0]) / 1048576
            elif k == "MemAvailable":
                avail = int(v.split()[0]) / 1048576
    except Exception:
        pass
    return total, avail


def read_psi() -> float:
    try:
        for line in open("/proc/pressure/memory"):
            if line.startswith("some"):
                for part in line.split():
                    if part.startswith("avg10="):
                        return float(part.split("=")[1])
    except Exception:
        pass
    return -1.0


def read_load1() -> float:
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:
        return -1.0


def read_oom_kill() -> int:
    try:
        for line in open("/proc/vmstat"):
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except Exception:
        pass
    return -1


def gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip().splitlines()
        mem = []
        util = []
        temp = []
        power = []
        free = []
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            used, total_mib = int(float(parts[1])), int(float(parts[2]))
            mem.append(used)
            util.append(int(float(parts[3])))
            temp.append(int(float(parts[4])))
            power.append(float(parts[5]))
            free.append(total_mib - used)
        # gpu_free_min = **最满那张卡**还剩多少 MiB。
        # 【为什么加它 · 2026-10-26】该实例崩溃前的真实形态是"显存涨到顶就不涨了":
        #   LEAK_SUSPECT 只看**斜率**(30 min 窗口),涨到天花板后斜率归 0 ⇒ **全程无告警** ✗
        #   而崩溃前 1 小时该值只有 13.5 MiB(引擎日志里就是 allocation failed)。
        #   见 dev-docs/INCIDENT_2026-10-26_XID31.md §3。
        return {"gpu_mem_sum": sum(mem) if mem else -1, "gpu_mem_max": max(mem) if mem else -1,
                "gpu_free_min": min(free) if free else -1,
                "gpu_util_max": max(util) if util else -1, "gpu_temp_max": max(temp) if temp else -1,
                "gpu_power_sum": round(sum(power), 1) if power else -1}
    except Exception:
        return {"gpu_mem_sum": -1, "gpu_mem_max": -1, "gpu_free_min": -1,
                "gpu_util_max": -1, "gpu_temp_max": -1, "gpu_power_sum": -1}


def port_inode(port: int) -> str:
    """从 /proc/net/tcp{,6} 找到监听该端口的 socket inode。"""
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in open(path).read().splitlines()[1:]:
                f = line.split()
                local, state, inode = f[1], f[3], f[9]
                if state != "0A":
                    continue
                if int(local.split(":")[1], 16) == port:
                    return inode
        except Exception:
            continue
    return ""


def pid_for_inode(inode: str) -> int:
    if not inode:
        return 0
    needle = f"socket:[{inode}]"
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fddir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fddir):
                try:
                    if os.readlink(os.path.join(fddir, fd)) == needle:
                        return int(pid)
                except Exception:
                    continue
        except Exception:
            continue
    return 0


def proc_tree(root_pid: int) -> list[int]:
    """返回 root_pid 及其全部后代（vLLM 的 EngineCore / Worker 都是它的子进程）。"""
    if root_pid <= 0:
        return []
    pids = [root_pid]
    i = 0
    while i < len(pids):
        parent = pids[i]
        i += 1
        try:
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                try:
                    with open(f"/proc/{d}/stat") as f:
                        ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                except Exception:
                    continue
                if ppid == parent:
                    pids.append(int(d))
        except Exception:
            continue
    return pids


def tree_stats(root_pid: int) -> dict:
    pids = proc_tree(root_pid)
    rss = threads = fds = 0
    counts = 0
    for pid in pids:
        st = proc_stats(pid)
        if st["rss_mib"] < 0:
            continue
        counts += 1
        rss += st["rss_mib"]
        threads += max(0, st["threads"])
        fds += max(0, st["fds"])
    return {"procs": counts, "rss_sum_mib": round(rss, 1), "threads_sum": threads, "fds_sum": fds}


def proc_stats(pid: int) -> dict:
    if pid <= 0:
        return {"rss_mib": -1, "threads": -1, "fds": -1}
    try:
        rss = threads = -1
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
            elif line.startswith("Threads:"):
                threads = int(line.split()[1])
        fds = len(os.listdir(f"/proc/{pid}/fd"))
        return {"rss_mib": round(rss, 1), "threads": threads, "fds": fds}
    except Exception:
        return {"rss_mib": -1, "threads": -1, "fds": -1}


def resolve_port(preferred: int = 8000) -> int:
    """跟随"当前真实在跑"的 vLLM api_server 端口（开发端会反复启停、可能换端口）。"""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
        cands = []
        for line in out.splitlines():
            pm = re.search(r":(\d+)\s", line)
            m = re.search(r"pid=(\d+)", line)
            if not (pm and m):
                continue
            port = int(pm.group(1))
            try:
                cmd = open(f"/proc/{m.group(1)}/cmdline", "rb").read().decode(errors="ignore")
            except OSError:
                continue
            if "vllm.entrypoints" in cmd and "api_server" in cmd:
                cands.append(port)
        if preferred in cands:
            return preferred
        return cands[0] if cands else preferred
    except Exception:
        return preferred


def engine_log_path(port: int = 8000) -> str:
    """定位【当前实例】的引擎日志。

    2026-10-27 教训：开发端把工作目录换到了 ~/lvllm/prod_YYYYMMDD/，旧硬编码路径里的
    文件停在几天前，监控却一直"成功地"抓旧日志 ⇒ 致命行全丢。
    现在优先跟着监听端口的那个进程的 stdout/stderr（/proc/<pid>/fd/1,2），
    失败再退回"最近 10 分钟内被改过的 .log"，最后才按 mtime 猜。
    """
    # ① 端口 → pid → fd/1|2
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if f":{port}" not in line:
                continue
            m = re.search(r"pid=(\d+)", line)
            if not m:
                continue
            pid = m.group(1)
            for fd in ("1", "2"):
                try:
                    target = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if target.endswith(".log") and os.path.exists(target):
                    return target
    except Exception:
        pass
    # ② 退回：所有已知树里，最近 10 分钟被写过的 .log
    import glob
    cands = []
    for pat in ("~/lvllm/*/dev-docs/report/tuning/logs/*.log",
                "~/lvllm/vllm-xiaotu-moe/dev-docs/report/tuning/logs/*.log"):
        cands.extend(glob.glob(os.path.expanduser(pat)))
    cands = [c for c in cands if not os.path.islink(c) and os.path.isfile(c)]
    if not cands:
        return ""
    cutoff = time.time() - 600
    fresh = [c for c in cands if os.path.getmtime(c) >= cutoff]
    pool = fresh or cands
    return max(pool, key=os.path.getmtime)


def engine_log_tail(limit_bytes: int = 2_000_000) -> bytes:
    """取当前实例引擎日志的尾部；找不到返回空。"""
    try:
        newest = engine_log_path()
        if not newest:
            return b""
        size = os.path.getsize(newest)
        with open(newest, "rb") as src:
            src.seek(max(0, size - limit_bytes))
            return src.read()
    except Exception:
        return b""


def capture_forensics(out_dir: str, port: int) -> str:
    """健康检查判定服务 DOWN 时，自动把现场落到 incident-<ts>/（只读、尽力而为）。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(out_dir, f"incident-{stamp}")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        return ""

    def dump(cmd, dest):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout
            with open(os.path.join(d, dest), "w", encoding="utf-8") as f:
                f.write(out)
        except Exception:
            pass

    dump(["nvidia-smi"], "nvidia-smi.txt")
    dump(["journalctl", "-k", "--since", "-20 min", "--no-pager"], "kernel-20min.txt")
    dump(["ps", "-eo", "pid,ppid,etime,rss,args"], "ps.txt")
    dump(["ss", "-ltnp"], "listen.txt")
    try:
        newest = engine_log_path()
        if newest:
            size = os.path.getsize(newest)
            with open(newest, "rb") as src:
                src.seek(max(0, size - 3_000_000))
                data = src.read()
            with open(os.path.join(d, "engine-tail.log"), "wb") as f:
                f.write(data)
    except Exception:
        pass
    return d


class Slope:
    """滑动窗口线性斜率（每分钟增量），窗口按秒给定。"""

    def __init__(self, window_s: float):
        self.window = window_s
        self.buf = deque()

    def add(self, t: float, v: float):
        if v is None or v < 0:
            return
        self.buf.append((t, v))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()

    def per_minute(self):
        if len(self.buf) < 6:
            return None
        (t0, v0), (t1, v1) = self.buf[0], self.buf[-1]
        if t1 - t0 < self.window * 0.5:
            return None
        return (v1 - v0) / ((t1 - t0) / 60.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM 生产服务长稳监控（只读）")
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--rss-slope-mib-min", type=float, default=1.0, help="RSS 每分钟增长阈值")
    ap.add_argument("--gpu-slope-mib-min", type=float, default=4.0, help="显存每分钟增长阈值")
    # 【2026-10-26 新增】"贴顶"判据:最满那张卡剩余显存 < 该值 ⇒ GPU_FREE_LOW。
    #   它是 LEAK_SUSPECT 的**补充而非替代**:贴顶后斜率归零,只有这项还看得见。
    #   ⚠️ 默认值 **2048** 是**用真实事故数据回放定的**,不是拍的:
    #     拿 r41 冻结的 monitor.csv(623 样本)灌进本判据,各阈值的"首次触发 → 崩溃前提前量":
    #        512 / 1024 MiB ⇒ **0 条**(永世不触发, 死告警✗ —— 我第一版就是 512)
    #        1536 ⇒ 提前 27 min(2 条)
    #        2048 ⇒ **提前 46 min(4 条)**  ← 取这一档:足够时间反应,又不会天天吵
    #        3072 ⇒ 提前 58 min(5 条)    ← 想更早预警就设它
    #     该事故里 nvidia-smi 最低剩余 = **1127 MiB**(对应 torch 自己 arena 只剩 13.5 MiB ——
    #     **两个量不是一回事**,别拿 torch 那个数当阈值,否则告警永远不响)。
    ap.add_argument("--gpu-free-mib", type=int, default=2048,
                    help="最满 GPU 剩余显存低于此值(MiB)告警；0 = 关闭。默认 2048(实测定的,见注释)")
    ap.add_argument("--idle-kv", type=float, default=0.05, help="空闲时 KV 使用率告警阈值")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.expanduser("~"), "lvllm", "stress-20261026")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "monitor.csv")
    evt_path = os.path.join(out_dir, "monitor-events.log")
    alert_path = os.path.join(out_dir, "alerts.log")

    fields = ["ts", "health_ok", "health_ms", "running", "waiting", "kv_usage", "gpu_mem_sum", "gpu_mem_max",
              "gpu_util_max", "gpu_temp_max", "gpu_power_sum", "mem_total_gib", "mem_avail_gib", "psi_some10",
              "load1", "oom_kill", "server_pid", "procs", "rss_sum_mib", "rss_max_mib", "threads_sum", "fds_sum",
              "prompt_tokens_total", "gen_tokens_total", "requests_ok_total", "preemptions_total"]
    new_file = not os.path.exists(csv_path)
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=fields)
    if new_file:
        writer.writeheader()
        csv_fh.flush()
    evt = open(evt_path, "a", encoding="utf-8")
    alerts = open(alert_path, "a", encoding="utf-8")

    def event(kind: str, detail: str):
        line = json.dumps({"ts": iso(), "kind": kind, "detail": detail}, ensure_ascii=False)
        evt.write(line + "\n")
        evt.flush()
        alerts.write(f"[{iso()}] {kind}: {detail}\n")
        alerts.flush()
        print(f"[{iso()}] {kind}: {detail}", flush=True)

    rss_slope = Slope(1800)
    gpu_slope = Slope(1800)
    kv_slope = Slope(1800)

    deadline = now() + args.hours * 3600
    last_pid = 0
    down_streak = 0
    last_oom = read_oom_kill()
    last_alert = {}
    samples = 0
    inode_cache = (0.0, "")

    event("MONITOR_START", f"port={args.port} interval={args.interval}s hours={args.hours}")

    while now() < deadline:
        try:
            t = now()
            ok, ms = health(args.port)
            metrics = read_metrics(args.port)
            total, avail = read_meminfo()
            gpu = gpu_stats()
            oom = read_oom_kill()

            # 端口 → pid（每 60 s 重新解析一次，或 pid 失效时）
            inode = ""
            if t - inode_cache[0] > 60 or last_pid <= 0 or not os.path.exists(f"/proc/{last_pid}"):
                inode = port_inode(args.port)
                pid = pid_for_inode(inode)
                inode_cache = (t, inode)
                if pid != last_pid:
                    if last_pid:
                        event("SERVICE_RESTARTED", f"pid {last_pid} -> {pid}")
                    last_pid = pid
            pstat = proc_stats(last_pid)
            tstat = tree_stats(last_pid)

            row = {
                "ts": iso(), "health_ok": int(ok), "health_ms": round(ms, 1),
                "running": metrics.get("running", -1), "waiting": metrics.get("waiting", -1),
                "kv_usage": metrics.get("kv_usage", -1), "gpu_mem_sum": gpu["gpu_mem_sum"],
                "gpu_mem_max": gpu["gpu_mem_max"], "gpu_util_max": gpu["gpu_util_max"],
                "gpu_temp_max": gpu["gpu_temp_max"], "gpu_power_sum": gpu["gpu_power_sum"],
                "mem_total_gib": round(total, 1), "mem_avail_gib": round(avail, 1),
                "psi_some10": read_psi(), "load1": read_load1(), "oom_kill": oom,
                "server_pid": last_pid, "procs": tstat["procs"], "rss_sum_mib": tstat["rss_sum_mib"],
                "rss_max_mib": max(pstat["rss_mib"], 0), "threads_sum": tstat["threads_sum"], "fds_sum": tstat["fds_sum"],
                "prompt_tokens_total": metrics.get("prompt_tokens_total", -1),
                "gen_tokens_total": metrics.get("gen_tokens_total", -1),
                "requests_ok_total": metrics.get("requests_ok_total", -1),
                "preemptions_total": metrics.get("preemptions_total", -1),
            }
            writer.writerow(row)
            csv_fh.flush()
            samples += 1
            if samples % 60 == 0:
                os.fsync(csv_fh.fileno())

            rss_slope.add(t, tstat["rss_sum_mib"])
            gpu_slope.add(t, gpu["gpu_mem_sum"])
            kv_slope.add(t, metrics.get("kv_usage", -1))

            # ---- 判定 ----
            if not ok:
                down_streak += 1
                if down_streak == 3:
                    event("SERVICE_DOWN", f"/health 连续 3 次失败（端口 {args.port}）")
                    _np = resolve_port(args.port)
                    if _np != args.port:
                        event("PORT_SWITCH", f"检测到服务改到端口 {_np}（原 {args.port}），已跟随")
                        args.port = _np
                        down_streak = 0
                    snap = capture_forensics(out_dir, args.port)
                    if snap:
                        event("FORENSICS_SAVED", snap)
            else:
                if down_streak >= 3:
                    event("SERVICE_RECOVERED", f"健康检查恢复，失败 {down_streak} 次")
                down_streak = 0

            if oom >= 0 and last_oom >= 0 and oom > last_oom:
                event("OOM_KILL", f"vmstat oom_kill {last_oom} -> {oom}")
            if oom >= 0:
                last_oom = oom

            if row["running"] == 0 and row["waiting"] == 0 and row["kv_usage"] > args.idle_kv:
                if t - last_alert.get("kv_idle", 0) > 600:
                    event("KV_NOT_RELEASED", f"空闲但 KV 使用率 {row['kv_usage']:.3f}")
                    last_alert["kv_idle"] = t

            r_slope = rss_slope.per_minute()
            if r_slope is not None and r_slope > args.rss_slope_mib_min and t - last_alert.get("rss", 0) > 900:
                event("LEAK_SUSPECT", f"服务进程树 RSS 每分钟 +{r_slope:.1f} MiB（30 分钟窗口，当前 {row['rss_sum_mib']} MiB / {row['procs']} 进程）")
                last_alert["rss"] = t

            g_slope = gpu_slope.per_minute()
            if g_slope is not None and g_slope > args.gpu_slope_mib_min and t - last_alert.get("gpu", 0) > 900:
                event("LEAK_SUSPECT", f"GPU 显存每分钟 +{g_slope:.1f} MiB（30 分钟窗口，当前 {row['gpu_mem_sum']} MiB）")
                last_alert["gpu"] = t

            # ---- 【2026-10-26 新增】显存"贴顶"告警 ----------------------------------
            # 与 LEAK_SUSPECT 的区别(本实例崩溃正是漏在这一条):
            #   LEAK_SUSPECT = "涨得快"（斜率）⇒ 涨到天花板后斜率归 0 ⇒ 不再告警 ✗
            #   GPU_FREE_LOW = "已经没有余量"（水平）⇒ **贴顶期间每 10 分钟反复提醒** ✓
            # 判据取自现场:崩溃前 1 小时最满卡只剩 13.5 MiB，引擎日志里已有 4 次
            # `memory allocation failed with OOM`。见 dev-docs/INCIDENT_2026-10-26_XID31.md §3。
            free_min = gpu.get("gpu_free_min", -1)
            if (args.gpu_free_mib > 0 and free_min >= 0 and free_min < args.gpu_free_mib
                    and t - last_alert.get("gpu_free", 0) > 600):
                event("GPU_FREE_LOW",
                      f"最满 GPU 仅剩 {free_min} MiB（阈值 {args.gpu_free_mib} MiB，"
                      f"该卡已用 {row['gpu_mem_max']} MiB）⇒ 随时可能出现 "
                      f"`memory allocation failed with OOM`，是 Xid 31 的先兆")
                last_alert["gpu_free"] = t

            # 每小时一行摘要，便于快速看趋势
            if int(t) % 3600 < args.interval:
                event("HOURLY",
                      f"health={int(ok)} rss_sum={row.get('rss_sum_mib')}MiB procs={row.get('procs')} gpu={row.get('gpu_mem_sum')}MiB "
                      f"kv={row.get('kv_usage')} running={row.get('running')} waiting={row.get('waiting')} "
                      f"mem_avail={row.get('mem_avail_gib')}GiB fds={row.get('fds_sum')}")

        except Exception as exc:
            event("MONITOR_ERROR", f"采样异常（继续运行）：{type(exc).__name__}: {exc}")
        # 每 10 分钟落一次引擎日志尾：即使进程被 SIGKILL，也有最近的内核日志可查
        if samples % max(1, int(600 / args.interval)) == 0:
            tail = engine_log_tail()
            if tail:
                try:
                    with open(os.path.join(out_dir, "engine-tail-latest.log"), "wb") as f:
                        f.write(tail)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    pass

        time.sleep(max(0.0, args.interval - (now() - t)))

    event("MONITOR_STOP", "到达运行时长上限")
    csv_fh.close()
    evt.close()
    alerts.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise  # 正常退出/用法错误不算崩溃，不写 crash 文件
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        crash = os.path.join(os.path.expanduser("~"), "lvllm", "stress-20261026", f"crash-monitor-{stamp}.log")
        try:
            os.makedirs(os.path.dirname(crash), exist_ok=True)
            with open(crash, "w", encoding="utf-8") as f:
                f.write(tb)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        sys.stderr.write(tb)
        sys.stderr.flush()
        sys.exit(1)
    sys.exit(rc)
