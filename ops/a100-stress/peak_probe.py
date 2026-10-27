#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整卡显存峰值探针（用于把预算里的 activation peak 算准，而不是用 profile 期的估值）。

用途
----
对一个目标上下文长度 N：发一条 ≈N token 的请求，同时以 `--hz` 频率采样
`nvidia-smi --query-gpu=memory.used`，记录**每张卡**在请求期间的峰值；
请求前后各采一次“静息值”。输出一行 CSV：

    ctx_tokens,peak_gpu0,peak_gpu1,peak_gpu2,idle_gpu1,idle_gpu2,elapsed_s,ok

这样 `activation(N) ≈ peak − idle` 就是这一档上下文的真实工作区峰值，
把它代进
    KV = util×total − consumed − activation(MAXLEN) − graph − reserve
即可判断“2×512K KV + PIN + draft 常驻”到底还剩多少余量。

安全
----
* 只发 loopback HTTP（禁用 proxy），只读 nvidia-smi；不写被测机器的任何文件。
* 单请求超时 `--timeout`（默认 1800 s），失败即记录 ok=0 继续下一档。
* 不做并发：一档测完再测下一档，避免与长稳压测的峰值混叠。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def gpu_used() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [int(x) for x in out.split()]


class Sampler(threading.Thread):
    def __init__(self, hz: float) -> None:
        super().__init__(daemon=True)
        self.period = 1.0 / hz
        self.peak: list[int] = []
        self.stop_evt = threading.Event()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                cur = gpu_used()
            except Exception:
                time.sleep(self.period)
                continue
            if not self.peak:
                self.peak = list(cur)
            else:
                self.peak = [max(a, b) for a, b in zip(self.peak, cur)]
            time.sleep(self.period)


def make_body(n_tokens: int, max_tokens: int) -> dict:
    # 一行 ≈ 13 token；行内容刻意贴近真实记录，避免特殊字符影响分词
    line = "record {i}: alpha beta gamma delta epsilon zeta eta theta iota kappa\n"
    n_lines = max(1, int(n_tokens / 13))
    body = "".join(line.format(i=i) for i in range(n_lines))
    return {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Summarize in one sentence.\n" + body}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }


def one(port: int, n_tokens: int, max_tokens: int, timeout: float, hz: float) -> str:
    idle = gpu_used()
    body = json.dumps(make_body(n_tokens, max_tokens)).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    s = Sampler(hz)
    t0 = time.perf_counter()
    s.start()
    ok, prompt_tokens = 0, 0
    try:
        with _OPENER.open(req, timeout=timeout) as f:
            resp = json.loads(f.read())
        usage = resp.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        ok = 1
    except Exception as e:
        print(f"# ctx={n_tokens} 失败: {type(e).__name__} {e}", flush=True)
    finally:
        elapsed = time.perf_counter() - t0
        s.stop_evt.set()
        s.join(timeout=5)
    peak = s.peak or [0] * len(idle)
    return ",".join(str(x) for x in [
        prompt_tokens or n_tokens,
        *peak,
        *idle[: max(0, len(idle) - 1)],
        f"{elapsed:.1f}", ok,
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--ctx", type=int, nargs="+", default=[32768, 131072, 262144, 524288])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--out", default="/tmp/peak_probe.csv")
    args = ap.parse_args()

    header = "ctx_tokens," + ",".join(f"peak_gpu{i}" for i in range(3)) + ",idle_gpu1,idle_gpu2,elapsed_s,ok"
    with open(args.out, "w") as f:
        f.write(header + "\n")
    print(header, flush=True)
    for ctx in args.ctx:
        line = one(args.port, ctx, args.max_tokens, args.timeout, args.hz)
        with open(args.out, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
