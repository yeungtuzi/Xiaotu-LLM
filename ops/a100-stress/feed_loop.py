#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只喂数据的微循环加压器（不解码）：专打「宿主侧喂数据 → GPU 消费」这条链。

为什么要它
----------
长稳压测里每次死亡要 2–3 小时，无法做参数级定位。本探针把**喂数据这一环**单独拎出来
反复压：尺寸从很小一路涨到「历史最大单次 DMA 的 2.5 倍」，看是否出现
①挂住（看门狗抓）②数据损坏（校验抓）③显存单调上涨（采样抓）。

三种喂法（对应三类嫌疑）
------------------------
  ring     轮转 tile 环的 nslots 个槽（生产 `_stage_tiles` 的形态）
  landing  两条侧流**交替写同一块 landing**（复刻 C=2 时两请求共享落点）
  grow     单块缓冲尺寸不断增大（复刻「第一次长上下文触发扩容」的那一刻）

安全
----
* 只用 GPU（默认 0 号卡），不碰端口、不发 HTTP、不写被测服务；
* `--minutes` 到时自动退出 0；看门狗触发打印全线程栈 + (轮次, 尺寸, 槽, 流) 后 exit 3；
* 不修改任何仓库文件，只写 `--out` 指定的日志。
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import threading
import time

import torch

from vllm_xiaotu_moe import gpu_prefill as gp


class Watchdog(threading.Thread):
    """主循环心跳停滞超过 hang_ms 就 dump 全线程栈并退出（把"挂住"变成快速失败）。"""

    def __init__(self, hang_ms: int, state: dict, out) -> None:
        super().__init__(daemon=True)
        self.hang_ms = hang_ms
        self.state = state
        self.out = out

    def run(self) -> None:
        last, last_t = -1, time.monotonic()
        while True:
            time.sleep(0.25)
            cur = self.state.get("iter", 0)
            now = time.monotonic()
            if cur != last:
                last, last_t = cur, now
                continue
            if (now - last_t) * 1000.0 > self.hang_ms:
                print(f"[feed] ⛔ 看门狗: 心跳停滞 {(now - last_t):.1f}s "
                      f"iter={cur} mode={self.state.get('mode')} "
                      f"bytes={self.state.get('bytes')} slot={self.state.get('slot')} "
                      f"stream={self.state.get('stream')}", flush=True)
                faulthandler.dump_traceback(all_threads=True)
                self.out.flush()
                os._exit(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--hang-ms", type=int, default=8000)
    ap.add_argument("--cap-mult", type=float, default=2.5,
                    help="尺寸上限 = 历史最大单次 DMA × 该倍数")
    ap.add_argument("--out", default="/tmp/feed_loop.out")
    args = ap.parse_args()

    out = open(args.out, "w", buffering=1)
    faulthandler.enable(file=out, all_threads=True)

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)

    # —— 真实几何：与生产 ds41f 一致，这样单次 DMA 尺寸就是生产形状 ——
    E, H, I, GK = 384, 5120, 1152, 32
    total = gp.prealloc_tile_ring(dev, E, H, I, group_k=GK, nslots=2)
    rings = [v for k, v in gp._TILE_RINGS.items() if k[0] == str(dev)]
    assert rings, "tile 环未分配"
    slot_tensors = rings[-1]
    nslots = len(slot_tensors)
    max_slot = max(int(t.numel()) * int(t.element_size())
                   for slot in slot_tensors for t in slot)
    cap = int(max_slot * args.cap_mult)
    print(f"[feed] prealloc={total / 2**30:.3f} GiB nslots={nslots} "
          f"max_slot_dma={max_slot / 2**20:.1f} MiB cap={cap / 2**20:.1f} MiB", flush=True)

    # 宿主侧：整块 pin 内存 + 一个可增长的 device 缓冲
    host = torch.empty(cap, dtype=torch.uint8, pin_memory=True)
    host.fill_(0x5A)
    big = torch.empty(cap, dtype=torch.uint8, device=dev)
    landing_key = next((k for k in gp._LANDING if k[0] == str(dev)), None)
    landing = gp._LANDING.get(landing_key) if landing_key else None
    assert landing is not None, "landing 未分配"
    land_b = int(landing.numel())

    streams = [torch.cuda.Stream(device=dev) for _ in range(2)]
    main_stream = torch.cuda.current_stream(dev)

    state = {"iter": 0, "mode": "init", "bytes": 0, "slot": -1, "stream": "main"}
    Watchdog(args.hang_ms, state, out).start()

    t_end = time.monotonic() + args.minutes * 60.0
    sizes_kb = [4, 64, 512, 4096, 16384]          # 起步尺寸（KiB）
    n = 0
    baseline_alloc = None
    drift_peak = 0

    while time.monotonic() < t_end:
        mode = ("ring", "landing", "grow")[n % 3]
        # 尺寸：按轮次阶梯放大，直到 cap
        sz = min(cap, sizes_kb[(n // 3) % len(sizes_kb)] * 1024 * (1 + (n // 15) % 8))
        sz = max(4096, sz)
        state.update(mode=mode, bytes=sz, iter=n)
        try:
            if mode == "ring":
                slot_i = n % nslots
                st = streams[n % 2]
                state["slot"] = slot_i
                state["stream"] = "side%d" % (n % 2)
                with torch.cuda.stream(st):
                    for t in slot_tensors[slot_i]:
                        nb = int(t.numel()) * int(t.element_size())
                        if nb <= sz:
                            t.copy_(host[:nb].view(t.shape), non_blocking=True)
                main_stream.wait_stream(st)
            elif mode == "landing":
                st = streams[n % 2]
                state["slot"] = -1
                state["stream"] = "side%d" % (n % 2)
                nb = min(sz, land_b)
                with torch.cuda.stream(st):
                    landing[:nb].copy_(host[:nb], non_blocking=True)
                    # 同一落点立刻被另一条流覆盖 ⇒ 复刻 C=2 共享落点
                    if nb > 4096:
                        landing[4096:nb].copy_(host[: nb - 4096], non_blocking=True)
                main_stream.wait_stream(st)
            else:
                state["slot"] = -1
                state["stream"] = "main"
                big[:sz].copy_(host[:sz], non_blocking=True)
        except Exception as e:                     # noqa: BLE001
            print(f"[feed] ⛔ 异常 mode={mode} bytes={sz}: {type(e).__name__} {e}", flush=True)
            faulthandler.dump_traceback(all_threads=True)
            return 4

        if n % 32 == 0:
            torch.cuda.synchronize(dev)
            # 校验：把刚写过的一段拷回来比对（抓静默损坏）
            got = big[:4096].clone()
            if n % 96 == 0 and mode == "grow":
                if not torch.equal(got.cpu(), host[:4096]):
                    print(f"[feed] ⛔ 数据损坏 iter={n} bytes={sz}", flush=True)
                    return 5
            alloc = torch.cuda.memory_allocated(dev)
            if baseline_alloc is None:
                baseline_alloc = alloc
            drift_peak = max(drift_peak, alloc - baseline_alloc)
        n += 1

    torch.cuda.synchronize(dev)
    print(f"[feed] ✅ 完成 iter={n} 峰值分配漂移={drift_peak / 2**20:.1f} MiB "
          f"cap={cap / 2**20:.1f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
