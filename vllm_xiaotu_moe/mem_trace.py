# -*- coding: utf-8 -*-
"""逐次显存分配的插桩（回答"到底是谁在涨"）。

为什么需要它
------------
`nvidia-smi` / `memory_allocated` 只给**总量**：看到 reserved 从 30 GiB 涨到 40 GiB，
仍然不知道是权重、KV、工作区、还是某个 dict 持有者。PyTorch 自带
`torch.cuda.memory._record_memory_history()`，会为**每一次** alloc/free 记录
尺寸与 **Python 调用栈**，`_dump_snapshot()` 落盘后可离线比对两份快照：

    在两个时间点各 dump 一次 ⇒ 按「调用栈」聚合存活块字节 ⇒ 差值最大的栈
    就是**在涨的那一处**（不是"某个模块大概有问题"）。

用法（env 门控，默认全关，不影响生产）
--------------------------------------
    XIAOTU_MEM_TRACE=1                 总开关
    XIAOTU_MEM_TRACE_MAX_ENTRIES=...   记录条数上限（默认 200000，约占数百 MiB 宿主内存）
    XIAOTU_MEM_TRACE_DIR=/tmp/xtu_mem  快照目录
    XIAOTU_MEM_TRACE_EVERY_S=600       每隔多久自动 dump 一次

纪律
----
* 只在 `XIAOTU_MEM_TRACE=1` 时生效；关闭时 `start()/maybe_dump()` 是空操作。
* 记录本身**不分配显存**，只占宿主内存；`_dump_snapshot` 是宿主 IO。
* 不修改 vLLM 源码：用 PyTorch 公开的诊断 API 旁路采集。
"""
from __future__ import annotations

import os
import threading
import time

import torch

_STARTED = False
_LAST = [0.0]
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("XIAOTU_MEM_TRACE", "0") == "1"


def start() -> None:
    """尽早调用（模型加载前），否则权重的分配点抓不到。"""
    global _STARTED
    if not enabled() or _STARTED:
        return
    with _LOCK:
        if _STARTED:
            return
        try:
            max_entries = int(os.environ.get("XIAOTU_MEM_TRACE_MAX_ENTRIES", "200000"))
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python",
                max_entries=max_entries,
            )
            _STARTED = True
            print(f"[xtu-mem] 逐次分配记录已开启 (max_entries={max_entries})", flush=True)
        except Exception as e:                    # noqa: BLE001
            print(f"[xtu-mem] 开启失败: {type(e).__name__} {e}", flush=True)


def dump(tag: str = "") -> str | None:
    """落一份快照，返回路径。失败返回 None（绝不影响服务）。"""
    if not enabled():
        return None
    d = os.environ.get("XIAOTU_MEM_TRACE_DIR", "/tmp/xtu_mem")
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"snap_{time.strftime('%Y%m%dT%H%M%S')}_{tag or 'x'}.pickle")
        torch.cuda.memory._dump_snapshot(path)
        print(f"[xtu-mem] 快照已落盘: {path}", flush=True)
        return path
    except Exception as e:                        # noqa: BLE001
        print(f"[xtu-mem] dump 失败: {type(e).__name__} {e}", flush=True)
        return None


def maybe_dump(tag: str = "periodic") -> None:
    """按 `XIAOTU_MEM_TRACE_EVERY_S` 限频 dump（供 60 s 心跳调用）。"""
    if not enabled():
        return
    every = float(os.environ.get("XIAOTU_MEM_TRACE_EVERY_S", "600") or 600)
    now = time.monotonic()
    if now - _LAST[0] < every:
        return
    _LAST[0] = now
    dump(tag)
