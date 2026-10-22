"""GPU0 侧车运行时：池 + worker + 真实 compute（ds41f MXFP4 / qfn FP8 两条内核路径）。

用法（在 mixed_experts 的非捕获初始化块里）：
    rt = get_sidecar_runtime(engine, hidden, inter, topk, table, n_layers, n_experts)
    pool = build_layer_pool(..., device=torch.device("cuda:0"), table_path=table,
                            require_pin=False, layout=...)
    if pool is not None: rt.add_pool(layer_idx, pool)

worker 的 compute 与 `mixed_experts` 的 PIN 路径同构：
  slotmap → res → part_ids/-1 → 内核 → D2H 到 pinned out。
"""
from __future__ import annotations

import ctypes
import os
import struct
import threading

import torch

from vllm_xiaotu_moe import sidecar as _sc

_RT = None
_RT_LOCK = threading.Lock()


def _table_n_layers(path: str) -> int:
    with open(path, "rb") as f:
        hdr = f.read(20)
    _magic, _ver, nl, _emax, _ne = struct.unpack("<4sIIII", hdr)
    return int(nl)


def _hid_view(hid_ptr: int, qlen: int, hidden: int) -> torch.Tensor:
    buf = (ctypes.c_uint16 * (qlen * hidden)).from_address(hid_ptr)
    return torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16).reshape(qlen, hidden)


class SidecarRuntime:
    def __init__(self, engine, hidden: int, inter: int, topk: int, table: str,
                 n_layers: int, n_experts: int, use_fp8: bool, device: int = 0,
                 swiglu_limit: float = 0.0, max_qlen: int = 32, max_k: int = 8):
        self.engine = engine
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.topk = int(topk)
        self.table = table
        self.use_fp8 = bool(use_fp8)
        self.device = int(device)
        self.swiglu_limit = float(swiglu_limit or 0.0)
        self.pools: dict[int, object] = {}
        self._bufs: dict[int, tuple] = {}
        self._worker = _sc.SidecarWorker(
            engine, table, n_layers, n_experts, hidden, max_qlen, max_k,
            compute=self.compute,
        )
        self._dev = f"cuda:{self.device}"

    def add_pool(self, layer: int, pool) -> None:
        self.pools[int(layer)] = pool

    def _buf(self, qlen: int):
        b = self._bufs.get(qlen)
        if b is None:
            b = (
                torch.empty(qlen, self.hidden, dtype=torch.bfloat16, device=self._dev),
                torch.empty(qlen, self.topk, dtype=torch.int32, device=self._dev),
                torch.empty(qlen, self.topk, dtype=torch.float32, device=self._dev),
            )
            self._bufs[qlen] = b
        return b

    def compute(self, layer: int, qlen: int, k: int, ids_host, wts_host, out_host,
                hid_ptr: int) -> None:
        pool = self.pools.get(int(layer))
        if pool is None:
            out_host.zero_()
            return
        h_gpu, ids_gpu, wts_gpu = self._buf(qlen)
        torch.cuda.set_device(self.device)
        h_gpu.copy_(_hid_view(hid_ptr, qlen, self.hidden))
        ids_gpu.copy_(ids_host[: qlen * k].view(qlen, k))
        wts_gpu.copy_(wts_host[: qlen * k].view(qlen, k))

        sl = pool.slotmap[ids_gpu.long()]
        res = (sl >= pool.n_local) & (ids_gpu >= 0)
        part_ids = torch.where(
            res, (sl - pool.n_local).to(torch.int32), torch.full_like(ids_gpu, -1)
        )
        zero = torch.zeros((), dtype=torch.float32, device=self._dev)
        pw = torch.where(res, wts_gpu, zero)
        if self.use_fp8:
            from vllm_xiaotu_moe.decode_moe_fp8 import gpu_moe_decode_fp8
            p = gpu_moe_decode_fp8(
                h_gpu, part_ids, pw, *pool.planes,
                self.hidden, self.inter, k,
                device=self._dev, swiglu_limit=self.swiglu_limit,
            )
        else:
            from vllm_xiaotu_moe import gpu_prefill as _gp
            p = _gp.gpu_moe_layer(
                h_gpu, part_ids, pw, *pool.planes,
                self.hidden, self.inter, k,
                device=self._dev, slot=pool.as_slot(),
                out_dtype=torch.float32, swiglu_limit=self.swiglu_limit,
            )
        out_host[: qlen * self.hidden].copy_(p.reshape(-1))

    def close(self) -> None:
        self._worker.close()


def get_sidecar_runtime(engine, hidden: int, inter: int, topk: int, table: str,
                        n_layers: int, n_experts: int, use_fp8: bool,
                        swiglu_limit: float = 0.0) -> SidecarRuntime | None:
    """进程级单例；未设 XTU_SIDECAR=gpu0 或表为空 ⇒ None。"""
    global _RT
    if _sc.sidecar_device() is None or not table or not os.path.exists(table):
        return None
    if int(n_layers) <= 0:
        n_layers = _table_n_layers(table)
    with _RT_LOCK:
        if _RT is None:
            _RT = SidecarRuntime(
                engine, hidden, inter, topk, table, n_layers, n_experts,
                use_fp8=use_fp8, swiglu_limit=swiglu_limit,
            )
        return _RT
