"""GPU0 挎斗运行时：池 + worker + 真实 compute（ds41f MXFP4 / qfn FP8 两条内核路径）。

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
from vllm_xiaotu_moe import sidecar_nmajor as _nm

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
        self.max_k = int(max_k)
        self.table = table
        self.use_fp8 = bool(use_fp8)
        self.device = int(device)
        self.swiglu_limit = float(swiglu_limit or 0.0)
        self.pools: dict[int, object] = {}
        self._slotmap_cpu: dict[int, object] = {}
        self._bufs: dict[int, tuple] = {}
        # [xtu 2026-10-24] MXFP4 N-major GEMV 只对 qlen=1 正确；qlen>1 时 host_begin 直接放弃
        # （不置 0 CPU 权重），避免命中专家被静默丢弃（见 docs/QUALITY_SIDECAR_SPEC_CONFLICT.md）。
        self._eff_qlen = int(max_qlen) if self.use_fp8 else 1
        self._worker = _sc.SidecarWorker(
            engine, table, n_layers, n_experts, hidden, self._eff_qlen, max_k,
            compute=self.compute,
        )
        self._dev = f"cuda:{self.device}"
        K = self.max_k
        with torch.inference_mode(False):
            self._nbuf = (
                torch.empty(K, 2 * self.inter, dtype=torch.float32, device=self._dev),
                torch.empty(K, self.inter, dtype=torch.bfloat16, device=self._dev),
                torch.empty(K, self.hidden, dtype=torch.float32, device=self._dev),
                torch.empty(self.hidden, dtype=torch.float32, device=self._dev),
            )
            self.slots_dev = torch.empty(K, dtype=torch.int32, device=self._dev)
            self.wts_dev = torch.empty(K, dtype=torch.float32, device=self._dev)
            self.h_gpu = torch.empty(self.hidden, dtype=torch.bfloat16, device=self._dev)
            self.slots_cpu = torch.empty(K, dtype=torch.int32, pin_memory=True)
            self.wts_cpu = torch.empty(K, dtype=torch.float32, pin_memory=True)
            self.slots_np = self.slots_cpu.numpy()
            self.wts_np = self.wts_cpu.numpy()
        self._t_dbg = __import__("os").environ.get("XTU_SIDECAR_DEBUG") == "2"
        self._warm_kernels()

    def _warm_kernels(self) -> None:
        """用合成 planes 编译全部 Triton kernel（一次即可）。"""
        try:
            torch.cuda.set_device(self.device)
            H, I = self.hidden, self.inter
            z = lambda *sh: torch.zeros(*sh, dtype=torch.uint8, device=self._dev)
            pl = (z(1, 2 * I, H // 2), z(1, 2 * I, H // 32),
                  z(1, H, I // 2), z(1, H, I // 32))
            h = torch.zeros(H, dtype=torch.bfloat16, device=self._dev)
            st = torch.zeros(1, dtype=torch.int32, device=self._dev)
            wt = torch.ones(1, dtype=torch.float32, device=self._dev)
            _nm.forward(h, st, wt, pl, H, I, 32)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"[xtu-sidecar] warmup 失败: {type(e).__name__}: {e}", flush=True)

    def add_pool(self, layer: int, pool) -> None:
        layer = int(layer)
        # MXFP4：转成 N-major（[S,N,K]），并释放 K-major，避免 2× 常驻
        if not self.use_fp8:
            try:
                pt = _nm.transpose_planes(pool)
                pt = tuple(t.to(self._dev) for t in pt)
                pool.planes = pt
                if layer in (0, 22):
                    print(f"[xtu-sidecar] layer{layer} planes={tuple(t.shape for t in pt)} "
                          f"dev={pt[0].device}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[xtu-sidecar] layer{layer} transpose 失败: {e}", flush=True)
        self.pools[layer] = pool
        self._slotmap_cpu[layer] = pool.slotmap.detach().cpu()


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
        torch.cuda.set_device(self.device)
        # MXFP4 只支持 qlen=1（N-major GEMV）；其它形状 fail-closed
        if not self.use_fp8:
            if qlen != 1:
                out_host.zero_(); return
        # MXFP4 + qlen=1：N-major 行 GEMV，只算命中 slot
        if (not self.use_fp8) and qlen == 1:
            sm = self._slotmap_cpu.get(int(layer))
            if sm is None:
                out_host.zero_(); return
            slots, ws = [], []
            for j in range(k):
                e = int(ids_host[j]); w = float(wts_host[j])
                if 0 <= e < sm.numel():
                    s = int(sm[e])
                    if s >= pool.n_local:
                        slots.append(s - pool.n_local); ws.append(w)
            if not slots:
                out_host.zero_(); return
            import time as _t
            _t0 = _t.perf_counter()
            n = len(slots)
            self.slots_np[:n] = slots
            self.wts_np[:n] = ws
            self.h_gpu.copy_(_hid_view(hid_ptr, 1, self.hidden)[0])
            self.slots_dev[:n].copy_(self.slots_cpu[:n], non_blocking=True)
            self.wts_dev[:n].copy_(self.wts_cpu[:n], non_blocking=True)
            _t1 = _t.perf_counter()
            _nm.forward(self.h_gpu, self.slots_dev[:n], self.wts_dev[:n], pool.planes,
                        self.hidden, self.inter, int(pool.gk), bufs=self._nbuf)
            _t2 = _t.perf_counter()
            out_host[: self.hidden].copy_(self._nbuf[3])
            if self._t_dbg:
                print(f"[xtu-sidecar] layer={layer} hits={n} io={( _t1-_t0)*1e3:.3f}ms "
                      f"kernels={(_t2-_t1)*1e3:.3f}ms", flush=True)
            return
        h_gpu, ids_gpu, wts_gpu = self._buf(qlen)
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
                use_fp8=use_fp8, device=int(_sc.sidecar_device()), swiglu_limit=swiglu_limit,
            )
        return _RT
