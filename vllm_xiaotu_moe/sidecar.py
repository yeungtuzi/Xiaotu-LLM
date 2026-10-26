"""辅助 GPU（挎斗）worker —— Python 侧。

开关（用户 2026-10-23 定）：**`XTU_SIDECAR=gpu0`** 才启用；
不设 / `none` / `0` / `off` / `false` ⇒ 标准路径（零行为变化）。

协议见 `dev-docs/SPEC_SIDECAR_HOSTFN.md` 与 `xiaotu_moe/csrc/moe/sidecar_ctl.hpp`：
  C++ host-function（每层）→ host_begin（命中置 0/发请求）→ CPU MoE →
  host_end（等 done/合并）；本模块的线程轮询 req_seq，算完写 side_out 并置 done_seq。

本文件先提供**可测试的管线**：`compute` 回调由调用方注入（mock 或真实 GPU0 池）。
"""
from __future__ import annotations

import os
import struct
import threading

import numpy as np
import time

import torch

_DISABLED = {"", "none", "0", "off", "false", "no"}


def sidecar_device() -> int | None:
    """解析 `XTU_SIDECAR`；返回 GPU 编号，或 None（标准路径）。"""
    v = os.environ.get("XTU_SIDECAR", "").strip().lower()
    if v in _DISABLED:
        return None
    if v == "gpu0":
        # "gpu0" 指【物理 GPU0】，必须按 CUDA_VISIBLE_DEVICES 换算成进程内 logical id。
        # 例：CUDA_VISIBLE_DEVICES=1,0 ⇒ 物理0 在 logical 1（主服务占 logical 0 = 物理1）。
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cvd:
            return 0
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        try:
            phys = [int(p) for p in parts]
        except ValueError:
            return 0  # UUID/MIG 形式：退回旧行为
        if 0 in phys:
            return phys.index(0)
        raise ValueError(
            "XTU_SIDECAR=gpu0 但物理 GPU0 不在 CUDA_VISIBLE_DEVICES="
            f"{cvd!r} 中；请用 GPUS=1,0 这类顺序让主服务占物理1、挎斗占物理0"
        )
    raise ValueError(
        f"XTU_SIDECAR={v!r} 不认识；只支持 'gpu0'，或 none/0/off 表示关闭"
    )


def load_hit_masks(path: str, n_layers: int, n_experts: int) -> torch.Tensor:
    """XTUP 表 → uint64 hit mask [n_layers, ceil(n_experts/64)]。"""
    with open(path, "rb") as f:
        buf = f.read()
    magic, ver, nl, _emax, n_ent = struct.unpack_from("<4sIIII", buf, 0)
    assert magic == b"XTUP" and ver == 1, f"bad pin table {magic!r} v{ver}"
    assert nl == n_layers, f"table has {nl} layers, model has {n_layers}"
    hdr = struct.calcsize("<4sIIII")
    w = (n_experts + 63) // 64
    mask = np.zeros((n_layers, w), dtype=np.uint64)
    for i in range(n_ent):
        layer, exp = struct.unpack_from("<HH", buf, hdr + 4 * i)
        if layer < n_layers and exp < n_experts:
            mask[layer, exp >> 6] |= np.uint64(1) << np.uint64(exp & 63)
    return torch.from_numpy(mask)


class SidecarWorker:
    """轮询 C++ ctl 的 worker；`compute` 由调用方注入。"""

    def __init__(self, engine, table_path: str, n_layers: int, n_experts: int,
                 hidden: int, max_qlen: int, max_k: int, compute=None):
        self.dev = sidecar_device()
        self.enabled = self.dev is not None
        self.engine = engine
        self.compute = compute or self._mock_compute
        self.stop = threading.Event()
        self.thread = None
        self.ctl = 0
        if not self.enabled:
            return

        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.hidden = int(hidden)
        self.max_qlen = int(max_qlen)
        self.max_k = int(max_k)
        # 必须在 InferenceMode(False) 下分配：worker 常在模型 forward 里首次创建，
        # 否则 pinned 缓冲会变成 inference tensor，之后 copy_ 会报
        # "Inplace update to inference tensor outside InferenceMode is not allowed"。
        with torch.inference_mode(False):
            self.masks = load_hit_masks(table_path, n_layers, n_experts)
            self.ids = torch.zeros(max_qlen * max_k, dtype=torch.int32, pin_memory=True)
            self.wts = torch.zeros(max_qlen * max_k, dtype=torch.float32, pin_memory=True)
            self.out = torch.zeros(max_qlen * hidden, dtype=torch.float32, pin_memory=True)

        self.ctl = engine.sidecar_new()
        engine.sidecar_config(
            self.ctl, n_layers, n_experts, hidden, max_qlen, max_k,
            self.masks.data_ptr(), self.ids.data_ptr(), self.wts.data_ptr(),
            self.out.data_ptr(),
        )
        engine.sidecar_attach(self.ctl)
        try:
            _mh = int(__import__("os").environ.get("XTU_SIDECAR_MIN_HITS", "1"))
        except Exception:
            _mh = 1
        engine.sidecar_set_min_hits(self.ctl, max(1, _mh))
        engine.sidecar_set_enabled(self.ctl, True)
        self.thread = threading.Thread(target=self._loop, name="xtu-sidecar", daemon=True)
        self.thread.start()

    def _mock_compute(self, layer, qlen, k, ids, wts, out, hid_ptr):
        out[: qlen * self.hidden].fill_(1.0)

    def _loop(self):
        last = 0
        while not self.stop.is_set():
            req, done = self.engine.sidecar_seq(self.ctl)
            if req == last:
                time.sleep(5e-5)
                continue
            last = req
            layer, qlen, k, ids_ptr, wts_ptr, hid_ptr, out_ptr = \
                self.engine.sidecar_request(self.ctl)
            try:
                _t0 = time.perf_counter()
                self.compute(layer, qlen, k, self.ids, self.wts, self.out, hid_ptr)
                if os.environ.get("XTU_SIDECAR_DEBUG"):
                    print(f"[xtu-sidecar] layer={layer} qlen={qlen} k={k} "
                          f"{(time.perf_counter()-_t0)*1e3:.2f}ms", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[xtu-sidecar] compute FAILED layer={layer}: "
                      f"{type(_e).__name__}: {_e}", flush=True)
            self.engine.sidecar_done(self.ctl, req)

    def close(self):
        if not self.enabled:
            return
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.engine.sidecar_set_enabled(self.ctl, False)
        self.engine.sidecar_detach()
        self.engine.sidecar_free(self.ctl)
        self.ctl = 0


if __name__ == "__main__":
    # 自测：FakeEngine 实现同一组方法，验证线程/请求/完成管线。
    class FakeEngine:
        def __init__(self):
            self.req = 0
            self.done = 0
            self.req_info = (0, 0, 0, 0, 0, 0, 0)
            self.attached = False
            self.out = None

        def sidecar_new(self):
            return 1

        def sidecar_config(self, ctl, nl, ne, h, mq, mk, masks, ids, wts, out):
            self.cfg = (nl, ne, h, mq, mk)

        def sidecar_attach(self, ctl):
            self.attached = True

        def sidecar_set_enabled(self, ctl, e):
            self.en = e

        def sidecar_seq(self, ctl):
            return self.req, self.done

        def sidecar_request(self, ctl):
            return self.req_info

        def sidecar_done(self, ctl, seq):
            self.done = seq

        def sidecar_detach(self):
            self.attached = False

        def sidecar_free(self, ctl):
            pass

    eng = FakeEngine()
    # 表：layer 0，expert 1
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    hdr = struct.pack("<4sIIII", b"XTUP", 1, 2, 64, 1)
    with open(path, "wb") as f:
        f.write(hdr + struct.pack("<HH", 0, 1))
    w = SidecarWorker(eng, path, 2, 64, 8, 4, 3)
    assert w.enabled and eng.attached
    eng.req_info = (0, 2, 2, 0, 0, 0, 0)
    eng.req = 1
    t0 = time.time()
    while eng.done != 1 and time.time() - t0 < 2:
        time.sleep(1e-4)
    assert eng.done == 1, "worker did not ack"
    assert float(w.out[: 2 * 8].max()) == 1.0, "mock compute not written"
    w.close()
    os.unlink(path)
    print("sidecar self-test OK; device =", sidecar_device())
