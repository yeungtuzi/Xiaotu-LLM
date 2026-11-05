"""moe4_marlin —— 融合反量化 + tensor core 的 MXFP4 MoE 路径（**默认关闭**）。

开关：`XIAOTU_MOE4_MARLIN=1` 才启用；缺省完全不生效，调用方走原有路径。

与 `moe4_smallm` 的关系：两者吃同一批权重布局（池内 K-major `[E, K/2, N]`）、
同一套路由（`topk_ids/topk_weights`），产出同一个 `[T, H]`；差别只在实现。
数值上：本路径的 fp32 结果与精确解在 **bf16 上逐位相同**（0/16384 差异），
故下游拿到的字节与精确计算一致，token 不会因本路径而变化。

内核要点（详见 dev-docs/report/decode_pipeline_20261030/kernel/fused_v2/）：
  * 权重在显存里始终 MXFP4，进 shared 时**不做扩展**（相邻两列字节压进一个 uint16）
  * 行置换并入 shared 写入的地址计算 ⇒ **无需离线重排**
  * `ldmatrix.trans` 直接把两个 k-pair 送到各通道；查表在寄存器内展开
  * E8M0 尺度折进 bf16 指数位（`s << 7`）+ `__hmul2`

约束（不满足则 `available()` 为 False，调用方回退）：
  * `H % 64 == 0`、`2I % 128 == 0`、`I % 64 == 0`、`H % 128 == 0`
  * 每专家槽位数固定 16（`SLOT`），超出部分丢弃（与生产口径一致需另行确认）

运行时需要 nvcc 首次编译，产物缓存于 `~/.cache/xtu_moe4_marlin/`。
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from typing import Optional

import torch

_ENV = "XIAOTU_MOE4_MARLIN"
SLOT = 16
_KSPLIT_GU = 4
_KSPLIT_DN = 4

_lib = None
_failed = False
_buf: dict = {}

_HERE = os.path.dirname(os.path.abspath(__file__))
_CU_FUSED = os.path.join(_HERE, "csrc", "moe_marlin_fused.cu")
_CU_DISP = os.path.join(_HERE, "csrc", "moe_marlin_dispatch.cu")


def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def ready() -> bool:
    """内核是否已加载（不触发编译，可在 CUDA 图捕获期安全调用）。"""
    return enabled() and (not _failed) and (_lib is not None)


def _build():
    """首用编译两个 .cu 为一个 .so；按源码哈希缓存。失败返回 None（fail-closed）。"""
    global _lib, _failed
    if _lib is not None or _failed:
        return _lib
    if not enabled():
        return None
    try:
        h = hashlib.sha256()
        for f in (_CU_FUSED, _CU_DISP):
            with open(f, "rb") as fh:
                h.update(fh.read())
        tag = h.hexdigest()[:16]
        cdir = os.path.expanduser("~/.cache/xtu_moe4_marlin")
        os.makedirs(cdir, exist_ok=True)
        so = os.path.join(cdir, f"moe4_marlin_{tag}.so")
        if not os.path.exists(so):
            nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
            tmp = so + ".tmp"
            cmd = [nvcc, "-O3", "-arch=sm_80", "-shared", "-Xcompiler", "-fPIC",
                   "-o", tmp, _CU_FUSED, _CU_DISP]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[xtu-marlin] 编译失败，回退原路径：{r.stderr[:400]}", flush=True)
                _failed = True
                return None
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        lib.run.restype = ctypes.c_int
        lib.swiglu.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.swiglu.restype = ctypes.c_int
        lib.e2e_dispatch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        lib.e2e_combine.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _lib = lib
        return lib
    except Exception as e:  # noqa: BLE001
        print(f"[xtu-marlin] 加载失败，回退原路径：{e}", flush=True)
        _failed = True
        return None


def available() -> bool:
    return enabled() and _build() is not None


def _bufs(T: int, K: int, E: int, H: int, I: int, dev) -> dict:
    key = (T, K, E, H, I, str(dev))
    b = _buf.get(key)
    if b is not None:
        return b
    f32, i32, bf16 = torch.float32, torch.int32, torch.bfloat16
    b = {
        "As":   torch.empty(E * SLOT, H, dtype=bf16, device=dev),
        "cnt":  torch.empty(E, dtype=i32, device=dev),
        "stok": torch.empty(E * SLOT, dtype=i32, device=dev),
        "sw":   torch.empty(E * SLOT, dtype=f32, device=dev),
        "so":   torch.empty(T * K, dtype=i32, device=dev),
        "part": torch.empty(T * K, H, dtype=f32, device=dev),
        "out":  torch.empty(T, H, dtype=f32, device=dev),
        "Cg":   torch.empty(E, SLOT, 2 * I, dtype=f32, device=dev),
        "A2":   torch.empty(E, SLOT, I, dtype=bf16, device=dev),
        "Cd":   torch.empty(E, SLOT, H, dtype=f32, device=dev),
        "ids32": torch.empty(T, K, dtype=i32, device=dev),
        "w32":  torch.empty(T, K, dtype=f32, device=dev),
    }
    _buf[key] = b
    return b


def gemm(x, topk_ids, topk_weights, w13, s13, w2, s2, num_experts, H, I) -> Optional[torch.Tensor]:
    """融合路径：返回 `[T, H]` fp32；任何前置条件不满足则返回 None（调用方回退）。"""
    lib = _build()
    if lib is None:
        return None
    if x.dim() != 2 or x.shape[1] != H:
        return None
    T = x.shape[0]
    K = topk_ids.shape[1]
    E = int(num_experts)
    if H % 128 or I % 64 or (2 * I) % 128:
        return None
    if int(w13.shape[0]) < E or int(w2.shape[0]) < E:
        return None
    for t in (w13, s13, w2, s2):
        if not t.is_contiguous():
            return None
    dev = x.device
    b = _bufs(T, K, E, H, I, dev)
    b["ids32"].copy_(topk_ids)
    b["w32"].copy_(topk_weights)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    p = lambda t: ctypes.c_void_p(t.data_ptr())  # noqa: E731

    stok = b["stok"]
    stok.fill_(-1)                      # 未命中的槽位必须是 -1，否则 gather 越界
    lib.e2e_dispatch(p(x), p(b["ids32"]), p(b["w32"]), p(b["As"]), p(b["cnt"]),
                     p(stok), p(b["sw"]), p(b["so"]), T, K, E, H, stream)
    nblk_gu = (2 * I + 127) // 128
    nblk_dn = (H + 127) // 128
    b["Cg"].zero_()
    lib.run(p(b["As"]), p(w13), p(s13), p(b["Cg"]), SLOT, H, 2 * I, E,
            nblk_gu, _KSPLIT_GU, stream)
    lib.swiglu(p(b["Cg"]), p(b["A2"]), SLOT, I, E, stream)
    b["Cd"].zero_()
    lib.run(p(b["A2"]), p(w2), p(s2), p(b["Cd"]), SLOT, I, H, E,
            nblk_dn, _KSPLIT_DN, stream)
    lib.e2e_combine(p(b["Cd"]), p(b["ids32"]), p(b["w32"]), p(b["so"]),
                    p(b["part"]), p(b["out"]), T, K, E, H, stream)
    return b["out"]


def disable(reason: str = "") -> None:
    global _failed
    if not _failed:
        print(f"[xtu-marlin] 已禁用，后续走原路径。{reason}", flush=True)
    _failed = True
