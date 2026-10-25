"""FP8 (e4m3, block-128) GPU prefill kernels for the hybrid plugin.

Companion to ``gpu_prefill.py`` (which does MXFP4/block-32). Differences that
matter:

* weights are **1 byte/element** (not two elements/byte), so the K-major byte
  transpose in ``byte_transpose.ktranspose_bytes`` works unchanged -- it is a
  plain uint8 transpose;
* the block scale is **128x128** (not groupN=1/groupK=32), so with tile sizes
  that divide 128 the scale is a **single fp32 scalar per (k-block, n-block)**
  and can be loaded as a scalar instead of a [BK/32, BN] tile plus broadcast;
* A100 has no FP8 tensor cores, so the tile is decoded to bf16 and the MMA runs
  in bf16 -- the same thing the MXFP4 path does.

Decoding uses ``_e4m3_uint8_to_f32`` from vLLM's fp8_utils, the arithmetic
(fp8e4nv-free) decoder added for Ampere by the DSv4/V4.1 SM80 work; Triton has no
fp8e4nv type on SM80.

The down kernel applies the gated-SiLU clamp (``swiglu_limit``) the way the CPU
engine does: gate clamped on the upper side only, up clamped on both sides.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_uint8_to_f32(x):
    """uint8 里装的 E4M3 原始字节 -> float32（按位定义精确还原）。

    ⚠️ 本函数原来从 vllm...fp8_utils 导入，但 vLLM 0.32 已删除该符号
    （`ImportError: cannot import name '_e4m3_uint8_to_f32'`），于是在这里自带一份。
    E4M3: s(1) e(4) m(3)，bias=7；e==0 为次正规数（2^-6 * m/8）。
    """
    i = x.to(tl.int32)
    s = (i >> 7) & 1
    e = (i >> 3) & 0xF
    m = i & 0x7
    # 正规数：直接拼 float32 位模式（无 exp2）
    bits = ((e + 120) << 23) | (m << 20)
    normal = bits.to(tl.float32, bitcast=True)
    # 次正规数：m * 2^-9（精确）
    sub = m.to(tl.float32) * 1.953125e-03
    val = tl.where(e == 0, sub, normal)
    return tl.where(s == 1, -val, val)


from vllm_xiaotu_moe.byte_transpose import ktranspose_into
from vllm_xiaotu_moe.gpu_prefill import (
    _build_segmentation,
    _dma_hostbuf,
    _dma_hostbuf_range,
    _kmajor_bytes,
    _pin_engine_hostbufs,
    _prefetch_stream,
    _stage_begin,
    _stage_mark,
    _tile_ring,
    stage_tiles,
    staging_tile_experts,
)

FP8_BLOCK = 128

# Dedicated named buffers. NOTE: we deliberately do NOT use gpu_prefill._reuse
# here -- its cache key is (device, shape) and ignores the logical `key`, so for
# GLM-5.3 (H == 2I == 4096) the w13 row layout [E, 2I, H] and its K-major target
# [E, H, 2I] have the SAME shape and would hand back the same tensor, i.e. an
# in-place transpose that corrupts itself. (w2 escaped it only because
# [E, H, I] != [E, I, H].) Names keep the buffers distinct.
_FP8_BUF: dict = {}


def _dma2d_available(engine) -> bool:
    """Whether the engine exposes the pitched 2-D shard DMA (binding >= 2026-09-19).

    Cached per engine class because an installed .so built before that commit would
    otherwise make every layer take the slow staging path silently.
    """
    if not hasattr(engine, "copy_hostbuf_to_device_2d"):
        return False
    return os.environ.get("XIAOTU_GPF_DMA2D", "1") != "0"


def _buf(name: str, shape, device, dtype=torch.uint8):
    sh = tuple(int(v) for v in shape)
    k = (str(device), str(name), sh, dtype)
    t = _FP8_BUF.get(k)
    if t is None:
        t = torch.empty(sh, dtype=dtype, device=device)
        _FP8_BUF[k] = t
    return t


@triton.jit
def gate_up_kernel_fp8(
    x_ptr, x_stride, tok_ptr,
    seg_ptr,
    w13t_ptr, w13t_row_stride, s13t_ptr, s13t_row_stride,
    inter_ptr, inter_ld,
    W13_E, S13_E,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
):
    """inter[g, 0:2I] = x[tok[g], :] @ dequant(w13)^T for one expert's rows.

    w13t is [E, H, 2I] uint8 K-major; s13t is [E, H/128, 2I/128] fp32 K-major.
    Requires BK <= 128 (divides it) and BN <= 128 with the n-tile aligned, so the
    block scale is constant across the tile.
    """
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    n = pid_n * BN + tl.arange(0, BN)
    n_blk = (pid_n * BN) // 128
    # int64: an expert's K-major block exceeds 2^31 bytes for these models.
    w_off = pid_e.to(tl.int64) * W13_E
    s_off = pid_e.to(tl.int64) * S13_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, H, BK, num_stages=NS):
            a = tl.load(
                x_ptr + toks[:, None] * x_stride + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0)
            b_u8 = tl.load(
                w13t_ptr + w_off
                + (kk + tl.arange(0, BK))[:, None] * w13t_row_stride + n[None, :])
            sc = tl.load(s13t_ptr + s_off + (kk >> 7) * s13t_row_stride + n_blk)
            b = (_e4m3_uint8_to_f32(b_u8) * sc).to(tl.bfloat16)
            acc = tl.dot(a, b, acc)
        tl.store(
            inter_ptr + g_rows[:, None] * inter_ld + n[None, :],
            acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def down_kernel_fp8(
    inter_ptr, inter_ld, tok_ptr, wts_ptr,
    seg_ptr,
    w2t_ptr, w2t_row_stride, s2t_ptr, s2t_row_stride,
    out_ptr, out_ld,
    W2_E, S2_E,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """out[tok, :] += w * (silu_gate(gate) * up) @ dequant(w2)^T, per assignment."""
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    h = pid_h * BH + tl.arange(0, BH)
    h_blk = (pid_h * BH) // 128
    w_off = pid_e.to(tl.int64) * W2_E
    s_off = pid_e.to(tl.int64) * S2_E
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g_rows, mask=rmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for kk in tl.range(0, I, BK, num_stages=NS):
            gate = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(
                inter_ptr + g_rows[:, None] * inter_ld + (I + kk + tl.arange(0, BK))[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
            if LIMIT > 0.0:
                gate = tl.minimum(gate, LIMIT)
                up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
            act = (gate / (1.0 + tl.exp(-gate))) * up
            b_u8 = tl.load(
                w2t_ptr + w_off
                + (kk + tl.arange(0, BK))[:, None] * w2t_row_stride + h[None, :])
            sc = tl.load(s2t_ptr + s_off + (kk >> 7) * s2t_row_stride + h_blk)
            b = (_e4m3_uint8_to_f32(b_u8) * sc).to(tl.bfloat16)
            acc = tl.dot(act.to(tl.bfloat16), b, acc)
        tl.atomic_add(
            out_ptr + toks[:, None] * out_ld + h[None, :],
            acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


# ---------------------------------------------------------------------------
# Host-side assembly + staging (FP8 twins of gpu_prefill's MXFP4 versions)
# ---------------------------------------------------------------------------


def _fp8_tile_ring(device, n_experts: int, hidden: int, inter: int,
                   tile: "int | None" = None, nslots: int = 2):
    """⭐ 【微 ping/pong】FP8 的 tile 环(进程级复用,复用 `gpu_prefill._tile_ring`)。

    槽内 4 项与 `fp8_staging_bytes` 的 tile 分支**一一对应**:
      K-major w13 `[tile, H, 2I]` / 尺度 `[tile, H/128, 2I/128]` fp32 /
      K-major w2  `[tile, I, H]` / 尺度 `[tile, I/128, H/128]` fp32

    ⚠️ **尺度是 fp32(4 B/项)** ⇒ 这里用 **uint8 槽 + 末维 ×4** 再 `.view(torch.float32)`;
       若按**元素数**申请,只会拿到 1/4 字节 ⇒ **静默越界** ✗
    ⚠️ 为什么必须**进程级复用**(不自建、不每次层分配)⇒ 见 `gpu_prefill._tile_ring`
       的两条理由(不可回收碎片 + **跨流 use-after-free**),此处不重复 ✗
    """
    E = int(n_experts)
    t = int(tile if tile is not None else staging_tile_experts(E))
    b = FP8_BLOCK
    H, I = int(hidden), int(inter)
    # ⭐ `XIAOTU_GPF_W13TN=1`：w13 槽改按 **N-major `(t, 2I, H)`** 分配（尺度同步换轴）。
    #   动机：w13 的宿主分片本来就是 `[cr13, H]` 的 N-major 连续块 ⇒ N-major 环 = **纯拷贝**，
    #   省掉每 band 的 `ktranspose_into`（实测 `[gpf-km] tr=140–173 ms/8band`）。
    #   注意：w2 槽**不需要**改形状（`_w2tn()` 用的是同一块连续内存的另一种 view，见其注释）✓
    if _w13tn():
        shapes = ((t, 2 * I, H),
                  (t, (2 * I) // b, H // b * 4),
                  (t, I, H),
                  (t, I // b, H // b * 4))
    else:
        shapes = ((t, H, 2 * I),
                  (t, H // b, (2 * I) // b * 4),
                  (t, I, H),
                  (t, I // b, H // b * 4))
    return _tile_ring(device, shapes, max(2, int(nslots)))


def prealloc_fp8_buffers(device, n_experts: int, hidden: int, inter: int) -> None:
    """Allocate this layer's persistent named buffers on the CURRENT stream.

    Call this before switching to a side stream for the assembly. The buffers are
    cached by name and outlive every layer, so allocating them under the side
    stream would tie them to a stream they are later read from only through
    events; allocating up front on the main stream keeps the caching allocator's
    stream association simple.
    """
    E, H, I = int(n_experts), int(hidden), int(inter)
    # ⭐⭐ 2026-10-15 用户裁定:所有 GPU 预填充路径**一律**走 tile 形态(微 ping/pong),
    #    旧的"按层持久缓冲"机制**已删除** ✗
    #    ⇒ `staging_tile_experts` 现在**永不返回 E** ⇒ 下面**不再有**整层分支 ✓
    #    ⇒ 整层的 `raw13`/`raw2`/`km13`/`km2` **一个都不分配** ✓(那是 §601 里永不释放的整层槽)
    tile = staging_tile_experts(E)
    _fp8_tile_ring(device, E, H, I, tile)


def fp8_staging_bytes(n_experts: int, hidden: int, inter: int, ns: int = 2) -> int:
    """Peak device bytes the FP8 streaming path holds for one layer's weights.

    This is **not** the MXFP4 three-part accounting, because the FP8 assembler
    keeps different things alive: it fills persistent named buffers (``_buf``) for
    the raw *and* the K-major copy of BOTH matrices, and never frees ``raw13``
    before allocating w2. So the peak is

        raw13 + km13 + raw2 + km2 + scales + (one node's DMA staging per block)

    which for GLM-5.3 at TP=2 is **7.59 GiB** -- the caller passes the engine's
    per-rank geometry (E=288 experts, H=4096, I=2048/TP=1024; measured at runtime
    by the `GPU prefill ACTIVE ... preflight: staging ~7.59 GiB` line). The
    earlier "~3.7 GiB for E=144" figure here was an arithmetic slip: 144 is not
    the expert count the buffers are allocated with, and being 2x low is exactly
    the kind of error that lets a preflight pass and the forward OOM (§601).
    """
    E, H, I = int(n_experts), int(hidden), int(inter)
    b = FP8_BLOCK
    w13d = E * (2 * I) * H
    w2d = E * H * I
    s13d = E * ((2 * I) // b) * (H // b) * 4
    s2d = E * (H // b) * (I // b) * 4
    n = max(1, int(ns))
    dma = (w13d + w2d) // n          # one node's staging buffer, per weight block
    tile = staging_tile_experts(E)
    # ⭐⭐ 2026-10-15 用户裁定:所有 GPU 预填充路径**一律** tile 形态
    #    ⇒ `staging_tile_experts` **永不返回 E** ⇒ **整层记账分支已删除** ✗
    # ⭐ 【微 ping/pong】tile 环走 noraw 直填 ⇒ **不再有 raw 那一份**,只留 2 个
    #    (K-major 权重 + 尺度) 槽,外加按 tile 缩小的 DMA staging。
    # ⚠️ 与 `_fp8_tile_ring`(分配)**共用 `staging_tile_experts`** ⇒ 口径结构上不可能分叉
    #    —— 这正是 §600/§601 两次"预检说够、实际 OOM"的唯一防线 ✓
    # ⭐ 现在两者**都只可能是 tile 形态** ⇒ 分叉在结构上不可能发生 ✓
    frac = tile / E
    return int(2 * frac * (w13d + w2d + s13d + s2d) + frac * dma)


def kmajor_from_engine_shards_fp8(engine, device, hidden: int, inter: int,
                                  n_experts: int, dst=None):
    """K-major FP8 device weights built from the engine's own NUMA shards.

    FP8 twin of ``gpu_prefill.kmajor_from_engine_shards``. Two differences from
    the MXFP4 version, both structural (not just widths):

    * a shard row is ``H`` (gate/up) / ``I`` (down) **bytes**, not ``H/2``/``I/2``
      -- but the reassembly is byte-identical, and the K-major transpose is the
      same plain uint8 ``ktranspose_bytes``;
    * the block scale is ``[E, N/128, K/128]`` fp32 (groupN = groupK = 128),
      whereas MXFP4 is ``[E, N, K/32]`` e8m0. So the scale is read as fp32 and
      transposed to K-major with torch (it is ~1 MB/layer -- negligible).

    Returns ``(w13t, s13t, w2t, s2t)`` with the *weights* K-major and ready for
    the kernels, or ``None`` when the engine has no shards (caller falls back).
    """
    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    if ns < 2 or not geo["w13_node_bytes"] or not geo["w2_node_bytes"]:
        return None
    # Page-lock the engine's shards in place before the first DMA. They are
    # mmap'd/numa_alloc_onnode memory, i.e. pageable, and pageable H2D measured
    # 16.5 GB/s here vs 26.85 GB/s pinned -- the assembly is 91% of the FP8
    # prefill wall time, so this is the single biggest lever. Idempotent; it
    # locks existing pages and copies nothing.
    _pin_engine_hostbufs(engine)
    H, I, E = int(hidden), int(inter), int(n_experts)
    b = FP8_BLOCK

    # Assemble the canonical [E, 2I, H] / [E, H, I] row layout. Each node's shard is
    # a contiguous run of crows rows of `H` (resp. `I`) bytes that maps onto a row
    # range of the canonical layout at a larger pitch, i.e. a plain 2-D copy. We do
    # it with `cudaMemcpy2DAsync` **straight into the destination** rather than
    # DMA-to-staging + strided `copy_`: the strided form measured ~84 GB/s against
    # ~1361 GB/s contiguous, and (worse) one staging buffer per weight block forced
    # every DMA in the layer to wait for the previous copy.
    import time as _time

    _t_dma = _stage_begin()
    w13_raw = _buf("raw13", (E, 2 * I, H), device)
    c13 = int(geo["w13_cbytes"])
    cr13 = int(geo["w13_crows"])
    # 【②c】raw2 复用 raw13 的存储:raw13=(E,2I,H)=2.25GiB 恰好是 raw2=(E,H,I)=1.12GiB 的 2 倍,
    # 因而 raw13 的存储装得下 raw2 ⇒ 省 1.12 GiB 设备显存。
    # ⚠️ 代价:DMA/转置流水被串行化(见下方 w13 转置被提前到 w2 DMA 之前)。
    # `XIAOTU_GPF_RAW2_SHARE=0` 可退回独立缓冲(一键回滚)。
    if os.environ.get("XIAOTU_GPF_RAW2_SHARE", "0") == "1":
        w2_raw = w13_raw.view(-1)[: E * H * I].view(E, H, I)
    else:
        w2_raw = _buf("raw2", (E, H, I), device)
    c2 = int(geo["w2_cbytes"])
    cr2 = int(geo["w2_crows"])
    _dma2d = _dma2d_available(engine)
    # Phase timing with CUDA events (no syncs, so it cannot perturb what it
    # measures). `XIAOTU_GPF_STAGE=1` prints one line per layer.
    import os as _os

    _trace = _os.environ.get("XIAOTU_GPF_STAGE") == "1"
    _ev = {}

    def _hit(name):
        if _trace:
            e = torch.cuda.Event(enable_timing=True)
            e.record(torch.cuda.current_stream(device))
            _ev[name] = e

    w13t = None
    _hit("start")
    if _dma2d:
        stream = torch.cuda.current_stream(device).cuda_stream
        p13 = w13_raw.data_ptr()
        p2 = w2_raw.data_ptr()
        for n in range(ns):
            c0 = n * cr13
            # gate rows [c0, c0+cr13) then up rows [I+c0, I+c0+cr13) of [E, 2I, H].
            # The shard is `[E][gate cbytes][up cbytes]` -- i.e. per expert the gate
            # run is followed by the up run -- so the *source* run stride is
            # `2*c13`, NOT `cr13*H`. Each run is `cr13*H` contiguous bytes (the
            # node's gate rows for one expert), and the destination rows for
            # consecutive experts are `2*I*H` apart. Getting the source stride
            # wrong silently reads the up rows as gate rows for the next expert
            # (caught by scripts/test_gpu_prefill_fp8_assembly.py).
            for _off, _row in ((0, c0), (c13, I + c0)):
                got = engine.copy_hostbuf_to_device_2d(
                    0, n, _off, p13 + _row * H, 2 * I * H, 2 * c13, cr13 * H, E, stream)
                if int(got) != cr13 * H * E:
                    raise RuntimeError(
                        f"gpu-prefill: 2-D shard DMA failed (w13 node={n} off={_off}: "
                        f"{got}/{cr13 * H * E} bytes)")
        _hit("w13")
        # 【②c】w13 转置必须在 w2 的 DMA 之前完成 —— 否则 w2 的 DMA 会覆盖 raw13。
        if os.environ.get("XIAOTU_GPF_RAW2_SHARE", "0") == "1":
            w13t = _kmajor_bytes(w13_raw, _buf("km13", (E, H, 2 * I), device))
        for n in range(ns):
            c0 = n * cr2
            got = engine.copy_hostbuf_to_device_2d(
                1, n, 0, p2 + c0 * I, H * I, cr2 * I, cr2 * I, E, stream)
            if int(got) != cr2 * I * E:
                raise RuntimeError(
                    f"gpu-prefill: 2-D shard DMA failed (w2 node={n}: "
                    f"{got}/{cr2 * I * E} bytes)")
        _hit("w2")
    else:
        for n in range(ns):
            buf = _dma_hostbuf(engine, 0, n, int(geo["w13_node_bytes"]), device)
            blk = buf.view(E, 2, c13)
            c0 = n * cr13
            w13_raw[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(E, cr13, H))
            w13_raw[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(E, cr13, H))
            del buf, blk
        # 【②c】同上:先把 w13 转置做完,再让 w2 的 DMA 覆盖 raw13。
        if os.environ.get("XIAOTU_GPF_RAW2_SHARE", "0") == "1":
            w13t = _kmajor_bytes(w13_raw, _buf("km13", (E, H, 2 * I), device))
        for n in range(ns):
            buf = _dma_hostbuf(engine, 1, n, int(geo["w2_node_bytes"]), device)
            c0 = n * cr2
            w2_raw[:, c0:c0 + cr2, :].copy_(buf.view(E, cr2, I))
            del buf

    # Scales are a single full host copy (not sharded): [E, N/128, K/128] fp32.
    s13_raw = (_dma_hostbuf(engine, 2, 0, int(geo["w13_scale_bytes"]), device)
               .view(torch.float32).view(E, (2 * I) // b, H // b))
    s2_raw = (_dma_hostbuf(engine, 3, 0, int(geo["w2_scale_bytes"]), device)
              .view(torch.float32).view(E, H // b, I // b))
    _hit("scales")
    # Up to here the DMA launches and the strided assembly copies are all queued on
    # one stream, so a single sync closes them out together; time the K-major
    # transpose separately (it is GPU-only, no DMA).
    _t_dma_end = _stage_mark(_t_dma, "dma")
    s13t = s13_raw.transpose(1, 2).contiguous()      # [E, H/128, 2I/128]
    s2t = s2_raw.transpose(1, 2).contiguous()        # [E, I/128, H/128]

    # ⭐⭐ 2026-10-15 用户裁定:所有 GPU 预填充路径**一律** tile 形态;
    #    **旧的"按层缓冲"机制已删除** ✗ ⇒ `dst`(tile 槽)**必需** ✓
    #    ⛔ 不再有 `dst is None` 的整层兜底 —— 缺槽即**显式失败**,
    #       绝不静默退回按层持久缓冲(那正是 §601 里"永不释放的整层槽")✗
    if dst is None:
        raise RuntimeError(
            "gpu-prefill(fp8): 需要 tile 槽(dst);"
            "按层持久缓冲已按 2026-10-15 用户裁定删除")
    w13t = w13t if w13t is not None else _kmajor_bytes(w13_raw, dst[0])
    w2t = _kmajor_bytes(w2_raw, dst[1])
    _stage_mark(_t_dma_end, "tr")
    if _trace:
        _hit("done")
        torch.cuda.synchronize(device)

        def _ms(a, b):
            return _ev[a].elapsed_time(_ev[b]) if a in _ev and b in _ev else -1.0
        print(f"[fp8-asm] w13={_ms('start', 'w13'):6.1f} w2={_ms('w13', 'w2'):6.1f} "
              f"scales={_ms('w2', 'scales'):6.2f} tr={_ms('scales', 'done'):6.1f} "
              f"total={_ms('start', 'done'):6.1f} ms", flush=True)
    return w13t, s13t, w2t, s2t


def gpu_moe_layer_fp8(x, topk_ids, topk_weights, w13t, s13t, w2t, s2t,
                      H: int, I: int, K: int, *, device,
                      swiglu_limit: float = 0.0,
                      bm: int = int(os.environ.get("XIAOTU_GPF_GEMM_BM", 64)),
                      bn: int = int(os.environ.get("XIAOTU_GPF_GEMM_BN", 64)),
                      bk: int = int(os.environ.get("XIAOTU_GPF_GEMM_BK", 64)),
                      bh: int = int(os.environ.get("XIAOTU_GPF_GEMM_BH", 64)),
                      ns: int = int(os.environ.get("XIAOTU_GPF_GEMM_STAGES", 2)),
                      warps: int = int(os.environ.get("XIAOTU_GPF_GEMM_WARPS", 4))):
    """One layer's routed MoE on GPU from K-major FP8 weights.

    Returns an **fp32** tensor: accumulating the top_k partial sums into a bf16
    output (what the MXFP4 path does, since its `out` is bf16) costs ~2-4e-3 rms
    relative error; fp32 accumulation measures ~5e-7. The caller casts once.
    """
    # ⛔⛔⛔【2026-10-17 B923 修复:尺度槽可能是 **uint8 字节槽**】✗
    #   实测根因(⭐ 同缓冲双入口对照,rms_rel 806 倍 vs 4.4e-3):
    #     热专家池 `expert_pool.planes` 与 `_fp8_tile_ring` 的尺度平面都按
    #     **`uint8` + 末维 ×4** 分配(注释自述:省空间,末维 ×4 后 `.view(torch.float32)`)
    #     ⇒ 此时 `s13t.stride(0)/stride(1)` 是 **【字节】步长** ✗,
    #   而内核 `gate_up_kernel_fp8`/`down_kernel_fp8` 按 **【fp32 元素】** 寻址
    #     (`:127` `s13t_ptr + s_off + (kk>>7)*row_stride + n_blk`
    #      / `:181` 同理)
    #     ⇒ 尺度整体错位 **4×** ⇒ 解量化全错 ⇒ 输出与真值无关 ✗
    #   ⭐ 对照证据:`_moe_tile_launch_fp8` 的调用方
    #     `gpu_moe_layer_from_engine_tiled:712` **显式**传
    #     `b[1][:e1-e0].view(torch.float32)` ⇒ 它做对了 ⇒ 4.4e-3 ✓;
    #     而本入口**原样接收** `s13t` ⇒ 806 ✗
    #   ⭐ 修法与另两处**逐字同构**(`_fp8_kmajor_tiled:565-568`、
    #     `gpu_moe_layer_from_engine_tiled:712`)⇒ 三处统一 ✓
    #   ⚠️ 兼容性:浮点尺度(如 harness 的 `kmajor_from_engine_shards_fp8` 输出)
    #     走 `else` 分支,行为**逐字不变** ✓
    if s13t.dtype == torch.uint8:
        s13t = s13t.view(torch.float32)
    if s2t.dtype == torch.uint8:
        s2t = s2t.view(torch.float32)
    # ⛔ MUSA 的 Triton 没有 fp8e4nv 类型：位运算要求权重以 uint8 呈现
    if w13t.dtype == torch.float8_e4m3fn:
        w13t = w13t.view(torch.uint8)
    if w2t.dtype == torch.float8_e4m3fn:
        w2t = w2t.view(torch.uint8)
    T = x.shape[0]
    E = int(w13t.shape[0])
    tok, wts, seg_start, A = _build_segmentation(topk_ids, topk_weights, E, device)
    out = torch.zeros((T, H), dtype=torch.float32, device=device)
    if T == 0 or K == 0:
        return out
    inter = _buf("inter", (A, 2 * I), device, torch.bfloat16)
    gate_up_kernel_fp8[(E, triton.cdiv(2 * I, bn))](
        x, x.stride(0), tok, seg_start,
        w13t, w13t.stride(1), s13t, s13t.stride(1),
        inter, inter.stride(0), w13t.stride(0), s13t.stride(0),
        H=H, BM=bm, BN=bn, BK=bk, NS=ns, num_warps=warps,
    )
    down_kernel_fp8[(E, triton.cdiv(H, bh))](
        inter, inter.stride(0), tok, wts, seg_start,
        w2t, w2t.stride(1), s2t, s2t.stride(1),
        out, out.stride(0), w2t.stride(0), s2t.stride(0),
        H=H, I=I, BM=bm, BH=bh, BK=bk, NS=ns, LIMIT=float(swiglu_limit),
        num_warps=warps,
    )
    return out


# 【2026-10-16 goal-A1 · 废除"按层缓冲"】此处原有
#   `def gpu_moe_layer_fp8_from_engine(x, topk_ids, topk_weights, engine, H, I, K, ...)`
# —— 它是"装【整层】K-major 权重再算一层"的死入口(内部 `kmajor_from_engine_shards_fp8`
#    不带 dst ⇒ 每次新建整层缓冲)。**已删除** ✓
# 证据:`grep -rn gpu_moe_layer_fp8_from_engine` 在插件内**零调用者** ✓
#   (仅 build/lib/ 下的旧拷贝与自身定义)⇒ 删除无行为变化 ✓
# 保留:`kmajor_from_engine_shards_fp8` 本体 ✓ —— tile 版(见下方 `..._tiled`)
#   与 3 个 harness(test_gpu_prefill_fp8_assembly / bench_gpf_asm_phases /
#   test_gpu_prefill_fp8_vs_cpu)仍在用 ✓


# ---------------------------------------------------------------------------
# Backend facade
# ---------------------------------------------------------------------------
# `mixed_experts._gpu_prefill_mod()` picks this module or `gpu_prefill` purely by
# the engine class, so the two must expose the same names with the same
# signatures. The MXFP4-specific `group_k` is accepted and ignored here: the FP8
# block is a fixed 128x128, and the staging/compute sizes do not depend on it.


def staging_bytes(n_experts: int, hidden: int, inter: int, group_k: int = 128,
                  ns: int = 2) -> int:
    """``gpu_prefill.staging_bytes``-compatible entry point (see fp8_staging_bytes)."""
    del group_k                      # fp8 block is fixed at 128x128
    return fp8_staging_bytes(n_experts, hidden, inter, ns)


def kmajor_from_engine_shards(engine, device, hidden: int, inter: int,
                              n_experts: int, group_k: int = 128, dst=None):
    """``gpu_prefill.kmajor_from_engine_shards``-compatible entry point.

    ``group_k`` is ignored (fixed 128) and ``dst`` is unused: the assembly writes
    into the module's own persistent named buffers (see ``_buf``), which is what
    keeps the per-layer allocator churn away.
    """
    del group_k
    # `dst` MUST be forwarded: the ping/pong prefetch passes its own K-major slot
    # here, and dropping it would silently write into the shared single-slot
    # buffers that another layer's GEMM may still be reading.
    return kmajor_from_engine_shards_fp8(engine, device, hidden, inter, n_experts,
                                         dst=dst)


def gpu_moe_layer(x, topk_ids, topk_weights, w13t, s13t, w2t, s2t,
                  H: int, I: int, K: int, *, device, slot=None,
                  swiglu_limit: float = 0.0, out_dtype=None, **kw):
    """``gpu_prefill.gpu_moe_layer``-compatible entry point.

    ``slot`` is accepted for interface parity but unused -- the weights arrive
    **positionally** as the 4 tensors, and for the residency pool those 4 ARE the
    pool's planes (``mixed_experts`` unpacks ``*_pinp.planes`` into exactly these
    four slots), so there is nothing to look up here ✓

    ⭐ ``out_dtype``: MXFP4's ``gpu_moe_layer`` takes it and returns bf16. The FP8
    path **always returns fp32** (fp32 accumulation measures ~5e-7 vs ~2-4e-3 for
    bf16 accumulation) and the caller casts once, so we accept the kwarg and
    ignore it.  ⛔ 2026-10-15 修:**先前没有这个参数** —— 而 `mixed_experts` 的驻留
    分流路**总是**传 `out_dtype=torch.float32` ⇒ 经 `**kw` 转发后在此处
    **TypeError** ⇒ **FP8 热专家驻留一启用就会崩** ✗(此路此前从未被执行过)
    """
    del slot, out_dtype
    return gpu_moe_layer_fp8(x, topk_ids, topk_weights, w13t, s13t, w2t, s2t,
                             H, I, K, device=device, swiglu_limit=swiglu_limit, **kw)


# ---------------------------------------------------------------------------
# 【微 ping/pong】tile 环路径(FP8 版)
# ---------------------------------------------------------------------------
# ⭐【2026-10-17 注释更正】下面这条已**过期**,勿再引用 ✗:
#    ~~默认关闭:`XIAOTU_GPF_STAGE_TILE_E=0`(或 >= E)⇒ `stage_tiles` 只给 `[(0, E)]`,
#      走的仍是上面的**整层**装配 ⇒ 与改动前**逐字节等价**~~
#    ⇒ 事实(读码,`gpu_prefill.py:1081-1118`):2026-10-15 用户裁定去掉按层缓冲后,
#      `staging_tile_experts()` **永不返回 `E`** ✓,`stage_tiles()` 是它的**纯派生** ✓
#      ⇒ `XIAOTU_GPF_STAGE_TILE_E=0`(或 >= E)现在等价于**默认** `max(1, E//3)` ✓
#      ⇒ **tile 环是唯一形态**;该 env 只用来**调 tile 宽度**,**不能**再退回整层 ✓
# ⭐ 与 MXFP4 版(`gpu_prefill.gpu_moe_layer_from_engine_tiled`)**逐字同构**:环、事件、
#    调度顺序全部照抄,只把"权重装配"换成 FP8、把"内核"换成 fp8 内核 ✓



def _fp8_kmajor_tiled_impl(engine, device, hidden: int, inter: int, n_experts: int,
                      e0: int, e1: int, dst):
    """把专家区间 `[e0, e1)` 的 K-major FP8 权重**逐 NUMA node 直填**进 `dst`。

    ⭐ **无 raw 缓冲**(noraw):每个 node 的切片直接转置进目标的 K-major 列区间
    ⇒ 省掉整层 raw 那一份(FP8 下 raw13+raw2 与 K-major 同尺寸,是**成对**的开销)✓

    与 `gpu_prefill.kmajor_from_engine_shards_noraw` 同构;两处差异:
      * 布局是 FP8 的 `[E, H, 2I]` / `[E, I, H]`(node 切片 `[gate cbytes][up cbytes]`);
        ⛔⛔【2026-10-17 B916/B923 勘误】**原文写的是 `[E, 2I, H]` / `[E, H, I]`,是错的** ✗
        ⇒ 实测(`scripts/test_fp8_pool_planes_vs_ring.py`,E=16 H=2048 I=1024):
          按 `[E, 2I, H]`/`[E, H, I]` 分配必抛
          `ValueError: ktranspose_into shape mismatch: x(1, 256, 1024) -> y(1, 2048, 256)` ✗;
          按 `[E, H, 2I]`/`[E, I, H]` 分配 ⇒ `rms_rel=4.407e-03` ✅
        ⭐ 判据:函数体按【末维】切 N(`:126` 用 `kk*row_stride + n`,`n` 覆盖 `2I`),
          且 `ktranspose_into` 负责 K-major→目标的转置 ⇒ **N 必须在最后一维** ✓
        ⭐ 对接口径与 `_fp8_tile_ring`(`shapes=((t,H,2I),(t,H//b,(2I)//b*4),(t,I,H),(t,I//b,H//b*4))`)
          和 `expert_pool.planes` **完全一致** ✓
      * **尺度是 fp32** ⇒ 用 torch `transpose(1,2).contiguous()`,**不走** `ktranspose`
        (`ktranspose_into` 只接受 uint8;尺度量很小,整层路径也是这么做的 ✓)
    """
    # ⭐⭐⭐【B985 修】tile 路径缺少锁页⇒ DMA 跑在 pageable 上✗
    #   MXFP4 路径在 `kmajor_from_engine_shards` 开头就调用它（`:1638`）✓，
    #   而 FP8 只在**整层**入口 `kmajor_from_engine_shards_fp8`（`:300`，已废）里调✗
    #   ⇒ benchmark：pageable 15-21 GB/s vs pinned 26.85 GB/s（同文件注释实测）
    #   幂等、失败只告警、`XIAOTU_GPF_PIN=0` 可关 ✓
    _pin_engine_hostbufs(engine)
    geo = engine.shard_geometry()
    ns = int(geo["ns"])
    E, Et = int(n_experts), int(e1) - int(e0)
    if Et <= 0:
        return dst
    H, I = int(hidden), int(inter)
    b = FP8_BLOCK
    c13, cr13 = int(geo["w13_cbytes"]), int(geo["w13_crows"])
    c2, cr2 = int(geo["w2_cbytes"]), int(geo["w2_crows"])
    w13t, s13t, w2t, s2t = dst
    # ⭐ 尺度槽在环里是 **uint8 字节槽**(见 `_fp8_tile_ring` 的 ×4 说明)⇒ **由本函数**
    #    统一转 fp32,避免"转换知识"散落到每个调用点(实测:散落会立刻写出形状不匹配)✗
    if s13t.dtype == torch.uint8:
        s13t = s13t.view(torch.float32)
    if s2t.dtype == torch.uint8:
        s2t = s2t.view(torch.float32)
    # ⛔ MUSA 的 Triton 没有 fp8e4nv 类型：位运算要求权重以 uint8 呈现
    if w13t.dtype == torch.float8_e4m3fn:
        w13t = w13t.view(torch.uint8)
    if w2t.dtype == torch.float8_e4m3fn:
        w2t = w2t.view(torch.uint8)

    # 入口事件：用于量出 "入口 → 第一次 DMA 结束" 这段（台账 §157 发现它才是大头）
    _k0 = _evt_new()
    if _k0 is not None:
        _k0.record()
    # w13:每个 node 的切片 = [gate c13 B][up c13 B];K-major 目标行区间 [c0,c0+cr13) ✓
    for n in range(ns):
        import time as _kmt
        _km_a = _kmt.perf_counter()
        _ke_a = _evt_new()
        if _ke_a is not None:
            _ke_a.record()
        buf = _dma_hostbuf_range(engine, 0, n, device, int(e0) * 2 * c13, Et * 2 * c13)
        _ke_b = _evt_new()
        if _ke_b is not None:
            _ke_b.record()
            _evt_add("km13_dma", _ke_a, _ke_b)
            if n == 0 and _k0 is not None:
                _evt_add("km_setup", _k0, _ke_b)
        _PH['dma_k'] = _PH.get('dma_k', 0.0) + (_kmt.perf_counter() - _km_a)
        _km_b = _kmt.perf_counter()
        if buf is None:
            print(f"[gpf-tile] _dma_hostbuf_range(w13) returned None (node={n})", flush=True)
            return None
        c0 = n * cr13
        blk = buf.view(Et, 2, c13)
        if (not getattr(_fp8_kmajor_tiled_impl, "_shape_dbg_done", False)
                and __import__("os").environ.get("XIAOTU_GPF_SHAPE_DBG") == "1"):
            _fp8_kmajor_tiled_impl._shape_dbg_done = True
            try:
                print("[gpf-shape] I=%d H=%d Et=%d e0=%s" % (I, H, Et, e0), flush=True)
                print("[gpf-shape] geo w13_cbytes=%s w13_crows=%s w2_cbytes=%s w2_crows=%s"
                      % (geo.get("w13_cbytes"), geo.get("w13_crows"),
                         geo.get("w2_cbytes"), geo.get("w2_crows")), flush=True)
                print("[gpf-shape] w13t=%s stride=%s" % (tuple(w13t.shape),
                                                         w13t.stride()), flush=True)
                print("[gpf-shape] w2t=%s stride=%s" % (tuple(w2t.shape),
                                                        w2t.stride()), flush=True)
                print("[gpf-shape] buf=%s blk=%s c13=%s cr13=%s c0=%s"
                      % (tuple(buf.shape), tuple(blk.shape), c13, cr13, c0), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("[gpf-shape] dump failed:", exc, flush=True)
        _kt_a = _evt_new()
        if _kt_a is not None:
            _kt_a.record()
        if _w13tn():
            # N-major 环：宿主分片本来就是 `[cr13, H]` 连续 ⇒ **纯拷贝**（无转置）。
            # ⚠️ 与 W2TN **不同**：w13 的环是 `_fp8_tile_ring` 里**按本开关换轴分配**的
            #    （`(t, 2I, H)`）⇒ 这里**直接用 `w13t`，不需要再 view 换轴**。
            #    （第一次我就照 W2TN 抄了一遍 `.view(shape[0], shape[2], shape[1])`，
            #      等于把已换好的轴又换回去 ⇒ `Target sizes: [170, 192, 768]` ✗）
            # ⚠️ 行区间用**相对**下标（槽已是本 tile 的缓冲），这是 W2TN 踩过的坑 ✓
            _w13n = w13t
            _w13n[:, c0:c0 + cr13, :].copy_(blk[:, 0, :].reshape(Et, cr13, H))
            _w13n[:, I + c0:I + c0 + cr13, :].copy_(blk[:, 1, :].reshape(Et, cr13, H))
        else:
            ktranspose_into(blk[:, 0, :].reshape(Et, cr13, H), w13t[:, :, c0:c0 + cr13])
            ktranspose_into(blk[:, 1, :].reshape(Et, cr13, H),
                            w13t[:, :, I + c0:I + c0 + cr13])
        _kt_b = _evt_new()
        if _kt_b is not None:
            _kt_b.record()
            _evt_add("km13_tr", _kt_a, _kt_b)
        del buf, blk
    # w2:每个 node 的切片 = (cr2, I) 连续 ✓
    for n in range(ns):
        _w2a = __import__("time").perf_counter()
        _kw_a = _evt_new()
        if _kw_a is not None:
            _kw_a.record()
        buf = _dma_hostbuf_range(engine, 1, n, device, int(e0) * c2, Et * c2)
        _kw_b = _evt_new()
        if _kw_b is not None:
            _kw_b.record()
            _evt_add("km2_dma", _kw_a, _kw_b)
        if buf is None:
            return None
        c0 = n * cr2
        if _w2tn():
            # ⛔⛔【2026-10-24 修】行区间必须**相对**，不能再用全局专家号 `e0`：
            #   ring 槽 `w2t` **已经是本 tile 的缓冲**（只有 `Et` 行，对应专家 `[e0,e1)`）⇒
            #   从第 2 个 tile（e0=170）起 `[e0:e0+Et]` 在 170 行上切出空集，实测报
            #   `The expanded size of the tensor (0) must match the existing size (170)
            #    at non-singleton dimension 0. Target sizes: [0, 1280, 384]` ✗
            #   首 tile（e0=0）恰好成立 ⇒ 该 bug **只在多 tile 时暴露**（探针必须跑 e0≠0）✓
            #   对照：非 W2TN 分支写 `w2t[:, :, c0:c0+cr2]`，本来就没有 e0 偏移 ✓
            _w2n = w2t.view(int(w2t.shape[0]), int(w2t.shape[2]), int(w2t.shape[1]))
            _w2n[:, c0:c0 + cr2, :].copy_(buf.view(Et, cr2, I))
        else:
            ktranspose_into(buf.view(Et, cr2, I), w2t[:, :, c0:c0 + cr2])
        _PH['w2_k'] = _PH.get('w2_k', 0.0) + (__import__("time").perf_counter() - _w2a)
        del buf
    # 尺度:整块按专家维连续,不按 node 切(与整层路径同一假设)✓
    s13_per = int(geo["w13_scale_bytes"]) // E
    s2_per = int(geo["w2_scale_bytes"]) // E
    _PH['tr_k'] = _PH.get('tr_k', 0.0) + (__import__('time').perf_counter() - _km_b)
    _km_c = __import__('time').perf_counter()
    _sda = __import__("time").perf_counter()
    b13 = _dma_hostbuf_range(engine, 2, 0, device, int(e0) * s13_per, Et * s13_per)
    b2 = _dma_hostbuf_range(engine, 3, 0, device, int(e0) * s2_per, Et * s2_per)
    _PH['scd_k'] = _PH.get('scd_k', 0.0) + (__import__("time").perf_counter() - _sda)
    if b13 is None or b2 is None:
        return None
    if _w13tn():
        # N-major：尺度槽已按 `(t, 2I/128, H/128)` 分配，宿主也是 `[2I/b, H/b]` 连续 ⇒ 纯拷贝
        s13t.view(torch.float32).view(Et, (2 * I) // b, H // b).copy_(
            b13.view(torch.float32).view(Et, (2 * I) // b, H // b))
    else:
        s13t.copy_(b13.view(torch.float32).view(Et, (2 * I) // b, H // b)
               .transpose(1, 2).contiguous())
    _PH['scc_k'] = _PH.get('scc_k', 0.0) + (__import__('time').perf_counter() - _km_c)
    if _w2tn():
        # 环里的 scale 是 uint8 且末维 x4（表示 fp32 字节）⇒ 必须先 view(f32) 再 reshape
        s2t.view(torch.float32).view(Et, H // b, I // b).copy_(
            b2.view(torch.float32).view(Et, H // b, I // b))
    else:
        s2t.copy_(b2.view(torch.float32).view(Et, H // b, I // b)
                  .transpose(1, 2).contiguous())
    # ⚠️ `view` 成立的前提 = 字节数恰好等于 Et*per:整层路径对**整块**用了同一个 view,
    #    故只要整块无尾部填充就成立;若有填充,整层路径同样会错 ⇒ 不是本条新引入的风险 ✓
    del b13, b2
    # ⭐ 返回**归一化后**的元组(尺度已是 fp32 视图,与内核/对拍期望的 dtype 一致)
    #    ⛔ 曾经 `return dst` ⇒ 调用方拿到 uint8 字节槽,而函数内部用的是 fp32 视图
    #    ⇒ dtype 与布局在各调用点分叉(实测立刻写出形状不匹配)✗
    return (w13t, s13t, w2t, s2t)


@triton.jit
def _decode_flat_kernel(w_ptr, s_ptr, o_ptr, ET, ROWS, N, W_E, S_E,
                        W_ROW, S_ROW, BLK: tl.constexpr):
    """fp8 环 -> bf16（带 128x128 block scale）。**单次 launch 扁平网格**。

    ⚠️ 教训：按专家逐个 launch（512 次）只有 1.3 GB/s；扁平单次是 44.5 GB/s。
    """
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    tot = ET * ROWS * N
    mask = off < tot
    e = off // (ROWS * N)
    rem = off % (ROWS * N)
    r = rem // N
    c = rem % N
    i = tl.load(w_ptr + e.to(tl.int64) * W_E + r * W_ROW + c, mask=mask, other=0).to(tl.int32)
    s = tl.load(s_ptr + e.to(tl.int64) * S_E + (r // 128) * S_ROW + (c // 128),
                mask=mask, other=0.0)
    ex = (i >> 3) & 0xF
    m = i & 0x7
    bits = ((ex + 120) << 23) | (m << 20)
    v = tl.where(ex == 0, m.to(tl.float32) * 1.953125e-03,
                 bits.to(tl.float32, bitcast=True))
    v = tl.where(((i >> 7) & 1) == 1, -v, v) * s
    tl.store(o_ptr + off, v.to(tl.bfloat16), mask=mask)


def _decode_blk_default() -> int:
    """decode 内核的块大小（可 A/B）。

    ⭐ 为什么做成旋钮：这个内核是**纯带宽受限**（读 1.44 GB + 写 2.82 GB），
    实测只有 **44–79 GB/s**，远低于 HBM 上限 ⇒ 块大小/线程数很可能没调好。
    默认值保持 4096（与历史一致）⇒ **不开这个 env 时行为完全不变** ✓
    """
    try:
        v = int(_os.environ.get("XIAOTU_GPF_DECODE_BLK", "4096"))
    except Exception:      # noqa: BLE001
        return 4096
    return v if v >= 512 else 4096


def _decode_warps_default() -> int:
    try:
        v = int(_os.environ.get("XIAOTU_GPF_DECODE_WARPS", "8"))
    except Exception:      # noqa: BLE001
        return 8
    return v if v in (1, 2, 4, 8, 16, 32) else 8


# ── 【P2 施工中】分块「转置 + 解码」融合内核（未接线，`XIAOTU_GPF_DECODE_FUSED` 待实现）
# 目的：删掉 `ktranspose_into` 那一整趟（每层 3.02 GB 流量 ≈ 40 ms）。
# 现状两趟：DMA 落 [cr, H] → ktranspose_into 写 [H, cr]（fp8）→ decode 写 [H, cr]（bf16）
# 目标一趟：DMA 落 [cr, H] → 本内核直接写 [H, cr]（bf16）
# ⚠️ 必须与 `_ktranspose_bytes_kernel` 的**同一分块映射**，并用 `torch.equal` 逐位对拍 ✓
@triton.jit
def _decode_tr_kernel(w, s, o, A, B, w_e, s_e, w_row, s_row, o_e, o_row,
                      BM: tl.constexpr, BN: tl.constexpr):
    """读 [A, B] 的 fp8(+128×128 block scale)，**转置**写成 bf16 [B, A]。

    `A` = 行（原始布局的第一维，如 cr）；`B` = 列（如 H）。
    输出行步长 `o_row`（元素数），所以可以用它写进 bf16 大缓冲的某个切片。
    """
    e = tl.program_id(0).to(tl.int64)
    pm = tl.program_id(1)
    pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mask = (rm[:, None] < A) & (rn[None, :] < B)
    i = tl.load(w + e * w_e + rm[:, None] * w_row + rn[None, :],
                mask=mask, other=0).to(tl.int32)
    sc = tl.load(s + e * s_e + (rm[:, None] // 128) * s_row + (rn[None, :] // 128),
                 mask=mask, other=0.0)
    ex = (i >> 3) & 0xF
    m = i & 0x7
    bits = ((ex + 120) << 23) | (m << 20)
    v = tl.where(ex == 0, m.to(tl.float32) * 1.953125e-03,
                 bits.to(tl.float32, bitcast=True))
    v = tl.where(((i >> 7) & 1) == 1, -v, v) * sc
    tl.store(o + e * o_e + rn[:, None] * o_row + rm[None, :],
             tl.trans(v).to(tl.bfloat16),
             mask=(rn[:, None] < B) & (rm[None, :] < A))


def _decode_ring_fp8_to_bf16(w_u8, s_f32, out_bf, et, rows, n, blk=None):
    """把 [et, rows, n] 的 fp8 + scale 解码到 out_bf（bf16，同形状）。"""
    w_e = int(w_u8.stride(0))
    s_e = int(s_f32.stride(0))
    if blk is None:
        blk = _decode_blk_default()
    total = int(et) * int(rows) * int(n)
    grid = (triton.cdiv(total, blk),)
    _decode_flat_kernel[grid](
        w_u8, s_f32, out_bf, int(et), int(rows), int(n), w_e, s_e,
        int(w_u8.stride(1)), int(s_f32.stride(1)), BLK=blk,
        num_warps=_decode_warps_default(),
    )


# 【2026-10-25】`XIAOTU_GPF_DMA_CPU=1`：给每次 `_dma_hostbuf_range` 记 **CPU 墙钟**。
# 目的：判 `copy_hostbuf_to_device_2d`（musaMemcpy2DAsync）是**阻塞**还是**异步入队**：
#   * 若 CPU 墙钟 ≈ event 侧 asm（293 ms/层）⇒ 阻塞 ⇒ Python 线程被占住 ⇒
#     `_issue(i+1)` 根本来不及提前发 ⇒ **双槽 ping/pong 名存实亡** ⇒ 改整层预取/真异步；
#   * 若 CPU 墙钟 ≪ asm ⇒ 已异步 ⇒ asm 是**带宽**上限（~7.2 GB/s）⇒ 只能减字节。
_DBG = {}          # 一次性诊断用的进程级开关状态（DMA bench 等）
# ============================================================================
# 【事件计时 `XIAOTU_GPF_EVT=1`】—— 台账 §149 的教训：`perf_counter` 包异步调用
# 量到的是"流控排队"，不是工作。所以层内归因**必须**用 CUDA/MUSA 事件。
# 采样：只测 1/`XIAOTU_GPF_EVT_EVERY` 层（默认 8），把开销压到可忽略。
# ============================================================================
_EV = {"pend": [], "acc": {}, "n_s": 0, "bad": False, "sample": False}
# 【`XIAOTU_GPF_NOSYNC=1`】本层段边界的**主机副本**。
# 由 `_gmlet_impl` 在层开始时一次性 `.tolist()` 得到（那一次同步本来就有），
# 于是 `_bmm_chunk` 里就**不必**再为 `R`/`avg` 各做一次 `.item()`. ⭐
_SC = {"bounds": None}


def _ln_is_stat(ln: str) -> bool:
    """pstats 的数据行形如 `   ncalls  tottime  percall  cumtime  percall filename:lineno(func)`。"""
    parts = ln.split(None, 5)
    if len(parts) < 6:
        return False
    try:
        float(parts[1])
        float(parts[3])
    except Exception:      # noqa: BLE001
        return False
    return True


def _evt_on() -> bool:
    import os as _eos
    return _eos.environ.get("XIAOTU_GPF_EVT", "0") == "1"


def _evt_arm() -> None:
    """在每层入口调用：决定本层是否采样。"""
    if not _evt_on() or _EV["bad"]:
        _EV["sample"] = False
        return
    import os as _eos
    try:
        every = max(1, int(_eos.environ.get("XIAOTU_GPF_EVT_EVERY", "8") or 8))
    except Exception:      # noqa: BLE001
        every = 8
    _EV["sample"] = (_EV.get("nl", 0) % every) == 0
    _EV["nl"] = _EV.get("nl", 0) + 1


def _evt_new():
    """新建 timing 事件；不可用则置 bad 并返回 None（绝不抛）。"""
    if not _EV["sample"] or _EV["bad"]:
        return None
    try:
        import torch as _ot
        return _ot.cuda.Event(enable_timing=True)
    except Exception as _exc:      # noqa: BLE001
        _EV["bad"] = True
        print("[gpf-evt] ⚠️ event API 不可用: %r" % (_exc,), flush=True)
        return None


def _evt_add(name: str, a, b) -> None:
    if a is not None and b is not None:
        _EV["pend"].append((name, a, b))


def _evt_harvest() -> None:
    keep = []
    for name, a, b in _EV["pend"]:
        try:
            if not b.query():
                keep.append((name, a, b))
                continue
            _EV["acc"][name] = _EV["acc"].get(name, 0.0) + a.elapsed_time(b)
        except Exception as _exc:      # noqa: BLE001
            _EV["bad"] = True
            print("[gpf-evt] ⚠️ harvest 失败: %r" % (_exc,), flush=True)
            return
    _EV["pend"] = keep


def _evt_report(layer_ms: float) -> None:
    per = max(1, int(_EV["n_s"]))
    if not _EV["acc"]:
        return
    parts = " ".join("%s=%.1f" % (k, v / per) for k, v in sorted(_EV["acc"].items()))
    print("[gpf-evt] sampled=%d avg/layer: %s | layer=%.1f"
          % (per, parts, layer_ms), flush=True)
    _EV["acc"] = {}
    _EV["n_s"] = 0


_PH = {"decode": 0.0, "gather": 0.0, "bmm1": 0.0, "act": 0.0, "bmm2": 0.0,
       "scatter": 0.0, "n": 0}


# ── 逐调用调试（`XIAOTU_GPF_DMA_DEBUG=1`）：打印**前 8 次** DMA 的 host 墙钟 ──
#   判据：若单次 host ≈ (size / 4 GB/s) 且 ≫ torch 等效调用(0.01 ms) ⇒
#   说明 `musaMemcpy2DAsync` 对 `musaHostRegister` 过的内存**仍然阻塞**（注册 ≠ DMA 认可的锁页）
if os.environ.get("XIAOTU_GPF_DMA_DEBUG", "0") == "1":
    import time as _ddb_t

    _ddb_raw = _dma_hostbuf_range
    _DDB = {"n": 0}

    def _dma_hostbuf_range(*a, **kw):        # noqa: F811
        _t0 = _ddb_t.perf_counter()
        try:
            return _ddb_raw(*a, **kw)
        finally:
            _DDB["n"] += 1
            if _DDB["n"] <= 8:
                _ms = (_ddb_t.perf_counter() - _t0) * 1e3
                _sz = a[5] if len(a) > 5 else kw.get("size", -1)
                print("[gpf-dma-dbg] call#%d which=%s node=%s size=%.1fMB host=%.2fms"
                      % (_DDB["n"], a[1] if len(a) > 1 else "?",
                         a[2] if len(a) > 2 else "?",
                         (int(_sz) / 2**20) if str(_sz).lstrip("-").isdigit() else -1,
                         _ms), flush=True)


if os.environ.get("XIAOTU_GPF_DMA_CPU", "0") == "1":
    import time as _dpc_t

    _dma_raw_fn = _dma_hostbuf_range

    def _dma_hostbuf_range(*a, **kw):        # noqa: F811 —— 有意覆盖模块全局
        _dpc0 = _dpc_t.perf_counter()
        try:
            return _dma_raw_fn(*a, **kw)
        finally:
            _PH["dma_cpu"] = _PH.get("dma_cpu", 0.0) + (_dpc_t.perf_counter() - _dpc0)
            # ⭐ 同时累计**字节数**：`asm` 到底是"带宽低"还是"字节多"，只有字节能定案
            try:
                _sz = a[5] if len(a) > 5 else kw.get("size", 0)
                _PH["dma_bytes"] = _PH.get("dma_bytes", 0) + int(_sz)
                _PH["dma_calls"] = _PH.get("dma_calls", 0) + 1
            except Exception:      # noqa: BLE001
                pass


def _ph_on() -> bool:
    import os as _os
    return _os.environ.get("XIAOTU_GPF_PHASE_LOG", "1") == "1"


def _ph_add(key: str, dt: float) -> None:
    if _ph_on():
        _PH[key] += dt


def _ph_tick() -> None:
    if not _ph_on():
        return
    import time as _pt
    _PH["n"] += 1
    _now = _pt.perf_counter()
    _prev = _PH.get("wall_prev")
    if _prev is not None:
        _PH["wall_seg"] = _PH.get("wall_seg", 0.0) + (_now - _prev)
    _PH["wall_prev"] = _now
    if _PH["n"] % 8:
        return
    n = _PH["n"]
    tot = sum(_PH[k] for k in ("decode", "gather", "bmm1", "act", "bmm2", "scatter"))
    print("[gpf-ph] n=%d decode=%.1fms gather=%.1fms bmm1=%.1fms act=%.1fms "
          "bmm2=%.1fms scatter=%.1fms total=%.1fms"
          % (n, *(_PH[k] * 1000 for k in ("decode", "gather", "bmm1", "act",
                                          "bmm2", "scatter")), tot * 1000),
          flush=True)
    print("[gpf-tile2] tile_ms_per_bandgroup=%.1f" % (_PH.get("tile", 0.0) * 1000.0),
          flush=True)
    _PH["tile"] = 0.0
    print("[gpf-wall] seq=%d seg=%.0fms bands=%d => %.1f ms/band"
          % (_PH.get("seq", 0) + 1, _PH.get("wall_seg", 0.0) * 1000, 8,
             _PH.get("wall_seg", 0.0) * 1000 / 8.0), flush=True)
    _PH["wall_seg"] = 0.0
    print("[gpf-km2] w2=%.1fms scd=%.1fms scc=%.1fms"
          % (_PH.get("w2_k", 0.0) * 1000, _PH.get("scd_k", 0.0) * 1000,
             _PH.get("scc_k", 0.0) * 1000), flush=True)
    _PH["w2_k"] = _PH["scd_k"] = _PH["scc_k"] = 0.0
    # ⭐ 2026-10-26：加**层调用序号** `seq`，用于定位 A73「卡在第几层」。
    #   原打印只有时间、没有归属 ⇒ 只能看出"停住了"，看不出"停在哪一层"，
    #   而常驻层是 3/9/11/15/16/25/31/39，必须知道序号才能判断停点是否落在常驻层上。
    _PH["seq"] = _PH.get("seq", 0) + 1
    print("[gpf-km] seq=%d dma=%.1fms tr=%.1fms scale=%.1fms"
          % (_PH["seq"], _PH.get("dma_k", 0.0) * 1000, _PH.get("tr_k", 0.0) * 1000,
             _PH.get("sc_k", 0.0) * 1000), flush=True)
    _PH["dma_k"] = _PH["tr_k"] = _PH["sc_k"] = 0.0
    for k in ("decode", "gather", "bmm1", "act", "bmm2", "scatter"):
        _PH[k] = 0.0


def _moe_tile_launch_bmm(x, tok_all, wts_all, seg_start, e0, e1,
                         w13_bf, w2_bf, H, I, *, out, w_sorted, tok_sorted,
                         row0, row1, rows_abs):
    """bmm 版 tile 计算：gather -> bmm(gate_up) -> SwiGLU -> bmm(down) -> index_add。"""
    import torch as _t

    dev = x.device
    ne_all = int(e1 - e0)
    if ne_all == 0:
        return
    if not getattr(_moe_tile_launch_bmm, "_dbg", False):
        _moe_tile_launch_bmm._dbg = True
        print(f"[gpf-bmm] ne={ne_all} H={H} I={I} row0={row0} row1={row1} "
              f"x={tuple(x.shape)}", flush=True)
    # 子块：把专家再切成小块，限制 gather/bmm 的峰值内存
    CHUNK = int(__import__("os").environ.get("XIAOTU_GPF_BMM_CHUNK", "32"))
    for c0 in range(e0, e1, CHUNK):
        c1 = min(e1, c0 + CHUNK)
        _bmm_chunk(x, tok_all, wts_all, seg_start, c0, c1, w13_bf, w2_bf, H, I,
                   out=out, w_sorted=wts_all, tok_sorted=tok_all,
                   row0=row0, row1=row1, base_e=e0)


def _perexpert(x, tok_all, wts_all, seg_start, e0, e1, w13_bf, w2_bf, H, I, *,
              out, w_sorted, tok_sorted, row0, row1, base_e, seg_l):
    """逐专家 matmul（零补零）+ **单次** index_add_ 写回。

    教训：逐专家调用 `out.index_add_` 在 MUSA 上单次 ~45 ms（512 次 = 23 s/层）；
    拼接后只调一次，成本降到毫秒级。
    """
    import time as _tt

    import torch as _t

    o = int(e0 - base_e)
    dns = []
    toks = []
    wts = []
    t_bmm = 0.0
    t_act = 0.0
    lim = float(getattr(_moe_tile_launch_bmm, "limit", 0.0))
    for k, e in enumerate(range(int(e0), int(e1))):
        s0, s1 = int(seg_l[e]), int(seg_l[e + 1])
        if s1 <= s0:
            continue
        tl_ = tok_sorted[s0:s1].long()
        xg = x[tl_]
        a = _tt.perf_counter()
        gu = _t.matmul(xg, w13_bf[o + k])
        t_bmm += _tt.perf_counter() - a
        b = _tt.perf_counter()
        g, u = gu[..., :I], gu[..., I:]
        if lim > 0.0:
            g = _t.clamp(g, max=lim)
            u = _t.clamp(u, min=-lim, max=lim)
        act = (g * _t.sigmoid(g)) * u
        t_act += _tt.perf_counter() - b
        dns.append(_t.matmul(act, w2_bf[o + k]))
        toks.append(tl_)
        wts.append(w_sorted[s0:s1])
    if not dns:
        return
    _ph_add("bmm1", t_bmm)
    _ph_add("act", t_act)
    dn_all = _t.cat(dns, dim=0)
    tok_all_rows = _t.cat(toks, dim=0)
    w_all = _t.cat(wts, dim=0).to(_t.float32)
    c = _tt.perf_counter()
    out.index_add_(0, tok_all_rows, dn_all.to(_t.float32) * w_all[:, None])
    _ph_add("scatter", _tt.perf_counter() - c)
    _ph_tick()


def _pack_groups(cnt, cap):
    """把连续的专家行数列表 `cnt` 贪心切成若干组，使 `len(group) * max(cnt[group]) ≤ cap`。

    ⭐ 关键是**重专家单独成组**：否则一个 745 行的专家会把整组的 R 顶到 745，
    让同组的轻专家(均 73 行)白白多算 10 倍（台账 §170 实测放大 ~10×）。
    返回 [(start, end), ...]（相对本 chunk 的偏移，左闭右开）。
    """
    groups = []
    i = 0
    n = len(cnt)
    while i < n:
        j = i
        mx = 0
        while j < n:
            m2 = max(mx, int(cnt[j]))
            if m2 * (j - i + 1) > cap and j > i:
                break
            mx = m2
            j += 1
            # 重专家（自己一个就占掉 cap 的 1/4 以上）⇒ 立刻收组
            if mx * (j - i) * 4 > cap:
                break
        if j == i:
            j = i + 1
        groups.append((i, j))
        i = j
    return groups


def _bmm_chunk(x, tok_all, wts_all, seg_start, e0, e1, w13_bf, w2_bf, H, I, *,
               out, w_sorted, tok_sorted, row0, row1, base_e):
    import torch as _t

    import os as _os

    dev = x.device
    ne = int(e1 - e0)
    seg = seg_start[e0:e1 + 1]                       # 全局行号
    counts = (seg[1:] - seg[:-1]).to(_t.long)
    # ⭐【`XIAOTU_GPF_NOSYNC=1`】`counts.max().item()` 与 `counts.mean().item()` 是
    # **每次调用两次强制 D2H 同步**（每层 4 个 tile ⇒ 8 次），每次都把 DMA+内核的流水**排空**
    # ⇒ 台账 §150 实测：GPU 只有 ~300 ms/层，主机却 690 ms/层，差的 ~390 ms 就是这里。
    # 层开始时已经 `.tolist()` 过一次（`_SC["bounds"]`），直接用它在**主机**上算，
    # 数值与 `counts` 完全一致（同一个 seg_start 的差分），零同步 ✓
    # 2026-10-25 实测采纳：同 boot A/B 层时 665.6 → 572.9–615.4 ms、TTFT 41.88 → 38.80 s
    # ⇒ **默认开启**（数值等价：R/avg 由同一份 seg_start 差分在主机侧算出）。
    # 回滚：`XIAOTU_GPF_NOSYNC=0`
    _nosync = _os.environ.get("XIAOTU_GPF_NOSYNC", "1") == "1"
    _hb = _SC.get("bounds") if _nosync else None
    if _hb is not None and int(e1) + 1 <= len(_hb):
        _sl = _hb[int(e0):int(e1) + 1]
        _cnt = [_sl[_j + 1] - _sl[_j] for _j in range(len(_sl) - 1)]
        R = max(_cnt) if _cnt else 0
    else:
        R = int(counts.max().item()) if ne else 0
    if R == 0:
        return
    if _hb is not None and int(e1) + 1 <= len(_hb):
        _bmm_chunk._cnt = list(_cnt)
    else:
        _bmm_chunk._cnt = None
    if not getattr(_bmm_chunk, "_dbg", False):
        _bmm_chunk._dbg = True
        print(f"[gpf-bmm2] ne={ne} R={R} rows={int(counts.sum())} "
              f"cap={_os.environ.get('XIAOTU_GPF_BMM_MAXROWS', '8192')}", flush=True)
    # 峰值保护：ne*R 过大就二分（R 异常时也能跑起来而不是 OOM）
    _mode = _os.environ.get("XIAOTU_GPF_BMM_MODE", "auto").lower()
    if _hb is not None and int(e1) + 1 <= len(_hb):
        _avg = (sum(_cnt) / float(len(_cnt))) if _cnt else 0.0
    else:
        _avg = float(counts.float().mean().item()) if ne else 0.0
    if _mode == "perexpert" or (_mode == "auto" and _avg > 0 and R / _avg > 2.0):
        _perexpert(x, tok_all, wts_all, seg_start, e0, e1, w13_bf, w2_bf, H, I,
                   out=out, w_sorted=w_sorted, tok_sorted=tok_sorted,
                   row0=row0, row1=row1, base_e=base_e,
                   seg_l=_hb if _hb is not None else seg_start.tolist())
        return
    max_rows = int(_os.environ.get("XIAOTU_GPF_BMM_MAXROWS", "8192"))
    # ⭐ `XIAOTU_GPF_BMM_PACK=1`：**贪心打包**代替"对半二分"。
    #    动机（台账 §170）：`_bmm_chunk` 的代价 ≈ ne × R（R=chunk 内最大 count），
    #    真实路由 R/avg≈10 ⇒ 对半二分会让每个 chunk 都被一个"重专家"把 R 顶到 745，
    #    补零放大约 10×（每层 gather ≈2.8 GB，其中 ~90% 是废活）。
    #    贪心做法：按"加入后 ne*R 仍 ≤ cap"尽量装；**一旦遇到重专家就让它单独成组**。
    #    ⇒ 补零从 ~10× 压到 ~2×，且**不引入新内核**（纯 Python 侧分块，S5000 同样有效）✓
    if (_os.environ.get("XIAOTU_GPF_BMM_PACK") == "1"
            and ne > 1 and not getattr(_bmm_chunk, "_packing", False)):
        _pack = getattr(_bmm_chunk, "_cnt", None)
        if _pack is not None and len(_pack) == ne:
            groups = _pack_groups(_pack, max_rows)
            if len(groups) > 1:
                _bmm_chunk._packing = True
                try:
                    for _g0, _g1 in groups:
                        _bmm_chunk(x, tok_all, wts_all, seg_start, e0 + _g0, e0 + _g1,
                                   w13_bf, w2_bf, H, I,
                                   out=out, w_sorted=w_sorted, tok_sorted=tok_sorted,
                                   row0=row0, row1=row1, base_e=base_e)
                finally:
                    _bmm_chunk._packing = False
                return
    if ne * R > max_rows and ne > 1:
        mid = e0 + ne // 2
        _bmm_chunk(x, tok_all, wts_all, seg_start, e0, mid, w13_bf, w2_bf, H, I,
                   out=out, w_sorted=w_sorted, tok_sorted=tok_sorted,
                   row0=row0, row1=row1, base_e=base_e)
        _bmm_chunk(x, tok_all, wts_all, seg_start, mid, e1, w13_bf, w2_bf, H, I,
                   out=out, w_sorted=w_sorted, tok_sorted=tok_sorted,
                   row0=row0, row1=row1, base_e=base_e)
        return
    ar = _t.arange(R, device=dev)
    # `int(seg_start[e0])` 会把 0 维张量取成 Python int ⇒ **又一次强制同步**（每个 chunk 一次）
    # ⇒ NOSYNC 模式下直接用主机副本（`_SC["bounds"]`），零同步 ✓
    _base_row = _hb[int(e0)] if _hb is not None else int(seg_start[e0])
    starts = (seg[:-1] - _base_row).to(_t.long)      # 相对 tile 起点
    idx = starts[:, None] + ar[None, :]
    valid = ar[None, :] < counts[:, None]
    tok_local = tok_sorted[int(row0):int(row1)]               # 该 tile 的排序行
    flat = _t.zeros(ne * R, dtype=_t.long, device=dev)
    flat[valid.reshape(-1)] = tok_local[idx.reshape(-1)[valid.reshape(-1)]].long()
    import time as _tt
    _a = _tt.perf_counter()
    _eg_a = _evt_new()
    if _eg_a is not None:
        _eg_a.record()
    rows_g = x[flat].reshape(ne, R, H)
    _eg_b = _evt_new()
    if _eg_b is not None:
        _eg_b.record()
        _evt_add("c_gather", _eg_a, _eg_b)
    _ph_add("gather", _tt.perf_counter() - _a)
    o = int(e0 - base_e)
    _eb_a = _evt_new()
    if _eb_a is not None:
        _eb_a.record()
    _b = _tt.perf_counter()
    if _w13tn():
        # N-major 环 ⇒ TN matmul（§120 实测：比 K-major bmm 只慢 13%，但**省掉整趟转置**）
        _nb13 = int(w13_bf.shape[0])          # 缓冲里的专家数（≠ chunk 的 ne）
        gu = _t.matmul(rows_g, w13_bf.view(_nb13, 2 * I, H)[o:o + ne].transpose(1, 2))
    else:
        gu = _t.bmm(rows_g, w13_bf[o:o + ne])
    _eb_b = _evt_new()
    if _eb_b is not None:
        _eb_b.record()
        _evt_add("c_bmm1", _eb_a, _eb_b)
    _ph_add("bmm1", _tt.perf_counter() - _b)
    g, u = gu[..., :I], gu[..., I:]
    lim = float(_moe_tile_launch_bmm.limit) if hasattr(_moe_tile_launch_bmm, "limit") else 0.0
    if lim > 0.0:
        # 与 Triton 路径 `LIMIT` 完全同语义（vLLM silu_and_mul_with_clamp）
        g = _t.clamp(g, max=lim)
        u = _t.clamp(u, min=-lim, max=lim)
    _c = _tt.perf_counter()
    _ea_a = _evt_new()
    if _ea_a is not None:
        _ea_a.record()
    act = (g * _t.sigmoid(g)) * u
    _ea_b = _evt_new()
    if _ea_b is not None:
        _ea_b.record()
        _evt_add("c_act", _ea_a, _ea_b)
    _ph_add("act", _tt.perf_counter() - _c)
    _e = _tt.perf_counter()
    _e2_a = _evt_new()
    if _e2_a is not None:
        _e2_a.record()
    if _w2tn():
        _nb2 = int(w2_bf.shape[0])            # 同上：用缓冲维，不用 chunk 的 ne
        dn = _t.matmul(act, w2_bf.view(_nb2, H, I)[o:o + ne].transpose(1, 2))
    else:
        dn = _t.bmm(act, w2_bf[o:o + ne])
    _e2_b = _evt_new()
    if _e2_b is not None:
        _e2_b.record()
        _evt_add("c_bmm2", _e2_a, _e2_b)
    _ph_add("bmm2", _tt.perf_counter() - _e)
    wv = _t.zeros(ne * R, dtype=_t.float32, device=dev)
    wv[valid.reshape(-1)] = w_sorted[int(row0):int(row1)][
        idx.reshape(-1)[valid.reshape(-1)]]
    _f = _tt.perf_counter()
    _es_a = _evt_new()
    if _es_a is not None:
        _es_a.record()
    out.index_add_(0, flat, dn.reshape(ne * R, H).to(_t.float32) * wv[:, None])
    _es_b = _evt_new()
    if _es_b is not None:
        _es_b.record()
        _evt_add("c_scatter", _es_a, _es_b)
    _ph_add("scatter", _tt.perf_counter() - _f)
    _ph_tick()


_BMM_BUF: dict = {}

# ── 段边界流水化（`XIAOTU_GPF_BOUNDS_PIPE=1`）────────────────────────────────
# 痛点：`seg_start.tolist()` 每层 1 次 = 151 ms，且它**同步整条主流**，
#       连刚发出去的 tile-0 DMA 也要等。
# 做法：把 seg_start 的 D2H 拷到**侧流**上的锁页缓冲，等这个拷贝时主流的 DMA 照常跑；
#       拿到 host 值后 `.tolist()` 是纯主机操作（零设备同步）。
_PIN_BND: dict = {}
_D2H_STREAM: dict = {}


def _bnd_pin_like(seg_start):
    """按 (长度, dtype) 缓存锁页缓冲；必须保持引用，否则悬空。"""
    key = (int(seg_start.numel()), str(seg_start.dtype))
    b = _PIN_BND.get(key)
    if b is None:
        b = _t.empty(int(seg_start.numel()), dtype=seg_start.dtype, pin_memory=True)
        _PIN_BND[key] = b
    return b


def _d2h_stream(device):
    key = str(device)
    st = _D2H_STREAM.get(key)
    if st is None:
        st = _t.cuda.Stream(device=device)
        _D2H_STREAM[key] = st
    return st



def _w2tn() -> bool:
    """XIAOTU_GPF_W2TN=1：w2 环存 N-major（[E,H,I]），用 TN matmul 代替 1280x384 转置。"""
    import os as _os
    if _os.environ.get("XIAOTU_GPF_W2TN", "0") != "1":
        return False
    # ⛔ 只对 bmm 后端成立：Triton 的 `down_kernel_fp8` 直接读 w2 环并按 `stride(1)` 定 K/N，
    #    N-major 环会让它读到转置的数据（静默错，不报错）⇒ 显式拦掉，宁可不省。
    if not _use_bmm():
        return False
    return True


def _w13tn() -> bool:
    """XIAOTU_GPF_W13TN=1：w13 环存 N-major（[E,2I,H]），用 TN matmul 代替每 band 的转置。

    与 `_w2tn()` 同一模式；量级更大（`[gpf-km] tr` 实测 140–173 ms/8band）。
    ⚠️ 同时**必须**把环槽形状换轴（`_fp8_tile_ring` 里按本开关分配），否则 `view` 只是
    把同一块内存换个名字、数据仍是 K-major ⇒ 数值静默错 ✗
    """
    import os as _os
    if _os.environ.get("XIAOTU_GPF_W13TN", "0") != "1":
        return False
    if not _use_bmm():
        return False
    return True


def _use_bmm() -> bool:
    import os as _os
    return _os.environ.get("XIAOTU_GPF_GEMM", "bmm").lower() == "bmm"


def _bmm_bufs(ne: int, H: int, I: int, device):
    key = (int(ne), int(H), int(I), str(device))
    buf = _BMM_BUF.get(key)
    if buf is None:
        import torch as _t
        buf = (
            _t.empty(int(ne), int(H), 2 * int(I), dtype=_t.bfloat16, device=device),
            _t.empty(int(ne), int(I), int(H), dtype=_t.bfloat16, device=device),
        )
        _BMM_BUF[key] = buf
    return buf


def _moe_tile_launch_fp8(x, tok, wts, seg_local, w13_t, s13_t, w2_t, s2_t,
                         inter_view, out, H, I, *, BM, BN, BK, BH, NS, NW,
                         LIMIT: float = 0.0):
    """按【专家 tile】发一次 FP8 gate_up + down(内核**零改动**,只传视图)。

    与 `gpu_prefill._moe_tile_launch` 同一条理由:`pid_e` 只经 `seg_ptr` 与专家 stride
    (`W*_E`)寻址,而 `inter/tok/wts` 全按**全局排序行号**索引 ⇒ 按专家切分后一个 tile
    恰好拥有连续区间 `[seg_start[e0], seg_start[e1])` ⇒ 切好视图 + 平移 seg 即可;
    `out` 在 `down_kernel_fp8` 里是 `tl.atomic_add` ⇒ 多次 launch 自然累加 ✓
    """
    if not globals().get("_XTU_DEV_PRINTED"):
        globals()["_XTU_DEV_PRINTED"] = True
        print(
            "[gpf-dev] x=%s tok=%s wts=%s seg=%s w13_t=%s s13_t=%s w2_t=%s "
            "s2_t=%s inter=%s out=%s" % (
                x.device, tok.device, wts.device, seg_local.device,
                w13_t.device, s13_t.device, w2_t.device, s2_t.device,
                inter_view.device, out.device),
            flush=True,
        )
    E_t = int(w13_t.shape[0])
    W13_E, S13_E = w13_t.stride(0), s13_t.stride(0)
    W2_E, S2_E = w2_t.stride(0), s2_t.stride(0)
    gate_up_kernel_fp8[(E_t, triton.cdiv(2 * I, BN))](
        x, x.stride(0), tok, seg_local,
        w13_t, w13_t.stride(1), s13_t, s13_t.stride(1),
        inter_view, inter_view.stride(0), W13_E, S13_E,
        H=H, BM=BM, BN=BN, BK=BK, NS=NS, num_warps=NW,
    )
    down_kernel_fp8[(E_t, triton.cdiv(H, BH))](
        inter_view, inter_view.stride(0), tok, wts, seg_local,
        w2_t, w2_t.stride(1), s2_t, s2_t.stride(1),
        out, out.stride(0), W2_E, S2_E,
        H=H, I=I, BM=BM, BH=BH, BK=BK, NS=NS, LIMIT=LIMIT, num_warps=NW,
    )


def _gmlet_impl(engine, x, topk_ids, topk_weights,
                                    H: int, I: int, E: int, K: int, *, device,
                                    group_k: int = 128, nslots: int = 2, out=None,
                                    swiglu_limit: float = 0.0):
    """【微 ping/pong】FP8 孪生:`gpu_prefill.gpu_moe_layer_from_engine_tiled`。

    调度逐字同构:`issue(t)` 在侧流上 wait(该槽上一个 free) → `_fp8_kmajor_tiled`
    → record ready;`compute(t)` 在主流 wait(ready) → `_moe_tile_launch_fp8`(共享
    `out`,原子累加) → record free ✓
    `tile_e<=0/>=E` ⇒ 只有 1 个 tile,等价于整层一次算完(便于先做数值对拍)✓
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    T, H, I, E = int(x.shape[0]), int(H), int(I), int(E)
    del group_k                      # FP8 的块固定 128x128
    tiles = stage_tiles(E)
    if not tiles:
        return None
    tile = staging_tile_experts(E)
    # ⭐ B981：tile 环深度可调（原仅函数参数默认 2，没有 env）✗
    #   每个槽 = 一个 tile 的 w13+s13+w2+s2；tile=96 时约 1.125 GiB✓
    #   更深 = 给 H2D 更多提前量（与 MXFP4 路径的 XIAOTU_MOE_PREFETCH_SLOTS 同意）✓
    _ns_env = os.environ.get("XIAOTU_GPF_STAGE_NSLOTS")
    if _ns_env:
        nslots = max(2, int(_ns_env))
    nslots = max(2, int(nslots))
    ring = _fp8_tile_ring(device, E, H, I, tile, nslots)
    free = [None] * nslots
    ready = [None] * nslots

    tok_all, wts_all, seg_start, A = _build_segmentation(topk_ids, topk_weights, E, device)
    if out is None:
        out = torch.zeros((T, H), dtype=torch.float32, device=device)
    if T == 0 or int(K) == 0:
        return out
    inter = _buf("inter", (A, 2 * I), device, torch.bfloat16)

    BM = int(os.environ.get("XIAOTU_GPF_GEMM_BM", 64))
    BN = int(os.environ.get("XIAOTU_GPF_GEMM_BN", 64))
    BK = int(os.environ.get("XIAOTU_GPF_GEMM_BK", 64))
    BH = int(os.environ.get("XIAOTU_GPF_GEMM_BH", 64))
    NS = int(os.environ.get("XIAOTU_GPF_GEMM_STAGES", 2))
    NW = int(os.environ.get("XIAOTU_GPF_GEMM_WARPS", 4))

    cur = torch.cuda.current_stream(device)
    side = _prefetch_stream(device)

    # ========================================================================
    # 【2026-10-25 `XIAOTU_GPF_CPT=1`】把"发 DMA"搬到**独立 producer 线程**。
    #
    # 为什么（台账 §147，三种独立测量交叉验证）：
    #   * DMA **本身是异步的**（逐调用打印 host=0.05–0.4 ms）；
    #   * 但平均 **21 ms/次**（24 次/层 = 500 ms）⇒ **异步队列填满后发起调用被节流**；
    #   * 于是主线程在 `_issue()` 里被卡住 ⇒ `compute(i)` 无法与 `_issue(i+1)` 并行
    #     ⇒ 层时 = DMA + 计算**相加**（293 + 352 ≈ 645 ms，实测吻合）。
    # ⇒ 把"发指令"挪到别的线程，阻塞就发生在那个线程里 ⇒ 主线程可以与计算重叠。
    # 预期：层时 → max(293, 352) ≈ 352 ms（−45%），TTFT 42 s → ~24 s。
    # 回滚：`XIAOTU_GPF_CPT=0`（默认）⇒ 与改动前逐字等价 ✓
    # ========================================================================
    _cpt_on = os.environ.get("XIAOTU_GPF_CPT") == "1"
    _cpt = {"th": None, "jobs": None, "hs": {}, "done": {}, "exc": None}

    if _cpt_on:
        import queue as _cpt_q
        import threading as _cpt_th

        _cpt["jobs"] = _cpt_q.Queue()

        def _cpt_producer():
            while True:
                job = _cpt["jobs"].get()
                if job is None:
                    return
                i, e0, e1, s, fev = job
                try:
                    if fev is not None:
                        side.wait_event(fev)
                    # ⛔ 必须在**本线程**再进一次 `inference_mode()`：
                    #   vLLM 的 forward 在主线程的 `torch.inference_mode()` 里跑，
                    #   而"inference tensor 的 inplace 保护"是**线程局部**的 ⇒
                    #   本线程若不在 inference_mode 里，写环缓冲会抛
                    #   `Inplace update to inference tensor outside InferenceMode is not allowed`（已踩）
                    with torch.inference_mode():
                        with torch.cuda.stream(side):
                            km = _fp8_kmajor_tiled(engine, device, H, I, E, e0, e1,
                                                   tuple(b[:e1 - e0] for b in ring[s]))
                            ev = torch.cuda.Event()
                            ev.record(side)
                    _cpt["done"][i] = (km, ev)
                except BaseException as _exc:      # noqa: BLE001
                    _cpt["exc"] = _exc
                    _cpt["done"][i] = (None, None)
                finally:
                    _hs = _cpt["hs"].get(i)
                    if _hs is not None:
                        _hs.set()

        _cpt["th"] = _cpt_th.Thread(target=_cpt_producer, name="gpf-cpt", daemon=True)
        _cpt["th"].start()

    def _cpt_post(i: int) -> None:
        import threading as _cpt_th
        _cpt["hs"][i] = _cpt_th.Event()
        _cpt["jobs"].put((i, tiles[i][0], tiles[i][1], i % nslots, free[i % nslots]))

    def _cpt_wait(i: int) -> bool:
        """等 producer 把第 i 个 tile 排好；成功则设置 `ready[slot]`。"""
        _hsj = _cpt["hs"].get(i)
        if _hsj is None:
            return False
        _hsj.wait()
        km, ev = _cpt["done"].pop(i, (None, None))
        if _cpt["exc"] is not None:
            print("[gpf-cpt] producer 异常: %r" % (_cpt["exc"],), flush=True)
            return False
        if km is None or ev is None:
            return False
        ready[i % nslots] = ev
        return True

    def _cpt_close() -> None:
        if not _cpt_on or _cpt["th"] is None:
            return
        try:
            _cpt["jobs"].put(None)
            _cpt["th"].join(timeout=30)
        except Exception:      # noqa: BLE001
            pass
        _cpt["th"] = None

    # 【2026-10-25 `XIAOTU_GPF_DMA_BENCH=1`】在**引擎进程内**单独跑一遍整层 DMA。
    # 目的：把"引擎的 host 缓冲/目的地是否天生慢"与"运行期被计算积压节流"彻底分开 ——
    #   前者 ⇒ bench 里也慢（要查缓冲/NUMA/内存类型）；
    #   后者 ⇒ bench 里很快（~90 ms），说明 500 ms 是**运行期队列节流**，要改的是调度而不是拷贝。
    # 只跑一次（每个进程），且不改变后续正常流程。
    if os.environ.get("XIAOTU_GPF_DMA_BENCH") == "1" and not _DBG.get("dma_bench_done"):
        _DBG["dma_bench_done"] = True
        import time as _bt
        _tot_mb = 0.0
        for _rep in range(2):
            _t0 = _bt.perf_counter()
            with torch.cuda.stream(side):
                for (_e0, _e1) in tiles:
                    _fp8_kmajor_tiled(engine, device, H, I, E, _e0, _e1,
                                      tuple(b[:_e1 - _e0] for b in ring[0]))
            side.synchronize()
            _dtm = (_bt.perf_counter() - _t0) * 1e3
            _mb = sum((b[:_e1 - _e0].numel() for (_e0, _e1) in tiles for b in ring[0])) / 2.0 ** 20
            _tot_mb = _mb
            print("[gpf-dmabench] rep=%d tiles=%d bytes=%.0fMB wall=%.1f ms => %.2f GB/s"
                  % (_rep, len(tiles), _mb, _dtm, _mb / _dtm if _dtm > 0 else 0), flush=True)

    def _issue(i: int) -> bool:
        s = i % nslots
        e0, e1 = tiles[i]
        if _cpt_on:
            _cpt_post(i)
            return True
        if free[s] is not None:
            side.wait_event(free[s])          # 该槽上次的使用者(主流内核)必须先完成
        # 事件起点记在 **wait 之后** ⇒ dma 段只含"拷贝+组装"，不含等槽（否则会把它读成 DMA 时间）
        _da = _evt_new()
        if _da is not None:
            _da.record(side)
        with torch.cuda.stream(side):
            km = _fp8_kmajor_tiled(engine, device, H, I, E, e0, e1,
                                   tuple(b[:e1 - e0] for b in ring[s]))
            ev = torch.cuda.Event()
            ev.record(side)
        _db = _evt_new()
        if _db is not None:
            _db.record(side)
            _evt_add("dma", _da, _db)
        if km is None:
            return False
        ready[s] = ev
        return True

    # ⭐ 先把 seg_start 异步拷到侧流的锁页缓冲（在 tile-0 DMA **之前**入队，
    #    但它在**另一条流**上 ⇒ 后面等它时不会连带等主流的 DMA）
    _pipe_bnd = os.environ.get("XIAOTU_GPF_BOUNDS_PIPE") == "1"
    _ev_bnd = None
    _pin_bnd = None
    if _pipe_bnd:
        try:
            _pin_bnd = _bnd_pin_like(seg_start)
            _side_d = _d2h_stream(device)
            _side_d.wait_stream(_t.cuda.current_stream(device))
            with _t.cuda.stream(_side_d):
                _pin_bnd.copy_(seg_start, non_blocking=True)
                _ev_bnd = _t.cuda.Event()
                _ev_bnd.record(_side_d)
        except Exception as _exc:      # noqa: BLE001
            print("[gpf-bnd] 侧流 D2H 失败，回退同步路径: %r" % (_exc,), flush=True)
            _pipe_bnd = False
            _pin_bnd = None

    if not _issue(0):
        print("[gpf-tile] _issue(0) failed -> give up (layer)", flush=True)
        _cpt_close()
        return None
    # ⭐ 每层**只做一次**设备→主机同步取段边界(逐 tile `.item()` 会把流水反复打断)✓
    # ⚠️ `XIAOTU_GPF_BOUNDS_REUSE=1` 是**测量专用**：复用上一层的 bounds，跳过 tolist 同步。
    #    ⛔ 结果**数值错误**（路由逐层不同），只用来量"这 151 ms 值不值"的上限。
    #    估完必须关掉；生产路径永远是 `else` 分支。
    if os.environ.get("XIAOTU_GPF_BOUNDS_REUSE") == "1" and _SC.get("bounds") is not None:
        _bounds = _SC["bounds"]
        _DBG["bounds_reused"] = _DBG.get("bounds_reused", 0) + 1
    elif _pipe_bnd and _pin_bnd is not None and _ev_bnd is not None:
        # 只等侧流上的那个 2 KB 拷贝（主流的 tile-0 DMA 不受牵连）
        _ev_bnd.synchronize()
        _bounds = _pin_bnd.tolist()          # 纯主机，零设备同步
        _SC["bounds"] = _bounds
    else:
        _bounds = seg_start.tolist()
        _SC["bounds"] = _bounds      # 供 _bmm_chunk 零同步使用

    for i, (e0, e1) in enumerate(tiles):
        s = i % nslots
        if _cpt_on and not _cpt_wait(i):      # 等 producer 线程把该 tile 的 DMA 排好
            _cpt_close()
            return None
        if ready[s] is None:                  # 该槽 DMA 没排上(引擎无分片)⇒ 整体放弃
            print(f"[gpf-tile] slot {s} DMA not issued -> give up (layer)", flush=True)
            _cpt_close()
            return None
        cur.wait_event(ready[s])
        row0, row1 = int(_bounds[e0]), int(_bounds[e1])
        if row1 > row0:
            seg_local = (seg_start[e0:e1 + 1] - row0).contiguous()
            b = ring[s]
            if _use_bmm():
                import time as _dt
                _d0 = _dt.perf_counter()
                ne = int(e1 - e0)
                w13_bf, w2_bf = _bmm_bufs(ne, H, I, device)
                if _w13tn():
                    # N-major：环是 `[ne, 2I, H]`，尺度是 `[ne, 2I/128, H/128]`；
                    # 解到同一块 bf16 缓冲的 `view(ne, 2I, H)`（GEMM 侧再 transpose）✓
                    _dec1 = b[1][:ne].view(torch.float32).view(ne, (2 * I) // 128, H // 128)
                    _decode_ring_fp8_to_bf16(b[0][:ne].view(ne, 2 * I, H), _dec1,
                                             w13_bf.view(ne, 2 * I, H), ne, 2 * I, H)
                else:
                    _decode_ring_fp8_to_bf16(b[0][:ne], b[1][:ne].view(torch.float32),
                                             w13_bf, ne, H, 2 * I)
                if _w2tn():
                    _dec2 = b[3][:ne].view(torch.float32).view(ne, H // 128, I // 128)
                    _decode_ring_fp8_to_bf16(b[2][:ne].view(ne, H, I), _dec2,
                                             w2_bf.view(ne, H, I), ne, H, I)
                else:
                    _decode_ring_fp8_to_bf16(b[2][:ne], b[3][:ne].view(torch.float32),
                                             w2_bf, ne, I, H)
                _ph_add("decode", _dt.perf_counter() - _d0)
                _moe_tile_launch_bmm.limit = float(swiglu_limit)
                _moe_tile_launch_bmm(
                    x, tok_all, wts_all, seg_start, e0, e1, w13_bf, w2_bf, H, I,
                    out=out, w_sorted=wts_all, tok_sorted=tok_all,
                    row0=row0, row1=row1, rows_abs=None,
                )
            else:
                _moe_tile_launch_fp8(
                    x, tok_all[row0:row1], wts_all[row0:row1], seg_local,
                    b[0][:e1 - e0], b[1][:e1 - e0].view(torch.float32),
                    b[2][:e1 - e0], b[3][:e1 - e0].view(torch.float32),
                    inter[row0:row1], out, H, I,
                    BM=BM, BN=BN, BK=BK, BH=BH, NS=NS, NW=NW,
                    LIMIT=float(swiglu_limit),
                )
        fe = torch.cuda.Event()
        fe.record(cur)
        free[s] = fe
        ready[s] = None
        if i + 1 < len(tiles):
            if not _issue(i + 1):
                _cpt_close()
                return None
    _cpt_close()
    return out


_tile_timer_installed = True


def _fp8_kmajor_tiled(*a, **kw):
    """计时包装：DMA + 转置（band 之外的主要嫌疑）。"""
    import time as _tw

    _t0 = _tw.perf_counter()
    try:
        return _fp8_kmajor_tiled_impl(*a, **kw)
    finally:
        _PH["tile"] = _PH.get("tile", 0.0) + (_tw.perf_counter() - _t0)


_layer_wrap_installed = True


def gpu_moe_layer_from_engine_tiled(*a, **kw):
    """每层墙钟 + 该层 band 数的计时包装（不改逻辑）。"""
    import os as _os
    import time as _t

    on = _os.environ.get("XIAOTU_GPF_PHASE_LOG", "1") == "1"
    # 【`XIAOTU_GPF_PROF=1`】对**第 3 层**做一次 cProfile，直接点名主机时间花在哪
    # （不再靠猜：本轮已试过 tile/chunk/环深/线程/锁页，只有"去同步"有效 ⇒ 需要真凭据）
    if _os.environ.get("XIAOTU_GPF_PROF", "0") == "1" and not _DBG.get("prof_done"):
        _DBG["prof_n"] = _DBG.get("prof_n", 0) + 1
        if _DBG["prof_n"] >= 3:
            _DBG["prof_done"] = True
            import cProfile as _cp
            import io as _io
            import pstats as _ps
            _pr = _cp.Profile()
            _pr.enable()
            try:
                return _gmlet_impl(*a, **kw)
            finally:
                _pr.disable()
                _sb = _io.StringIO()
                _ps.Stats(_pr, stream=_sb).sort_stats("tottime").print_stats(14)
                _txt = _sb.getvalue()
                _keep = [ln for ln in _txt.splitlines()
                         if ln.strip() and ("cumtime" in ln or "tottime" in ln
                                            or _ln_is_stat(ln))]
                print("[gpf-prof] 层内主机热点（tottime 前 14）：\n" + "\n".join(_keep[:24]),
                      flush=True)
    n0 = _PH.get("n", 0)
    _evt_arm()                       # 决定本层是否采样（1/EVT_EVERY）
    _el0 = _evt_new()
    if _el0 is not None:
        _el0.record()
    t0 = _t.perf_counter()
    try:
        return _gmlet_impl(*a, **kw)
    finally:
        _el1 = _evt_new()
        if _el1 is not None:
            _el1.record()
            _evt_add("layer", _el0, _el1)
            _EV["n_s"] = _EV.get("n_s", 0) + 1
        if on:
            dt = (_t.perf_counter() - t0) * 1000.0
            bands = _PH.get("n", 0) - n0
            _PH["lw_n"] = _PH.get("lw_n", 0) + 1
            _PH["lw_ms"] = _PH.get("lw_ms", 0.0) + dt
            _PH["lw_bands"] = _PH.get("lw_bands", 0) + bands
            if _PH["lw_n"] % 8 == 0:
                _mb = _PH.pop("dma_bytes", 0) / 2.0 ** 20 / 8.0
                _nc = _PH.pop("dma_calls", 0) / 8.0
                _dcpu = _PH.pop("dma_cpu", 0.0) * 1000.0 / 8.0
                print("[gpf-layer] n=%d avg_ms=%.1f bands/layer=%.1f dma_cpu=%.1f "
                      "dma_mb=%.0f dma_calls=%.1f"
                      % (_PH["lw_n"], _PH["lw_ms"] / 8.0,
                         _PH["lw_bands"] / 8.0, _dcpu, _mb, _nc), flush=True)
                if _evt_on() and not _EV["bad"]:
                    _evt_harvest()
                    _evt_report(_PH["lw_ms"] / 8.0)
                _PH["lw_ms"] = 0.0
                _PH["lw_bands"] = 0
