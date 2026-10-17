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

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _e4m3_uint8_to_f32,
)

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



def _fp8_kmajor_tiled(engine, device, hidden: int, inter: int, n_experts: int,
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

    # w13:每个 node 的切片 = [gate c13 B][up c13 B];K-major 目标行区间 [c0,c0+cr13) ✓
    for n in range(ns):
        buf = _dma_hostbuf_range(engine, 0, n, device, int(e0) * 2 * c13, Et * 2 * c13)
        if buf is None:
            return None
        c0 = n * cr13
        blk = buf.view(Et, 2, c13)
        ktranspose_into(blk[:, 0, :].reshape(Et, cr13, H), w13t[:, :, c0:c0 + cr13])
        ktranspose_into(blk[:, 1, :].reshape(Et, cr13, H),
                        w13t[:, :, I + c0:I + c0 + cr13])
        del buf, blk
    # w2:每个 node 的切片 = (cr2, I) 连续 ✓
    for n in range(ns):
        buf = _dma_hostbuf_range(engine, 1, n, device, int(e0) * c2, Et * c2)
        if buf is None:
            return None
        c0 = n * cr2
        ktranspose_into(buf.view(Et, cr2, I), w2t[:, :, c0:c0 + cr2])
        del buf
    # 尺度:整块按专家维连续,不按 node 切(与整层路径同一假设)✓
    s13_per = int(geo["w13_scale_bytes"]) // E
    s2_per = int(geo["w2_scale_bytes"]) // E
    b13 = _dma_hostbuf_range(engine, 2, 0, device, int(e0) * s13_per, Et * s13_per)
    b2 = _dma_hostbuf_range(engine, 3, 0, device, int(e0) * s2_per, Et * s2_per)
    if b13 is None or b2 is None:
        return None
    s13t.copy_(b13.view(torch.float32).view(Et, (2 * I) // b, H // b)
               .transpose(1, 2).contiguous())
    s2t.copy_(b2.view(torch.float32).view(Et, H // b, I // b)
              .transpose(1, 2).contiguous())
    # ⚠️ `view` 成立的前提 = 字节数恰好等于 Et*per:整层路径对**整块**用了同一个 view,
    #    故只要整块无尾部填充就成立;若有填充,整层路径同样会错 ⇒ 不是本条新引入的风险 ✓
    del b13, b2
    # ⭐ 返回**归一化后**的元组(尺度已是 fp32 视图,与内核/对拍期望的 dtype 一致)
    #    ⛔ 曾经 `return dst` ⇒ 调用方拿到 uint8 字节槽,而函数内部用的是 fp32 视图
    #    ⇒ dtype 与布局在各调用点分叉(实测立刻写出形状不匹配)✗
    return (w13t, s13t, w2t, s2t)


def _moe_tile_launch_fp8(x, tok, wts, seg_local, w13_t, s13_t, w2_t, s2_t,
                         inter_view, out, H, I, *, BM, BN, BK, BH, NS, NW,
                         LIMIT: float = 0.0):
    """按【专家 tile】发一次 FP8 gate_up + down(内核**零改动**,只传视图)。

    与 `gpu_prefill._moe_tile_launch` 同一条理由:`pid_e` 只经 `seg_ptr` 与专家 stride
    (`W*_E`)寻址,而 `inter/tok/wts` 全按**全局排序行号**索引 ⇒ 按专家切分后一个 tile
    恰好拥有连续区间 `[seg_start[e0], seg_start[e1])` ⇒ 切好视图 + 平移 seg 即可;
    `out` 在 `down_kernel_fp8` 里是 `tl.atomic_add` ⇒ 多次 launch 自然累加 ✓
    """
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


def gpu_moe_layer_from_engine_tiled(engine, x, topk_ids, topk_weights,
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

    def _issue(i: int) -> bool:
        s = i % nslots
        e0, e1 = tiles[i]
        if free[s] is not None:
            side.wait_event(free[s])          # 该槽上次的使用者(主流内核)必须先完成
        with torch.cuda.stream(side):
            km = _fp8_kmajor_tiled(engine, device, H, I, E, e0, e1,
                                   tuple(b[:e1 - e0] for b in ring[s]))
            ev = torch.cuda.Event()
            ev.record(side)
        if km is None:
            return False
        ready[s] = ev
        return True

    if not _issue(0):
        return None
    # ⭐ 每层**只做一次**设备→主机同步取段边界(逐 tile `.item()` 会把流水反复打断)✓
    _bounds = seg_start.tolist()

    for i, (e0, e1) in enumerate(tiles):
        s = i % nslots
        if ready[s] is None:                  # 该槽 DMA 没排上(引擎无分片)⇒ 整体放弃
            return None
        cur.wait_event(ready[s])
        row0, row1 = int(_bounds[e0]), int(_bounds[e1])
        if row1 > row0:
            seg_local = (seg_start[e0:e1 + 1] - row0).contiguous()
            b = ring[s]
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
                return None
    return out
