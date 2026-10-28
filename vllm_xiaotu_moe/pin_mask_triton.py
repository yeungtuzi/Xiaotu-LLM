"""PIN 路径的掩码/分流：把每层 ~19 个 torch 小算子压成 1 个 Triton 内核。

背景（`dev-docs/AUDIT_CPU_DECODE_LATENCY_2026-10-28.md` §11）
------------------------------------------------------------
有池（`XIAOTU_PIN=1`）路径每层 decode 要发 ~31 个 kernel，其中 19 个是纯元素级/索引：
`clamp_min` → `slotmap[]` → `<0` → `>=n_local` → `~` → `&` → `where×3` → `sum` → `add_×3`。
它们的数据量只有 `M*K ≤ 288` 个元素，却要 19 次 launch（eager 下 3–5 µs/次）。

本模块把它们合成 1 个内核，**语义逐位对齐**下面的 torch 参考实现：

    _ids_safe = ids.clamp_min(0)
    _sl       = slotmap[_ids_safe.long()]
    _bad      = ids < 0
    _res      = (_sl >= n_local) & (~_bad)
    cpu_ids   = where(_res | _bad, 0, ids)
    cpu_w     = where(_res, 0, wts)
    part_ids  = where(_res, (_sl - n_local).to(int32), -1)

注意三点（都是从现有代码里抄下来的**硬约束**，改动前先读它们）：
1. `slotmap` 是**本地** id 空间，值域 `[n_local, n_local+S)`；比较基准必须是
   `n_local`，不是全局专家数（`mixed_experts.py:2938` 的 B889 修复）。
2. `ids < 0` 是 profile/warmup 期的哨兵 ⇒ 必须并入「非常驻」，且 CPU 侧收到 0 而不是 -1
   （否则引擎按 uint32 解释成 0xFFFFFFFF 越界，`mixed_experts.py:2952`）。
3. `cpu_w` 对无效 id 也必须是 0（`_zw`），与旧实现一致。

Triton 不可用或张量过大时自动退回 torch 实现（本模块只做加速，不改变语义）。
"""

from __future__ import annotations

import os

import torch

try:  # Triton 在 vLLM 环境里总是存在；缺席时静默退回 torch
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None
    tl = None
    _HAS_TRITON = False

# 默认只在显式开启时启用（第一阶段：A/B 用），保持生产行为不变。
_ENV_ON = "XIAOTU_PIN_MASK_TRITON"
_MAX_ELEMS = 8192          # 超过则退回 torch（prefill 的大 M 不走这里）


if _HAS_TRITON:

    @triton.jit
    def _pin_mask_kernel(ids_ptr, wts_ptr, sm_ptr,
                         cpu_ids_ptr, cpu_w_ptr, part_ptr, hits_ptr,
                         n_elem, n_local, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        m = off < n_elem
        ids = tl.load(ids_ptr + off, mask=m, other=-1)
        w = tl.load(wts_ptr + off, mask=m, other=0.0)
        bad = ids < 0
        safe = tl.where(bad, 0, ids)
        slot = tl.load(sm_ptr + safe, mask=m, other=0)
        res = (slot >= n_local) & (ids >= 0)
        cpu_ids = tl.where(res | bad, 0, ids)
        cpu_w = tl.where(res, 0.0, w)
        part = tl.where(res, slot - n_local, -1)
        tl.store(cpu_ids_ptr + off, cpu_ids, mask=m)
        tl.store(cpu_w_ptr + off, cpu_w, mask=m)
        tl.store(part_ptr + off, part, mask=m)
        if hits_ptr is not None:
            tl.atomic_add(hits_ptr, tl.sum(res.to(tl.int64), axis=0))


def torch_pin_mask(ids: torch.Tensor, wts: torch.Tensor, slotmap: torch.Tensor,
                   n_local: int):
    """参考实现：与 `mixed_experts.py` 现有 19 个算子的语义逐位一致。"""
    ids_i32 = ids.to(torch.int32)
    wts_f32 = wts.to(torch.float32)
    ids_safe = ids_i32.clamp_min(0)
    sl = slotmap[ids_safe.long()]
    bad = ids_i32 < 0
    res = (sl >= n_local) & (~bad)
    cpu_ids = torch.where(res | bad, torch.zeros_like(ids_i32), ids_i32)
    cpu_w = torch.where(res, torch.zeros_like(wts_f32), wts_f32)
    part_ids = torch.where(res, (sl - n_local).to(torch.int32),
                           torch.full_like(ids_i32, -1))
    return cpu_ids, cpu_w, part_ids, res


class PinMaskFused:
    """复用输出缓冲的融合实现（每层调用一次，稳态零分配）。

    `step()` 的入参与 `mixed_experts.py` 的 decode 分支一一对应；返回
    `(cpu_ids, cpu_w, part_ids)`，与 `torch_pin_mask` 的前三项逐位相同。
    命中数走内部 int64 累加器（`hits()`），对应旧实现的 `_hacc.add_(_res.sum())`。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._bufs: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}
        self._hits = torch.zeros((), dtype=torch.int64, device=device)
        self._enabled = os.environ.get(_ENV_ON, "") not in ("", "0")

    # ---- 内部：按 (形状, dtype) 复用缓冲 -----------------------------------
    def _buf(self, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
        """按 (语义名, 长度, dtype) 复用；**必须带 name** —— 否则 cpu_ids 与 part_ids
        都是 int32 同长，会拿到同一块内存，后被 part 覆盖（实测踩过）。"""
        key = (name, n, dtype)
        t = self._bufs.get(key)
        if t is None:
            t = torch.empty(n, dtype=dtype, device=self.device)
            self._bufs[key] = t
        return t

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and _HAS_TRITON)

    def step(self, ids: torch.Tensor, wts: torch.Tensor, slotmap: torch.Tensor,
             n_local: int):
        n = int(ids.numel())
        if (not self.enabled) or n == 0 or n > _MAX_ELEMS:
            return torch_pin_mask(ids, wts, slotmap, n_local)

        ids_i32 = ids.to(torch.int32)
        wts_f32 = wts.to(torch.float32)
        cpu_ids = self._buf("cpu_ids", n, torch.int32)
        cpu_w = self._buf("cpu_w", n, torch.float32)
        part_ids = self._buf("part_ids", n, torch.int32)

        block = 1 << max(5, (n - 1).bit_length())
        _pin_mask_kernel[(1,)](
            ids_i32, wts_f32, slotmap,
            cpu_ids, cpu_w, part_ids, self._hits,
            n, int(n_local), BLOCK=block,
            num_warps=4,
        )
        # 返回与输入同形状（内核按扁平索引工作）；否则调用方拿到 (n,) 会与 (M,K) 对不上
        shp = ids.shape
        return (cpu_ids.view(shp), cpu_w.view(shp), part_ids.view(shp))

    def hits(self) -> torch.Tensor:
        """累计命中数（int64 标量，device）。读取方需自行 `.item()` 并清零。"""
        return self._hits
