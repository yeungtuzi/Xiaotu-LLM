"""moe4_smallm — 小 M 4-bit 分组 MoE 的 CUDA 内核接入层（**默认关闭**）。

开关：`XIAOTU_PIN_CUDA_MOE=1` 才启用；缺省（未设 / 其它值）= 完全不生效，
      调用方走原有 Triton 路径，行为逐位不变。

适用条件（不满足 ⇒ `available()` 返回 False，调用方回退原路径）：
* 权重是 MXFP4（2 值/字节 + e8m0 分组尺度，`gk=32`）；
* K（hidden）与 N 满足内核对齐要求（N%4==0、K%32==0）；
* plane 是池内 **K-major** 布局 `[S, K/2, N]`（`expert_pool` 的 MXFP4 分支）。

验证记录：`docs/EXPERIMENTS.md` B1099–B1106
  * 真实 checkpoint 权重 + 池内 K-major：相对误差 **1.01e-6**（fp64 参照）
  * 生产形状（T=6, topk=6, 6 个不同专家）：**352 µs/层**
    （CPU 引擎 871 µs / 旧 GPU 路径 1272 µs）
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

_ENV = "XIAOTU_PIN_CUDA_MOE"

_module = None
_failed = False

# 小缓存：这些张量只由 (T/K/E/device) 决定，逐层重建纯属浪费（实测各 14/39 µs）
_FIRST_LOGGED = False
_FIRST_LOGGED_SEG = False
# 【2026-10-31】图捕获安全：捕获期禁止任何 host 分配 ⇒ 用**持久缓冲**（按形状缓存）。
#   key = (T, TOPSEL, E, H, I, device)
_buf_cache: dict = {}
_tok_ids_cache: dict = {}
_arange_cache: dict = {}


def enabled() -> bool:
    """总开关：只有显式 `XIAOTU_PIN_CUDA_MOE=1` 才算开。"""
    return os.environ.get(_ENV, "0") == "1"


def ready() -> bool:
    """内核是否**已加载好**（不触发加载/初始化，可在 CUDA 图捕获期安全调用）。"""
    return enabled() and (not _failed) and (_module is not None)


def available() -> bool:
    """开关开着 **且** 扩展里确实有内核符号时才为真（fail-closed）。"""
    if not enabled() or _failed:
        return False
    return _load() is not None


def disable(reason: str = "") -> None:
    """永久禁用（本进程内）：调用期出错时用，避免每层每步重复失败。"""
    global _failed
    if not _failed:
        print(f"[xtu-moe4] 已禁用，后续全部走原路径。{reason}", flush=True)
    _failed = True


def _load():
    global _module, _failed
    if _module is not None:
        return _module
    if _failed:
        return None
    try:
        import xiaotu_moe as _xtu  # 复用既有 loader（按 CPU 特性选变体）

        mod = _xtu.load()
        _need = ("moe4_smallm_gu", "moe4_smallm_dn", "moe4_smallm_init")
        if not all(hasattr(mod, n) for n in _need):
            # 引擎 .so 还没带内核（未重建）⇒ 明确失败一次后禁用，不反复探测
            print("[xtu-moe4] 内核不在当前变体 .so 里（需要重编译引擎），已禁用",
                  flush=True)
            _failed = True
            return None
        mod.moe4_smallm_init()
        # 快路径需要 gu2/dn2；旧 .so 只有 gu/dn ⇒ 自动回退段表版（而不是永久 disable）
        if not (hasattr(mod, "moe4_smallm_gu2") and hasattr(mod, "moe4_smallm_dn2")):
            os.environ["XIAOTU_MOE4_FAST"] = "0"
            print("[xtu-moe4] 当前 .so 无 gu2/dn2（旧产物）⇒ 自动回退段表版", flush=True)
        _module = mod
        print("[xtu-moe4] 小 M 4-bit 分组内核已启用（XIAOTU_PIN_CUDA_MOE=1）",
              flush=True)
        return _module
    except Exception as exc:  # noqa: BLE001
        print(f"[xtu-moe4] 加载失败，退回原路径：{type(exc).__name__}: {exc}",
              flush=True)
        _failed = True
        return None


def _segmentation(topk_ids, topk_weights, num_experts: int, device):
    """本模块自带的分段（**不依赖 gpu_prefill 的实现**）。

    实测 `gpu_prefill._build_segmentation` 在其 buffer 复用路径下会让本内核得到
    非确定结果（B1108 补：同输入两次差 1.5e-3）⇒ 这里用完全本地的实现，
    语义与之一致：按 expert id 稳定排序，(token,expert) 行按段连续，
    `seg_start` 长度 `E+1`、`tok`/`wts` 为 int32/fp32。

    返回 `(tok, wts, seg_start, A)`，`A = T*K`（= 行数上界）。
    """
    T, K = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    ids = topk_ids.to(torch.int32).reshape(-1)
    w = topk_weights.to(torch.float32).reshape(-1)
    _tk = (T, K, str(device))
    tok_ids = _tok_ids_cache.get(_tk)
    if tok_ids is None:
        tok_ids = torch.arange(T, device=device, dtype=torch.int32).repeat_interleave(K)
        _tok_ids_cache[_tk] = tok_ids
    bad = (ids < 0) | (ids >= num_experts)
    ids_c = torch.where(bad, torch.full((), num_experts, dtype=torch.int32, device=device), ids)
    order = torch.argsort(ids_c, stable=True)
    tok = tok_ids[order].to(torch.int32).contiguous()
    wts = w[order].to(torch.float32).contiguous()
    # ⛔ 不用 bincount：它会按数据的 max 决定输出长度 ⇒ **隐式 device→host 同步**
    #    （实测每层多花 ~200 µs；`gpu_prefill.py:880` 已有同一条教训）。
    #    改用固定长度的 scatter_add，零同步。
    ones = torch.ones_like(ids_c, dtype=torch.int32)
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, ids_c.long(), ones)
    counts = counts[:num_experts]
    seg_start = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    seg_start[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return tok, wts, seg_start.contiguous(), T * K


def _buffers(T, TOPSEL, E, H, I, device):
    """按形状缓存持久缓冲（捕获期复用，零分配）。"""
    key = (T, TOPSEL, E, H, I, str(device))
    b = _buf_cache.get(key)
    if b is not None:
        return b
    rows = int(E) * 8                      # 8 = 内核 kMaxA
    b = {
        "ids32": torch.empty((T, TOPSEL), dtype=torch.int32, device=device),
        "w32": torch.empty((T, TOPSEL), dtype=torch.float32, device=device),
        "inter": torch.zeros((rows, 2 * I), dtype=torch.float32, device=device),
        "tmp": torch.empty((rows, I), dtype=torch.float32, device=device),
        "act": torch.empty((rows, I), dtype=torch.float32, device=device),
        "res": torch.empty((T, H), dtype=torch.float32, device=device),
    }
    _buf_cache[key] = b
    return b


def moe_layer(
    planes: Sequence[torch.Tensor],
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    H: int,
    I: int,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    slot_map: Optional[torch.Tensor] = None,
    slot_base: int = 0,
    gk: int = 32,
    num_experts: Optional[int] = None,
    swiglu_limit: float = 0.0,
    group_k: Optional[int] = None,
    scale_kind: Optional[str] = None,
    s13_ts=None,
    s2_ts=None,
) -> Optional[torch.Tensor]:
    """一层 routed MoE（gate_up → SwiGLU → down），返回 `[T, H]`。

    `planes` 为池内 K-major 布局：`(w13[S,H/2,2I], s13[S,H/gk,2I], w2[S,I/2,H], s2[S,I/gk,H])`。
    ⚠️ **slot 路径下 `topk_ids` 已经是池内 plane 行号**（`mixed_experts.py:3130`
    `part_ids = (_sl - n_local)`）⇒ 应传 `slot_map=None`（恒等）。
    `slot_map[e]-slot_base` 仅在调用方传的是**专家号**时才需要。

    条件不满足时返回 None，由调用方回退原路径（fail-closed）。

    验证状态（**别混为一谈**）：
      * 内核本体：真实权重 + 池内布局，相对误差 1.0e-6（B1103/B1106/B1108）；
      * 接入链：已按审计整改（B1108），**尚未做服务级 logits 对拍**。
    """
    # 【2026-10-31】图安全改造后：**捕获期也允许**，但前提是内核已加载好
    #   （`ready()` 不触发加载；捕获期内绝不能再做 cudaMemcpyToSymbol 这类同步操作）。
    if torch.cuda.is_current_stream_capturing():
        if not ready():
            return None
    elif not available():
        return None
    mod = _module
    assert mod is not None
    w13, s13, w2, s2 = planes
    T = int(x.shape[0])
    if T <= 0:
        return None
    # ---- 格式硬校验（NVFP4 是 gk=16 + e4m3 块尺度 + 每专家 fp32 二级尺度，本内核不支持）----
    if group_k is not None and int(group_k) != 32:
        return None
    if scale_kind not in (None, "e8m0"):
        return None
    if s13_ts is not None or s2_ts is not None:
        return None
    if s13.dtype != torch.uint8 or s2.dtype != torch.uint8:
        return None
    if s13.shape[1] != H // 32 or s2.shape[1] != I // 32:
        return None
    # ---- SwiGLU 夹取（ds41f 生产就是 clamp@10.0）：在 Python 侧按 Triton 的语义做，
    #      `gate = min(gate, LIMIT)`、`up = clamp(up, -LIMIT, +LIMIT)`、`act = silu(gate)*up`
    #      （见 gpu_prefill.py:349-359）。此前为了保守直接回退，会导致内核在 ds41f 上永不运行。
    _limit = float(swiglu_limit or 0.0)
    if (H % 32) or (I % 32) or (gk != 32):
        return None
    if x.dtype != torch.bfloat16 or w13.dtype != torch.uint8:
        return None
    if w13.shape[1] != H // 2 or w13.shape[2] != 2 * I:
        return None
    if w2.shape[1] != I // 2 or w2.shape[2] != H:
        return None

    # ⚠️ 内核每个专家最多处理 8 个 (token,expert) 对（kMaxA=8）。
    #    T<=8 且"每 token 的 top-k 互异"时单专家最多 8 对，恰好不截断；
    #    为防静默截断，这里再要求 topk <= 8（正常 decode 是 6）。
    if T > 8 or int(topk_ids.shape[1]) > 8:
        return None
    # ---- 形状/连续性/device 硬门（内核按下标死算地址，非连续会静默读错行）----
    if x.shape[-1] != H or not x.is_contiguous():
        return None
    if int(topk_ids.shape[0]) != T or tuple(topk_weights.shape) != tuple(topk_ids.shape):
        return None
    for _t in (w13, s13, w2, s2):
        if not _t.is_contiguous() or _t.device != x.device:
            return None
    if topk_ids.device != x.device or topk_weights.device != x.device:
        return None
    # nexp 必须不超过权重行数（否则内核按 blockIdx.z 越界读 W/SC）
    if int(num_experts) > int(w13.shape[0]) or int(num_experts) > int(w2.shape[0]):
        return None
    # slot_map 语义：快路径把 id 直接当 plane 行号，传了 slot_map 就是"专家号"语义
    # ⇒ 必须回退段表版，绝不能静默忽略（R38 审计 P1-1）
    if slot_map is not None:
        return None
    # out 形状不符要在**发内核之前**判掉，否则白算一遍 + 调用方再算一遍
    if out is not None and (out.shape != (T, H) or out.dtype not in (torch.float32, torch.bfloat16)):
        return None

    device = x.device
    if num_experts is None:
        num_experts = max(int(topk_ids.max().item()) + 1 if topk_ids.numel() else 1,
                          int(w13.shape[0]))
    num_experts = int(num_experts)
    # 【2026-10-31】优先走**内核内 dispatch**（无主机侧分段，省 ~157 µs/层）。
    #   `XIAOTU_MOE4_FAST=0` 可回退到段表版（两版都已对 fp64 验证）。
    TOPSEL = int(topk_ids.shape[1])
    if os.environ.get("XIAOTU_MOE4_FAST", "1") == "1":
        # 【2026-10-31】**图捕获安全版**：捕获期不允许任何 host 分配，
        #   所以 ids/权重/inter/act/res 全部用**持久缓冲**，所有算子用 in-place 或 out=。
        b = _buffers(T, TOPSEL, int(num_experts), H, I, device)
        b["ids32"].copy_(topk_ids)          # dtype/设备转换在拷贝里完成，不分配
        b["w32"].copy_(topk_weights)
        inter2, tmp, act2, res2 = b["inter"], b["tmp"], b["act"], b["res"]
        inter2.zero_()
        stream = torch.cuda.current_stream().cuda_stream
        mod.moe4_smallm_gu2(
            w13.data_ptr(), s13.data_ptr(), x.data_ptr(), inter2.data_ptr(),
            b["ids32"].data_ptr(),
            T, TOPSEL, int(num_experts), int(H), int(2 * I), stream)
        g2 = inter2[:, :I]
        u2 = inter2[:, I:]
        if _limit > 0.0:
            g2.clamp_(max=_limit)           # ⚠️ 原地改 inter 的 gate 半区（inter 本就是临时量）
            u2.clamp_(-_limit, _limit)
        torch.sigmoid(g2, out=tmp)          # silu(g) = g * sigmoid(g)，全程无分配
        tmp.mul_(g2)
        tmp.mul_(u2)
        act2.copy_(tmp)
        global _FIRST_LOGGED
        if not _FIRST_LOGGED:
            _FIRST_LOGGED = True
            print(f"[xtu-moe4] first launch fast=1 T={T} K={TOPSEL} "
                  f"E={int(num_experts)} H={H} I={I} capture="
                  f"{torch.cuda.is_current_stream_capturing()}", flush=True)
        res2.zero_()
        mod.moe4_smallm_dn2(
            w2.data_ptr(), s2.data_ptr(), act2.data_ptr(), res2.data_ptr(),
            b["ids32"].data_ptr(), b["w32"].data_ptr(),
            T, TOPSEL, int(num_experts), int(I), int(H), stream)
        if out is not None:
            out.copy_(res2)
            return out
        return res2

    tok, wts, seg_start, A = _segmentation(topk_ids, topk_weights, num_experts, device)
    # ⛔ `_build_segmentation` 的 tok 是 **int64**（`torch.arange` 默认），
    #    内核按 `const int*` 读 ⇒ 必须显式转 int32，否则 token 号交替错位（静默错数）。
    tok = tok.to(torch.int32).contiguous()
    wts = wts.to(torch.float32).contiguous()
    seg_start = seg_start.to(torch.int32).contiguous()

    # ⛔ 不用 `nonzero` 求活跃专家列表（实测 39 µs/层）：直接传 arange(E) 当段号，
    #    空段（seg_start[e+1]==seg_start[e]）由内核 `if (na <= 0) return;` 跳过 ⇒ 零成本。
    _ak = (num_experts, str(device))
    seg2exp = _arange_cache.get(_ak)
    if seg2exp is None:
        seg2exp = torch.arange(num_experts, dtype=torch.int32, device=device)
        _arange_cache[_ak] = seg2exp
    nseg = int(num_experts)
    if nseg == 0:
        return None

    stream = torch.cuda.current_stream().cuda_stream
    global _FIRST_LOGGED_SEG
    if not _FIRST_LOGGED_SEG:
        _FIRST_LOGGED_SEG = True
        print(f"[xtu-moe4] first launch fast=0 T={T} E={int(num_experts)}", flush=True)
    inter = torch.zeros((A, 2 * I), dtype=torch.float32, device=device)
    _sm = slot_map.data_ptr() if slot_map is not None else 0
    mod.moe4_smallm_gu(
        w13.data_ptr(), s13.data_ptr(), x.data_ptr(), inter.data_ptr(),
        seg2exp.data_ptr(), seg_start.data_ptr(), tok.data_ptr(),
        _sm, int(slot_base), nseg, int(H), int(2 * I), stream,
    )
    _g1 = inter[:, :I]
    _u1 = inter[:, I:]
    if _limit > 0.0:
        _g1 = _g1.clamp(max=_limit)
        _u1 = _u1.clamp(-_limit, _limit)
    act = (torch.nn.functional.silu(_g1) * _u1).contiguous()

    res = torch.zeros((T, H), dtype=torch.float32, device=device)
    mod.moe4_smallm_dn(
        w2.data_ptr(), s2.data_ptr(), act.data_ptr(), res.data_ptr(),
        seg2exp.data_ptr(), seg_start.data_ptr(), tok.data_ptr(), wts.data_ptr(),
        _sm, int(slot_base), nseg, int(I), int(H), stream,
    )
    if out is not None:
        if out.shape != res.shape:
            return None          # 调用方的持久缓冲对不上 ⇒ 回退，绝不静默换张量
        out.copy_(res.to(out.dtype))
        return out
    return res.to(out_dtype)
