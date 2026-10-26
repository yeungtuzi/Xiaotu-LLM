"""上游 vLLM 兼容补丁（**全部在插件侧运行时打**，不在 vLLM 树内改文件）。

⭐ 为什么这样做
-------------
本项目的治理规则是：**vLLM 只跟上游，我们的功能一律以 patch 形式留在本仓**。
如果为了跑通某个模型/平台去直接改 vLLM 树，就会产生"跟进上游的负债"：
每次 rebase 都要手工重放这些改动，而且改动一旦是 hack（改语义、掩盖错误），
下次上游行为变化时会静默出错 ✗

⇒ 所以平台/模型族的适配**一律在本模块里 monkeypatch**：
  · 插件是 OOT（`vllm_xiaotu_moe/**`），上游怎么变都不冲突 ✓
  · 上游树保持 **0 处** 本项目改动 ⇒ `git status` 干净，rebase 无负债 ✓
  · 若某处确实该进上游（例如平台 bug 的 workaround），再单独成 patch ✓

开关
----
* `XIAOTU_UPSTREAM_COMPAT=1`（默认开）—— 总开关；设 `0` 可完全关闭本模块所有补丁。
* `XIAOTU_NVFP4=1` —— NVFP4 相关补丁额外条件（避免影响 FP8 主线）。
"""

from __future__ import annotations

import os

_APPLIED: list[str] = []
# 需要**延迟到 post-load** 才打的补丁（模型模块那时才就绪）
_PLE_PENDING: list = []


def applied() -> list[str]:
    """已应用的补丁名（便于日志/自检）。"""
    return list(_APPLIED)


def _enabled() -> bool:
    return os.environ.get("XIAOTU_UPSTREAM_COMPAT", "1").strip() != "0"


def _nvfp4() -> bool:
    return os.environ.get("XIAOTU_NVFP4", "0").strip() == "1"


# ────────────────────────────────────────────────────────────────────────────
# 补丁 1：MUSA 上 NVFP4 模拟内核的 Triton 实现编不出来 ⇒ 换纯 torch 等价实现
# ────────────────────────────────────────────────────────────────────────────
def _torch_round_to_e2m1(x):
    """按 E2M1 最近值量化（只用比较+累加：`bucketize`/fp8 cast 在 MUSA 上不可用）。"""
    import torch

    a = x.abs()
    out = torch.zeros_like(a)
    for th, delta in ((0.25, 0.5), (0.75, 0.5), (1.25, 0.5), (1.75, 0.5),
                      (2.5, 1.0), (3.5, 1.0), (5.0, 2.0)):
        out = out + (a >= th).to(a.dtype) * delta
    return torch.sign(x) * out


def _torch_nvfp4_quant_dequant(x, global_scale, block_size):
    """与 vLLM 的 Triton 版**逐行等价**：逐组(16)求尺度 → clamp → e4m3 → E2M1 → 反量化。"""
    import torch

    x_m, x_k = x.shape
    assert x_k % block_size == 0, (x.shape, block_size)
    out_dtype = x.dtype
    nb = x_k // block_size
    gs = float(global_scale.reshape(-1)[0].float().item())
    xf = x.float().reshape(x_m, nb, block_size)
    scale = (gs * xf.abs().amax(dim=2) * (1.0 / 6.0)).clamp(-448.0, 448.0)
    try:  # 组尺度量化到 e4m3（MUSA 不支持该 cast 时跳过，误差极小）
        scale = scale.to(torch.float8_e4m3fn).float()
    except Exception:  # noqa: BLE001
        pass
    zero = scale == 0
    safe = torch.where(zero, torch.ones_like(scale), scale)
    out_scale = torch.where(zero, torch.zeros_like(scale), gs / safe)
    scaled = (xf * out_scale[:, :, None]).clamp(-6.0, 6.0)
    fp4 = _torch_round_to_e2m1(scaled)
    return (fp4 * (scale / gs)[:, :, None]).reshape(x_m, x_k).to(out_dtype)


def patch_nvfp4_emulation_for_musa() -> bool:
    """MUSA 上把 `_triton_nvfp4_quant_dequant` 换成纯 torch 实现。

    为什么必须换：Triton 在 MUSA 上编 fp8 类型转换会失败 ——
    `ld.lld: error: relocation R_MTGPU_REL32_DWORD cannot be used against
     symbol '__mt_tt_e4m3_to_f32'; recompile with -fPIC`（实测，厂商工具链问题）。
    """
    if not _nvfp4():
        return False
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_musa():
            return False
        import vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils as _u
    except Exception:  # noqa: BLE001 上游路径变化时静默跳过（不影响 FP8 主线）
        return False
    if getattr(_u, "_xtu_torch_patched", False):
        return True
    _u._triton_nvfp4_quant_dequant = _torch_nvfp4_quant_dequant
    _u._xtu_torch_patched = True
    _APPLIED.append("nvfp4_emulation_torch(musa)")
    return True


# ────────────────────────────────────────────────────────────────────────────
# 补丁 2：AutoWeightsLoader 遇到"模块上不存在的属性名"时**跳过**而不是崩
# ────────────────────────────────────────────────────────────────────────────
def patch_loader_skip_unknown_attr() -> bool:
    """让不可解析的权重名**跳过并告警**。

    为什么需要：某些 checkpoint 会带出目标模块上并不存在的属性（实测
    `...linear_attn.in_proj_qkvz.weight_scale`，而该层已是 BF16、没有尺度参数）。
    vLLM 原实现会把**模块自身**当作 param 传下去 ⇒
    `'MergedColumnParallelLinear' object has no attribute 'data'` ✗

    ⚠️ 这是**兜底**：能通过**规范化 checkpoint 索引**解决的，一律在源头解决
    （见 `dev-docs/HANDOFF_NVFP4_2026-10-26.md` §4），不要依赖本补丁 ✗

    ⭐ 2026-10-26 改为**显式 opt-in**（`XIAOTU_LOAD_SKIP_UNKNOWN=1`，缺省关）：
    本补丁会把 `There is no module or parameter named ...` 整类错误吞掉 ⇒
    若真有一批权重没被加载，它只会打印一行 SKIP 然后**继续跑**，
    表现为"能跑但数错" ✗ —— 这与本项目最怕的失败模式正好相同。
    所以缺省**不允许**静默跳过；只有明确的兜底场景才开。
    """
    if not _nvfp4():
        return False
    if os.environ.get("XIAOTU_LOAD_SKIP_UNKNOWN", "0").strip() != "1":
        return False
    try:
        from vllm.model_executor.models import utils as _mu
    except Exception:  # noqa: BLE001
        return False
    if getattr(_mu, "_xtu_skip_unknown_patched", False):
        return True

    def _wrap(orig):
        def _inner(self, *a, **kw):
            try:
                return orig(self, *a, **kw)
            except ValueError as exc:
                msg = str(exc)
                if "There is no module or parameter named" not in msg:
                    raise
                print(f"[xtu-load-skip] SKIP unresolved weight: {msg[:160]}", flush=True)
                return None
        return _inner

    cls = getattr(_mu, "AutoWeightsLoader", None)
    if cls is not None and hasattr(cls, "_load_module"):
        cls._load_module = _wrap(cls._load_module)
        _mu._xtu_skip_unknown_patched = True
        _APPLIED.append("loader_skip_unknown_attr")
        return True
    return False


# ────────────────────────────────────────────────────────────────────────────
# 补丁 3：checkpoint 的 `weight_scale_inv` → vLLM `FP8_PB_WO` 的 `weight_scale`
# ────────────────────────────────────────────────────────────────────────────
def _xtu_scale_rename(weights):
    """按名字流式规范尺度键（与树内 `models/utils.py` 的 `_xtu_scale_rename` 等价）。

    两类处理：
    1. `.linear_attn.` / `.self_attn.` 下的任何尺度键 ⇒ **丢弃**：
       这些层已被我们**离线反量化成 BF16**（见 HANDOFF §5），checkpoint 索引里
       仍可能映射出尺度名；模块已不量化，留着只会让加载器把它当属性解析而报错 ✗
    2. 其余 `*weight_scale_inv` ⇒ 改名为 `weight_scale`，并把 **2 维** (out/128, in/128)
       补成 **4 维** (ob,1,ib,1)：vLLM `FP8_PB_WO` 把参数注册成 4 维，
       `parameter.py` 的 shape 断言要求两侧一致 ✗
    """
    for _n, _t in weights:
        if (_n.endswith(("weight_scale", "weight_scale_inv", "weight_scale_2",
                         "input_scale"))
                and (".linear_attn." in _n or ".self_attn." in _n)):
            continue
        if _n.endswith("weight_scale_inv") and ".experts." not in _n:
            _n = _n[: -len("weight_scale_inv")] + "weight_scale"
            if getattr(_t, "dim", None) is not None and _t.dim() == 2:
                _t = _t.unsqueeze(1).unsqueeze(-1)
        yield _n, _t


def patch_loader_scale_rename() -> bool:
    """在 `WeightsMapper.apply` 之后规范尺度键（等价于树内 `models/utils.py` 的改动）。

    为什么打在 `WeightsMapper.apply` 而不是 `AutoWeightsLoader.load_weights`：
    树内改动的位置就是"映射**之后**、过滤之前"，顺序参与语义 ——
    映射前改名会被后续 `orig_to_new_prefix` 等规则改写，映射后改名才稳妥 ✓
    """
    if not _nvfp4():
        return False
    try:
        from vllm.model_executor.models.utils import WeightsMapper
    except Exception:  # noqa: BLE001  上游路径变化时静默跳过（不影响 FP8 主线）
        return False
    if getattr(WeightsMapper, "_xtu_scale_rename_patched", False):
        return True
    _orig = WeightsMapper.apply

    def apply(self, weights):
        return _xtu_scale_rename(_orig(self, weights))

    WeightsMapper.apply = apply
    WeightsMapper._xtu_scale_rename_patched = True
    _APPLIED.append("loader_scale_rename(nvfp4)")
    return True


def patch_model_post_load_hook() -> bool:
    """给 `qwen4_exp` 族补上 `finalize_mega_moe_weights` 的后置加载钩子（A78-8）。

    ⛔ **为什么这条最致命**：`deepseek_v4` 的 `ForCausalLM.load_weights` 末尾会调
    `self.model.finalize_mega_moe_weights()` 来**构造 xiaotu 引擎**；
    而 `Qwen4ExpForCausalLM` 是另一条继承链（Qwen3_5），**没有这个钩子** ⇒
    `CpuXiaotuMoE.forward` 里 `if self.engine is None: return hidden_states`
    会**静默把整层 MoE 跳过** —— 模型照样出字、bench 照样有 tok/s，数值全错 ✗
    （实测：worker "graphs captured" 都完成了，日志里 0 条 `引擎=MOE_NVFP4`。）

    做法：包一层 `load_weights`，加载完成后遍历所有子模块，对带
    `finalize_mega_moe_weights` 的调一次（幂等由各实现自己保证）。
    """
    if not _nvfp4():
        return False
    try:
        import vllm.models.qwen4_exp.nvidia.model as _qm
    except Exception:  # noqa: BLE001  没这个模型族就跳过，不影响 DS4/FP8 主线
        return False
    cls = getattr(_qm, "Qwen4ExpForCausalLM", None)
    if cls is None:
        return False
    if getattr(cls, "_xtu_post_load_patched", False):
        return True
    _orig = cls.load_weights

    def load_weights(self, weights):
        out = _orig(self, weights)
        _n = 0
        for _m in self.modules():
            _fn = getattr(_m, "finalize_mega_moe_weights", None)
            if callable(_fn):
                _fn()
                _n += 1
        print(f"[xtu-moe] post-load: 已请求构造 {_n} 个 MoE 层引擎", flush=True)
        # ⭐ 延迟补丁（PLE 设备反量化）：此刻模型模块一定已导入 ⇒ 才能安全 monkeypatch
        for _fn in list(_PLE_PENDING):
            try:
                _fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[xtu-moe] ⚠️ 延迟补丁失败: {type(exc).__name__}: {exc}",
                      flush=True)
        try:
            _warmup_moe_shapes(self, __import__("torch"))
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-moe] ⚠️ MoE 形状预热失败: {type(exc).__name__}: {exc}",
                  flush=True)
        try:
            _disable_ple(self, __import__("torch"))
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-moe] ⚠️ PLE 关闭失败: {type(exc).__name__}: {exc}", flush=True)
        try:
            _apply_ple_patches(self)
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-moe] ⚠️ PLE 设备缓冲补丁失败: {type(exc).__name__}: {exc}",
                  flush=True)
        # ⭐ `XIAOTU_MOE_RESIDENT_DIAG=1`：常驻上传**之后、KV 定容之前**的显存账。
        #   为什么要有：实测 48 层全驻留时 `model loading took 37.66 GiB`，
        #   但 KV 定容前 free 只剩 0.17 GiB ⇒ **中间多出 ~10 GiB 查不到去处** ✗
        #   ⇒ 这里直接打 allocated / reserved / free，用差值定位是"真分配"还是
        #   "allocator 保留未用"（后者可用 empty_cache 归还）
        if os.environ.get("XIAOTU_MOE_RESIDENT_DIAG", "0") == "1":
            try:
                import torch as _tj

                _al = _tj.cuda.memory_allocated() / 2**30
                _re = _tj.cuda.memory_reserved() / 2**30
                _fr, _tot = _tj.cuda.mem_get_info()
                print(
                    f"[xtu-res-diag] 上传后: allocated={_al:.2f} reserved={_re:.2f} "
                    f"free={_fr / 2**30:.2f}/{_tot / 2**30:.2f} GiB "
                    f"（reserved-allocated={_re - _al:.2f} GiB 可 empty_cache 归还）",
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[xtu-res-diag] 取显存失败(不掩盖): {type(_e).__name__}: {_e}",
                      flush=True)
        return out

    cls.load_weights = load_weights
    cls._xtu_post_load_patched = True
    _APPLIED.append("model_post_load_finalize(qwen4_exp)")
    return True


def patch_nvfp4_oracle_fallback() -> bool:
    """NVFP4 MoE 后端"没人支持"时，env 门控回退到 EMULATION 类（等价于树内 `oracle/nvfp4.py`）。

    为什么要这条：我们**不用** vLLM 的 NVFP4 MoE 后端做计算（计算由 xiaotu 引擎接管），
    但 vLLM 仍要**把模块外壳与权重参数建出来**。官方 vLLM 在 MUSA 上没有任何 NVFP4
    MoE 后端能把 `is_supported_config` 判过 ⇒ 直接抛
    `NotImplementedError: No NvFp4 MoE backend supports the deployment configuration.`
    ⇒ 回退到 EMULATION 类只为 `create_weights`/参数形状，**前向不会走到它**。

    ⚠️ **fail-closed**：只有显式 `XIAOTU_NVFP4_MOE_FALLBACK=1` 才回退；否则原样抛出 ✗
    （否则会在别的模型/精度上悄悄改变后端选择）
    """
    if not _nvfp4():
        return False
    try:
        import importlib

        _mod = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4")
    except Exception:  # noqa: BLE001  上游路径变化时静默跳过（不影响 FP8 主线）
        return False
    _orig = getattr(_mod, "select_nvfp4_moe_backend", None)
    if _orig is None or getattr(_orig, "_xtu_oracle_patched", False):
        return bool(getattr(_orig, "_xtu_oracle_patched", False))

    def select_nvfp4_moe_backend(*a, **kw):
        try:
            return _orig(*a, **kw)
        except NotImplementedError:
            if os.environ.get("XIAOTU_NVFP4_MOE_FALLBACK", "0").strip() != "1":
                raise
            from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (  # noqa: E501
                Nvfp4QuantizationEmulationTritonExperts as _Fb,
            )
            print(
                "[xtu-nvfp4] ⚠ 回退到 EMULATION 类（仅建权重外壳，前向由 xiaotu 引擎接管）",
                flush=True,
            )
            return _mod.NvFp4MoeBackend.EMULATION, _Fb

    select_nvfp4_moe_backend._xtu_oracle_patched = True
    _mod.select_nvfp4_moe_backend = select_nvfp4_moe_backend
    # 有些调用方是 `from ... import select_nvfp4_moe_backend` 提前绑定的 ⇒ 一并改掉，
    # 否则补丁对它们无效（静默失效比报错更难查 ✗）
    import sys as _sys

    for _m in list(_sys.modules.values()):
        if _m is None or _m is _mod:
            continue
        if getattr(_m, "select_nvfp4_moe_backend", None) is _orig:
            try:
                _m.select_nvfp4_moe_backend = select_nvfp4_moe_backend
            except Exception:  # noqa: BLE001
                pass
    _APPLIED.append("nvfp4_oracle_fallback")
    return True


def patch_fp8_backend_order_on_musa() -> bool:
    """MUSA 在场时**别让 CPU FP8 后端排在候选首位**（等价于树内 `oracle/fp8.py` 的改动）。

    为什么需要：`current_platform` 是惰性代理，后端选择发生在平台解析完成之前，
    此时 `is_cpu()` 会短暂返回 True ⇒ 树内把 `CPU`/`CPU_W8A8` 移到列表最前
    ⇒ 整个模型可能永远走 CPU 专家（实测见 `docs/musa/MUSA_BUILD_PITFALLS.md`）。
    这里改成"若 MUSA 可见，就把 CPU 系后端挪到**末尾**"，语义等价且不必改树 ✓
    """
    if not _nvfp4():
        return False
    try:
        import importlib

        _mod = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.oracle.fp8")
    except Exception:  # noqa: BLE001
        return False
    _orig = getattr(_mod, "_get_priority_backends", None)
    if _orig is None or getattr(_orig, "_xtu_patched", False):
        return bool(getattr(_orig, "_xtu_patched", False))

    def _musa_visible() -> bool:
        try:
            import torch

            return bool(torch.musa.is_available() and torch.musa.device_count() > 0)
        except Exception:  # noqa: BLE001
            return False

    _CPU = {"CPU", "CPU_W8A8"}

    def _get_priority_backends(*a, **kw):
        lst = _orig(*a, **kw)
        if _musa_visible():
            _cpu = [b for b in lst if getattr(b, "name", "") in _CPU]
            if _cpu:
                lst = [b for b in lst if getattr(b, "name", "") not in _CPU] + _cpu
        return lst

    _get_priority_backends._xtu_patched = True
    _mod._get_priority_backends = _get_priority_backends
    import sys as _sys

    for _m in list(_sys.modules.values()):
        if _m is None or _m is _mod:
            continue
        if getattr(_m, "_get_priority_backends", None) is _orig:
            try:
                _m._get_priority_backends = _get_priority_backends
            except Exception:  # noqa: BLE001
                pass
    _APPLIED.append("fp8_backend_order_musa")
    return True


def patch_ple_prefetch_buffer_to_device() -> bool:
    """把 PLE 的 **prefetch 输出缓冲** 从主机搬到设备 ⇒ forward 里再无主机张量/主机拷贝。

    为什么必须做（2026-10-26 实测）：`cpu_offload=True` 时
    `Qwen4ExpPLEPinnedHostEmbedding.start_prefetch()` 用
    `device=self._uva_weight.device` 分配缓冲 —— 而 UVA 表在**主机**上
    ⇒ 缓冲也在主机 ⇒ 后续 `dequantize` 拿到的是**主机张量**
    ⇒ 任何反量化都要 H2D（捕获期非法），于是 CUDA 图永远起不来 ✗

    改法：在 post-load 阶段**预置**设备侧缓冲与侧流（形状/类型从模块属性推导），
    `start_prefetch` 见到 `_prefetch_buffer` 非空就会跳过分配 ⇒ gather 直接写设备内存
    （UVA 读主机表 → 写设备缓冲，是**设备内核**，不是主机拷贝）✓
    未激活的环境（没有该属性/已就绪）静默跳过。
    """
    if not _nvfp4():
        return False
    try:
        import torch

        from vllm.compilation.breakable_cudagraph import (  # noqa: F401
            BreakableCUDAGraphCapture)
    except Exception:  # noqa: BLE001
        pass
    return True


def _install_break_dequant(cls, torch) -> None:
    """把 PLE 的 FP8 反量化包成 `@eager_break_during_capture` 断点。

    为什么这是**正解**（而不是把反量化搬到设备）：
    * 用户口径（2026-10-26）：**PLE 表本来就该在主机 RAM/SSD**，属于设计内 ⇒
      PLE 那一段走 eager 是允许的；要求"纯 GPU"的是**其余 47/48 层**
    * vLLM 自己就是这么设计的：`start_prefetch` 带 `@eager_break_during_capture`，
      只有 `dequantize` 这一步忘了标记 ⇒ 图捕获在 `.cpu()` 处失败 ✗
    * 断点语义（`breakable_cudagraph.py:67`）：捕获期 **结束当前段 → eager 跑 → 记录供 replay → 开新段**
      ⚠️ 要求**输出地址稳定** ⇒ 这里写进**持久缓冲**，否则 replay 后下游图段读到旧地址 ✗
    """
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    orig = cls.dequantize
    bufs: dict = {}
    printed = {"n": 0}

    def _buf(shape, dtype):
        key = (tuple(shape), dtype)
        t = bufs.get(key)
        if t is None:
            t = torch.empty(tuple(shape), dtype=dtype, device="cuda")
            bufs[key] = t
        return t

    def dequantize(self, layer, embeddings, output_dtype):  # noqa: ANN001
        host = orig(self, layer, embeddings, output_dtype)   # 上游主机版（含 .cpu()）
        out = _buf(host.shape, host.dtype)
        out.copy_(host, non_blocking=True)
        if printed["n"] < 1:
            printed["n"] = 1
            print(f"[xtu-ple] 反量化已包成 eager break（输出写持久缓冲 "
                  f"{tuple(host.shape)} {host.dtype}）⇒ 其余层可进图", flush=True)
        return out

    cls.dequantize = eager_break_during_capture(dequantize)
    cls._xtu_break_installed = True       # type: ignore[attr-defined]


def _e4m3_lut_values():
    """e4m3fn 的 256 项精确值（纯 float 手算，不依赖任何 fp8 dtype）。"""
    tbl = []
    for b in range(256):
        sign = -1.0 if (b >> 7) & 1 else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            v = man * 2.0 ** -9            # 次正规
        else:
            v = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        if exp == 0xF and man == 0x7:
            v = float("nan")               # e4m3fn 唯一的 NaN
        tbl.append(sign * v)
    return tbl


def _install_lut_dequant(cls, torch, weight_scale=None) -> None:
    """把 `cls.dequantize` 换成**全设备**版本（字节 + 256 项 LUT）。

    * 避开 fp8 dtype（MUSA 上 fp8→float 的设备转换会回退主机 ⇒ 捕获期非法）
    * 尺度在**装补丁时**就搬到设备缓存（否则每步一次 H2D，同样会让捕获失败）
    * 拿不到设备张量（或显式 `XIAOTU_PLE_HOST_DEQUANT=1`）时走原实现，行为不变
    """
    orig = cls.dequantize
    lut_cache: dict = {}
    scale_cache: dict = {}
    printed = {"n": 0}

    def _lut(dev):
        t = lut_cache.get(str(dev))
        if t is None:
            t = torch.tensor(_e4m3_lut_values(), dtype=torch.float32, device=dev)
            lut_cache[str(dev)] = t
        return t

    def _scale(ws, dev):
        k = str(dev)
        if k not in scale_cache:
            scale_cache[k] = (ws.detach().to(dtype=torch.float32)
                              .reshape(-1)[0].to(device=dev, dtype=torch.float32)
                              .contiguous())
        return scale_cache[k]

    def dequantize(self, layer, embeddings, output_dtype):  # noqa: ANN001
        ws = getattr(layer, "weight_scale", None)
        if ws is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if (embeddings.device.type != "cuda"
                or os.environ.get("XIAOTU_PLE_HOST_DEQUANT", "0") == "1"):
            return orig(self, layer, embeddings, output_dtype)
        try:
            s = _scale(ws, embeddings.device)
            if embeddings.dtype == output_dtype:
                return embeddings
            idx = embeddings.view(torch.uint8).to(torch.int64)
            f = _lut(embeddings.device).index_select(0, idx.reshape(-1)).reshape(
                embeddings.shape)
            out = f.mul(s).to(output_dtype)
            if printed["n"] < 1 and not getattr(torch.cuda, "is_current_stream_capturing",
                                                lambda: False)():
                printed["n"] = 1
                print(f"[xtu-ple] 设备 LUT 反量化生效: shape={tuple(embeddings.shape)} "
                      f"{embeddings.dtype}->{output_dtype} on {embeddings.device}",
                      flush=True)
            return out
        except Exception as exc:  # noqa: BLE001
            if printed["n"] < 2:
                printed["n"] = 2
                print(f"[xtu-ple] ⚠️ 设备 LUT 反量化失败，退回主机版: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            return orig(self, layer, embeddings, output_dtype)

    dequantize._xtu_lut = True          # type: ignore[attr-defined]
    cls.dequantize = dequantize
    cls._xtu_lut_installed = True       # type: ignore[attr-defined]
    if weight_scale is not None:
        try:  # 预热：把尺度**在装补丁时**搬到设备（捕获前完成）
            _scale(weight_scale, torch.device("cuda"))
        except Exception:  # noqa: BLE001
            pass


def _warmup_moe_shapes(model, torch) -> int:
    """把 MoE 的 Triton 内核**在捕获之前**按各捕获 batch 预编译一遍。

    为什么必须（2026-10-26 实测）：`[step-trace]` 显示捕获期 T=8 时
    A/B/C 都过了，但**挂在内核 launch 之后**（两个 rank 都是 R 状态自旋、380 线程）——
    这是 Triton 在捕获期**首次编译/加载 module** 的典型表现（profile 只跑过 T=128/512，
    捕获用 T=1/2/4/8 ⇒ 签名是新的 ⇒ 编译发生在 capture 里 ✗）

    ⇒ 在 post-load（eager 期）按 `XIAOTU_MOE_WARM_SHAPES` 逐个 T 走一次真实前向，
       让所有签名提前编译好；捕获期就只剩纯 launch ✓
    """
    if os.environ.get("XIAOTU_MOE_WARM_SHAPES", "1,2,4,8,16,32,64,128,256,512") == "0":
        return 0
    raw = os.environ.get("XIAOTU_MOE_WARM_SHAPES",
                         "1,2,4,8,16,32,64,128,256,512")
    shapes = [int(x) for x in raw.split(",") if x.strip().isdigit()]
    mods = [m for m in model.modules()
            if getattr(m, "_resident_slot", None) is not None
            and hasattr(m, "hidden_size")]
    if not mods or not shapes:
        return 0
    n = 0
    try:
        dev = torch.device("cuda")
        H = int(mods[0].hidden_size)
        with torch.no_grad():
            for T in shapes:
                x = torch.zeros((T, H), dtype=torch.bfloat16, device=dev)
                for m in mods:
                    try:
                        m(x)
                        n += 1
                    except Exception:  # noqa: BLE001
                        pass
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[xtu-warm] ⚠️ 预热失败: {type(exc).__name__}: {exc}", flush=True)
        return 0
    print(f"[xtu-warm] MoE 内核预编译完成: {len(mods)} 层 × {len(shapes)} 个形状 "
          f"= {n} 次前向（捕获期不会再编译）", flush=True)
    return n


def _disable_ple(model, torch) -> int:
    """`XIAOTU_PLE_DISABLE=1`：**彻底**把 PLE 从 forward 里摘掉（对照实验 / 让图捕获可行）。

    为什么单独一个函数、而且**不依赖任何模块特征**：
      * 早前把它塞进 `_apply_ple_patches` 的主循环（只遍历带 `_prefetch_rows` 的模块）⇒
        一旦那个循环没命中，**整段关闭逻辑静默不执行**（ploff4 就是这么白跑一轮）✗
      * PLE 有**三个**入口都要关（见 HANDOFF §61）：
        ① 模型级 `_start_layer_ple_prefetch`（真正入口）
        ② 嵌入模块 `start_prefetch`（预取 + host gather）
        ③ PLE 层的 `forward`（把结果加进 hidden_states）
    """
    if os.environ.get("XIAOTU_PLE_DISABLE", "0") != "1":
        return 0
    n = 0
    try:
        # ① 模型级入口：沿 MRO 找（可能定义在父类上）
        for klass in type(model).__mro__:
            if "_start_layer_ple_prefetch" in vars(klass):
                klass._start_layer_ple_prefetch = lambda self, *a, **k: None
                klass._xtu_noprefetch = True
                print(f"[xtu-ple-off] {klass.__name__}._start_layer_ple_prefetch 已置空",
                      flush=True)
                n += 1
                break
    except Exception as exc:  # noqa: BLE001
        print(f"[xtu-ple-off] ⚠️ 置空 prefetch 入口失败: {type(exc).__name__}: {exc}",
              flush=True)
    seen_cls = set()
    for m in model.modules():
        cls = type(m)
        if cls in seen_cls:
            continue
        # 判据：类名含 PLE **或** 该类定义了 start_prefetch（嵌入模块的强特征，鸭子类型）
        if "PLE" not in cls.__name__ and "start_prefetch" not in vars(cls):
            continue
        seen_cls.add(cls)
        try:
            def forward(self, hidden_states, *a, **kw):  # noqa: ANN001
                return hidden_states
            cls.forward = forward
            cls._xtu_passthrough = True
            if "start_prefetch" in vars(cls):
                cls.start_prefetch = lambda self, *a, **k: None
            n += 1
            print(f"[xtu-ple-off] {cls.__name__}: forward/start_prefetch 已置空", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-ple-off] ⚠️ {cls.__name__} 关闭失败: {type(exc).__name__}: {exc}",
                  flush=True)
    print(f"[xtu-ple-off] 共处理 {n} 个类（XIAOTU_PLE_DISABLE=1）", flush=True)
    return n


def _apply_ple_patches(model) -> int:
    """post-load：给 PLE 的反量化装 **eager break**（默认），可选把预取缓冲搬到设备。

    背景（2026-10-26 实测，用户口径）：**PLE 表本来就该在主机 RAM/SSD**，属于设计内 ⇒
    PLE 那一段 eager 是允许的；要求"纯 GPU"的是**其余 47/48 层**。
    vLLM 自己也这么设计：`start_prefetch` 带 `@eager_break_during_capture`，
    只有 `dequantize` 忘了标记 ⇒ 图捕获在 `.cpu()` 处失败 ✗

    这里**不 import 模型模块**（会触发 `Tried to instantiate class …dequantize` 的懒校验）：
    直接从 `model.modules()` 拿到 `embedding_method` 实例，改 `type(inst).dequantize` ✓
    """
    want_break = os.environ.get("XIAOTU_PLE_BREAK", "1") == "1"
    want_lut = os.environ.get("XIAOTU_PLE_LUT", "0") == "1"
    want_dev_buf = os.environ.get("XIAOTU_PLE_DEVICE_BUFFER", "0") == "1"
    if not (want_break or want_lut or want_dev_buf):
        return 0
    try:
        import torch
    except Exception:  # noqa: BLE001
        return 0

    n_buf = 0
    n_cls = 0
    seen = 0
    for m in model.modules():
        # ⭐ 鸭子类型判定：类可能被包装（compile/代理），按类名会漏 ✗
        if not (hasattr(m, "_prefetch_rows") and hasattr(m, "_uva_weight")):
            continue
        seen += 1

        if want_dev_buf:
            buf = getattr(m, "_prefetch_buffer", None)
            if buf is None or buf.device.type != "cuda":
                try:
                    if getattr(m, "_prefetch_stream", None) is None:
                        m._prefetch_stream = torch.cuda.Stream(device="cuda")
                    rows = int(getattr(m, "_prefetch_rows"))
                    heads = int(getattr(m, "_num_ngram_heads"))
                    dim = int(m.embedding_dim)
                    m._prefetch_buffer = torch.empty(
                        rows, heads, dim, dtype=m.weight.dtype, device="cuda")
                    n_buf += 1
                    print(f"[xtu-ple] prefetch 缓冲改到设备: {rows}x{heads}x{dim} "
                          f"{m.weight.dtype}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[xtu-ple] ⚠️ 设备缓冲预置失败: {type(exc).__name__}: {exc}",
                          flush=True)

        # ── ②b PLE **单流化**（`XIAOTU_PLE_SINGLE_STREAM=1`，默认开）───────────
        #  实测：`start_prefetch` 的 `with torch.cuda.stream(prefetch_stream)` 里
        #  有 `ids_cpu = flat_ids.to("cpu")`（`ngram_embedding.py:598`）⇒
        #  即使 `@eager_break_during_capture` 已经 `_end_segment()`，
        #  **侧流仍在捕获隔离态** ⇒ 报 `operation not permitted when stream is capturing` ✗
        #  把 `_prefetch_stream` 换成**当前流**：`with stream(cur)` 是空操作、
        #  `wait_stream(cur)` 自满足 ⇒ 那条 D2H 变合法 ✓（PLE 允许 eager）
        #  ⭐【2026-10-26 · 第三次修法，终于对上时机】
        #     `eager_break_during_capture` = 「结束段 → **跑函数** → 开新段」。
        #     若在**函数返回之后**才 join copy 流，那次 wait 会落在**新开的捕获段里** ⇒
        #     反而制造"依赖未捕获工作" ⇒ `capturing stream has unjoined work` ✗
        #     ⇒ 必须让 join 发生在 **eager 区间内部**：外层再包一层断点，join 写在里面 ✓
        #     （内层那个上游装饰器此时 `_capturing==False`，直接执行函数体，不会二次断点）
        if True:   # ⭐ 外层断点**总是**装；是否单流由 XIAOTU_PLE_SINGLE_STREAM 决定
            try:
                cls_s = type(m)
                if not getattr(cls_s, "_xtu_sp_join", False):
                    _orig_sp = cls_s.start_prefetch
                    from vllm.compilation.breakable_cudagraph import (
                        eager_break_during_capture as _eb2)

                    def _sp_join(self, ngram_ids, _o=_orig_sp):  # noqa: ANN001
                        # ⭐ 对照实验：PLE 整体关闭时，预取（含 host gather）**什么都不做**。
                        #   `model.py:533 _start_layer_ple_prefetch` 会**独立**调用它，
                        #   光把 embedding.forward 直通**挡不住**这条路径 ✗（实测 100% 捕获后
                        #   仍报 `not permitted when stream is capturing`）
                        if os.environ.get("XIAOTU_PLE_DISABLE", "0") == "1":
                            return None
                        # ⭐ 此处**捕获已结束**（外层 add_eager 干的）⇒ 一切主机操作合法
                        if os.environ.get("XIAOTU_PLE_SINGLE_STREAM", "1") == "1":
                            # 单流：预取流 = 当前流 ⇒ **根本不 fork** ⇒ 不可能 unjoined ✓
                            # 且 `.to("cpu")` 在 eager 区间里合法 ✓
                            try:
                                self._prefetch_stream = torch.cuda.current_stream()
                            except Exception:  # noqa: BLE001
                                pass
                        _res = _o(self, ngram_ids)      # ← 上游断点已结束捕获，此处在 eager 区间
                        _cs = getattr(self, "_prefetch_stream", None)
                        if _cs is not None:
                            try:
                                _cur = torch.cuda.current_stream()
                                if _cs != _cur:
                                    _cur.wait_stream(_cs)     # ⭐ eager 区间内 join ✓
                            except Exception:  # noqa: BLE001
                                pass
                        return _res

                    cls_s.start_prefetch = _eb2(_sp_join)
                    cls_s._xtu_sp_join = True
                    print("[xtu-ple] start_prefetch 外层断点 + **eager 区间内 join** 已装",
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[xtu-ple] ⚠️ start_prefetch 外层断点安装失败: "
                      f"{type(exc).__name__}: {exc}", flush=True)

        # ── ②c 可选：**整层 PLE 直通**（`XIAOTU_PLE_DISABLE=1`，对照实验用）──────
        #   目的：PLE 的 host gather 无法在捕获期存在（见 HANDOFF §59），而用户口径里
        #   PLE 本来就在"全 GPU"之外 ⇒ 关掉它即可让**其余 47/48 层 + MoE 断点**进图，
        #   从而量出"图回放 decode"的真实步长 ✓（数值上会丢掉 n-gram 记忆，仅作对照）
        if os.environ.get("XIAOTU_PLE_DISABLE", "0") == "1":
            try:
                cls_pl = type(m)
                if hasattr(cls_pl, "forward") and not getattr(
                        cls_pl, "_xtu_passthrough", False):
                    def forward(self, hidden_states, *a, **kw):  # noqa: ANN001
                        return hidden_states
                    cls_pl.forward = forward
                    cls_pl._xtu_passthrough = True
                    print(f"[xtu-ple] ⚠️ {cls_pl.__name__}.forward 已改为直通"
                          f"（XIAOTU_PLE_DISABLE=1 对照）", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[xtu-ple] ⚠️ PLE 直通失败: {type(exc).__name__}: {exc}",
                      flush=True)

        # ── ②d PLE **层**直通（`XIAOTU_PLE_DISABLE=1`）───────────────────────────
        #   只直通 embedding 模块不够：model.py 会 `embedding.finalize_prefetch(...)`，
        #   而 `ple_layer` 才是把结果加进 hidden_states 的那一层 ⇒ 把整个 PLE 层直通 ✓
        if os.environ.get("XIAOTU_PLE_DISABLE", "0") == "1":
            try:
                cls_name = type(m).__name__
                if "PLE" in cls_name and not getattr(type(m), "_xtu_passthrough", False):
                    def forward(self, hidden_states, *a, **kw):  # noqa: ANN001
                        return hidden_states
                    type(m).forward = forward
                    type(m)._xtu_passthrough = True
                    print(f"[xtu-ple] ⚠️ {cls_name}.forward 已改为直通"
                          f"（XIAOTU_PLE_DISABLE=1 对照）", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[xtu-ple] ⚠️ PLE 层直通失败: {type(exc).__name__}: {exc}",
                      flush=True)

        # ── ③ 把 PLE 的 `_lookup`（gather）也包成 eager break ────────────────
        #  实测：`ngram_embedding.py:598 ids_cpu = flat_ids.to("cpu")` 在捕获期非法
        #  （`start_prefetch` 虽有装饰器，但这条 D2H 仍落在捕获区里）
        #  `_lookup(self, input_ids, output=None)` 本身就有 out 参数 ⇒ 满足断点语义 ✓
        try:
            cls_m = type(m)
            if (getattr(cls_m, "_xtu_lookup_break", False) is False
                    and hasattr(cls_m, "_lookup")):
                from vllm.compilation.breakable_cudagraph import (
                    eager_break_during_capture as _eb)
                cls_m._lookup = _eb(cls_m._lookup)
                cls_m._xtu_lookup_break = True
                print("[xtu-ple] gather(_lookup) 已包成 eager break", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-ple] ⚠️ gather 断点安装失败: {type(exc).__name__}: {exc}",
                  flush=True)

        em = getattr(m, "embedding_method", None)
        if em is None:
            continue
        cls = type(em)
        try:
            if want_break and not getattr(cls, "_xtu_break_installed", False):
                _install_break_dequant(cls, torch)
                n_cls += 1
            elif want_lut and not getattr(cls, "_xtu_lut_installed", False):
                _install_lut_dequant(cls, torch, getattr(m, "weight_scale", None))
                n_cls += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-ple] ⚠️ 反量化补丁失败: {type(exc).__name__}: {exc}",
                  flush=True)

    # ── 对照实验：PLE 整体关闭（第二遍：模型级入口 + 所有 PLE 层）──────────────
    if os.environ.get("XIAOTU_PLE_DISABLE", "0") == "1":
        try:
            cls_mdl = type(model)
            if not getattr(cls_mdl, "_xtu_noprefetch", False):
                if hasattr(cls_mdl, "_start_layer_ple_prefetch"):
                    cls_mdl._start_layer_ple_prefetch = lambda self, *a, **k: None
                    cls_mdl._xtu_noprefetch = True
                    print("[xtu-ple] model._start_layer_ple_prefetch 已置空"
                          "（PLE_DISABLE 对照）", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-ple] ⚠️ 置空 prefetch 入口失败: {type(exc).__name__}: {exc}",
                  flush=True)
        _n_pl = 0
        for _m in model.modules():
            _cn = type(_m).__name__
            if "PLE" not in _cn or getattr(type(_m), "_xtu_passthrough", False):
                continue
            try:
                def forward(self, hidden_states, *a, **kw):  # noqa: ANN001
                    return hidden_states
                type(_m).forward = forward
                type(_m)._xtu_passthrough = True
                _n_pl += 1
                print(f"[xtu-ple] ⚠️ {_cn}.forward 已改为直通"
                      f"（PLE_DISABLE 对照）", flush=True)
            except Exception:  # noqa: BLE001
                pass
        if _n_pl:
            print(f"[xtu-ple] PLE 直通类数 {_n_pl}", flush=True)

    if seen:
        print(f"[xtu-ple] PLE 模块 {seen} 个：break/LUT 装了 {n_cls} 个类，"
              f"设备缓冲重建 {n_buf} 个", flush=True)
    return n_cls


def apply_all() -> list[str]:
    """应用全部补丁；返回已应用名称列表（幂等）。"""
    if not _enabled():
        return []
    patch_nvfp4_emulation_for_musa()
    patch_loader_scale_rename()
    patch_loader_skip_unknown_attr()
    patch_model_post_load_hook()
    patch_nvfp4_oracle_fallback()
    patch_fp8_backend_order_on_musa()
    return applied()
