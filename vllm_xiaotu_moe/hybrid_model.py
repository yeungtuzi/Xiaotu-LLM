"""vllm-xtu-moe: 主线 DS-V4 的 OOT hybrid 模型覆盖(把 MoE 层挪 CPU 走 xiaotu).

机制(已源码实证,见 dev-docs/ARCHITECTURE.md):
  - vLLM 主线原生支持 DeepSeek-V4(vllm.models.deepseek_v4,nvidia 平台)。
  - ModelRegistry.register_model("DeepseekV4ForCausalLM", "<module>:<class>") 直接覆盖主线该
    arch(registry.py overwrite 明确);main 里 entry-points 把 register() 挂到 vllm.general_plugins。
  - 构造期 OOM 根因:主线把 138GB fp4 MoE 在 GPU 上 materialize(或 offload 只能搬"已构造"权重,
    阻止不了构造期分配)。=> 真解法 = 让每层 ffn 从构造起就持 CPU 权重。
  - 本模块替换 dv4_nvidia.DeepseekV4MoE → CpuXiaotuMoE:
      * gate(GPU,小)+ shared_experts(DeepseekV4MLP,GPU,小)复用主线;
      * routed experts = 自建 CPU 参数模块 CpuMegaExpertsParams,镜像主线 MegaMoE 的原始 fp4
        布局(w13_weight [E,2I,H//2] uint8 / w2_weight [E,H,I//2] uint8 / *_scale raw e8m0 uint8,
        groupK=32)—— 与该 checkpoint 的原始 fp4 及 xiaotu 引擎契约完全一致。
      * 让 self.use_mega_moe=True → 主线的 get_expert_mapping 走 mega mapping,主线的 load_weights
        直接把 checkpoint 原始 fp4 绑定进我们这组 CPU 参数(no GPU 138GB materialize)。
  - 权重加载完成后,process_weights_after_loading -> model.finalize_mega_moe_weights() ->
    layer.ffn.finalize_mega_moe_weights() 这一后置钩子里用这批 CPU 权重构造 xiaotu 引擎
    (数据指针直达,零拷贝)。
  - forward: 路由(GPU fused_topk_bias,含 hash/vision/biased 专家) + engine.cpu_decode(GPU 张量,
    内部 D2H→CPU forward_many→H2D) + 共享专家(GPU)。
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import torch
import torch.nn as nn

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.models.deepseek_v4.nvidia.model import extract_layer_index
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP
from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

# 主线 DS-V4 模型(不 fork、只 OOT 覆盖)
import vllm.models.deepseek_v4.nvidia.model as dv4_nvidia  # noqa: F401
from vllm_xiaotu_moe.gpu_prefill import gpu_prefill_min_tokens  # noqa: E402

_T: dict[str, float] = {"eng": 0.0, "calls": 0, "t0": time.perf_counter()}

MODEL_ARCH = "DeepseekV4ForCausalLM"


def gpu_resident_layers() -> set[int]:
    """常驻 GPU 的专家层(`XIAOTU_MOE_GPU_RESIDENT_LAYERS`,逗号+区间,如 "0-4,10")。

    这些层的专家权重**一次性**放进显存并常驻(不再每步走 CPU 引擎),因此它们
    贡献 0 往返、0 DRAM 权重流量;剩下的层仍在 CPU。对齐 lk-moe 的
    `LVLLM_GPU_RESIDENT_MOE_LAYERS`。默认空 = 全部走 CPU(与旧行为一致)。
    """
    spec = os.environ.get("XIAOTU_MOE_GPU_RESIDENT_LAYERS", "").strip()
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        except ValueError:
            continue
    return out


def _resident_state() -> dict:
    """**进程级**共享状态(挂在 builtins 上)。

    不能只用模块级全局:实测(2026-09-11)插件的 hybrid_model 在同一进程里被加载成
    **两个模块对象**,各自的模块级全局互不相通 —— 结果同一个 prefix 的常驻层被建了
    两份(`[xiaotu] GPU-resident ...` 同 PID 同 prefix 出现两条),13 层吃掉 41 GiB
    ⇒ KV 8 GiB 分配失败 OOM。builtins 是解释器级的,模块复制也共享。
    """
    import builtins

    st = getattr(builtins, "_xiaotu_resident_state", None)
    if st is None:
        st = {"seen": {}, "used_bytes": 0}
        builtins._xiaotu_resident_state = st
    return st


def _instance_index(prefix: str) -> int:
    """同一 prefix 的第几次构造(0 = 目标模型,>=1 = draft/speculator 副本)。"""
    st = _resident_state()
    n = st["seen"].get(prefix, 0)
    st["seen"][prefix] = n + 1
    return n


def resident_already_built(prefix: str, instance: int = 0) -> bool:
    """这个 prefix 的常驻槽位是否已经建过(进程级)。

    插件模块被复制成两份时会各自调用一次 `finalize_mega_moe_weights`,同一个层就会
    建两份常驻显存(实测 13 层 ×2×1.59 GiB ≈ 41 GiB ⇒ KV OOM)。用进程级集合去重,
    比模块级 `self._resident_slot` 可靠。
    """
    st = _resident_state()
    keys = st.setdefault("built", set())
    # 必须带实例号:目标层与 draft 层的 prefix 相同(都叫 model.layers.0…),
    # 只按 prefix 去重会把 draft 的常驻槽位误判成"已建过"而跳过它。
    key = f"{prefix}#{instance}"
    if key in keys:
        return True
    keys.add(key)
    return False


def _spec_decode_requested() -> bool:
    """是否请求了投机解码(仅用于"预留 draft 常驻额度"这一件事)。

    在层构造期拿不到 `vllm_config`,而 `--speculative-config` 一定在 `sys.argv` 里,
    所以直接看命令行;也允许用 `XIAOTU_SPEC_DECODE=0/1` 显式覆盖。
    """
    flag = os.environ.get("XIAOTU_SPEC_DECODE")
    if flag == "1":
        return True
    if flag == "0":
        return False
    return any("speculative" in a for a in sys.argv)


def _draft_layer_count() -> int:
    """DSpark 起草块的层数(本 checkpoint 是 `mtp.0/1/2` ⇒ 3)。"""
    try:
        return max(1, int(os.environ.get("XIAOTU_DRAFT_LAYERS", "3")))
    except ValueError:
        return 3


def register_spec_layer_floor(num_hidden_layers: int | None) -> None:
    """记录**目标模型**的层数(进程级),用来认出 MTP 起草层。

    背景:GLM-5.3-Flash 的 MTP 起草层是 `layers.45`(目标模型 0..44),它在整个
    模型里**只构造一次** ⇒ DSpark 那套"同一 prefix 第二次出现"的判定
    (`_instance_index(prefix) > 0`)**认不出它**,于是它被当普通层、默认落 CPU,
    ⛔ 旧文写「实测 CPU draft 会让投机净负(7.11 vs 10.76 t/s,NOTES §245)」——
    ⭐ **该依据已失效**(2026-10-16):今天同口径实测 draft 落 CPU **30.1 tok/s** vs
    常驻 **15.0 tok/s** vs 不开投机 **18.3 tok/s**(真 agent 载荷/k=4/MBT=8192/TP=2/outlen=256,
    见 EXPERIMENTS **B840**)⇒ **默认改为"落 CPU"** ✓,只有显式 `=1` 才强制常驻。

    这里记下目标模型的 `num_hidden_layers`(取值取**最大**:起草模型的 config 可能
    把 `num_hidden_layers` 设成 0,见 MiMo MTP),任何 index ≥ 它的层就是起草层。
    """
    import builtins

    try:
        n = int(num_hidden_layers or 0)
    except (TypeError, ValueError):
        return
    if n <= 0:
        return
    cur = getattr(builtins, "_xiaotu_spec_layer_floor", None)
    if cur is None or n > cur:
        builtins._xiaotu_spec_layer_floor = n


def spec_layer_floor() -> int | None:
    """目标模型的层数(未捕获时 None)。"""
    return getattr(__import__("builtins"), "_xiaotu_spec_layer_floor", None)


def is_spec_draft_layer_index(layer_idx: int) -> bool:
    """`layer_idx` 是否落在目标模型之外(⇒ MTP/speculator 起草层)。

    只在**请求了投机解码**时有意义:不开投机时根本不会构造起草层,这里恒 False,
    行为逐字不变。GLM-5.3-Flash 是 `idx >= 45`;MiMo-V2.5 的 `model.mtp.layers.*`
    也会被这个下界覆盖(它的起草模块层号同样是接在目标层之后)。
    """
    if not _spec_decode_requested():
        return False
    floor = spec_layer_floor()
    if floor is None:
        return False
    try:
        return int(layer_idx) >= int(floor)
    except (TypeError, ValueError):
        return False


def reserve_draft_bytes(per_layer_bytes: int) -> None:
    """把 draft 层要占的显存**先从预算里预留出来**(只做一次)。

    构造顺序是"目标模型在前、draft 在后"(`LLMBaseProposer.load_model()`),若不预留,
    `XIAOTU_MOE_RESIDENT_BUDGET_GB` 会被 42 层目标层吃光,draft 只能落回 CPU ——
    ⛔ 旧文写「而实测 draft 走 CPU 时投机是**净负**的(7.11 vs 不开投机 10.76 t/s)」——
    ⭐ **已失效**(2026-10-16):今天同口径(draft 落 CPU 30.1 / 常驻 15.0 / 不开投机 18.3 tok/s,
    EXPERIMENTS B840)⇒ **落 CPU 反而最优**;7.11/10.76 两数**都低于今天的不开投机基线** ⇒
    必然来自更早更慢的配置。⇒ **默认已改为"落 CPU"** ✓。
    因此按用户要求:**draft 永远在 GPU**,预算先扣掉它的份额。
    """
    st = _resident_state()
    if st.get("draft_reserved"):
        return
    st["draft_reserved"] = True
    if not _spec_decode_requested():
        st["reserved_bytes"] = 0
        return
    st["reserved_bytes"] = per_layer_bytes * _draft_layer_count()
    print(
        f"[xiaotu] reserved {st['reserved_bytes'] / 2**30:.2f} GiB for "
        f"{_draft_layer_count()} GPU-resident draft layer(s)",
        flush=True,
    )


def resident_budget_ok(nbytes: int, mandatory: bool = False) -> bool:
    """本次常驻是否还在显存预算内(`XIAOTU_MOE_RESIDENT_BUDGET_GB`,0=不限)。

    `XIAOTU_MOE_RESIDENT_BUDGET_GB` 是控制**目标模型**常驻层数的硬旋钮(直接对应
    "塞进 40 GiB")。`mandatory=True` 表示这是 **draft 层**:按用户要求 draft 永远在
    GPU,所以不受预算限制(它很小:3 层 ≈ 4.8 GiB/rank),预算是留给目标层的。
    """
    gb = float(os.environ.get("XIAOTU_MOE_RESIDENT_BUDGET_GB", "0") or 0)
    st = _resident_state()
    if mandatory:
        st["used_bytes"] += nbytes
        return True
    if gb <= 0:
        return True
    reserved = int(st.get("reserved_bytes", 0))
    if st["used_bytes"] + reserved + nbytes > int(gb * (1 << 30)):
        return False
    st["used_bytes"] += nbytes
    return True


def _is_duplicate_model_instance(prefix: str) -> bool:
    """同一 MoE 层 prefix 是否**第二次**被构造(⇒ 是 draft/speculator 模型)。

    vLLM 的 DSpark 起草模型是目标模型的又一份完整实例,层名(prefix)相同;插件
    的模块构造顺序是"目标模型在前、draft 在后"(`LLMBaseProposer.load_model()`
    在目标模型加载完成后调用)。用它把 draft 排除在常驻层之外,省下的显存换更多
    目标模型常驻层(预填充的关键)。同一进程内每个模型实例只构造一次。
    """
    st = _resident_state()
    n = st["seen"].get(prefix, 0)
    st["seen"][prefix] = n + 1
    return n > 0


def ep_shm_enabled() -> bool:
    """EP 部分和改走 /dev/shm 归约(默认开;`XIAOTU_MOE_EP_SHM=0` 关回 NCCL)。

    两个 rank 在同一台机器上,跨 rank 求和用共享内存只需一次 memcpy + 两个自旋
    barrier(几十 µs),而 GPU 的 NCCL all-reduce 实测每层要 +4~6.5 ms。
    """
    return os.environ.get("XIAOTU_MOE_EP_SHM", "1") != "0"


_EP_SHM_FDS: list = []   # 保持 mmap/fd 存活


# 【第 193 轮·修 draft 计算 bug】shm 文件名必须区分**模型实例**。
# 背景:vLLM 的 DSpark 起草模型是目标模型的**又一份完整实例,层名(prefix)相同**
# ⇒ 只用 layer_idx 命名会让"目标第 L 层"和"draft 第 L 层"共用同一个 barrier 头与
# 同一块部分和区域 ⇒ 两套调用序列交织推进世代、互相覆写 ⇒ **draft 的 MoE 算错**
# ⇒ 接受率塌陷(实测 position-0: 26% vs 修好后的 72.5%,见 NOTES §236/§238)。
# 这里用一个**独立**的计数(不碰 _is_duplicate_model_instance 的 seen,避免互相干扰)。
_SHM_INSTANCE: dict = {"seen": {}}


def _shm_instance_tag(prefix: str) -> str:
    n = _SHM_INSTANCE["seen"].get(prefix, 0)
    _SHM_INSTANCE["seen"][prefix] = n + 1
    return "T" if n == 0 else f"D{n}"


def _ep_shm_attach(layer_idx: int, tokens: int, hidden: int, world: int,
                   instance_tag: str = "T"):
    """为某一层创建/打开跨 rank 共享的归约缓冲,返回 (mmap, stride_bytes)。

    布局:[Header 128B][rank0 部分和 stride B][rank1 部分和 stride B ...]
    Header 的前 64B 是两个 atomic(arrive/gen)与 read_done/gen2,由引擎侧解释;
    这里只需要保证两个 rank 打开**同名文件**且大小一致。
    """
    import mmap

    stride = (tokens * hidden * 4 + 63) // 64 * 64
    total = 128 + stride * world
    path = (f"/dev/shm/xiaotu_ep_L{layer_idx}_{instance_tag}"
            f"_{hidden}_{tokens}_{world}.bin")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, total)
        mm = mmap.mmap(fd, total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        pass
    _EP_SHM_FDS.append((fd, mm, path))
    return mm, stride


_EP_REQUESTED: "bool | None" = None


def ep_enabled() -> bool:
    """Expert-parallel CPU decode for TP>1 (default on; `XIAOTU_MOE_EP=0` off).

    Without it every rank runs the *full* CPU expert set on the same tokens
    (redundant 2x work at TP=2), which is why TP=2 decode used to be slower than
    single-card. With it each rank owns `E/tp` experts, masks the routing pairs
    that belong to another rank (weight 0 -> the engine skips them, see
    `forward_many`'s `weights[ai] != 0.f` filter), and the partial MoE outputs are
    summed by one TP all-reduce.

    `vllm --enable-expert-parallel` sets the module-level `_EP_REQUESTED=True`
    from `CpuXiaotuMoE.__init__` (see `[xtu-ep]` log); `XIAOTU_MOE_REDUNDANT=1`
    still forces the old redundant behaviour.
    """
    if _EP_REQUESTED is not None:
        return _EP_REQUESTED
    if os.environ.get("XIAOTU_MOE_REDUNDANT", "0") == "1":
        return False
    return os.environ.get("XIAOTU_MOE_EP", "1") != "0"

# Layer-index -> MoE module, so layer L can prefetch layer L+1's GPU weights.
_LAYERS: dict[int, "CpuXiaotuMoE"] = {}
_PREBUILD_STARTED = False


_PROF_STATE: dict = {"prof": None, "calls": 0}


def _capture_guard() -> bool:
    """True while vLLM is capturing a CUDA graph.

    Every debug/profiling helper must bail out here: anything that synchronizes
    the stream (``torch.profiler.__enter__``, ``.item()``, ``.cpu()``) inside a
    capture region raises ``cudaErrorStreamCaptureUnsupported`` and invalidates
    the whole capture, which aborts engine startup. Costing a 13-minute reload to
    learn that is not worth it (TRIED_AND_REVERTED R101).
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _maybe_profile() -> None:
    """Env-gated torch profiler around the first N GPU-prefill layers.

    Used to decompose the long-prefill TTFT into MoE vs attention/indexer
    kernels (the model runs in vLLM's worker process, so an external nsys
    session does not see its kernels). Set XIAOTU_TORCH_PROFILE=<path> to dump
    a chrome trace plus a per-kernel table; XIAOTU_TORCH_PROFILE_CALLS=N
    (default 86 = two passes) controls the window.
    """
    path = os.environ.get("XIAOTU_TORCH_PROFILE")
    if not path:
        return
    if _capture_guard():
        return
    st = _PROF_STATE
    if st["prof"] is None:
        import torch.profiler as tp

        st["prof"] = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]
        )
        st["prof"].__enter__()
    st["calls"] += 1
    limit = int(os.environ.get("XIAOTU_TORCH_PROFILE_CALLS", "86"))
    if st["calls"] >= limit:
        prof = st["prof"]
        st["prof"] = None
        prof.__exit__(None, None, None)
        print("[xiaotu-profile]\n" + prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=40), flush=True)
        prof.export_chrome_trace(f"{path}.r{os.getpid()}")
        print(f"[xiaotu-profile] chrome trace -> {path}.r{os.getpid()}", flush=True)


_PROF_DEC: dict = {"prof": None, "calls": 0}


def _maybe_profile_decode() -> None:
    """Env-gated torch profiler for the DECODE path.

    XIAOTU_TORCH_PROFILE_DECODE=<path> dump(默认抓 2×43 层)――decode 的
    step 由 43 次 D2H→host 回调→H2D + GPU 注意力组成,只有拿到进程内 trace
    才能看清"每层 6 ms 里有多少是内核、多少是等 CPU"。
    """
    path = os.environ.get("XIAOTU_TORCH_PROFILE_DECODE")
    if not path:
        return
    if _capture_guard():          # 捕获期启动 profiler 会作废捕获(见 R101)
        return
    st = _PROF_DEC
    if st["prof"] is None:
        import torch.profiler as tp

        st["prof"] = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]
        )
        st["prof"].__enter__()
    st["calls"] += 1
    limit = int(os.environ.get("XIAOTU_TORCH_PROFILE_DECODE_CALLS", "86"))
    if st["calls"] >= limit:
        prof = st["prof"]
        st["prof"] = None
        prof.__exit__(None, None, None)
        print("[xiaotu-dec-profile]\n" + prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30), flush=True)
        prof.export_chrome_trace(f"{path}.r{os.getpid()}")
        print(f"[xiaotu-dec-profile] chrome trace -> {path}.r{os.getpid()}", flush=True)


def _start_pinned_prebuild() -> None:
    """Build every layer's K-major pinned host cache in the background (once).

    Without this the first long prefill pays ~1 s per layer synchronously inside
    its forward (43 layers). The prebuild runs on a daemon thread while the first
    layers' kernels execute, so steady state is reached within the first few
    layers.
    """
    global _PREBUILD_STARTED
    if _PREBUILD_STARTED:
        return
    _PREBUILD_STARTED = True
    try:
        from vllm_xiaotu_moe.gpu_prefill import prebuild_pinned_kmajor

        quads = []
        for m in list(_LAYERS.values()):
            try:
                w13, s13, w2, s2, _, _, _ = m._gpu_shard()
                quads.append((w13, s13, w2, s2))
            except Exception:  # noqa: BLE001
                pass

        def _runner():
            # 阶段 1(纯 CPU 转置)立刻开始:warmup 期间就把转置做完,否则服务路径
            # 会单线程逐层转置+锁页,启动拖成好几分钟。阶段 2(cudaHostRegister)
            # 由 prebuild_pinned_kmajor 自己等 CUDA graph 捕获静止后再做
            # (捕获期间调用会作废捕获甚至让驱动 segfault,见 TRIED_AND_REVERTED R4)。
            prebuild_pinned_kmajor(quads, workers=6)

        threading.Thread(target=_runner, daemon=True).start()
    except Exception as e:  # noqa: BLE001
        print(f"[xiaotu] pinned prebuild not started: {type(e).__name__}: {e}",
              flush=True)


def _map_global_expert_id(expert_id: int) -> list[int]:
    # 单 rank、无 EPLB:全局专家 id == 本进程参数里的行号(我们持有全部 256 个)。
    return [expert_id]


class CpuMegaExpertsParams(nn.Module):
    """镜像主线 DeepseekV4MegaMoEExperts 的原始 fp4 参数布局,但分配在 CPU。

    形状/数据类型与该 checkpoint 的原始 fp4(及 xiaotu MOE_MXFP4 契约)完全一致:
      w13_weight        [E, 2I, H//2]  uint8
      w13_weight_scale  [E, 2I, H//32] uint8(raw e8m0,groupK=32)
      w2_weight         [E, H, I//2]   uint8
      w2_weight_scale   [E, H, I//32]  uint8
    weight_loader 语义与主线 mega 版一致(TP=1/EP=1:per-expert w1/w3 分半写入 w13)。
    关键:分配在 CPU => 模型构造不再在 GPU 上 materialize 138GB => 解决构造期 OOM。
    """

    def __init__(self, num_experts, hidden_size, intermediate_size, device="cpu",
                 ep_start=0, ep_local=None, ep_global=None,
                 group_k: int = 32, nvfp4: bool = False):
        super().__init__()
        # ⭐ 2026-10-26 NVFP4 W4A16（`XIAOTU_NVFP4=1` 开启；默认关闭 ⇒ MXFP4 路径零回归）
        #   差异只有三处：① 组尺度 groupK 32→**16**；② 尺度字节 e8m0→**e4m3**（同为 1 字节，
        #   形状只改除数）；③ 多一组**每张量**二级尺度 `weight_scale_2`（fp32）。
        self._group_k = int(group_k)
        self._nvfp4 = bool(nvfp4)
        # 【NOTES §365】EP:只分配**本 rank 的专家分片**。
        # 主线原先按 config.n_routed_experts(=256) 全量分配 ⇒ 每 rank 138 GB,
        # 而 fork 用 `moe_config.num_local_experts`(=256/TP) ⇒ 只要一半。
        # 同机实测:fork+我们的引擎 105 GB/worker vs 主线+我们的插件 260 GB/worker。
        self.ep_start = int(ep_start)
        self.ep_local = int(ep_local if ep_local is not None else num_experts)
        self.ep_global = int(ep_global if ep_global is not None else num_experts)
        num_experts = self.ep_local          # 下面的 zeros 全部按本地数分配
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        weight_attrs = {"weight_loader": self.weight_loader}
        kw = dict(device=device, dtype=torch.uint8)
        self.w13_weight = nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, hidden_size // 2, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, hidden_size // self._group_k, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"
        self.w2_weight = nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size // 2, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size // self._group_k, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        # ⭐ NVFP4 二级尺度（每张量 per-tensor，fp32）。仅 `nvfp4=True` 时创建。
        #   w13 由 gate(w1)/up(w3) **融成一路**，而 checkpoint 里两者 scale_2 **不同**
        #   （实测抽样 400 组有 395 组不同）⇒ 必须分开存，见引擎构造处的折叠说明。
        if self._nvfp4:
            _kw2 = dict(device=device, dtype=torch.float32)
            self.w13_weight_scale_2 = nn.Parameter(
                torch.zeros(num_experts, 2, **_kw2), requires_grad=False)
            self.w2_weight_scale_2 = nn.Parameter(
                torch.zeros(num_experts, **_kw2), requires_grad=False)
            for _p in (self.w13_weight_scale_2, self.w2_weight_scale_2):
                set_weight_attrs(_p, {"weight_loader": self.weight_scale_2_loader})

        # ⭐ 2026-10-26：`qwen4_exp` 的 AutoWeightsLoader 按**逐专家模块**解析
        #   `mlp.experts.<i>.{gate,up,down}_proj.weight[_scale[_2]]` ⇒ 我们的**融合**参数
        #   无法被它定位（实测 `There is no module or parameter named 'layers.0.mlp.experts.0'`）。
        #   ⇒ 这里给每个本地专家建一组**视图 Parameter**（与融合张量共享存储），
        #   加载器写视图 == 写融合张量，**零拷贝**，也不需要改 vLLM ✓
        # ⭐ 2026-10-26【A78-3】原本这里给每个本地专家建 `experts.<j>.<proj>.<attr>` 子模块，
        #   因为 `qwen4_exp` 的 `AutoWeightsLoader` 按**模块路径**解析。
        #   ⚠️ 但那个做法有个硬伤：加载器会把**整个 checkpoint** 的专家键都喂给每个 rank，
        #   而本 rank 只持 `ep_local` 个（TP=2 时 rank0=0–255 / rank1=256–511）
        #   ⇒ 一遇到 `experts.256` 就报 `There is no module or parameter named` ✗
        #   ⇒ 改为在 `load_weights()` 里**自己解析 id**（见下面的方法），
        #     非本地专家按 EP 语义**静默跳过**（那是正常分片，不是丢权重）。
        #   本开关仅保留给排障：把视图也建出来（会让每层多 ~3k 个 Module/Parameter）。
        if os.environ.get("XIAOTU_PER_EXPERT_VIEWS", "0").strip() == "1":
            self._xt_build_per_expert_views()

    # ---- `AutoWeightsLoader` 的 per-module 钩子（A78-3）------------------------
    _XT_PROJ_SHARD = {
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
    }

    def load_weights(self, weights):
        """自己消化 `experts.<gid>.<proj>.<attr>`（`_groupby_prefix` 已剥掉 `experts.` 前缀）。

        ⭐ 收益：① 非本 rank 的专家 id 直接跳过（EP 正常语义）⇒ 不再需要逐专家子模块
        （每层少 ~3k 个 Module/Parameter 对象）；② 名称→融合张量的映射集中在一处，
        不会再出现"尺度视图写进 weight 张量"那类错配（A78-2）。
        """
        loaded: list[str] = []
        for name, tensor in weights:
            parts = name.split(".")
            if len(parts) < 3:
                continue
            try:
                geid = int(parts[0])
            except ValueError:      # 不是专家键（不该发生）；交给别的钩子
                continue
            proj, attr = parts[1], ".".join(parts[2:])
            shard = self._XT_PROJ_SHARD.get(proj)
            if shard is None:
                continue
            if shard in ("w1", "w3"):
                w_name, s_name = "w13_weight", "w13_weight_scale"
            else:
                w_name, s_name = "w2_weight", "w2_weight_scale"
            if attr == "weight":
                self.weight_loader(getattr(self, w_name), tensor, w_name, shard, geid)
            elif attr == "weight_scale":
                self.weight_loader(getattr(self, s_name), tensor, s_name, shard, geid)
            elif attr == "weight_scale_2":
                # 每专家全局尺度：gate/up 各占半格，down 一格（见 weight_scale_2_loader）
                if shard in ("w1", "w3"):
                    self.weight_scale_2_loader(
                        self.w13_weight_scale_2, tensor, "w13_weight_scale_2", shard, geid)
                else:
                    self.weight_scale_2_loader(
                        self.w2_weight_scale_2, tensor, "w2_weight_scale_2", shard, geid)
            elif attr == "input_scale":
                pass                # 引擎不用（NVFP4 只有 per-tensor global，已折进组尺度）
            else:
                print(f"[xtu-nvfp4] 未知专家键（已忽略）: {name}", flush=True)
                continue
            loaded.append(name)
        return loaded

    def _xt_build_per_expert_views(self) -> None:
        """给 `qwen4_exp` 的 AutoWeightsLoader 建**逐专家子模块**（名为 "0","1",…）。

        ⭐ 关键：这些子模块的 Parameter 是融合张量的**视图**（共享存储），
        且每个 Parameter 都挂上**转交给我们自己的 `weight_loader`** 的加载函数
        ⇒ TP 分片/专家映射/形状校验全部复用已验证的 `CpuMegaExpertsParams.weight_loader`，
        加载器写进来的数据直接落进融合张量（`w13_weight`/`w2_weight`/尺度） ✓
        """
        import torch as _torch
        import torch.nn as _nn

        E = int(self.w13_weight.shape[0])
        I = int(self.intermediate_size)

        def _mk_fused(wid: str, shard: str, geid: int):
            def _ld(param, loaded_weight, weight_name="", *a, **k):
                return self.weight_loader(getattr(self, wid), loaded_weight,
                                          wid, shard, geid)
            return _ld

        def _mk_copy():
            def _ld(param, loaded_weight, weight_name="", *a, **k):
                with _torch.no_grad():
                    param.data.copy_(loaded_weight.to(param.dtype).reshape(param.shape))
                return True
            return _ld

        for lid in range(E):
            geid = int(getattr(self, "ep_start", 0)) + lid
            em = _nn.ModuleDict()
            spec = (
                ("gate_proj", "w13_weight", "w1", self.w13_weight.data[lid, :I],
                 self.w13_weight_scale.data[lid, :I],
                 self.w13_weight_scale_2.data[lid, 0] if self._nvfp4 else None),
                ("up_proj", "w13_weight", "w3", self.w13_weight.data[lid, I:2 * I],
                 self.w13_weight_scale.data[lid, I:2 * I],
                 self.w13_weight_scale_2.data[lid, 1] if self._nvfp4 else None),
                ("down_proj", "w2_weight", "w2", self.w2_weight.data[lid],
                 self.w2_weight_scale.data[lid],
                 self.w2_weight_scale_2.data[lid] if self._nvfp4 else None),
            )
            for nm, wid, shard, w, ws, gs in spec:
                m = _nn.Module()
                m.weight = _nn.Parameter(w, requires_grad=False)
                m.weight.weight_loader = _mk_fused(wid, shard, geid)
                m.weight_scale = _nn.Parameter(ws, requires_grad=False)
                # ⭐ 2026-10-26【必须分开转交】尺度视图要交给**尺度**那张融合张量：
                #   若沿用 weight 的 `wid`，`weight_loader` 会去写 `w2_weight`/`w13_weight`
                #   ⇒ 实测报 `param (2560, 320) vs ckpt (2560, 40)`（320 是权重宽、40 才是
                #   H/I 组的尺度宽）✗。名字里必须仍带 `w13_`/`w2_`（shard 判据靠它）。
                m.weight_scale.weight_loader = _mk_fused(wid + "_scale", shard, geid)
                if gs is not None:
                    m.weight_scale_2 = _nn.Parameter(gs, requires_grad=False)
                    m.weight_scale_2.weight_loader = _mk_copy()
                # checkpoint 里还有 input_scale（标量）：我们的引擎用不到，
                # 但要有个地方接住它，否则加载器按路径解析会报 "no module named" ✗
                m.input_scale = _nn.Parameter(
                    _torch.zeros((), dtype=_torch.float32), requires_grad=False)
                m.input_scale.weight_loader = _mk_copy()
                em[nm] = m
            # ⭐ 直接挂成名为 "0"/"1"… 的**子模块** ⇒ 路径正是 experts.<i>.<proj>.<attr> ✓
            self.add_module(str(lid), em)

    def weight_scale_2_loader(self, param, loaded_weight, weight_name,
                              shard_id, expert_id, return_success=False):
        """NVFP4 的 `*_weight_scale_2`（每张量标量）→ 按专家、按 gate/up 半边写入。"""
        try:
            _eid = int(expert_id)
        except Exception:  # noqa: BLE001
            return False if return_success else None
        _lid = _eid - self.ep_start
        if not (0 <= _lid < self.ep_local):
            return False if return_success else None
        _v = loaded_weight.to(torch.float32).reshape(())
        if "w13_" in weight_name:
            param.data[_lid, 0 if shard_id == "w1" else 1] = _v
        elif "w2_" in weight_name:
            param.data[_lid] = _v
        else:
            return False if return_success else None
        return True if return_success else None

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        # global -> local(§365):不属于本 rank 的专家直接跳过(不写、不占内存)。
        try:
            _eid = int(expert_id)
        except Exception:  # noqa: BLE001 - 0-dim tensor / numpy 标量
            _eid = expert_id
        if isinstance(_eid, int):
            _lid = _eid - self.ep_start
            local_ids = [_lid] if 0 <= _lid < self.ep_local else []
        else:
            local_ids = _map_global_expert_id(expert_id)
        # ⭐ 2026-10-26【静默错数守卫】NVFP4 的组尺度在 checkpoint 里是 **float8_e4m3fn**，
        #   而我们的尺度参数是 **uint8**（引擎按**原始字节**当 e4m3 读，见构造引擎处
        #   `s13.view(torch.float8_e4m3fn)`）。`copy_` 遇到不同 dtype 会做**数值转换**：
        #   实测 fp8 [0.5,0.75,1.5,6.0] → uint8 [0,0,1,6] ✗ ⇒ 全部尺度被静默毁掉，
        #   表现为"能跑但输出全错"。正确做法是**位拷贝**（view 成 uint8 再 copy_）。
        if (
            param.dtype == torch.uint8
            and loaded_weight.dtype != torch.uint8
            and loaded_weight.element_size() == 1
        ):
            loaded_weight = loaded_weight.view(torch.uint8)
        loaded_any = False
        for local_expert_id in local_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                shard_offset = 0 if shard_id == "w1" else self.intermediate_size
                expert_data = expert_data.narrow(
                    0, shard_offset, self.intermediate_size
                )
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")
            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"[xiaotu] expert weight shape mismatch {weight_name}: "
                    f"param {tuple(expert_data.shape)} vs ckpt {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True
        if return_success:
            return loaded_any
        return None

    def update_expert_map(self):
        # 无 EPLB,no-op(与主线 FusedMoEFactory 同名钩子对齐)。
        return None

    def finalize_weights(self, shared_experts=None):
        # 无 DeepGEMM 融合,no-op(引擎由 CpuXiaotuMoE.finalize_mega_moe_weights 构造)。
        return None


CpuMegaExpertsParams.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class CpuXiaotuMoE(nn.Module):
    """把一层 DS-V4 的 routed MoE 挪到 CPU 用 xiaotu 引擎算;gate/共享专家留在 GPU。

    - 该模块即"该层 ffn"的 CPU 权重持有者:routed 专家参数在 CPU(不占 GPU,解决 138GB OOM)。
    - finalize_mega_moe_weights()(主线的模型后置加载钩子)里构造 xiaotu 引擎。
    - forward: 路由(GPU)+ engine.cpu_decode + 共享专家(GPU)。
    """

    def __init__(
        self,
        vllm_config,
        prefix: str = "",
        use_sequence_parallel: bool = False,
        *,
        num_hash_layers: int | None = None,
        n_routed_experts: int | None = None,
        n_activated_experts: int | None = None,
        image_sentinel_lo: int | None = None,
        **kwargs,   # ⭐ 2026-10-26：兼容 `Qwen4ExpSparseMoeBlock` 的额外关键字
                    #   （实测会传 `reduce_results`）——我们内部自己处理归约，
                    #   这里只接受并忽略，保持签名可替换 ✓
    ):
        # Upstream's `DeepseekV4MoE` grew these keyword-only args (mainline
        # dabc4362b, DS-V4 hash routing). Accept them and fall back to the
        # config, so the plugin still works on the older tree
        # (<= 6c73b08dec) that constructed us positionally.
        super().__init__()
        # ⭐ 2026-10-26：`qwen4_exp` 的模型配置是**外层壳**，真正的超参在 `hf_text_config`
        #   （实测 `Qwen4ExpConfig` 没有 `hidden_size` ⇒ AttributeError）。
        #   vLLM 自己的 qwen4_exp 代码也用 `hf_text_config` ⇒ 我们跟随 ✓
        _mc = vllm_config.model_config
        config = getattr(_mc, "hf_text_config", None) or _mc.hf_config
        # ⭐ 2026-10-26：不同模型族的**字段名不同**（DS-V4 用 n_routed_experts/n_shared_experts，
        #   qwen4_exp 用 num_experts/（无 shared 计数））⇒ 这里统一成 DS-V4 的别名，
        #   后面代码只认一套名字，避免为每个模型族写分支 ✓
        if getattr(config, "n_routed_experts", None) is None:
            try:
                config.n_routed_experts = int(getattr(config, "num_experts"))
            except Exception:  # noqa: BLE001
                pass
        if getattr(config, "n_shared_experts", None) is None:
            # qwen4_exp 只给 `shared_expert_intermediate_size`，专家数按 1 处理 ✓
            config.n_shared_experts = 1 if getattr(
                config, "shared_expert_intermediate_size", None) else 0
        if getattr(config, "num_experts_per_tok", None) and not hasattr(
                config, "num_experts_per_tok_alias"):
            config.num_experts_per_tok_alias = config.num_experts_per_tok
        self._vllm_config = vllm_config
        # ⭐ 2026-10-19:把 vLLM 的 --enable-expert-parallel 接到 CPU 专家分片。
        #   CPU 引擎本来就默认按 TP 分片;此前这个 flag 不影响插件 ⇒ 用户无法从
        #   服务参数上确认"EP 是否真的开了"。这里显式记录并强制打开。
        try:
            _pc = getattr(vllm_config, "parallel_config", None)
            if _pc is not None and bool(getattr(_pc, "enable_expert_parallel", False)):
                global _EP_REQUESTED
                _EP_REQUESTED = True
                print("[xtu-ep] vLLM enable_expert_parallel=True -> CPU expert sharding ON "
                      f"(tp={getattr(_pc, 'tensor_parallel_size', '?')}, "
                      f"ep={getattr(_pc, 'expert_parallel_size', '?')})", flush=True)
        except Exception:  # noqa: BLE001
            pass
        self.prefix = prefix
        self.use_sequence_parallel = bool(use_sequence_parallel)
        # 关键:让主线的 get_expert_mapping() 走 mega mapping(绑定原始 fp4 进下面的 CPU 参数)。
        self.use_mega_moe = True
        self.use_fi_mega_moe = False
        self.tp_size = 1

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.n_routed_experts = (
            config.n_routed_experts if n_routed_experts is None else n_routed_experts
        )
        self.n_activated_experts = (
            config.num_experts_per_tok
            if n_activated_experts is None
            else n_activated_experts
        )
        self.top_k = self.n_activated_experts
        self.num_hash_layers = (
            getattr(config, "num_hash_layers", 0)
            if num_hash_layers is None
            else num_hash_layers
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        # ⭐⭐⭐【B971 · E 项修复】GLM 的 `swiglu_limit` 在 `text_config` 里(顶层 None)✗
        #   B903 只修了 mixed_experts.py ⇒ 经本文件的路径(MTP draft)仍漏掉 clamp✗
        #   修法同 B903:仅在顶层取不到时回退到 text_config ⇒ 对已能取到的模型零影响✓
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        if self.swiglu_limit is None:
            _tc = getattr(config, "text_config", None)
            if _tc is not None:
                self.swiglu_limit = getattr(_tc, "swiglu_limit", None)
        # ⭐ B972 临时诊断(1 行,验证完即删):config 到底是哪一层?
        if os.environ.get("XIAOTU_CFG_DIAG", "0") == "1":
            _t = getattr(config, "text_config", None)
            print("[xtu-cfg] type=%s has_text_config=%s swiglu_limit=%r routed_scaling=%r "
                  "scoring_func=%r norm_topk_prob=%r | text_config: scaling=%r scoring=%r"
                  % (type(config).__name__, _t is not None, getattr(config, "swiglu_limit", "<NA>"),
                     getattr(config, "routed_scaling_factor", "<NA>"),
                     getattr(config, "scoring_func", "<NA>"),
                     getattr(config, "norm_topk_prob", "<NA>"),
                     getattr(_t, "routed_scaling_factor", "<NA>") if _t else "<NA>",
                     getattr(_t, "scoring_func", "<NA>") if _t else "<NA>"), flush=True)
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        self.n_shared_experts = config.n_shared_experts or 0
        # ---- EP 存储分片(§365)----------------------------------------------
        # 默认跟 fork 一样:每个 rank **只分配/加载** n_routed_experts/TP 个专家。
        # XIAOTU_MOE_EP_SHARD_STORAGE=0 可回到旧的"每 rank 持全部"行为(排障用)。
        self._shard_storage = os.environ.get(
            "XIAOTU_MOE_EP_SHARD_STORAGE", "1") != "0"
        _tp, _rk = 1, 0
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            _tp = int(get_tensor_model_parallel_world_size())
            _rk = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            _tp, _rk = 1, 0
        if not self._shard_storage:
            _tp, _rk = 1, 0
        self._ep_tp, self._ep_rank = max(1, _tp), max(0, _rk)
        self._stor_local = self.n_routed_experts // self._ep_tp
        self._stor_start = self._ep_rank * self._stor_local
        self.n_local_experts = self._stor_local
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_routed_experts  # 无 redundant
        self.n_local_physical_experts = self._stor_local
        self.n_redundant_experts = 0
        self.experts_start_idx = self._stor_start
        self.experts_end_idx = self._stor_start + self._stor_local
        self.hash_indices_dtype = torch.int64  # mega 语义
        self.image_sentinel_lo = (
            (IMAGE_SENTINEL_BASE_ID if image_sentinel_lo is None else image_sentinel_lo)
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )

        # ---- gate(GPU,小)— 与主线一致(含 hash / biased 专家) ----
        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        is_hash_moe = extract_layer_index(prefix) < self.num_hash_layers
        if is_hash_moe:
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    self.n_routed_experts,
                    (config.vocab_size, self.n_activated_experts),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        if getattr(config, "topk_method", None) == "noaux_tc" and not is_hash_moe:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        # ---- 共享专家(GPU,小) ----
        # ⭐ A78-7（2026-10-26 实测 checkpoint）：qwen4_exp 族的共享专家键是
        #   `mlp.shared_expert.{gate,up,down}_proj.weight` **外加 `mlp.shared_expert_gate.weight`**，
        #   语义 = `out = routed(x) + sigmoid(shared_expert_gate(x)) * shared(x)`
        #   ⇒ 必须用上游带 `expert_gate` 的 Qwen2MoeMLP（即 Qwen3NextMLP）；
        #     用 DS4 的 `DeepseekV4MLP` 会**丢掉这道 sigmoid 门** ⇒ 静默错数 ✗
        #   ⚠️ 该族共享专家是 BF16（`config.quantized_layers` 只标了 `...mlp.experts`）
        #      ⇒ quant_config 必须 None，否则会去找不存在的 weight_scale/weight_scale_2 ✗
        #   ⚠️ 名字差异：checkpoint 是**未融合**的 gate_proj/up_proj，模块里是融合的
        #      `gate_up_proj` ⇒ 由本类的 `load_weights` 显式折叠（上游 AutoWeightsLoader
        #      只按子模块名找，必然报 `There is no module ... shared_expert` ✗）
        _shared_is = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        self.shared_expert = None
        self.shared_expert_gate = None
        self.shared_experts = None
        if _shared_is > 0:
            from vllm.model_executor.layers.linear import ReplicatedLinear
            from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )
            # reduce_results=True：routed 由 forward 里单独 all-reduce（EP），
            # 共享专家在这里自归约 —— 与主线(及 DS4 分支)的组合方式一致。
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=_shared_is,
                hidden_act=config.hidden_act,
                quant_config=None,
                reduce_results=True,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        elif self.n_shared_experts:
            intermediate_size = (
                config.moe_intermediate_size * self.n_shared_experts
            )
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=vllm_config.quant_config,
                reduce_results=True,
                is_sequence_parallel=False,
                prefix=f"{prefix}.shared_experts",
            )

        # ---- routed experts:CPU 参数(主线的 load_weights 会把原始 fp4 绑进来) ----
        # ⭐ NVFP4 检测：`XIAOTU_NVFP4=1`（launcher 显式给）**或** 量化配置写着 nvfp4/modelopt
        _qcfg = getattr(config, "quantization_config", None) or {}
        _is_nvfp4 = (os.environ.get("XIAOTU_NVFP4", "0").strip() == "1") or (
            str(_qcfg.get("quant_algo", "")).lower().find("nvfp4") >= 0
            or str(_qcfg.get("quant_method", "")).lower() == "modelopt"
            and "nvfp4" in str(_qcfg).lower()
        )
        self._xt_nvfp4 = bool(_is_nvfp4)
        self._xt_group_k = 16 if self._xt_nvfp4 else 32
        if self._xt_nvfp4:
            print(f"[xtu-nvfp4] 本层按 NVFP4-W4A16 处理: group_k=16, 尺度 e4m3, "
                  f"+ weight_scale_2（prefix={prefix}）", flush=True)
        self.experts = CpuMegaExpertsParams(
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            device="cpu",
            ep_start=self._stor_start,
            ep_local=self._stor_local,
            ep_global=config.n_routed_experts,
            group_k=self._xt_group_k,
            nvfp4=self._xt_nvfp4,
        )
        # ⭐ EPLB(`Qwen4ExpMixtureOfExperts.set_moe_parameters`)会读这些属性；
        #   我们的块被当作 SparseMoeBlock 使用时必须提供，否则 AttributeError ✓
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_routed_experts
        self.n_local_physical_experts = self.n_routed_experts
        self.n_shared_experts = int(getattr(config, "n_shared_experts", 1) or 1)
        self.engine = None
        self._xt_cpu_decode = None      # None=未探测；True=引擎有 cpu_decode（A78-11）
        self._xt_last_ref = None        # 自检用：最近一次参考输出（A78-8 追查）
        self._xt_last_ti = -1
        self._timing = os.environ.get("XIAOTU_TIMING") == "1"
        self._slot = None
        self._slot_device = None
        # ---- GPU 常驻层(见 gpu_resident_layers()) ----
        self._is_draft = False
        try:
            _li = extract_layer_index(prefix)
            # 每个 prefix 恰好登记一次(目标模型先构造,draft 后构造)。
            self._resident_instance = _instance_index(prefix)
            _dup = self._resident_instance > 0
            if _dup:
                # 同一 prefix 第二次出现 = **draft/speculator 模型**。
                # 【用户要求·第 210 轮】draft 层默认**在 GPU**:它只有几层,GPU 常驻
                # 让 draft 前向不必走 CPU MoE(那条路每 token 都要重读专家权重)。
                # ⚠️【2026-10-16 勘误 · B745】原注释引用的旧实测「draft 走 CPU 时投机净负
                #   (7.11 vs 10.76 t/s)」**已被本轮同口径实测推翻** ✗:
                #   `XIAOTU_MOE_RESIDENT_DRAFT=0`(draft 回 CPU)= 接受率 **72.0%** ✓、
                #   解码 **22.3 tok/s** vs 无投机 **18.4 tok/s**(**+21%** ✓),
                #   见 `docs/EXPERIMENTS.md` **B730/B739**。
                # ⇒ 正确表述:CPU draft 让 draft 前向**变慢**(每 token 读专家权重),
                #   但**不必然**使投机净负 —— 净收益取决于
                #   `spec_period / rest` 的比值(rest 是与 batch 无关的固定层间开销)。
                # ⇒ 因此本告警**不再断言"net-negative"**,只说明"会让 draft 前向变慢"。
                # ⛔⛔ 2026-10-16 修【E 项根因】:默认**由"常驻"改为"落 CPU"** ✓
                #    B840 同口径三臂(真 agent 载荷/k=4/MBT=8192/TP=2/outlen=256):
                #      * 无投机        = 18.3 tok/s / TPOT 54.6 ms
                #      * draft 落 CPU  = **30.1 tok/s / TPOT 33.2 ms** ✓(+64.5%,最优)
                #      * draft GPU 常驻 = 15.0 tok/s / TPOT 66.9 ms ⛔(**比不开投机还慢**)
                #    ⇒ 默认必须是"落 CPU";只有显式 `=1` 才常驻 ✓
                self._is_draft = True
                self._gpu_resident = (
                    os.environ.get("XIAOTU_MOE_RESIDENT_DRAFT", "0") == "1"
                )
                if self._gpu_resident:
                    print(
                        f"[xiaotu] WARNING: draft layer {prefix} 被显式 "
                        f"XIAOTU_MOE_RESIDENT_DRAFT=1 强制常驻 GPU。"
                        f"⛔ 2026-10-16 同口径实测它**比落 CPU 慢一倍、且比不开投机还慢**"
                        f"(15.0 vs 30.1 vs 18.3 tok/s;真 agent 载荷/k=4/MBT=8192/TP=2/outlen=256)"
                        f"⇒ 除非你正在复核该测量,否则请 unset 该变量。",
                        flush=True,
                    )
            else:
                self._gpu_resident = _li in gpu_resident_layers()
                # 【诊断】一次性看清"常驻层为什么没生效":打印解析到的层号、prefix
                # 与集合。XIAOTU_RESIDENT_DIAG=1 才开(避免刷屏)。
                if os.environ.get("XIAOTU_RESIDENT_DIAG") == "1":
                    _G = globals().setdefault("_RESIDENT_DIAG_N", [0])
                    if _G[0] < 6 or self._gpu_resident:
                        print(
                            f"[xtu-resident] idx={_li} prefix={prefix} "
                            f"set={sorted(gpu_resident_layers())} "
                            f"resident={self._gpu_resident}",
                            flush=True,
                        )
                    _G[0] += 1
        except Exception as _e:
            if os.environ.get("XIAOTU_RESIDENT_DIAG") == "1":
                print(f"[xtu-resident] EXCEPTION {type(_e).__name__}: {_e}", flush=True)
            self._gpu_resident = False
        self._resident_slot = None
        # ---- expert parallelism state (see ep_enabled()) ----
        self._ep = False
        self._ep_start = 0
        # 图安全的输出缓冲(见 _graph_out_buffers);cap 在 finalize 里定
        self._out_buf = None
        self._io_buf = None
        self._out_cap = 8
        self._ep_local = int(self.n_routed_experts)
        try:
            # 只登记**第一份**(目标模型)。draft 模型层名相同,若覆盖会让
            # prefetch-ahead(_LAYERS.get(L+1))与 pinned 预构建都作用在 draft 权重上,
            # 主模型的 H2D 重叠失效。
            _li = extract_layer_index(prefix)
            if _li not in _LAYERS:
                _LAYERS[_li] = self
        except Exception:
            pass

    # ---- long-prefill GPU path helpers ----
    def _graph_out_buffers(self, qlen: int, device):
        """CUDA 图安全的输出缓冲(地址**跨调用稳定**)。

        背景(第 18 轮,NOTES §347/§348):CPU 路径原先每次调用都
        `torch.empty(qlen,H,fp32)` 再 `out.to(bf16)`,**每次拿到新地址**。
        图捕获时内核把结果写到捕获期的地址,而 Python 侧返回的是新分配的张量
        ⇒ vLLM 读到的是未初始化数据 ⇒ 序列永远"没前进" ⇒ 每步重做整段预填充
        (`qlen=257 × 200/200`,1136.8 ms/token,实测 1.25 s/token)。
        修法:按**捕获范围内**的最大尺寸预分配两块持久缓冲,fp32 结果先 `copy_`
        成 bf16 再以**视图**返回(视图基址固定,replay 命中同一地址)。

        超过 `cap` 的形状(超长预填充)不会进图,退回临时分配即可,不浪费显存。
        """
        cap = self._out_cap
        if qlen > cap:
            return None
        buf = self._out_buf
        if buf is None or buf[0].device != device:
            f32 = torch.empty(cap, self.hidden_size, dtype=torch.float32, device=device)
            bf16 = torch.empty(cap, self.hidden_size, dtype=torch.bfloat16, device=device)
            buf = (f32, bf16)
            self._out_buf = buf
        # 【第 26 轮】ids/weights 也必须用持久缓冲。
        # fork 的 `_cpu_decode` 是**直接透传** `topk_ids.data_ptr()/topk_weights.data_ptr()`;
        # 我们这边为了做 EP 重映射,每层新建 4 个张量(减/夹/转 int32/where/转 f32)。
        # 在图重放下,图里记录的 D2H 源地址是**捕获期**的地址,而 Python 侧每层给的是新地址
        # ⇒ 引擎读到陈旧/未就绪数据,而且这种"读到的不是刚写的那份"会让异步握手的
        # 完成判定出问题,表现为**周期性等待**(实测主线 28 ms/层 vs fork 0.9 ms/层)。
        rb = self._io_buf
        if rb is None or rb[0].device != device:
            ids = torch.empty(cap, self.top_k, dtype=torch.int32, device=device)
            wts = torch.empty(cap, self.top_k, dtype=torch.float32, device=device)
            mask = torch.empty(cap, self.top_k, dtype=torch.bool, device=device)
            rb = (ids, wts, mask)
            self._io_buf = rb
        return buf[0][:qlen], buf[1][:qlen], rb[0][:qlen], rb[1][:qlen], rb[2][:qlen]

    def _gpu_shard(self):
        """(w13, s13, w2, s2, start, local_E, tp) CPU views for the GPU path.

        ⭐ A78-4（2026-10-26）：GPU 路径(NVFP4)吃的是
        **checkpoint 原样的 1 字节 e4m3 组尺度**，`weight_scale_2` 由 Triton 内核按
        **每专家**的 fp32 标量乘上去（`ts_ptr`，见 `_xt_nvfp4_ts()`）。
        ⇒ 不要在这里换成"折叠后的 fp32 尺度"：那样每层要多占 3 倍显存(48 层 +11 GiB/卡)，
          而 CPU 引擎才需要 fp32（内核契约不同，见 `_xt_fold_nvfp4_scales`）。
        """
        ex = self.experts
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp > 1:
            if getattr(self, "_shard_storage", False):
                # 【§365】参数已经是本 rank 分片,整份返回,不要再切。
                return (
                    ex.w13_weight.data, ex.w13_weight_scale.data,
                    ex.w2_weight.data, ex.w2_weight_scale.data,
                    self._stor_start, self._stor_local, tp,
                )
            # Expert-parallel: each rank streams only its 1/TP expert shard.
            rank = get_tensor_model_parallel_rank()
            L = self.n_routed_experts // tp
            st = rank * L
            return (
                ex.w13_weight.data[st:st + L], ex.w13_weight_scale.data[st:st + L],
                ex.w2_weight.data[st:st + L], ex.w2_weight_scale.data[st:st + L],
                st, L, tp,
            )
        return (
            ex.w13_weight.data, ex.w13_weight_scale.data,
            ex.w2_weight.data, ex.w2_weight_scale.data,
            0, self.n_routed_experts, tp,
        )

    def prefetch_gpu_weights(self, device) -> None:
        """Async H2D of this layer's GPU weights (issued by the previous layer)."""
        if self.engine is None or self._gpu_resident:
            return   # 常驻层权重已在显存,不需要 ping-pong 预取
        from vllm_xiaotu_moe.gpu_prefill import _pinned_kmajor, prefetch_layer

        w13, s13, w2, s2, _, _, _ = self._gpu_shard()
        self._slot = prefetch_layer(
            _pinned_kmajor(w13), _pinned_kmajor(s13),
            _pinned_kmajor(w2), _pinned_kmajor(s2), device,
        )
        self._slot_device = device

    # ---- 引擎构造(后置加载钩子,权重已就位) ----
    def finalize_mega_moe_weights(self) -> None:
        if self.engine is not None:
            return
        if self._gpu_resident:
            # 【幂等 + 预算】常驻层不设 self.engine,这个钩子可能被调用两次(同一进程
            # 里插件模块被复制成两份时尤其明显)⇒ 必须自己防重复分配,否则显存翻倍。
            if self._resident_slot is not None or resident_already_built(
                self.prefix, getattr(self, "_resident_instance", 0)
            ):
                return
            _nb = self._resident_bytes()
            if _nb <= 0:   # 形状取不到时的兜底估计(打包 fp4 ≈ 12.6MB/专家 + 尺度)
                _nb = int(13.5e6 * max(1, int(self._ep_local)))
            if not self._is_draft:
                # 目标层第一个决预算之前,先把 draft 的份额扣掉(用户要求:
                # draft 永远在 GPU;预算只用来限制目标层)。
                reserve_draft_bytes(_nb)
            _gb = float(os.environ.get("XIAOTU_MOE_RESIDENT_BUDGET_GB", "0") or 0)
            _st = _resident_state()
            print(f"[xiaotu] resident try {self.prefix}: {_nb/2**30:.2f} GiB, "
                  f"used={_st['used_bytes']/2**30:.2f}, "
                  f"reserved={_st.get('reserved_bytes', 0)/2**30:.2f}, "
                  f"budget={_gb:.1f} GiB"
                  f"{' [draft, mandatory]' if self._is_draft else ''}", flush=True)
            if resident_budget_ok(_nb, mandatory=self._is_draft):
                self._build_resident_slot()
                return
            if self._is_draft:
                # draft 是 mandatory ⇒ 正常不会走到这里;真走到了说明显存彻底不够。
                print(
                    f"[xiaotu] ERROR: draft layer {self.prefix} could not be made "
                    f"GPU-resident. The draft will run on CPU and pay the CPU MoE "
                    f"cost per draft step (slower; not necessarily net-negative: "
                    f"measured +21% decode in EXPERIMENTS B730/B739). To avoid it, "
                    f"lower XIAOTU_MOE_GPU_RESIDENT_LAYERS / "
                    f"raise XIAOTU_MOE_RESIDENT_BUDGET_GB.",
                    flush=True,
                )
            else:
                print(f"[xiaotu] resident budget exhausted -> {self.prefix} stays on CPU",
                      flush=True)
            # 预算用尽 ⇒ 本层当普通 CPU 层:继续往下把 CPU 引擎建起来,否则前向会
            # 因为 engine=None 而静默返回输入(算错)。
            self._gpu_resident = False
        # ---- 线程池自旋窗口(重要) -------------------------------------------
        # 引擎 worker 在两次调用之间自旋 spin_idle_us 等下一次任务,引擎默认 5 ms。
        # 但解码步里相邻两次调用只隔 ~2-4 ms ⇒ 192 个 worker 在整个解码期间几乎
        # 全在自旋,把整机烧满(实测:解码期间进程占 166-259 核,而真正的 MoE
        # 算力只需 ~21 核),并抢走驱动 GPU 的主线程 ⇒ 每层 `rest` 被推高。
        # 实测(scripts/sweep_spin.sh,负载 50-100 的同程序列,单位 ms/层):
        #   spin=5000: rest 3.30-3.47, period 6.6-7.1, C=1 7.53 tok/s(load 100)
        #   spin=1000: rest 1.27-1.42, period 4.3-4.9, C=1 10.74 tok/s(load 75)
        #   spin= 300: rest 0.94-0.97, period 3.7-3.9, C=1  9.61 tok/s(load 52)
        # 超时后 caller 会 cv_.notify_all() 唤醒(limited 与 non-limited 路径都有,
        # 见 numa_pool.hpp parallel_for 的 fallback),所以不会死锁。
        # 可用环境变量覆盖(例如 prefill 密集场景想要更大自旋)。
        os.environ.setdefault("XIAOTU_MOE_SPIN_IDLE_US", "300")
        # ---- 线程数:按"每 CCD 4-5 核"定(用户给定规则,2026-09-11)-----------
        # 规则:解码性能的最优解是**每个 CCD 只开 4-5 个核心**;更多核心拿不到更多
        # 带宽,反而抢 L3 与散热/power 预算,实测更慢。本机 24 CCD ⇒ 96-120 线程。
        # 佐证:本仓库 process_data/decomp/NUMA_BANDWIDTH_CCD.md(1 CCD ~30 GB/s、
        # 越过 6-8 CCD 后每 CCD 效率下降);LvLLM/lk_moe 官方 README(线程数 = 物理核
        # ÷ GPU 数;声明 L3 命中率 >50%、跨 node 通信 <3%)。
        # 实测(本仓库 NOTES §43/§44):192 线程 8.25-9.98 tok/s(C=1)、176→11.44、
        # 160→11.66、144→12.57、128→12.66 ⇒ 与"少开核"方向一致。
        # ⇒ 默认取 n_ccd × 5(上限 = 核数),用户仍可用 XIAOTU_MOE_THREADS 覆盖。
        if os.environ.get("XIAOTU_MOE_THREADS") is None:
            _ncpu = os.cpu_count() or 8
            _nccd = 0
            try:
                _l3 = set()
                for _c in range(_ncpu):
                    _p = f"/sys/devices/system/cpu/cpu{_c}/cache/index3/id"
                    if os.path.exists(_p):
                        with open(_p) as _f:
                            _l3.add(_f.read().strip())
                _nccd = len(_l3)
            except OSError:
                _nccd = 0
            if _nccd > 0:
                _want = _nccd * 5                     # 每 CCD 5 核
            else:
                _want = max(1, int(_ncpu * 5 / 8))    # 回退:核数 * 5/8
            os.environ["XIAOTU_MOE_THREADS"] = str(max(1, min(_ncpu, _want)))
        import xiaotu_moe

        ex = self.experts
        w13 = ex.w13_weight.detach()
        w2 = ex.w2_weight.detach()
        s13 = ex.w13_weight_scale.detach()
        s2 = ex.w2_weight_scale.detach()

        # ---- expert parallelism: this rank's engine holds only its shard ----
        # The CPU weights are still the FULL set on every rank (the mainline
        # loader is told tp_size=1 for this module), so slicing is a pure pointer
        # offset: the engine copies `expert_num` experts out of the given base
        # pointer, and the slice keeps dim-0 contiguity.
        tp = 1
        if ep_enabled():
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp = int(get_tensor_model_parallel_world_size())
            if tp > 1:
                rank = int(get_tensor_model_parallel_rank())
                if getattr(self, "_shard_storage", False):
                    # 【§365】参数**已经只含本 rank 分片**(分配期就切了)⇒ 绝不能再切一次,
                    # 否则切出空张量。直接沿用分配时记下的 start/local。
                    E = self.n_logical_experts
                    st, L = self._stor_start, self._stor_local
                else:
                    E = int(w13.shape[0])
                    L = E // tp
                    st = rank * L
                    w13 = w13[st:st + L].contiguous()
                    w2 = w2[st:st + L].contiguous()
                    s13 = s13[st:st + L].contiguous()
                    s2 = s2[st:st + L].contiguous()
                self._ep = True
                self._ep_start = st
                self._ep_local = L
                print(
                    f"[xiaotu] EP {self.prefix}: rank {rank}/{tp} owns experts "
                    f"[{st}, {st + L}) of {E}",
                    flush=True,
                )

        cfg = xiaotu_moe.MOEConfigV2()
        # 【v0.2·第 15 轮·NUMA 放置修复】把真实的 TP rank/world 告诉引擎,
        # **只用于 NUMA 权重分片与线程池核表切分**(归约仍走下面的 configure_ep
        # 共享内存,所以必须同时关掉引擎自建的 auto EP,否则重复归约)。
        #
        # 原先硬编码 `1 / 0` ⇒ `moe_v2.hpp` 里 `_world = num_processes = 1` ⇒
        #   `nshard_ = numa_node_count()/_world = 8`,
        #   `rank_node_base_ = process_id * (numnodes/_world) = 0`
        # ⇒ **两个 rank 都把各自那份权重分片绑到全部 8 个 NUMA node** ⇒ 每个 node
        # 承受两个 rank 的量。本机每个 node 只有 193 GB,而单 worker 需要 219–242 GB:
        #   oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=0,
        #             task=VLLM::Worker_TP, anon-rss:218866316kB
        # 进程**静默死亡**(无 Traceback)⇒ 表现为"日志只加载到一半就没了"。
        # 同一个 process_id 还决定 `shared_numa_pool(process_id, num_processes)` 的核表,
        # 硬编码 1 会让两个 rank 的线程池都横跨全部 24 CCD 互抢核(与 mode B 的
        # NOTES §334p "compute 双峰 0.41 vs 9.5 ms" 同源)。
        # 修法与 `mixed_experts.py`(mode B)完全一致,两处务必保持同步。
        os.environ.setdefault("XIAOTU_MOE_NO_AUTO_EP", "1")
        _tp, _rank = 1, 0
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            _tp = int(get_tensor_model_parallel_world_size())
            _rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            _tp, _rank = 1, 0
        cfg.num_processes = max(1, _tp)
        cfg.process_id = max(0, _rank)
        cfg.gpu_id = torch.cuda.current_device()
        cfg.has_gate_proj = True
        cfg.expert_num = int(w13.shape[0])
        cfg.top_k = self.top_k
        cfg.hidden_size = self.hidden_size
        cfg.intermediate_size = self.moe_intermediate_size
        cfg.max_batch_size = 8192
        cfg.max_num_seqs = 256
        cfg.stride = 32
        cfg.group_min_len = 10
        # 引擎按 group_max_len 给 input_fp32_/gate_input_/… 定尺寸,而 forward_many_m
        # 用 token_id(<qlen)直接索引 → **必须 group_max_len >= qlen**(否则堆越界;
        # 曾经 qlen=8192/profile 而 group_max_len=4224,越界 ~2x)。canonical 公式:
        # min(4096, max_num_batched_tokens) + 128。
        _mnbt = int(
            getattr(
                getattr(self._vllm_config, "scheduler_config", None),
                "max_num_batched_tokens",
                4096,
            )
            or 4096
        )
        # V2 引擎(MOE_MXFP4/MOE_FP8/MOE_BF16)的每 token 缓冲是**按调用动态
        # 分配**的(qlen = 本次 forward 的 token 数),不再依赖 group_max_len;
        # group_max_len 只对旧 moe.cpp 引擎的静态缓冲有意义。这里仍然把它设成
        # max_num_batched_tokens,并在两者不一致时给出**信息性**提示(而不是
        # 让用户以为必须把 --max-num-batched-tokens 压到 4096)。
        cfg.group_max_len = int(
            os.environ.get("XIAOTU_GROUP_MAX_LEN", str(max(4096, _mnbt) + 128))
        )
        if _mnbt > cfg.group_max_len:
            print(
                f"[xiaotu] WARNING: max_num_batched_tokens={_mnbt} > "
                f"group_max_len={cfg.group_max_len}; the legacy per-token "
                "buffers would overflow (V2 engine allocates per call, so this "
                "is informational).",
                flush=True,
            )
        # ---- 图安全输出缓冲的上界(见 _graph_out_buffers)------------------
        # 只需要覆盖"会被 CUDA 图捕获"的形状:
        #   * 显式限了 --cudagraph-capture-sizes ⇒ max_cudagraph_capture_size 即上界;
        #   * 否则主线 PIECEWISE 会捕获到 MBT,取 MBT(MBT=256 时仅 6.3 MB/层)。
        # XIAOTU_MOE_GRAPH_OUT_MAX>0 可再兜一层显存(cap×H×(4+2)B×层数×rank)。
        _env_cap = int(os.environ.get("XIAOTU_MOE_GRAPH_OUT_MAX", "0") or 0)
        _cap = int(_mnbt)
        _cc = getattr(self._vllm_config, "compilation_config", None)
        _mcs = getattr(_cc, "max_cudagraph_capture_size", None) if _cc is not None else None
        if _mcs:
            _cap = int(_mcs)
        if _env_cap > 0:
            _cap = min(_cap, _env_cap)
        self._out_cap = max(8, _cap)
        cfg.activation_type = 0
        cfg.use_gpu_prefill = False
        cfg.groupN = 1
        # ⭐ NVFP4 的组大小是 16（e4m3 尺度），MXFP4 是 32（e8m0）
        cfg.groupK = int(getattr(self, "_xt_group_k", 32))

        # ---- 引擎只能吃**主机**指针:构造前显式断言(NOTES §482)----------------
        # 引擎内部按 [E][2I][H/2] 定长步长 memcpy;喂设备/空/非连续张量会在 C++ 里
        # 直接 SIGSEGV(**没有 Python 栈**),表现为"进程静默消失"。
        # Mode B(mixed_experts)早就有等价的 _assert_host_source;Mode A 一直没有,
        # 于是 DSpark draft(走这条路径)的专家权重把 SIGSEGV 暴露了出来。
        for _nm, _t in (("w13", w13), ("w2", w2), ("s13", s13), ("s2", s2)):
            _bad = (
                _t is None
                or _t.device.type != "cpu"
                or _t.numel() == 0
                or not _t.is_contiguous()
            )
            if _bad:
                raise RuntimeError(
                    f"xiaotu(Mode A) {self.prefix}: refusing to build the engine from "
                    f"a non-host source ({_nm}: device="
                    f"{getattr(_t, 'device', None)} numel="
                    f"{getattr(_t, 'numel', lambda: '?')()} contiguous="
                    f"{getattr(_t, 'is_contiguous', lambda: '?')()} "
                    f"shape={tuple(getattr(_t, 'shape', ()))}) — handing this to the "
                    f"C++ engine would segfault in shard_fill_*."
                )
        if getattr(self, "_xt_nvfp4", False) or getattr(
                self.experts, "_nvfp4", False):
            # ⭐⭐ NVFP4-W4A16：内核只有**每专家一个**全局尺度（`w13_gs[eid]`），
            #   而 checkpoint 的 gate/up 两个 `weight_scale_2` **不同**（实测 395/400 不同）
            #   ⇒ 采用【路线 A·折叠】：把 scale_2 乘进各自半边的 e4m3 组尺度，
            #      引擎侧传 gs=0（内核按 1.0f 处理）。
            #   ⚠️ 代价：e4m3 只有 3 位尾数 ⇒ 组尺度多一次舍入（≤~6% 相对误差/组），
            #      且乘积低于 e4m3 次正规下限的组会被压到最小次正规（实测 61/102400 ≈0.06%）。
            #   ✅ 精确做法（路线 B）：改内核按 half 取 `gs[2*eid+half]`，无需舍入 —— 见台账 231.1
            self._xt_fold_nvfp4_scales()
            print(f"[xtu-nvfp4] 引擎=MOE_NVFP4  组尺度(fp32)已折叠 scale_2 "
                  f"(E={int(s13.shape[0])}, I={int(self.experts.intermediate_size)}, "
                  f"groupK={cfg.groupK}, "
                  f"scales={self._nvfp4_s13.nbytes / 2**20:.0f}MiB+"
                  f"{self._nvfp4_s2.nbytes / 2**20:.0f}MiB)", flush=True)
            self.engine = xiaotu_moe.MOE_NVFP4(
                cfg,
                w13.data_ptr(), w2.data_ptr(),
                self._nvfp4_s13.data_ptr(), self._nvfp4_s2.data_ptr(),
                0, 0,
            )
        else:
            self.engine = xiaotu_moe.MOE_MXFP4(
                cfg,
                w13.data_ptr(), w2.data_ptr(),
                s13.data_ptr(), s2.data_ptr(),
                0, 0,
            )
        # CUDA graph 安全:在**捕获之前**把 pinned 缓冲按最大捕获尺寸预分配好。
        # 捕获期间任何 cudaHostAlloc/cudaFreeHost 都会让 capture 失效
        # (实测:cudaErrorStreamCaptureInvalidated → 引擎初始化直接失败),
        # 所以这里一次性分配,稳态不再扩容;prefill 若需要更大缓冲会在
        # 非捕获路径上扩容(那时 sync+free 是合法的)。
        # ---- EP:部分和走共享内存(替代每层一次 NCCL all-reduce) ----
        self._ep_shm_tokens = 0
        if self._ep and ep_shm_enabled() and hasattr(self.engine, "configure_ep"):
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            _world = int(get_tensor_model_parallel_world_size())
            _rank = int(get_tensor_model_parallel_rank())
            # 覆盖 CPU 路径可能出现的最大 qlen:GPU prefill 阈值以上走 GPU 通路,
            # 所以这里取 max(预分配, 阈值) 即可(两者都可由环境变量调整)。
            _toks = int(os.environ.get("XIAOTU_MOE_EP_SHM_TOKENS", "1024"))
            if _world > 1:
                _mm, _stride = _ep_shm_attach(
                    extract_layer_index(self.prefix), _toks, self.hidden_size, _world,
                    _shm_instance_tag(self.prefix),
                )
                self.engine.configure_ep(_rank, _world, _mm, _stride)
                self._ep_shm_tokens = _toks
                self._ep_shm_mm = _mm
                print(
                    f"[xiaotu] EP-shm {self.prefix}: rank {_rank}/{_world} "
                    f"stride={_stride} tokens={_toks}",
                    flush=True,
                )
        _pre = int(os.environ.get("XIAOTU_CD_PREALLOC_TOKENS", "512"))
        if hasattr(self.engine, "prepare_decode_buffers"):
            try:
                self.engine.prepare_decode_buffers(max(1, _pre), self.top_k)
            except Exception as _e:  # 老 .so 没有该方法时静默跳过
                print(f"[xiaotu] prepare_decode_buffers skipped: {_e}", flush=True)
        # 【第 16 轮更正】原注释说"引擎持有这些参数的内存(达 data_ptr),必须防被
        # 替换/释放" —— 该前提已被 `scripts/test_engine_copies_weights.py` 证伪:
        # 引擎构造时就 memcpy 进自己的 NUMA 分片内存,原地改写调用方缓冲后输出逐位不变。
        # 这里保留强引用**另有原因**:`gpu_prefill._pin_key()` 会把源 storage 存进
        # `_PIN_CACHE`(见 NOTES §346c),GPU 预填充/常驻层路径需要这些 host 张量存活。
        self._w13, self._w2 = w13, w2
        self._s13, self._s2 = s13, s2
        del w13, w2, s13, s2
        print(f"[xiaotu] engine built {self.prefix} "
              f"E={self.n_routed_experts} topk={self.top_k}", flush=True)

    def _xt_nvfp4_ts(self, device):
        """每专家 fp32 二级尺度（GPU/Triton 路径用），在设备上缓存。

        内核侧 `ts_ptr` 按 `pid_e` 取一个 **标量** ⇒ gate/up 两半必须共享同一个
        `weight_scale_2`。真实 checkpoint 实测相等（HANDOFF §13.1）；
        若不相等，这里**直接报错**（静默取其一 = 静默错数 ✗）。
        """
        ex = self.experts
        _c = getattr(self, "_nvfp4_ts_cache", None)
        if _c is not None and _c[0] == device:
            return _c[1], _c[2]
        _g13 = ex.w13_weight_scale_2.detach().to(torch.float32)   # [E,2]
        _g2 = ex.w2_weight_scale_2.detach().to(torch.float32)     # [E]
        if _g13.ndim == 2 and _g13.shape[1] > 1:
            _ne = int((_g13[:, 0] != _g13[:, 1]).sum())
            if _ne:
                raise RuntimeError(
                    f"[xtu-nvfp4] {getattr(self, 'prefix', '?')}: gate/up 的 "
                    f"weight_scale_2 有 {_ne}/{_g13.shape[0]} 个不相等，而 GPU 路径的 "
                    "二级尺度是**每专家一个标量**(gate/up 共享) ⇒ 不能走 GPU 路径。"
                    "要么给内核加第二列 ts，要么让这层留在 CPU 引擎。"
                )
        # 兼容 [E]（只给一个标量）与 [E,2]（gate/up 各一格）两种 ckpt 形状
        _col0 = _g13[:, 0] if _g13.ndim == 2 else _g13.reshape(-1)
        _ts13 = _col0.contiguous().to(device)
        _ts2 = _g2.contiguous().to(device)
        self._nvfp4_ts_cache = (device, _ts13, _ts2)
        return _ts13, _ts2

    def _xt_fold_nvfp4_scales(self):
        """把每张量二级尺度 `weight_scale_2` 折进组尺度，**结果保持 fp32**（幂等）。

        ⭐⭐ 为什么必须是 fp32（A78-3，2026-10-26 实测内核源码）：
        `NVFP4WeightTraits = Packed4WeightTraitsBase<E2M1, NVFP4Tag, /*E8M0=*/false, ...>`
        ⇒ `gate_up_impl`/`down_impl` 里 `byte_off = eid * (nb*kb) * sizeof(float)`
        ⇒ **block scale 按 fp32 读**（单字节 e8m0 是 MXFP4 的契约）。上一版传折叠好的
        e4m3 单字节 ⇒ 内核按 4 字节步长解读、只覆盖 1/4 范围 ⇒ 读越界 + 尺度全错（不报错）✗

        ⭐ 为什么可以直接折进组尺度：e4m3 → fp32 是**精确**的，再乘 fp32 的 `scale_2`
        只多 ~1 ulp 舍入 ⇒ **不会**再出现 e4m3 再舍入那种 9% 量级的误差
        （真实 checkpoint 实测见 `dev-docs/HANDOFF_NVFP4_2026-10-26.md` §13.2）✓

        ⭐ 单一来源：CPU 引擎（`MOE_NVFP4`）与 GPU 路径（`_gpu_shard`→常驻/prefill）都从这里取，
        避免"两条路各折一次/一条忘了折"（A78-4 就是这么错的）。
        """
        s13f = getattr(self, "_nvfp4_s13", None)
        s2f = getattr(self, "_nvfp4_s2", None)
        if s13f is not None and s2f is not None:
            return s13f, s2f
        ex = self.experts
        s13 = ex.w13_weight_scale.data          # [E,2I,H/gk] uint8（e4m3 的位模式）
        s2 = ex.w2_weight_scale.data            # [E,H,I/gk]   uint8
        _E = int(s13.shape[0])
        _I = int(ex.intermediate_size)
        _g13 = ex.w13_weight_scale_2.detach().to(torch.float32)   # [E,2] gate/up 各一格
        _g2 = ex.w2_weight_scale_2.detach().to(torch.float32)     # [E]
        # ⭐ 防**静默错数**守卫：`scale_2` 全 0 说明 loader/名称映射没接上 ⇒ 折叠后组尺度
        #   全 0 ⇒ 输出恒 0（"能跑但全错"）⇒ 宁可当场报错 ✗
        if float(_g13.abs().max()) == 0.0 or float(_g2.abs().max()) == 0.0:
            raise RuntimeError(
                "[xtu-nvfp4] ⛔ weight_scale_2 全为 0 ⇒ 尺度未被加载"
                f"（w13 max={float(_g13.abs().max()):.3g}, "
                f"w2 max={float(_g2.abs().max()):.3g}）"
                "；检查 load_weights/weight_scale_2_loader 是否命中"
            )
        f13 = s13.view(torch.float8_e4m3fn).to(torch.float32)
        f13[:, :_I, :] *= _g13[:, 0].view(_E, 1, 1)      # gate(w1)
        f13[:, _I:, :] *= _g13[:, 1].view(_E, 1, 1)      # up(w3)
        f2 = s2.view(torch.float8_e4m3fn).to(torch.float32)
        f2 *= _g2.view(_E, 1, 1)
        self._nvfp4_s13 = f13.contiguous()
        self._nvfp4_s2 = f2.contiguous()
        print(f"[xtu-nvfp4] scale_2 已装载: w13∈[{float(_g13.min()):.3g},"
              f"{float(_g13.max()):.3g}] w2∈[{float(_g2.min()):.3g},"
              f"{float(_g2.max()):.3g}]", flush=True)
        return self._nvfp4_s13, self._nvfp4_s2

    def _resident_bytes(self) -> int:
        """本层本 rank 常驻权重需要多少字节(只看形状,不做拷贝)。"""
        try:
            w13, s13, w2, s2, _st, _L, _tp = self._gpu_shard()
            return int(sum(t.numel() * t.element_size()
                           for t in (w13, s13, w2, s2) if t is not None))
        except Exception:  # noqa: BLE001
            return 0

    def _build_resident_slot(self) -> None:
        """把本层(本 rank 分片)的专家权重一次性放进显存并常驻。

        复用 gpu_prefill 的 K-major + PrefetchSlot 机制:`slot.bufs` 是设备缓冲,
        这里用一次性阻塞 H2D 填好并记录 ready 事件,之后每次 forward 只需
        `gpu_moe_layer(..., slot=self._resident_slot)`,不再有任何 H2D。
        """
        from vllm_xiaotu_moe.gpu_prefill import PrefetchSlot, _pinned_kmajor

        if self._resident_slot is not None:   # 幂等:见 finalize_mega_moe_weights()
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        w13, s13, w2, s2, st, L, tp = self._gpu_shard()
        src = (
            _pinned_kmajor(w13), _pinned_kmajor(s13),
            _pinned_kmajor(w2), _pinned_kmajor(s2),
        )
        slot = PrefetchSlot()
        slot.alloc(src, dev)
        for buf, s_ in zip(slot.bufs, src):
            buf.copy_(s_, non_blocking=False)
        slot.ready = torch.cuda.Event()
        slot.ready.record()
        slot.busy = None
        self._resident_slot = slot
        self._slot_device = dev
        gb = sum(b.numel() for b in slot.bufs) / 2**30
        print(
            f"[xiaotu] GPU-resident {self.prefix}: {gb:.2f} GiB on {dev} "
            f"(tp={tp}, experts={L})",
            flush=True,
        )

    def _xt_selfcheck(self, x_bf16, ids_i32, wts_f32, out_f32):
        """⭐ 进程内自检（`XIAOTU_MOE_SELFCHECK=1`）：把**模型自己的参数**按引擎约定
        解一遍，和引擎刚算出的 `out` 对比。

        为什么必须有它：离线单测只能证明"引擎 + 我理解的约定"自洽，证明不了
        **运行时真的把正确的权重/隐藏态喂进去了**。这条自检失败 ⇒ 问题在加载/映射；
        通过 ⇒ 问题在别处（路由 / 别的层），可以立刻往外挪，不必再猜。
        """
        try:
            import torch as _t

            # ⛔ 只在**真实数据**上跑：profile_run 的 dummy 输入可能是**全零**
            #   （实测 hidden std=0.0）⇒ 参考解与引擎输出都是 0，会得到假通过 ✗
            if x_bf16.numel() == 0 or float(x_bf16.abs().max()) == 0.0:
                print("[xtu-selfcheck] hidden 全零（profile/dummy）⇒ 稍后重试", flush=True)
                return False
            # 逐层统计（`XIAOTU_MOE_LAYERSTATS=1`）：只看**第一个真实解码 token**，
            # 目的是看残差流在哪一层开始崩（std/absmax 突变 ⇒ 上游 attention/PLE 有问题）
            if (os.environ.get("XIAOTU_MOE_LAYERSTATS", "0") == "1"
                    and x_bf16.shape[0] <= 64
                    and not getattr(self, "_xt_ls_done", False)):
                self._xt_ls_done = True
                _p = str(getattr(self, "prefix", "?"))
                _li = -1
                for _tok in _p.replace("layers.", "|").split("|"):
                    if _tok.isdigit():
                        _li = int(_tok)
                        break
                print(f"[xtu-layerstats] layer={_li:02d} in_std={float(x_bf16.float().std()):.4e} "
                      f"in_absmax={float(x_bf16.abs().max()):.4e}", flush=True)
                return False
            if x_bf16.shape[0] > 64:
                if not getattr(self, "_xt_sc_prefill_noted", False):
                    self._xt_sc_prefill_noted = True
                    print(f"[xtu-selfcheck] qlen={x_bf16.shape[0]} > 64（prefill）⇒ 等解码步",
                          flush=True)
                return False
            s13 = getattr(self, "_nvfp4_s13", None)
            s2 = getattr(self, "_nvfp4_s2", None)
            if s13 is None or s2 is None:
                print("[xtu-selfcheck] ⚠ 没有折叠后的 fp32 组尺度，跳过", flush=True)
                return True
            GKS = int(getattr(self, "_xt_group_k", 16))
            E = int(self.experts.w13_weight.shape[0])
            I = int(self.experts.intermediate_size)
            H = int(self.experts.hidden_size)
            lut = _t.tensor([0.0, .5, 1., 1.5, 2., 3., 4., 6.,
                             -0., -.5, -1., -1.5, -2., -3., -4., -6.])
            # 【关键】必须挑一个**真的有本地专家**的 token：
            #   profile_run 期的 dummy 输入常常让 topk 权重全 0（EP 掩码把不属于本 rank 的
            #   专家置 0）⇒ 用 token 0 会得到"专家数=0、误差 0"的**假通过** ✗
            _ids_all = ids_i32.tolist()
            _w_all = wts_f32.tolist()
            _ti = -1
            for _i, _row in enumerate(_w_all):
                if any(float(v) != 0.0 and 0 <= int(e) < E
                       for e, v in zip(_ids_all[_i], _row)):
                    _ti = _i
                    break
            if _ti < 0:
                print(f"[xtu-selfcheck] 本次 {len(_w_all)} 个 token 都无本地专家"
                      f"（profile/dummy 期正常）⇒ 稍后重试", flush=True)
                return False
            _ids = _ids_all[_ti]; _w = _w_all[_ti]
            # ⚠️ 只解**用到的**专家：整层 256 个专家解开会要 3.3 GB，白费 ✗
            _use = [(int(e), float(wv)) for e, wv in zip(_ids, _w)
                    if float(wv) != 0.0 and 0 <= int(e) < E]
            x = x_bf16[_ti].to(_t.float32)
            print(f"[xtu-selfcheck] 取 token {_ti}/{len(_w_all)}；hidden std="
                  f"{float(x.std()):.4e} absmax={float(x.abs().max()):.4e}", flush=True)
            acc = _t.zeros(H)
            for e, wv in _use:
                w13 = self.experts.w13_weight[e]
                w2 = self.experts.w2_weight[e]
                d13 = _t.empty(2 * I, H)
                d13[:, 0::2] = lut[(w13 & 0x0F).long()]
                d13[:, 1::2] = lut[((w13 >> 4) & 0x0F).long()]
                d13 = d13 * s13[e].repeat_interleave(GKS, dim=1)
                d2 = _t.empty(H, I)
                d2[:, 0::2] = lut[(w2 & 0x0F).long()]
                d2[:, 1::2] = lut[((w2 >> 4) & 0x0F).long()]
                d2 = d2 * s2[e].repeat_interleave(GKS, dim=1)
                gate = x @ d13[:I, :].T
                up = x @ d13[I:, :].T
                acc += wv * ((_t.nn.functional.silu(gate) * up) @ d2.T)
            used = len(_use)
            got = out_f32[_ti].to(_t.float32)
            scale = acc.abs().mean().clamp_min(1e-8)
            rel = float((got - acc).abs().mean() / scale)
            self._xt_last_ref = acc
            self._xt_last_ti = _ti
            print(f"[xtu-selfcheck] 专家数={used} |ref|={float(acc.abs().mean()):.4e} "
                  f"|eng|={float(got.abs().mean()):.4e} 相对误差={rel:.4e} "
                  f"{'✅ 引擎与模型参数一致' if rel < 5e-2 else '⛔ 运行时喂给引擎的东西不对'}",
                  flush=True)
            return True
        except Exception as _exc:  # noqa: BLE001 自检失败不许影响主流程
            print(f"[xtu-selfcheck] ⚠ 自检异常: {type(_exc).__name__}: {_exc}", flush=True)
            return True

    def _xt_selfcheck2(self, x_bf16, ids_i32, wts_f32, out_f32, is_local_all):
        """第二轮自检（`XIAOTU_MOE_SELFCHECK=1`）：查**路由**、**共享专家**、
        以及**跨 rank 归约之后**的数值。第一轮已经证明"引擎算得对"，所以这里只查
        "喂进来的路由对不对 / 别的部分对不对 / 归约有没有把值搞坏"。"""
        try:
            import torch as _t

            _ids0 = [int(v) for v in ids_i32[0].tolist()]
            _w0 = [round(float(v), 4) for v in wts_f32[0].tolist()]
            print(f"[xtu-selfcheck2] ep_start={getattr(self, '_ep_start', 0)} "
                  f"L={getattr(self, 'experts', None) and self.experts.num_experts} "
                  f"ids={_ids0}", flush=True)
            print(f"[xtu-selfcheck2] w={_w0} sum={sum(_w0):.4f} "
                  f"非零={sum(1 for v in _w0 if v != 0.0)} "
                  f"去重={len(set(_ids0))} 全本地={bool(is_local_all[0])}", flush=True)
            gate = getattr(self, "gate", None)
            if gate is not None and hasattr(gate, "weight"):
                with _t.no_grad():
                    _lg = gate(x_bf16[:1].to(gate.weight.dtype))
                    if isinstance(_lg, tuple):
                        _lg = _lg[0]
                    _lg = _lg.to(_t.float32).reshape(-1)
                _top = _t.topk(_lg, min(10, _lg.numel()))
                print(f"[xtu-selfcheck2] router: |w|max={gate.weight.abs().max().item():.4e} "
                      f"logits std={_lg.std().item():.4e} top1={int(_top.indices[0])} "
                      f"top1-{_top.indices.numel()} 差="
                      f"{float(_top.values[0] - _top.values[-1]):.4e} "
                      f"模型给的 ids={ids_i32[0].tolist()[:4]}", flush=True)
            se = getattr(self, "shared_expert", None)
            if se is not None:
                with _t.no_grad():
                    _se = self._shared_out(x_bf16[:1])
                if _se is not None:
                    _se = _se.to(_t.float32).reshape(-1)
                    _gt = out_f32[0].to(_t.float32)
                    print(f"[xtu-selfcheck2] shared_expert |mean|={float(_se.abs().mean()):.4e} "
                          f"|out|mean={float(_gt.abs().mean()):.4e}", flush=True)
        except Exception as _exc:  # noqa: BLE001
            print(f"[xtu-selfcheck2] ⚠ 异常: {type(_exc).__name__}: {_exc}", flush=True)

    def _shared_out(self, hidden_states):
        """共享专家输出。

        qwen4_exp 族 = `Qwen2MoeMLP`（内部含 `sigmoid(expert_gate(x))` 门，见 A78-7）；
        DS4 族 = `DeepseekV4MLP`（无此门）。两者互斥，由 `__init__` 决定。
        """
        if getattr(self, "shared_expert", None) is not None:
            return self.shared_expert(hidden_states)
        if getattr(self, "shared_experts", None) is not None:
            return self.shared_experts(hidden_states)
        return None

    # ---- 权重加载（A78-3b / A78-7）----
    def load_weights(self, weights):
        """本块自己解析 checkpoint 键（**不要**交给上游 AutoWeightsLoader 递归）。

        上游递归只按 `named_children()` / `named_parameters()` 找名字，本块两处必然对不上：
          * routed experts 在 CPU 参数里且是 **EP 分片**，checkpoint 却给全部 512 个 id
            ⇒ `There is no module or parameter named '...experts.256'`（A78-3b）
          * 共享专家的键名/形状与模块结构不同（A78-7）

        分派规则：
          * `experts.*` → `self.experts.load_weights`（非本地 id 静默跳过 = EP 语义）
          * `shared_expert.*` → 交给对应子模块自己的 `load_weights`：
            上游 WeightsMapper 已把 `gate_proj/up_proj` 合并成 `gate_up_proj.weight` 并把
            `shard_id` 挂在**张量属性**上，`MergedColumnParallelLinear.load_weights` 会读它
            ⇒ TP 切分/分片偏移全部由它负责 ✓（手写 `weight_loader(...,0/1)` 会漏掉 stacked 语义）
          * `gate.weight` / `shared_expert_gate.weight` → 直接 copy
          * 其它键 ⇒ **抛错**（宁可当场失败，也不静默丢权重 ✗）
        """
        loaded: list[str] = []
        exp_sub: list[tuple[str, torch.Tensor]] = []
        sp_pairs: list[tuple[str, "nn.Module"]] = []
        for _pre, _aname in (("shared_expert.", "shared_expert"),
                             ("shared_experts.", "shared_experts")):
            _m = getattr(self, _aname, None)
            if _m is not None:
                sp_pairs.append((_pre, _m))
        for name, w in weights:
            if name.startswith("experts."):
                exp_sub.append((name[len("experts."):], w))
                continue
            if name == "gate.weight":
                _p = self.gate.weight
                _p.data.copy_(w.to(_p.dtype))
                loaded.append(name)
                continue
            if name == "shared_expert_gate.weight":
                _p = self.shared_expert_gate.weight      # [1, H]
                _p.data.copy_(w.to(_p.dtype))
                loaded.append(name)
                continue
            _done = False
            for _pre, _mod in sp_pairs:
                if not name.startswith(_pre):
                    continue
                _rest = name[len(_pre):]
                _child, _, _attr = _rest.partition(".")
                _sub = getattr(_mod, _child, None)
                if _sub is None:
                    continue
                _lw = getattr(_sub, "load_weights", None)
                if callable(_lw):
                    list(_lw([(_attr, w)]))
                else:
                    _pl = getattr(_sub, "weight_loader", None)
                    if _pl is None or not hasattr(_sub, _attr):
                        continue
                    _pl(getattr(_sub, _attr), w)
                loaded.append(name)
                _done = True
                break
            if _done:
                continue
            raise ValueError(
                f"[xtu-moe] {getattr(self, 'prefix', '?')}: 未处理的 checkpoint 键 "
                f"{name!r}（shape={tuple(w.shape)}）—— 拒绝静默丢弃"
            )
        if exp_sub:
            for _n in (self.experts.load_weights(exp_sub) or []):
                loaded.append(f"experts.{_n}")
        return loaded

    def build_engine(self):
        self.finalize_mega_moe_weights()

    # ---- forward ----
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # ⭐ 对照实验开关（A78-14 追查用）：**在一切之前**跳过整个 MLP（含共享专家），
        # 只保留 attention/残差 ⇒ 用来区分"MoE 坏了"与"上游（attention/PLE）坏了"。
        # ⚠️ 必须放在 `engine is not None` 之前 —— 早先放在里面 ⇒ 引擎已建好时永远不生效 ✗
        if os.environ.get("XIAOTU_MOE_SKIP", "0") == "1":
            if not getattr(self, "_xt_skip_noted", False):
                self._xt_skip_noted = True
                print(f"[xtu-moe] XIAOTU_MOE_SKIP=1 ⇒ {self.prefix} 的 MLP 被跳过（对照）",
                      flush=True)
            return hidden_states
        if self._gpu_resident and self._resident_slot is not None:
            # 常驻层:每步都在 GPU 上算,不需要阈值判断、不需要 CPU 引擎。
            self._resident_forward_kwargs = True
        if self.engine is None and not self._gpu_resident:
            # ⛔【A78-8 · 2026-10-26】这是**结构性退化**路径：引擎还没构造时直接返回输入，
            #   即**整层 MoE 被跳过**。它原本只为"模型构造阶段的前向探测"兜底，但如果
            #   后置加载钩子没被调用（qwen4_exp 族就缺 `finalize_mega_moe_weights`），
            #   就会**一直**走这里 ⇒ 模型能出字、bench 有读数、数值全错 ✗
            #   ⇒ 必须吵：每个进程每层只喊一次，且日志里可被机械检索（`MoE 被跳过`）。
            if not getattr(self, "_xt_warned_no_engine", False):
                self._xt_warned_no_engine = True
                print(
                    f"[xtu-moe] ⛔ 引擎未就绪 ⇒ **MoE 被跳过**（{self.prefix}）："
                    "后置加载钩子 `finalize_mega_moe_weights()` 没被调用。"
                    "qwen4_exp 族需 `compat_patches.patch_model_post_load_hook()`；"
                    "否则输出必然全错（A78-8）",
                    flush=True,
                )
            return hidden_states
        if getattr(self.gate, "tid2eid", None) is not None and input_ids is None:
            raise ValueError("DS-V4 hash MoE routing requires input_ids.")

        gate_out = self.gate(hidden_states)
        router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
        bias = getattr(self.gate, "e_score_correction_bias", None)
        bias_vl = getattr(self.gate, "bias_vl", None)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=bias.data if bias is not None else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=getattr(self.gate, "tid2eid", None),
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=bias_vl.data if bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
        )

        qlen = hidden_states.size(0)

        # ---- long-prefill GPU MoE path (env-gated) ----
        # When this layer's prefill batch is large enough, compute the MoE on
        # GPU by streaming this layer's raw MXFP4 weights H2D (one layer at a
        # time), instead of the CPU engine. Activations already live on the GPU
        # (attention path), so only weights stream. Short prefills stay on CPU.
        _gp_min = gpu_prefill_min_tokens()
        _resident = self._gpu_resident and self._resident_slot is not None
        if _resident or (
            _gp_min > 0
            and qlen >= _gp_min
            and not torch.cuda.is_current_stream_capturing()
        ):
            # ⭐ A78-4（2026-10-26）：NVFP4 的 `weight_scale_2` 由 **Triton 内核按每专家标量**乘上去
            #   （`ts_ptr` + `HAS_TS`），组尺度仍是 checkpoint 原样的 1 字节 e4m3
            #   ⇒ 组尺度不需要 fp32 化（那是 CPU 引擎的契约），显存零额外开销 ✓
            _ts13 = _ts2 = None
            _sk = None
            if getattr(self, "_xt_nvfp4", False) or getattr(
                    self.experts, "_nvfp4", False):
                _ts13, _ts2 = self._xt_nvfp4_ts(hidden_states.device)
                # ⭐ 显式告诉内核"尺度是 e4m3 字节"：真模型里尺度参数是 uint8，
                #   光看 dtype 无法区分 NVFP4(e4m3) 与 MXFP4(e8m0)（A78-4）
                _sk = "e4m3"
            from vllm.distributed import tensor_model_parallel_all_reduce
            from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer

            # Overlap: kick off the NEXT layer's H2D before doing this layer's
            # kernels, so the DMA runs concurrently with the tensor cores.
            _ov = (not _resident) and os.environ.get("XIAOTU_GPU_PREFETCH_AHEAD", "1") == "1"
            if not _resident:
                _start_pinned_prebuild()
            if os.environ.get("XIAOTU_DEBUG_QLEN") == "1":
                # Diagnostic: shows how the scheduler batches (one entry per MoE
                # forward). Used to explain why concurrent requests do not share
                # one weight stream (see docs/OPTIMIZATION_ROUND2.md §4).
                _dbg = _T.setdefault("qlen_log", [])
                if len(_dbg) < 3000:
                    _dbg.append(qlen)
                    if len(_dbg) % 43 == 0:
                        _tail = _dbg[-43:]
                        print(f"[qlen] pass#{len(_dbg) // 43} "
                              f"qlen={sorted(set(_tail))} layers={len(_tail)}",
                              flush=True)
            if _ov:
                try:
                    _nxt = _LAYERS.get(extract_layer_index(self.prefix) + 1)
                except Exception:
                    _nxt = None
                if _nxt is not None:
                    _nxt.prefetch_gpu_weights(hidden_states.device)

            _nvtx = os.environ.get("XIAOTU_NVTX") == "1"
            if _nvtx:
                torch.cuda.nvtx.range_push("xiaotu_moe")
            # ⭐【纯 GPU 模式 · 2026-10-26】把本层 MoE（含跨 rank 归约）包成
            #   `@eager_break_during_capture` 断点。原因（实测）：
            #   MUSA Triton 的内核在**捕获期 launch 会挂死**（两 rank 自旋、~380 线程；
            #   `[step-trace]` 显示 A/B/C 都过、挂在内核之后；按形状预编译也救不了）✗
            #   断点语义 = 捕获期「结束当前段 → eager 跑 → 记录供 replay → 开新段」
            #   ⇒ 除 MoE 外的层仍在图里 ✓；输出写**持久缓冲**保证 replay 地址稳定 ✓
            _caller = getattr(self, "_xt_break_caller", None)
            _outbuf = None
            if _caller is None and self._gpu_resident:
                def _fwd(hs, ids, wts, w13_, s13_, w2_, s2_, slot_, out_buf,
                         tp_, group_k_, sk_, ts13_, ts2_):
                    # ⭐【可选】把 Triton 内核放到**非捕获侧流**上跑，eager 窗口内 join 回来。
                    #   假设（HANDOFF §70）：`invalid argument` 来自 Triton launcher 触碰
                    #   "捕获中的流"；换到普通流即可规避。
                    _side = None
                    _main = None
                    if os.environ.get("XIAOTU_MOE_SIDE_STREAM", "0") == "1":
                        _side = getattr(self, "_xtu_moe_stream", None)
                        if _side is None:
                            _side = torch.cuda.Stream()
                            self._xtu_moe_stream = _side
                        _main = torch.cuda.current_stream()
                        _side.wait_stream(_main)
                    _ctx = (torch.cuda.stream(_side) if _side is not None
                            else contextlib.nullcontext())
                    with _ctx:
                        _o = gpu_moe_layer(
                            hs, ids, wts, w13_, s13_, w2_, s2_,
                        H=self.hidden_size, I=self.moe_intermediate_size,
                        K=self.top_k, device=hs.device, slot=slot_,
                        group_k=group_k_, scale_kind=sk_,
                        s13_ts=ts13_, s2_ts=ts2_,
                        out_dtype=(torch.float32 if sk_ else None),
                        swiglu_limit=float(self.swiglu_limit or 0.0),
                        out=out_buf,
                    )
                    if _side is not None:
                        _main.wait_stream(_side)
                    if tp_ > 1 and os.environ.get("XIAOTU_SKIP_AR", "0") != "1":
                        _o = tensor_model_parallel_all_reduce(_o)
                        out_buf.copy_(_o)
                        _o = out_buf
                    if os.environ.get("XIAOTU_BREAK_SYNC", "0") == "1":
                        # 诊断：MCCL 归约可能在自己的内部流上 fork ⇒ 捕获收尾报 unjoined
                        torch.cuda.synchronize()
                    return _o
                try:
                    from vllm.compilation.breakable_cudagraph import (
                        eager_break_during_capture as _ebdc)
                    _caller = _ebdc(_fwd)
                    print(f"[xtu-moe] {self.prefix}: MoE 已包成 eager-break"
                          f"（捕获期 eager、其余层进图）", flush=True)
                except Exception:   # noqa: BLE001
                    _caller = None
                self._xt_break_caller = _caller
            if _caller is not None:
                try:
                    _gb2 = self._graph_out_buffers(qlen, hidden_states.device)
                    if _gb2 is not None:
                        _outbuf = _gb2[0]
                except Exception:   # noqa: BLE001
                    _outbuf = None
                if _outbuf is None:
                    _caller = None
            w13, s13, w2, s2, st, L, tp = self._gpu_shard()
            if _resident:
                slot = self._resident_slot
                w13, s13, w2, s2 = slot.bufs  # 已在显存(K-major),不需再传 host 张量
            else:
                slot = self._slot if _ov else None
            self._slot = None
            if tp > 1:
                # Expert-parallel: remap to the local shard, then all-reduce the
                # partial MoE output across ranks.
                ids_l = topk_ids - st
                mask = (ids_l >= 0) & (ids_l < L)
                ids_l = torch.where(mask, ids_l, torch.full_like(ids_l, -1))
                wts_l = torch.where(mask, topk_weights, torch.zeros_like(topk_weights))
                _gk = int(getattr(self, "_xt_group_k", 32))
                if _caller is not None:
                    gpu_out = _caller(hidden_states, ids_l, wts_l, w13, s13, w2,
                                      s2, slot, _outbuf, tp, _gk, _sk, _ts13, _ts2)
                else:
                    gpu_out = gpu_moe_layer(
                        hidden_states, ids_l, wts_l, w13, s13, w2, s2,
                        H=self.hidden_size, I=self.moe_intermediate_size,
                        K=self.top_k, device=hidden_states.device, slot=slot,
                        group_k=_gk,
                        scale_kind=_sk, s13_ts=_ts13, s2_ts=_ts2,
                        out_dtype=(torch.float32 if _sk else None),
                        swiglu_limit=float(self.swiglu_limit or 0.0),
                    )
                    if os.environ.get("XIAOTU_SKIP_AR", "0") != "1":
                        gpu_out = tensor_model_parallel_all_reduce(gpu_out)
                # XIAOTU_SKIP_AR=1 仅供**计时诊断**:跳过这条每层跨 rank 归约
                # (MoE 输出 (T,4096) bf16 = 33MB,A100-PCIE 无 NVLink;数值会不完整)。
                # 用来量"每层归约"在 GPU 预填充里占多少;默认关,交付路径不受影响。
            else:
                _gk = int(getattr(self, "_xt_group_k", 32))
                if _caller is not None:
                    gpu_out = _caller(hidden_states, topk_ids, topk_weights,
                                      w13, s13, w2, s2, slot, _outbuf, tp, _gk,
                                      _sk, _ts13, _ts2)
                else:
                    gpu_out = gpu_moe_layer(
                        hidden_states, topk_ids, topk_weights,
                        w13, s13, w2, s2,
                        H=self.hidden_size, I=self.moe_intermediate_size,
                        K=self.top_k, device=hidden_states.device, slot=slot,
                        group_k=_gk,
                        scale_kind=_sk, s13_ts=_ts13, s2_ts=_ts2,
                        out_dtype=(torch.float32 if _sk else None),
                        swiglu_limit=float(self.swiglu_limit or 0.0),
                    )
            if _nvtx:
                torch.cuda.nvtx.range_pop()
            if gpu_out.dtype != hidden_states.dtype:   # fp32 累加 ⇒ 转回激活 dtype
                gpu_out = gpu_out.to(hidden_states.dtype)
            _maybe_profile()
            _sh = self._shared_out(hidden_states)
            if _sh is not None:
                gpu_out = gpu_out + _sh
            if os.environ.get("XIAOTU_DEBUG_L1") == "1" and not _capture_guard():
                print(
                    f"[dbg] {self.prefix} GPU-prefill qlen={qlen} tp={tp} "
                    f"out_mean={gpu_out.abs().mean().item():.3e}",
                    flush=True,
                )
            return gpu_out

        # routed experts:xiaotu 引擎(CPU)。engine.cpu_decode 内部 D2H->CPU forward_many->H2D。
        # 【图契约】优先用地址稳定的持久缓冲(见 _graph_out_buffers);超出捕获范围
        # 的形状(超长预填充)才临时分配。
        _gb = self._graph_out_buffers(qlen, hidden_states.device)
        if _gb is not None:
            out, _out_bf16, _ids_b, _wts_b, _mask_b = _gb
        else:
            out = torch.empty(
                qlen, self.hidden_size, dtype=torch.float32,
                device=hidden_states.device
            )
            _out_bf16 = None
            _ids_b = _wts_b = _mask_b = None
        stream = torch.cuda.current_stream()
        # 引擎 binding 的指针提取只认 numpy 数组或整数 data_ptr()(对 torch 张量返回
        # nullptr → cudaMemcpyAsync(nullptr) → invalid argument),故传 data_ptr()。
        h_bf16 = hidden_states.to(torch.bfloat16)
        if self._ep:
            # Expert-parallel: this rank only owns experts [st, st+L). Pairs that
            # belong to another rank get weight 0 — the engine's assignment scan
            # (`weights[ai] != 0.f`) drops them, so they cost nothing — and their
            # id is remapped into the local range so it can never index OOB.
            _st, _L = self._ep_start, self._ep_local
            if _ids_b is not None:
                # 就地写(地址稳定,图重放安全);见 _graph_out_buffers 的说明。
                _ids_b.copy_(topk_ids)          # int64 -> int32,copy_ 自带转换
                _ids_b.sub_(_st).clamp_(0, _L - 1)
                ids_i32 = _ids_b
                _wts_b.copy_(topk_weights)      # -> float32
                torch.logical_and(topk_ids >= _st, topk_ids < _st + _L, out=_mask_b)
                _wts_b.masked_fill_(~_mask_b, 0.0)
                wts_f32 = _wts_b
            else:
                ids_i32 = (topk_ids - _st).clamp_(0, _L - 1).to(torch.int32)
                _in = (topk_ids >= _st) & (topk_ids < _st + _L)
                wts_f32 = torch.where(
                    _in,
                    topk_weights,
                    torch.zeros((), dtype=topk_weights.dtype,
                                device=topk_weights.device),
                ).to(torch.float32)
        else:
            if _ids_b is not None:
                _ids_b.copy_(topk_ids)
                _wts_b.copy_(topk_weights)
                ids_i32, wts_f32 = _ids_b, _wts_b
            else:
                ids_i32 = topk_ids.to(torch.int32)
                wts_f32 = topk_weights.to(torch.float32)
        _t0 = time.perf_counter() if self._timing else 0.0
        # ⭐ A78-11（2026-10-26 实测）：引擎的 ISA 变体可能是 **NO_CUDA 构建**
        #   （`cpu_decode` 在 `#ifndef XIAOTU_MOE_NO_CUDA` 里；本机 CPU **没有 avx512**，
        #   loader 只能选 `_avx2`，而那个 .so 编译时带了 NO_CUDA）
        #   ⇒ 这里必须有**同步兜底**：D2H hidden → `cpu_prefill` → H2D out。
        #   代价：多两次同步拷贝（decode 变慢），但**数值与 prefill 同一条 CPU 路径**。
        #   ⚠️ 仅供 `enforce_eager`（本 bench 就是 eager）**：图捕获下同步拷贝会破坏捕获。
        if self._xt_cpu_decode is None:
            self._xt_cpu_decode = hasattr(self.engine, "cpu_decode")
            if not self._xt_cpu_decode:
                print(
                    "[xtu-moe] ⚠️ 本引擎无 cpu_decode（NO_CUDA 变体）⇒ decode 走同步 "
                    "cpu_prefill 兜底；仅 eager 模式可用（A78-11）",
                    flush=True,
                )
        if self._xt_cpu_decode:
            self.engine.cpu_decode(
                stream.cuda_stream, qlen, self.top_k,
                h_bf16.data_ptr(),
                ids_i32.data_ptr(),
                wts_f32.data_ptr(),
                out.data_ptr(),
            )
        else:
            # ⚠️ 必须传 **data_ptr() 整数**：binding 的 `as_ptr()` 对 torch 张量
            #   返回 nullptr ⇒ 引擎解引用空指针 **SIGSEGV**（2026-10-26 实测，
            #   与 cpu_decode 那条注释同源）。张量要保持存活到调用结束 ⇒ 全用局部变量。
            _xo = torch.empty((qlen, self.hidden_size), dtype=torch.float32)
            _xh = h_bf16.to("cpu").contiguous()
            _xi = ids_i32.to("cpu").contiguous()
            _xw = wts_f32.to("cpu").contiguous()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "cpu_prefill 兜底不支持 CUDA graph 捕获（A78-11）；"
                    "请用 --enforce-eager，或重编一个带 CUDA 的 ISA 变体"
                )
            self.engine.cpu_prefill(
                qlen, self.top_k,
                _xi.data_ptr(), _xw.data_ptr(),
                _xh.data_ptr(), _xo.data_ptr(),
            )
            out.copy_(_xo.to(out.device))
            # ⚠️ EP：引擎的 `cpu_decode` 会用 /dev/shm 做跨 rank 部分和归约，
            #   而 `cpu_prefill` **不做** ⇒ 兜底必须自己补一次 all-reduce，
            #   否则每个 rank 只给出"自己那半专家"的贡献（数值错一半，且不会报错）。
            #   下面的 `_need_allreduce` 只在 qlen > _ep_shm_tokens 时才触发，
            #   解码头恰好不触发 ⇒ 必须在这里显式做一次。
            if (os.environ.get("XIAOTU_MOE_SELFCHECK", "0") == "1"
                    and not getattr(self, "_xt_did_selfcheck", False)):
                # 只有**真的算过**才打已完成标记；没本地专家就下次再试
                self._xt_did_selfcheck = bool(
                    self._xt_selfcheck(_xh, _xi, _xw, _xo))
            #   条件与引擎自身一致：**只有 qlen ≤ _ep_shm_tokens 时**引擎才走 shm 归约，
            #   而这正是尾段 `_need_allreduce` **不会**触发的区间 ⇒ 在这里补一次；
            #   qlen 更大时交给尾段的 NCCL all-reduce（否则会**叠加两次** = 数值 2×）✗
            _local_all = None
            if (os.environ.get("XIAOTU_MOE_SELFCHECK", "0") == "1"
                    and not getattr(self, "_xt_did_selfcheck2", False)
                    and qlen <= 64 and float(hidden_states.abs().max()) > 0.0):
                # ⚠️ 早先版本要求"10 个专家全在本 rank"才跑 —— 概率 ~1e-3，
                #    实测**一次都没触发** ✗ ⇒ 路由/共享专家部分**无条件**跑；
                #    归约比对仍然只看"全本地"的 token（那是它的正确前提）。
                _mask = (ids_i32 >= 0) & (ids_i32 < self.experts.num_experts)
                _local_all = (_mask.sum(dim=1) == ids_i32.shape[1])
                self._xt_selfcheck2(hidden_states, ids_i32, wts_f32, out, _local_all)
                self._xt_did_selfcheck2 = True
            if self._ep and qlen <= self._ep_shm_tokens:
                from vllm.distributed import tensor_model_parallel_all_reduce

                out.copy_(tensor_model_parallel_all_reduce(out))
            if _local_all is not None and bool(_local_all.any()):
                _ti2 = int(_local_all.nonzero()[0, 0])
                if (getattr(self, "_xt_last_ref", None) is not None
                        and _ti2 == getattr(self, "_xt_last_ti", -1)):
                    _got = out[_ti2].to(torch.float32)
                    _ref = self._xt_last_ref
                    _rel = float((_got - _ref).abs().mean()
                                 / _ref.abs().mean().clamp_min(1e-8))
                    print(f"[xtu-selfcheck2] 归约后 token {_ti2}: 相对误差={_rel:.4e} "
                          f"{'✅ 归约没问题' if _rel < 5e-2 else '⛔ 归约把值搞坏了'}",
                          flush=True)
        _maybe_profile_decode()
        if self._timing:
            _T["eng"] += time.perf_counter() - _t0
            _T["calls"] += 1
            if _T["calls"] == 1:
                _T["t0"] = time.perf_counter()
            if _T["calls"] % 43 == 0:
                _now = time.perf_counter()
                _step = _now - _T["t0"]
                print(
                    f"[xiaotu-timing] engine={_T['eng']:.3f}s step_wall={_step:.3f}s "
                    f"other={_step - _T['eng']:.3f}s qlen={qlen} "
                    f"engine_frac={_T['eng'] / max(_step, 1e-9):.2f}",
                    flush=True,
                )
                _T["eng"] = 0.0
                _T["t0"] = _now
        if _out_bf16 is not None:
            # 就地 dtyped copy ⇒ 结果落在**固定地址**的持久缓冲上,返回其视图。
            _out_bf16.copy_(out)
            final_hidden_states = _out_bf16
        else:
            final_hidden_states = out.to(hidden_states.dtype)
        # EP 的跨 rank 求和:默认已在引擎的 host 回调里用 /dev/shm 完成
        # (`XIAOTU_MOE_EP_SHM=1`,qlen ≤ _ep_shm_tokens 时);只有超出该容量
        # (极长 prefill 走 CPU 路径)才回退到 NCCL all-reduce。
        _need_allreduce = self._ep and qlen > self._ep_shm_tokens
        if _need_allreduce:
            from vllm.distributed import tensor_model_parallel_all_reduce

            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )
        if os.environ.get("XIAOTU_DEBUG_L1") == "1" and not _capture_guard():
            nz = (final_hidden_states != 0).sum().item()
            nn = final_hidden_states.numel()
            fn = (final_hidden_states != final_hidden_states).sum().item()  # NaN
            tids = topk_ids[:6].cpu().tolist()
            tw = topk_weights[:6].cpu().tolist()
            hin = hidden_states.abs().mean().item()
            ex = self.experts
            w13m = float(ex.w13_weight.abs().float().mean().item())
            w2m = float(ex.w2_weight.abs().float().mean().item())
            print(f"[dbg] {self.prefix} out-nz={nz}/{nn} NaN={fn} "
                  f"mean={final_hidden_states.abs().mean().item():.3e} "
                  f"hin={hin:.3e} w13m={w13m:.3e} w2m={w2m:.3e} "
                  f"topk={tids} wts={tw}", flush=True)

        _sh = self._shared_out(hidden_states)
        if _sh is not None:
            if _out_bf16 is not None:
                # 就地累加(缓冲区是持久视图,不能再产生新张量)
                final_hidden_states.add_(_sh)
            else:
                final_hidden_states = final_hidden_states + _sh
        return final_hidden_states


class HybridDeepseekV4ForCausalLM(dv4_nvidia.DeepseekV4ForCausalLM):
    """OOT 覆盖 DeepseekV4ForCausalLM:decoder 层的 ffn 全部是 CpuXiaotuMoE(CPU MoE + GPU 注意力)。

    复用主线 model_cls=DeepseekV4Model(其层 ffn 因 register() 把 DeepseekV4MoE 换成
    CpuXiaotuMoE 而成为 CPU 版);引擎在 process_weights_after_loading ->
    model.finalize_mega_moe_weights() -> layer.ffn.finalize_mega_moe_weights() 里构造。
    """


def register():
    """vllm.general_plugins 入口:OOT 覆盖 DS-V4 arch(主线零改动)。

    可用 `XIAOTU_OOT_OVERRIDE=0` 关掉本覆盖,改用 **lk/lvllm 同构**的路径:
    主线的 `RoutedExperts` 保持不动,由 `mixed_experts.register_mixed_cpu_backend()`
    把 `Mxfp4MoeBackend.CPU` 指向 `xiaotu_moe` 引擎(等价于上游 PR #56118 的
    `VLLM_EXPERTS_LOAD_DEVICE=cpu` + CPU 后端)。两条路的数值应一致,但 lk 那条
    能原生吃下主线的 cudagraph / prefix caching / chunked prefill / spec decode。
    """
    if os.environ.get("XIAOTU_OOT_OVERRIDE", "1") == "0":
        print(
            "[vllm-xtu-moe] OOT DS-V4 model override DISABLED "
            "(XIAOTU_OOT_OVERRIDE=0) -> mainline RoutedExperts + xiaotu CPU backend",
            flush=True,
        )
        return None

    # ⭐ 2026-10-26【平台适配一律留在插件侧】把原本"要改 vLLM 树"的兼容逻辑
    #   在这里 monkeypatch 打上（见 `compat_patches.py` 的模块说明）：
    #   目标形态是 **vLLM 树 0 处本项目改动** ⇒ rebase 无负债 ✓
    #   ⚠️ 幂等；失败一律只告警、不影响 FP8 主线（补丁内部还各自要求 XIAOTU_NVFP4=1）
    try:
        from .compat_patches import apply_all as _apply_upstream_compat

        _patches = _apply_upstream_compat()
        if _patches:
            print(f"[vllm-xtu-moe] upstream compat patches applied: {_patches}",
                  flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[vllm-xtu-moe] compat patches skipped: {type(_e).__name__}: {_e}",
              flush=True)

    if not getattr(dv4_nvidia, "_xiaotu_patched", False):
        dv4_nvidia.DeepseekV4MoE = CpuXiaotuMoE

        # ⭐ 2026-10-26：`qwen4_exp` 架构**不用** DeepseekV4MoE，而是它自己的
        #   `Qwen4ExpSparseMoeBlock`（内部是 vLLM 的 FusedMoE）⇒ 之前我们的引擎
        #   **根本没被使用**（实测日志出现 `fused_moe.py:1213 Using default MoE config`
        #   之后 muDNN 报 NOT_SUPPORTED ⇒ 挂起）。
        #   该块与 `CpuXiaotuMoE` **同签名** `(vllm_config, prefix=...)` ⇒ 直接替换 ✓
        try:
            import vllm.models.qwen4_exp.nvidia.model as _q4
            if hasattr(_q4, "Qwen4ExpSparseMoeBlock"):
                _q4.Qwen4ExpSparseMoeBlock = CpuXiaotuMoE
                print("[vllm-xtu-moe] registered OOT override of "
                      "Qwen4ExpSparseMoeBlock -> CpuXiaotuMoE", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[vllm-xtu-moe] qwen4_exp override skipped: {_e}", flush=True)
        dv4_nvidia._xiaotu_patched = True  # 幂等

    ModelRegistry.register_model(
        MODEL_ARCH,
        f"{__name__}:HybridDeepseekV4ForCausalLM",
    )
    print(
        f"[vllm-xtu-moe] registered OOT override of {MODEL_ARCH} "
        f"(DeepseekV4MoE -> CpuXiaotuMoE)",
        flush=True,
    )
    return None


if __name__ == "__main__":
    register()
    print("registered OK")
