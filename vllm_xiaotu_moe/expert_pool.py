# vllm_xiaotu_moe/expert_pool.py
"""专家驻留池(缩水版 MVP,SPEC_expert_residency v1 §8)。

不变式:
* 总开关 XIAOTU_PIN=0 默认关(未开启时本模块一行都不执行);
* 池 = KV profiling 之前(torch 树内,权重加载期)分配的 plane 张量(MBT 科目记账见日志行);
* 槽加载 = kmajor_from_engine_shards_noraw(e0=e, e1=e+1, dst=槽 view) —— 零新 C++;
* slot-map 编码(每层一个 device int32 [E_local]):未钉 → 值 = 本地 id(< E_local,兼作 CPU id);
  钉住 → 值 = E_local + 槽号(≥ E_local 即"交 GPU");kernel/CPU 分流只靠这个数值。
Phase A(本轮):静态底表(XTUP bin,源 = P1 真实流);Phase B:LRU 换入线程只改 plane + 表内容,不改图。
"""
from __future__ import annotations

import os
import struct

import torch

_ENV_ON = "XIAOTU_PIN"
_ENV_TABLE = "XIAOTU_PIN_TABLE"
_ENV_GIB = "XIAOTU_PIN_GIB"
# ---- 【运行中自适应】的三个开关(全部默认**关** ⇒ 不开时本模块行为与改动前一致)----
_ENV_WINDOW = "XIAOTU_PIN_WINDOW"     # 每多少个 forward 重新规划一次;0/缺省 = 永不(关)
_ENV_HYST = "XIAOTU_PIN_HYST"         # 滞回系数 θ:新候选须 > 被换出者×(1+θ)
_ENV_MAXSWAP = "XIAOTU_PIN_MAXSWAP"   # 单次换入上限;<=0 ⇒ 取 max(1, nslots//4)


def plan_swap(resident, counts, nslots: int, *, hyst: float = 0.25,
              maxswap: int = 0, min_hits: int = 1, preserve_oldest: bool = False):
    """⭐ **纯策略**(无 torch / 无 device ⇒ 可在 CPU 上机械测试)。

    给定:当前驻留集合 `resident`(专家号列表,长度 = nslots)与窗口内命中计数
    `counts`(dict 专家→次数),给出 `(evict, load)`:要把哪些换出、哪些换入。

    规则(每条都可机械检查):
      1. **换出优先挑最冷的**:按 `(counts.get(e,0), 入池序)` 升序 ⇒ 取前 k 个;
      2. **候选** = 不在驻留集合里、且窗口计数 ≥ `min_hits` 的专家,按计数降序;
      3. ⭐ **滞回**:候选的计数必须 **> 被它顶掉的那个驻留者计数 × (1+hyst)**
         ⇒ 否则**不动**(防止计数接近时来回抖,把 PCIe 打满);
      4. ⭐ **限流**:一次最多换 `k = maxswap if maxswap > 0 else max(1, nslots//4)` 个;
      5. `resident` 为空或 `nslots<=0` ⇒ 不换(空池不做任何事)。

    ⭐⭐ **`preserve_oldest`(2026-10-15 自 SGLang 移植,`expert_pack.py:473/559`)**:
       平局时的换出方向。默认 `False` = 同计数时**先换最旧的**(LRU ✓);
       置 `True` ⇒ **先换最新的**(MRU ✓)。语义:**prefill 阶段刚摸到的专家很可能是一次性的**
       ⇒ 该保住**解码期**建立起来的驻留集,而不是被 prefill 的长尾挤掉 ✓
       (SGLang 原文 `preserve_oldest=is_prefill`,且有单测钉死该语义 ✓)

    ⭐⭐ **不变式(机械可查)**:换出集合**绝不包含**本窗口的热点候选 `load`
       —— 由"候选按计数降序、受害按计数升序"**结构性保证** ✓;
       下面用一条 `assert` 把它钉成**运行时不变式**(零成本,契合 fail-closed 哲学 ✓
       参照 SGLang `_victim_slot` 的 `protected` + raise,`expert_pack.py:477-486`)。

    ⚠️ 本函数**只做决策**,不碰任何张量 ⇒ "策略对不对"与"接线对不对"可以分开验证 ✓
    """
    E_n = int(nslots)
    if E_n <= 0 or not resident:
        return [], []
    res = list(resident)
    k = int(maxswap) if int(maxswap) > 0 else max(1, E_n // 4)
    # ⭐ 平局方向:preserve_oldest ⇒ i 倒序(最新者先被换出)
    cold = sorted(range(len(res)),
                  key=(lambda i: (int(counts.get(res[i], 0)), -i)) if preserve_oldest
                  else (lambda i: (int(counts.get(res[i], 0)), i)))[:k]
    in_res = set(res)
    cand = sorted(((e, int(c)) for e, c in counts.items()
                   if e not in in_res and int(c) >= int(min_hits)),
                  key=lambda kv: -kv[1])[:k]
    evict, load = [], []
    for i, (e, c) in enumerate(cand):
        victim = res[cold[i]]
        # ⭐ 滞回:只有"明显更热"才值得付出一次换入代价
        if c > int(counts.get(victim, 0)) * (1.0 + float(hyst)):
            evict.append(victim)
            load.append(e)
    # ⭐ 不变式:绝不换出本窗口比任何 load 候选更热的专家(结构性保证的运行期复核)
    assert not (set(evict) & set(load)), f"plan_swap: evict∩load={set(evict) & set(load)}"
    assert all(int(counts.get(v, 0)) <= int(counts.get(l, 0)) for v, l in zip(evict, load)), \
        "plan_swap: 受害者比装入者更热(选取方向反了)"
    return evict, load


def pin_enabled() -> bool:
    return os.environ.get(_ENV_ON, "0") == "1"


def load_pin_table(path: str) -> dict[int, list[int]]:
    """XTUP v0:header<magic4,ver,n_layers,E,n_entries> + entries(layer u16, expert u16 频率降序)。

    ⚠️ 头长 = `struct.calcsize("<4sIIII")` = **20** 字节(不是 16)。原先按 16 偏移读条目 ⇒
    每条读到的是**上一条**(最后一条丢失),且第 0 条读到的是头部尾部 4 字节的**垃圾键**
    (它的 layer 值恰等于 n_entries,永不匹配任何真实层)⇒ 净效果 = 表少最后 1 条 + 1 条垃圾。
    量级无害(实测 342→341 真条目,命中率 23.5%→23.4%),但属真 bug,2026-10-14 修正。
    """
    with open(path, "rb") as f:
        buf = f.read()
    magic, ver, n_layers, e_max, n_ent = struct.unpack_from("<4sIIII", buf, 0)
    assert magic == b"XTUP" and ver == 1, f"bad pin table {magic!r} v{ver}"
    hdr = struct.calcsize("<4sIIII")
    out: dict[int, list[int]] = {}
    for i in range(n_ent):
        l, e = struct.unpack_from("<HH", buf, hdr + 4 * i)
        out.setdefault(int(l), []).append(int(e))
    return out


class LayerPool:
    """单层池。nslots = 该层钉住数(本 rank 视角,已按 expert_map 折算)。"""

    def __init__(self, engine, device: torch.device, hidden: int, inter: int,
                 group_k: int, n_local: int, pinned_local: list[int], layer_idx: int,
                 layout: str = "mxfp4", topk: int = 0):
        """⭐ 2026-10-15:`layout` 让同一个池同时服务两条后端的**不同槽布局**
        (`mxfp4` = `gpu_prefill`;`fp8` = `gpu_prefill_fp8`)✓

        为什么必须分开(⛔ 不能共用一套形状):
          * **mxfp4**:权重 4-bit 打包 ⇒ 行维减半(`H//2`),尺度是 e8m0 **uint8**;
          * **fp8**  :权重 e4m3 全字节(`H`),尺度是 **fp32**(4 B/项)⇒ 槽按
            **uint8 + 末维 ×4** 申请再 `.view(torch.float32)`(与 `_fp8_tile_ring` 逐字同构);
          ⚠️ 若两者混用 ⇒ **静默越界/错读**(正是 `_fp8_tile_ring` 注释里警告的那类)✗
        """
        from vllm_xiaotu_moe import gpu_prefill as _gp
        self.engine, self.dev = engine, device
        self.H, self.I, self.gk = int(hidden), int(inter), int(group_k)
        self.n_local, self.layer_idx = int(n_local), int(layer_idx)
        self.layout = str(layout)
        self.pinned = list(pinned_local)
        self.nslots = len(self.pinned)
        # ---- 【运行中自适应】状态(默认不激活;是否启用由 XIAOTU_PIN_WINDOW 决定)----
        self.expert_of: list[int] = [-1] * self.nslots   # 槽 → 专家(swap_out 必需的反向表)
        self._hits = None        # [E_local] int32 device 计数(懒建;下面会在 nslots>0 时**立即**建)
        self._side = None        # 换入用的侧流(懒建)
        # 【2026-10-16 goal-B · 图安全】`_win` 原为 **Python int** ✗ ⇒ 在 CUDA graph 里
        # `self._win += 1` 只在 **capture 期执行一次**,replay 不会加 ⇒ 窗口永远攒不满
        # ⇒ 自适应恒不触发 ✗(实测证据:B550 图开时 `swapped_in=0`)
        # ⇒ 改为 **device 侧标量张量** `_win_t`:它是 device kernel 的一部分,
        #    被捕获后**每步 replay 都会重放** ⇒ 计数随步数增长 ✓(SPEC_expert_residency §8.4)
        self._win_t = None       # int64 标量 device 计数(距上次自适应跑了几个 forward)
        self.swaps_in = 0        # 可观测:累计换入/换出次数
        self.swaps_out = 0
        if self.nslots == 0:
            self.planes = None
            self.slotmap = torch.arange(self.n_local, dtype=torch.int32, device=device)
            return
        # ⭐ 【图安全】两个累加器必须**在捕获之前**建好(捕获区内不许再有 host 分配 ✗):
        #    `_hits` 与 `_win_t` 都建在 device 上 ⇒ `note_routing` 变成**纯 device 调用** ✓
        self._hits = torch.zeros(self.n_local, dtype=torch.int32, device=device)
        self._win_t = torch.zeros((), dtype=torch.int64, device=device)
        S = self.nslots
        if self.layout == "fp8":
            # ⭐ 与 `gpu_prefill_fp8._fp8_tile_ring:212-215` 的 shapes **逐字相同** ✗
            #    (尺度末维 ×4 = fp32 的 4 字节;见该函数的两条警告)
            b = 128                              # FP8_BLOCK 常见值;gpu_prefill_fp8 正式取它
            try:
                from vllm_xiaotu_moe.gpu_prefill_fp8 import FP8_BLOCK as b  # type: ignore
            except Exception:  # noqa: BLE001
                pass
            self.planes = (
                torch.empty((S, self.H, 2 * self.I), dtype=torch.uint8, device=device),
                torch.empty((S, self.H // b, (2 * self.I) // b * 4), dtype=torch.uint8, device=device),
                torch.empty((S, self.I, self.H), dtype=torch.uint8, device=device),
                torch.empty((S, self.I // b, self.H // b * 4), dtype=torch.uint8, device=device),
            )
        else:
            rb13, rb2 = self.H // 2, self.I // 2   # 与 noraw:1383 逐字同式
            self.planes = (
                torch.empty((S, rb13, 2 * self.I), dtype=torch.uint8, device=device),
                torch.empty((S, self.H // self.gk, 2 * self.I), dtype=torch.uint8, device=device),
                torch.empty((S, rb2, self.H), dtype=torch.uint8, device=device),
                torch.empty((S, self.I // self.gk, self.H), dtype=torch.uint8, device=device),
            )
        self.slot_of = {e: i for i, e in enumerate(self.pinned)}
        self.expert_of = list(self.pinned)   # ⭐ 反向表:与 slot_of 同步建立
        sm = torch.arange(self.n_local, dtype=torch.int32, device=device)
        for e, s in self.slot_of.items():
            sm[e] = self.n_local + s
        self.slotmap = sm.to(device=device, non_blocking=False)
        self._slot = None          # ⭐ 2026-10-20:复用 as_slot(),不要每层每次 forward 新建 Event
        self._load_all(_gp)
        gib = sum(p.numel() for p in self.planes) / 2**30
        print(f"[xtu-pin] layer{layer_idx}: slots={S} pool={gib:.2f} GiB "
              f"layout={self.layout} (树内分配,KV profiling 前)", flush=True)
        # ⭐ 池容量最小守卫(2026-10-15 自 SGLang 移植:`expert_pack.py:376-380`
        #    在 `slot_count < top_k` 时直接 `raise MemoryError`)✓
        #    我们的设计**有意允许** `nslots < topk`(SPEC §1.1:槽缺席只损失 bitmap 判定的
        #    µs 级成本)⇒ 改成**显式告警**而非抛错,但把"这一层的池连一个完整 top-k 工作集
        #    都装不下"暴露出来 —— 否则现场只会看到"命中率莫名很低" ✗
        if 0 < int(topk) and S < int(topk):
            print(f"[xtu-pin] ⚠️ layer{layer_idx}: slots={S} < top_k={int(topk)} "
                  f"⇒ 本层池**装不下一个完整 top-k 工作集**,命中率上界被结构性压到 "
                  f"{100.0 * S / int(topk):.0f}% 以下(不是策略问题,是先给少了)✗", flush=True)

    def _load_slot(self, gp, e: int, s: int):
        """把专家 `e` 装进槽 `s`(⭐ 后端分派:两条路的**装载器签名不同** ✗)。

        * `mxfp4` ⇒ `gpu_prefill.kmajor_from_engine_shards_noraw(..., gk, dst=, e0=, e1=)`
          返回 dst(或 None);
        * `fp8`   ⇒ `gpu_prefill_fp8._fp8_kmajor_tiled(engine, dev, H, I, n_local, e0, e1, dst)`
          —— **没有 `group_k` 参数**(FP8 块固定 128),且 `dst` 是**位置参数** ✓
        """
        dst = tuple(p[s:s + 1] for p in self.planes)
        if self.layout == "fp8":
            # ⛔⛔ 2026-10-15 实测修正:原写作 `gp._fp8_kmajor_tiled(...)` ✗ ——
            #     实测报错 `AttributeError: module 'vllm_xiaotu_moe.gpu_prefill'
            #     has no attribute '_fp8_kmajor_tiled'` ⇒ 传进来的 `gp` 是
            #     【generic 模块】,而 `_fp8_kmajor_tiled` **只定义在
            #     `gpu_prefill_fp8.py:542`**,generic 模块**没有 re-export** ✗
            #     ⇒ 改为**按定义模块显式取**,与 `gp` 是谁无关 ✓(鲁棒)
            from vllm_xiaotu_moe import gpu_prefill_fp8 as _f8  # noqa: PLC0415
            return _f8._fp8_kmajor_tiled(  # noqa: SLF001  同包私有孪生 ✓
                self.engine, self.dev, self.H, self.I, self.n_local,
                e0=int(e), e1=int(e) + 1, dst=dst)
        return gp.kmajor_from_engine_shards_noraw(
            self.engine, self.dev, self.H, self.I, self.n_local, self.gk,
            dst=dst, e0=int(e), e1=int(e) + 1)

    def _load_all(self, gp) -> None:
        """槽加载 = 逐专家复用 noraw 子路径(e0..e1=单专家,dst=槽 view)。

        ⛔ **fail-closed**(2026-10-19 审计 MAJOR 修复):`_load_slot` 在引擎缺少
        `copy_hostbuf_to_device_2d` 等情况下会**返回 None 且不抛**。原实现丢弃返回值
        ⇒ `torch.empty` 的 planes 原样参与计算 ⇒ 输出是垃圾,而日志一个字都没有 ✗。
        ⇒ 现在任一槽装载失败 ⇒ 直接抛错(建池失败 ⇒ 上层明确失败,绝不静默算错)。
        """
        for e, s in self.slot_of.items():
            if self._load_slot(gp, e, s) is None:
                raise RuntimeError(
                    f"expert_pool: 槽装载失败(layer{self.layer_idx} expert={e} slot={s})"
                    f" ⇒ 拒绝使用未初始化的 plane")
        torch.cuda.synchronize()

    def as_slot(self):
        # 喂给 gpu_moe_layer 的 slot-like:bufs=planes,ready=构建期已完成事件。
        # ⭐ 2026-10-20:旧实现**每次调用**都 new 一个 Event;PIN 路径每层每 forward 调一次
        #   ⇒ 长时间压测累计大量 CUDA event(并可能留下未释放资源)。槽内容不变,
        #   复用同一个 _ReadySlot 即可;`busy` 由 gpu_moe_layer 每次覆盖。
        if self._slot is None:
            ev = torch.cuda.Event()
            ev.record()   # 前面 _load_all 已 synchronize,这里只是保序凭证
            self._slot = _ReadySlot(self.planes, ev)
        return self._slot

    # ---- 【运行中自适应】两个入口:只改 plane 内容 + 表内容,不改图、不改池子大小 ----
    def swap_out(self, s: int) -> int:
        """把槽 s 里的专家退回 CPU 路径;返回被换出的专家(-1 = 槽本来就空)。

        ⛔⛔ **红线:绝不释放 plane**(不 del / 不 resize / 不 empty / 不 empty_cache)——
            plane 是 boot 期、**KV profiling 之前**分配的**常量**;一旦释放,
            `staging_bytes()` 的记账口径("池常驻、且大小不变")就与事实分叉
            ⇒ §600/§601 那类"预检说够、实际 OOM"会重演 ✗
        ⚠️ 顺序:先改 `slotmap`(内核据此分流),**之后**才允许复用该槽 ✓
        """
        e = self.expert_of[s]
        if e is None or e < 0:
            return -1
        self.slotmap[e] = e          # ⭐ 值 < E_local ⇒ 内核把它分流到 CPU 路径
        self.expert_of[s] = -1
        self.swaps_out += 1
        return int(e)

    def note_routing(self, topk_ids) -> None:
        """零同步累加本窗口的专家命中(device 侧 index_add_)⇒ 不打断流水 ✓

        ⭐ 【2026-10-16 goal-B · 图安全】本方法现在是**纯 device 调用** ✓:
        * `_hits` / `_win_t` 已在 `__init__` 里**预先建在 device** ⇒ 这里**不再新建张量** ✓
          (捕获区内做 host 分配会把图作废 ✗)
        * `_win_t` 是 device 标量 ⇒ 本方法整体可被 CUDA graph 捕获,
          **每步 replay 都会重放** ⇒ 命中计数与窗口计数都随步数增长 ✓
        ⇒ 因此调用方**必须**在捕获区**之外**的判断里也调用它(即无条件调用)✓
        """
        if self.nslots == 0 or self._hits is None or self._win_t is None:
            return
        idx = topk_ids.reshape(-1).to(torch.long)
        # ⛔⛔【2026-10-19 · 图捕获期 `device-side assert` 的**真凶**】
        #   profile/warmup/捕获期的路由 id 是 **-1 哨兵**;本方法为了"图内也要计数"
        #   被**移到捕获守卫之外**无条件调用 ⇒ `index_add_(idx=-1)` 越界 ✗
        #   ⇒ `CUDA error: device-side assert triggered`(B1049 逐字复现)✗
        #   ⭐ 修法与 B706/B889 同一族:**device 侧把无效 id 归零 + 权重 0**,
        #     不删除、不改形状、不做 `.item()` ⇒ 仍可被 CUDA graph 捕获 ✓
        _n = self._hits.numel()
        _bad = (idx < 0) | (idx >= _n)
        _safe = torch.where(_bad, torch.zeros_like(idx), idx)
        _w = torch.where(_bad, torch.zeros_like(idx, dtype=torch.int32),
                         torch.ones_like(idx, dtype=torch.int32))
        self._hits.index_add_(0, _safe, _w)
        self._win_t += 1

    def swap_in(self, gp, new_expert: int, s: int):
        """把专家 `new_expert` 压进槽 s;成功返回 s,失败/已在池中返回 None。

        ⚠️ 顺序是**硬要求**:① 侧流写 plane[s] → ② 主流 `wait_event` → ③ **最后**翻表。
            表一旦翻,后续 forward 就会去读 plane ⇒ 内容必须先就绪 ✓
            用 `wait_event` 而非 `synchronize()`,不阻塞主机 ✓
        """
        if new_expert in self.expert_of:
            return None                              # 已在池里,别重复搬(纯 Python 判,无 D2H)✓
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.dev)
        st = torch.cuda.current_stream(self.dev)
        self._side.wait_stream(st)                   # 该槽上一次的使用者(主流)先完成
        with torch.cuda.stream(self._side):
            got = self._load_slot(gp, new_expert, s)   # ⭐ 后端分派(mxfp4 / fp8)✓
            ev = torch.cuda.Event()
            ev.record(self._side)
        if got is None:
            return None
        st.wait_event(ev)                            # ⭐ 保序:内容先就绪
        self.slotmap[new_expert] = self.n_local + s  # ⭐ **最后**才翻表
        self.expert_of[s] = int(new_expert)
        self.swaps_in += 1
        return s

    def maybe_adapt(self, gp, *, staging_active: bool = True) -> int:
        """按窗口自适应一次;返回实际换入的个数(0 = 什么都没动)。

        ⚠️ **默认 `staging_active=True` ⇒ 默认不做任何事(fail-closed)**:
            ⛔ 本项目**目前没有**可查询的 "staging 突发" 标志(全仓 grep 只有注释里的
            `staging_active` 字样)⇒ 既然**无法机械判定"现在安不安全"**,就不许猜 ✗
            ⇒ 调用方必须**显式**传入它确认过的 `staging_active=False`;
              或在 `gpu_prefill` 里先落一个真标志(那是**下一步**,与"接线"同批)✓
        ⚠️ 每窗口**只做一次** D2H(`.tolist()`);`index_add_` 全程在 device 上 ✓
        """
        if self.nslots == 0 or self._hits is None or staging_active:
            return 0
        w = int(os.environ.get(_ENV_WINDOW, "0") or 0)
        if w <= 0:
            return 0
        # ⭐ 【2026-10-16 goal-B · 图安全】窗口计数现在来自 **device 标量**
        #    ⇒ 图内 re播放也会累加 ✓;`item()` 是本方法**唯一**一次 device→host 同步
        #    (与 `_hits.tolist()` 同一批,按窗口摊薄 ✓,不进每步关键路径 ✓)
        if self._win_t is None or int(self._win_t.item()) < w:
            return 0
        counts = self._hits.tolist()                 # ⭐ 唯一一次同步,按窗口摊薄
        self._hits.zero_()
        self._win_t.zero_()
        cd = {e: c for e, c in enumerate(counts) if c}
        # ⭐ 2026-10-18 诊断(gated,默认关):把"热度到底有多集中"打成一行 ——
        #    没有这个数,就无法判断"自适应 0 次"是【路由均匀】还是【策略/装载坏了】✗
        #    只在窗口边界执行一次,且不碰任何 CUDA API ⇒ 对被测物零影响 ✓
        if os.environ.get("XIAOTU_PIN_DEBUG", "0") == "1" and cd:
            tot = sum(cd.values())
            top = sorted(cd.values(), reverse=True)
            cum = []
            for n in (8, 16, 32, 64, 128, 256):
                cum.append(f"top{n}={100.0 * sum(top[:n]) / tot:.1f}%")
            print(f"[xtu-pin][dist] layer{self.layer_idx}: experts={len(cd)}/"
                  f"{self.n_local} events={tot} slots={self.nslots} " + " ".join(cum),
                  flush=True)
        evict, load = plan_swap(
            self.expert_of, cd, self.nslots,
            hyst=float(os.environ.get(_ENV_HYST, "0.25") or 0.25),
            maxswap=int(os.environ.get(_ENV_MAXSWAP, "0") or 0),
            # ⭐ SGLang 的 `preserve_oldest=is_prefill`(`expert_pack.py:559`):
            #    置 1 ⇒ 平局时换出**最新**用过的(它们多是 prefill 的一次性路由),
            #    保住解码期建立的驻留集 ✓ 默认 0 = LRU(既有行为,逐字不变)✓
            preserve_oldest=(os.environ.get("XIAOTU_PIN_PRESERVE_OLDEST", "0") == "1"))
        _dbg = os.environ.get("XIAOTU_PIN_DEBUG", "0") == "1"
        if _dbg:
            print(f"[xtu-pin][plan] layer{self.layer_idx}: evict={len(evict)} "
                  f"load={len(load)} slots={self.nslots}", flush=True)
        n = 0
        for victim, e in zip(evict, load):
            if victim not in self.expert_of:
                continue
            s = self.expert_of.index(victim)
            self.swap_out(s)
            try:
                _got = self.swap_in(gp, e, s)
            except Exception as _se:  # noqa: BLE001
                # ⛔ 原先此异常被调用方 `except: pass` 吞掉 ⇒ 现场只看到"0 次换入"
                if _dbg:
                    print(f"[xtu-pin][swap-err] layer{self.layer_idx} expert{e} slot{s}: "
                          f"{type(_se).__name__}: {str(_se)[:160]}", flush=True)
                raise
            if _got is not None:
                n += 1
            elif _dbg:
                print(f"[xtu-pin][swap-none] layer{self.layer_idx} expert{e} slot{s}: "
                      f"loader returned None", flush=True)
        if n:
            print(f"[xtu-pin-adapt] layer{self.layer_idx}: swapped_in={n} "
                  f"total_in={self.swaps_in} total_out={self.swaps_out}", flush=True)
        return n


def build_layer_pool(engine, device, hidden: int, inter: int, group_k: int,
                     e_local: int, layer_idx: int, em_global_to_local, topk: int = 0,
                     layout: str = "mxfp4"):
    """env 门 + 表命中才建;返回 LayerPool 或 None。

    ⭐ `layout`:见 `LayerPool.__init__` —— `"mxfp4"`(gpu_prefill)/`"fp8"`
    (gpu_prefill_fp8)。默认 `"mxfp4"` ⇒ 对既有 ds41f 路径**逐字不变** ✓
    """
    if not pin_enabled():
        return None
    path = os.environ.get(_ENV_TABLE, "")
    if not path or not os.path.exists(path):
        # ⛔ 2026-10-16 修:这一行原先**每层每次建池都打** ⇒ 实测在
        #    `logs/glm68_p1_pin1.log:588-1814` 一次性刷出 **1088 行**同样的话 ⇒
        #    纯日志噪音(把真正有用的行淹掉)✓
        # ⭐ 改为**按"原因+path"去重**:同一进程内同样原因只打**第一次** ✓
        # ⛔ 行为**逐字不变**:表缺失 ⇒ 本层纯 CPU 路径 ⇒ `return None` ✓
        # ⚠️ 注意:本条**不是**任何崩溃的原因(实测那两臂的崩因在别处,见 EXPERIMENTS B870)✗
        _wkey = f"missing:{path}"
        if build_layer_pool._warned != _wkey:            # noqa: SLF001
            print(f"[xtu-pin] 开关开但表缺失({path!r})⇒ 本层纯 CPU 路径"
                  f"（同类提示本次进程内只报一次）", flush=True)
            build_layer_pool._warned = _wkey             # noqa: SLF001
        return None
    # ⛔ 2026-10-16 修:缓存原先**只看"有没有"**、⛔ 不看 `path` ✗ ⇒ 同一进程内
    #    先给模型 A 建池、再给模型 B 建池时,**B 会沿用 A 的表**(热专家集合完全错)✗
    #    —— 而 `XIAOTU_PIN_TABLE` 是**按模型**打的表(如 `expert_pin.ds41f.seed10.bin`)✓
    #    ⇒ 缓存必须以 **path 为键** ✓(与 `test_pin_stage` 的"换模型必重载"一致性要求相符 ✓)
    if build_layer_pool._table_path != path:                 # noqa: SLF001
        build_layer_pool._table = load_pin_table(path)       # noqa: SLF001
        build_layer_pool._table_path = path                  # noqa: SLF001
    table = build_layer_pool._table                          # noqa: SLF001
    g_list = table.get(layer_idx, [])
    if not g_list:
        return None
    if em_global_to_local is None:  # 无 EP(TP=1):全局==本地
        pinned_local = sorted({int(g) for g in g_list if 0 <= int(g) < int(e_local)})
    else:
        em = em_global_to_local.to(torch.long)
        pinned_local = sorted({int(em[g]) for g in g_list if 0 <= int(em[g]) < int(e_local)})
    if not pinned_local:
        return None
    return LayerPool(engine, device, int(hidden), int(inter), int(group_k),
                     int(e_local), pinned_local, layer_idx, layout=str(layout),
                     topk=int(topk))


build_layer_pool._table = None  # 进程级缓存(懒),键 = 表路径
build_layer_pool._table_path = None  # noqa: SLF001  ⭐ 缓存键:换了 path 必须重载 ✓
build_layer_pool._warned = None  # noqa: SLF001  ⭐ 2026-10-16:"表缺失"提示去重键(只报一次)✓


class _ReadySlot:
    __slots__ = ("bufs", "busy", "ready")

    def __init__(self, bufs, ready):
        self.bufs = bufs
        self.busy = None
        self.ready = ready
