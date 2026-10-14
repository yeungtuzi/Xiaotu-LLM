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


def pin_enabled() -> bool:
    return os.environ.get(_ENV_ON, "0") == "1"


def load_pin_table(path: str) -> dict[int, list[int]]:
    """XTUP v0:header<magic4,ver,n_layers,E,n_entries> + entries(layer u16, expert u16 频率降序)。"""
    with open(path, "rb") as f:
        buf = f.read()
    magic, ver, n_layers, e_max, n_ent = struct.unpack_from("<4sIIII", buf, 0)
    assert magic == b"XTUP" and ver == 1, f"bad pin table {magic!r} v{ver}"
    out: dict[int, list[int]] = {}
    for i in range(n_ent):
        l, e = struct.unpack_from("<HH", buf, 16 + 4 * i)
        out.setdefault(int(l), []).append(int(e))
    return out


class LayerPool:
    """单层池。nslots = 该层钉住数(本 rank 视角,已按 expert_map 折算)。"""

    def __init__(self, engine, device: torch.device, hidden: int, inter: int,
                 group_k: int, n_local: int, pinned_local: list[int], layer_idx: int):
        from vllm_xiaotu_moe import gpu_prefill as _gp
        self.engine, self.dev = engine, device
        self.H, self.I, self.gk = int(hidden), int(inter), int(group_k)
        self.n_local, self.layer_idx = int(n_local), int(layer_idx)
        self.pinned = list(pinned_local)
        self.nslots = len(self.pinned)
        if self.nslots == 0:
            self.planes = None
            self.slotmap = torch.arange(self.n_local, dtype=torch.int32, device=device)
            return
        rb13, rb2 = self.H // 2, self.I // 2   # 与 noraw:1383 逐字同式
        S = self.nslots
        self.planes = (
            torch.empty((S, rb13, 2 * self.I), dtype=torch.uint8, device=device),
            torch.empty((S, self.H // self.gk, 2 * self.I), dtype=torch.uint8, device=device),
            torch.empty((S, rb2, self.H), dtype=torch.uint8, device=device),
            torch.empty((S, self.I // self.gk, self.H), dtype=torch.uint8, device=device),
        )
        self.slot_of = {e: i for i, e in enumerate(self.pinned)}
        sm = torch.arange(self.n_local, dtype=torch.int32, device=device)
        for e, s in self.slot_of.items():
            sm[e] = self.n_local + s
        self.slotmap = sm.to(device=device, non_blocking=False)
        self._load_all(_gp)
        gib = sum(p.numel() for p in self.planes) / 2**30
        print(f"[xtu-pin] layer{layer_idx}: slots={S} pool={gib:.2f} GiB (树内分配,KV profiling 前)",
              flush=True)

    def _load_all(self, gp) -> None:
        """槽加载 = 逐专家复用 noraw 子路径(e0..e1=单专家,dst=槽 view)。"""
        for e, s in self.slot_of.items():
            dst = tuple(p[s:s + 1] for p in self.planes)
            gp.kmajor_from_engine_shards_noraw(
                self.engine, self.dev, self.H, self.I, self.n_local, self.gk,
                dst=dst, e0=e, e1=e + 1)
        torch.cuda.synchronize()

    # ---- Phase B 预留:LRU 换入只走这两个入口(step 间 event 窗口内调用) ----
    def swap_in(self, gp, old_slot_expert: int, new_expert: int) -> None:
        raise NotImplementedError  # P2-B:plane[s] 重写 + slotmap 原子改值 + 表更新
        # 纪律:prefill staging 突发段(staging_active flag)内不得触发(QA #167 支路③)


def build_layer_pool(engine, device, cfg, layer_idx: int, em_global_to_local: torch.Tensor):
    """env 门 + 表命中才建;返回 LayerPool 或 None。"""
    if not pin_enabled():
        return None
    path = os.environ.get(_ENV_TABLE, "")
    if not path or not os.path.exists(path):
        print(f"[xtu-pin] 开关开但表缺失({path!r})⇒ 本层纯 CPU 路径", flush=True)
        return None
    table = build_layer_pool._table or load_pin_table(path)  # noqa: SLF001
    build_layer_pool._table = table
    g_list = table.get(layer_idx, [])
    if not g_list:
        return None
    if em_global_to_local is None:  # 无 EP(TP=1):全局==本地
        pinned_local = sorted({int(g) for g in g_list if 0 <= int(g) < int(cfg.expert_num)})
    else:
        em = em_global_to_local.to(torch.long)
        pinned_local = sorted({int(em[g]) for g in g_list if 0 <= int(em[g]) < int(cfg.expert_num)})
    if not pinned_local:
        return None
    return LayerPool(engine, device, cfg.hidden_size, cfg.intermediate_size,
                     int(getattr(cfg, "groupK", 32) or 32), int(cfg.expert_num),
                     pinned_local, layer_idx)


build_layer_pool._table = None  # 进程级缓存(懒)
