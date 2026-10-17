#!/usr/bin/env python
"""量【会话级重瞄】的载荷假设:该会话 **prompt 的 prefill 路由** 能否预测它的 **decode 路由**?

背景(2026-10-15,用户在另一会话提出的结构):
  在 GPU prefill 期间,CPU 是闲的 —— 而 **prompt 各层的专家选择在 prefill 时已经算出来了**
  (不是预测问题,是已经发生的事实)⇒ 于是可以在 prefill 结束前把"本会话会用到"的专家
  **一次性压进显存**,把"每步流 3 槽"的抠搜模式换成**一次会话重瞄**。

本脚本量它的**载荷假设**(唯一的未知量):
  ⇒ 对每个会话,令 P(layer) = 该会话 **prefill 触达**的专家集合,
     D(layer) = 该会话 **decode 真正用到**的专家(带频次)⇒ 算:
       ① **覆盖**:decode 的 (layer,expert) 命中中,落在 P 里的比例(按 pair 加权)
       ② **P 的规模**:|P(layer)| 的分布(若 ≈ E ⇒ 该集合无鉴别力,方案不成立 ✗)
       ③ **受槽位约束的可达性**:只允许每层常驻 S 个槽,在 P 内**按 prefill 频次降序**取 S 个
          ⇒ 得到的 decode 命中率(这才是"会话重瞄"能给到的上界)
       ④ 对照:**全局静态底表**(全 trace 频次 top-S,留一折外评估,避免 B394 的 in-sample 自欺)

⚠️ 口径与前提(必须同时成立才可用):
  * trace 必须是 **conc=1**(否则时间相邻 ≠ 会话归属);脚本会打印并发线索供人判断;
  * 记录格式 = XTU-RDUMP,`[u64 tag][u32 qlen][u32 k][u64 ts][i32 ids...]`(见 binding.cpp xtu_rd);
  * `qlen > PREFILL_MIN` 判为 prefill,否则 decode;
  * 层身份 = tag 首见序(与 expert_profile.py 同一约定)。

用法:
  python scripts/prefill_decode_overlap.py <route.bin> [--slots 128,256,512] [--prefill-min 16]
"""
from __future__ import annotations

import argparse
import collections
import glob as _glob
import statistics
import struct
import sys

REC_HDR = 24          # u64 tag + u32 qlen + u32 k + u64 ts


def parse(path: str):
    """产出 (layer, ts, qlen, k, ids) 按 ts 升序;不做任何 decode 过滤。"""
    files = sorted(_glob.glob(path)) if any(c in path for c in "*?[") else [path]
    if len(files) > 1:
        print(f"[overlap] 检测到 {len(files)} 个 rank 文件 ⇒ 只用 {files[0]}(TP 路由冗余,B399)✗")
        files = files[:1]
    recs, lay, bad = [], {}, 0
    for fp in files:
        with open(fp, "rb") as f:
            buf = f.read()
        off = 0
        while off + REC_HDR <= len(buf):
            tag, qlen, k, ts = struct.unpack_from("<QIIQ", buf, off)
            n = qlen * k
            if n <= 0 or qlen > 1000000 or k > 64:
                bad += 1
                break
            if off + REC_HDR + 4 * n > len(buf):
                bad += 1
                break
            ids = list(struct.unpack_from(f"<{n}i", buf, off + REC_HDR))
            off += REC_HDR + 4 * n
            if tag not in lay:
                lay[tag] = len(lay)
            recs.append((lay[tag], ts, qlen, k, ids))
    recs.sort(key=lambda r: r[1])
    return recs, len(lay), bad


def segment(recs, prefill_min: int):
    """按时间相邻切会话:一段 prefill 记录 → 其后到下一个 prefill 之前的 decode 记录。

    返回 [(prefill_recs, decode_recs), ...];开头若没有 prefill(流被截断)则丢弃。
    """
    sessions, cur_pf, cur_dec, started = [], [], [], False
    for r in recs:
        is_pf = r[2] > prefill_min
        if is_pf:
            if started and cur_dec:            # 上一段 prefill 已经跟着 decode ⇒ 收一个会话
                sessions.append((cur_pf, cur_dec))
                cur_pf, cur_dec = [], []
            cur_pf.append(r)
            started = True
        elif started:
            cur_dec.append(r)
    if started and cur_dec:
        sessions.append((cur_pf, cur_dec))
    return sessions


def top_s_in_P(pf_ids_by_layer, dec_cnt_by_layer, slots: int):
    """在 P 内按 **prefill 频次** 降序取 slots 个 ⇒ 命中该会话 decode 的比例(pair 加权)。"""
    hit = tot = 0
    for layer, dc in dec_cnt_by_layer.items():
        pf = pf_ids_by_layer.get(layer)
        if not pf:
            tot += sum(dc.values())
            continue
        cand = sorted(pf.items(), key=lambda kv: -kv[1])[:slots]
        keep = {e for e, _ in cand}
        tot += sum(dc.values())
        hit += sum(c for e, c in dc.items() if e in keep)
    return hit, tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binfile")
    ap.add_argument("--slots", default="128,256,512")
    ap.add_argument("--prefill-min", type=int, default=16)
    a = ap.parse_args()
    slots_list = [int(x) for x in a.slots.split(",") if x.strip()]

    recs, n_layers, bad = parse(a.binfile)
    if not recs:
        print("⛔ 没解析出任何记录")
        return 2
    qb = collections.Counter(r[2] for r in recs)
    n_pf = sum(1 for r in recs if r[2] > a.prefill_min)
    e_max = max((max(r[4]) for r in recs if r[4]), default=-1)
    print(f"文件={a.binfile}")
    print(f"记录={len(recs)}  层数(去重 tag)={n_layers}  截断/损坏停止={bad}")
    print(f"⭐ 专家号上界 E>={e_max + 1}(trace 里出现过的最大 id+1;真实 E 只会 ≥ 它)")
    print(f"prefill 记录(qlen>{a.prefill_min})={n_pf}  decode 记录={len(recs) - n_pf}")
    print(f"qlen 分布(前 8)={sorted(qb.items(), key=lambda kv: -kv[1])[:8]}")
    if n_pf == 0:
        print("⛔ 该 trace **没有 prefill 记录** ⇒ 无法量'prompt→decode'的重合(需 --all-prefill 采集的流)✗")
        return 3

    sess = segment(recs, a.prefill_min)
    print(f"\n切出会话数={len(sess)}"
          f"{'  ⚠️ 会话过少/过大 ⇒ 该 trace 可能是多路并发或含大量重复 prefill,结论需谨慎' if len(sess) < 2 else ''}")
    if not sess:
        print("⛔ 切不出会话(prefill 后没有 decode)⇒ 无法评估")
        return 3

    # ---- ①/② 逐会话:P 的规模 与 decode 覆盖率 ----
    cov_hit = cov_tot = 0
    p_size_all, cov_rates = [], []
    for pf, dec in sess:
        P = collections.defaultdict(collections.Counter)   # layer -> expert -> prefill 次数
        for layer, _ts, _q, _k, ids in pf:
            P[layer].update(ids)
        D = collections.defaultdict(collections.Counter)
        for layer, _ts, _q, _k, ids in dec:
            D[layer].update(ids)
        h = t = 0
        for layer, dc in D.items():
            keep = set(P.get(layer, {}))
            t += sum(dc.values())
            h += sum(c for e, c in dc.items() if e in keep)
        cov_hit += h
        cov_tot += t
        cov_rates.append(h / t if t else float("nan"))
        p_size_all += [len(v) for v in P.values()]
    print("\n① 【覆盖】decode 的 (layer,expert) 命中中落在本会话 prefill 触达集 P 里的比例:")
    print(f"     总体(pair 加权)= {cov_hit / cov_tot:.4f}  ({cov_hit}/{cov_tot})")
    print(f"     逐会话:中位={statistics.median(cov_rates):.4f} 最小={min(cov_rates):.4f} 最大={max(cov_rates):.4f}")
    print("\n② 【P 的规模】每层 prefill 触达专家数 |P(layer)|:")
    print(f"     n={len(p_size_all)} 中位={statistics.median(p_size_all):.0f} "
          f"均值={statistics.mean(p_size_all):.1f} 最大={max(p_size_all)}")

    # ---- ③ 受槽位约束的可达性(会话内:P 内 top-slots)----
    print("\n③ 【会话重瞄的上界】每层只常驻 S 槽,在 P 内按 prefill 频次取 S 个 ⇒ decode 命中率:")
    for S in slots_list:
        h = t = 0
        for pf, dec in sess:
            P = collections.defaultdict(collections.Counter)
            for layer, _ts, _q, _k, ids in pf:
                P[layer].update(ids)
            D = collections.defaultdict(collections.Counter)
            for layer, _ts, _q, _k, ids in dec:
                D[layer].update(ids)
            hh, tt = top_s_in_P(P, D, S)
            h += hh
            t += tt
        print(f"     S={S:<5} 命中率={h / t:.4f}")

    # ---- ⑤ oracle:任何"会话感知预测器"的**上界**(用未来真实 decode 频次取 top-S)----
    # ⭐ 这是回答"要不要上深度学习模型"的唯一判据:
    #    若 oracle 与 ③(零模型的 prefill 直方图)差距很小 ⇒ 模型无处施展 ✗
    #    若差距很大 ⇒ 才谈得上模型(且必须先排除更廉价的排名改进)
    print("\n⑤ 【oracle 上界(偷看未来 ⇒ 只作上界,不可实现)】:")
    for S in slots_list:
        h_in = t_in = h_all = 0
        for pf, dec in sess:
            P = collections.defaultdict(collections.Counter)
            for layer, _ts, _q, _k, ids in pf:
                P[layer].update(ids)
            D = collections.defaultdict(collections.Counter)
            for layer, _ts, _q, _k, ids in dec:
                D[layer].update(ids)
            for layer, dc in D.items():
                t_in += sum(dc.values())
                h_all += sum(c for e, c in dc.most_common(S))
                pin = [(e, c) for e, c in dc.items() if e in P.get(layer, {})]
                keep_in = {e for e, _ in sorted(pin, key=lambda kv: -kv[1])[:S]}
                h_in += sum(c for e, c in pin if e in keep_in)
        print(f"     S={S:<5} (a)限制在 P 内={h_in / t_in:.4f}   "
              f"(b)不限制(绝对上界)={h_all / t_in:.4f}")

    # ---- ⑥ 零模型的【运行时自适应】:会话内 LRU,初值 = P 内 prefill 频次 top-S ----
    # ⭐ 这是 item③(expert_pool.swap_in/swap_out)真正要选的策略:
    #    它既用 prefill 信号(冷启动),又用 decode 自身的新信息(随步自适应)
    #    ⇒ 若它能逼近 ⑤a,则【连"预填直方图"都不必是最优的】,更不需要模型 ✓
    print("\n⑥ 【会话内 LRU(容量 S/layer,初值=prefill 频次 top-S;零模型)】:")
    for S in slots_list:
        h = t = 0
        for _pf, dec in sess:
            P = collections.defaultdict(collections.Counter)
            for layer, _ts, _q, _k, ids in _pf:
                P[layer].update(ids)
            res = {l: collections.OrderedDict(
                (e, None) for e, _ in c.most_common(S)) for l, c in P.items()}
            for layer, _ts, _q, _k, ids in dec:
                r = res.setdefault(layer, collections.OrderedDict())
                t += len(ids)
                h += sum(1 for e in ids if e in r)
                for e in ids:                      # LRU:命中/新入都移到最近端
                    if e in r:
                        r.move_to_end(e)
                    else:
                        r[e] = None
                        if len(r) > S:
                            r.popitem(last=False)
        print(f"     S={S:<5} 命中率={h / t:.4f}")

    # ---- ④ 对照:全局静态底表(留一折外)----
    print("\n④ 【对照】全局静态底表(全 trace 频次 top-S,10 折留一 ⇒ 折外评估):")
    per_fold = collections.defaultdict(list)
    for r in recs:
        per_fold[r[1] % 10].append(r)
    for S in slots_list:
        h = t = 0
        for f in range(10):
            top = collections.defaultdict(collections.Counter)
            for f2, rs in per_fold.items():
                if f2 == f:
                    continue
                for layer, _ts, _q, _k, ids in rs:
                    top[layer].update(ids)
            keep = {l: {e for e, _ in c.most_common(S)} for l, c in top.items()}
            for layer, _ts, _q, _k, ids in per_fold[f]:
                t += len(ids)
                k_l = keep.get(layer, set())
                h += sum(1 for e in ids if e in k_l)
        print(f"     S={S:<5} 命中率={h / t:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
