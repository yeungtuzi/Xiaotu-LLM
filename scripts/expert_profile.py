#!/usr/bin/env python3
"""热度画像工具(P0.1,PLAN_EXPERT_RESIDENT)—— Strata make_profile/routing_kfold 的对应物。

输入 : dump jsonl,每行 {"k": <模块标识>, "ids": [[e0..eK], ...]}(一行 = 一次某层的 dump)
产出 : ① 频率降序热度表(钉选表 v0 的原料)
       ② top-N 覆盖率曲线(in-sample ⇒ 上界)
       ③ ⭐ 十折留一 h@N(用其余 9 折的 top-N 预测 held-out 选择 ⇒ 真实可预测性,防 in-sample 自欺)
用法 : expert_profile.py <e2.jsonl> [--slots 64,128,192,256] [--folds 10]
判据 : 校准点 = qfn e2 上 h@cache 应与 Strata 公布的 h_expert≈0.64 同量级(≠ 精确相等,口径不同)。
⚠️ 若输入混合多层(按 k 区分),分层分别算;混池统计 = 粒度错解(Strata h_layer=0.0456 教训)。
"""
import argparse
import collections
import itertools
import json
import sys


def load(path):
    """返回 {k: [expert_ids 扁平序列]}(一行一 dump;行内按 token×topk 展开)。"""
    per_key = collections.defaultdict(list)
    bad = 0
    for ln in open(path, encoding="utf-8"):
        try:
            d = json.loads(ln)
        except Exception:
            bad += 1
            continue
        flat = []
        for row in d.get("ids", []):
            if isinstance(row, list):
                flat.extend(x for x in row if isinstance(x, int))
            elif isinstance(row, int):
                flat.append(row)
        per_key[d.get("k", "?")].append(flat)
    if bad:
        print(f"[warn] {bad} 行解析失败", file=sys.stderr)
    return per_key


def coverage(counter, slots):
    tot = sum(counter.values()) or 1
    c = sorted(counter.values(), reverse=True)[:slots]
    return sum(c) / tot


def loo_hit_rate(all_lines, slots, folds=10):
    """十折(按 dump 折分):训练折 top-N 预测 held-out 折的选择 ⇒ 可预测性。"""
    lines = list(all_lines)  # dumps(list of int);不可展平(展平=把 dump 拆成单事件,折就没了)
    n = len(lines)
    if n < folds:
        return float("nan")
    hits = tot = 0
    for f in range(folds):
        test = lines[f::folds]
        train = (e for i, ln in enumerate(lines) if i % folds != f for e in ln)
        top = {e for e, _ in collections.Counter(train).most_common(slots)}
        hits += sum(1 for ln in test for e in ln if e in top)
        tot += sum(len(ln) for ln in test)
    return hits / tot if tot else float("nan")


def read_bin(path, decode_only=True):
    """读 XTU-RDUMP 二进制流:rec = [u64 tag][u32 qlen][u32 k][u64 ts_ns][i32 ids...](见 binding.cpp xtu_rd)。
    产出 (tag, qlen, k, ts, ids) 序列;tag(引擎指针) ⇒ 层身份(首见序 → 层号)。"""
    import struct, glob as _glob
    files = sorted(_glob.glob(path)) if any(c in path for c in "*?[") else [path]
    recs = []
    for fp in files:
        lay = {}  # ⚠️ 层身份映射**每文件独立**:engine 指针跨进程不同,但注册首见序一致 ⇒ 首见序=层号
        with open(fp, "rb") as f:
            buf = f.read()
        off = 0
        while off + 24 <= len(buf):
            tag, qlen, k, ts = struct.unpack_from("<QIIQ", buf, off)
            n = qlen * k
            if n <= 0 or qlen > 1000000 or k > 64:
                break  # 流损坏/截断 ⇒ 到此为止(尽力而为)
            if off + 24 + 4 * n > len(buf):
                break
            ids = list(struct.unpack_from(f"<{n}i", buf, off + 24))
            off += 24 + 4 * n
            if decode_only and qlen > 16:
                continue  # decode 形态(M≤16);prefill 记录单独口径
            if tag not in lay:
                lay[tag] = len(lay)
            recs.append((lay[tag], ts, ids))
    recs.sort(key=lambda r: r[1])  # 多 pid 文件(TP=2 双 worker)按时间归并
    return recs, len(lay)


def lru_sweep(recs, caps, slot_gib=0.01752, base_frac=0.0):  # 18.81MB/槽=0.01752GiB(SPEC §3 定稿口径)
    """全局 (layer,expert) 槽的 LRU 时序模拟 ⇒ 会话级命中率主判据(QA #166)。
    ids 是层内专家号(引擎对象本身带层身份)⇒ 缓存键必须带层维。
    base_frac>0 = **Strata 形状**:按全局频率取 cap×base_frac 个槽做受保护常驻(永不逐出),
    LRU 只在剩余槽活动;0 = 纯 LRU(会话局部性的下界探针)。
    ⚠️ in-sample 陷阱:base 表必须来自**独立 trace**(留一),同流自算会复刻 +10~20pp 自欺差(B394)。"""
    import collections
    cnt = collections.Counter()
    if base_frac > 0:
        cnt = collections.Counter((l, e) for l, _, ids in recs for e in ids)
    for cap in caps:
        base_keys = {k for k, _ in cnt.most_common(int(cap * base_frac))} if base_frac else set()
        cache = collections.OrderedDict()  # 末端=MRU,popitem(last=False) 逐出最久未用
        hits = tot = 0
        for (l, _, ids) in recs:
            for e in ids:
                key = (l, e)
                if key in base_keys:
                    hits += 1
                elif key in cache:
                    cache.move_to_end(key)
                    hits += 1
                else:
                    tot += 1
                    if cap - len(base_keys) <= 0:
                        continue  # 全部槽被保护段占用 ⇒ 非 base 只计 miss,不入缓存
                    if len(cache) >= cap - len(base_keys):
                        cache.popitem(last=False)
                    cache[key] = None
        rate = hits / (hits + tot) if hits + tot else float("nan")
        tag = f"base={base_frac*100:.0f}%" if base_frac else "纯LRU"
        print(f"  [{tag:8s}] cap={cap:6d} 槽(≈{cap*slot_gib:5.2f} GiB): 命中={rate*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?")
    ap.add_argument("--read-bin", default=None, help="XTU-RDUMP 二进制文件")
    ap.add_argument("--lru", default="200,400,700,1200,2400,5120")
    ap.add_argument("--base", default="0,0.25,0.5", help="受保护常驻占比扫描(Strata 形状 vs 纯LRU)")
    ap.add_argument("--all-prefill", action="store_true", help="read-bin 时不排除 prefill 记录")
    ap.add_argument("--slots", default="64,128,192,256")
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    if a.read_bin:
        recs, n_layers = read_bin(a.read_bin, decode_only=not a.all_prefill)
        print(f"records={len(recs)} 层数={n_layers}(首见序)")
        caps = [int(x) for x in a.lru.split(",")]
        for bf in [float(x) for x in a.base.split(",")]:
            lru_sweep(recs, caps, base_frac=bf)
        return
    if not a.jsonl:
        ap.error("需要 jsonl 或 --read-bin")
    per_key = load(a.jsonl)
    slot_list = [int(x) for x in a.slots.split(",")]
    print(f"dumps per key: { {k: len(v) for k, v in list(per_key.items())[:5]} } "
          f"keys={len(per_key)}")
    for k, dumps in list(per_key.items())[:4]:
        cnt = collections.Counter(itertools.chain.from_iterable(dumps))
        n_exp = len(cnt)
        print(f"\n== key={k[:18]} 事件={sum(cnt.values()):,} 出现过的专家={n_exp}")
        for s in slot_list:
            in_s = coverage(cnt, s)
            loo = loo_hit_rate(dumps, s, a.folds)
            print(f"  slots={s:4d}: in-sample={in_s*100:5.1f}%   留一 h@N={loo*100:5.1f}%"
                  f"   (自欺差={100*(in_s-loo):+.1f}pp)")


if __name__ == "__main__":
    main()
