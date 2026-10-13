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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--slots", default="64,128,192,256")
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
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
