#!/usr/bin/env python3
"""按内核名统计 Triton 缓存变体数，用于验证"形状爆炸是否被治好"。

背景（2026-11-01）：真实流量下 1–2h 崩溃，实测稳态 prefill 期间 Triton 缓存
5 分钟新增 464 个文件。缓存快照按内核名统计出的 top3 正是
`_fp8_mqa_logits_kernel`(1121) / `_fp8_paged_mqa_logits_rowwise_kernel`(567) /
`_deepseek_v4_sm12x_fp8_einsum_kernel`(513) —— 三者都把"随请求变化的长度"
声明成 tl.constexpr，每来一个新长度就编译一份 cubin。

用法：
  verify_kernel_variants.py --snapshot /tmp/kv_before.json     # 存基线
  verify_kernel_variants.py --diff /tmp/kv_before.json         # 与基线比，退出码 1 = 有增长
  verify_kernel_variants.py --top 12                           # 只看排行

判据（验收口径）：
  * 稳态运行（有真实流量）≥10 分钟后，任一内核变体增长 > 5 ⇒ 仍有形状特化；
  * 总变体增长 > 20 ⇒ 同上。
"""
import argparse, collections, json, os, sys


def _kernel_name_of(dirpath):
    for f in os.listdir(dirpath):
        if f.endswith(".json"):
            try:
                with open(os.path.join(dirpath, f)) as fh:
                    j = json.load(fh)
            except Exception:
                continue
            name = j.get("name") or j.get("kernel_name")
            if name:
                return name
    return None


def scan(cache_dir):
    counts = collections.Counter()
    unknown = 0
    if not os.path.isdir(cache_dir):
        return counts, unknown
    for d in os.listdir(cache_dir):
        p = os.path.join(cache_dir, d)
        if not os.path.isdir(p):
            continue
        name = _kernel_name_of(p)
        if name:
            counts[name] += 1
        else:
            unknown += 1
    return counts, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/.triton/cache"))
    ap.add_argument("--snapshot")
    ap.add_argument("--diff")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--max-kernel-growth", type=int, default=5)
    ap.add_argument("--max-total-growth", type=int, default=20)
    a = ap.parse_args()

    counts, unknown = scan(a.cache)
    total = sum(counts.values())
    print(f"缓存目录: {a.cache}")
    print(f"变体总数: {total}  内核名数: {len(counts)}  未识别目录: {unknown}")
    print(f"\n按内核名 top{a.top}:")
    for name, c in counts.most_common(a.top):
        print(f"  {c:>6}  {name}")

    if a.snapshot:
        with open(a.snapshot, "w") as fh:
            json.dump(dict(counts), fh, indent=1)
        print(f"\n基线已写入 {a.snapshot}")
        return 0

    if a.diff:
        with open(a.diff) as fh:
            before = json.load(fh)
        growth = {n: c - before.get(n, 0) for n, c in counts.items() if c > before.get(n, 0)}
        gone = [n for n in before if n not in counts]
        total_growth = total - sum(before.values())
        print(f"\n=== 与基线对比（总增长 {total_growth:+d}）===")
        if growth:
            for n, g in sorted(growth.items(), key=lambda kv: -kv[1])[:15]:
                flag = "❌" if g > a.max_kernel_growth else "✅"
                print(f"  {flag} {n}: +{g}")
        else:
            print("  ✅ 无任何内核新增变体")
        if gone:
            print(f"  （基线中有 {len(gone)} 个内核名已不在缓存里，属缓存被清理，不是问题）")
        bad = total_growth > a.max_total_growth or any(g > a.max_kernel_growth for g in growth.values())
        print("\n判定: " + ("❌ FAIL — 仍在按形状特化" if bad else "✅ PASS — 形状特化已停止"))
        return 1 if bad else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
