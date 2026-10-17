#!/usr/bin/env python3
"""make_dsh_len_prompt.py —— 用【真实 DSH 载荷】拼出指定长度的 prefill 基准 prompt。

为什么需要它(B427/B428 的后续):
  · README 的 prefill 列用的是 ~9.4k 的 DSH 载荷;而 MBT=8192 时它会被切成
    `8192 + ~1200`,**尾块 < GPU 预填门槛 2560 ⇒ 整块退回 CPU(实测 127 tok/s)**,
    于是那一档的 prefill 读数里混进了 ~50% 的 CPU 时间(B427)。
  · 要单独观测"纯 GPU 预填"与 MBT 的影响,就需要一个**不含亚门槛尾块**的长度:
    `2 × MBT` 附近(15–16K)最合适 —— 两块的 qlen 都 ≫ 门槛。

口径(与 dsh_longctx_needle 一致):
  · 源 = 本仓装配好的真实 DSH 载荷(默认 `dsh_assembled_bench.jsonl`);
  · 长度换算用**实测比值** `chars/token ≈ 4.03`(由 `471931 chars → 117107 token` 反解),
    ⚠️ 这只是选长度用的估值;最终**以引擎 `/tokenize` 的实际 prompt_tokens 为准**;
  · 切点收在**行边界**,不把一行token切碎;
  · 变体之间**起点错开**(轮转同一份长缓冲),避免 3 条只是同一条的复制
    (⚠️ QA #187 的教训:自造数据集必须多样,否则测的是数据集的属性)。

用法:
  python3 scripts/make_dsh_len_prompt.py --tokens 15500 --n 3 \
      --out /path/dsh_15k.jsonl \
      [--src .../dsh_assembled_bench.jsonl] [--chars-per-token 4.03] [--outlen 8]
"""
import argparse
import json
import os


def build_buffer(recs, need_chars):
    """轮转拼接记录,直到缓冲长度 >= need_chars + 变体错开所需的余量。"""
    buf = []
    total = 0
    i = 0
    while total < need_chars and recs:
        p = recs[i % len(recs)].get("prompt", "")
        buf.append(p)
        total += len(p) + 1
        i += 1
        if i > 100000:      # 防御:空 prompt 死循环
            break
    return "\n".join(buf)


def cut_line(buf, start, target_chars):
    seg = buf[start:start + target_chars]
    nl = seg.rfind("\n")
    if nl > 0:
        seg = seg[:nl + 1]
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=15500, help="目标 token 数(估值)")
    ap.add_argument("--n", type=int, default=3, help="变体条数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="process_data/data/dsh_assembled_bench.jsonl")
    ap.add_argument("--chars-per-token", type=float, default=4.03)
    ap.add_argument("--outlen", type=int, default=8)
    a = ap.parse_args()

    recs = [json.loads(ln) for ln in open(a.src, encoding="utf-8") if ln.strip()]
    if not recs:
        raise SystemExit("空源: %s" % a.src)
    tgt = int(a.tokens * a.chars_per_token)
    span = tgt
    buf = build_buffer(recs, tgt * (a.n + 1))
    out = []
    for i in range(a.n):
        start = i * span
        seg = cut_line(buf, start, tgt)
        if len(seg) < tgt // 2:
            seg = cut_line(buf, 0, tgt)      # 缓冲不够长时退回同一段(会告警)
            print("⚠️ 变体 %d 缓冲不足,与变体 0 相同(源太短)" % i)
        out.append({"prompt": seg, "output_tokens": a.outlen})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("写出 %s : n=%d 目标=%d token(≈%d chars)" % (a.out, len(out), a.tokens, tgt))
    for i, r in enumerate(out):
        print("  [%d] chars=%d 估~%.0f tok" % (i, len(r["prompt"]), len(r["prompt"]) / a.chars_per_token))


if __name__ == "__main__":
    main()
