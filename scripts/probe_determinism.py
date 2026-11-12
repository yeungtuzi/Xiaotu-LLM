#!/usr/bin/env python3
"""在同一 prompt 上反复取样，定位“输出不唯一”的**具体差异位置**。

背景（台账 #433）：`accept_degen.py --checks A` 实测约 1/8 的**偶发**签名不一致；
确定性归约（`XIAOTU_MOE4_KSPLIT_DET`）已修掉主因，残余疑为稀疏索引器 top-k 并列取舍。
本脚本用于**抓住一次不一致并把差异摆出来**：

* 每组发 N 条同 prompt（同 uuid），保存**完整文本 + reasoning**；
* 报告签名种类；若 ≥2 种，打印前两条的**首个差异字符位置**与差异片段；
* 可循环多组（`--groups`），直到抓到一次不一致。

用法（远端，需服务已就绪）：
    PORT=8700 python3 scripts/probe_determinism.py --n 200 --per-group 6 --groups 8
退出码：0 = 抓到了不一致（并已打印差异）；1 = 所有组都一致；2 = 环境问题。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

PORT = int(os.environ.get("PORT", "8700"))
NEEDLE = int(os.environ.get("NEEDLE", "740317"))


def doc(n: int, needle_row: int, needle: int) -> str:
    out = []
    for i in range(n):
        if i == needle_row:
            out.append("Entry %d: the access code for project Q is %d." % (i, needle))
        else:
            out.append("Entry %d: the access code for project P%d is %d."
                       % (i, i, 100000 + (i * 37 + n) % 899999))
    return "\n".join(out)


def one(n: int, uuid: str, maxtok: int) -> tuple[str, str]:
    needle_row = int(n * 0.5)
    body = doc(n, needle_row, NEEDLE)
    prompt = ("You are a helpful assistant. Answer using only the document.\n\n"
              + uuid + "\n\n" + body
              + "\n\nWhat is the access code for project Q? Answer with the number only.")
    payload = {"model": "DeepSeek-V4.1-Flash",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": maxtok, "temperature": 0, "top_p": 1}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        d = json.loads(resp.read())
    m = d["choices"][0].get("message") or {}
    return (m.get("content") or ""), (m.get("reasoning") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--maxtok", type=int, default=512)
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--uuid", default="det-probe")
    a = ap.parse_args()

    for g in range(1, a.groups + 1):
        texts = []
        for _ in range(a.per_group):
            t, r = one(a.n, a.uuid, a.maxtok)
            texts.append((t, r))
        sigs = [hashlib.sha256((t + "\u0000" + r).encode()).hexdigest()[:16] for t, r in texts]
        uniq = sorted(set(sigs))
        print(f"组 {g}/{a.groups}: 签名种类 {len(uniq)}  ({' '.join(s[:8] for s in uniq)})", flush=True)
        if len(uniq) > 1:
            # 找前两条不同样本，打印首个差异位置
            first = texts[0]
            for k, cur in enumerate(texts[1:], start=2):
                if cur == first:
                    continue
                for idx, (c1, c2) in enumerate(zip(first[0], cur[0])):
                    if c1 != c2:
                        lo = max(0, idx - 40)
                        print(f"  ✅ 抓到不一致：第 1 条 vs 第 {k} 条，content 首差在字符 {idx}")
                        print(f"     第1条: …{first[0][lo:idx+60]!r}")
                        print(f"     第{k}条: …{cur[0][lo:idx+60]!r}")
                        break
                else:
                    print(f"  ✅ 抓到不一致：第 1 条 vs 第 {k} 条（content 相同，差异在 reasoning 或长度）")
                print(f"     长度：第1条 content={len(first[0])} reasoning={len(first[1])}；"
                      f"第{k}条 content={len(cur[0])} reasoning={len(cur[1])}")
                return 0
        print("  全部一致", flush=True)
    print("== 全部组都一致：本轮没抓到不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())
