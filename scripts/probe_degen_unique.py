#!/usr/bin/env python3
"""单臂探针：每份文档一个**唯一**针值 —— 用来区分「没检索到」与「读到了别的文档」。

背景（dev-docs/DEGEN_STAGE_COMPARISON.md #24/#25/#27）：
已有探针里**所有文档共用同一个针值 981273**，因此"读到别的文档的旧 KV"与"读到本文档"
在判分上无法区分（都算命中）。本探针让每份文档的针值唯一，于是：

* 回答里出现**本文档**的针 ⇒ `ok`；
* 回答里出现**历史文档**的针 ⇒ `cross`（= 读到别的请求的 KV，H1 的直接证据）；
* 出现别的数字/不回答 ⇒ `miss`。

用法：
    python3 scripts/probe_degen_unique.py                    # N=400, 4 轮
    N=800 ROUNDS=6 python3 scripts/probe_degen_unique.py
    ARM=shared python3 scripts/probe_degen_unique.py         # unique|shared
"""
import json
import os
import random
import re
import sys
import urllib.request

PORT = int(os.environ.get("PORT", "8700"))
N = int(os.environ.get("N", "400"))
ROUNDS = int(os.environ.get("ROUNDS", "4"))
ARM = os.environ.get("ARM", "unique")
MAXTOK = int(os.environ.get("MAXTOK", "64"))
DEPTH = float(os.environ.get("DEPTH", "0.5"))
UUID_FIXED = os.environ.get("UUID_FIXED", "")

BASE = "You are a helpful assistant. Answer using only the document.\n\n"
Q = ("\n\nWhat is the access code for project X? "
     "Answer with the number only.")
rng = random.Random(20261107)

# 本进程见过的所有针值 ⇒ 用来识别"跨文档污染"
SEEN_NEEDLES: dict = {}


def make_doc(n, needle, depth=DEPTH):
    row = min(n - 1, max(0, int(n * depth)))
    lines = []
    for i in range(n):
        if i == row:
            lines.append("Entry %d: the access code for project X is %d." % (i, needle))
        else:
            lines.append("Entry %d: the access code for project P%d is %d."
                         % (i, i, 100000 + (i * 37 + n) % 899999))
    return "\n".join(lines)


def ask(prompt):
    body = json.dumps({
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK, "temperature": 0, "top_p": 1,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        d = json.loads(resp.read())
    m = d["choices"][0]["message"]
    out = (m.get("content") or "") + (m.get("reasoning") or "")
    usage = d.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return out, cached, usage.get("prompt_tokens")


def main():
    st = {"ok": 0, "miss": 0, "cross": 0, "n": 0}
    print("# probe_degen_unique PORT=%d ARM=%s N=%d ROUNDS=%d DEPTH=%.2f uuid=%s"
          % (PORT, ARM, N, ROUNDS, DEPTH, UUID_FIXED or "random"), flush=True)
    for rnd in range(1, ROUNDS + 1):
        needle = 100000 + rng.randrange(899999)
        SEEN_NEEDLES[needle] = rnd
        body = make_doc(N, needle)
        u = UUID_FIXED or os.urandom(8).hex()
        if ARM == "shared":
            prompt = BASE + body + Q
        else:
            prompt = BASE + u + "\n\n" + body + Q
        out, cached, ptok = ask(prompt)
        nums = set(int(x) for x in re.findall(r"\b(\d{6})\b", out))
        if needle in nums:
            verdict = "ok"
        elif nums & set(SEEN_NEEDLES):
            verdict = "cross"
        else:
            verdict = "miss"
        st[verdict] += 1
        st["n"] += 1
        print("轮%d cached=%-6s ptok=%-6s %-5s 期望=%d 实得=%s  out=%s"
              % (rnd, cached, ptok, verdict, needle, sorted(nums)[:4],
                 out.strip()[:80].replace("\n", " ")), flush=True)
    print("\n# 汇总 ok=%d miss=%d cross=%d n=%d" % (st["ok"], st["miss"], st["cross"], st["n"]),
          flush=True)
    return 0 if st["ok"] == st["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
