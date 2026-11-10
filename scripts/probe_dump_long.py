#!/usr/bin/env python3
"""发**一条**固定内容的长 prompt，打印 cached/ptok/命中判定 —— 用于"同一 prompt 的两次转储"。

内容由 `N`/`UUID`/`DEPTH` 全确定（同一组参数 ⇒ 逐 token 相同），
针（needle）放在第 `int(N*DEPTH)` 行，答案唯一。
"""
from __future__ import annotations

import argparse
import json
import os
import time
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--uuid", default="")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--maxtok", type=int, default=16)
    a = ap.parse_args()
    needle_row = int(a.n * a.depth)
    uid = a.uuid or os.urandom(8).hex()
    body = doc(a.n, needle_row, NEEDLE)
    prompt = ("You are a helpful assistant. Answer using only the document.\n\n"
              + uid + "\n\n" + body
              + "\n\nWhat is the access code for project Q? Answer with the number only.")
    # ⚠️ 2026-11-10 预检发现：1017 的官方权重目录**不含 chat template**
    #    （tokenizer_config.json 里没有 chat_template 键，也没有 chat_template.jinja），
    #    因此 `/v1/chat/completions` 这条路径不可依赖 ⇒ 这里直接用 `/v1/completions`。
    #    语义不变：仍然是把整段文档喂进去、问一个数字。
    payload = {"model": "DeepSeek-V4.1-Flash", "prompt": prompt,
               "max_tokens": a.maxtok, "temperature": 0, "top_p": 1}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        d = json.loads(resp.read())
    dt = time.time() - t0
    ch = d["choices"][0]
    text = ch.get("text") or ""
    usage = d.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens", "?")
    ptok = usage.get("prompt_tokens", "?")
    print("cached=%s ptok=%s %.1fs 命中=%s 期望=%d 实得=%s  out=%s"
          % (cached, ptok, dt, "✓" if str(NEEDLE) in text else "✗",
             NEEDLE, [x for x in text.split() if x.isdigit()][:3], text[:60].replace("\n", " ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
