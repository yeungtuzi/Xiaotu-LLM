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
    ap.add_argument("--maxtok", type=int, default=2048)
    ap.add_argument("--sig", action="store_true",
                    help="额外打印完整回复文本的 sha256 前 16 位（用于“输出签名唯一”判据）")
    a = ap.parse_args()
    needle_row = int(a.n * a.depth)
    uid = a.uuid or os.urandom(8).hex()
    body = doc(a.n, needle_row, NEEDLE)
    prompt = ("You are a helpful assistant. Answer using only the document.\n\n"
              + uid + "\n\n" + body
              + "\n\nWhat is the access code for project Q? Answer with the number only.")
    # ⛔ 2026-11-10 23:xx 实测更正（台账 #274）：**必须走 `/v1/chat/completions`**。
    #    早先据"tokenizer_config 里没有 chat_template 键"推断 chat 路径不可用 ⇒ 改用 /v1/completions，
    #    结果真权重下模型**立刻 EOS**（text='' / completion_tokens=1），加 ignore_eos 后输出
    #    完全不相关的胡话（它把这当成续写任务）。
    #    实测 `/v1/chat/completions` ⇒ `740317`（正确、44 tokens）⇒ 本模型是**指令/对话**形态，
    #    服务端有其自带的对话模板。
    payload = {"model": "DeepSeek-V4.1-Flash",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": a.maxtok, "temperature": 0, "top_p": 1}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        d = json.loads(resp.read())
    dt = time.time() - t0
    ch = d["choices"][0]
    text = (ch.get("message") or {}).get("content") or ""
    usage = d.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens", "?")
    ptok = usage.get("prompt_tokens", "?")
    # ⭐ 台账 #278：必须打印 finish_reason 与完成 token 数 —— "空输出"到底是
    #    finish=length（思考吃掉预算）还是 finish=stop（模型真的立刻收尾）完全不同。
    fin = ch.get("finish_reason")
    ctok = usage.get("completion_tokens")
    rz = (ch.get("message") or {}).get("reasoning") or ""
    _sig = ""
    if a.sig:
        import hashlib
        _h = hashlib.sha256((text + "\u0000" + rz).encode()).hexdigest()[:16]
        _sig = " sig=%s" % _h
    print("cached=%s ptok=%s %.1fs 命中=%s 期望=%d 实得=%s  finish=%s ctok=%s rzlen=%d  out=%s"
          % (cached, ptok, dt, "✓" if str(NEEDLE) in text else "✗",
             NEEDLE, [x for x in text.split() if x.isdigit()][:3], fin, ctok, len(rz),
             text[:60].replace("\n", " ")), end="")
    print(_sig if a.sig else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
