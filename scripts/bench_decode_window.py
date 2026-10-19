#!/usr/bin/env python3
"""纯解码窗口基准:先让 C 条 15K 请求全部进入解码,再在窗口内计 token 速率。

为什么需要它(2026-10-19 教训):
  `bench_agent_throughput` 的"聚合解码"分母是**整条请求的解码段**;
  在"16 条 × 15K、自然结束"下,输出只有几十~几百 token,而每换一批就有一次
  15K 冷预填插进来 ⇒ **等预填的时间也被算进解码段** ⇒ 聚合解码被严重低估 ✗
  ⇒ 本脚本只在"所有请求都已在解码"的窗口里计数,量的是**纯解码扩展性** ✓

用法:
  python scripts/bench_decode_window.py --port 8140 --model Qwen3.8-Flash-Next \
      --conc 4 --dataset /home/user/lvllm/process_data/data/dsh_15k_v2.jsonl \
      --window 12 --max-tokens 512
输出: 窗口内聚合 tok/s(以及每请求速率、步时中位)。
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request


def stream_request(prompt, port, model, max_tokens, out):
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % port,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    t0 = time.time()
    times = out.setdefault("times", [])
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                j = json.loads(d)
                ch = j.get("choices") or []
                if not ch:
                    continue
                if ch[0].get("text"):
                    now = time.time()
                    if "first" not in out:
                        out["first"] = now      # ⭐ 必须**实时**写:轮询线程靠它判"已首 token"
                    times.append(now)
                    # ⭐ 2026-10-19:DSH 的 DSpark 一次 step 可能在一个 SSE chunk 里带多个 token;
                    #   只数 chunk 会把 decode 低估 AL 倍。保存文本,窗口结束后用 tokenizer 精确计数。
                    out.setdefault("texts", []).append((now, ch[0]["text"]))
    except Exception as exc:  # noqa: BLE001
        out["err"] = f"{type(exc).__name__}: {exc}"
        return
    out["t0"] = t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--window", type=float, default=12.0, help="解码窗口秒数")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=0, help="用前 n 条(默认=conc)")
    ap.add_argument("--tokenizer", default="",
                    help="模型目录/tokenizer 路径;给了就按【token 数】算 decode(推荐,DSH/DSpark 必给)")
    a = ap.parse_args()

    rows = [json.loads(x) for x in open(a.dataset, encoding="utf-8") if x.strip()]
    n = a.n or a.conc
    prompts = [rows[i % len(rows)]["prompt"] for i in range(n)]
    outs = [{} for _ in range(n)]
    ths = [threading.Thread(target=stream_request,
                            args=(p, a.port, a.model, a.max_tokens, o))
           for p, o in zip(prompts, outs)]
    for t in ths:
        t.start()

    # 等全部首 token(最多 120s)
    t_wait = time.time()
    while time.time() - t_wait < 120:
        if all(o.get("first") for o in outs):
            break
        time.sleep(0.05)
    t_ready = time.time()

    time.sleep(a.window)
    t_end = time.time()

    # 步时口径:窗口内每请求相邻 token 间隔的中位(对短输出也稳)
    import statistics as st
    steps = []
    for o in outs:
        ts = [t for t in o.get("times", []) if t_ready <= t <= t_end]
        if len(ts) >= 2:
            steps.append(st.median([b - a for a, b in zip(ts, ts[1:])]) * 1000.0)
    if not steps:
        diag = [(len(o.get("times", [])), o.get("err", "")) for o in outs]
        print(f"并发={a.conc} ⚠️ 窗口内样本不足 diag={diag}"); return 1
    step = st.median(steps)

    # ⭐ token 口径(主判据):窗口内文本用 tokenizer 重新计数;没有 tokenizer 才退回 chunk 数。
    tok = None
    if a.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
            print(f"[decode-window] tokenizer={a.tokenizer}")
        except Exception as exc:  # noqa: BLE001
            print(f"[decode-window] ⚠️ tokenizer 加载失败,退回 chunk 口径: {exc}")
    rates, ntok_sum = [], 0
    for o in outs:
        pairs = [(t, x) for (t, x) in o.get("texts", []) if t_ready <= t <= t_end]
        txt = "".join(x for _, x in pairs)
        nt = len(tok.encode(txt, add_special_tokens=False)) if tok is not None else len(pairs)
        if nt > 0:
            rates.append(nt / a.window)
            ntok_sum += nt
    if rates:
        print(f"并发={a.conc} 就绪耗时={t_ready - t_wait:.1f}s 步时中位={step:.1f} ms "
              f"⇒ **每请求 {st.median(rates):.1f} tok/s,聚合 {ntok_sum / a.window:.1f} tok/s** "
              f"(token 口径;窗口内 tokens={ntok_sum})")
    else:
        print(f"并发={a.conc} 就绪耗时={t_ready - t_wait:.1f}s 步时中位={step:.1f} ms "
              f"⇒ 每请求 {1000 / step:.1f} tok/s,聚合 {a.conc * 1000 / step:.1f} tok/s "
              f"(chunk 口径;⚠️ 未给 --tokenizer,DSH/DSpark 会低估 AL 倍)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
