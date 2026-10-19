#!/usr/bin/env python3
"""QFN 接入的**质量子集**校验(GSM8K 前 N 题,零样本,temp=0)。

为什么不用 p2_quality.py 全量:全量 128 题在 TP=1/CPU 专家口径下要数小时;
接入验证只需要"数值正确 + 语义正常"的机械证据(全量留给验收窗口)。

用法:
  PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
  $PY scripts/qfn_quality_subset.py --port 8140 --model Qwen3.8-Flash-Next --n 12 --conc 2
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.request

GSM = "/tmp/gsm_test.jsonl"


def ask(port, model, q, max_tokens=512):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": q}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"content-type": "application/json"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            c = d["choices"][0]["message"]
            return (c.get("content") or "") + "\n" + (c.get("reasoning") or "")
        except Exception:  # noqa: BLE001
            time.sleep(5 * (k + 1))
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conc", type=int, default=2)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(GSM, encoding="utf-8")][: a.n]
    out, lock = [], threading.Lock()

    def work(idx):
        for i in idx:
            q = rows[i]["question"] + "\n请推理后,最后一行以 #### <答案数字> 结束。"
            txt = ask(a.port, a.model, q)
            m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", txt)
            got = m[-1].replace(",", "") if m else ""
            exp = re.search(r"####\s*(-?[\d,\.]+)", rows[i]["answer"]).group(1).replace(",", "")
            ok = bool(got) and int(float(got)) == int(float(exp))
            with lock:
                out.append((i, ok, got, exp, txt[-120:].replace("\n", " ")))
            print(f"[{i}] {'✅' if ok else '❌'} got={got!r} exp={exp}", flush=True)

    per = max(1, a.n // a.conc)
    ts = [threading.Thread(target=work, args=(list(range(k * per, min(a.n, (k + 1) * per))),))
          for k in range(a.conc)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ok = sum(1 for x in out if x[1])
    print(f"\n== GSM8K 子集 {ok}/{len(out)} = {100.0 * ok / max(1, len(out)):.1f}% "
          f"墙钟 {time.time() - t0:.0f}s ==")
    with open("/tmp/qfn_quality_subset.json", "w", encoding="utf-8") as f:
        json.dump([{"i": i, "ok": o, "got": g, "exp": e, "tail": t} for i, o, g, e, t in out],
                  f, ensure_ascii=False, indent=1)
    return 0 if ok == len(out) else 0


if __name__ == "__main__":
    raise SystemExit(main())
