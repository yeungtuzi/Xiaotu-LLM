#!/usr/bin/env python3
"""README 口径(v2,固定输出长度)基准:prefill / decode / 步时。

与 `bench_agent_throughput.py` 的区别:后者按 QA #264 废止了"强制输出",
而 README 表格(**BENCH_PROTOCOL §2**)要求 `max_tokens=min_tokens=N` + `ignore_eos=true`。
本脚本只做这一件事,便于把 qfn 的数字放回 README 同一张表。

用法:
  python scripts/bench_fixed_protocol.py --port 8140 --model Qwen3.8-Flash-Next \
      --conc 1 --n 16 --outlen 1024 --dataset /home/user/lvllm/process_data/data/dsh_assembled_bench_15k.jsonl
输出一行 JSON(便于入库)+ 人读摘要。
"""
import argparse
import json
import statistics as st
import threading
import time
import urllib.request


def one(prompt, outlen, port, model, timeout):
    body = {
        "model": model, "prompt": prompt,
        "max_tokens": outlen, "min_tokens": outlen, "ignore_eos": True,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % port,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    t0 = time.time()
    first = None
    last = None
    gaps = []
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if not ch:
                continue
            txt = ch[0].get("text") or ""
            if not txt:
                continue
            now = time.time()
            if first is None:
                first = now
            else:
                gaps.append(now - last)
            last = now
    end = time.time()
    return {
        "ttft": (first - t0) if first else None,
        "decode": (end - first) if first else None,
        "end": end - t0,
        "toks": int((usage or {}).get("completion_tokens") or 0),
        "ptok": int((usage or {}).get("prompt_tokens") or 0),
        "gaps": gaps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--outlen", type=int, default=1024)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--jsonl", default="")
    a = ap.parse_args()

    rows = [json.loads(x) for x in open(a.dataset, encoding="utf-8") if x.strip()][: a.n]
    out = [None] * len(rows)
    slots = threading.Semaphore(a.conc)

    def work(i, prompt):
        with slots:
            try:
                out[i] = one(prompt, a.outlen, a.port, a.model, a.timeout)
            except Exception as exc:  # noqa: BLE001
                out[i] = {"err": f"{type(exc).__name__}: {exc}"}

    ths = [threading.Thread(target=work, args=(i, r["prompt"])) for i, r in enumerate(rows)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0

    ok = [r for r in out if r and "err" not in r and r.get("toks")]
    if not ok:
        print("== 全失败 ==", [r for r in out][:2])
        return 1
    sum_tok = sum(r["toks"] for r in ok)
    sum_dec = sum(r["decode"] for r in ok)
    sum_pre = sum(r["ptok"] for r in ok)
    last_start = max(r["ttft"] for r in ok)
    prefill = sum_pre / last_start if last_start > 0 else 0.0
    per_stream = sum_tok / sum_dec if sum_dec > 0 else 0.0
    agg = per_stream * a.conc
    gaps = [g for r in ok for g in r["gaps"]]
    res = {
        "conc": a.conc, "n_ok": len(ok), "wall_s": round(wall, 1),
        "prefill_tok_s": round(prefill, 0),
        "decode_tok_s": round(agg, 1),
        "decode_per_stream": round(per_stream, 1),
        "step_ms_median": round(1000 * st.median(gaps), 2) if gaps else None,
        "prompt_tokens_sum": sum_pre, "out_tokens_sum": sum_tok,
    }
    print(json.dumps(res, ensure_ascii=False))
    if a.jsonl:
        with open(a.jsonl, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps({k: v for k, v in (r or {}).items() if k != "gaps"},
                                   ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
