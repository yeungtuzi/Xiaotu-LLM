#!/usr/bin/env python3
"""Compare two OpenAI-compatible endpoints for greedy correctness.

Pure stdlib; no torch. Intended for CUDA reference vs MUSA candidate.

Example:
  python scripts/musa/compare_openai.py \
    --ref http://cuda-host:8000 --ref-model qwen38-fp8 \
    --cand http://musa-host:8000 --cand-model qwen38-fp8 \
    --prompts docs/musa/fixtures/prompts.jsonl \
    --max-tokens 256 --out /tmp/musa-correctness.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(base: str, model: str, messages: list[dict[str, str]], max_tokens: int,
         timeout: float, seed: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }
    return post_json(base.rstrip("/") + "/v1/chat/completions", payload, timeout)


def first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference base URL")
    ap.add_argument("--cand", required=True, help="candidate base URL")
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--cand-model", required=True)
    ap.add_argument("--prompts", type=Path, required=True, help="JSONL: {id,messages} or {id,prompt}")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    failures = 0
    for line in args.prompts.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "messages" in item:
            messages = item["messages"]
        else:
            messages = [{"role": "user", "content": item["prompt"]}]
        print(f"[{item.get('id', '?')}] ref ...", flush=True)
        t0 = time.time()
        r = chat(args.ref, args.ref_model, messages, args.max_tokens, args.timeout, args.seed)
        print(f"[{item.get('id', '?')}] cand ...", flush=True)
        c = chat(args.cand, args.cand_model, messages, args.max_tokens, args.timeout, args.seed)
        rt = r["choices"][0]["message"].get("content") or ""
        ct = c["choices"][0]["message"].get("content") or ""
        ok = rt == ct
        if not ok:
            failures += 1
        pos = first_diff(rt, ct)
        rows.append({
            "id": item.get("id"),
            "ok": ok,
            "ref_finish": r["choices"][0].get("finish_reason"),
            "cand_finish": c["choices"][0].get("finish_reason"),
            "ref_chars": len(rt),
            "cand_chars": len(ct),
            "first_diff": pos,
            "ref_head": rt[:300],
            "cand_head": ct[:300],
            "ref_usage": r.get("usage"),
            "cand_usage": c.get("usage"),
            "wall_s": round(time.time() - t0, 3),
        })
        print(f"[{item.get('id', '?')}] {'OK' if ok else 'MISMATCH'} "
              f"ref={len(rt)}c cand={len(ct)}c pos={pos}", flush=True)

    print(f"\n{len(rows) - failures}/{len(rows)} exact matches")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print("wrote", args.out)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as e:
        print("HTTP error:", e.code, e.read().decode(errors="replace")[:1000], file=sys.stderr)
        raise SystemExit(2)
