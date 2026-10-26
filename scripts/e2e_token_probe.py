#!/usr/bin/env python3
"""贪心 e2e 探针：同一组 prompt 喂两个树，输出逐字对比（含长上下文用例）。

用法:
    python3 scripts/e2e_token_probe.py --port 8701 --out /tmp/e2e_new.json [--nonce xyz]

判据:
    * 两个 JSON 的 `texts` 列表**逐元素完全相同** ⇒ 逐 token 一致；
    * `usage.prompt_tokens` 也必须相同（否则不是同一个输入）。

⚠️ 为什么带 `--nonce`：vLLM 有 prefix cache，同一 prompt 第二次命中会**跳过 prefill**
   （实测 0.99 s vs 21.3 s）⇒ 长上下文用例必须破坏缓存，否则测的不是同一条路径。
   本脚本默认每次运行都加随机 nonce（写在 prompt 末尾的注释里，不影响语义）。
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
import urllib.request

PROMPTS = [
    # 1) 短：算术 + 推理，考查采样是否为贪心
    "Compute 17 * 23 - 19 step by step, then give the final number on its own line.",
    # 2) 短：代码，考查 tokenizer/模板一致性
    "Write a Python one-liner that returns the second largest element of a list, "
    "then explain in one sentence.",
    # 3) 中文：考查 chat template / 思考强度
    "用一句话解释什么是 NUMA，然后列出两个和显存带宽有关的指标。",
]


def _long_context_prompt(target_chars: int = 12000) -> str:
    """构造一个长上下文用例（约 12k 字符 ≈ 3-4k token），末尾才是真问题。

    内容用确定性填充（不随机），保证两个树拿到**完全相同**的正文；
    只有末尾 nonce 注释不同，用来打破 prefix cache。
    """
    head = (
        "You are given a long log excerpt. Read it, then answer the question at the end.\n\n"
        "--- BEGIN LOG ---\n"
    )
    body = "".join(
        f"2026-10-26T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z "
        f"shard={i % 8} layer={i % 40} event=stream_sync_stage status=ok "
        f"cost_ms={(i * 37) % 991}.{(i * 13) % 10}\n"
        for i in range(220)
    )
    tail = (
        "--- END LOG ---\n\n"
        "Question: in the log above, which layer value appears most often in the "
        "'layer=' field? Answer with just the integer.\n"
    )
    s = head + body + tail
    return s[:target_chars] if len(s) > target_chars else s


def post(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    out["_elapsed_s"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--nonce", default=None, help="缺省每次随机（破 prefix cache）")
    a = ap.parse_args()

    nonce = a.nonce or "".join(random.choices(string.ascii_letters + string.digits, k=12))
    long_p = _long_context_prompt() + f"\n<!-- probe nonce: {nonce} -->\n"

    cases = [(f"short{i + 1}", p, a.max_tokens) for i, p in enumerate(PROMPTS)]
    cases.append(("long_context", long_p, 16))

    rec = {"nonce": nonce, "cases": []}
    for name, prompt, mt in cases:
        try:
            r = post(a.port, a.model, prompt + f"\n<!-- nonce {nonce} -->", mt)
            rec["cases"].append({
                "name": name,
                "text": r["choices"][0]["text"],
                "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                "finish_reason": r["choices"][0].get("finish_reason"),
                "elapsed_s": r["_elapsed_s"],
            })
            print(f"  ✓ {name}: prompt_tokens={rec['cases'][-1]['prompt_tokens']} "
                  f"{rec['cases'][-1]['elapsed_s']}s")
        except Exception as e:  # noqa: BLE001
            rec["cases"].append({"name": name, "error": f"{type(e).__name__}: {e}"})
            print(f"  ✗ {name}: {type(e).__name__}: {e}")

    with open(a.out, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"已写 {a.out}（nonce={nonce}）")
    return 0 if all("text" in c for c in rec["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
