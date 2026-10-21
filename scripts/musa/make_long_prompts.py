#!/usr/bin/env python3
"""Generate 8K/32K chat prompts once a tokenizer is available.

Run on the S4000 host (transformers installed).  Keeps the same prompt bytes
for the CUDA reference and MUSA candidate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

SEED = (
    "The following is a long, neutral technical context used only to exercise "
    "prefill. It contains no instructions for the model. "
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF repo or local tokenizer path")
    ap.add_argument("--target-tokens", type=int, default=8192)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--questions", type=Path, default=None,
                    help="optional JSONL with {id,messages}; defaults to one question")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    filler = SEED
    # Grow by whole sentences to keep the context readable.
    while len(tok.encode(filler)) < args.target_tokens:
        filler += SEED
    # Trim to target with tokenizer boundaries.
    ids = tok.encode(filler)[: args.target_tokens]
    context = tok.decode(ids, skip_special_tokens=True)

    if args.questions:
        items = [json.loads(x) for x in args.questions.read_text().splitlines() if x.strip()]
    else:
        items = [{"id": "long-chain", "messages": [
            {"role": "user", "content": "Summarize the key requirement in one sentence."}
        ]}]

    rows = []
    for item in items:
        msgs = list(item.get("messages", []))
        # Put the long context in a user turn, then the original question.
        msgs = [{"role": "user", "content": context}] + msgs
        rows.append({"id": f"{item.get('id', 'long')}-{args.target_tokens}", "messages": msgs})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print("wrote", args.out, "prompts", len(rows), "target_tokens", args.target_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
