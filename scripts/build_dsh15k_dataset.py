#!/usr/bin/env python3
"""从**真实 DSH 会话**日志构建 ~15K token 的基准数据集。

背景(2026-10-19 用户裁定):现有 `dsh_assembled_bench_15k.jsonl` 只有 8 条,
自然结束口径下解码步样本太少;而本机 DSH 会话"足够多也足够长" ⇒ 直接从
`~/.dsh/sessions/**/session.v4.jsonl.zstd` 挖真实多轮会话,切成 15K 窗口。

用法:
  TOKENIZER=<qwen snapshot> python scripts/build_dsh15k_dataset.py \
      --sessions /home/user/.dsh/sessions \
      --out /home/user/lvllm/process_data/data/dsh_15k_v2.jsonl \
      --rows 32 --target 15000 --stride 15000
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

from transformers import AutoTokenizer


def iter_events(path):
    """zstd -dc 解压后逐行 JSON。"""
    p = subprocess.run(["zstd", "-dc", path], capture_output=True)
    if p.returncode != 0:
        return
    for line in p.stdout.decode("utf-8", "ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:  # noqa: BLE001
            continue


def session_text(path):
    """把会话里的 user/assistant 文本按顺序拼起来(跳过 reasoning/tool 参数)。"""
    parts = []
    for e in iter_events(path):
        t = e.get("type")
        if t == "user/message":
            for c in (e.get("data") or {}).get("content") or []:
                if c.get("type") == "text" and c.get("text"):
                    parts.append(f"\n[user]\n{c['text']}")
        elif t == "assistant/message":
            msg = (e.get("data") or {}).get("message") or {}
            for c in msg.get("content") or []:
                if c.get("type") == "text" and c.get("text"):
                    parts.append(f"\n[assistant]\n{c['text']}")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="/home/user/.dsh/sessions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=15000)
    a = ap.parse_args()
    stride = a.stride or a.target

    tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER"], trust_remote_code=True)
    files = sorted(glob.glob(os.path.join(a.sessions, "**", "session.v4.jsonl.zstd"),
                             recursive=True))
    print(f"会话文件 {len(files)} 个", flush=True)

    rows, seen = [], set()
    for fi, f in enumerate(files):
        if len(rows) >= a.rows:
            break
        text = session_text(f)
        if len(text) < 4000:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < a.min_tokens:
            continue
        for off in range(0, len(ids) - a.target + 1, stride):
            win = ids[off:off + a.target]
            prompt = tok.decode(win, skip_special_tokens=False)
            h = hashlib.sha1(prompt[:400].encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            rows.append({"prompt": prompt, "tokens": len(win),
                         "source": os.path.basename(os.path.dirname(f))})
            if len(rows) >= a.rows:
                break
        if (fi + 1) % 20 == 0:
            print(f"  扫描 {fi + 1}/{len(files)} ⇒ 已得 {len(rows)} 条", flush=True)

    if len(rows) < a.rows:
        print(f"⚠️ 只凑到 {len(rows)} 条(目标 {a.rows});可调小 --target 或 --min-tokens",
              file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {len(rows)} 条 → {a.out};token 中位 "
          f"{sorted(r['tokens'] for r in rows)[len(rows)//2] if rows else 0}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
