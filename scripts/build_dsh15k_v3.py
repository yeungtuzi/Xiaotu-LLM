#!/usr/bin/env python3
"""Build a **well-posed** ~15K benchmark from real DSH sessions (v3).

Difference from v2:
  * cut at a **complete user turn**, never in the middle of assistant text;
  * render with the official DeepSeek-V4.1 encoding (`encoding/encoding.py`),
    so the prompt ends with `<｜Assistant｜><think>` (thinking mode);
  * the final user message is a real task (length/selection filters), and the
    assistant response after it is **not** included.

Usage (remote, with the 1017 tokenizer):
  TOKENIZER=<1017 snapshot> python scripts/build_dsh15k_v3.py \
      --sessions /home/user/.dsh/sessions \
      --out /home/user/lvllm/process_data/data/dsh_15k_v3.jsonl \
      --rows 32 --target 15000
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_v41_encoding import encode_messages  # noqa: E402


def iter_events(path):
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


def _text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for c in content:
        if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
            out.append(c["text"])
    return "\n".join(out)


def session_messages(path):
    """Return [{'role','content'}] in event order; skip tool/runtime-context."""
    msgs = []
    for e in iter_events(path):
        t = e.get("type")
        d = e.get("data") or {}
        if t == "system/message":
            m = d.get("message") or {}
            txt = _text(m.get("content"))
            if txt and (not msgs or msgs[0].get("role") != "system"):
                msgs.append({"role": "system", "content": txt})
        elif t == "user/message":
            src = d.get("source") or {}
            if src.get("kind") != "user":
                continue
            txt = _text(d.get("content"))
            if txt:
                msgs.append({"role": "user", "content": txt})
        elif t == "assistant/message":
            m = d.get("message") or {}
            txt = _text(m.get("content"))
            if txt:
                msgs.append({"role": "assistant", "content": txt})
    return msgs


def looks_like_task(s: str) -> bool:
    if len(s) < 200:
        return False
    low = s.lower()
    keys = ("?", "？", "请", "写", "实现", "检查", "修复", "分析", "review",
            "add ", "run ", "fix ", "implement", "```", "|", "任务", "要求")
    return any(k in low for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="/home/user/.dsh/sessions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--target", type=int, default=15000)
    # ⭐【2026-10-31 放宽】原 14500–15500 是"整段前缀必须恰好落在该区间"，
    #   实测 203 个会话一条都凑不出（前缀 token 数是跳跃的）。改为：
    #   "允许从最前面丢弃更早的回合"，只要最终 prompt 落在 [lo, hi] 且**以 user 回合结尾**。
    #   上界 16,300 ≈ MBT=8192 × 2 ⇒ 预填正好两个 chunk。
    ap.add_argument("--lo", type=int, default=10500)
    ap.add_argument("--hi", type=int, default=16300)
    ap.add_argument("--per-session", type=int, default=2)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER"], trust_remote_code=True)
    files = sorted(glob.glob(os.path.join(a.sessions, "**", "session.v4.jsonl.zstd"),
                             recursive=True))
    print(f"会话文件 {len(files)} 个", flush=True)

    rows, seen = [], set()
    for fi, f in enumerate(files):
        if len(rows) >= a.rows:
            break
        msgs = session_messages(f)
        if not msgs or msgs[0].get("role") != "system":
            continue
        best = []
        for i, m in enumerate(msgs):
            if m["role"] != "user" or not looks_like_task(m["content"]):
                continue
            # ⭐ 从"最近的 user 回合"往回试不同起点（丢弃更早的回合），取最接近 target 的窗口
            cand = None
            starts = [j for j in range(0, i + 1)
                      if msgs[j]["role"] == "user" or j == 0]
            for st in reversed(starts):
                sub = ([msgs[0]] if st > 0 else []) + msgs[st:i + 1]
                try:
                    pr = encode_messages(sub, thinking_mode="thinking",
                                         drop_thinking=True, reasoning_effort="high")
                except Exception:
                    continue
                if not pr.endswith("<｜Assistant｜><think>"):
                    continue
                nn = len(tok(pr, add_special_tokens=False)["input_ids"])
                if nn > a.hi:
                    continue
                if nn >= a.lo:
                    cand = (abs(nn - a.target), nn, i, pr, len(m["content"]))
                    break
            if cand is None:
                continue
            _, n, i, prompt, uc = cand
            if True:
                h = hashlib.sha1(prompt[-2000:].encode()).hexdigest()[:16]
                if h not in seen:
                    # ⭐ 同时存下原始对话（供官方 API 生成参考回答做语义对照）
                    best.append((abs(n - a.target), n, i, prompt, h,
                                 len(m["content"]), sub))
        best.sort()
        for _, n, i, prompt, h, uc, sub in best[:a.per_session]:
            if len(rows) >= a.rows:
                break
            if h in seen:
                continue
            seen.add(h)
            rows.append({"prompt": prompt, "tokens": n,
                         "source": os.path.basename(os.path.dirname(f)),
                         "turn": i, "user_chars": uc,
                         "messages": sub,
                         "final_user": sub[-1]["content"]})
        if (fi + 1) % 20 == 0:
            print(f"  扫描 {fi + 1}/{len(files)} ⇒ 已得 {len(rows)} 条", flush=True)

    if len(rows) < a.rows:
        print(f"⚠️ 只凑到 {len(rows)} 条(目标 {a.rows})", file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ts = sorted(r["tokens"] for r in rows)
    print(f"✓ {len(rows)} 条 → {a.out};token 中位 "
          f"{ts[len(ts)//2] if ts else 0};范围 {ts[0] if ts else 0}-{ts[-1] if ts else 0}",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
