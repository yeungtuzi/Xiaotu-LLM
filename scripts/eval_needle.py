#!/usr/bin/env python3
"""Needle-in-a-haystack 检索评测（走 OpenAI 兼容接口，纯 stdlib）。

用途：验证长上下文检索能力在优化前后是否退化。合成填充文本 + 埋入唯一"针"，
按深度（上下文比例）与长度取样，判分只看回复里是否出现针的内容。

用法：
    PORT=8071 N=10 LEN=8192 bash -c 'python scripts/eval_needle.py'
    PORT=8700 N=20 LEN=32768 DEPTHS=0.1,0.5,0.9 python scripts/eval_needle.py

环境变量：
    PORT    服务端口（默认 8071）
    N       总样本数（默认 12，在 DEPTHS × 长度上轮转）
    LEN     上下文目标字符数（默认 32000，约 8K token；8192/32768 亦可）
    DEPTHS  针的插入位置（占上下文比例，逗号分隔；默认 0.1,0.5,0.9）
    MAXTOK  生成上限（默认 32）
    OUT     结果 JSON 路径（默认 /tmp/needle_result.json）

判据：回复中出现针里的数字串即算命中（数字唯一、不会被填充文本误撞）。
"""
import json
import os
import random
import re
import sys
import urllib.request

PORT = os.environ.get("PORT", "8071")
N = int(os.environ.get("N", "12"))
LEN = int(os.environ.get("LEN", "32000"))
DEPTHS = [float(x) for x in os.environ.get("DEPTHS", "0.1,0.5,0.9").split(",")]
MAXTOK = int(os.environ.get("MAXTOK", "32"))
OUT = os.environ.get("OUT", "/tmp/needle_result.json")

FILLER = (
    "The archive room holds maintenance logs from the previous decade. "
    "Shelves are labelled by quarter, and each binder records routine checks, "
    "spare parts orders, and shift handovers. Nothing in these pages is unusual; "
    "the entries describe inspections of pumps, valves, and ventilation ducts. "
)


def build_context(target_chars, depth, needle, rng):
    """在 target_chars 长度的填充文本中，按 depth 比例插入针。"""
    filler = (FILLER * (target_chars // len(FILLER) + 2))[:target_chars]
    pos = int(len(filler) * depth)
    # 断在空格处，避免把单词切开
    pos = filler.rfind(" ", 0, pos) if pos > 0 else 0
    return filler[:pos] + "\n" + needle + "\n" + filler[pos:]


def ask(prompt, max_tokens):
    body = json.dumps({
        "model": "DeepSeek-V4.1-Flash", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "ignore_eos": False,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % int(PORT), data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["text"]


def main():
    rng = random.Random(20261029)
    rows = []
    hits = 0
    for i in range(N):
        depth = DEPTHS[i % len(DEPTHS)]
        code = "%06d" % rng.randrange(1000000)          # 唯一针
        needle = ("IMPORTANT NOTE: the vault access code for today is %s. "
                  "This code must be repeated verbatim when asked." % code)
        ctx = build_context(LEN, depth, needle, rng)
        prompt = (ctx + "\n\nQuestion: What is the vault access code for today? "
                  "Answer with the code only.\nAnswer:")
        try:
            out = ask(prompt, MAXTOK)
            ok = code in out
        except Exception as e:                            # noqa: BLE001
            out, ok = "ERROR: %s" % e, False
        hits += 1 if ok else 0
        rows.append({"i": i, "depth": depth, "chars": len(ctx), "code": code,
                     "answer": out.strip()[:200], "hit": ok})
        print("[needle] i=%d depth=%.1f chars=%d hit=%s ans=%r"
              % (i, depth, len(ctx), ok, out.strip()[:40]))

    res = {"port": PORT, "n": N, "len_chars": LEN, "depths": DEPTHS,
           "hits": hits, "acc": hits / max(1, N), "details": rows}
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("\n[needle] %d/%d 命中 (acc=%.3f) ⇒ %s" % (hits, N, hits / max(1, N), OUT))
    return 0 if hits == N else 1


if __name__ == "__main__":
    sys.exit(main())
