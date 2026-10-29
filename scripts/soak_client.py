#!/usr/bin/env python3
"""soak_client.py —— 固定短上下文的持续压测（对照用）

目的：与"上下文爬升"的压测并行，用来分辨两类死因
  A 类（有限资源泄漏）: 与上下文长度无关，只随**累计工作量**累加 ⇒ 本流也会推高死亡速度
  B 类（积压/上下文增长）: 本流（固定短上下文）不制造积压 ⇒ 死亡仍只由爬升流驱动

特性
  · 固定 prompt 长度（默认 ~2K token 的中文填充），max_tokens 固定
  · 串行单并发（生产 MAXSEQS=2，用户压测占 1 个槽，本流占另一个）
  · 每行一条记录：时间 / 状态 / 耗时 / prompt_tokens，便于与 metrics.csv 对齐
  · 失败不退出，退避重试（服务重启后自动继续）
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("SOAK_URL", "http://127.0.0.1:8700/v1/chat/completions")
MODEL = os.environ.get("SOAK_MODEL", "DeepSeek-V4.1-Flash")
CHARS = int(os.environ.get("SOAK_CHARS", "3000"))      # ≈2K token 中文
MAXTOK = int(os.environ.get("SOAK_MAXTOK", "64"))
SLEEP = float(os.environ.get("SOAK_SLEEP", "1"))

FILLER = "这是一段用于稳定压测的固定长度技术说明文本，内容不重要，只需要长度恒定。" * 200
PROMPT = FILLER[:CHARS] + "\n请用一句话总结。"


def main() -> int:
    n = 0
    fails = 0
    while True:
        n += 1
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXTOK,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        req = urllib.request.Request(
            URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            print(
                "OK %d %.1fs pt=%s ct=%s" % (n, time.time() - t0, u.get("prompt_tokens"), u.get("completion_tokens")),
                flush=True,
            )
            fails = 0
        except Exception as e:  # noqa: BLE001 - 压测需要吞掉所有异常并继续
            fails += 1
            print(
                "ERR %d %.1fs %s: %s" % (n, time.time() - t0, type(e).__name__, str(e)[:120]),
                flush=True,
            )
            time.sleep(min(30, 2 * fails))
            continue
        time.sleep(SLEEP)


if __name__ == "__main__":
    sys.exit(main())
