#!/usr/bin/env python3
"""tail_chunk_probe.py —— 精确复现"小尾块 + 长上下文"的特化

依据（2026-10-29 三次活捉到同一栈：r101 / r106 / r108）
    崩溃栈恒为 `_init_handles` ← `fp8_mqa_logits_triton` ← `_idx_emit`（稀疏索引器）
    而每次死亡前正在跑的都是 CPU-MoE 预填，其 `qlen` 依次为 174 / 188 / 203 / 822 / 1194
    ⇒ 这些是**调度器切出的 chunk 大小**，不是整条 prompt 长度。

    `spec_sweep.py` 之所以没复现：它发的是"整条 prompt = 某个怪长度"，因此
    `seq_len_kv` 与 `num_q` 同步变化；而真实场景是 **`num_q` 很小、`seq_len_kv` 很大**。

构造（关键）
    让 prompt 长度 = `CHUNK * k + q`（CHUNK=8192 为分块大小）
      ⇒ 前 k 块各 8192 token，**最后一块正好 q 个 token**，此时 `seq_len_kv ≈ CHUNK*k`
    ⇒ 精确造出 (num_q=q, seq_len_kv≈几十万) 这个组合，而它在正常短请求里不会出现。

用法：python scripts/tail_chunk_probe.py
环境：TCP_URL / TCP_HEALTH / TCP_CHUNK / TCP_KS / TCP_QS
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("TCP_URL", "http://127.0.0.1:8700/v1/chat/completions")
MODEL = os.environ.get("TCP_MODEL", "DeepSeek-V4.1-Flash")
HEALTH = os.environ.get("TCP_HEALTH", "http://127.0.0.1:8700/health")
WAIT = os.environ.get("TCP_WAIT_HEALTH", "1") == "1"

CHUNK = int(os.environ.get("TCP_CHUNK", "8192"))
KS = [int(x) for x in os.environ.get("TCP_KS", "12,48").split(",")]
QS = [int(x) for x in os.environ.get("TCP_QS", "203,174,188,822,1194").split(",")]

UNIT = "分布式张量并行与拓扑排序的填充文本，用于精确控制请求长度，内容不参与判断。"


def text_for(tokens: int) -> str:
    need = max(1, int(tokens * 1.7))
    return (UNIT * (need // len(UNIT) + 2))[:need]


def wait_health(timeout_s: int = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(HEALTH, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def ask(tokens: int, max_tokens: int = 4, timeout: int = 3600):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text_for(tokens) + "\n用一句话总结。"}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        u = d.get("usage", {})
        return True, time.time() - t0, u.get("prompt_tokens"), ""
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, None, "%s: %s" % (type(e).__name__, str(e)[:130])


def main() -> int:
    if WAIT and not wait_health():
        print("FATAL 服务未就绪", flush=True)
        return 2
    print("== 尾块特化探针开始（CHUNK=%d）==" % CHUNK, flush=True)
    for k in KS:
        for q in QS:
            n = CHUNK * k + q
            ok, dt, pt, err = ask(n)
            tag = "k=%d q=%d prompt=%d" % (k, q, n)
            if ok:
                print("OK   %-34s pt=%-7s %.1fs" % (tag, pt, dt), flush=True)
            else:
                print("FAIL %-34s %.1fs %s" % (tag, dt, err), flush=True)
                print("== 命中：(num_q≈%d, seq_len_kv≈%d) 复现 ==" % (q, CHUNK * k), flush=True)
                return 1
            time.sleep(1)
    print("== 全部通过：尾块组合也未触发 ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
