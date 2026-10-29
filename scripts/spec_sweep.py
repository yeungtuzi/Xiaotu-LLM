#!/usr/bin/env python3
"""spec_sweep.py —— 系统性扫"从未出现过的特化"，用来证实/否证
                    "_fp8_mqa_logits_kernel 首次加载新特化 ⇒ 崩溃" 这个假设。

背景（已由 py-spy 活捉两次：r101/r106）
    崩溃点：`_init_handles` ← `fp8_mqa_logits_triton` ← `_idx_emit`(稀疏索引器)
    `_init_handles` 只在该 Triton 内核**某个特化在本进程首次 launch** 时被调用。

Triton 的特化维度（本内核相关）
    · `num_q`      = 本次前向的 token 数（prefill 是 chunk 大小，decode 是 批×投机长）
    · `seq_len_kv` = KV 长度
    · 实参整除性质：Triton 默认对 `==1`、`%16==0` 生成不同特化
    · stride 的整除性质
⇒ 因此"没见过的特化"不只来自 prompt 变长，还来自：
    ① 单 token（num_q==1）  ② 16 的边界(15/16/17…)  ③ 8192 chunk 后的 1-token 尾巴
    ④ 两路并发（MAXSEQS=2）的批组合  ⑤ max_tokens/投机长的组合

用法（服务就绪后自动开始；任一步失败即停，把现场留给看门狗）：
    python scripts/spec_sweep.py
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get("SWEEP_URL", "http://127.0.0.1:8700/v1/chat/completions")
MODEL = os.environ.get("SWEEP_MODEL", "DeepSeek-V4.1-Flash")
HEALTH = os.environ.get("SWEEP_HEALTH", "http://127.0.0.1:8700/health")
WAIT = os.environ.get("SWEEP_WAIT_HEALTH", "1") == "1"
CHARS_PER_TOK = float(os.environ.get("SWEEP_CHARS_PER_TOK", "1.7"))

FILLER = "拓扑排序与张量并行填充文本，长度需精确控制，内容不参与判断。" * 900


def body_for(tokens: int, max_tokens: int = 16):
    chars = max(1, int(tokens * CHARS_PER_TOK))
    prompt = FILLER[:chars] + "\n用一句话总结。"
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
    }


def call(payload, timeout=1800):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        u = d.get("usage", {})
        return True, time.time() - t0, u.get("prompt_tokens"), u.get("completion_tokens"), ""
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, None, None, "%s: %s" % (type(e).__name__, str(e)[:130])


def wait_health(timeout_s=900):
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


def stage(name, items):
    """items: [(描述, tokens, max_tokens)] —— 串行执行，遇错即停。"""
    print("--- [%s] ---" % name, flush=True)
    for desc, tok, mt in items:
        ok, dt, pt, ct, err = call(body_for(tok, mt))
        if ok:
            print("OK   %-28s target=%-6d pt=%-6s ct=%-4s %.1fs" % (desc, tok, pt, ct, dt), flush=True)
        else:
            print("FAIL %-28s target=%-6d %.1fs %s" % (desc, tok, dt, err), flush=True)
            print("== 命中：该请求构造了此前未见过的特化，崩溃已复现 ==", flush=True)
            return False
    return True


def stage_concurrent(name, pairs):
    """两路并发（MAXSEQS=2）：同时发两个不同长度，制造新的批组合。"""
    print("--- [%s] ---" % name, flush=True)
    for toks in pairs:
        res = {}

        def run(i, n):
            res[i] = call(body_for(n, 16))

        ts = [threading.Thread(target=run, args=(i, n)) for i, n in enumerate(toks)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        bad = [k for k, v in res.items() if not v[0]]
        if bad:
            for k in bad:
                print("FAIL 并发 %s 第%d路 %s" % (toks, k, res[k][4]), flush=True)
            print("== 命中：并发批组合触发崩溃 ==", flush=True)
            return False
        print("OK   并发 %-16s %s" % (toks, ["%.1fs" % res[i][1] for i in range(len(toks))]), flush=True)
    return True


def main():
    if WAIT and not wait_health():
        print("FATAL 服务未就绪", flush=True)
        return 2
    print("== 特化扫描开始 ==", flush=True)

    # ① num_q==1 与极小值（Triton 对 ==1 单独特化；decode 单步也是 1）
    if not stage("极小/单token", [("1 token", 1, 1), ("2 tokens", 2, 1), ("3 tokens", 3, 1),
                                 ("1 token 长生成", 1, 64)]):
        return 1
    # ② 16 的边界
    if not stage("16 边界", [("15", 15, 8), ("16", 16, 8), ("17", 17, 8),
                            ("31", 31, 8), ("32", 32, 8), ("33", 33, 8),
                            ("47", 47, 8), ("48", 48, 8), ("49", 49, 8)]):
        return 1
    # ③ 质数/奇数长度（整除性质不同）
    if not stage("质数长度", [("101", 101, 8), ("103", 103, 8), ("107", 107, 8),
                            ("109", 109, 8), ("113", 113, 8), ("1009", 1009, 8),
                            ("1021", 1021, 8), ("2039", 2039, 8)]):
        return 1
    # ④ 8192 chunk 边界：让尾部 chunk 恰好 1 token（num_q==1 但 seq_len_kv 很大）
    if not stage("chunk 尾部 1 token", [("8191", 8191, 8), ("8192", 8192, 8), ("8193", 8193, 8),
                                       ("16385", 16385, 8)]):
        return 1
    # ⑤ 并发批组合
    if not stage_concurrent("并发组合", [(17, 33), (1, 8193), (101, 103), (2048, 2051)]):
        return 1
    # ⑥ 长上下文收尾
    if not stage("长上下文", [("24577", 24577, 8), ("32771", 32771, 8)]):
        return 1

    print("== 扫描全部通过：**否证**『新特化首次加载触发崩溃』 ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
