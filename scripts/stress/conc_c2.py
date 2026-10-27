#!/usr/bin/env python3
"""conc_c2.py —— 并发（C=2）正确性与稳定性验证。

为什么单独做:
    长稳压测是**单请求串行**，只覆盖了 max-num-seqs=1 的路径。C=2 会真正走到批处理、
    KV 块分配/回收、调度抢占；这一层的历史故障（KV 未释放、显存贴顶）单请求压不出来。

设计（关键是"串话检测"，不只是吞吐）:
    每一轮同时发 2 个请求，**两条 prompt 各自埋一个不同的事实**（不同代号 → 不同人名）；
    判分要看**每条回答是否只包含自己那条 prompt 的事实**：
      - 两个都对   ⇒ 并发路径正确
      - 互相串了   ⇒ 严重的 KV/批处理正确性 bug（比吞吐掉更值得报）
      - 都答不出   ⇒ 能力问题
    同时记录：每轮墙钟耗时、两条各自延迟、生成 token 速率、失败数、以及采样到的 GPU 峰值。

用法:
    python3 conc_c2.py --port 8700 --rounds 20 --ctx-tokens 8000
    python3 conc_c2.py --port 8700 --rounds 5 --ctx-tokens 32000   # 更大的 KV 压力
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_driver as rd  # noqa: E402
import intel_probe as ip     # noqa: E402  # 复用 _corpus_roots / build 语料逻辑

FACTS = [
    ("KX-7", "林昭"),
    ("MVP-3", "周恕"),
]


def build_prompt(code, who, ctx_tokens, corpus):
    """拼一段长上下文，在中间埋入 (代号 → 负责人) 事实，末尾问该代号负责人。"""
    limit = ctx_tokens * 4
    text = "\n\n".join(corpus)[:limit]
    fact = f"【项目记录】代号 {code}；负责人：{who}。"
    pos = len(text) // 2
    text = text[:pos] + "\n\n" + fact + "\n\n" + text[pos:]
    assert fact in text
    return text + f"\n\n问题：代号 {code} 的项目的负责人是谁？只回答人名。"


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=os.path.expanduser("~/conc_c2.jsonl"))
    args = ap.parse_args()

    api_root = f"http://127.0.0.1:{args.port}"
    api = api_root + "/v1/chat/completions"
    model = "DeepSeek-V4.1-Flash"

    code, _ = rd.http_code(api_root + "/health", timeout=5.0)
    if code != 200:
        print(f"⛔ {api_root}/health = {code}，请确认 --port", flush=True)
        return 2

    corpus = []
    for root in ip._corpus_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _d, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith((".md", ".py")):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as f:
                        t = f.read()
                except OSError:
                    continue
                for i in range(0, len(t), 3000):
                    if len(t[i:i + 3000]) > 200:
                        corpus.append(t[i:i + 3000])
    if not corpus:
        print("⛔ 语料为空（可用 STRESS_CORPUS_ROOTS 指定）", flush=True)
        return 2
    print(f"语料块={len(corpus)} 并发=2 轮数={args.rounds} ctx≈{args.ctx_tokens} tokens", flush=True)

    fh = open(args.out, "a", encoding="utf-8")
    stats = {"rounds": 0, "both_ok": 0, "cross_talk": 0, "partial": 0, "error": 0}
    gpu_peak = 0

    for r in range(args.rounds):
        prompts = [build_prompt(c, w, args.ctx_tokens, corpus) for c, w in FACTS]
        results = [None, None]

        def worker(i):
            t0 = time.time()
            try:
                res = rd.chat(api, model, [{"role": "user", "content": prompts[i]}],
                              args.max_tokens, 0.0, 1.0, args.timeout)
                results[i] = {"ok": True, "latency_s": round(res["latency_s"], 2),
                              "content": (res.get("content_raw") or ""),
                              "prompt_tokens": res["prompt_tokens"],
                              "completion_tokens": res["completion_tokens"],
                              "finish_reason": res["finish_reason"]}
            except Exception as e:
                results[i] = {"ok": False, "error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - t0, 2)}

        t0 = time.time()
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = round(time.time() - t0, 2)

        gpu = gpu_used_mib()
        if gpu:
            gpu_peak = max(gpu_peak, max(gpu))

        mine = [results[i]["content"] if results[i] else "" for i in range(2)]
        expect = [w for _c, w in FACTS]
        hit_own = [expect[i] in mine[i] for i in range(2)]
        hit_other = [expect[1 - i] in mine[i] for i in range(2)]
        cross = any(hit_other)
        if not all(x["ok"] for x in results):
            kind = "error"
        elif all(hit_own) and not cross:
            kind = "both_ok"
        elif cross:
            kind = "cross_talk"
        else:
            kind = "partial"
        stats["rounds"] += 1
        stats[kind] = stats.get(kind, 0) + 1

        rec = {"ts": rd.iso(), "round": r, "kind": kind, "wall_s": wall, "gpu_mib": gpu,
               "hit_own": hit_own, "hit_other": hit_other,
               "results": [{k: (v if k != "content" else v[:120]) for k, v in (x or {}).items()} for x in results]}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"[{r+1}/{args.rounds}] {kind:<9} wall={wall:6.1f}s "
              f"lat={[x['latency_s'] if x else None for x in results]} gpu_peak={gpu_peak}", flush=True)
        if kind in ("cross_talk", "error"):
            print("   ⛔ 异常样本:", json.dumps(rec, ensure_ascii=False)[:500], flush=True)

    fh.close()
    print(f"\n=== 汇总: {stats}   GPU 峰值 {gpu_peak} MiB")
    return 0 if stats.get("cross_talk", 0) == 0 and stats.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
