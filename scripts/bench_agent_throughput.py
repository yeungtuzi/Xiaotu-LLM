#!/usr/bin/env python3
"""⭐ agent 编程场景的标准吞吐基准 **v2**(行业口径 + 与 README 可比)。

v1 的三个口径缺陷(2026-10-15 用户指出"比 README 差一些"后定位):
  ① **波次同步**:v1 按 C 个一组提交、等全组完成才发下一组 ⇒ 组内最慢者空转其余槽位,
     且每条都串了自己那段冷预填 ⇒ 测出来的是"波次 + 冷预填"的端到端,**不是稳态吞吐**。
     ⇒ v2 改为 **满 C 连续灌入**(谁完成就补谁)= `vllm bench serve` 的模型 ✓
  ② **分母混入预填**:`Σtok ÷ Σ墙钟` 含每条长 prompt 的冷预填 ⇒ 与 README 的"每请求解码段"
     不可比。⇒ v2 **同时给两个数**:
       * `聚合输出吞吐`(含预填 = vLLM `Output token throughput`,行业口径)
       * `聚合解码吞吐`(仅解码段 `Σ(token−1) ÷ Σ(t_last − t_first)`)+ **×C 外推**
         —— 后者与 README 的 decode 列**同口径**,可直接对比 ✓
  ③ **缺 TPOT/TTFT**:⇒ v2 走流式,记录每条的 `ttft` / `t_last` / `chunks`,
     报 **TPOT 中位**、**每请求解码速率中位**、**TTFT 中位**。
     ⚠️ TPOT/步时是**轨迹无关**判据:不受"两臂因数值分岔而生成不同文本 ⇒ AL 不同"的干扰
     (见 `docs/BENCH_PROTOCOL.md` §1、`docs/EXPERIMENTS.md` B416)。

用法:
  PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
  $PY scripts/bench_agent_throughput.py --conc 4 --limit 16 --outlen 1024 --tag pin0 --jsonl /tmp/x.jsonl
  $PY scripts/bench_agent_throughput.py --port 8070 --model GLM-5.3-Flash --tag glm53 …
数据集:`process_data/data/dsh_assembled_bench.jsonl`(真实 DSH 载荷,预装配 `<|im_start|>`)
"""
from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
import urllib.request

DS = "/home/user/lvllm/process_data/data/dsh_assembled_bench.jsonl"


def deg(t, n=8):
    if len(t) < n * 2:
        return 0.0
    seen, dup, tot = set(), 0, 0
    for i in range(len(t) - n + 1):
        g = t[i:i + n]
        tot += 1
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / tot


def one(prompt, outlen, port, model, sink, tag):
    """流式发一条:记 ttft / 末 token 时刻 / chunks(≈解码步数)/ 全文。"""
    # ⭐⭐【2026-10-17 用户裁定 · QA #264】**不得强制输出**:`max_tokens` 仅作上限,模型自然结束 ✓
    #   废止旧口径 `"min_tokens": outlen` + `"ignore_eos": True` ✗
    #   ⛔ 旧口径的副作用(协议 §2.7/§4.2 自述):强制长度把生成推进"复制循环"体制
    #      ⇒ **AL 虚高、tok/s 假快**、`deg` 告警 ⇒ **我们自己的数字被自己的口径污染** ✗
    #   ⚠️ 连带重构(缺一即不可读):
    #     ① 长度不再固定 ⇒ `tok/s` **失去可比性** ⇒ ⭐ **主判据改为"步时"**(协议 §4.1 ✓)
    #     ② **必须同报**【实际输出长度中位】与【被上限截断的比例】⇒ 否则等于"偷偷变短" ✗
    body = {"model": model, "prompt": prompt,
            "max_tokens": outlen, "ignore_eos": False,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % port,
                                data=json.dumps(body).encode(),
                                headers={"content-type": "application/json"})
    t0 = time.time()
    first = last = None
    texts, ntok, nchunk, ptok = [], None, 0, None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                if not raw or not raw.startswith(b"data: "):
                    continue
                p = raw[6:].strip()
                if p == b"[DONE]":
                    break
                try:
                    d = json.loads(p)
                except Exception:  # noqa: BLE001
                    continue
                ch = (d.get("choices") or [{}])[0].get("text")
                if ch:
                    now = time.time()
                    if first is None:
                        first = now - t0
                    last = now - t0
                    nchunk += 1
                    texts.append(ch)
                u = d.get("usage") or {}
                if u.get("completion_tokens"):
                    ntok = int(u["completion_tokens"])
                    ptok = u.get("prompt_tokens")
    except Exception as e:  # noqa: BLE001
        sink[tag] = {"err": repr(e)}
        return
    # ⭐ 步时(协议 §4.1 指定:本口径里**最接近"引擎速度"**的量,因为它不含 AL)= 解码段耗时 ÷ 步数
    #   ⚠️ 步数以 SSE delta 数近似;`chunks` 含首块与 usage 尾块 ⇒ `chunks-1` 并做下限保护 ✓
    _nt = int(ntok) if ntok else nchunk
    _step = ((last - first) / (nchunk - 1) * 1000
             if (first and last and nchunk > 1) else None)
    sink[tag] = {"tok": _nt,
                 "n_src": "usage" if ntok else "chunks",
                 "ptok": ptok, "ttft": first, "tlast": last,
                 "dt": time.time() - t0, "chunks": nchunk,
                 "outlen": _nt, "cap": int(outlen),
                 "capped": int(_nt >= int(outlen)),
                 "step_ms": _step,
                 "deg": deg("".join(texts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--outlen", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="取多少条载荷(0=全部)")
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--jsonl", default="", help="把每条原始记录写到该文件(供事后分层)")
    a = ap.parse_args()
    recs = [json.loads(ln) for ln in open(a.dataset, encoding="utf-8") if ln.strip()]
    if a.limit:
        recs = recs[:a.limit]
    tasks = [(rd, i, r["prompt"]) for rd in range(a.rounds) for i, r in enumerate(recs)]

    q: queue.Queue = queue.Queue()
    for t in tasks:
        q.put(t)
    sink, lock = {}, threading.Lock()
    t_start, t_end = [None], [0.0]

    def worker():
        while True:
            try:
                rd, i, pr = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                if t_start[0] is None:
                    t_start[0] = time.time()
            one(pr, a.outlen, a.port, a.model, sink, "r%d_%d" % (rd, i))
            with lock:
                t_end[0] = time.time()

    print("并发=%d(**满 C 连续灌入**)载荷=%d 条 × %d 轮 ⭐**自然结束**、上限=%d token %s"
          % (a.conc, len(recs), a.rounds, a.outlen, ("[%s]" % a.tag) if a.tag else ""), flush=True)
    ths = [threading.Thread(target=worker) for _ in range(a.conc)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()

    good = [v for v in sink.values() if not v.get("err")]
    if not good:
        print("无有效样本"); return
    wall = t_end[0] - t_start[0]
    stok = sum(v["tok"] for v in good)
    dec = [v for v in good if v.get("ttft") and v.get("tlast")
           and v["tlast"] > v["ttft"] and v["tok"] > 1]
    sdur = sum(v["tlast"] - v["ttft"] for v in dec)
    stokd = sum(v["tok"] - 1 for v in dec)
    per_req = sorted((v["tok"] - 1) / (v["tlast"] - v["ttft"]) for v in dec)
    tpot = sorted((v["tlast"] - v["ttft"]) / (v["tok"] - 1) * 1000 for v in dec)
    ttfts = sorted(v["ttft"] * 1000 for v in good if v.get("ttft"))
    degs = [v["deg"] for v in good]
    # ⭐ 新口径必报三列(QA #264):步时 / 实际输出长度 / 被上限截断率
    stepms = sorted(v["step_ms"] for v in good if v.get("step_ms"))
    outs = sorted(v.get("outlen", v["tok"]) for v in good)
    ncap = sum(1 for v in good if v.get("capped"))
    ptok = sum(v["ptok"] or 0 for v in good)
    if a.jsonl:
        with open(a.jsonl, "w") as f:
            for k, v in sink.items():
                f.write(json.dumps(dict(v, tag=k)) + "\n")
    print()
    print("===== 标准口径 v2(满 C 连续灌入)=====")
    print("① 聚合输出吞吐(含预填 = vLLM `Output token throughput`)= %d / %.1fs = **%.1f tok/s**"
          % (stok, wall, stok / wall))
    if sdur > 0:
        print("② 聚合解码吞吐(仅解码段,与 README 的 decode 列同口径)= %d / %.1fs = **%.1f tok/s**"
              " ⇒ **×C 外推 = %.1f**" % (stokd, sdur, stokd / sdur, stokd / sdur * a.conc))
    print("③ 每请求解码速率中位 = %.1f tok/s ;**TPOT 中位 = %.1f ms/token** ;TTFT 中位 = %.1f ms"
          % (statistics.median(per_req) if per_req else float("nan"),
             statistics.median(tpot) if tpot else float("nan"),
             statistics.median(ttfts) if ttfts else float("nan")))
    print("③b ⭐**步时中位 = %.1f ms/步**(⭐ 主判据,不含 AL)⇒ 等效解码步率 = %.1f 步/s"
          % (statistics.median(stepms) if stepms else float("nan"),
             1000.0 / statistics.median(stepms) if stepms else float("nan")))
    print("③c ⭐**实际输出长度**:中位=%.0f token(上限=%d);⭐ **被上限截断 = %d/%d = %.0f%%**"
          " ;最短=%d 最长=%d"
          % (statistics.median(outs) if outs else float("nan"),
             int(good[0].get("cap", 0)) if good else 0,
             ncap, len(good), 100.0 * ncap / len(good) if good else float("nan"),
             outs[0] if outs else 0, outs[-1] if outs else 0))
    print("④ 每请求端到端耗时中位 = %.1fs ;输入 token 合计=%d ⇒ 总吞吐(含输入)= %.1f tok/s"
          % (statistics.median(v["dt"] for v in good), ptok,
             (ptok + stok) / wall if ptok else float("nan")))
    print("⑤ 健康列:deg 中位=%.2f ;deg>0.30 占比=%.0f%% ;解码段中位=%.1fs ;步数中位=%.0f ;失败=%d"
          " (token 口径:usage=%d/chunks=%d)"
          % (statistics.median(degs), 100 * sum(1 for d in degs if d > 0.30) / len(degs),
             statistics.median((v["tlast"] - v["ttft"]) for v in dec) if dec else float("nan"),
             statistics.median(v["chunks"] for v in good),
             len(sink) - len(good),
             sum(1 for v in good if v["n_src"] == "usage"),
             sum(1 for v in good if v["n_src"] == "chunks")))


if __name__ == "__main__":
    main()
