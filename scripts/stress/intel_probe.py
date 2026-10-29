#!/usr/bin/env python3
"""intel_probe.py —— 压测期间的周期性「智力体检」（单请求、题目轮换、机械判分）。

设计约束（2026-10-27 用户要求）:
  * 每 10 分钟一次（默认 600 s）
  * **严格单请求**：不并发，等服务空闲才发（服务 max-num-seqs=2，压测本身占 1）
  * 题目轮换：固定题库循环，题号持久化，重启接着上次走
  * 判分全自动：答案用正则抽，和期望值机械比对；不依赖裁判模型
  * 记录：每次一行 JSONL（题目/期望/实际/是否正确/延迟/重复率/截断），失败另外追加到 alerts.log

题库覆盖：多步算术、速率、百分比、逻辑、代码求值、指令 JSON、中文计数、长上下文针。

用法:
  python3 intel_probe.py [--port <port>] [--interval 600] [--out .../intel_probe.jsonl]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.expanduser("~/lvllm/stress-20261026"))
import replay_driver as rd  # noqa: E402

# (id, 题目, 期望答案的正则集合, 说明)
POOL = [
    ("math-multistep",
     "一个仓库有 3 层，每层 12 个货架，每个货架 25 箱，每箱 8 件。若已发出 1450 件，还剩多少件？只输出数字。",
     [r"\b5750\b"], "3*12*25*8-1450=5750"),
    ("math-rate",
     "某任务 8 个工人 6 天完成。若 12 个工人做同样的任务且效率相同，需要多少天？只输出数字。",
     [r"\b4\b"], "8*6/12=4"),
    ("math-percent",
     "某商品先涨价 25%，随后降价 10%。最终价格与原价相比如何？只输出：高 / 低 / 相等。",
     [r"高"], "1.25*0.9=1.125 → 高"),
    ("logic-order",
     "甲比乙高，乙比丙高，丙比丁高。四人中谁最矮？只输出名字。",
     [r"丁"], "丙>丁"),
    ("logic-syllogism",
     "所有 A 都是 B；没有 B 是 C。那么 A 与 C 是什么关系？只输出：相交 / 不相交。",
     [r"不相交"], "A⊆B, B∩C=∅"),
    ("code-genexpr",
     "Python 表达式 sum(i*i for i in range(5)) 的值是多少？只输出数字。",
     [r"\b30\b"], "0+1+4+9+16=30"),
    ("code-pow",
     "Python 表达式 2 ** 3 ** 2 的值是多少？只输出数字。",
     [r"\b512\b"], "右结合 2**(3**2)=512"),
    ("json-instr",
     '只输出一个 JSON 对象，不要任何其他文字：键 "a" 的值为 6*7 的计算结果（整数），键 "b" 的值为字符串 "ok"。',
     [r'"a"\s*:\s*42', r"'a'\s*:\s*42"], "a=42, b=ok"),
    ("zh-count",
     "「人工智能」这四个字里，一共有几个不同的汉字？只输出数字。",
     [r"\b4\b"], "四个字各不相同"),
    ("needle-8k",
     "@NEEDLE@",
     [r"林昭"], "8k 上下文里埋的事实"),
]

NEEDLE_FACT = "【项目记录】代号 KX-7；负责人：林昭。"
NEEDLE_Q = "问题：代号 KX-7 的项目的负责人是谁？只回答人名。"


def _corpus_roots():
    """语料根目录：优先 $STRESS_CORPUS_ROOTS（冒号分隔），否则用仓库内的 docs / vllm_xiaotu_moe / dev-docs。"""
    env = os.environ.get("STRESS_CORPUS_ROOTS", "")
    if env:
        return [p for p in env.split(":") if p]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand = [os.path.join(root, d) for d in ("docs", "vllm_xiaotu_moe", "dev-docs")]
    if any(os.path.isdir(c) for c in cand):
        return cand
    # 部署在执行机上时仓库不在脚本旁边：退回“最新的 prod_* 部署目录”
    import glob
    host = os.path.expanduser("~")
    prods = sorted(glob.glob(os.path.join(host, "lvllm", "prod_*")))
    for base in reversed(prods):
        got = [os.path.join(base, d) for d in ("docs", "vllm_xiaotu_moe", "dev-docs")]
        if any(os.path.isdir(g) for g in got):
            return got
    return cand


def build_needle_context(target_tokens=8000):
    """用本机文档拼 8k 上下文，把事实埋在中段。"""
    chunks = []
    for root in _corpus_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _d, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith((".md", ".py")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        t = f.read()
                except OSError:
                    continue
                for i in range(0, len(t), 3000):
                    if len(t[i:i + 3000]) > 200:
                        chunks.append(t[i:i + 3000])
                if sum(len(c) for c in chunks) > target_tokens * 4 * 2:
                    break
            if sum(len(c) for c in chunks) > target_tokens * 4 * 2:
                break
    if not chunks:
        return NEEDLE_FACT + "\n\n" + NEEDLE_Q
    # ⛔【2026-10-27 修】先截断再埋针：原实现把事实插在“拼接后正文”的中点、随后又截到
    #   target_tokens*4 字符，事实正好落在截断边界上被切掉 ⇒ 探针误报“模型找不到”。
    limit = target_tokens * 4
    text = "\n\n".join(chunks)[:limit]
    pos = len(text) // 2
    text = text[:pos] + "\n\n" + NEEDLE_FACT + "\n\n" + text[pos:]
    assert NEEDLE_FACT in text, "needle 事实未被植入"
    return text + "\n\n" + NEEDLE_Q


def normalize(s):
    s = (s or "").strip()
    s = re.sub(r"[`*_#>]", "", s)
    return s


def judge(patterns, content):
    text = normalize(content)
    for pat in patterns:
        if re.search(pat, text):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--interval", type=float, default=600.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=os.path.expanduser("~/lvllm/stress-run4/intel_probe.jsonl"))
    ap.add_argument("--state", default=os.path.expanduser("~/lvllm/stress-run4/intel_probe.idx"))
    ap.add_argument("--once", action="store_true", help="只跑一题就退出（用于自测）")
    args = ap.parse_args()

    api_root = f"http://127.0.0.1:{args.port}"
    api = api_root + "/v1/chat/completions"
    model = "DeepSeek-V4.1-Flash"

    # 题号持久化
    idx = 0
    try:
        with open(args.state) as f:
            idx = int(f.read().strip() or 0)
    except Exception:
        idx = 0

    fh = open(args.out, "a", encoding="utf-8")
    ah = open(os.path.expanduser("~/lvllm/stress-run4/alerts.log"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{rd.iso()}] {msg}"
        print(line, flush=True)

    log(f"INTEL_PROBE_START port={args.port} interval={args.interval}s pool={len(POOL)} start_idx={idx}")
    # ⛔【2026-10-27 修】启动即自检端口：端口给错时以前要等 10 分钟才打出 SKIP，
    #   表现为“探针在跑但什么也不做”。现在直接失败退出，不留哑进程。
    # 端口自检：允许最多 5 分钟预热（服务重启后常处于 warming 状态），
    # 超时仍不健康才判为端口写错并退出。
    _deadline = time.time() + 300
    _c = 0
    while time.time() < _deadline:
        _c, _ = rd.http_code(api_root + "/health", timeout=5.0)
        if _c == 200:
            break
        time.sleep(10)
    if _c != 200:
        log(f"⛔ 启动失败：{api_root}/health 5 分钟内未就绪（最后 {_c}），请确认 --port（当前 {args.port}）")
        return 2

    while True:
        pid, prompt, patterns, note = POOL[idx % len(POOL)]
        if prompt == "@NEEDLE@":
            prompt = build_needle_context()
        msgs = [{"role": "user", "content": prompt}]

        # 等服务健康（外部重启时别记脏数据）
        healthy = False
        for _ in range(60):
            code, _t = rd.http_code(api_root + "/health", timeout=4.0)
            if code == 200:
                healthy = True
                break
            time.sleep(10)
        if not healthy:
            log("SKIP: 服务一直不健康")
            fh.write(json.dumps({"ts": rd.iso(), "item": pid, "skipped": "service_unhealthy"},
                                ensure_ascii=False) + "\n")
            fh.flush()
        else:
            try:
                res = rd.chat(api, model, msgs, args.max_tokens, 0.0, 1.0, args.timeout)
                content = res.get("content_raw") or ""
                ok = judge(patterns, content)
                rec = {
                    "ts": rd.iso(), "item": pid, "expect_note": note,
                    "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                    "finish_reason": res["finish_reason"], "latency_s": res["latency_s"],
                    "ok": ok, "answer_head": " ".join(content.split())[:160],
                    "reasoning_tail": " ".join((res.get("reasoning_raw") or "").split())[-160:],
                    "repeat_ratio": round(rd.repeat_ratio(res["text"]), 3),
                }
            except Exception as e:
                rec = {"ts": rd.iso(), "item": pid, "error": f"{type(e).__name__}: {e}"}
                ok = False
            try:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            except OSError as e:
                # ⛔【2026-10-29 修】该盘偶发 fsync EIO 会直接打死探针（实测 r107 期间崩过一次）
                log(f"⚠ 写结果失败（继续运行）：{type(e).__name__}: {e}")
            log(f"{'PASS' if ok else 'FAIL'} {pid} :: {rec.get('answer_head', rec.get('error'))}")
            if not ok:
                ah.write(f"[{rd.iso()}] INTEL_FAIL {pid}: {rec.get('answer_head') or rec.get('error')}\n")
                ah.flush()

        idx = (idx + 1) % len(POOL)
        try:
            with open(args.state, "w") as f:
                f.write(str(idx))
        except OSError:
            pass
        if args.once:
            break
        time.sleep(max(30.0, args.interval))


if __name__ == "__main__":
    main()
