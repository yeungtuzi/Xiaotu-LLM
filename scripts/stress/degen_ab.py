#!/usr/bin/env python3
"""degen_ab.py —— 区分「长上下文退化」与「max_tokens 不够被截断」的对照实验。

背景（2026-10-27）:
    回放统计里，'只有思考 + finish_reason=length' 在短上下文也占 28%，长上下文升到 64%。
    但 driver 固定 max_tokens=1024，思考型模型很容易把 1024 全花在思考上 ⇒ 存在混淆。
    本脚本在同一段上下文上分别用不同 max_tokens 发请求，看"答不出正文"的比例是否随
    ①上下文长度 ②max_tokens 变化。

设计:
    4 个上下文档位（50k/100k/150k/200k 估算 token）× 2 个 max_tokens（1024/4096）
    每格重复 N 次（默认 5）⇒ 40 个请求，串行，避免与压测抢并发。
    上下文从真实会话转录拼出来（同 driver 的 parse_session），末尾接一个固定的总结问题。

用法:
    python3 degen_ab.py [--port <port>] [--reps 5] [--out ~/lvllm/stress-run4/degen_ab.jsonl]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/lvllm/stress-20261026"))
import replay_driver as rd  # noqa: E402

BASE = os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
BUCKETS = [50_000, 100_000, 150_000, 200_000]
MAXTOK = [1024, 4096]
QUESTION = "用三句话总结上面这段对话要解决的核心问题，以及当前卡在哪里。"


def wait_healthy(api_root, limit_s=900):
    """服务被外部重启时等它就绪（最多 limit_s 秒），避免把"连接被拒"记成实验数据。"""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        code, _ = rd.http_code(api_root + "/health", timeout=4.0)
        if code == 200:
            return True
        time.sleep(10)
    return False


def est(msgs):
    return sum(len(m.get("content") or "") for m in msgs) // 4


def build_pool():
    sessions = rd.find_sessions(BASE, 3)
    pool = []
    for _total, _name, path in sessions:   # find_sessions 返回 (total, name, path)
        try:
            system, turns = rd.parse_session(path, tool_clip=2000)
        except Exception:
            continue
        for t in turns:
            u, a = (t.get("user") or "").strip(), (t.get("assistant") or "").strip()
            if u:
                pool.append(("user", u))
            if a:
                pool.append(("assistant", a))
    return pool


def make_ctx(pool, target):
    """从池子里顺序取内容，拼到 target 估算 token 为止（user/assistant 交替）。"""
    msgs = []
    for i, (role, text) in enumerate(pool):
        want = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": want, "content": rd.clip(text, 20000)})
        if est(msgs) >= target:
            break
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=os.path.expanduser("~/lvllm/stress-run4/degen_ab.jsonl"))
    args = ap.parse_args()

    api_root = f"http://127.0.0.1:{args.port}"
    api = api_root + "/v1/chat/completions"
    model = "DeepSeek-V4.1-Flash"
    pool = build_pool()
    if not pool:
        print("no transcript pool", file=sys.stderr)
        return 2
    print(f"pool={len(pool)} 段；计划 {len(BUCKETS)}×{len(MAXTOK)}×{args.reps} 个请求")

    fh = open(args.out, "a", encoding="utf-8")
    for target in BUCKETS:
        base = make_ctx(pool, target)
        for mt in MAXTOK:
            for r in range(args.reps):
                msgs = base + [{"role": "user", "content": QUESTION}]
                res = None
                err = None
                for attempt in range(4):
                    if not wait_healthy(api_root):
                        err = "health timeout"
                        break
                    try:
                        res = rd.chat(api, model, msgs, mt, 0.0, 1.0, args.timeout)
                        break
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        print(f"  {target} mt={mt} rep={r} attempt={attempt} {err} → 等待恢复")
                        time.sleep(15)
                if res is None:
                    rec = {"ts": rd.iso(), "target": target, "max_tokens": mt, "rep": r, "error": err}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"  {target} mt={mt} rep={r} GIVEUP {err}")
                    continue
                text = res["text"]
                rec = {
                    "ts": rd.iso(), "target": target, "max_tokens": mt, "rep": r,
                    "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                    "finish_reason": res["finish_reason"],
                    "reasoning_only": res["reasoning_only"],
                    "answered": bool((res.get("content_raw") or "").strip()),
                    "repeat_ratio": round(rd.repeat_ratio(text), 3),
                    "latency_s": res["latency_s"],
                    "content_head": " ".join((res.get("content_raw") or "").split())[:200],
                    "reasoning_tail": " ".join((res.get("reasoning_raw") or "").split())[-200:],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                print(f"  target={target} mt={mt} rep={r} prompt={res['prompt_tokens']} "
                      f"comp={res['completion_tokens']} fin={res['finish_reason']} "
                      f"answered={rec['answered']} rep_ratio={rec['repeat_ratio']} {res['latency_s']}s")
    fh.close()

    # 汇总
    rows = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip().startswith("{")]
    print("\n=== 汇总（answered = 是否写出了正文）")
    print(f"{'target':>8} {'mt':>5} {'n':>3} {'answered':>9} {'仅思考撞限':>10} {'平均延迟':>9}")
    for target in BUCKETS:
        for mt in MAXTOK:
            sel = [x for x in rows if x.get("target") == target and x.get("max_tokens") == mt and "error" not in x]
            if not sel:
                continue
            answered = sum(1 for x in sel if x.get("answered"))
            trunc = sum(1 for x in sel if x.get("reasoning_only") and x.get("finish_reason") == "length")
            lat = sum(x.get("latency_s") or 0 for x in sel) / len(sel)
            print(f"{target:>8} {mt:>5} {len(sel):>3} {answered:>9} {trunc:>10} {lat:>8.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
