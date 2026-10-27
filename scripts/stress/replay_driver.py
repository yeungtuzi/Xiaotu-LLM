#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSH 会话回放驱动器（A100 生产服务长稳压力测试用，dry-run）

用途
----
把 A100 上已有的 DSH 会话记录（~/.dsh/sessions/<bucket>/session-*/session.*.jsonl.zstd）
解析成"用户轮次"，按**真实对话方式**逐轮回放：发一段 prompt → 等 API 应答 → 再发下一段。
**不执行任何工具**：请求体里不带 tools 定义，也没有任何文件/进程改动；工具输出只是把
会话记录里已有的文本（截断后）放回上下文，用来让上下文长度接近真实开发会话。

三种回放模式
------------
  transcript （默认）  上下文 = 记录里的 user 文本 + 记录里的 assistant 文本/工具输出（截断）
                       + 本次 API 的真实应答。最接近真实会话的上下文规模。
  chat                 上下文 = 只累积 user 文本 + 本次 API 的真实应答（最干净，上下文增长慢）。
  chained              一个会话回放结束后，接着回放下一个会话（上下文不重置，直到超过上限）。

安全边界
--------
* 只发 HTTP 请求到 127.0.0.1:<port>/v1/chat/completions 与 /v1/models、/health；
* 从不调用 /bin/sh、不写被测机器的任何文件（本程序的日志除外）；
* 单条请求超时 <timeout> 秒，连续失败到阈值即退出（退出码 3），不会无限重试烧机器。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ⛔ 只用 loopback：显式禁用 proxy 解析（不读 HTTP(S)_PROXY，URL 也只指向 127.0.0.1）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ---------------- 基础工具 ----------------


def now() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts if ts is not None else now()))


def log(msg: str, fh=None) -> None:
    line = f"[{iso()}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def http_json(url: str, payload=None, timeout: float = 60.0, method: str | None = None):
    data = None
    headers = {"accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    with _OPENER.open(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def http_code(url: str, timeout: float = 5.0) -> tuple[int, float]:
    t0 = now()
    try:
        req = urllib.request.Request(url, method="GET")
        with _OPENER.open(req, timeout=timeout) as resp:
            resp.read(64)
            return resp.status, now() - t0
    except urllib.error.HTTPError as e:
        return e.code, now() - t0
    except Exception:
        return 0, now() - t0


# ---------------- 会话解析 ----------------

def iter_events(path: str):
    """流式读取 .jsonl.zstd / .jsonl，逐行 yield dict。"""
    if path.endswith(".zstd"):
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        stream = proc.stdout
    else:
        stream = open(path, "rb")
    try:
        for raw in stream:  # type: ignore[union-attr]
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except Exception:
                continue
    finally:
        try:
            stream.close()  # type: ignore[union-attr]
        except Exception:
            pass
        if path.endswith(".zstd"):
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


def message_text(message: dict) -> str:
    """把 DSH message 的 content[] 拼成纯文本（忽略 reasoning / tool-call）。"""
    content = message.get("content")
    if isinstance(content, str):
        return content
    out = []
    for item in content or []:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            out.append(item["text"])
    return "\n".join(out)


def clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return text[:head] + "\n…[截断 %d 字符]…\n" % (len(text) - limit) + text[-tail:]


def parse_session(path: str, tool_clip: int, max_turns: int = 0):
    """返回 (system_prompt, turns)；turns 每项 {'user','assistant','tools':[str]}。"""
    system = ""
    turns: list[dict] = []
    cur = None
    for ev in iter_events(path):
        t = ev.get("type")
        d = ev.get("data") or {}
        if t == "system/message":
            txt = message_text(d.get("message") or {})
            if txt and len(txt) > len(system):
                system = txt
        elif t == "user/message":
            if (d.get("source") or {}).get("kind") != "user":
                continue  # 跳过审批变更/系统注入等非人类输入
            txt = message_text(d)
            if not txt.strip():
                continue
            cur = {"user": txt, "assistant": "", "tools": []}
            turns.append(cur)
            if max_turns and len(turns) >= max_turns:
                break
        elif t == "assistant/message":
            if cur is None:
                continue
            txt = message_text(d.get("message") or {})
            if txt:
                cur["assistant"] += txt
        elif t == "tool/result":
            if cur is None:
                continue
            txt = message_text(d.get("message") or {})
            if txt:
                cur["tools"].append(clip(txt, tool_clip))
    return system, turns


def find_sessions(root: str, pick: int, prefer_newest: bool = True, exclude: set[str] | None = None):
    """按【会话目录总字节数】挑最大的 N 个；每个会话取最新的可解析快照（优先 v4）。"""
    exclude = exclude or set()
    entries = []
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or name in exclude:
            continue
        files = [f for f in os.listdir(d) if f.startswith("session") and (f.endswith(".jsonl.zstd") or f.endswith(".jsonl"))]
        if not files:
            continue
        total = sum(os.path.getsize(os.path.join(d, f)) for f in files)
        # 版本优先级：v4 > v3 > 其它；同版本取最新
        def rank(f):
            m = re.search(r"session\.v(\d+)\.jsonl", f)
            return (int(m.group(1)) if m else 0, os.path.getmtime(os.path.join(d, f)))
        best = max(files, key=rank if prefer_newest else lambda f: os.path.getsize(os.path.join(d, f)))
        entries.append((total, name, os.path.join(d, best)))
    entries.sort(reverse=True)
    return entries[:pick]


# ---------------- 请求与质检 ----------------

REPEAT_N = 8


def repeat_ratio(text: str) -> float:
    toks = re.findall(r"\S+", text)
    if len(toks) < REPEAT_N * 2:
        return 0.0
    grams = [tuple(toks[i:i + REPEAT_N]) for i in range(len(toks) - REPEAT_N + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for ch in text if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\t\r"))
    return bad / len(text)


PROBES = [
    ("只输出数字：37 * 19 = ?", "703"),
    ("把字符串 abcdef 反转，只输出结果。", "fedcba"),
    ("北京是中国的首都吗？只输出 YES 或 NO。", "YES"),
    ("字符串 mississippi 里有几个字母 s？只输出数字。", "4"),
    ("1 到 20 之间有多少个质数？只输出数字。", "8"),
]


def run_probe(api: str, model: str, timeout: float, temperature: float = 0.0):
    """固定题目 + 全新上下文，用来量化"智力退化"。返回 (score, details)。"""
    hits = 0
    details = []
    for q, expect in PROBES:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 24,
            "temperature": temperature,
        }
        try:
            _, body = http_json(api, payload, timeout=min(timeout, 120))
            text = (body["choices"][0]["message"].get("content") or "").strip()
            ok = expect.lower() in text.lower()
            hits += 1 if ok else 0
            details.append({"q": q, "expect": expect, "got": text[:80], "ok": ok})
        except Exception as e:
            details.append({"q": q, "expect": expect, "error": str(e)[:120], "ok": False})
    return hits / len(PROBES), details


def chat(api: str, model: str, messages, max_tokens: int, temperature: float, top_p: float,
         timeout: float, extra: dict | None = None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    t0 = now()
    _, body = http_json(api, payload, timeout=timeout)
    dt = now() - t0
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    text = msg.get("content") or ""
    reasoning_only = not text.strip() and bool(reasoning.strip())
    if reasoning_only:
        text = reasoning  # 只有思考时按"有内容"处理，避免误报空应答
    usage = body.get("usage") or {}
    return {
        "text": text,
        "content_raw": msg.get("content") or "",
        "reasoning_raw": reasoning,
        "reasoning_only": reasoning_only,
        "reasoning_chars": len(reasoning),
        "finish_reason": choice.get("finish_reason"),
        "latency_s": round(dt, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


# ---------------- 主流程 ----------------

def main() -> int:
    ap = argparse.ArgumentParser(description="DSH 会话回放压力驱动器（dry-run，不执行工具）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8070, help="被测 vLLM 端口（实际端口见 dev-docs/ENV_TWO_MACHINES.md）")
    ap.add_argument("--model", default="", help="留空则取 /v1/models 的第一个")
    ap.add_argument("--sessions-root", default=os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--"))
    ap.add_argument("--pick", type=int, default=3, help="选最大的几个会话")
    ap.add_argument("--mode", choices=["transcript", "chat", "chained"], default="transcript")
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--max-ctx-tokens", type=int, default=200000, help="上下文估算上限，超过则重开会话")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--tool-clip", type=int, default=6000, help="单条工具输出保留字符数（0=不截断）")
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--probe-every", type=int, default=20, help="每 N 次请求跑一次退化探针（0=关闭）")
    ap.add_argument("--max-consecutive-failures", type=int, default=5, help="连续失败多少次后进入“中断等待”")
    ap.add_argument("--max-outage-min", type=float, default=30.0, help="允许服务中断多久后放弃（分钟）")
    ap.add_argument("--max-requests", type=int, default=0, help="最多发多少个请求（0=不限；冒烟用）")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--describe-only", action="store_true", help="只解析并打印会话统计，不发请求")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.expanduser("~"), "lvllm", "stress-20261026")
    os.makedirs(out_dir, exist_ok=True)
    api = f"http://{args.host}:{args.port}/v1/chat/completions"
    base = f"http://{args.host}:{args.port}"

    # ---- 预检 ----
    code, latency = http_code(f"{base}/health", timeout=8)
    log(f"preflight /health -> {code} ({latency:.2f}s)", None)
    if code != 200 and not args.describe_only:
        log("⛔ /health 非 200，拒绝启动。请确认端口（实际端口见 dev-docs/ENV_TWO_MACHINES.md）", None)
        return 2

    model = args.model
    if not model:
        try:
            _, models = http_json(f"{base}/v1/models", timeout=10)
            model = models["data"][0]["id"]
        except Exception as e:
            log(f"⛔ 取模型名失败: {e}", None)
            return 2
    log(f"model = {model}", None)

    # ---- 选会话 ----
    picks = find_sessions(args.sessions_root, args.pick)
    if not picks:
        log(f"⛔ {args.sessions_root} 下没有可解析的会话", None)
        return 2
    sessions = []
    for total, name, path in picks:
        t0 = now()
        system, turns = parse_session(path, args.tool_clip)
        n_users = len(turns)
        prompt_chars = sum(len(t["user"]) for t in turns)
        log(f"session {name}: file={os.path.basename(path)} {total/1e6:.1f}MB "
            f"turns={n_users} user_chars={prompt_chars} system_chars={len(system)} parse={now()-t0:.1f}s", None)
        if turns:
            sessions.append({"name": name, "path": path, "system": system, "turns": turns})

    if args.describe_only:
        for s in sessions:
            est = sum(len(t["user"]) + len(t["assistant"]) + sum(len(x) for x in t["tools"]) for t in s["turns"]) // 4
            log(f"  {s['name']}: 估算总文本 ~{est} tokens（截断后）", None)
        return 0

    if not sessions:
        log("⛔ 解析后没有有效会话", None)
        return 2

    # ---- 回放 ----
    log_path = os.path.join(out_dir, "replay.jsonl")
    fh_log = open(log_path, "a", encoding="utf-8")
    failure_streak = 0
    outage_since = None
    deadline = now() + args.hours * 3600
    idx = 0
    req_no = 0
    probe_no = 0
    stats = {"requests": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0, "probes": 0, "probe_hits": 0}
    cur = None  # 当前会话上下文

    def start_session(s):
        sys_msg = clip(s["system"], 8000) if s["system"] else "You are a helpful coding assistant."
        return {"name": s["name"], "msgs": [{"role": "system", "content": sys_msg}],
                "turn": 0, "turns": s["turns"], "est": len(sys_msg) // 4}

    def est_tokens(msgs):
        return sum(len(m.get("content") or "") for m in msgs) // 4

    while now() < deadline:
        if args.max_requests and stats["requests"] >= args.max_requests:
            log(f"=== 已达 --max-requests={args.max_requests}，停止", fh_log)
            break
        if cur is None:
            cur = start_session(sessions[idx % len(sessions)])
            idx += 1
            log(f"=== 开始回放 {cur['name']}（共 {len(cur['turns'])} 轮）", fh_log)

        if cur["turn"] >= len(cur["turns"]):
            log(f"=== 回放完毕 {cur['name']}，切换下一个", fh_log)
            cur = None
            continue

        turn = cur["turns"][cur["turn"]]
        if args.mode != "chained" and est_tokens(cur["msgs"]) > args.max_ctx_tokens:
            log(f"=== 上下文估算 {est_tokens(cur['msgs'])} > {args.max_ctx_tokens}，重开会话", fh_log)
            cur = None
            continue

        cur["msgs"].append({"role": "user", "content": turn["user"]})
        try:
            res = chat(api, model, cur["msgs"], args.max_tokens, args.temperature, args.top_p,
                       args.request_timeout)
            failure_streak = 0
            if outage_since is not None:
                log(f"✓ 服务恢复，本次中断 {(now() - outage_since) / 60:.1f} 分钟", fh_log)
                fh_log.write(json.dumps({"ts": iso(), "event": "OUTAGE_END",
                                         "minutes": round((now() - outage_since) / 60, 1)}, ensure_ascii=False) + "\n")
                fh_log.flush()
                outage_since = None
        except Exception as e:
            failure_streak += 1
            stats["errors"] += 1
            cur["msgs"].pop()
            if outage_since is None:
                outage_since = now()
                log(f"‼ 请求失败，进入中断等待：{type(e).__name__}: {e}"
                    f"（最多等 {args.max_outage_min:.0f} 分钟）", fh_log)
                fh_log.write(json.dumps({"ts": iso(), "event": "OUTAGE_START",
                                         "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
                fh_log.flush()
            waited = (now() - outage_since) / 60
            log(f"‼ 连续失败 {failure_streak} 次（中断 {waited:.1f}/{args.max_outage_min:.0f} min）："
                f"{type(e).__name__}: {e}", fh_log)
            if waited >= args.max_outage_min:
                log(f"⛔ 服务中断超过 {args.max_outage_min:.0f} 分钟，放弃（退出码 3）", fh_log)
                fh_log.close()
                return 3
            time.sleep(min(60, 10 + 5 * failure_streak))
            continue

        text = res["text"]
        cur["msgs"].append({"role": "assistant", "content": text})
        req_no += 1
        stats["requests"] += 1
        stats["tokens_in"] += res["prompt_tokens"] or 0
        stats["tokens_out"] += res["completion_tokens"] or 0

        # 把这次回放"本应产生"的记录型上下文补回去（transcript 模式）
        if args.mode in ("transcript", "chained"):
            if turn["assistant"]:
                cur["msgs"].append({"role": "assistant", "content": clip(turn["assistant"], args.tool_clip)})
            for t in turn["tools"]:
                cur["msgs"].append({"role": "user", "content": "[记录中的工具输出]\n" + t})

        rec = {
            "ts": iso(), "epoch": round(now(), 3), "session": cur["name"], "turn": cur["turn"] + 1,
            "mode": args.mode, "latency_s": res["latency_s"], "finish_reason": res["finish_reason"],
            "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
            "ctx_est_tokens": est_tokens(cur["msgs"]),
            "out_chars": len(text), "reasoning_chars": res.get("reasoning_chars", 0),
            "reasoning_only": res.get("reasoning_only", False), "repeat_ratio": round(repeat_ratio(text), 3),
            "garbage_ratio": round(garbage_ratio(text), 5),
        }
        # 【2026-10-27】退化取证：只对可疑应答（高重复 / 有乱码 / 只有思考）存正文与思考的头部摘要，
        # 避免整份日志膨胀；正常应答不存，隐私与体积都可控。
        _rr = rec["repeat_ratio"]
        _gr = rec["garbage_ratio"]
        if _rr > 0.6 or _gr > 0.0 or rec["reasoning_only"]:
            def _head(t: str, n: int = 500) -> str:
                t = " ".join((t or "").split())
                return t[:n]
            rec["content_head"] = _head(res.get("content_raw", ""))
            rec["reasoning_head"] = _head(res.get("reasoning_raw", ""))
            rec["reasoning_tail"] = " ".join((res.get("reasoning_raw") or "").split())[-300:]
        fh_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh_log.flush()
        if req_no % 20 == 0:
            os.fsync(fh_log.fileno())

        # 退化现象即时告警
        if not text.strip():
            log(f"⚠ 空应答（{cur['name']} turn {cur['turn']+1}）", fh_log)
        elif rec["repeat_ratio"] > 0.6:
            log(f"⚠ 疑似重复退化 repeat_ratio={rec['repeat_ratio']}（{cur['name']} turn {cur['turn']+1}）", fh_log)

        cur["turn"] += 1

        # 周期性探针
        if args.probe_every and req_no % args.probe_every == 0:
            score, details = run_probe(f"{base}/v1/chat/completions", model, args.request_timeout)
            probe_no += 1
            stats["probes"] += 1
            stats["probe_hits"] += score
            fh_log.write(json.dumps({"ts": iso(), "probe": probe_no, "score": score, "details": details},
                                    ensure_ascii=False) + "\n")
            fh_log.flush()
            log(f"— 探针 #{probe_no}: {score:.0%}", fh_log)

    summary = {
        "finished_at": iso(), "mode": args.mode, "hours": args.hours, "model": model,
        "port": args.port, "sessions": [s["name"] for s in sessions], **stats,
        "probe_accuracy": (stats["probe_hits"] / stats["probes"]) if stats["probes"] else None,
    }
    with open(os.path.join(out_dir, "replay-summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"=== 结束：{json.dumps(summary, ensure_ascii=False)}", None)
    fh_log.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise  # 正常退出/用法错误不算崩溃，不写 crash 文件
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        crash = os.path.join(os.path.expanduser("~"), "lvllm", "stress-20261026", f"crash-driver-{stamp}.log")
        try:
            os.makedirs(os.path.dirname(crash), exist_ok=True)
            with open(crash, "w", encoding="utf-8") as f:
                f.write(tb)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        sys.stderr.write(tb)
        sys.stderr.flush()
        sys.exit(1)
    sys.exit(rc)
