#!/usr/bin/env python3
"""degen_suite.py —— 自建「退化」检测套件（不依赖外网数据集，全部本地生成）。

四个子测试，都按【上下文长度】分档，输出逐条 JSONL + markdown 汇总：

  ladder      长度阶梯同题回归：同一问题塞进不同长度的上下文，看是否还能正常作答
  instruction 指令遗忘：长上下文末尾给硬约束（JSON/前缀/字数/中文），看是否还遵守
  multihop    多跳 + 干扰：埋 3 条互相引用的硬事实，混入结构相似的干扰事实，看是否答对
  consistency 一致性：同一 prompt 温度 0 跑 N 次，看输出是否发散

判分全部是机械规则，不需要裁判模型：
  answered    正文非空
  truncated   reasoning_only 且 finish_reason=length（思考把预算用光，没写出正文）
  repeat/garbage  复用 replay_driver 的重复率/乱码率
  compliant   instruction 子测试：JSON 可解析 + 键齐全 + 前缀/字数/语言约束全满足
  correct     multihop 子测试：期望实体出现在正文中

用法:
  python3 degen_suite.py --suite ladder,instruction,multihop,consistency \
      --lengths 32000,64000,128000,256000 --reps 3 --port <port> \
      --out ~/lvllm/stress-run4/degen_suite.jsonl
"""
import argparse
import difflib
import json
import os
import random
import sys
import time
import re

sys.path.insert(0, os.path.expanduser("~/lvllm/stress-20261026"))
import replay_driver as rd  # noqa: E402

CORPUS_ROOTS = None  # 运行时由 _corpus_roots() 决定

LADDER_Q = "读完上面全部内容后，用五句话总结其中提到的三个最关键的工程约束，并说明它们各自的原因。"

INSTR_TPL = (
    "下面是一段很长的工程资料。请严格按要求回答最后一个问题。\n\n{ctx}\n\n"
    "【硬性输出要求，必须全部满足】\n"
    "1) 只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块；\n"
    "2) JSON 必须恰好包含三个键：\"answer\"（字符串）、\"count\"（整数）、\"lang\"（固定为 \"zh\"）；\n"
    "3) \"answer\" 的值必须以「答：」开头，且总长度不超过 40 个字符；\n"
    "4) 全部使用中文。\n\n"
    "问题：这段资料主要讨论的是什么？"
)

FACTS = [
    ("代号", "KX-7", "负责人", "林昭"),
    ("代号", "KX-7", "上线季度", "2027 年第二季度"),
    ("代号", "MVP-3", "负责人", "周恕"),
]
DISTRACTORS = [
    ("代号", "KX-9", "负责人", "陈铎"),
    ("代号", "KX-7B", "负责人", "许岸"),
    ("代号", "MVP-13", "负责人", "何湛"),
    ("代号", "KX-7", "备用负责人", "苏黎"),
]


def _corpus_roots():
    """语料根目录：优先 $STRESS_CORPUS_ROOTS（冒号分隔），否则用仓库内的 docs / vllm_xiaotu_moe / dev-docs。"""
    env = os.environ.get("STRESS_CORPUS_ROOTS", "")
    if env:
        return [p for p in env.split(":") if p]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [os.path.join(root, d) for d in ("docs", "vllm_xiaotu_moe", "dev-docs")]


def est(text):
    return len(text) // 4


def build_corpus(max_chars=4_000_000):
    chunks = []
    total = 0
    for root in _corpus_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith((".md", ".py", ".txt", ".json", ".html")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > 2_000_000:
                        continue
                    with open(p, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except OSError:
                    continue
                # 按 4000 字符切块
                for i in range(0, len(text), 4000):
                    piece = text[i:i + 4000].strip()
                    if len(piece) > 200:
                        chunks.append(piece)
                        total += len(piece)
                if total >= max_chars:
                    return chunks
    return chunks


def make_ctx(chunks, target_tokens, rng, prefix=""):
    need = target_tokens * 4
    out = []
    size = 0
    pool = chunks[:]
    rng.shuffle(pool)
    i = 0
    while size < need and pool:
        c = pool[i % len(pool)]
        out.append(c)
        size += len(c)
        i += 1
        if i > 20000:
            break
    return prefix + "\n\n".join(out)


def plant_facts(ctx, rng, facts):
    """把事实随机插进上下文（按段落切分后插入），返回新上下文。"""
    paras = ctx.split("\n\n")
    for subj_k, subj_v, attr, val in facts:
        line = f"【项目记录】{subj_k}：{subj_v}；{attr}：{val}。"
        pos = rng.randrange(0, max(1, len(paras)))
        paras.insert(pos, line)
    for subj_k, subj_v, attr, val in DISTRACTORS:
        line = f"【项目记录】{subj_k}：{subj_v}；{attr}：{val}。"
        pos = rng.randrange(0, max(1, len(paras)))
        paras.insert(pos, line)
    return "\n\n".join(paras)


def judge_json(text):
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        d = json.loads(s)
    except Exception:
        return False, "json_parse_failed"
    if not isinstance(d, dict):
        return False, "not_object"
    if set(d) != {"answer", "count", "lang"}:
        return False, f"keys={sorted(d)}"
    if not isinstance(d.get("answer"), str) or not isinstance(d.get("count"), int) or d.get("lang") != "zh":
        return False, "types"
    a = d["answer"]
    if not a.startswith("答："):
        return False, "prefix"
    if len(a) > 40:
        return False, f"len={len(a)}"
    if not re.search(r"[\u4e00-\u9fff]", a):
        return False, "no_chinese"
    return True, "ok"


def call(api_root, api, model, msgs, max_tokens, timeout, log):
    for attempt in range(4):
        code, _ = rd.http_code(api_root + "/health", timeout=4.0)
        if code != 200:
            log(f"    health={code} → 等 10s")
            time.sleep(10)
            continue
        try:
            return rd.chat(api, model, msgs, max_tokens, 0.0, 1.0, timeout)
        except Exception as e:
            log(f"    attempt={attempt} {type(e).__name__}: {e} → 等 15s")
            time.sleep(15)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--suite", default="ladder,instruction,multihop,consistency")
    ap.add_argument("--lengths", default="32000,64000,128000,256000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096, help="给足预算，避免把截断误判成退化")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=os.path.expanduser("~/lvllm/stress-run4/degen_suite.jsonl"))
    ap.add_argument("--md", default=os.path.expanduser("~/lvllm/stress-run4/degen_suite.md"))
    args = ap.parse_args()

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    api_root = f"http://127.0.0.1:{args.port}"
    api = api_root + "/v1/chat/completions"
    model = "DeepSeek-V4.1-Flash"

    def log(msg):
        print(f"[{rd.iso()}] {msg}", flush=True)

    chunks = build_corpus()
    log(f"语料块={len(chunks)}")
    if not chunks:
        log("⛔ 没有语料")
        return 2

    fh = open(args.out, "a", encoding="utf-8")
    rows = []

    def rec(**kw):
        kw.setdefault("ts", rd.iso())
        rows.append(kw)
        fh.write(json.dumps(kw, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    rng = random.Random(args.seed)
    total = 0
    for suite in suites:
        for L in lengths:
            if suite == "consistency" and L != lengths[-1]:
                continue
            for rep in range(args.reps if suite != "consistency" else 5):
                total += 1
                ctx = make_ctx(chunks, L, random.Random(args.seed + L + rep))
                if suite == "ladder":
                    msgs = [{"role": "user", "content": ctx + "\n\n" + LADDER_Q}]
                elif suite == "instruction":
                    msgs = [{"role": "user", "content": INSTR_TPL.format(ctx=ctx)}]
                elif suite == "multihop":
                    ctx2 = plant_facts(ctx, random.Random(args.seed + L + rep), FACTS[:2])
                    msgs = [{"role": "user", "content": ctx2 + "\n\n问题：代号 KX-7 的项目负责人是谁？只回答人名。"}]
                else:  # consistency
                    msgs = [{"role": "user", "content": ctx + "\n\n" + LADDER_Q}]
                log(f"[{total}] suite={suite} L={L} rep={rep} 预计 prompt≈{est(''.join(m['content'] for m in msgs))} tokens")
                res = call(api_root, api, model, msgs, args.max_tokens, args.timeout, log)
                if res is None:
                    rec(suite=suite, length=L, rep=rep, error="giveup")
                    continue
                text = res["text"]
                content = res.get("content_raw") or ""
                row = dict(
                    suite=suite, length=L, rep=rep,
                    prompt_tokens=res["prompt_tokens"], completion_tokens=res["completion_tokens"],
                    finish_reason=res["finish_reason"], latency_s=res["latency_s"],
                    answered=bool(content.strip()),
                    truncated=bool(res["reasoning_only"] and res["finish_reason"] == "length"),
                    repeat_ratio=round(rd.repeat_ratio(text), 3),
                    garbage_ratio=round(rd.garbage_ratio(text), 5),
                    content_head=" ".join(content.split())[:200],
                    reasoning_tail=" ".join((res.get("reasoning_raw") or "").split())[-200:],
                )
                if suite == "instruction":
                    ok, why = judge_json(content)
                    row["compliant"] = ok
                    row["compliance_note"] = why
                if suite == "multihop":
                    row["correct"] = "林昭" in content
                if suite == "consistency":
                    row["similar_to_first"] = None
                    if rep > 0:
                        first = next((r for r in reversed(rows) if r.get("suite") == "consistency" and r.get("length") == L and r.get("rep") == 0 and "content_head" in r), None)
                        if first:
                            row["similar_to_first"] = round(difflib.SequenceMatcher(None, first["content_head"], row["content_head"]).ratio(), 3)
                rec(**row)
                log(f"    → answered={row['answered']} trunc={row['truncated']} "
                    f"rep={row['repeat_ratio']} {row['latency_s']}s"
                    + (f" compliant={row.get('compliant')}({row.get('compliance_note')})" if suite == "instruction" else "")
                    + (f" correct={row.get('correct')}" if suite == "multihop" else ""))
    fh.close()

    # ---- markdown 汇总 ----
    lines = ["# 自建退化套件结果\n", f"- 输出: `{args.out}`", f"- 长度档: {lengths}，max_tokens={args.max_tokens}", ""]
    for suite in suites:
        sel = [r for r in rows if r.get("suite") == suite and "error" not in r]
        if not sel:
            continue
        lines.append(f"## {suite}\n")
        lines.append("| 长度 | n | 答出 | 截断 | 平均重复率 | 延迟 | 其它 |")
        lines.append("|---|---|---|---|---|---|---|")
        for L in sorted({r["length"] for r in sel}):
            ss = [r for r in sel if r["length"] == L]
            extra = ""
            if suite == "instruction":
                extra = f"合规 {sum(1 for r in ss if r.get('compliant'))}/{len(ss)}"
            elif suite == "multihop":
                extra = f"正确 {sum(1 for r in ss if r.get('correct'))}/{len(ss)}"
            elif suite == "consistency":
                sims = [r["similar_to_first"] for r in ss if r.get("similar_to_first") is not None]
                extra = f"一致性 min={min(sims):.2f}" if sims else "-"
            lines.append(f"| {L} | {len(ss)} | {sum(1 for r in ss if r['answered'])} | "
                         f"{sum(1 for r in ss if r['truncated'])} | "
                         f"{sum(r['repeat_ratio'] for r in ss)/len(ss):.3f} | "
                         f"{sum(r['latency_s'] for r in ss)/len(ss):.0f}s | {extra} |")
        lines.append("")
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n写出 {args.md}")
    print(open(args.md, encoding="utf-8").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
