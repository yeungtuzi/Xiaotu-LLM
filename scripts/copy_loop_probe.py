#!/usr/bin/env python3
"""复制循环溯源:
 ① 抓 **完整输出**(completions 传输,与基准同口径);
 ② 同一批转录走 **chat/completions**(真实 DSH 服务实际用的端点)作对照;
 ③ 报告:输出长度、字符 8-gram 复制率、**循环起点**(滑动窗复制率首次>0.5 的位置),
    以及重复片段是否**出自 prompt**(在 prompt 里找同一 40 字符片段)。

用法: python copy_loop_probe.py <outdir>
"""
import json
import os
import statistics
import sys
import time
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/copyloop"
os.makedirs(OUT, exist_ok=True)
W = "/home/user/tmp/c4rerun_20261012"
DS = [("f_r11", "r11"), ("f_r13", "r13"), ("f_r20", "r20"), ("f_r32", "r32")]
URL = "http://127.0.0.1:8070/v1/"


def post(path, body, timeout=1800):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def deg_char(t, n=8):
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


def onset(t, win=400, step=50, thr=0.5, n=8):
    """滑动窗复制率首次 > thr 的字符位置(近似循环起点)。"""
    for i in range(0, max(1, len(t) - win), step):
        if deg_char(t[i:i + win], n) > thr:
            return i
    return -1


recs = {k: json.loads(open("%s/%s.jsonl" % (W, f)).readline())["prompt"] for f, k in DS}
summary = []
for f, k in DS:
    prompt = recs[k]
    # ---- ① completions(基准同口径)----
    j = post("completions", {"model": "DeepSeek-V4.1-Flash", "prompt": prompt,
                             "max_tokens": 1024, "temperature": 0})
    t = (j["choices"][0] or {}).get("text") or ""
    u = j["usage"]
    on = onset(t)
    # 重复片段是否出自 prompt?
    src = None
    if on > 0:
        seg = t[on:on + 40]
        if seg and seg in prompt:
            src = "**prompt 原文**"
        elif seg and t.find(seg, on + 40) > 0:
            src = "自身复读"
        else:
            src = "其它"
    open("%s/compl_%s.txt" % (OUT, k), "w").write(t)
    summary.append(("completions", k, len(t), u["completion_tokens"],
                    j["choices"][0].get("finish_reason"), deg_char(t), on, src))
    print("completions %s: chars=%d tok=%d finish=%s deg=%.2f 循环起点(字符)=%d 来源=%s" % (
        k, len(t), u["completion_tokens"], j["choices"][0].get("finish_reason"),
        deg_char(t), on, src), flush=True)
    # ---- ② chat(真实服务端点)----
    try:
        jc = post("chat/completions", {"model": "DeepSeek-V4.1-Flash",
                                       "messages": [{"role": "user", "content": prompt}],
                                       "max_tokens": 1024, "temperature": 0})
        ch = (jc["choices"][0].get("message") or {})
        tc = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
        uc = jc["usage"]
        open("%s/chat_%s.txt" % (OUT, k), "w").write(tc)
        print("chat        %s: chars=%d tok=%d finish=%s deg=%.2f 循环起点=%d" % (
            k, len(tc), uc["completion_tokens"], jc["choices"][0].get("finish_reason"),
            deg_char(tc), onset(tc)), flush=True)
        summary.append(("chat", k, len(tc), uc["completion_tokens"],
                        jc["choices"][0].get("finish_reason"), deg_char(tc), onset(tc), "-"))
    except Exception as e:  # noqa: BLE001
        print("chat %s ERROR %r" % (k, e), flush=True)

print()
print("| 传输 | 转录 | 字符数 | token | finish | deg(8-gram复制率) | 循环起点(字符) | 重复来源 |")
print("|---|---|---:|---:|---|---:|---:|---|")
for row in summary:
    print("| %s | %s | %d | %d | %s | **%.2f** | %d | %s |" % row)
comp = [r for r in summary if r[0] == "completions"]
chat = [r for r in summary if r[0] == "chat"]
for lab, S in (("completions", comp), ("chat", chat)):
    if S:
        print("\n%s: token 中位=%.0f, deg 中位=%.2f, deg>0.3 的比例=%d/%d" % (
            lab, statistics.median(r[3] for r in S), statistics.median(r[5] for r in S),
            sum(1 for r in S if r[5] > 0.3), len(S)))
