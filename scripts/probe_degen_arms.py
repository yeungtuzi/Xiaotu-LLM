#!/usr/bin/env python3
"""退化归因：把「前缀缓存未命中」与「长 prefill 计算」两个变量拆开的四臂对照。

背景（dev-docs/DEGEN_STAGE_COMPARISON.md #23）：
    既有 soak_ab.py 的 unique 臂在 prompt **开头**插 uuid ⇒ 该臂同时"未命中"且"全量长 prefill"，
    与 shared 臂（全命中 + 几乎不 prefill）差了两个变量，无法区分：
      (a) 未命中/插入路径写坏 KV；       (b) 长 prefill 计算本身有问题。

本探针的四臂（同一份含针文档，同一问题）：

    shared      BASE + doc            + q    全前缀命中，prefill≈0
    unique      BASE + uuid + doc     + q    第 0 块不匹配 ⇒ 全量长 prefill
    tail        BASE + doc + uuid     + q    长前缀命中，仅尾部几 token 为新
    suffix      BASE + docA + docB    + q    docA 命中、docB 全新（docB 内含针）

判读：
    tail  成功、unique 失败 ⇒ 缓存里的长 KV 可读、长上下文注意力正常
                              ⇒ 失败只发生在「长 prefill 的那一次请求」⇒ 支持 (b)
    tail  失败              ⇒ 缓存中的 KV 内容被污染（同一份 KV 之前写进去的）⇒ 支持 (a)
    suffix 失败、shared 成功 ⇒ 与"是否新内容"有关，与"是否全量"无关 ⇒ 进一步支持 (b)

机械自检（每次请求都记录，不靠"参数传了"）：
    * 每请求读 `usage.prompt_tokens_details.cached_tokens`；缺失时回落到 /metrics 的
      `prefix_cache_hits_total` 前后差值。四臂的 cached_tokens 必须呈现
      shared/tail 高、unique/suffix 低，否则该轮作废并打印 ARM-BROKEN。

用法：
    python3 scripts/probe_degen_arms.py                 # 默认 PORT=8700，3 轮
    PORT=8700 ROUNDS=3 N=1200 python3 scripts/probe_degen_arms.py
    ARM_FILTER=tail python3 scripts/probe_degen_arms.py
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("PORT", "8700"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
N = int(os.environ.get("N", "1200"))
ARM_FILTER = [a for a in os.environ.get("ARM_FILTER", "").split(",") if a]
NEEDLE = int(os.environ.get("NEEDLE", "981273"))
MAXTOK = int(os.environ.get("MAXTOK", "64"))
UUID_FIXED = os.environ.get("UUID_FIXED", "")

BASE = "You are a helpful assistant. Answer using only the document.\n\n"
Q = "\n\nWhat is the access code for project X? Answer with the number only."

rng = random.Random(20261107)


def doc(n, salt="", depth=0.5):
    """第 int(n*depth) 行是针（access code for project X = NEEDLE），其余为噪声。"""
    lines = []
    needle_row = min(n - 1, max(0, int(n * depth)))
    for i in range(n):
        if i == needle_row:
            lines.append("Entry %d: the access code for project X is %d." % (i, NEEDLE))
        else:
            lines.append("Entry %d: the access code for project P%s%d is %d."
                         % (i, salt, i, 100000 + (i * 37 + n) % 899999))
    return "\n".join(lines)


# docA 固定（可被缓存），docB 每轮新盐（必然未命中）
DOC_A = doc(N, salt="A")
_round_salts = {}


def doc_b(round_idx):
    if round_idx not in _round_salts:
        _round_salts[round_idx] = doc(N, salt="B%d-%s" % (round_idx, rng.getrandbits(40)))
    return _round_salts[round_idx]


def metrics_hits():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % PORT, timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:prefix_cache_hits_total"):
                    return float(line.split()[-1])
    except Exception:
        return None
    return None


def ask(prompt):
    # ⚠️ 2026-11-10：官方 1017 权重**没有 chat template**（tokenizer_config 里没有该键），
    #    因此 `/v1/chat/completions` 在真权重上不可依赖 ⇒ 与 probe_dump_long.py 一致改用 completions。
    body = json.dumps({
        "model": "DeepSeek-V4.1-Flash",
        "prompt": prompt,
        "max_tokens": MAXTOK, "temperature": 0, "top_p": 1,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT,
                                 data=body, headers={"Content-Type": "application/json"})
    h0 = metrics_hits()
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        d = json.loads(resp.read())
    dt = time.time() - t0
    h1 = metrics_hits()
    out = d["choices"][0].get("text") or ""
    usage = d.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and h0 is not None and h1 is not None:
        cached = int(h1 - h0)
    return {
        "ok": str(NEEDLE) in out,
        "out": out.strip()[:110].replace("\n", " "),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached": cached,
        "sec": round(dt, 1),
    }


DEPTH = float(os.environ.get("DEPTH", "0.5"))


def build(arm, rnd, u):
    if arm == "shared":
        return BASE + doc(N, depth=DEPTH) + Q
    if arm == "unique":
        return BASE + u + "\n\n" + doc(N, depth=DEPTH) + Q
    if arm == "tail":
        return BASE + doc(N, depth=DEPTH) + "\n\nSession tag: %s" % u + Q
    if arm == "suffix":
        return BASE + DOC_A + "\n\n" + doc_b(rnd) + Q
    raise SystemExit("unknown arm %s" % arm)


def main():
    arms = ["shared", "unique", "tail", "suffix"]
    if ARM_FILTER:
        arms = [a for a in arms if a in ARM_FILTER]
    st = {a: {"ok": 0, "n": 0, "cached": [], "sec": []} for a in arms}
    print("# probe_degen_arms PORT=%d ROUNDS=%d N=%d DEPTH=%.2f uuid=%s arms=%s"
          % (PORT, ROUNDS, N, DEPTH, UUID_FIXED or "random", ",".join(arms)),
          flush=True)

    # warm：先把 BASE+doc(N)、BASE+DOC_A 预热进缓存（各自 1 次），不计分
    if "shared" in arms or "tail" in arms:
        ask(BASE + doc(N, depth=DEPTH) + Q)
    if "suffix" in arms:
        ask(BASE + DOC_A + "\n\n" + doc_b(-1) + Q)

    for rnd in range(1, ROUNDS + 1):
        order = arms[:]
        rng.shuffle(order)
        for arm in order:
            u = UUID_FIXED or os.urandom(8).hex()
            try:
                r = ask(build(arm, rnd, u))
            except Exception as e:  # noqa: BLE001
                print("轮%d %-7s EXC %s" % (rnd, arm, str(e)[:80]), flush=True)
                continue
            s = st[arm]
            s["n"] += 1
            s["ok"] += 1 if r["ok"] else 0
            s["cached"].append(r["cached"])
            s["sec"].append(r["sec"])
            print("轮%d %-7s 命中=%s cached=%s ptok=%s %5.1fs  %s"
                  % (rnd, arm, "✓" if r["ok"] else "✗", r["cached"], r["prompt_tokens"],
                     r["sec"], r["out"][:70]), flush=True)

    print("\n# 汇总", flush=True)
    bad = []
    for arm in arms:
        s = st[arm]
        c = [x for x in s["cached"] if isinstance(x, int)]
        print("%-7s 命中 %d/%d  cached=%s  avg=%.1fs"
              % (arm, s["ok"], s["n"],
                 (("%d-%d" % (min(c), max(c))) if c else "n/a"),
                 (sum(s["sec"]) / len(s["sec"])) if s["sec"] else -1), flush=True)
        if arm in ("shared", "tail") and c and min(c) < N // 4:
            bad.append(arm)
        if arm == "unique" and c and max(c) > N // 4:
            bad.append(arm)
        # ⚠️ 2026-11-10 修正：`suffix` = BASE + DOC_A + docB ⇒ DOC_A 是**预热过的共享前缀**
        #    ⇒ 它的 cached 本来就应该"大"。旧检查把它和 `unique` 归为一类（要求 cached 小），
        #    与 design 自相矛盾 —— 一旦生效就会在"一切正常"时打印 ARM-BROKEN。
        #    改为：只报告、不否决（其值在汇总里可见）。
    if bad:
        print("ARM-BROKEN: 这些臂的 cached_tokens 与预期不符 ⇒ 本轮结论作废：%s" % bad, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
