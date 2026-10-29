#!/usr/bin/env python3
"""vary_soak.py —— 高多样性压测：把"数小时的负载多样性"压缩到几十分钟

为什么需要它（2026-10-29 已排除了五个假设，见事故文档 §33.26）
    真实故障的负载形态是「**数小时内、成千上万次、长度分布很宽、并发有起伏**」；
    而此前所有实验都是「几十个精心构造的请求」。唯一还没复刻的自变量就是**多样性 × 次数**。
    本脚本按真实形态的分布随机化：长度跨 1K–300K、采样参数随机、偶发双路并发。
    ⇒ 若死亡时间随之**按比例缩短**，则确认该自变量，并把 3 小时的等待压缩成分钟级实验窗口。

分布（可用环境变量改）
    · 60%  1K–8K     （短问答，最常见）
    · 25%  8K–40K    （中等）
    · 15%  40K–300K  （长上下文，最容易撞上大 seq_len_kv）
    · max_tokens 随机 1–512（偶发长生成 ⇒ decode 步数多）
    · 10% 的轮次发**两路并发**（模拟真实并发起伏；服务 MAXSEQS=2）

安全
    · 默认单流；失败即**停止并保留现场**（不无限重试，避免把证据冲掉）
    · 每步一行：时刻 / 目标长度 / 实际 prompt_tokens / max_tokens / 耗时 / 结果
"""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get("VS_URL", "http://127.0.0.1:8700/v1/chat/completions")
MODEL = os.environ.get("VS_MODEL", "DeepSeek-V4.1-Flash")
HEALTH = os.environ.get("VS_HEALTH", "http://127.0.0.1:8700/health")
WAIT = os.environ.get("VS_WAIT_HEALTH", "1") == "1"
SEED = int(os.environ.get("VS_SEED", "20261029"))
PAIR_P = float(os.environ.get("VS_PAIR_P", "0.10"))
STOP_ON_FAIL = os.environ.get("VS_STOP_ON_FAIL", "1") == "1"

UNIT = "分布式张量并行与拓扑排序的填充文本，用于精确控制请求长度，内容不参与判断。"


def text_for(tokens: int) -> str:
    need = max(1, int(tokens * 1.7))
    reps = need // len(UNIT) + 2
    return (UNIT * reps)[:need]


def pick_tokens(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.60:
        return rng.randint(1000, 8000)
    if r < 0.85:
        return rng.randint(8000, 40000)
    return rng.randint(40000, 300000)


def call(tokens: int, max_tokens: int, temperature: float, top_p: float, timeout: int = 3600):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text_for(tokens) + "\n用一句话总结。"}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        u = d.get("usage", {})
        return True, time.time() - t0, u.get("prompt_tokens"), u.get("completion_tokens"), ""
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, None, None, "%s: %s" % (type(e).__name__, str(e)[:140])


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


def main() -> int:
    rng = random.Random(SEED)
    if WAIT and not wait_health():
        print("FATAL 服务未就绪", flush=True)
        return 2
    print("== 高多样性压测开始（seed=%d, pair_p=%.2f）==" % (SEED, PAIR_P), flush=True)

    i = 0
    while True:
        i += 1
        tok = pick_tokens(rng)
        mt = rng.choice([1, 8, 32, 64, 128, 256, 512])
        temp = round(rng.uniform(0.0, 1.2), 2)
        topp = round(rng.uniform(0.5, 1.0), 2)

        if rng.random() < PAIR_P:
            # 双路并发：两路长度不同，制造新的批组合
            tok2 = pick_tokens(rng)
            res = {}

            def run(k, t_, m_, tm_, tp_):
                res[k] = call(t_, m_, tm_, tp_)

            ts = [
                threading.Thread(target=run, args=(0, tok, mt, temp, topp)),
                threading.Thread(target=run, args=(1, tok2, mt, temp, topp)),
            ]
            [t.start() for t in ts]
            [t.join() for t in ts]
            bad = [k for k, v in res.items() if not v[0]]
            desc = "PAIR %d+%d mt=%d" % (tok, tok2, mt)
            if bad:
                for k in bad:
                    print("FAIL %s 第%d路 %.1fs %s" % (desc, k, res[k][1], res[k][4]), flush=True)
                if STOP_ON_FAIL:
                    print("== 失败即停：现场留给看门狗 ==", flush=True)
                    return 1
            else:
                print(
                    "OK   %s pt=%s/%s %.1fs/%.1fs"
                    % (desc, res[0][2], res[1][2], res[0][1], res[1][1]),
                    flush=True,
                )
        else:
            ok, dt, pt, ct, err = call(tok, mt, temp, topp)
            if ok:
                print(
                    "OK   #%d tok~%-6d pt=%-6s ct=%-4s mt=%-3d T=%.2f %.1fs"
                    % (i, tok, pt, ct, mt, temp, dt),
                    flush=True,
                )
            else:
                print("FAIL #%d tok~%d mt=%d %.1fs %s" % (i, tok, mt, dt, err), flush=True)
                if STOP_ON_FAIL:
                    print("== 失败即停：现场留给看门狗 ==", flush=True)
                    return 1
                time.sleep(5)
        time.sleep(rng.uniform(0, 2))


if __name__ == "__main__":
    sys.exit(main())
