#!/usr/bin/env python3
"""state_probe.py —— 验证组合假设：**新特化 × 累积状态**（而不是新特化本身）

为什么需要它（2026-10-29 结论）
    · `spec_sweep.py` 在**刚启动、内存宽松**的实例上把所有"没见过的特化"打了个遍，全部通过
      ⇒ "新特化首次加载"**单独**不足以触发崩溃；
    · 真实故障都发生在**跑了数小时后**：KV 占用高、分配碎片多、pinned/段都扩到最大。
      ⇒ 需要复刻"**最没空间的时候**才第一次遇到某形状"这个交叉条件。

两阶段
    A 累积：**同时保留一个长请求在飞**（长 prompt + 长生成），使 KV 块持续被占住；
            ⚠️ 2026-10-29 实测教训：若像先前那样**串行**发长请求，两次请求之间 KV 会被释放，
               `kv_cache_usage_perc` 采样永远是 0 —— 那样根本没有"高占用状态"，实验无效。
    B 发射：**只在这时**发出"保留长度"（从未在本实例上出现过），逐个记录；任一步失败即停。

保留长度（刻意避开 sweep 已用过的值：1,2,3,15,16,17,31,32,33,47,48,49,101,103,107,109,113,
          1009,1021,2039,8191,8192,8193,16385,24577,32771）
    ⇒ 1709 / 2741 / 4099 / 6151 / 8209 / 10009 / 12281 / 20483 / 32779

用法：python scripts/state_probe.py
环境：PROBE_URL / PROBE_HEALTH / PROBE_KV_TARGET / PROBE_MAX_ACC_TOKENS
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("PROBE_URL", "http://127.0.0.1:8700/v1/chat/completions")
MODEL = os.environ.get("PROBE_MODEL", "DeepSeek-V4.1-Flash")
HEALTH = os.environ.get("PROBE_HEALTH", "http://127.0.0.1:8700/health")
METRICS = os.environ.get("PROBE_METRICS", "http://127.0.0.1:8700/metrics")
KV_TARGET = float(os.environ.get("PROBE_KV_TARGET", "0.45"))
HOLD_TOKENS = int(os.environ.get("PROBE_HOLD_TOKENS", "400000"))
HOLD_GEN = int(os.environ.get("PROBE_HOLD_GEN", "1500"))
HOLD_WAIT = float(os.environ.get("PROBE_HOLD_WAIT", "180"))

UNIT = "分布式张量并行与拓扑排序的填充文本，用于精确控制请求长度，内容不参与判断。"  # ~37 字

# sweep 用过的长度全部排除，避免 novelty 被提前消耗
RESERVED = [1709, 2741, 4099, 6151, 8209]


def text_for(tokens: int) -> str:
    """按"1 token ≈ 1.7 字"生成**任意长度**的文本（不像旧脚本那样被 FILLER 封顶）。"""
    need = max(1, int(tokens * 1.7))
    reps = need // len(UNIT) + 2
    return (UNIT * reps)[:need]


def get(url: str, timeout: int = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def kv_perc() -> float:
    try:
        for line in get(METRICS).splitlines():
            if line.startswith("vllm:kv_cache_usage_perc"):
                return float(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def ask(tokens: int, max_tokens: int = 8, timeout: int = 1800):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text_for(tokens) + "\n用一句话总结。"}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        u = d.get("usage", {})
        return True, time.time() - t0, u.get("prompt_tokens"), ""
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, None, "%s: %s" % (type(e).__name__, str(e)[:130])


def main() -> int:
    print("== 组合假设探针：先累积状态，再发射保留长度 ==", flush=True)
    print("起始 kv=%.3f 目标=%.2f" % (kv_perc(), KV_TARGET), flush=True)

    # ---- A 累积：起一个"在飞"的长请求，让 KV 块持续被占住 ----
    import threading

    holder = {"kv": 0.0, "done": False, "err": ""}

    def hold():
        """长 prompt + 长生成；在这段时间内 KV 块一直占着，slot 1 也一直占着。"""
        ok, dt, pt, err = ask(HOLD_TOKENS, max_tokens=HOLD_GEN, timeout=3600)
        holder.update(kv=kv_perc(), done=True, err="" if ok else err)
        print("长请求结束 pt=%s %.1fs %s" % (pt, dt, holder["err"]), flush=True)

    th = threading.Thread(target=hold, daemon=True)
    th.start()
    print("已发起长请求（%d token, 生成 %d），等待 KV 抬升…" % (HOLD_TOKENS, HOLD_GEN), flush=True)
    t0 = time.time()
    while time.time() - t0 < HOLD_WAIT and not holder["done"]:
        kv = kv_perc()
        if kv >= KV_TARGET:
            print("累积到位：kv=%.3f（%.0fs）" % (kv, time.time() - t0), flush=True)
            break
        time.sleep(2)
    else:
        print("累积等待结束：kv=%.3f" % kv_perc(), flush=True)

    # ---- B 发射保留长度（此刻第一次出现） ----
    print("--- 发射保留长度（从未在本实例出现）---", flush=True)
    for t in RESERVED:
        kv = kv_perc()
        ok, dt, pt, err = ask(t)
        if ok:
            print("OK   target=%-6d pt=%-6s %.1fs kv=%.3f" % (t, pt, dt, kv), flush=True)
        else:
            print("FAIL target=%-6d %.1fs kv=%.3f %s" % (t, dt, kv, err), flush=True)
            print("== 命中：『新特化 × 高占用状态』复现 ==", flush=True)
            return 1
    print("== 保留长度全部通过，最终 kv=%.3f ==" % kv_perc(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
