#!/usr/bin/env python3
"""ladder_client.py —— "异常长度阶梯"探针：快速复现 / 否证"新形状触发"假设

假设：崩溃点在 `_init_handles`（Triton 内核模块**首次加载**）⇒ 触发条件是某个
      **此前从未出现过的长度/形状特化**。若成立，则用一串"平时不会出现的长度"
      逐级打过去，应在**分钟级**复现；若整串都过，则否证该假设。

做法：串行、单并发，长度逐一递增且刻意避开常见值（不以 0/5 结尾、多为质数或 2^n±1），
      每一步记录：字符数 / 实际 prompt_tokens / 耗时 / 结果。
      任一步失败（500 / 连接拒绝 / 超时）⇒ 记 FAIL 并**立即停止**（现场留给看门狗）。

用法：
  LADDER_URL=http://127.0.0.1:8700/v1/chat/completions \
  python scripts/ladder_client.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("LADDER_URL", "http://127.0.0.1:8700/v1/chat/completions")
MODEL = os.environ.get("LADDER_MODEL", "DeepSeek-V4.1-Flash")
MAXTOK = int(os.environ.get("LADDER_MAXTOK", "32"))
WAIT_HEALTH = os.environ.get("LADDER_WAIT_HEALTH", "1") == "1"
HEALTH = os.environ.get("LADDER_HEALTH", "http://127.0.0.1:8700/health")

# 目标 token 数（中文≈1.7 字符/token，按比例生成后再用返回值核对）
TARGETS = [1700, 1701, 1703, 1711, 1733, 1741, 2047, 2049, 2731, 3071,
           4093, 4099, 5003, 6141, 8191, 10007, 12289, 16381, 20011]

FILLER = "压测填充文本，长度需精确控制，内容本身不参与判断。" * 400


def wait_healthy(timeout_s: int = 900) -> bool:
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


def ask(target_tokens: int):
    chars = int(target_tokens * 1.7)
    prompt = FILLER[:chars] + "\n用一句话总结。"
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.6,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        u = d.get("usage", {})
        return True, time.time() - t0, u.get("prompt_tokens"), ""
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, None, "%s: %s" % (type(e).__name__, str(e)[:140])


def main() -> int:
    if WAIT_HEALTH and not wait_healthy():
        print("FATAL 服务未在超时内就绪", flush=True)
        return 2
    print("== 阶梯开始（目标长度 → 实际 prompt_tokens）==", flush=True)
    for t in TARGETS:
        ok, dt, pt, err = ask(t)
        if ok:
            print("OK   target=%d chars=%d pt=%s %.1fs" % (t, int(t * 1.7), pt, dt), flush=True)
        else:
            print("FAIL target=%d chars=%d %.1fs %s" % (t, int(t * 1.7), dt, err), flush=True)
            print("== 复现成功：该长度附近即为触发点，现场已留给看门狗 ==", flush=True)
            return 1
        time.sleep(1)
    print("== 阶梯全部通过：**否证**『新长度特化触发』假设 ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
