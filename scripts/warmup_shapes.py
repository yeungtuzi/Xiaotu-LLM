#!/usr/bin/env python3
"""warmup_shapes.py —— 启动后预热真实形状：文本多长度 + 1/4 张图 + 并发 2。

目的：把"运行期首次编译"（Triton JIT / cuBLAS workspace / ViT 路径）提前到启动期，
      消掉 2026-10-31 那类 `Triton Error [CUDA]: out of memory` 崩溃。

用法：python3 warmup_shapes.py --port 8700 --model DeepSeek-V4.1-Flash
"""
import argparse, base64, json, os, struct, threading, time, urllib.request, zlib

_API_KEY = ""   # 由 --api-key 填；空则回落到 env VLLM_API_KEY


def make_png(w=768, h=768):
    """纯 zlib 生成 PNG（不依赖 PIL）：渐变 + 色块，足够触发视觉编码。"""
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(((x * 255) // w, (y * 255) // h, ((x ^ y) & 0xFF)))
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")).decode()


def post(port, body, timeout=600):
    # ⭐ 鉴权（2026-11-15）：引擎启用 API key 后，预热请求也必须带 Bearer，
    #    否则会 401 ⇒ warmup ok=0 fail=8（服务起来但预热全失败）。
    #    密钥来源优先级：--api-key > 环境变量 VLLM_API_KEY；两者都空 ⇒ 不鉴权（与旧行为一致）。
    hdrs = {"content-type": "application/json"}
    _key = _API_KEY or os.environ.get("VLLM_API_KEY") or ""
    if _key:
        hdrs["Authorization"] = "Bearer " + _key
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode(), headers=hdrs)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return time.time() - t0, (data.get("usage") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--text-lengths", default="512,2048,8192,32768")
    ap.add_argument("--image-counts", default="1,4")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--api-key", default="", help="引擎启用鉴权时必填（或设 env VLLM_API_KEY）")
    a = ap.parse_args()
    global _API_KEY
    _API_KEY = a.api_key or ""

    png = make_png()
    ok = fail = 0

    def run(label, text, images):
        nonlocal ok, fail
        content = [{"type": "text", "text": text}]
        for _ in range(images):
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + png}})
        body = {"model": a.model, "max_tokens": a.max_tokens, "temperature": 0.0,
                "messages": [{"role": "user", "content": content}]}
        t0 = time.time()
        try:
            dt, usage = post(a.port, body)
            ok += 1
            print("[warmup] %-22s %.1fs  prompt_tokens=%s" %
                  (label, dt, (usage or {}).get("prompt_tokens", "?")), flush=True)
        except Exception as e:                                     # noqa: BLE001
            fail += 1
            print("[warmup] %-22s ✗ %s: %s" % (label, type(e).__name__, str(e)[:120]),
                  flush=True)

    # ① 文本长度档（覆盖 Inductor/compile_ranges 与 attention 变体）
    for n in [int(x) for x in a.text_lengths.split(",") if x]:
        run("text-%d" % n, "warmup " * (n // 2), 0)

    # ② 图片档（覆盖 ViT / mm 路径 —— 压测从未跑过这条路）
    for k in [int(x) for x in a.image_counts.split(",") if x]:
        run("image-%d" % k, "describe these images", k)

    # ③ 并发 2（覆盖 batch=2 的图与 kernel 变体）
    ts = [threading.Thread(target=run, args=("concurrent-2#%d" % i, "warmup " * 1024, 1))
          for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]

    print("[warmup] 完成: ok=%d fail=%d" % (ok, fail), flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
