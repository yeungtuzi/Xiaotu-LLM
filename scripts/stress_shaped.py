#!/usr/bin/env python3
"""形状多样压测：按【真实流量的 prompt 长度分布】采样，模拟 agent 多轮增长。

为什么需要它：固定形状回放（同一输入重复 N 小时）是低熵输入，不触达形状相关的
崩溃路径；真实 agent 流量的杀伤力来自【形状多样性】。判据见 dev-docs/STRESS_METHODOLOGY.md。

用法：
  python3 scripts/stress_shaped.py --port 8700 --hours 24 --out /tmp/stress.csv
  python3 scripts/stress_shaped.py --port 8700 --selftest     # 只验证采样逻辑，不发请求

设计：
  * 分布来自 /metrics 的 vllm:request_prompt_tokens_bucket（累计计数器 → 差分得每桶计数）
  * 长度在所选桶内按 log-uniform 采样（避免长桶被线性偏置）
  * 会话模型：一定比例的请求是"续写"，长度 = 上轮长度 + 增量 ⇒ 模拟上下文增长
  * 图片数 {0,1,4} 按比例采样；并发可配
  * 每 30s 采样一次：显存/KV%/health/已完成请求数 → CSV
"""
import argparse, json, math, os, random, threading, time, urllib.error, urllib.request

PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

def fetch(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout).read().decode()

def parse_hist(text):
    """→ [(le, cumulative_count), ...] 按 le 升序"""
    import re
    pat = re.compile(r'request_prompt_tokens_bucket\{[^}]*le="([0-9.]+)"[^}]*\}\s+([0-9.]+)')
    rows = [(float(a), float(b)) for a, b in pat.findall(text)]
    return sorted(rows)

def bucket_weights(hist):
    """累计 → 每桶计数 → 归一化权重（跳过空桶）"""
    buckets, prev = [], 0.0
    for i, (le, cum) in enumerate(hist):
        cnt = max(0.0, cum - prev); prev = cum
        lo = hist[i-1][0] if i else 0.0
        if cnt > 0 and le > lo:
            buckets.append((lo, le, cnt))
    total = sum(c for _, _, c in buckets) or 1.0
    return [(lo, hi, c / total) for lo, hi, c in buckets]

def sample_len(weights, rng):
    """按桶权重选桶，桶内 log-uniform"""
    r, acc = rng.random(), 0.0
    for lo, hi, w in weights:
        acc += w
        if r <= acc:
            a, b = max(lo, 1.0), hi
            return int(math.exp(rng.uniform(math.log(a), math.log(b))))
    lo, hi, _ = weights[-1]
    return int(hi)

def words_for(n_tokens, rng):
    """粗粒度构造约 n_tokens 的文本（比例在 responses 的 usage.prompt_tokens 里复核）"""
    vocab = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
             "omicron pi rho sigma tau upsilon phi chi psi omega").split()
    out, approx = [], 0
    while approx < n_tokens:
        w = rng.choice(vocab); out.append(w); approx += 1
    return " ".join(out)

def build_payload(n_tokens, n_images, rng, session_tail=None):
    text = (session_tail + " " if session_tail else "") + words_for(n_tokens, rng)
    content = [{"type": "text", "text": text}]
    for _ in range(n_images):
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + PNG}})
    return {"model": "ds41f", "messages": [{"role": "user", "content": content}],
            "max_tokens": 8, "temperature": 0.0, "stream": False}

def post(port, payload, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--continue-ratio", type=float, default=0.45, help="续写请求比例（模拟上下文增长）")
    ap.add_argument("--img-ratio", default="0.6,0.3,0.1", help="图片 0/1/4 的比例")
    ap.add_argument("--out", default="/tmp/stress_shaped.csv")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    if a.selftest:
        hist = [(500, 3), (2000, 12), (10000, 20), (50000, 27), (200000, 30)]
        w = bucket_weights(hist)
        lens = [sample_len(w, rng) for _ in range(20000)]
        print("桶权重:", [(int(lo), int(hi), round(p,3)) for lo,hi,p in w])
        print("采样长度分位: p10=%d p50=%d p90=%d max=%d" % (
            sorted(lens)[int(.1*len(lens))], sorted(lens)[int(.5*len(lens))],
            sorted(lens)[int(.9*len(lens))], max(lens)))
        return

    hist = parse_hist(fetch(f"http://127.0.0.1:{a.port}/metrics"))
    weights = bucket_weights(hist)
    if not weights:
        raise SystemExit("❌ /metrics 里没有 request_prompt_tokens 直方图数据，无法采样真实分布")
    print("[shaped] 真实分布桶:", [(int(lo), int(hi), round(p,3)) for lo, hi, p in weights])

    imgs = [int(x) for x in a.img_ratio.split(",")]
    img_choices = [0, 1, 4]

    out = open(a.out, "w")
    out.write("ts,event,prompt_tokens,images,latency_s,ok,gpu1_mib,kv_perc\n"); out.flush()
    stop = threading.Event()
    lock = threading.Lock()
    sessions = []          # 每个活跃会话最近一轮的文本尾部（模拟上下文增长）
    stats = {"req": 0, "fail": 0, "tok": 0}

    def sampler():
        while not stop.wait(30):
            try:
                m = fetch(f"http://127.0.0.1:{a.port}/metrics", timeout=5)
                import re
                kv = re.search(r'kv_cache_usage_perc\{[^}]*\}\s+([0-9.]+)', m)
                gpu = os.popen("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits").read()
                g1 = next((l.split(",")[1].strip() for l in gpu.strip().splitlines()
                           if l.split(",")[0].strip() == "1"), "")
                with lock:
                    out.write(f"{int(time.time())},sample,,,,,{g1},{kv.group(1) if kv else ''}\n"); out.flush()
            except Exception as e:
                with lock:
                    out.write(f"{int(time.time())},sample_err,,,,,,{type(e).__name__}\n"); out.flush()

    threading.Thread(target=sampler, daemon=True).start()
    deadline = time.time() + a.hours * 3600

    def worker(wid):
        cur = None          # 当前会话的 prompt 长度（模拟 agent 上下文单调增长）
        while time.time() < deadline:
            try:
                if cur is None or rng.random() > a.continue_ratio:
                    n = sample_len(weights, rng)          # 新会话：按真实分布采样
                    cur = n
                else:
                    n = min(cur + rng.randint(256, 2048), 200000)   # 续写：追加一轮
                    cur = n
                n_img = rng.choices(img_choices, weights=imgs)[0]
                payload = build_payload(n, n_img, rng)
                body, dt = post(a.port, payload)
                ptok = (body.get("usage") or {}).get("prompt_tokens", 0)
                if ptok and cur < ptok: cur = ptok   # 用真实 token 数校准
                with lock:
                    stats["req"] += 1; stats["tok"] += ptok
                    out.write(f"{int(time.time())},req,{ptok},{n_img},{dt:.2f},1,,\n"); out.flush()
            except Exception as e:
                with lock:
                    stats["fail"] += 1
                    out.write(f"{int(time.time())},req_err,,,0,0,,{type(e).__name__}\n"); out.flush()
                time.sleep(5)
        return

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.concurrency)]
    for t in ths: t.start()
    try:
        while any(t.is_alive() for t in ths):
            for t in ths: t.join(timeout=5)
    except KeyboardInterrupt:
        pass
    stop.set()
    out.close()
    print(f"[shaped] 完成：请求 {stats['req']}，失败 {stats['fail']}，累计 prompt_tokens {stats['tok']}")

if __name__ == "__main__":
    main()
