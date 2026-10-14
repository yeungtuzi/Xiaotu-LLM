#!/usr/bin/env python3
"""语义一致门测三联(GSM8K / NIAH / 视觉合成探针),temp=0 双臂同形。
用法: p2_quality.py <port> <model> <out.json>"""
import base64, json, random, re, struct, subprocess, sys, threading, time, urllib.request, zlib

PORT, MODEL, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}"


def chat(content, max_tokens, image=None):
    parts = []
    if image:
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}})
    parts.append({"type": "text", "text": content})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": parts}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    for t in range(4):
        try:
            c = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
            return ((c.get("content") or "") + "\n" + (c.get("reasoning") or ""))
        except Exception:
            time.sleep(10 * (t + 1))
    return ""


# ---- 1) GSM8K 128 零样本 ----
def gsm():
    rows = [json.loads(l) for l in open("/tmp/gsm_test.jsonl")][:128]
    res, lock = [], threading.Lock()
    def worker(idxs):
        for i in idxs:
            q = rows[i]["question"] + "\n请推理后,最后一行以 #### <答案数字> 结束。"
            txt = chat(q, 640)
            m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", txt)
            got = m[-1].replace(",", "") if m else ""
            exp = re.search(r"####\s*(-?[\d,\.]+)", rows[i]["answer"]).group(1).replace(",", "")
            with lock:
                res.append((i, int(float(got)) == int(float(exp)) if got else False))
    ths = [threading.Thread(target=worker, args=(list(range(k, 128, 4)),)) for k in range(4)]
    for t in ths: t.start()
    for t in ths: t.join()
    return res

# ---- 2) NIAH ----
def niah():
    rng = random.Random(7)
    w = "数据 系统 网络 节点 缓存 请求 调度 内存 磁盘 线程 队列 算法 芯片 服务 日志 索引".split()
    probes = []
    for L, tag in ((1600, "32k"), (3000, "64k")):
        for d in (10, 30, 50, 70, 90, 99):
            key = f"紫电{rng.randint(100,999)}"
            n = max(1, int(L * d / 100))
            filler = []
            for j in range(L):
                filler.append(f"事项{j:05d}:" + "".join(rng.choice(w) + "的" for _ in range(5)) + rng.choice(w))
                if j == n:
                    filler.append(f"重要提醒:本次任务的密钥词是「{key}」。")
            probes.append((tag, d, key, "\n".join(filler)))
    out = []
    for tag, d, key, ctx in probes:
        txt = chat(ctx + "\n\n问题:上文提到的密钥词是什么?只回答密钥词本身。", 96)
        out.append((tag, d, key, key in txt.replace("「", "").replace("」", "")))
    return out

# ---- 3) 视觉 12 探针(手搓 PNG,零依赖) ----
def png_of(px, s=224):
    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(s)) for y in range(s))
    def ck(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
    R, G, B = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    return base64.b64encode(b"\x89PNG\r\n\x1a\n"
        + ck(b"IHDR", struct.pack(">IIBBBBB", s, s, 8, 2, 0, 0, 0))
        + ck(b"IDAT", zlib.compress(raw, 6)) + ck(b"IEND", b"")).decode()

def V():
    R, G, B, W, K, Y, GR = ((255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0),(255,255,0),(128,128,128))
    s = 224
    def sol(c): return lambda x, y: c
    def tb(a, b): return lambda x, y: a if y < s//2 else b
    def lr(a, b): return lambda x, y: a if x < s//2 else b
    def vst(n, a, b): return lambda x, y: a if (x*n//s) % 2 == 0 else b
    def hst(n, a, b): return lambda x, y: a if (y*n//s) % 2 == 0 else b
    def sq(c): return lambda x, y: c if s//4 <= x < 3*s//4 and s//4 <= y < 3*s//4 else K
    def circ(c): return lambda x, y: c if (x-s//2)**2+(y-s//2)**2 < (s//3)**2 else W
    def tri(c): return lambda x, y: c if (y > s//4 and abs(x-s//2) < (y-s//4)*0.8) else W
    def bars(hs): return lambda x, y: R if y > s - hs[min(2, x*3//s)] else W
    def squares(pairs):
        def f(x, y):
            for (x0, y0, c) in pairs:
                if x0 <= x < x0+30 and y0 <= y < y0+30: return c
            return W
        return f
    probes = [
        (png_of(sol(R)), "图中主要是什么颜色?一个字。", "红"),
        (png_of(sol(G)), "图中主要是什么颜色?一个字。", "绿"),
        (png_of(tb(R, B)), "图片上半部分是什么颜色?一个字。", "红"),
        (png_of(lr(K, W)), "图片右半部分是什么颜色?一个字。", "白"),
        (png_of(vst(6, B, W)), "图中有几条蓝色竖纹?只答数字。", "3"),
        (png_of(hst(10, R, W)), "图中有几条红色横纹?只答数字。", "5"),
        (png_of(sq(Y)), "中央方块是什么颜色?一个字。", "黄"),
        (png_of(circ(B)), "中央的圆形是什么颜色?一个字。", "蓝"),
        (png_of(tri(G)), "图中绿色的是什么形状?两个字。", "三角"),
        (png_of(bars([20, 80, 45])), "红黑柱状图,从左到右第几根最高?只答数字。", "2"),
        (png_of(squares([(10,10,R),(120,10,R),(10,120,R),(120,120,B)])),
         "哪种颜色的方块数量更多?一个字。", "红"),
        (png_of(lambda x, y: tuple(min(255, (x+y)*255//(2*s)) for _ in range(3))),
         "这是一幅灰度渐变图,中心区域的明度应描述为?一个字。", "灰"),
    ]
    return [(q, c, chat(q, 128, img)) for img, q, c in probes]

def dump(g, n, v):
    rep = {"gsm": {"ok": sum(1 for _, o in g if o), "n": len(g),
                   "wrong": [i for i, o in g if not o]},
           "niah": {"ok": sum(1 for *_, o in n if o), "n": len(n),
                    "miss": [(t, d) for t, d, k, o in n if not o]},
           "vision": {"ok": sum(1 for q, c, t in v if c in t), "n": len(v),
                      "miss": [q for q, c, t in v if c not in t]}}
    json.dump({"report": rep}, open(OUT, "w"), ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


legs = sys.argv[4] if len(sys.argv) > 4 else "gsm,niah,vision"
g = gsm() if "gsm" in legs else [(i, True) for i in range(128)]
if "gsm" in legs:
    dump(g, [], [])
n = niah() if "niah" in legs else []
if "niah" in legs:
    dump(g, n, [])
v = V() if "vision" in legs else []
dump(g, n, v)
