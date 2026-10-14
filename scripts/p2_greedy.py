# scripts/p2_greedy.py —— 对拍批量:固定 prompt 集 × greedy(temp=0)→ JSONL
# 用法: python p2_greedy.py <port> <model_id> <out.jsonl>
import sys, json, urllib.request

PORT, MID, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = [
    "用一句话解释什么是KV cache。", "Explain gradient descent in one sentence.",
    "写一个 Python 函数反转字符串。", "Why is the sky blue?",
    "把这句话翻译成英文:知识就是力量。", "List three prime numbers greater than 100.",
    "什么是幂律分布?一句话。", "Write a haiku about CPUs.",
    "简述 TCP 握手过程。", "Compute 17 times 23.",
    "推荐一本关于并行编程的书并说明理由。", "用一个比喻解释 CUDA graph。",
]
url = f"http://127.0.0.1:{PORT}/v1/completions"
with open(OUT, "w") as f:
    for i, p in enumerate(PROMPTS):
        body = json.dumps({"model": MID, "prompt": p, "temperature": 0,
                           "max_tokens": 64, "stream": False}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.load(r)
        f.write(json.dumps({"i": i, "text": j["choices"][0]["text"]}, ensure_ascii=False) + "\n")
        f.flush()
print("done", len(PROMPTS), "->", OUT)
