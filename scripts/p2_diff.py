# scripts/p2_diff.py —— 两臂 greedy 输出一致率(最长公共前缀 token 近似 + 全等率)
# 用法: python p2_diff.py a.jsonl b.jsonl
import sys, json

def toks(s):
    return s.split()

A = {r["i"]: r["text"] for r in map(json.loads, open(sys.argv[1]))}
B = {r["i"]: r["text"] for r in map(json.loads, open(sys.argv[2]))}
n = len(A)
ident = 0
lcp = tot = 0
for i in sorted(A):
    ta, tb = toks(A[i]), toks(B[i])
    k = 0
    while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
        k += 1
    lcp += k
    tot += max(len(ta), len(tb))
    ident += A[i] == B[i]
    if A[i] != B[i]:
        print(f"  prompt{i}: 分叉于第 {k} token(A长{len(ta)} B长{len(tb)})")
print(f"一致率: token级LCP={100*lcp/tot:.2f}% 全等={ident}/{n} (100*{ident}/n={100*ident/n:.1f}%)")
