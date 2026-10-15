#!/usr/bin/env python3
"""假设:预装配载荷走 completions 时缺 <|im_end|> 停止串 ⇒ 模型越过轮次边界继续伪造下一轮 ⇒ 模板循环。
检验:同一批载荷,completions 分别 ①不设 stop ②设 stop=["<|im_end|>"],比长度与 deg。
"""
import json, statistics, sys, time, urllib.request
DS="/home/user/lvllm/process_data/data/dsh_assembled_bench.jsonl"
N=int(sys.argv[1]) if len(sys.argv)>1 else 8
recs=[json.loads(l) for l in open(DS,encoding="utf-8") if l.strip()][:N]
def deg(t,n=8):
    if len(t)<n*2: return 0.0
    seen,dup,tot=set(),0,0
    for i in range(len(t)-n+1):
        g=t[i:i+n]; tot+=1
        if g in seen: dup+=1
        else: seen.add(g)
    return dup/tot
def run(r, stop):
    body={"model":"DeepSeek-V4.1-Flash","prompt":r["prompt"],"max_tokens":1024,"temperature":0}
    if stop: body["stop"]=stop
    req=urllib.request.Request("http://127.0.0.1:8070/v1/completions",data=json.dumps(body).encode(),
                               headers={"content-type":"application/json"})
    t0=time.time(); j=json.load(urllib.request.urlopen(req,timeout=1800)); ch=j["choices"][0]
    t=ch.get("text") or ""
    return {"tok":j["usage"]["completion_tokens"],"finish":ch.get("finish_reason"),"deg":deg(t),"dt":time.time()-t0,"text":t}
print("| 转录 | 无 stop: tok / finish / deg | **设 stop=['<|im_end|>']**: tok / finish / deg |")
print("|---:|---|---|")
res={0:[],1:[]}
for i,r in enumerate(recs):
    a=run(r,None); b=run(r,["<|im_end|>"])
    res[0].append(a); res[1].append(b)
    print("| %d | %d / %s / %.2f | **%d / %s / %.2f** |" % (i,a["tok"],a["finish"],a["deg"],b["tok"],b["finish"],b["deg"]))
for k,lab in ((0,"无 stop"),(1,"stop=<|im_end|>")):
    v=res[k]
    print("%-18s 输出 tok 中位=%.0f  finish=%s  deg 中位=%.2f  deg>0.3=%d/%d" % (
        lab, statistics.median(x["tok"] for x in v),
        dict((f,sum(1 for x in v if x["finish"]==f)) for f in {x["finish"] for x in v}),
        statistics.median(x["deg"] for x in v), sum(1 for x in v if x["deg"]>0.30), len(v)))
print()
print("无 stop 最长一条尾部 140:", repr(max(res[0],key=lambda x:x['tok'])['text'][-140:]))
print("有 stop 最长一条尾部 140:", repr(max(res[1],key=lambda x:x['tok'])['text'][-140:]))
