#!/usr/bin/env python3
"""退化扫描(两个新判据,只读 ✓):① 客观路径判据 ② 行为重复率"""
import json, os, re, sys, time, zstandard as z
from collections import Counter
base = os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
PATH = re.compile(r"/home/user/[A-Za-z0-9_.\-/]{3,}")
SKIP = ("re.compile", "CORRUPT=", "_degen_scan")
def scan(sid):
    f = os.path.join(base, sid, "session.v4.jsonl.zstd")
    if not os.path.exists(f): return None
    with open(f, "rb") as fh: d = z.ZstdDecompressor().stream_reader(fh).read()
    rr = [json.loads(l) for l in d.decode("utf-8", "replace").split("\n") if l.strip()]
    calls = []
    for r in rr:
        if r.get("type") != "tool/call": continue
        try: a = json.loads((r.get("data") or {}).get("arguments") or "{}")
        except Exception: a = {}
        c = str(a.get("command", "") or a.get("file_path", "") or a.get("pattern", ""))
        if any(k in c for k in SKIP): continue
        if c: calls.append(c)
    if not calls: return None
    # ① 客观判据:命令里的 /home/user 路径是否真实存在(排除将要创建的)
    badpath = 0
    for c in calls:
        if re.search(r"mkdir|touch|tee|>\s*/home|cp .* /home|mv .* /home", c): continue
        for p in PATH.findall(c):
            pp = p.rstrip("/.,);")
            if "*" in pp or "$" in pp or "{" in pp: continue
            if not os.path.exists(pp): badpath += 1; break
    # ② 行为重复率
    norm = [re.sub(r"\s+", " ", c).strip()[:70] for c in calls]
    dup = 1 - len(set(norm)) / max(1, len(norm))
    top = Counter(norm).most_common(1)
    return dict(calls=len(calls), badpath=badpath,
                badpath_pct=100*badpath/len(calls), dup_pct=100*dup,
                top=top[0] if top else ("", 0), age=int(time.time()-os.path.getmtime(f)))
if __name__ == "__main__":
    import glob
    rows = []
    for d in glob.glob(base + "/*"):
        sid = os.path.basename(d)
        if sid.startswith("session-ecd2c6e6"): continue
        m = scan(sid)
        if m: rows.append((m["dup_pct"], sid, m))
    rows.sort(reverse=True)
    print("  %-44s %-6s %-10s %-10s %s" % ("会话", "调用", "路径坏%", "重复率%", "最高重复项"))
    for dup, sid, m in rows[:8]:
        print("  %-44s %-6d %-10.2f %-10.1f %s×%d" % (
            sid[:44], m["calls"], m["badpath_pct"], m["dup_pct"], m["top"][0][:30], m["top"][1]))
