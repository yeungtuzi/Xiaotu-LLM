#!/usr/bin/env python3
"""退化计量器：读 DSH 会话日志，统计"工具调用缺必填字段"的比率，按 provider 分组。

为什么看这个指标（2026-11-06 实测）：
  同一会话内，deepseek-official 0/417，本地引擎 42 次且随调用数单调上升
  （601-700 为 1%、701-800 为 3%、801-900 为 38%）——
  这是退化最容易观测、且不需要发任何请求的指纹。

用法：
  python3 scripts/deg_meter.py                # 全部会话汇总
  python3 scripts/deg_meter.py --detail       # 逐会话明细
  python3 scripts/deg_meter.py --window 100   # 指定窗口大小
"""
import argparse, glob, json, os, re, subprocess, sys
from collections import defaultdict

SESS = os.path.expanduser("~/.dsh/sessions/*/*/session.v4.jsonl.zstd")
# 各工具名 -> 必填字段集合（按 DSH 实际 schema，缺一即计为一次违规）
REQUIRED = {
    "bash": {"description", "command"},
    "run_command": {"description", "command"},
    "read": {"file_path"},
    "write": {"file_path", "content"},
    "edit": {"file_path", "old_string", "new_string"},
    "glob": {"pattern"},
    "grep": {"pattern"},
}

def scan(path):
    try:
        raw = subprocess.run(["zstd", "-dc", path], capture_output=True, timeout=180).stdout
    except Exception:
        return None
    prov = "?"
    rows = []
    for line in raw.decode("utf-8", "replace").split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("type"); d = r.get("data") or {}
        if t == "request/header":
            m = re.search(r"'provider': '([^']*)'", str(d.get("header")))
            if m:
                prov = m.group(1)
        elif t == "tool/call":
            name = d.get("name") or ""
            try:
                a = json.loads(d.get("arguments") or "{}")
            except Exception:
                a = {}
            req = REQUIRED.get(name)
            bad = bool(req) and isinstance(a, dict) and not req.issubset(a.keys())
            rows.append((prov, bad))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    per_prov = defaultdict(lambda: [0, 0])          # prov -> [calls, bad]
    detail = []
    for f in sorted(glob.glob(SESS)):
        rows = scan(f)
        if not rows:
            continue
        d = defaultdict(lambda: [0, 0])
        for p, bad in rows:
            per_prov[p][0] += 1; per_prov[p][1] += 1 if bad else 0
            d[p][0] += 1; d[p][1] += 1 if bad else 0
        detail.append((os.path.basename(os.path.dirname(f))[:16], dict(d), len(rows)))

    print("%-22s %10s %8s %9s" % ("provider", "工具调用", "违规", "违规率"))
    for p, (n, b) in sorted(per_prov.items(), key=lambda kv: -kv[1][0]):
        if n < 20:
            continue
        print("%-22s %10d %8d %8.2f%%" % (p, n, b, 100.0*b/n))

    if args.detail:
        print("\n逐会话（仅列有违规的）:")
        for sid, d, n in sorted(detail, key=lambda x: -sum(v[1] for v in x[1].values())):
            tot = sum(v[1] for v in d.values())
            if tot == 0:
                continue
            parts = ["%s %d/%d" % (p, v[1], v[0]) for p, v in d.items()]
            print("  %-18s %s" % (sid, "  ".join(parts)))

if __name__ == "__main__":
    main()
