#!/usr/bin/env python3
"""验收回归：长会话退化的三条判据 + 最小复现（2026-11-12）。

对应目标里的验收项：

* **A 同 prompt ×10 输出签名唯一**：同一 prompt 连发 10 次，10 次输出必须完全一致且都命中针；
* **B 交错 shared/unique 四臂 unique 命中率 100%**：shared / unique 交错发，
  每一条 unique 都必须命中，shared 亦不得崩坏；
* **C 142K 针尖 >4h 后 ≥2/3**（可选，`--soak-hours N`）：先灌 N 小时负载（可 `0` 表示跳过预灌），
  再连发 3 条 142K，要求 ≥2 条命中；
* **D 最小复现 8×131K 连发 8/8**。

⚠️ 本脚本**只探测已在运行的服务**（不发 start/stop），因此不涉及起停脚本审计。
服务必须已就绪（`/health` 返回 200），默认端口 8700。

⛔ **必须在 GPU 执行机上跑**（脚本连的是 `127.0.0.1:<port>`）：
本机（harness VM）没有服务、也没有到远端端口的隧道，在本地跑只会得到“服务未就绪”。

```bash
scp scripts/accept_degen.py gpu:/home/user/lvllm/vllm-xiaotu-moe/scripts/
ssh gpu 'cd /home/user/lvllm/vllm-xiaotu-moe && python3 scripts/accept_degen.py --checks A --n 200'
```

用法：

```bash
bash scripts/serve_ds41.sh status TAG=prod              # 确认在跑
python3 scripts/accept_degen.py                         # 跑 A B D
python3 scripts/accept_degen.py --checks A,D --n 7430
python3 scripts/accept_degen.py --checks C --soak-hours 4
```

退出码：0 = 所选判据全部通过；1 = 有判据未通过；2 = 环境问题（服务未就绪）。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe_dump_long.py")
NEEDLE = 740317
LINE_RE = re.compile(r"cached=(\S+) ptok=(\S+) ([\d.]+)s 命中=(✓|✗) 期望=(\d+) "
                     r"实得=(\[[^\]]*\]) +finish=(\S+) ctok=(\S+) rzlen=(\d+)")
SIG_RE = re.compile(r" sig=([0-9a-f]{16})")


def health(port: int) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status
    except Exception:      # noqa: BLE001
        return 0


def probe(port: int, n: int, uuid: str, maxtok: int) -> dict:
    """跑一条探针请求，返回解析后的结果。"""
    env = dict(os.environ, PORT=str(port))
    out = subprocess.run(
        [sys.executable, "-u", PROBE, "--n", str(n), "--uuid", uuid,
         "--maxtok", str(maxtok), "--sig"],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    line = (out.stdout or "").strip().splitlines()[-1] if out.stdout.strip() else ""
    m = LINE_RE.search(line)
    if m is None and line:
        print(f"  [warn] 探针输出无法解析：{line[:120]}", flush=True)
    rec = {"raw": line, "hit": bool(m and m.group(4) == "✓"),
           "ptok": m.group(2) if m else "?", "signature": None}
    # 签名用「实得 + 输出前 60 字」—— 同 prompt 十连发时要求完全一致
    sm = SIG_RE.search(line)
    if sm:
        # 优先用完整回复文本的 sha256（真正代表“输出内容”）
        rec["signature"] = sm.group(1)
    elif m:
        # 退化路径：至少用「实得 + finish」（不含 ctok —— ctok 会随推理长度合理变化）
        rec["signature"] = (m.group(6), m.group(7))
    return rec


def check_A(port: int, n: int, maxtok: int, rounds: int = 1) -> tuple[bool, str]:
    """同 prompt ×10，可多轮采样：每一轮都必须 10/10 命中且签名唯一。

    ⚠️ 为什么默认只跑 1 轮、却支持 `--a-rounds`：台账 #433 实测 A 有约 1/8 的**偶发**
    非确定性（'DET' 已修掉主因，残余疑为 top-k 并列取舍）。单轮通过不足以证明“唯一”，
    判验收时建议 `--a-rounds 3`（三轮全绿才算过）。
    """
    sigs, hits = [], 0
    per_round = []
    for _ in range(max(1, rounds)):
        rsigs, rhits = [], 0
        for _ in range(10):
            r = probe(port, n, "same-prompt-A", maxtok)
            rhits += int(r["hit"])
            rsigs.append(r["signature"])
        hits += rhits
        sigs += rsigs
        per_round.append((rhits, len(set(rsigs))))
    total = 10 * max(1, rounds)
    all_rounds_unique = all(u == 1 and h == 10 for h, u in per_round)
    ok = hits == total and all_rounds_unique
    detail = "；".join(f"第{i+1}轮 {h}/10 签名 {u} 种" for i, (h, u) in enumerate(per_round))
    return ok, f"A 同 prompt ×10×{max(1, rounds)}：命中 {hits}/{total}，{detail}（要求每轮 1 种）"


def check_D(port: int, n: int, maxtok: int) -> tuple[bool, str]:
    """8×131K 背靠背（不同 uuid）：要求 8/8 命中。"""
    hits = 0
    for i in range(8):
        r = probe(port, n, f"degen-8x-{i}", maxtok)
        hits += int(r["hit"])
    return hits == 8, f"D 8×131K 连发：命中 {hits}/8"


def check_B(port: int, n: int, maxtok: int) -> tuple[bool, str]:
    """交错 shared / unique ×2 轮：unique 必须 100% 命中。"""
    unique_hits = unique_total = 0
    shared_hits = shared_total = 0
    for rnd in range(2):
        r = probe(port, n, "shared-B", maxtok)
        shared_total += 1
        shared_hits += int(r["hit"])
        for k in range(2):
            r = probe(port, n, f"unique-B-{rnd}-{k}", maxtok)
            unique_total += 1
            unique_hits += int(r["hit"])
    ok = unique_hits == unique_total and shared_hits == shared_total
    return ok, (f"B 交错四臂：unique {unique_hits}/{unique_total}，"
                f"shared {shared_hits}/{shared_total}")


def check_C(port: int, n: int, maxtok: int, soak_hours: float, soak_n: int) -> tuple[bool, str]:
    """142K 针尖：先灌 soak_hours 小时负载，再连发 3 条，要求 ≥2 命中。"""
    if soak_hours > 0:
        deadline = time.time() + soak_hours * 3600
        k = 0
        while time.time() < deadline:
            probe(port, soak_n, f"soak-{k}", max(64, maxtok // 8))
            k += 1
        print(f"  [C] 预灌完成：{k} 条", flush=True)
    hits = 0
    for i in range(3):
        r = probe(port, n, f"needle-C-{i}", maxtok)
        hits += int(r["hit"])
    return hits >= 2, f"C 142K 针尖 ×3：命中 {hits}/3（要求 ≥2）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8700")))
    ap.add_argument("--checks", default="A,B,D",
                    help="逗号分隔：A / B / C / D（默认 A,B,D）")
    ap.add_argument("--n", type=int, default=7430, help="探针 --n（7430 ≈ 131.8K token）")
    ap.add_argument("--maxtok", type=int, default=512)
    ap.add_argument("--a-rounds", type=int, default=1,
                    help="A 判据采样轮数（验收建议 3；见台账 #433 的偶发非确定性）")
    ap.add_argument("--soak-hours", type=float, default=0.0)
    ap.add_argument("--soak-n", type=int, default=2000)
    a = ap.parse_args()

    want = [c.strip().upper() for c in a.checks.split(",") if c.strip()]

    # ⚠️ 污染防护（台账 #399 的教训）：D/C 是“失败索引/失败率”类判据，
    #    必须在**重启后的第一个**跑；若同一次调用里排在 A/B 之后，先给出明确提示，
    #    避免把“实例已有存量”误读成“这个改动让退化变严重”。
    failure_index_checks = [c for c in want if c in ("C", "D")]
    earlier = [c for c in want[: want.index(failure_index_checks[0])]] if failure_index_checks else []
    if failure_index_checks and earlier:
        print(f"⚠️ 顺序提醒：{failure_index_checks[0]} 排在 {','.join(earlier)} 之后。")
        print("   若前面已发过成规模的长请求，实例会带有累计 token 存量，D/C 的失败点可能提前；")
        print("   结论只在『重启后第一个跑』时可比（见台账 #399）。")
    if health(a.port) != 200:
        print(f"⛔ 服务未就绪（port={a.port} /health 非 200）—— 先用 serve_ds41.sh 起服务")
        return 2


    results = []
    t0 = time.time()
    for c in want:
        if c == "A":
            results.append(check_A(a.port, a.n, a.maxtok, a.a_rounds))
        elif c == "B":
            results.append(check_B(a.port, a.n, a.maxtok))
        elif c == "C":
            results.append(check_C(a.port, a.n, a.maxtok, a.soak_hours, a.soak_n))
        elif c == "D":
            results.append(check_D(a.port, a.n, a.maxtok))
        else:
            print(f"⛔ 未知判据 {c}")
            return 2

    print()
    print(f"== 验收结果（用时 {(time.time() - t0) / 60:.1f} 分钟，端口 {a.port}）")
    all_ok = True
    for ok, msg in results:
        print(("✅ " if ok else "❌ ") + msg)
        all_ok = all_ok and ok
    print("== 汇总：" + ("全部通过" if all_ok else "存在未通过项"))
    return 0 if all_ok else 1



if __name__ == "__main__":
    raise SystemExit(main())
