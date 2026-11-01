#!/usr/bin/env python3
"""校验「本地镜像 == 远端部署树」—— 任何"门禁绿"都必须以此为前提。

⛔ 来由（2026-11-01 第 41 轮）：所有 scp 都带 `>/dev/null` 忽略退出码 ⇒ 部署**静默失败**，
   而门禁审的是远端树 ⇒ 连续数轮在"半旧对象"上判绿/判红，51 条里 43 条是假象。

用法：
  python3 scripts/check_deploy_consistency.py            # 校验并打印
  python3 scripts/check_deploy_consistency.py --deploy   # 先推送再校验
退出码：0=一致；1=不一致；2=无法连接
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SSH = ["ssh", "-F", "/root/lvllm/tmp/sshcfg", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "gpu"]
SCP = ["scp", "-F", "/root/lvllm/tmp/sshcfg", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

# (本地相对路径, 远端目录)
ROOTS = [
    ("/root/lvllm/data/repos/vllm-mainline", "/home/user/lvllm/process_data/ref/repos/vllm-mainline"),
    ("/root/lvllm/repos/xiaotu-llm/vllm_xiaotu_moe", "/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe"),
]


def _scope() -> list[str]:
    p = os.path.join(HERE, "kernel_audit_scope.txt")
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _resolve(rel: str):
    # scope 里插件文件写成 `plugin:vllm_xiaotu_moe/xxx.py`
    if rel.startswith("plugin:"):
        rel = rel[len("plugin:"):]
        # scope 里写成 `plugin:vllm_xiaotu_moe/xxx.py`，而 ROOTS[1] 已指向该包目录
        if rel.startswith("vllm_xiaotu_moe/"):
            rel = rel[len("vllm_xiaotu_moe/"):]
    for local, remote in ROOTS:
        lp = os.path.join(local, rel)
        if os.path.exists(lp):
            return lp, os.path.join(remote, rel)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="先推送再校验")
    a = ap.parse_args()

    pairs = []
    for rel in _scope():
        lp, rp = _resolve(rel)
        if lp is None:
            print(f"⚠️ 本地找不到：{rel}")
            continue
        pairs.append((lp, rp))

    if a.deploy:
        bad = 0
        for lp, rp in pairs:
            r = subprocess.run(SCP + [lp, "gpu:" + os.path.dirname(rp) + "/"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                print(f"❌ 推送失败 {os.path.basename(lp)}: {r.stderr.strip()[:100]}")
                bad += 1
        print(f"推送：{len(pairs) - bad}/{len(pairs)} 成功")
        if bad:
            return 1

    cmd = "md5sum " + " ".join(f'"{rp}"' for _, rp in pairs)
    try:
        r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("❌ ssh 超时")
        return 2
    if r.returncode != 0:
        print(f"❌ ssh 失败: {r.stderr.strip()[:120]}")
        return 2

    remote = {}
    for line in r.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            remote[parts[1].strip()] = parts[0]

    mismatch = []
    for lp, rp in pairs:
        lm = hashlib.md5(open(lp, "rb").read()).hexdigest()
        rm = remote.get(rp)
        if rm is None:
            mismatch.append((os.path.basename(lp), lm[:8], "MISSING"))
        elif rm != lm:
            mismatch.append((os.path.basename(lp), lm[:8], rm[:8]))

    if mismatch:
        print(f"❌ 本地镜像与远端部署树**不一致**：{len(mismatch)}/{len(pairs)}")
        for name, lm, rm in mismatch[:15]:
            print(f"   {name}: 本地={lm} 远端={rm}")
        print("\n⇒ 任何门禁结论在被审对象上**不可信**。先跑 --deploy 再重跑门禁。")
        return 1

    print(f"✅ 一致：{len(pairs)}/{len(pairs)} 文件 md5 相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
