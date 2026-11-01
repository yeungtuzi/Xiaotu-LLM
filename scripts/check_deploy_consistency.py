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
# (本地根, 远端根, 是否全量比对)
# ⛔ 第11轮 N1：上一版注释写"vLLM 本地镜像不是全量副本"——**这是假的**。
#    实测本地镜像就是全量树（2794 个 .py），且当时有 **12 个文件比远端旧**
#    （远端含我们自己的补丁 `[xtu-10]` / `XTU_VARLEN_K` / `_XTU_UNFUSED_NAMES`），
#    而本脚本仍报 45/45 绿 ⇒ 门禁绿建立在假前提上。
#    ⇒ 两棵树一律**全量比对**。
ROOTS = [
    ("/root/lvllm/data/repos/vllm-mainline", "/home/user/lvllm/process_data/ref/repos/vllm-mainline", True),
    ("/root/lvllm/repos/xiaotu-llm/vllm_xiaotu_moe", "/home/user/lvllm/vllm-xiaotu-moe/vllm_xiaotu_moe", True),
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
    for local, remote, _full in ROOTS:
        lp = os.path.join(local, rel)
        if os.path.exists(lp):
            return lp, os.path.join(remote, rel)
    return None, None


def main_verify_only(pairs, whole_pairs) -> int:
    """推送后**再校验一次**（避免"推完不验"）：重新执行本脚本但**不带 --deploy**。"""
    return subprocess.run([sys.executable, os.path.abspath(__file__)]).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="先校验，再推送**不一致**的文件")
    ap.add_argument("--force", action="store_true", help="允许用本地覆盖更旧的远端文件")
    a = ap.parse_args()

    pairs = []
    unresolved = []
    for rel in _scope():
        lp, rp = _resolve(rel)
        if lp is None:
            # ⛔ 第10轮：**解析不到的条目不许静默跳过**（旧版会打印 "✅ 一致：N/N"，分母悄悄变小）
            unresolved.append(rel)
            continue
        pairs.append((lp, rp))
    if unresolved:
        print(f"❌ scope 里有 {len(unresolved)} 个条目解析不到本地文件（分母不能悄悄变小）：")
        for rel in unresolved[:10]:
            print(f"   {rel}")
        return 1

    # ⛔ 第10轮：只比 scope 的 32 个文件不够 —— `hybrid_model.py` 已实测漂移而门禁报绿。
    #    ⇒ 额外对**整棵树**（两侧都存在的 *.py）做一次全量比对。
    whole = []
    for local, remote, full in ROOTS:
        if not full:
            continue   # vLLM 镜像不是全量副本 ⇒ 只比 scope（见 ROOTS 注释）
        for dirpath, _dirs, files in os.walk(local):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                lp = os.path.join(dirpath, fn)
                rel = os.path.relpath(lp, local)
                rp = os.path.join(remote, rel)
                whole.append((lp, rp))
    whole_pairs = [(lp, rp) for lp, rp in whole if lp not in {p[0] for p in pairs}]

    allpairs = pairs + whole_pairs
    remote = {}
    # ⛔ 命令行有长度上限 ⇒ 分批（每批 150 个）
    BATCH = 150
    for i in range(0, len(allpairs), BATCH):
        chunk = allpairs[i:i + BATCH]
        cmd = "md5sum " + " ".join(f'"{rp}"' for _, rp in chunk)
        try:
            r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print("❌ ssh 超时")
            return 2
        if r.returncode != 0 and not r.stdout:
            print(f"❌ ssh 失败: {r.stderr.strip()[:120]}")
            return 2
        for line in r.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                remote[parts[1].strip()] = parts[0]

    mismatch = []
    for lp, rp in allpairs:
        lm = hashlib.md5(open(lp, "rb").read()).hexdigest()
        rm = remote.get(rp)
        # ⛔ 第13轮：旧版只存 basename，推送时再按 basename 反查 ⇒
        #    全树 517 个 .py 有重名（`__init__.py` ×621）⇒ 推的是"第一个同名文件"，
        #    真正不一致的那个永远不推，还把计数报成成功。
        if rm is None:
            mismatch.append((os.path.basename(lp), lm[:8], "MISSING", lp, rp))
        elif rm != lm:
            mismatch.append((os.path.basename(lp), lm[:8], rm[:8], lp, rp))

    # ⛔ 第12轮：旧 `--deploy` 是**先推后验** ⇒ 会把本地旧文件覆盖远端补丁
    #    （第11轮根因的反向重演）。现在：**先比对，只推不一致项**；
    #    若远端该文件比本地**新**（mtime 更晚）⇒ 默认**拒绝推送**（除非 --force）。
    if a.deploy and mismatch:
        pushed = refused = 0
        for name, lm, rm, lp, rp in mismatch:
            if not a.force:
                try:
                    rmt = subprocess.run(
                        SSH + [f'stat -c %Y "{rp}"'], capture_output=True, text=True, timeout=30)
                    if rmt.returncode != 0 or not rmt.stdout.strip():
                        raise RuntimeError(rmt.stderr.strip()[:80] or "stat 无输出")
                    rtime = int(rmt.stdout.strip())
                except Exception as _e:  # noqa: BLE001
                    # ⛔ 第13轮：异常时旧版 rtime=0 ⇒ 恒放行（fail-open）。改为拒绝。
                    print(f"⛔ 拒绝推送 {name}：无法取远端 mtime（{_e}）⇒ fail-closed")
                    refused += 1
                    continue
                if rtime > os.path.getmtime(lp):
                    print(f"⛔ 拒绝推送 {name}：远端更新（远端 mtime {rtime} > 本地 "
                          f"{int(os.path.getmtime(lp))}）⇒ 用 --force 才覆盖")
                    refused += 1
                    continue
            r = subprocess.run(SCP + [lp, "gpu:" + os.path.dirname(rp) + "/"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                pushed += 1
            else:
                print(f"❌ 推送失败 {name}: {r.stderr.strip()[:100]}")
        print(f"推送：成功 {pushed}，拒绝 {refused}")
        if pushed:
            return main_verify_only(pairs, whole_pairs)

    _remote_only_reported = False
    # ⛔ 第12轮：旧版只枚举**本地**文件 ⇒ 远端独有文件永远看不见。
    # ⛔ 第13轮：该检测曾被 `if mismatch: return 1` 早退挡在前面 ⇒ 只要有任何不一致就永不报告；
    #    且 `find` 返回码未检查 ⇒ 失败即 rset 空 ⇒ 静默"✅ 一致"。
    remote_only = []
    for _local, _remote, _full in ROOTS:
        r = subprocess.run(SSH + [f'find "{_remote}" -name "*.py" -printf "%P\\n"'],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"❌ 无法枚举远端 {_remote}（find 返回 {r.returncode}）：{r.stderr.strip()[:100]}")
            return 2
        rset = {x.strip() for x in r.stdout.splitlines() if x.strip()}
        lset = set()
        for dirpath, _d, files in os.walk(_local):
            for fn in files:
                if fn.endswith(".py"):
                    lset.add(os.path.relpath(os.path.join(dirpath, fn), _local))
        for x in sorted(rset - lset):
            remote_only.append(os.path.join(_remote, x))

    if remote_only:
        print(f"⚠️ 远端独有（本地没有，可能是在远端直接改的）：{len(remote_only)} 个，示例：")
        for x in remote_only[:5]:
            print(f"   {x}")

    if mismatch:
        print(f"❌ 本地镜像与远端部署树**不一致**：{len(mismatch)}/{len(allpairs)}")
        for name, lm, rm, _lp, _rp in mismatch[:15]:
            print(f"   {name}: 本地={lm} 远端={rm}")
        print("\n⇒ 任何门禁结论在被审对象上**不可信**。先跑 --deploy 再重跑门禁。")
        return 1



    print(f"✅ 一致：{len(allpairs)}/{len(allpairs)} 文件 md5 相同"
          f"（scope {len(pairs)} + 全树其余 {len(whole_pairs)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
