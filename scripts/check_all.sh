#!/usr/bin/env bash
# 一键跑全部机械门禁（提交前/pre-commit 都走这里）。
set -o pipefail
cd "$(dirname "$0")/.." || exit 2
rc=0
echo "=== 1/5 内核资源门禁 ==="
python3 scripts/check_kernel_alloc_rules.py || rc=1
echo "=== 2/5 门禁自检（含源码反回归） ==="
python3 scripts/test_kernel_alloc_rules.py || rc=1
echo "=== 3/5 局部变量承载扫描（门禁对此盲） ==="
python3 scripts/scan_stride_relays.py || rc=1
echo "=== 4/5 门禁盲区扫描（别名/不可达同步/同名内核/@jit 拼写） ==="
python3 scripts/scan_gate_blindspots.py || rc=1
echo "=== 5/5 部署一致性（本地镜像 vs 远端） ==="
python3 scripts/check_deploy_consistency.py || rc=1
exit $rc
