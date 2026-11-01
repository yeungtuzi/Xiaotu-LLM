#!/usr/bin/env bash
# 一键跑全部机械门禁（提交前/pre-commit 都走这里）。
set -o pipefail
cd "$(dirname "$0")/.." || exit 2
rc=0
echo "=== 1/3 内核资源门禁 ==="
python3 scripts/check_kernel_alloc_rules.py || rc=1
echo "=== 2/3 门禁自检（含源码反回归） ==="
python3 scripts/test_kernel_alloc_rules.py || rc=1
echo "=== 3/3 部署一致性（本地镜像 vs 远端） ==="
python3 scripts/check_deploy_consistency.py || rc=1
exit $rc
