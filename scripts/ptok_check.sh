#!/usr/bin/env bash
# ptok_check.sh —— 从一行探针输出里取出 ptok 并判"上下文是否够长"
#
# 为什么单独成脚本：`degen_repro.sh` 只按 `命中=✓` 计分，**从不看 ptok** ⇒
# 若服务因 MAXLEN/截断只吃了 90K，repro 仍然会报"8/8 未复现"，而它测的根本不是 131K。
# 单独成脚本后可以被测试直接覆盖（见 scripts/test_ptok_check.sh）。
#
# 用法： bash scripts/ptok_check.sh "<一行探针输出>" <MIN_PTOK>
#   退出码 0 = ptok 达标；1 = ptok 不足；2 = 这一行里没有 ptok 字段（无证据）
set -uo pipefail
line=${1:-}
min=${2:-120000}

ptok=$(printf '%s' "$line" | sed -n 's/.*ptok=\([0-9][0-9]*\).*/\1/p')
if [ -z "${ptok:-}" ]; then
    printf '⛔ 无 ptok 字段（无法证明上下文长度）：%s\n' "${line:0:80}"
    exit 2
fi
if [ "$ptok" -lt "$min" ]; then
    printf '❌ ptok=%s < 门槛 %s ⇒ 这一条不算长上下文\n' "$ptok" "$min"
    exit 1
fi
printf '✅ ptok=%s ≥ %s\n' "$ptok" "$min"
exit 0
