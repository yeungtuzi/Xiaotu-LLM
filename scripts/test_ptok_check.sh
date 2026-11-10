#!/usr/bin/env bash
# test_ptok_check.sh —— ptok 门槛的三档行为（达标 / 不足 / 无字段）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHK="$HERE/ptok_check.sh"
fails=0
check() { # 名称 期望码 行 门槛
    out=$(bash "$CHK" "$3" "$4" 2>&1); rc=$?
    if [ "$rc" = "$2" ]; then printf '✅ %s（rc=%s）\n' "$1" "$rc"
    else printf '❌ %s：期望 rc=%s 实得 %s\n   %s\n' "$1" "$2" "$rc" "$out"; fails=$((fails+1)); fi
}
check "131K 达标" 0 "cached=0 ptok=131774 90.6s 命中=✗" 120000
check "90K 不足"  1 "cached=0 ptok=90000 60.0s 命中=✓" 120000
check "无 ptok"   2 "cached=0 90.6s 命中=✓" 120000
check "边界 120000" 0 "ptok=120000 命中=✓" 120000
check "边界 119999" 1 "ptok=119999 命中=✓" 120000
echo
if [ "$fails" -ne 0 ]; then echo "❌ $fails 个用例不符合预期"; exit 1; fi
echo "✅ 5/5 用例符合预期"
