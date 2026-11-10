#!/usr/bin/env bash
# test_d1_state.sh —— D1 阶段判据的四组合 + 用法错误
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fails=0
run() { out=$(bash "$HERE/check_d1_state.sh" "$1" "$2" 2>&1); rc=$?
    if [ "$rc" = "$3" ]; then printf '✅ 标记=%s expect=%s ⇒ rc=%s\n' "$1" "$2" "$rc"
    else printf '❌ 标记=%s expect=%s 期望 rc=%s 实得 %s\n%s\n' "$1" "$2" "$3" "$rc" "$out"; fails=$((fails+1)); fi; }
run 0 0 0
run 2 0 1
run 1 1 0
run 0 1 1
run "" 0 2
run "abc" 1 2
run 1 7 2
echo
[ "$fails" -eq 0 ] && { echo "✅ 7/7 用例符合预期"; exit 0; }
echo "❌ $fails 个不符合预期"; exit 1
