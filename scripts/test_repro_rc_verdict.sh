#!/usr/bin/env bash
# test_repro_rc_verdict.sh —— rc 语义翻译表（重点是 rc=2 必须中止）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fails=0
chk() { # 名称 rc mode 期望rc 期望文案
    out=$(bash "$HERE/repro_rc_verdict.sh" "$2" "$3" 2>&1); rc=$?
    if [ "$rc" = "$4" ] && printf '%s' "$out" | grep -q -- "$5"; then
        printf '✅ %s（rc=%s）\n' "$1" "$rc"
    else
        printf '❌ %s：期望 rc=%s 含「%s」，实得 rc=%s / %s\n' "$1" "$4" "$5" "$rc" "$out"; fails=$((fails+1))
    fi
}
chk "口径不符(rc=2, base) ⇒ 中止" 2 base 2 "口径不符"
chk "口径不符(rc=2, fix) ⇒ 中止"  2 fix  2 "口径不符"
chk "基线复现(rc=1) ⇒ 继续"        1 base 0 "基线成立"
chk "修后仍失败(rc=1) ⇒ 继续"      1 fix  0 "修复未生效"
chk "基线未复现(rc=0) ⇒ 继续+警告" 0 base 0 "未复现"
chk "修后全过(rc=0) ⇒ 继续"        0 fix  0 "8/8 全过"
chk "未知码(rc=9) ⇒ 继续+警告"     9 base 0 "未知"
echo
[ "$fails" -eq 0 ] && { echo "✅ 7/7 用例符合预期"; exit 0; }
echo "❌ $fails 个用例不符合预期"; exit 1
