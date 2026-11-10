#!/usr/bin/env bash
# test_report_check.sh —— check_report_complete.sh 的三档行为
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fails=0

mk_report() { # $1=文件 $2=状态行 $3=是否含待填 $4=证据
    {
        echo "# 报告"
        echo "$2"
        echo "## 6. 验收结果"
        echo "| 项 | 结果 | 证据文件 |"
        echo "|---|---|---|"
        if [ "$3" = "pending" ]; then
            echo "| 8×长请求 | 【待填】 | 【待填】|"
        else
            echo "| 8×长请求 | ✅ 8/8 | $4 |"
        fi
        echo ""
        echo "## 7. 复现"
    } > "$1"
}

run() { out=$(bash "$HERE/check_report_complete.sh" "$1" 2>&1); rc=$?; printf '%s\n' "$out" > "$TMP/last.out"; echo "$rc"; }

mk_report "$TMP/r1.md" "> 状态：**进行中**（数据待填）" pending "-"
rc=$(run "$TMP/r1.md")
[ "$rc" = 1 ] && echo "✅ 进行中 + 待填 ⇒ rc=1" || { echo "❌ 用例1 rc=$rc"; fails=$((fails+1)); }

mk_report "$TMP/r2.md" "> 状态：**已验收**" done "dev-docs/report/tuning/logs/x.txt"
rc=$(run "$TMP/r2.md")
[ "$rc" = 0 ] && echo "✅ 已验收 + 有证据 ⇒ rc=0" || { echo "❌ 用例2 rc=$rc"; fails=$((fails+1)); }

mk_report "$TMP/r3.md" "> 状态：**已验收**" done "无"
rc=$(run "$TMP/r3.md")
[ "$rc" = 1 ] && echo "✅ 已验收但证据列无路径 ⇒ rc=1" || { echo "❌ 用例3 rc=$rc"; fails=$((fails+1)); }

echo
[ "$fails" -eq 0 ] && { echo "✅ 3/3 用例符合预期"; exit 0; }
echo "❌ $fails 个用例不符合预期"; exit 1
