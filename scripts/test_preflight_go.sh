#!/usr/bin/env bash
# test_preflight_go.sh —— 总闸的离线冒烟：DRY_RUN 必须 GO，且三项本地判据真的被判过
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out=$(DRY_RUN=1 bash "$HERE/preflight_go.sh" 2>&1); rc=$?
fails=0
chk() { if printf '%s' "$out" | grep -q "$1"; then printf '✅ %s\n' "$2"; else printf '❌ %s\n' "$2"; fails=$((fails+1)); fi; }
[ "$rc" = 0 ] && echo "✅ DRY_RUN 退出码 0" || { echo "❌ DRY_RUN 退出码 $rc"; fails=$((fails+1)); }
chk "判据体系自检" "含『判据体系自检』一项"
chk "仪器补丁存在且 md5 正确" "含『仪器补丁』一项"
chk "D1 补丁 md5 白名单" "含『D1 补丁』一项"
chk "结论：GO" "结论为 GO"
chk "跳过" "远端项被跳过（DRY_RUN）"
# 反向：把自检打断（临时把某个脚本改坏）应得 NO-GO —— 用 PATH 注入一个假 bash 太脆，改为断言"缺文件会 NO-GO"
tmp=$(mktemp -d); mkdir -p "$tmp/scripts" "$tmp/patches/diag" "$tmp/patches/replay"
cp "$HERE/preflight_go.sh" "$tmp/scripts/"; cp "$HERE/run_harness_selftests.sh" "$tmp/scripts/"
out2=$(DRY_RUN=1 bash "$tmp/scripts/preflight_go.sh" 2>&1); rc2=$?
[ "$rc2" = 1 ] && echo "✅ 缺补丁时 NO-GO（rc=1）" || { echo "❌ 缺补丁时应 rc=1，实得 $rc2"; fails=$((fails+1)); }
rm -rf "$tmp"
echo
[ "$fails" -eq 0 ] && { echo "✅ 全部冒烟通过"; exit 0; }
echo "❌ $fails 项不符合预期"; exit 1
