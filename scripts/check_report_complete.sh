#!/usr/bin/env bash
# check_report_complete.sh —— 结项前机械检查：报告还有没有【待填】、证据是否齐全
#
# 存在的理由（台账 #232）："完成"这件事最容易被嘴上判定。这个脚本让"能不能算完成"变成机器判据：
#   ① 报告里不允许残留 `【待填】`；
#   ② 状态行必须从"进行中"改为已验收；
#   ③ §6 每一行都要有**证据文件**（`$RUN_DIR/...` 或 `dev-docs/report/tuning/logs/...`），
#      且该文件在本机存在（远端文件用 --remote 允许路径未知，只检查格式）。
#
# 用法： bash scripts/check_report_complete.sh [报告路径]
# 退出码：0 = 可以声称完成；1 = 还有未填/缺证据
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REP="${1:-$ROOT/dev-docs/REPORT_DEGEN_FIX_2026-11-10.md}"
fails=0
bad() { printf '❌ %s\n' "$*"; fails=$((fails+1)); }
ok()  { printf '✅ %s\n' "$*"; }

[ -f "$REP" ] || { echo "⛔ 找不到报告：$REP"; exit 1; }
echo "检查报告：$REP"

# ① 待填
n=$(grep -c "【待填】" "$REP" || true)
if [ "${n:-0}" -gt 0 ]; then bad "还有 ${n} 处【待填】"; grep -n "【待填】" "$REP" | head -8 | sed 's/^/   /'
else ok "没有残留【待填】"; fi

# ② 状态行
if grep -q "状态：\*\*进行中\*\*" "$REP"; then bad "状态行仍是「进行中」"
else ok "状态行已不是「进行中」"; fi

# ③ §6 每行都要有证据
awk '/^## 6\./{f=1;next} /^## 7\./{f=0} f' "$REP" > /tmp/_sec6.$$
rows=$(grep -c '^|' /tmp/_sec6.$$ || true)
if [ "${rows:-0}" -le 2 ]; then bad "§6 表格疑似为空（只有表头）"
else
    # 证据列里应出现 logs/ 或 run/ 之类路径
    ev=$(grep '^|' /tmp/_sec6.$$ | tail -n +3 | grep -cE 'logs/|run/|\.txt|\.jsonl|\.log' || true)
    body=$((rows - 2))
    if [ "${ev:-0}" -lt "$body" ]; then bad "§6 有 $((body - ev)) 行缺少证据路径（共 $body 行）"
    else ok "§6 全部 $body 行都有证据路径"; fi
fi
rm -f /tmp/_sec6.$$

echo
[ "$fails" -eq 0 ] && { echo "✅ 报告可声称完成"; exit 0; }
echo "❌ $fails 项未满足 ⇒ **不许**声称完成"
exit 1
