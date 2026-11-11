#!/usr/bin/env bash
# discriminate_needle_scale.sh —— 判别"长上下文失败"是引擎缺陷还是模型自身不稳定
#
# 背景（台账 #277）：E1/E2 的 8×131K 失败是**随机单条**，不像历史描述的"第 8 条起错"。
# 若不先分清下面两种可能，就会把模型噪声当成"修复无效"，在重启上无限打转：
#   A. 引擎 KV/回放缺陷   ⇒ 短上下文同题 8/8、长上下文失败
#   B. 模型自身检索不稳   ⇒ 短上下文同题也偶发失败
#
# 用法： bash scripts/discriminate_needle_scale.sh [PORT]
# 退出码：0 = 短 8/8 且长 8/8；3 = 短上下文也失败（⇒ 模型噪声/判据问题）；4 = 短 8/8、长有失败（⇒ 查引擎）
set -uo pipefail
PORT="${1:-${PORT:-8700}}"
PY_BIN="${PY_BIN:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUNDS="${ROUNDS:-8}"
SHORT_N="${SHORT_N:-60}"
LONG_N="${LONG_N:-7430}"

run_arm() {  # $1=n  $2=label  ⇒ 打印通过数
    local n=$1 label=$2 ok=0 i out
    for i in $(seq 1 "$ROUNDS"); do
        out=$(PORT="$PORT" "$PY_BIN" "$HERE/probe_dump_long.py" --n "$n" --uuid "${label}${i}$(date +%s%N | tail -c 6)" --maxtok 2048 2>&1 | tail -1)
        if printf '%s' "$out" | grep -q "命中=✓"; then ok=$((ok+1)); fi
        printf '  [%s %d/%d] %s\n' "$label" "$i" "$ROUNDS" "$(printf '%s' "$out" | cut -c1-110)" >&2
    done
    echo "$ok"
}

echo "== 短上下文 N=$SHORT_N（≈$((SHORT_N*18)) tokens）x $ROUNDS"
s=$(run_arm "$SHORT_N" short | tail -1)
echo "== 长上下文 N=$LONG_N（≈131K tokens）x $ROUNDS"
l=$(run_arm "$LONG_N" long | tail -1)
echo
echo "短=$s/$ROUNDS   长=$l/$ROUNDS"
if [ "${s:-0}" -lt "$ROUNDS" ]; then
    echo "⇒ 判定 3：**短上下文也失败** ⇒ 不是引擎路径问题；先修判据（解码参数/题目形式），别急着改引擎"
    exit 3
fi
[ "${l:-0}" -lt "$ROUNDS" ] && { echo "⇒ 判定 4：短 8/8、长有失败 ⇒ 指向引擎的长上下文路径（回放/pad/indexer）"; exit 4; }
echo "⇒ 判定 0：两档都 8/8 ⇒ 本轮长上下文无失败，需更多轮次才能区分"
exit 0
