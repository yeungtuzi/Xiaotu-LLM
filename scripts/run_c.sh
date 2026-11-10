#!/usr/bin/env bash
# =============================================================================
# run_c.sh —— 验收 C（142K 针尖 >4h ≥2/3）的**后台**启动器（不阻塞前台）
#
# 用法：
#   bash scripts/run_c.sh [PORT=8700] [HOURS=6] [INTERVAL=10] [ROUNDS=3]
# 产物：
#   dev-docs/report/tuning/logs/needle_c_<时间戳>.jsonl   逐次结果
#   dev-docs/report/tuning/logs/needle_c_<时间戳>.log     启动器自己的输出
# 说明：
#   * 用 setsid nohup 起，**立即返回**；之后任意时刻用 --aggregate 读结论；
#   * 不启停服务：C 的前提是"服务一直在线"，所以它只发请求。
# =============================================================================
set -uo pipefail
PORT="${PORT:-8700}"
HOURS="${HOURS:-6}"
INTERVAL="${INTERVAL:-10}"
ROUNDS="${ROUNDS:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-$HERE/../dev-docs/report/tuning/logs}"
mkdir -p "$LOGDIR" || { echo "[c] ⛔ 建不了日志目录 $LOGDIR" >&2; exit 2; }
TS="$(date +%Y%m%d-%H%M%S)"
JSONL="$LOGDIR/needle_c_${TS}.jsonl"
OUT="$LOGDIR/needle_c_${TS}.log"

if ! curl -s -o /dev/null --max-time 4 "http://127.0.0.1:${PORT}/health"; then
    echo "[c] ⛔ 端口 ${PORT} 没有服务 ⇒ 先起服务再做验收 C" >&2
    exit 2
fi

setsid nohup python3 "$HERE/long_uptime_needle.py" \
    --port "$PORT" --hours "$HOURS" --interval-min "$INTERVAL" --rounds "$ROUNDS" \
    --tokens 142000 > "$JSONL" 2> "$OUT" < /dev/null &
pid=$!
echo "[c] ✅ 已后台启动 pid=$pid"
echo "[c]    数据：$JSONL"
echo "[c]    日志：$OUT"
echo "[c]    预计结束：$(date -d "+${HOURS} hours" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "约 ${HOURS} 小时后")"
echo "[c] 读结论：python3 $HERE/long_uptime_needle.py --aggregate $JSONL"
