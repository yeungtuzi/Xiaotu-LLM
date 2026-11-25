#!/usr/bin/env bash
# start_gateway.sh —— 起/停 vLLM 前置网关（默认 :4000，纯透传 + 虚拟 key + 计量）
#
# 密钥文件（都 0600，不进仓库、不进 argv）：
#   <DIR>/upstream.key  上游 vLLM 的 VLLM_API_KEY
#   <DIR>/master.key    管理面 key（首次自动生成）
#
# 用法： bash start_gateway.sh start|stop|status
set -uo pipefail
DIR="${GATEWAY_DIR:-/home/user/lvllm/gateway}"
PY="${GATEWAY_PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
PY_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gateway.py"
PORT="${GATEWAY_PORT:-4000}"
PIDF="$DIR/gateway.pid"
LOG="$DIR/gateway.log"
UP="$DIR/upstream.key"
MK="$DIR/master.key"

die() { printf '[gw] ⛔ %s\n' "$1" >&2; exit "${2:-2}"; }
say() { printf '[gw] %s\n' "$*"; }

mkdir -p "$DIR" || die "无法创建 $DIR" 3
chmod 700 "$DIR" 2>/dev/null || true

case "${1:-status}" in
start)
    [ -s "$UP" ] || die "缺上游密钥文件 $UP（内容 = vLLM 的 VLLM_API_KEY）" 3
    if [ ! -s "$MK" ]; then
        ( umask 077; head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$MK" ) || die "生成 master key 失败" 4
        say "已生成管理面密钥：$MK"
    fi
    chmod 600 "$UP" "$MK" 2>/dev/null || true
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
        die "已在运行 pid=$(cat "$PIDF")；先 stop" 3
    fi
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then die "端口 $PORT 已被占用" 3; fi
    UPSTREAM_API_KEY="$(tr -d '\n' < "$UP")"
    MASTER="$(tr -d '\n' < "$MK")"
    cd "$(dirname "$PY_FILE")" || die "cd 失败" 3
    VLLM_API_KEY="$UPSTREAM_API_KEY" GATEWAY_MASTER_KEY="$MASTER" \
    GATEWAY_DB="$DIR/gateway.db" GATEWAY_PORT="$PORT" \
    INJECT_STREAM_USAGE="${GATEWAY_INJECT_STREAM_USAGE:-0}" \
    GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}" \
        setsid nohup "$PY" -u gateway.py >> "$LOG" 2>&1 < /dev/null &
    sleep 3
    pid="$(ss -ltnp 2>/dev/null | sed -n "s/.*:${PORT} .*pid=\([0-9]*\).*/\1/p" | head -1)"
    [ -n "$pid" ] || die "启动失败，见 $LOG" 4
    printf '%s' "$pid" > "$PIDF"
    say "已启动 pid=$pid 端口=$PORT 日志=$LOG"
    say "管理面密钥：$MK（用 Bearer 调 /admin/**）"
    ;;
stop)
    if [ -f "$PIDF" ]; then
        pid="$(tr -dc '0-9' < "$PIDF")"; [ -n "$pid" ] && kill "$pid" 2>/dev/null && say "已停止 pid=$pid"
        rm -f "$PIDF"
    else
        say "无 PID 文件（可能未运行）"
    fi
    ;;
status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; then
        say "运行中 pid=$(cat "$PIDF") 端口=$PORT"
        curl -s -o /dev/null -w '[gw] health=%{http_code}\n' --max-time 4 "http://127.0.0.1:${PORT}/health"
    else
        say "未运行"
    fi
    ;;
*) die "用法：$0 start|stop|status" 2 ;;
esac
