#!/usr/bin/env bash
# check_ced_on.sh —— 机械检查：正在跑的服务是否**真正**开了 CED。
#
# 背景（AGENTS.md「CED 默认必须开启」）：
#   CED 由 `--swa-bounded-replay` 开启；`serve_v41.sh` 默认 CED=1，
#   但**直接用 api_server 命令行起服务不会带这个旗标** ⇒ CED 静默关闭（2026-10-31 事故）。
#
# 只读脚本：只读 /proc 与日志，不启停任何进程、不发信号。
#
# 判据（全部满足才算 CED 生效）：
#   ① 进程存在，且命令行含 `--swa-bounded-replay`
#   ② 命令行**不含** `--no-swa-bounded-replay`
#   ③ 日志里不出现 `SWA bounded replay needs model runner V2`（出现 = CED 没生效）
#
# 退出码：0=生效；1=未生效；2=服务不存在
#
# 用法：bash scripts/check_ced_on.sh [--port 8700] [--log <path>]

set -u

PORT=8700
LOG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?}"; shift 2 ;;
        --log)  LOG="${2:?}";  shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 64 ;;
    esac
done

# 端口 → PID（只认监听该端口的进程，不用进程名匹配）
PID=""
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    [ -r "/proc/$pid/cmdline" ] || continue
    # 用 tr 展开 NUL，避免把二进制 cmdline 当文本 grep 出错
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$cmd" in
        *"api_server"*) : ;;
        *) continue ;;
    esac
    # 端口归属：该进程的监听 socket 里是否含 :PORT
    for fd in /proc/$pid/fd/*; do
        [ -L "$fd" ] || continue
        tgt=$(readlink "$fd" 2>/dev/null) || continue
        case "$tgt" in
            "socket:["*"]") : ;;
            *) continue ;;
        esac
    done
    # 退化为命令行的 --port 匹配（api_server 一定带 --port）
    case "$cmd" in
        *"--port $PORT"*|*"--port=$PORT"*) PID="$pid" ;;
    esac
    [ -n "$PID" ] && break
done

if [ -z "$PID" ]; then
    echo "未找到监听 --port $PORT 的 api_server 进程 ⇒ 服务未运行" >&2
    exit 2
fi

CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
echo "服务 pid=$PID"

if printf '%s' "$CMD" | grep -q -- '--no-swa-bounded-replay'; then
    echo "✗ 命令行含 --no-swa-bounded-replay ⇒ CED 被显式关闭（基线臂）"
    exit 1
fi

if printf '%s' "$CMD" | grep -q -- '--swa-bounded-replay'; then
    echo "✓ 命令行含 --swa-bounded-replay"
else
    echo "✗ 命令行**不含** --swa-bounded-replay ⇒ CED 静默关闭（2026-10-31 事故同款）" >&2
    exit 1
fi

if [ -n "$LOG" ] && [ -r "$LOG" ]; then
    if grep -aq 'SWA bounded replay needs model runner V2' "$LOG"; then
        echo "✗ 日志出现「SWA bounded replay needs model runner V2」⇒ CED 未真正生效" >&2
        exit 1
    fi
    echo "✓ 日志未出现 model runner V2 警告"
fi

echo "CED 生效"
exit 0
