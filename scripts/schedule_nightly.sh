#!/usr/bin/env bash
# =============================================================================
# schedule_nightly.sh —— **单次**定时器：在下一个 AT_SPEC（默认 03:00）执行一次
# nightly_restart_stable.sh，随后退出（不是每日任务）。
#
# 用法（远端、分离进程）：
#   setsid nohup bash scripts/schedule_nightly.sh --at 03:00 \
#       > /home/user/lvllm/run/nightly_scheduler.log 2>&1 < /dev/null &
#
# 行为：
#   * 计算**下一个** AT 时刻（今天未过则今天，否则明天），等待期间每 5 分钟心跳；
#   * 到点执行 nightly 脚本一次，透传退出码，然后**退出**；
#   * 启动时写 $RUN/nightly_pending（巡检据此判断"有一次待执行的重启"），
#     nightly 目标档成功后会删除该标记；
#   * 幂等：若今日该时刻已成功执行过，则直接退出 0（不重复重启）；
#   * 互斥：flock 全程持有，防重复调度；子进程一律 9>&- 关闭锁 fd。
# =============================================================================
set -uo pipefail

RUN="${RUN:-/home/user/lvllm/run}"
REPO="${REPO:-/home/user/lvllm/vllm-xiaotu-moe}"
AT_SPEC="03:00"
case "${1:-}" in
    "") ;;
    --at) [ "$#" -ge 2 ] || { echo "[sched] ⛔ --at 缺少时间参数" >&2; exit 2; }
          AT_SPEC="$2"; [ "$#" -le 2 ] || { echo "[sched] ⛔ 多余参数" >&2; exit 2; } ;;
    -h|--help) [ "$#" -le 1 ] || { echo "[sched] ⛔ -h 不接受其它参数" >&2; exit 2; }
               sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[sched] ⛔ 未知参数：'$1'" >&2; exit 2 ;;
esac
case "$AT_SPEC" in
    [0-9]:[0-5][0-9]|[01][0-9]:[0-5][0-9]|2[0-3]:[0-5][0-9]) ;;
    *) echo "[sched] ⛔ --at 非法：'$AT_SPEC'（应为 HH:MM）" >&2; exit 12 ;;
esac
export TZ="Asia/Shanghai"

mkdir -p "$RUN" || exit 12
PIDFILE="$RUN/nightly_scheduler.pid"
LOCK="$RUN/nightly_scheduler.lock"
PENDING="$RUN/nightly_pending"
SCHED_LAST="$RUN/nightly_sched_last"
TAG="${TAG:-prod}"; PORT="${PORT:-8700}"
exec 9>"$LOCK" || exit 12
flock -n 9 || { echo "[sched] ⛔ 已有调度进程在运行（$LOCK 被占用）" >&2; exit 12; }
echo "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

read -r TGT_EPOCH WAIT_S <<<"$(AT_SPEC="$AT_SPEC" python3 - <<'PY'
import os, datetime
h, m = (int(x) for x in os.environ["AT_SPEC"].split(":"))
now = datetime.datetime.now()
tgt = now.replace(hour=h, minute=m, second=0, microsecond=0)
if tgt <= now:
    tgt += datetime.timedelta(days=1)
print(int(tgt.timestamp()), int((tgt - now).total_seconds()))
PY
)" || { echo "[sched] ⛔ 时间计算失败" >&2; exit 12; }
case "${WAIT_S:-x}" in ''|*[!0-9]*) echo "[sched] ⛔ WAIT_S='${WAIT_S:-}'" >&2; exit 12 ;; esac

echo "[sched] 单次调度 pid=$$ 目标=$(date -d "@$TGT_EPOCH" '+%F %T') 距离=${WAIT_S}s"
printf '%s target=%s\n' "$(date '+%F %T')" "$(date -d "@$TGT_EPOCH" '+%F %T')" > "$PENDING"

END=$(( $(date +%s) + WAIT_S ))
while :; do
    LEFT=$(( END - $(date +%s) ))
    [ "$LEFT" -le 0 ] && break
    if [ "$LEFT" -gt 300 ]; then sleep 300 9>&-; echo "[sched] $(date '+%F %T') 心跳 剩余 $(( END - $(date +%s) ))s"
    else sleep "$LEFT" 9>&-; fi
done

echo "[sched] $(date '+%F %T') 到点，执行 nightly_restart_stable.sh（单次）"
cd "$REPO" || { echo "[sched] ⛔ 仓库不存在：$REPO" >&2; exit 12; }
TAG="$TAG" PORT="$PORT" RUN="$RUN" RUN_DIR="$RUN" \
    bash scripts/nightly_restart_stable.sh 9>&-
RC=$?
printf '%s exit=%s\n' "$(date '+%F %T')" "$RC" > "$SCHED_LAST"
echo "[sched] $(date '+%F %T') nightly 退出码=$RC（单次调度结束，不再重试）"
exit "$RC"
