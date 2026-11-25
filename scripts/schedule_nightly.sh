#!/usr/bin/env bash
# =============================================================================
# schedule_nightly.sh —— 常驻调度：每天 AT_SPEC（默认 03:00）跑 nightly_restart_stable.sh
#
# 启动（远端、分离进程）：
#   setsid nohup bash scripts/schedule_nightly.sh --at 03:00 \
#       > /home/user/lvllm/run/nightly_scheduler.log 2>&1 < /dev/null &
#
# 行为：
#   * **常驻循环**（每日一次，不是单发）；每轮重新计算下一个 AT 时刻；
#   * 等待期间每 5 分钟写心跳；到点执行 nightly 脚本，结果写 $RUN/nightly_last.json（由对方脚本写）
#     与本脚本的 $RUN/nightly_sched_last（时间/退出码）；
#   * 幂等：若今天的目标时刻已成功执行过（nightly_last.json 的时间 >= 今日 AT 且 exit=0/10），则跳过本轮；
#   * 互斥：flock 全程持有，防重复调度；
#   * 失败可见：nightly 退出码 11 时写 $RUN/nightly_alert.txt（保留），并继续下一轮。
# =============================================================================
set -uo pipefail

RUN="${RUN:-/home/user/lvllm/run}"
REPO="${REPO:-/home/user/lvllm/vllm-xiaotu-moe}"
AT_SPEC="03:00"
case "${1:-}" in
    "") ;;
    --at) [ "$#" -ge 2 ] || { echo "[sched] ⛔ --at 缺少时间参数" >&2; exit 2; }; AT_SPEC="$2"
          [ "$#" -le 2 ] || { echo "[sched] ⛔ 多余参数" >&2; exit 2; } ;;
    -h|--help) [ "$#" -le 1 ] || { echo "[sched] ⛔ -h 不接受其它参数" >&2; exit 2; }
               sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "[sched] ⛔ 未知参数：'$1'" >&2; exit 2 ;;
esac
export TZ="Asia/Shanghai"
case "$AT_SPEC" in
    [0-9]:[0-5][0-9]|[01][0-9]:[0-5][0-9]|2[0-3]:[0-5][0-9]) ;;
    *) echo "[sched] ⛔ --at 非法：'$AT_SPEC'（应为 HH:MM）" >&2; exit 12 ;;
esac

mkdir -p "$RUN" || exit 12
PIDFILE="$RUN/nightly_scheduler.pid"
LOCK="$RUN/nightly_scheduler.lock"
SCHED_LAST="$RUN/nightly_sched_last"
MISSED_LOG="$RUN/nightly_missed.log"
TAG="${TAG:-prod}"; PORT="${PORT:-8700}"
exec 9>"$LOCK" || exit 12
flock -n 9 || { echo "[sched] ⛔ 已有调度进程在运行（$LOCK 被占用）" >&2; exit 12; }
echo "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
echo "[sched] 常驻启动 pid=$$ at=$AT_SPEC=$(date '+%F %T')"

next_target(){  # 输出 "<epoch> <wait_s>"
    AT_SPEC="$1" python3 - <<'PY'
import os, datetime
h, m = (int(x) for x in os.environ["AT_SPEC"].split(":"))
now = datetime.datetime.now()
tgt = now.replace(hour=h, minute=m, second=0, microsecond=0)
if tgt <= now:
    tgt += datetime.timedelta(days=1)
print(int(tgt.timestamp()), int((tgt - now).total_seconds()))
PY
}

already_done_today(){  # 今天的目标时刻是否已成功执行（须 result/port/tag 全部匹配）
    AT_SPEC="$1" LAST="$RUN/nightly_last.json" PORT="$PORT" TAG="$TAG" python3 - <<'PY'
import os, json, datetime
try:
    d = json.load(open(os.environ["LAST"]))
except Exception:
    raise SystemExit(1)
h, m = (int(x) for x in os.environ["AT_SPEC"].split(":"))
today_at = datetime.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
try:
    ts = datetime.datetime.strptime(d.get("ts", ""), "%Y-%m-%d %H:%M:%S")
except Exception:
    raise SystemExit(1)
want_port = int(os.environ.get("PORT", "8700"))
want_tag = os.environ.get("TAG", "prod")
ok = (d.get("result") in ("target_ok", "rollback_ok", "rollback_ok_delayed")
      and d.get("port") == want_port and d.get("tag", "prod") == want_tag
      and ts >= today_at)
raise SystemExit(0 if ok else 1)
PY
}

while :; do
    read -r TGT_EPOCH WAIT_S <<<"$(next_target "$AT_SPEC")" || { echo "[sched] ⛔ 时间计算失败"; sleep 60; continue; }
    case "${WAIT_S:-x}" in ''|*[!0-9]*) echo "[sched] ⛔ WAIT_S='${WAIT_S:-}'"; sleep 60; continue ;; esac
    echo "[sched] 下一轮=$(date -d "@$TGT_EPOCH" '+%F %T') 距离=${WAIT_S}s"
    # 错过窗口告警：若距目标时刻超过 24h（说明启动时已过今日 AT 且今日未执行）
    case "$WAIT_S" in
        *[!0-9]*) ;; *)
            if [ "$WAIT_S" -gt 86000 ] && ! already_done_today "$AT_SPEC"; then
                echo "[sched] ⚠️ 今日 $(date '+%F') 的 $AT_SPEC 未执行（调度器在窗口后才启动）" \
                    | tee -a "$MISSED_LOG" >&2
            fi ;;
    esac

    END=$(( $(date +%s) + WAIT_S ))
    while :; do
        LEFT=$(( END - $(date +%s) ))
        [ "$LEFT" -le 0 ] && break
        if [ "$LEFT" -gt 300 ]; then sleep 300; echo "[sched] $(date '+%F %T') 心跳 剩余 $(( END - $(date +%s) ))s"
        else sleep "$LEFT"; fi
    done

    if already_done_today "$AT_SPEC"; then
        echo "[sched] $(date '+%F %T') 今日已执行过，跳过"
        sleep 60
        continue
    fi

    echo "[sched] $(date '+%F %T') 到点，执行 nightly_restart_stable.sh"
    cd "$REPO" || { echo "[sched] ⛔ 仓库不存在：$REPO"; sleep 300; continue; }
    TAG="$TAG" PORT="$PORT" RUN="$RUN" RUN_DIR="$RUN"         bash scripts/nightly_restart_stable.sh 9>&-
    RC=$?
    printf '%s exit=%s\n' "$(date '+%F %T')" "$RC" > "$SCHED_LAST"
    echo "[sched] $(date '+%F %T') nightly 退出码=$RC"
    sleep 120   # 防同一秒重复触发
done
