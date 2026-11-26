#!/usr/bin/env bash
# =============================================================================
# harness_patrol.sh —— harness 侧 24h 只读巡检（每 15 分钟）
#
# 只读：不启停任何服务、不改远端文件。单条 ssh（≤25s），输出一行 JSON 到
#   /root/lvllm/backup/patrol/patrol-YYYYMMDD.jsonl
# 异常时追加到
#   /root/lvllm/backup/patrol/ALERT.txt  （并保留最近 200 行）
#
# 判据（任一命中 ⇒ 记 alert=1 并写明原因）：
#   1 health != 200
#   2 PID 文件缺失 / 端口属主与 PID 文件不符
#   3 XIAOTU_PIN_CUDA_MOE != 1（稳定档口径）
#   4 夜间调度进程不在
#   5 $RUN/nightly_FAILED 存在
#   6 命中率 < HIT_MIN（默认 15%）
#   7 远端根分区可用 < DISK_MIN_GB（默认 20 GB）
#   8 主日志最近 300 行出现 Traceback/CUDA error/Xid
# =============================================================================
set -uo pipefail
REPO="${REPO:-/root/lvllm/repos/xiaotu-llm}"
OUTDIR="${OUTDIR:-/root/lvllm/backup/patrol}"
SSH="ssh -F /root/lvllm/tmp/sshcfg -o BatchMode=yes -o ConnectTimeout=8 gpu"
HIT_MIN="${HIT_MIN:-15}"
DISK_MIN_GB="${DISK_MIN_GB:-20}"
mkdir -p "$OUTDIR" || exit 1
OUT="$OUTDIR/patrol-$(date +%Y%m%d).jsonl"
ALERT="$OUTDIR/ALERT.txt"

read -r HEALTH PID OWNER CUDA SLOTS WINDOW HIT HIT_LAST SCHED LAST ALERTF FAILEDF DISK ERRKV <<<"$(timeout 25 $SSH '
R=/home/user/lvllm/run
H=$(curl -s --noproxy "*" -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8700/health 2>/dev/null)
P=$(cat $R/ds41_prod.pid 2>/dev/null)
O=$(ss -ltnpH "sport = :8700" 2>/dev/null | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
E=$(tr "\0" "\n" < /proc/$O/environ 2>/dev/null)
CUDA=$(printf "%s\n" "$E" | sed -n "s/^XIAOTU_PIN_CUDA_MOE=//p" | head -1)
SLOTS=$(printf "%s\n" "$E" | sed -n "s/^XIAOTU_PIN_MAX_PER_LAYER=//p" | head -1)
WIN=$(printf "%s\n" "$E" | sed -n "s/^XIAOTU_PIN_WINDOW=//p" | head -1)
HITW=$(grep -a "\[xtu-pin\]\[hit\]" $R/ds41_prod.log 2>/dev/null | grep -oE "=[0-9.]+%" | tr -d "=%" | tail -10)
HIT=$(printf "%s\n" "$HITW" | awk "NF{s+=\$1;n++} END{if(n) printf \"%.1f\", s/n}")
HIT_LAST=$(printf "%s\n" "$HITW" | tail -1)
S=$(pgrep -f "schedule_nightly.sh" | head -1)
L=$(head -c 400 $R/nightly_last.json 2>/dev/null | tr -d " \n\r")
AF=$([ -e $R/nightly_alert.txt ] && echo 1 || echo 0)
FF=$([ -e $R/nightly_FAILED ] && echo 1 || echo 0)
D=$(df -BG / 2>/dev/null | awk "NR==2{gsub(/[^0-9]/,\"\",\$4); print \$4}")
ERR=$(tail -300 $R/ds41_prod.log 2>/dev/null | grep -acE "Traceback \(most recent call last\)|CUDA error|Xid [0-9]")
printf "%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" "${H:-0}" "${P:-NA}" "${O:-NA}" "${CUDA:-NA}" "${SLOTS:-NA}" "${WIN:-NA}" "${HIT:-NA}" "${HIT_LAST:-NA}" "${S:-0}" "${L:-{}}" "$AF" "$FF" "${D:-0}" "${ERR:-0}"
')" || { read -r HEALTH PID OWNER CUDA SLOTS WINDOW HIT SCHED LAST ALERTF FAILEDF DISK ERRKV <<<"0 NA NA NA NA NA NA 0 {} 1 1 0 1"; }

REASONS=()
[ "$HEALTH" != "200" ] && REASONS+=("health=$HEALTH")
[ "$PID" = "NA" ] && REASONS+=("PID文件缺失")
[ "$PID" != "NA" ] && [ "$OWNER" != "NA" ] && [ "$PID" != "$OWNER" ] && REASONS+=("PID($PID)≠端口属主($OWNER)")
[ "$CUDA" != "1" ] && REASONS+=("CUDA_MOE=$CUDA(应1)")
[ "$SCHED" = "0" ] && REASONS+=("夜间调度未运行")
[ "$FAILEDF" = "1" ] && REASONS+=("nightly_FAILED 存在")
[ "$ALERTF" = "1" ] && REASONS+=("nightly_alert 存在")
# nightly 最近一次结果：0=目标成功 / 10=回滚成功 / 其它（11/12）=异常
NL_EXIT="$(printf '%s' "${LAST:-}" | sed -n 's/.*"exit":\([0-9-]*\).*/\1/p')"
NL_RES="$(printf '%s' "${LAST:-}" | sed -n 's/.*"result":"\([a-z_]*\)".*/\1/p')"
# 是否存在"一次待执行的重启"（单次调度器在启动时写，nightly 目标档成功后删）
# pending 的内容含目标时刻；未到点前不追责、不提前补跑
PENDING_N=0; PENDING_DUE=0
PINFO="$(timeout 10 $SSH 'cat /home/user/lvllm/run/nightly_pending 2>/dev/null' 2>/dev/null | tr -d "\r")"
P_TGT="$(printf '%s' "$PINFO" | sed -n 's/.*target=\([0-9: -]*\).*/\1/p')"
if [ -n "$PINFO" ]; then
    PENDING_N=1
    if [ -n "$P_TGT" ]; then
        TGT_EPOCH="$(date -d "$P_TGT" +%s 2>/dev/null || echo 0)"
        [ "$(date +%s)" -ge "$TGT_EPOCH" ] && PENDING_DUE=1
    else
        PENDING_DUE=1
    fi
fi
if [ "$PENDING_N" = "1" ] && [ "$PENDING_DUE" = "1" ]; then
    case "${NL_EXIT:-}" in
        0|10) ;;
        11) REASONS+=("nightly 回滚失败(exit=11)") ;;
        12) REASONS+=("nightly 未执行(exit=12,${NL_RES:-?})") ;;
        *) REASONS+=("nightly 异常退出(exit=${NL_EXIT:-none})") ;;
    esac
fi
case "$HIT" in NA) REASONS+=("命中率读不到") ;; *) awk -v h="$HIT" -v m="$HIT_MIN" 'BEGIN{exit !(h+0<m+0)}' && REASONS+=("命中率${HIT}%<${HIT_MIN}%") ;; esac
[ "${DISK:-0}" -lt "$DISK_MIN_GB" ] 2>/dev/null && REASONS+=("磁盘可用${DISK}GB<${DISK_MIN_GB}GB")
[ "${ERRKV:-0}" -gt 0 ] 2>/dev/null && REASONS+=("日志最近300行含${ERRKV}条致命行")

# ---- 守护动作：夜间任务失败且仍在可补跑窗口 ⇒ 立刻在远端补跑一次（每晚至多一次）----
AT_H="${AT_H:-3}"; RETRY_WINDOW_H="${RETRY_WINDOW_H:-4}"
RETRY_MARK="/root/lvllm/backup/patrol/.retried-$(date +%Y%m%d)"
NEED="/root/lvllm/backup/patrol/NEEDS_ATTENTION.txt"
ACTED=""
NL_EXIT_V="${NL_EXIT:-}"
if [ "$PENDING_N" = "1" ] && [ "$PENDING_DUE" = "1" ] && [ "${NL_EXIT_V:-}" != "0" ] && [ "${NL_EXIT_V:-}" != "10" ]; then
    NOW_H="$(date +%H | sed 's/^0//')"
    ELAPSED=$(( (NOW_H - AT_H + 24) % 24 ))
    if [ "$ELAPSED" -lt "$RETRY_WINDOW_H" ] && [ ! -e "$RETRY_MARK" ]; then
        : > "$RETRY_MARK"
        timeout 20 $SSH 'cd /home/user/lvllm/vllm-xiaotu-moe && setsid nohup bash scripts/nightly_restart_stable.sh > /home/user/lvllm/run/nightly_retry_'"$(date +%Y%m%d-%H%M%S)"'.log 2>&1 < /dev/null & echo RETRY_STARTED' >/dev/null 2>&1 \
            && ACTED="已自动补跑一次"
    elif [ -e "$RETRY_MARK" ]; then
        ACTED="今晚已补跑过，仍未成功"
    else
        ACTED="不在补跑窗口(03:00-07:00)"
    fi
fi

# ---- 守护动作 ②：口径漂移（CUDA_MOE≠1）在维护窗口内 ⇒ 触发一次稳定档重启 ----
MAINT_FROM="${MAINT_FROM:-2}"; MAINT_TO="${MAINT_TO:-6}"
if [ "$PENDING_N" = "1" ] && [ "$PENDING_DUE" = "1" ] && [ "${CUDA:-NA}" != "1" ] && [ "${HEALTH:-0}" = "200" ]; then
    NOW_H2="$(date +%H | sed 's/^0//')"
    IN_MAINT=0
    if [ "$MAINT_FROM" -le "$MAINT_TO" ]; then
        [ "$NOW_H2" -ge "$MAINT_FROM" ] && [ "$NOW_H2" -lt "$MAINT_TO" ] && IN_MAINT=1
    else
        { [ "$NOW_H2" -ge "$MAINT_FROM" ] || [ "$NOW_H2" -lt "$MAINT_TO" ]; } && IN_MAINT=1
    fi
    if [ "$IN_MAINT" = "1" ] && [ ! -e "$RETRY_MARK" ]; then
        : > "$RETRY_MARK"
        timeout 20 $SSH 'cd /home/user/lvllm/vllm-xiaotu-moe && setsid nohup bash scripts/nightly_restart_stable.sh > /home/user/lvllm/run/nightly_retry_drift_'"$(date +%Y%m%d-%H%M%S)"'.log 2>&1 < /dev/null & echo RETRY_STARTED' >/dev/null 2>&1 \
            && ACTED="${ACTED} 已在维护窗口内触发稳定档重启"
    elif [ "$IN_MAINT" = "0" ]; then
        ACTED="${ACTED} 口径漂移但在维护窗口外(02:00-06:00)，需人工确认"
    fi
fi

[ "$PENDING_N" = "1" ] && [ "$PENDING_DUE" = "0" ] && ACTED="${ACTED} 待 ${P_TGT:-下一个 03:00} 执行（未到点）"
ALERT_N=0; REASON_STR=""
if [ "${#REASONS[@]}" -gt 0 ]; then
    ALERT_N=1
    REASON_STR="$(printf '%s; ' "${REASONS[@]}")"
    [ -n "$ACTED" ] && REASON_STR="${REASON_STR}[$ACTED] "
    { printf '%s ALERT health=%s pid=%s cuda_moe=%s slots=%s window=%s hit=%s%% sched=%s disk=%sGB err=%s :: %s\n' \
        "$(date '+%F %T')" "$HEALTH" "$PID" "$CUDA" "$SLOTS" "$WINDOW" "$HIT" "$SCHED" "$DISK" "$ERRKV" "$REASON_STR"; } >> "$ALERT"
    tail -200 "$ALERT" > "$ALERT.tmp" && mv "$ALERT.tmp" "$ALERT"
fi
if [ "$ALERT_N" = "1" ]; then
    { echo "== 未处理告警（首次出现 $(date '+%F %T')）=="; printf '%s\n' "${REASONS[@]}"; } > "$NEED"
else
    rm -f "$NEED"
fi
printf '{"ts":"%s","alert":%s,"health":"%s","pid":"%s","owner":"%s","cuda_moe":"%s","slots":"%s","window":"%s","hit":"%s","sched":"%s","disk_gb":"%s","err":%s,"last_exit":"%s","last_result":"%s","last_ts":"%s","acted":"%s","reason":"%s"}\n' \
    "$(date '+%F %T')" "$ALERT_N" "$HEALTH" "$PID" "$OWNER" "$CUDA" "$SLOTS" "$WINDOW" "$HIT" "$SCHED" "$DISK" "${ERRKV:-0}" \
    "${NL_EXIT:-}" "${NL_RES:-}" "${P_TGT:-}" "${ACTED//\"/}" "${REASON_STR//\"/}" >> "$OUT"
[ "$ALERT_N" = "1" ] && { echo "⚠️ $REASON_STR"; exit 3; }
echo "✅ 巡检正常 health=$HEALTH cuda_moe=$CUDA slots=$SLOTS window=$WINDOW hit=${HIT}% sched=$SCHED disk=${DISK}GB"
exit 0
