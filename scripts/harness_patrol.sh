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

read -r HEALTH PID OWNER CUDA SLOTS WINDOW HIT SCHED LAST ALERTF FAILEDF DISK ERRKV <<<"$(timeout 25 $SSH '
R=/home/user/lvllm/run
H=$(curl -s --noproxy "*" -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8700/health 2>/dev/null)
P=$(cat $R/ds41_prod.pid 2>/dev/null)
O=$(ss -ltnpH "sport = :8700" 2>/dev/null | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
E=$(tr "\0" "\n" < /proc/$O/environ 2>/dev/null)
CUDA=$(printf "%s\n" "$E" | sed -n "s/^XIAOTU_PIN_CUDA_MOE=//p" | head -1)
SLOTS=$(printf "%s\n" "$E" | sed -n "s/^XIAOTU_PIN_MAX_PER_LAYER=//p" | head -1)
WIN=$(printf "%s\n" "$E" | sed -n "s/^XIAOTU_PIN_WINDOW=//p" | head -1)
HIT=$(grep -a "\[xtu-pin\]\[hit\]" $R/ds41_prod.log 2>/dev/null | tail -1 | grep -oE "=[0-9.]+%" | tail -1 | tr -d "=%")
S=$(pgrep -f "schedule_nightly.sh" | head -1)
L=$(head -c 400 $R/nightly_last.json 2>/dev/null | tr -d " \n\r")
AF=$([ -e $R/nightly_alert.txt ] && echo 1 || echo 0)
FF=$([ -e $R/nightly_FAILED ] && echo 1 || echo 0)
D=$(df -BG / 2>/dev/null | awk "NR==2{gsub(/[^0-9]/,\"\",\$4); print \$4}")
ERR=$(tail -300 $R/ds41_prod.log 2>/dev/null | grep -acE "Traceback \(most recent call last\)|CUDA error|Xid [0-9]")
printf "%s %s %s %s %s %s %s %s %s %s %s %s %s\n" "${H:-0}" "${P:-NA}" "${O:-NA}" "${CUDA:-NA}" "${SLOTS:-NA}" "${WIN:-NA}" "${HIT:-NA}" "${S:-0}" "${L:-{}}" "$AF" "$FF" "${D:-0}" "${ERR:-0}"
')" || { read -r HEALTH PID OWNER CUDA SLOTS WINDOW HIT SCHED LAST ALERTF FAILEDF DISK ERRKV <<<"0 NA NA NA NA NA NA 0 {} 1 1 0 1"; }

REASONS=()
[ "$HEALTH" != "200" ] && REASONS+=("health=$HEALTH")
[ "$PID" = "NA" ] && REASONS+=("PID文件缺失")
[ "$PID" != "NA" ] && [ "$OWNER" != "NA" ] && [ "$PID" != "$OWNER" ] && REASONS+=("PID($PID)≠端口属主($OWNER)")
[ "$CUDA" != "1" ] && REASONS+=("CUDA_MOE=$CUDA(应1)")
[ "$SCHED" = "0" ] && REASONS+=("夜间调度未运行")
[ "$FAILEDF" = "1" ] && REASONS+=("nightly_FAILED 存在")
[ "$ALERTF" = "1" ] && REASONS+=("nightly_alert 存在")
case "$HIT" in NA) REASONS+=("命中率读不到") ;; *) awk -v h="$HIT" -v m="$HIT_MIN" 'BEGIN{exit !(h+0<m+0)}' && REASONS+=("命中率${HIT}%<${HIT_MIN}%") ;; esac
[ "${DISK:-0}" -lt "$DISK_MIN_GB" ] 2>/dev/null && REASONS+=("磁盘可用${DISK}GB<${DISK_MIN_GB}GB")
[ "${ERRKV:-0}" -gt 0 ] 2>/dev/null && REASONS+=("日志最近300行含${ERRKV}条致命行")

ALERT_N=0; REASON_STR=""
if [ "${#REASONS[@]}" -gt 0 ]; then
    ALERT_N=1
    REASON_STR="$(printf '%s; ' "${REASONS[@]}")"
    { printf '%s ALERT health=%s pid=%s cuda_moe=%s slots=%s window=%s hit=%s%% sched=%s disk=%sGB err=%s :: %s\n' \
        "$(date '+%F %T')" "$HEALTH" "$PID" "$CUDA" "$SLOTS" "$WINDOW" "$HIT" "$SCHED" "$DISK" "$ERRKV" "$REASON_STR"; } >> "$ALERT"
    tail -200 "$ALERT" > "$ALERT.tmp" && mv "$ALERT.tmp" "$ALERT"
fi
printf '{"ts":"%s","alert":%s,"health":"%s","pid":"%s","owner":"%s","cuda_moe":"%s","slots":"%s","window":"%s","hit":"%s","sched":"%s","disk_gb":"%s","err":%s,"last":%s,"reason":"%s"}\n' \
    "$(date '+%F %T')" "$ALERT_N" "$HEALTH" "$PID" "$OWNER" "$CUDA" "$SLOTS" "$WINDOW" "$HIT" "$SCHED" "$DISK" "${ERRKV:-0}" "${LAST:-{\}}" "$REASON_STR" >> "$OUT"
[ "$ALERT_N" = "1" ] && { echo "⚠️ $REASON_STR"; exit 3; }
echo "✅ 巡检正常 health=$HEALTH cuda_moe=$CUDA slots=$SLOTS window=$WINDOW hit=${HIT}% sched=$SCHED disk=${DISK}GB"
exit 0
