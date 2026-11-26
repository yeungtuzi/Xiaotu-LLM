#!/usr/bin/env bash
# =============================================================================
# nightly_restart_stable.sh —— 完整稳定档重启 8700；失败自动回滚（唯一入口 serve_ds41.sh）
#
# 用法（在 GPU 执行机、仓库根目录）：
#   bash scripts/nightly_restart_stable.sh            # 执行：完整稳定档重启 + 失败回滚
#   bash scripts/nightly_restart_stable.sh --preflight # 只做前置检查（不碰服务、不写状态文件）
#   bash scripts/nightly_restart_stable.sh --dry-run   # 打印将执行的命令
#   bash scripts/nightly_restart_stable.sh -h
#
# 退出码：0=目标口径成功  10=回滚成功  11=回滚失败（写 FAILED/ALERT）  12=前置失败（服务未动）
#
# 设计（R38 四轮审计定稿）：
#   * 只用 PID 文件 + 端口属主判定进程，无任何名字/命令行模式匹配；
#   * 回滚快照在重启前落盘（强制 600），抓不到的项取**目标稳定档**同值（绝不用 serve 缺省）；
#   * 验收覆盖异步阶段：以**本轮字节偏移**为界，只看增量（append-only 日志不会污染判定），
#     等冒烟 busy 出现再消失、断言无 smoke.fail（mtime>本轮起点）、预热增量含 fail=0、
#     4 个日志增量无 Traceback/CUDA error/illegal memory access/Xid；
#   * 目标与回滚**用同一套验收**；回滚失败再给一次 30 分钟后的机会；
#   * 仅 run 模式写 LAST/ALERT/FAILED/DEGRADED（preflight/dry-run 不写任何状态文件）。
# =============================================================================
set -uo pipefail

REPO="${REPO:-/home/user/lvllm/vllm-xiaotu-moe}"
RUN="${RUN_DIR:-${RUN:-/home/user/lvllm/run}}"
export RUN_DIR="$RUN"
PORT="${PORT:-8700}"
TAG="${TAG:-prod}"
MODEL_DIR="${MODEL_DIR:-/home/user/lvllm/models/dsv41f-1017-weights}"
KEYFILE="${KEYFILE:-/home/user/lvllm/gateway/upstream.key}"
PIDFILE="$RUN/ds41_${TAG}.pid"
SNAP="$RUN/nightly_profile.snapshot"
LAST="$RUN/nightly_last.json"
ALERT="$RUN/nightly_alert.txt"
FAILEDF="$RUN/nightly_FAILED"
DEGRADED="$RUN/nightly_DEGRADED"
LOCK="$RUN/nightly_restart.lock"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
ASYNC_TIMEOUT="${ASYNC_TIMEOUT:-900}"
POLL="${POLL:-15}"
LOG="$RUN/nightly_restart_$(date '+%Y%m%d-%H%M%S').log"

usage(){ sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; }
MODE="run"
[ "$#" -gt 1 ] && { echo "⛔ 只接受一个参数（得到 $# 个）" >&2; exit 2; }
[ "$#" -eq 1 ] && [ -z "$1" ] && { echo "⛔ 参数为空（不带参数=执行重启；可用 --preflight/--dry-run）" >&2; exit 2; }
case "${1:-}" in
    "") MODE=run ;;
    --preflight) MODE=preflight ;;
    --dry-run) MODE=dryrun ;;
    -h|--help) usage; exit 0 ;;
    *) echo "⛔ 未知参数：'$1'（可用：--preflight / --dry-run / -h）" >&2; exit 2 ;;
esac
export TZ="Asia/Shanghai"

case "$HEALTH_TIMEOUT" in ''|*[!0-9]*) echo "⛔ HEALTH_TIMEOUT 非数字：'$HEALTH_TIMEOUT'" >&2; exit 2 ;; esac
case "$ASYNC_TIMEOUT"  in ''|*[!0-9]*) echo "⛔ ASYNC_TIMEOUT 非数字：'$ASYNC_TIMEOUT'" >&2; exit 2 ;; esac
case "$POLL"           in ''|*[!0-9]*) echo "⛔ POLL 非数字：'$POLL'" >&2; exit 2 ;; esac

say(){ printf '[nightly] %s\n' "$*" >>"$LOG"; printf '[nightly] %s\n' "$*" >&2; }
write_last(){ [ "$MODE" = "run" ] || return 0
    printf '{"ts":"%s","exit":%s,"result":"%s","log":"%s","port":%s,"tag":"%s"}\n' \
        "$(date '+%F %T')" "${1:-?}" "${2:-?}" "$LOG" "$PORT" "$TAG" > "$LAST" 2>/dev/null || true; }
mark_failed(){ [ "$MODE" = "run" ] || return 0
    { echo "nightly 失败 at $(date '+%F %T')（result=${1:-?}）"; echo "log=$LOG"; } > "$ALERT" 2>/dev/null || true
    : > "$FAILEDF" 2>/dev/null || true; }
die(){ say "⛔ $*"; mark_failed "preflight_failed"; write_last 12 "preflight_failed"; exit 12; }

health(){ curl -s --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null; }
pid_alive(){ [ -n "${1:-}" ] && [ -d "/proc/$1" ]; }
getenv_of(){ tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n "s/^$2=//p" | head -1; }
port_owner(){ ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }
owns_port(){ [ "$(port_owner)" = "$(cat "$PIDFILE" 2>/dev/null || true)" ]; }
foff(){ [ -e "$1" ] && wc -c < "$1" 2>/dev/null || echo 0; }
fino(){ [ -e "$1" ] && stat -c %i "$1" 2>/dev/null || echo 0; }
fdelta(){ [ -e "$1" ] && tail -c "+$(( ${2:-0} + 1 ))" "$1" 2>/dev/null; }
# 有效偏移：文件被替换(inode 变)或变短(归档重建) ⇒ 从 0 读全量，绝不漏扫
eff_off(){ local f="$1" off="$2" ino="$3" cur
    [ -e "$f" ] || { echo 0; return; }
    [ "$(fino "$f")" = "$ino" ] || { echo 0; return; }
    cur="$(foff "$f")"; [ "$cur" -lt "$off" ] && { echo 0; return; }
    echo "$off"; }
fmtime(){ [ -e "$1" ] && stat -c %Y "$1" 2>/dev/null || echo 0; }

TARGET=( "MODEL_DIR=$MODEL_DIR" "TAG=$TAG" "PORT=$PORT" "GPU_UTILIZATION=0.9" "MAX_NUM_SEQS=4"
         "MAX_NUM_BATCHED_TOKENS=8192" "MAXLEN=524288" "SPEC=1"
         "PIN=1" "PIN_SLOTS=19" "PIN_WINDOW=200" "PIN_CUDA_MOE=1"
         "MARLIN=0" "SWA_BOUNDED_REPLAY=1" "NO_PREFIX_CACHE=0" "PC=0"
         "EXPANDABLE=1" "CUDAGRAPH_MODE=FULL_DECODE_ONLY" "WARMUP=1" "THREADS=60" )

[ -d "$REPO" ] || die "仓库目录不存在：$REPO"
cd "$REPO" || die "无法进入 $REPO"
[ -f scripts/serve_ds41.sh ] || die "缺少 scripts/serve_ds41.sh"
# 部署版本一致性：目标档用到的每个参数名都必须在本机 serve_ds41.sh 的参数表里，
# 否则 restart 会在校验阶段拒绝、夜间静默变成 aborted_no_change（2026-11-26 实际发生）。
for kv in "${TARGET[@]}"; do
    n="${kv%%=*}"
    grep -qE "(^|[ |])${n}[|)]" scripts/serve_ds41.sh \
        || die "本机 scripts/serve_ds41.sh 不支持参数 ${n}（远端部署版本落后？先同步再重启）"
done
[ -d "$MODEL_DIR" ] || die "模型目录不存在：$MODEL_DIR"
mkdir -p "$RUN" || die "无法写 $RUN"

# 运行期互斥（防止人工与调度器同时执行）
if [ "$MODE" = "run" ]; then
    exec 8>"$LOCK" || die "无法创建锁 $LOCK"
    flock -n 8 || die "已有 nightly 在运行（$LOCK 被占用）"
fi

# 鉴权：env 优先，其次 0600 密钥文件
API_KEY_STATE="unset"
if [ -n "${API_KEY:-}" ]; then
    API_KEY_STATE="set(len=${#API_KEY})"
elif [ -r "$KEYFILE" ]; then
    _mode="$(stat -c %a "$KEYFILE" 2>/dev/null || echo '???')"
    case "$_mode" in 600|400) ;; *) die "密钥文件权限过宽：$KEYFILE mode=$_mode（要求 600/400）" ;; esac
    API_KEY="$(cat "$KEYFILE" 2>/dev/null | tr -d '[:space:]')"
    [ -n "$API_KEY" ] && API_KEY_STATE="set(len=${#API_KEY})"
fi
[ "$API_KEY_STATE" = "unset" ] && die "无 API_KEY（env 与 $KEYFILE 均不可用）⇒ 拒绝重启，避免静默降级为无鉴权"

# 回滚基线
CUR_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
pid_alive "$CUR_PID" || die "当前实例未运行或 PID 文件陈旧（PID='${CUR_PID:-}'）⇒ 无回滚基线，拒绝重启"
owns_port || die "端口 $PORT 属主($(port_owner))与 PID 文件($CUR_PID)不符 ⇒ 归属不明，拒绝动作"

g(){ getenv_of "$CUR_PID" "$1"; }
RB_SLOTS="$(g XIAOTU_PIN_MAX_PER_LAYER)"; RB_WIN="$(g XIAOTU_PIN_WINDOW)"
RB_CUDA="$(g XIAOTU_PIN_CUDA_MOE)";      RB_MARLIN="$(g XIAOTU_MOE4_MARLIN)"
RB_SWA="$(g SWA_BOUNDED_REPLAY)";        RB_SEQS="$(g MAX_NUM_SEQS)"
RB_MBT="$(g MAX_NUM_BATCHED_TOKENS)";    RB_LEN="$(g DS41_MAXLEN)"
RB_UTIL="$(g GPU_UTIL)";                 RB_CGM="$(g CUDAGRAPH_MODE)"
RB_TP="$(g DS41_TP)";                    RB_THREADS="$(g XIAOTU_MOE_THREADS)"
RB_SLOTS="${RB_SLOTS:-19}"; RB_WIN="${RB_WIN:-200}"; RB_CUDA="${RB_CUDA:-1}"
RB_MARLIN="${RB_MARLIN:-0}"; RB_SWA="${RB_SWA:-1}"; RB_SEQS="${RB_SEQS:-4}"
RB_MBT="${RB_MBT:-8192}"; RB_LEN="${RB_LEN:-524288}"; RB_UTIL="${RB_UTIL:-0.9}"
RB_CGM="${RB_CGM:-FULL_DECODE_ONLY}"

RB=( "MODEL_DIR=$MODEL_DIR" "TAG=$TAG" "PORT=$PORT" "GPU_UTILIZATION=$RB_UTIL" "MAX_NUM_SEQS=$RB_SEQS"
     "MAX_NUM_BATCHED_TOKENS=$RB_MBT" "MAXLEN=$RB_LEN" "SPEC=1"
     "PIN=1" "PIN_SLOTS=$RB_SLOTS" "PIN_WINDOW=$RB_WIN" "PIN_CUDA_MOE=$RB_CUDA"
     "MARLIN=$RB_MARLIN" "SWA_BOUNDED_REPLAY=$RB_SWA" "NO_PREFIX_CACHE=0" "PC=0"
     "EXPANDABLE=1" "CUDAGRAPH_MODE=$RB_CGM" "WARMUP=1" )
[ -n "${RB_TP:-}" ] && RB+=("TP=$RB_TP")
RB+=("THREADS=${RB_THREADS:-60}")

if [ "$MODE" = "run" ]; then
    umask 077
    install -m 600 /dev/null "$SNAP" 2>/dev/null || true
    { echo "# nightly rollback snapshot $(date '+%F %T') source_pid=$CUR_PID"
      for kv in "${RB[@]}"; do echo "$kv"; done
      echo "API_KEY_STATE=$API_KEY_STATE"; } > "$SNAP" || die "无法写回滚快照 $SNAP"
    chmod 600 "$SNAP" 2>/dev/null || true
fi

say "回滚基线：slots=$RB_SLOTS window=$RB_WIN cuda_moe=$RB_CUDA cgm=$RB_CGM auth=$API_KEY_STATE"
say "目标基线：slots=19 window=200 cuda_moe=1 cgm=FULL_DECODE_ONLY"

if [ "$MODE" = "preflight" ]; then say "✅ 前置检查通过（未触碰服务，未写状态文件）"; exit 0; fi
if [ "$MODE" = "dryrun" ]; then
    say "DRY-RUN 目标：serve_ds41.sh restart ${TARGET[*]}"
    say "DRY-RUN 回滚：serve_ds41.sh restart ${RB[*]}"
    exit 0
fi

# ---- 验收 -------------------------------------------------------------------
verify_profile(){
    local p; p="$(cat "$PIDFILE" 2>/dev/null || true)"
    pid_alive "$p" || { say "验收失败：PID 文件无有效进程"; return 1; }
    local a b c d e
    a="$(getenv_of "$p" XIAOTU_PIN_CUDA_MOE)"; b="$(getenv_of "$p" XIAOTU_PIN_MAX_PER_LAYER)"
    c="$(getenv_of "$p" XIAOTU_PIN_WINDOW)";   d="$(getenv_of "$p" CUDAGRAPH_MODE)"
    e="$(getenv_of "$p" DS41_API_KEY_STATE)"; [ -z "$e" ] && e="unset"
    say "实跑口径：cuda_moe=${a:-?} slots=${b:-?} window=${c:-?} cgm=${d:-?} auth=${e} (PID=$p)"
    [ "$a" = "$1" ] && [ "$b" = "$2" ] && [ "$c" = "$3" ] && [ "$d" = "$4" ] && [ "$e" = "$5" ] && return 0
    say "验收失败：期望 cuda_moe=$1 slots=$2 window=$3 cgm=$4 auth=$5"; return 1
}
wait_health(){ local t0=$SECONDS; while [ $((SECONDS-t0)) -lt "$HEALTH_TIMEOUT" ]; do
    [ "$(health)" = "200" ] && return 0; sleep "$POLL"; done; say "health 超时（${HEALTH_TIMEOUT}s）"; return 1; }

# 记录本轮起点（日志偏移 + 时间戳），供增量验收
mark_start(){ START_TS="$(date +%s)"
    LOGM="$RUN/ds41_${TAG}.log"; LOGS="$RUN/ds41_${TAG}_smoke.log"
    LOGE="$RUN/ds41_${TAG}_smoke.err"; LOGW="$RUN/ds41_${TAG}_warmup.log"
    OFF_MAIN="$(foff "$LOGM")";   INO_MAIN="$(fino "$LOGM")"
    OFF_SMOKE="$(foff "$LOGS")";  INO_SMOKE="$(fino "$LOGS")"
    OFF_SERR="$(foff "$LOGE")";   INO_SERR="$(fino "$LOGE")"
    OFF_WARM="$(foff "$LOGW")";   INO_WARM="$(fino "$LOGW")"
    say "验收起点 ts=$START_TS 偏移 main=$OFF_MAIN smoke=$OFF_SMOKE serr=$OFF_SERR warm=$OFF_WARM"; }

accept_async(){
    local busy="$RUN/ds41_${TAG}.smoke.busy" failf="$RUN/ds41_${TAG}.smoke.fail" t0=$SECONDS
    # ① 等冒烟开始（避免"尚未创建就跳过"）；若 60s 内没开始，按无冒烟继续（仍受后续日志检查）
    while [ ! -e "$busy" ] && [ $((SECONDS-t0)) -lt 60 ]; do sleep 5; done
    # ② 等冒烟结束
    while [ -e "$busy" ] && [ $((SECONDS-t0)) -lt "$ASYNC_TIMEOUT" ]; do sleep "$POLL"; done
    [ -e "$busy" ] && { say "验收失败：冒烟未在 ${ASYNC_TIMEOUT}s 内收尾"; return 1; }
    if [ -e "$failf" ] && [ "$(fmtime "$failf")" -ge "$START_TS" ]; then
        say "验收失败：冒烟失败标记 $(head -1 "$failf")"; return 1; fi
    # ③ 等预热完成行（只看本轮增量）
    local woff; woff="$(eff_off "$LOGW" "$OFF_WARM" "$INO_WARM")"
    while [ $((SECONDS-t0)) -lt "$ASYNC_TIMEOUT" ]; do
        woff="$(eff_off "$LOGW" "$OFF_WARM" "$INO_WARM")"
        fdelta "$LOGW" "$woff" > "$RUN/.nightly_warm_delta" 2>/dev/null
        grep -a "\[warmup\] 完成: ok=" "$RUN/.nightly_warm_delta" >/dev/null 2>&1 && break
        sleep "$POLL"
    done
    local wline; wline="$(grep -a "\[warmup\] 完成: ok=" "$RUN/.nightly_warm_delta" 2>/dev/null | tail -1)"
    if [ -z "$wline" ]; then say "验收失败：预热未在 ${ASYNC_TIMEOUT}s 内出现完成行（本轮）"; return 1; fi
    say "预热（本轮）：$wline"
    case "$wline" in *"fail=0"*) ;; *) say "验收失败：本轮预热存在失败请求"; return 1 ;; esac
    # ④ 本轮增量里的致命行
    local f off ino nm eoff
    for pair in "main:$LOGM:$OFF_MAIN:$INO_MAIN" "smoke:$LOGS:$OFF_SMOKE:$INO_SMOKE" \
                "smoke.err:$LOGE:$OFF_SERR:$INO_SERR" "warmup:$LOGW:$OFF_WARM:$INO_WARM"; do
        nm="${pair%%:*}"; rest="${pair#*:}"; f="${rest%%:*}"; rest="${rest#*:}"
        off="${rest%%:*}"; ino="${rest##*:}"
        [ -e "$f" ] || continue
        eoff="$(eff_off "$f" "$off" "$ino")"
        fdelta "$f" "$eoff" > "$RUN/.nightly_delta" 2>/dev/null
        if grep -aE "Traceback \(most recent call last\)|CUDA error|illegal memory access|Xid [0-9]" \
              "$RUN/.nightly_delta" >/dev/null 2>&1; then
            say "验收失败：$nm 日志本轮出现致命行（起点=$eoff）"; return 1; fi
    done
    rm -f "$RUN/.nightly_delta" "$RUN/.nightly_warm_delta" 2>/dev/null || true
    say "✅ 异步验收通过（冒烟无失败标记、预热 fail=0、无致命行）"
    return 0
}

# 8>&- ：关掉本脚本的 flock fd，否则长驻 vLLM 会继承并永久持锁（次轮 flock 必失败）
run_restart(){ say "启动：${1} 档 ${@:2}"; API_KEY="$API_KEY" bash scripts/serve_ds41.sh restart "${@:2}" 8>&- >>"$LOG" 2>&1; }
accept_all(){ local cu="$1" sl="$2" wi="$3" cg="$4"
    [ "$(health)" = "200" ] || { say "验收失败：health=$(health)"; return 1; }
    verify_profile "$cu" "$sl" "$wi" "$cg" "$API_KEY_STATE" || return 1
    accept_async || return 1
    return 0; }

# ---- 目标档 -----------------------------------------------------------------
mark_start
run_restart target "${TARGET[@]}"; RC=$?
TARGET_OK=0
if [ "$RC" = "0" ] && wait_health && accept_all 1 19 200 FULL_DECODE_ONLY; then TARGET_OK=1; fi
if [ "$TARGET_OK" = "1" ]; then
    say "✅ 目标口径成功"
    rm -f "$FAILEDF" "$ALERT" "$DEGRADED" "$RUN/nightly_pending"; write_last 0 "target_ok"; exit 0
fi

NOW_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
if [ "$NOW_PID" = "$CUR_PID" ] && [ "$(health)" = "200" ]; then
    say "ℹ️ 旧实例仍在运行（PID 未变）⇒ 未发生实际重启，跳过回滚"
    write_last 12 "aborted_no_change"; exit 12
fi

# ---- 回滚 -------------------------------------------------------------------
say "⏳ 目标档失败（rc=$RC health=$(health)）⇒ 回滚到重启前口径"
for attempt in 1 2; do
    say "回滚尝试 $attempt/2"
    mark_start
    run_restart rollback "${RB[@]}"; RC=$?
    if [ "$RC" = "0" ] && wait_health && accept_all "$RB_CUDA" "$RB_SLOTS" "$RB_WIN" "$RB_CGM"; then
        say "✅ 回滚成功（口径与重启前一致）"
        rm -f "$FAILEDF" "$ALERT"
        printf 'DEGRADED: 目标档失败，已回滚（%s）\n' "$(date '+%F %T')" > "$DEGRADED" 2>/dev/null || true
        write_last 10 "rollback_ok"; exit 10
    fi
    say "⚠️ 回滚尝试 $attempt 失败（rc=$RC health=$(health)）"
    [ "$attempt" = "1" ] && sleep 60
done

mark_failed "rollback_failed"     # 先落告警，再等待重试（可见性）
say "⏳ 回滚两轮失败，已落 FAILED/ALERT；30 分钟后最后重试一次"
sleep 1800
mark_start
run_restart rollback-delayed "${RB[@]}"; RC=$?
if [ "$RC" = "0" ] && wait_health && accept_all "$RB_CUDA" "$RB_SLOTS" "$RB_WIN" "$RB_CGM"; then
    say "✅ 延迟回滚成功"
    rm -f "$FAILEDF" "$ALERT"
    printf 'DEGRADED: 目标档失败，延迟回滚成功（%s）\n' "$(date '+%F %T')" > "$DEGRADED" 2>/dev/null || true
    write_last 10 "rollback_ok_delayed"; exit 10
fi
say "❌ 回滚失败：8700 不可用，需人工介入"
mark_failed "rollback_failed_final"
write_last 11 "rollback_failed"; exit 11
