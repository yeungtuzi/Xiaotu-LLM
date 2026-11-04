#!/usr/bin/env bash
# ds41f 存活看门狗（v2，2026-11-04 按独立审计 A 的报告重写）
#
# ⛔ 设计红线（审计 A 抓出的致命项，一条都不能碰回去）：
#   1. **默认只告警、不重启**。自动重启生产与"重启生产需用户当次许可"冲突
#      ⇒ 必须显式 `ALLOW_AUTO_RESTART=1` 才允许调用唯一入口 start。
#   2. **绝不删 `ds41_<TAG>.lock`**：flock 是 inode 锁，删路径 = 新进程锁新 inode = 互斥失效。
#      进程退出时内核自动释放锁，0 字节残留文件无害。
#   3. **绝不按进程名/命令行模式匹配**：归属只用 PID 文件 + 端口监听者比对。
#   4. **端口状态三态**（owned:<pid> / free / unknown）⇒ `unknown` 一律**不动作**（fail-closed）。
#   5. **健康判定必须 `--noproxy '*'`**，且必须同时满足 `端口归属 == PID 文件里的 pid`
#      （防"别的服务占了 8700 且回 200"被误判健康）。
#   6. **连续 N 次不健康才动作**（默认 3 次），避免单次抖动触发。
#   7. **永不因限速/异常退出**：限速只影响重启，检测与日志继续。
#   8. **锁打不开 ⇒ exit 66**，绝不假装"已在跑"。
#
# 用法（远端一律后台）：
#   setsid nohup bash scripts/watchdog_ds41.sh > /tmp/wd_ds41.log 2>&1 < /dev/null &
#   bash scripts/watchdog_ds41.sh --check     # 单次只读检查：0=健康 1=异常 2=启动中（cron 里 2 当 OK）
#   bash scripts/watchdog_ds41.sh --dry-run   # 单次检查，绝不重启
set -uo pipefail

TAG="${TAG:-prod}"
PORT="${PORT:-8700}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
REPO="${REPO:-/home/user/lvllm/vllm-xiaotu-moe}"
INTERVAL="${INTERVAL:-30}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
WINDOW="${WINDOW:-3600}"
ALLOW_AUTO_RESTART="${ALLOW_AUTO_RESTART:-0}"
START_GRACE="${START_GRACE:-1200}"
START_TIMEOUT="${START_TIMEOUT:-2400}"   # 单次 serve start 的最长等待（秒）
INFLIGHT_MAX=$((START_TIMEOUT + 300))   # 在飞重启记录的过期窗口（与 START_TIMEOUT 绑定）
REPEAT_ALERT="${REPEAT_ALERT:-1800}"   # 同一状态重复提醒间隔（秒）   # 启动宽限（秒）：PID 已写、端口未监听且 PID 文件很新 ⇒ STARTING
MODE="loop"

for a in "$@"; do
    case "$a" in
        --check)   MODE="once" ;;   # 只读：0=健康 1=异常 2=启动中
        --once)    MODE="once" ;;
        --dry-run) MODE="dry" ;;
        --help|-h) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
                   echo '退出码: 0=健康 1=异常 2=启动中 64=参数错误 65=参数值非法 66=环境/锁失败'; exit 0 ;;
        *) echo "未知参数: $a（--check | --once | --dry-run）" >&2; exit 64 ;;
    esac
done

LOG="${LOG:-$RUN_DIR/watchdog_${TAG}.log}"
LOCK="$RUN_DIR/watchdog_${TAG}.lock"
ALERT="$RUN_DIR/ds41_${TAG}.wd-alert"

die() { echo "[wd] ⛔ $1" >&2; exit "${2:-2}"; }
# ⚠️ 隐式契约：check_once 通过全局 last_state 回传状态，**只能在主 shell 调用**
#    （写成 $(check_once) 或放进管道，last_state 会丢在子 shell 里）
# 数值参数校验（审计 A · P1-4）：非数字会让 [ -lt ] 返回假 ⇒ 限速静默失效
for _v in INTERVAL FAIL_THRESHOLD MAX_RESTARTS WINDOW START_GRACE REPEAT_ALERT START_TIMEOUT; do
    case "${!_v}" in ''|*[!0-9]*) die "$_v 必须是非负整数（得到 '${!_v}'）" 65 ;; esac
done
[ "$INTERVAL" -ge 1 ] || die "INTERVAL 必须 ≥1" 65
[ "$FAIL_THRESHOLD" -ge 1 ] || die "FAIL_THRESHOLD 必须 ≥1" 65
[ "$WINDOW" -ge "$INTERVAL" ] || die "WINDOW 必须 ≥ INTERVAL" 65
case "$TAG" in *[!A-Za-z0-9_-]*) die "TAG 含非法字符" 65 ;; esac

mkdir -p "$RUN_DIR" || die "无法创建 $RUN_DIR" 66
# 锁：打不开 ⇒ 非 0 退出（审计 A · P0-5）
if ! exec 9>"$LOCK"; then die "无法打开锁文件 $LOCK" 66; fi
if [ "$MODE" = "loop" ]; then
    flock -n 9 || { echo "[wd] 已有一个 watchdog 在跑（$LOCK）⇒ 退出" >&2; exit 0; }
else
    # read-only（--check/--once/--dry-run）**不参与**互斥：它们永不动手，不应被 loop 挡住
    exec 9>&- 2>/dev/null || true
fi

say() {
    local line="[wd $(date '+%m-%d %H:%M:%S')] $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG" 2>/dev/null || printf '[wd] ⚠️ 无法写日志 %s\n' "$LOG" >&2
}

pidfile="$RUN_DIR/ds41_${TAG}.pid"
read_pid() { local p=""; [ -f "$pidfile" ] && p="$(tr -dc '0-9' < "$pidfile" 2>/dev/null)"; printf '%s' "$p"; }

# 活：排除僵尸/不可中断（对齐 serve 的 pid_alive）
pid_alive() {
    local p="$1" st
    [ -n "$p" ] || return 1
    st="$(sed -e 's/.*) //' -e 's/ .*//' "/proc/$p/stat" 2>/dev/null)" || return 1
    case "$st" in Z|X|x) return 1 ;; esac
    return 0
}

pidfile_age() {  # PID 文件年龄（秒）；不存在 ⇒ 极大值
    local m
    m="$(stat -c %Y "$pidfile" 2>/dev/null)" || { printf '999999999'; return; }
    printf '%s' "$(( $(date +%s) - m ))"
}

# 端口状态三态：owned:<pid> / free / unknown（fail-closed）
port_owner() {
    command -v ss >/dev/null 2>&1 || { printf 'unknown'; return; }
    local out
    out="$(ss -Hltnp "sport = :${PORT}" 2>/dev/null)" || { printf 'unknown'; return; }
    if [ -z "$out" ]; then printf 'free'; return; fi
    local p
    p="$(printf '%s' "$out" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
    case "$p" in
        ''|*[!0-9]*) printf 'unknown' ;;   # 有监听行但拿不到 pid ⇒ 不可判定（fail-closed）
        *)           printf 'owned:%s' "$p" ;;
    esac
}

health() {
    command -v curl >/dev/null 2>&1 || { printf 'nocurl'; return; }
    local c
    c="$(curl -sS --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 5 \
         "http://127.0.0.1:${PORT}/health" 2>/dev/null)" || c="000"
    printf '%s' "${c:-000}"
}

restarts_file="$RUN_DIR/watchdog_${TAG}.restarts"
restart_pid_file="$RUN_DIR/watchdog_${TAG}.restart.pid"
recent_restarts() {
    local now f t n=0
    now="$(date +%s)"
    [ -f "$restarts_file" ] || { printf '0'; return; }
    while read -r t; do
        case "$t" in ''|*[!0-9]*) continue ;; esac
        [ $(( now - t )) -ge 0 ] && [ $(( now - t )) -lt "$WINDOW" ] && n=$((n + 1))
    done < "$restarts_file"
    printf '%s' "$n"
}

consecutive=0
last_state=""
check_once() {  # 返回 0=健康 1=异常；设置 last_state
    local pid owner h
    pid="$(read_pid)"
    owner="$(port_owner)"
    if pid_alive "$pid"; then
        if [ "$owner" = "owned:$pid" ]; then
            h="$(health)"
            if [ "$h" = "200" ]; then last_state="HEALTHY pid=$pid health=200"; return 0; fi
            last_state="HUNG pid=$pid health=$h 端口归属=$owner"; return 1
        fi
        case "$owner" in
            unknown) last_state="UNKNOWN pid=$pid 存活但端口归属取不到（ss 不可用/权限）"; return 1 ;;
            free)
                # STARTING 只能在这里：**进程活着**、端口还没起来（serve 这段可达 10~15 分钟）。
                # ⛔ 绝不能放到"进程已死"分支 —— 那样"启动后 20 分钟内崩溃"会被静默吞掉
                #    （审计 A 实测 S2：pid 已死 + PID 文件新 ⇒ 误判 STARTING rc=2，无告警）。
                if [ "$(pidfile_age)" -lt "$START_GRACE" ]; then
                    last_state="STARTING pid=$pid 启动中（PID 文件 ${START_GRACE}s 宽限内）"; return 2
                fi
                last_state="HUNG pid=$pid 存活但端口 $PORT 无监听（已超启动宽限 ${START_GRACE}s）"; return 1 ;;
            *)       last_state="FOREIGN pid=$pid 端口 $PORT 被 $owner 占用"; return 1 ;;
        esac
    fi
    case "$owner" in
        owned:*) last_state="FOREIGN 无 PID 文件/进程已死，但端口 $PORT 被 $owner 占用"; return 1 ;;
        unknown) last_state="UNKNOWN 无存活 PID，端口归属取不到"; return 1 ;;
        free)    last_state="DEAD 无存活 PID，端口 $PORT 空闲"; return 1 ;;
    esac
    last_state="UNKNOWN 未判定"; return 1
}

last_alert_epoch=0
last_alert_state=""
activate() {  # 触发告警/重启
    local n now notify=1
    now="$(date +%s)"
    # 去重（审计 A · 建议 4）：状态迁移才通知；同一状态每 REPEAT_ALERT 秒才重提醒一次
    if [ "$last_state" = "$last_alert_state" ] && [ $(( now - last_alert_epoch )) -lt "$REPEAT_ALERT" ]; then
        notify=0
    fi
    say "🚨 $last_state（连续 $consecutive 次）"
    # ⛔ 只有真正要通知时才推进计时器（审计 A · D1）：
    #    以前无条件推进 ⇒ 每次 activate（默认 90s）都重置 ⇒ REPEAT_ALERT 永远到不了。
    if [ "$notify" = "1" ]; then
        last_alert_state="$last_state"; last_alert_epoch="$now"
        printf '[ds41] ⛔ 看门狗告警 %s：%s\n' "$(date '+%m-%d %H:%M:%S')" "$last_state" > "$ALERT" 2>/dev/null
    fi
    # B5：通知钩子（可选）。只有文件+日志没人看 = 8 小时事故重演 ⇒ 允许外部命令把告警送出去。
    if [ "$notify" = "1" ] && [ -n "${ALERT_CMD:-}" ]; then
        # 必须有超时：钩子一挂，主循环就停摆（审计 A · S4 实测）
        ( WD_STATE="$last_state" timeout 15 bash -c "$ALERT_CMD" ) >> "$LOG" 2>&1 \
            || say "⚠️ ALERT_CMD 执行失败或超时（不影响检测）"
    fi

    case "$last_state" in
        DEAD*)
            if [ "$ALLOW_AUTO_RESTART" != "1" ]; then
                say "ℹ️ 未开启自动重启（ALLOW_AUTO_RESTART=0）⇒ 仅告警。人工恢复：bash scripts/serve_ds41.sh start"
                return 0
            fi
            # B3：在飞保护 —— 上一轮 start 还在跑就别再起一个（否则额度被空转烧光）
            if [ -f "$restart_pid_file" ]; then
                local _rp _re _now
                read -r _rp _re < "$restart_pid_file" 2>/dev/null || true
                _now="$(date +%s)"
                _rp="${_rp//[!0-9]/}"; _re="${_re//[!0-9]/}"
                # 只有"进程还活着 **且** 时间戳没过期"才算在飞；否则视为陈旧（pid 可能已被复用）
                if [ -n "$_rp" ] && [ -n "$_re" ] && pid_alive "$_rp" \
                   && [ $(( _now - _re )) -ge 0 ] && [ $(( _now - _re )) -lt "$INFLIGHT_MAX" ]; then
                    say "ℹ️ 上一次启动仍在进行（pid=$_rp, ${_re}）⇒ 跳过本轮重启"
                    return 0
                fi
                say "ℹ️ 清理过期/失效的重启记录（pid=${_rp:-?}, epoch=${_re:-?}）"
                rm -f "$restart_pid_file" 2>/dev/null
            fi
            n="$(recent_restarts)"
            if [ "$n" -ge "$MAX_RESTARTS" ]; then
                say "⛔ ${WINDOW}s 内已重启 $n 次（上限 $MAX_RESTARTS）⇒ 本轮不重启（继续检测，不退出）"
                return 0
            fi
            say "▶️ 调用唯一入口重启（第 $((n + 1))/$MAX_RESTARTS 次）"
            # B3：子 shell 先脱掉 watchdog 锁 fd，别把它继承给 serve 的 pre-flock 窗口
            ( exec 9>&-
              cd "$REPO" || exit 1
              TAG="$TAG" PORT="$PORT" RUN_DIR="$RUN_DIR" \
                timeout "$START_TIMEOUT" setsid bash scripts/serve_ds41.sh start >> "$LOG" 2>&1 < /dev/null
              echo "[wd] start 返回码=$? " >> "$LOG"
              rm -f "$restart_pid_file" 2>/dev/null ) &
            printf '%s %s' "$!" "$(date +%s)" > "$restart_pid_file" 2>/dev/null
            date +%s >> "$restarts_file" 2>/dev/null
            ;;
        *)
            # HUNG / FOREIGN / UNKNOWN：不调 start（serve 会因 PID 文件/内存门拒绝，且 FOREIGN 更不能动）
            say "ℹ️ 该类异常不自动重启（需人工判断；自动重启只对 DEAD 生效）"
            ;;
    esac
}

if [ "$MODE" != "loop" ]; then
    check_once; rc=$?
    case "$rc" in
        0) say "OK $last_state" ;;
        2) say "$last_state" ;;
        *) say "ALERT $last_state" ;;
    esac
    exit "$rc"          # 0=健康 1=异常 2=启动中（read-only 模式永不动手）
fi

say "watchdog v2 启动：TAG=$TAG PORT=$PORT interval=${INTERVAL}s 阈值=$FAIL_THRESHOLD 自动重启=$ALLOW_AUTO_RESTART md5=$(md5sum "$0" 2>/dev/null | cut -c1-12)"
trap 'say "收到信号，退出"; exit 0' TERM INT
while :; do
    check_once; rc=$?
    case "$rc" in
        0)
            if [ "$consecutive" -gt 0 ]; then say "✅ 恢复：$last_state"; fi
            consecutive=0
            # ⛔ 必须清去重状态（审计 A · D2）：否则"崩→恢复→再崩"时，
            #    若第二次状态串与第一次相同且距上次通知 < REPEAT_ALERT，会静默不通知。
            last_alert_state=""; last_alert_epoch=0
            rm -f "$ALERT" 2>/dev/null ;;
        2)  # STARTING：不计数、不告警；只在状态迁移时打一行（否则一次启动刷 ~40 行）
            consecutive=0
            if [ "$last_state" != "$last_alert_state" ]; then say "…启动中（不告警）：$last_state"; fi
            last_alert_state="$last_state" ;;
        *)
            consecutive=$((consecutive + 1))
            if [ "$consecutive" -ge "$FAIL_THRESHOLD" ]; then
                activate
                consecutive=0
            else
                say "…异常 $consecutive/$FAIL_THRESHOLD：$last_state"
            fi ;;
    esac
    sleep "$INTERVAL"
done
