#!/usr/bin/env bash
# ============================================================================
# musa_watch.sh —— MUSA 目标机上【只读】的启动看门狗（2026-10-24 用户明令）
#
# 用户原话：
#   「长时间任务不许阻塞前台，用watchdog查看目标进程是否没起来/崩了，不准kill -9避免内存泄漏」
#
# ⛔ 本脚本的**边界（重要）**：
#    ① **绝不启动**任何进程；② **绝不 kill** 任何进程（连 TERM 都不发）。
#    ⇒ 它只做"观测 + 断言 + 落状态文件"，所以**不属于** AGENTS R38 的「起停脚本」类别。
#    ⛔ 若将来有人要给它加 kill，请先走 R38 审计（本脚本仍**不发** kill）。
#    ⚠️【2026-10-25 用户授权更新，覆盖旧结论】空转/卡死的 worker **可以 `kill -9`**（精确 PID），
#      实测清 44 GiB 显存无 panic、无需重启整机；之后须 `sleep 20` + `mthreads-gmi` 确认 VRAM≈0。
#      ⇒ 旧的「`kill -9` 永久排除」**已作废**；但**按模式匹配杀**依然永久禁止 ✗
#
# 它区分【三种死法】，而不是靠"日志停了"猜：
#   NO_START  进程没起来（pidfile 迟迟不出现 / 出现的 pid 不活）
#   CRASH     进程死了**且**日志里有崩溃特征（Traceback / FW_PANIC / CSW_TIMEOUT /
#             KILL_TIMEOUT / musaErrorUnknown / out of memory）
#   STALL     进程活着但日志长时间零增长（早于就绪串时阈值放宽，因为 profile 期本来就安静）
#   READY     命中就绪串（默认 `RUN [0-9]+ TTFT=`）
#
# 用法（**在目标机后台跑**，不许占 ssh 前台）：
#   nohup setsid bash musa_watch.sh --tag qfn_w2tn \
#       --pidfile /tmp/qfn_w2tn.pid --log /tmp/bench.log \
#       --status /tmp/qfn_w2tn.status > /tmp/qfn_w2tn.watch.out 2>&1 < /dev/null &
#   # 会话侧只看：tail -3 /tmp/qfn_w2tn.status
#
# 退出码：0 正常退出（有 READY 或无崩溃特征） / 3 CRASH / 4 STALL / 5 DEADLINE / 6 NO_START
# ============================================================================
set -uo pipefail

TAG="watch"
PIDFILE=""
LOG=""
STATUS=""
INTERVAL=10
READY_RE='RUN [0-9]+ TTFT='
STALL_SEC=1200        # 就绪后：日志零增长多少秒 = 卡住
PREREADY_STALL=1800   # 就绪前：profile/加载期本来就安静，给更长
DEADLINE=7200         # 总时长上限（2 h）
GRACE=300             # 等 pidfile 出现的时间
VRAM_EVERY=6          # 每 N 个 tick 采一次显存

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --pidfile) PIDFILE="$2"; shift 2;;
    --log) LOG="$2"; shift 2;;
    --status) STATUS="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --ready-re) READY_RE="$2"; shift 2;;
    --stall-sec) STALL_SEC="$2"; shift 2;;
    --deadline) DEADLINE="$2"; shift 2;;
    --grace) GRACE="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$PIDFILE" ] && [ -n "$LOG" ] || { echo "need --pidfile and --log" >&2; exit 2; }
[ -n "$STATUS" ] || STATUS="/tmp/${TAG}.status"

mkdir -p "$(dirname "$STATUS")" 2>/dev/null || true
: > "$STATUS"

ts() { date '+%F %T'; }
emit() { printf '%s %-10s %s\n' "$(ts)" "$1" "$2" | tee -a "$STATUS"; }

alive() {  # $1=pid ⇒ 活且非僵尸
  local p="$1" st
  [ -n "$p" ] || return 1
  [ -d "/proc/$p" ] || return 1
  st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null) || return 1
  [ "$st" != "Z" ]
}

logbytes() { stat -c %s "$LOG" 2>/dev/null || echo 0; }

crash_reason() {
  local pat
  for pat in 'FW_PANIC' 'CSW_TIMEOUT' 'KILL_TIMEOUT' 'musaErrorUnknown' \
             'Traceback (most recent call last)' 'out of memory' 'CUDA out of memory'; do
    if grep -qF "$pat" "$LOG" 2>/dev/null; then echo "$pat"; return 0; fi
  done
  return 1
}

vram() {
  mthreads-gmi -q --json 2>/dev/null \
    | /root/venvs/vllm310/bin/python -c 'import json,sys;d=json.load(sys.stdin);print([g["FB Memory Usage"]["Used"] for g in d["GPU"]])' 2>/dev/null \
    || echo "n/a"
}

emit START "tag=$TAG pidfile=$PIDFILE log=$LOG ready_re=[$READY_RE]"

t0=$(date +%s)
tick=0
ready=0
last_bytes=-1
last_grow=$(date +%s)
pid=""

while :; do
  now=$(date +%s)
  elapsed=$(( now - t0 ))
  tick=$(( tick + 1 ))

  # ---- 1) 进程身份：只从 pidfile 读，绝不按进程名匹配 ----
  if [ -z "$pid" ] && [ -f "$PIDFILE" ]; then
    pid=$(head -1 "$PIDFILE" 2>/dev/null | tr -dc '0-9')
    [ -n "$pid" ] && emit PID "$pid (from $PIDFILE)"
  fi

  if [ -z "$pid" ]; then
    if [ "$elapsed" -ge "$GRACE" ]; then
      emit NO_START "pidfile 在 ${GRACE}s 内未出现：$PIDFILE"
      exit 6
    fi
  elif ! alive "$pid"; then
    if why=$(crash_reason); then
      emit CRASH "pid=$pid 已死，日志特征：【$why】"
      tail -5 "$LOG" 2>/dev/null | sed 's/^/    | /' | tee -a "$STATUS"
      exit 3
    fi
    emit EXIT "pid=$pid 已退出，无崩溃特征（日志末尾见下）"
    grep -E "$READY_RE" "$LOG" 2>/dev/null | tail -3 | sed 's/^/    | /' | tee -a "$STATUS"
    [ "$ready" -eq 1 ] && exit 0
    emit NO_START "pid=$pid 在未就绪前退出"
    exit 6
  else
    # ---- 2) 活着的进程：看日志增长 + 就绪串 ----
    b=$(logbytes)
    if [ "$b" != "$last_bytes" ]; then
      last_bytes="$b"; last_grow=$now
    fi
    if [ "$ready" -eq 0 ] && grep -qE "$READY_RE" "$LOG" 2>/dev/null; then
      ready=1
      emit READY "命中就绪串；$(grep -E "$READY_RE" "$LOG" | tail -1)"
    fi
    idle=$(( now - last_grow ))
    lim=$STALL_SEC
    [ "$ready" -eq 0 ] && lim=$PREREADY_STALL
    if [ "$idle" -ge "$lim" ]; then
      emit STALL "pid=$pid 活着但日志已 ${idle}s 零增长（阈值 ${lim}s）"
      if why=$(crash_reason); then emit CRASH_NOTE "日志同时含【$why】"; fi
      emit NOTE "⛔ 不要 kill -9；排查顺序：mthreads-gmi -q → journalctl -k | grep -E 'FW_PANIC|CSW_TIMEOUT'"
      exit 4
    fi
  fi

  # ---- 3) 显存抽样（低频）----
  if [ $(( tick % VRAM_EVERY )) -eq 0 ]; then
    printf '%s %-10s %s\n' "$(ts)" "VRAM" "$(vram)" >> "$STATUS"
  fi

  if [ "$elapsed" -ge "$DEADLINE" ]; then
    emit DEADLINE "超过 ${DEADLINE}s 仍未结束（pid=${pid:-none}, ready=$ready）"
    exit 5
  fi

  sleep "$INTERVAL"
done
