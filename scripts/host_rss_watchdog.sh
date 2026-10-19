#!/usr/bin/env bash
# host_rss_watchdog.sh — 服务实例的【宿主 RSS 看门狗】(2026-10-19 用户明令)。
#
# 目的:2026-10-19 两个调试实例各 ~472 GiB anon-rss 把 1.5 TiB 机器拖到 OOM 死机。
# ⇒ 起服务后必须挂本看门狗:每 N 秒采样实例的 RSS,超阈值先杀自己,不等内核。
#
# ⭐ 归属用【进程组 PGID】而不是"根进程的后代":
#   根死后子进程会被 reparent 到 init,`/proc/<pid>/task/*/children` 立刻看不见它们 ✗;
#   而 PGID 会被子进程继承(除非它自己 setsid)⇒ 孤儿仍在同一 PGID 里 ✓
#
# 身份判据(⛔ 不用进程名做 kill):
#   ① 根 PID 只来自 `<logs>/<tag>.pid`,且 >1、活、非僵尸;
#   ② cmdline 含 `vllm.entrypoints.openai.api_server`(或 `VLLM::`);
#   ③ environ 含 `XIAOTU_ENV_FILE=<logs>/<tag>.envfile`(同 uid 可读,否则拒绝托管);
#   ④ 取根的 PGID,要求 PGID>=2、≠看门狗自身 PGID、≠1;记录根 starttime 防 PID 复用;
#   ⑤ 枚举 = `ps -eo pid,pgid,stat` 中 PGID 相同者;kill 前逐个复核 starttime。
#
# 用法(远端后台,别占 ssh):
#   nohup bash scripts/host_rss_watchdog.sh --tag qfn_tp2_pin16 --limit-gib 350 \
#       > /tmp/wd_qfn.out 2>&1 < /dev/null &
#
# 退出:0 实例已完全退出 / 3 超限已击杀且确认清空 / 5 击杀后仍有存活 / 2 身份或参数错
set -euo pipefail

TAG=""
LIMIT_GIB=""
INTERVAL=10
GRACE=5
KILL=0          # ⭐ 默认**只告警、不杀**(kill 是显式 opt-in,且要求根=进程组长)

die() { local rc="$1"; shift; echo "[wd] ⛔ $*" >&2; exit "$rc"; }
say() { echo "[wd $(date +%H:%M:%S)] $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="${2:-}"; shift 2 || die 2 "--tag 缺值" ;;
    --limit-gib) LIMIT_GIB="${2:-}"; shift 2 || die 2 "--limit-gib 缺值" ;;
    --interval) INTERVAL="${2:-}"; shift 2 || die 2 "--interval 缺值" ;;
    --grace) GRACE="${2:-}"; shift 2 || die 2 "--grace 缺值" ;;
    --kill) KILL=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) die 2 "未知参数: $1" ;;
  esac
done

[ -n "$TAG" ] || die 2 "必须给 --tag"
case "$TAG" in */*|*' '*) die 2 "--tag 非法('$TAG')" ;; esac
[ -n "$LIMIT_GIB" ] || die 2 "必须给 --limit-gib(不会用默认阈值替你杀进程)"
case "$LIMIT_GIB" in ''|0|0*|*[!0-9]*) die 2 "--limit-gib 必须是正整数('$LIMIT_GIB')" ;; esac
case "$INTERVAL" in ''|0|0*|*[!0-9]*) die 2 "--interval 必须是正整数('$INTERVAL')" ;; esac
case "$GRACE" in ''|0|0*|*[!0-9]*) die 2 "--grace 必须是正整数('$GRACE')" ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/dev-docs/report/tuning/logs"
PIDFILE="$LOGDIR/${TAG}.pid"
ENVFILE="$LOGDIR/${TAG}.envfile"
RSSLOG="$LOGDIR/${TAG}.rss.log"
mkdir -p "$LOGDIR"

_self_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
case "${_self_pgid:-}" in ''|*[!0-9]*) die 2 "读不到看门狗自身 PGID ⇒ 拒绝托管" ;; esac

starttime_of() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null || true; }
state_of() { awk '{print $3}' "/proc/$1/stat" 2>/dev/null || true; }

rss_kib() {
  local v
  v="$(awk '/^VmRSS:/{print $2}' "/proc/$1/status" 2>/dev/null || true)"
  case "${v:-}" in
    ''|*[!0-9]*)
      if [ -d "/proc/$1" ]; then die 2 "进程 $1 仍在但读不到 VmRSS ⇒ 拒绝继续(不许当 0)"; fi
      echo 0 ;;
    *) echo "$v" ;;
  esac
}

# PGID 枚举:输出 "pid:starttime"(排除僵尸;看门狗自身按 PID 排除)
snapshot_pg() {
  local pg="$1" p st out=""
  while read -r p; do
    [ -n "$p" ] || continue
    [ "$p" = "$$" ] && continue
    [ "$(state_of "$p")" = "Z" ] && continue
    st="$(starttime_of "$p")"
    [ -n "$st" ] || continue
    out="$out $p:$st"
  done < <(ps -eo pid=,pgid=,stat= 2>/dev/null | awk -v g="$pg" '$2==g{print $1}')
  # ⚠️ 进程替换里的 ps 退出码拿不到 ⇒ 单独再验一次"ps 是否可用"(失败即拒绝,不许当成 0 成员)
  ps -eo pid= >/dev/null 2>&1 || die 2 "ps 不可用 ⇒ 无法枚举进程,拒绝继续"
  printf '%s\n' "$out"
}

kill_snapshot() {  # $1 = "pid:start ...";先 TERM,grace 后 KILL(均复核 starttime)
  local snap="$1" item p st cur
  for item in $snap; do
    p="${item%%:*}"; st="${item##*:}"
    [ -d "/proc/$p" ] || continue
    cur="$(starttime_of "$p")"
    [ "$cur" = "$st" ] || { say "⚠️ pid=$p starttime 变了($st→$cur)⇒ 跳过"; continue; }
    kill -TERM "$p" 2>/dev/null || say "⚠️ TERM pid=$p 失败"
  done
  sleep "$GRACE"
  for item in $snap; do
    p="${item%%:*}"; st="${item##*:}"
    [ -d "/proc/$p" ] || continue
    cur="$(starttime_of "$p")"
    [ "$cur" = "$st" ] || continue
    [ "$(state_of "$p")" = "Z" ] && continue
    kill -KILL "$p" 2>/dev/null || say "⚠️ KILL pid=$p 失败"
  done
}

# ---- 身份钉死(一次)----
[ -f "$PIDFILE" ] || die 2 "找不到 PID 文件 $PIDFILE"
_root="$(cat "$PIDFILE" 2>/dev/null || true)"
case "${_root:-}" in ''|0|1|*[!0-9]*) die 2 "PID 文件内容非法('${_root:-}')⇒ 拒绝托管" ;; esac
[ -d "/proc/${_root}" ] || die 2 "PID ${_root} 已不在 ⇒ 拒绝托管(请人工核对)"
[ "$(state_of "$_root")" != "Z" ] || die 2 "PID ${_root} 是僵尸 ⇒ 拒绝托管"
_cmd="$(tr '\0' ' ' < "/proc/${_root}/cmdline" 2>/dev/null || true)"
case "$_cmd" in
  *vllm.entrypoints.openai.api_server*|*VLLM::*) : ;;
  *) die 2 "PID ${_root} 的 cmdline 不像 vllm 服务('${_cmd:0:120}')⇒ 拒绝托管" ;;
esac
[ -r "/proc/${_root}/environ" ] || die 2 "读不到 /proc/${_root}/environ ⇒ 无法确认归属,拒绝托管"
tr '\0' '\n' < "/proc/${_root}/environ" 2>/dev/null | grep -qxF "XIAOTU_ENV_FILE=${ENVFILE}" \
  || die 2 "PID ${_root} 的 environ 不含 XIAOTU_ENV_FILE=${ENVFILE} ⇒ 不属于 tag=${TAG},拒绝托管"
_root_st="$(starttime_of "$_root")"
case "${_root_st:-}" in ''|*[!0-9]*) die 2 "读不到 PID ${_root} 的 starttime ⇒ 拒绝托管" ;; esac
PG="$(ps -o pgid= -p "$_root" 2>/dev/null | tr -d ' ')"
case "${PG:-}" in ''|*[!0-9]*) die 2 "读不到根进程 PGID ⇒ 拒绝托管" ;; esac
[ "$PG" -ge 2 ] || die 2 "根进程 PGID=${PG}(=0/1)⇒ 不是独立进程组,拒绝托管"
[ "$PG" != "$_self_pgid" ] || die 2 "根进程与看门狗同一 PGID(${PG})⇒ 会杀到自己,拒绝托管"
say "托管 tag=$TAG root=${_root}(start=${_root_st}) PGID=${PG} limit=${LIMIT_GIB}GiB interval=${INTERVAL}s"
printf '# ts,root_alive,total_gib,pids,rss_gib_each\n' >> "$RSSLOG"

while true; do
  [ -f "$PIDFILE" ] || die 2 "PID 文件消失 ⇒ 无法确认身份"
  _cur="$(cat "$PIDFILE" 2>/dev/null || true)"
  [ "$_cur" = "$_root" ] || die 2 "PID 文件被改写(${_root} → ${_cur:-空})⇒ 拒绝继续托管"

  _alive=0
  if [ -d "/proc/${_root}" ] && [ "$(state_of "$_root")" != "Z" ]; then
    _alive=1
    _cur_st="$(starttime_of "$_root")"
    [ "$_cur_st" = "$_root_st" ] || die 2 "根 PID starttime 变了(${_root_st}→${_cur_st})⇒ PID 被复用"
  fi

  _snap="$(snapshot_pg "$PG")"
  _tot=0; _top=""; _np=0
  for item in $_snap; do
    p="${item%%:*}"; k="$(rss_kib "$p")"
    _tot=$(( _tot + k )); _top="$_top ${p}:$(( k / 1024 / 1024 ))"; _np=$(( _np + 1 ))
  done
  _gib=$(( _tot / 1024 / 1024 ))
  printf '%s,%s,%s,%s,%s\n' "$(date +%s)" "$_alive" "$_gib" "$_np" "$_top" >> "$RSSLOG"
  say "root_alive=${_alive} PGID=${PG} total=${_gib} GiB${_top}"

  if [ "$_np" = "0" ]; then
    if [ "$_alive" = "0" ]; then say "实例已完全退出(PGID ${PG} 无成员)"; exit 0; fi
    die 2 "根仍活但 PGID 枚举为空 ⇒ 身份异常,拒绝继续"
  fi

  if [ "$_tot" -gt "$(( LIMIT_GIB * 1024 * 1024 ))" ]; then   # 用 KiB 比较,避免整除截断
    if [ "$KILL" != "1" ]; then
      say "⛔ 超过阈值 ${LIMIT_GIB} GiB(默认只告警,不杀)⇒ 请按 PID 逐个处理:$(echo $_snap | tr ' ' ' ')"
      exit 3
    fi
    [ "$PG" = "$_root" ] || die 2 "根不是进程组长(pid=${_root} pgid=${PG})⇒ 不许按 PGID 击杀(会误伤同组)"
    say "⛔ 超过阈值 ${LIMIT_GIB} GiB ⇒ --kill 生效,击杀 PGID=${PG} 的 $_np 个成员"
    kill_snapshot "$_snap"
    _after="$(snapshot_pg "$PG")"
    if [ -n "$_after" ]; then
      say "⛔ 击杀后 PGID 仍有成员:${_after} ⇒ 需人工(exit 5)"
      exit 5
    fi
    say "✅ 已击杀,PGID ${PG} 已清空"
    exit 3
  fi
  sleep "$INTERVAL"
done
