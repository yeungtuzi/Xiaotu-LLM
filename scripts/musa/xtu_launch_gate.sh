#!/usr/bin/env bash
# ============================================================================
# xtu_launch_gate.sh —— 起实例前的【只检查】门禁（默认不杀任何进程）
#
# 来由：2026-10-22 事故（USER_QA_LEDGER #288）——上一实例未退干净就起新的，
#       两个 EngineCore × 60 线程抢 32 核，load average 80，测量全废。
#
# 用法：
#   bash scripts/musa/xtu_launch_gate.sh --need-gib 900
#   bash scripts/musa/xtu_launch_gate.sh --need-gib 900 --kill-orphans   # 显式清理孤儿
#
# 退出码：0 可启动 / 3 不可启动（任一检查不过，**fail-closed**）
# ============================================================================
set -uo pipefail

need_gib=900
kill_orphans=0
while [ $# -gt 0 ]; do
  case "$1" in
    --need-gib) need_gib="${2:-}"; shift 2 ;;
    --kill-orphans) kill_orphans=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

fail=0
say() { printf '%s\n' "$*"; }
bad() { printf '  ⛔ %s\n' "$*"; fail=1; }
ok()  { printf '  ✓ %s\n' "$*"; }

say "== 1) GPU 进程 =="
gpu_procs=$(timeout 8 mthreads-gmi 2>/dev/null | awk '/^Processes:/{f=1;next} f && /^[0-9]+/ {print $1}' || true)
if [ -z "$gpu_procs" ]; then ok "GPU 上零进程"; else bad "GPU 仍有进程: $gpu_procs"; fi

say "== 2) EngineCore（含孤儿）=="
# ⚠️ 不能用 `ps | grep VLLM::EngineCore`：那会命中【本门禁自己的命令行】。
# 只认 /proc/<pid>/comm（EngineCore 用 prctl 设置，comm 被内核截到 15 字符）。
ec_pids() {
  for d in /proc/[0-9]*; do
    c=$(cat "$d/comm" 2>/dev/null) || continue
    case "$c" in "VLLM::EngineCor"*) basename "$d" ;; esac
  done
}
ec=""
for p in $(ec_pids); do ec="$ec $p(ppid=$(awk '{print $4}' /proc/$p/stat 2>/dev/null))"; done
if [ -z "$ec" ]; then
  ok "零 EngineCore"
else
  bad "仍有 EngineCore:$ec"
  if [ "$kill_orphans" -eq 1 ]; then
    for p in $(ec_pids); do
      ppid=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
      [ "$ppid" = "1" ] && kill -9 "$p" && say "    已清理孤儿 $p"
    done
    say "    --kill-orphans 已执行；请【重跑本门禁】确认"
  fi
fi

say "== 3) host 内存 =="
avail_gib=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gib" -ge "$need_gib" ]; then
  ok "MemAvailable ${avail_gib} GiB >= need ${need_gib} GiB"
else
  bad "MemAvailable ${avail_gib} GiB < need ${need_gib} GiB"
fi

say "== 4) 线程预算 =="
cores=$(nproc)
threads_phys=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); c=$2} /^Socket\(s\)/{gsub(/ /,"",$2); s=$2} END{print c*s}')
say "  nproc=${cores}  物理核=${threads_phys:-?}  XIAOTU_MOE_THREADS=${XIAOTU_MOE_THREADS:-unset}"
if [ "${XIAOTU_MOE_THREADS:-0}" -gt "${cores}" ] 2>/dev/null; then
  bad "XIAOTU_MOE_THREADS 超过可用线程数 ${cores}"
else
  ok "线程预算未超配"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAIL —— 不允许启动实例"
  exit 3
fi
echo "RESULT: PASS —— 可以启动（单实例）"
