#!/usr/bin/env bash
# ============================================================================
# resmon.sh —— 资源上限全量采样（**只读**；不杀进程、不改配置）
#
# 目的：回答"生产实例跑几小时后失败，是不是 OS/驱动资源耗尽"。
# 归属判定：api 按【端口 8700 归属】；EngineCore/Worker 按【父链包含 api】筛选
#           —— ⛔ 禁止按进程名/命令行模式匹配（同机会有多个实例）。
# 口径：每分钟一行 CSV；只追加，不轮转（便于回看死亡前 10–30 分钟的趋势）。
# 用法： setsid nohup bash scripts/resmon.sh [out.csv] >/dev/null 2>&1 </dev/null &
# ============================================================================
set -uo pipefail
OUT=${1:-$HOME/lvllm/xtu-oct28-tools/logs/resmon/resmon.csv}
PORT=${RESMON_PORT:-8700}
mkdir -p "$(dirname "$OUT")"

if [ ! -f "$OUT" ]; then
  echo "ts,api,eng,mapA,mapE,mapW,fdA,fdE,fdW,thrA,thrE,thrW,rssA,rssE,rssW,pinA,pinE,swpA,majfltE,minfltE,shmf,shmMB,shmMaxKB,nvmapA,nvmapE,nvmapW,irq_nv,tcp,unix,psicpu,psimem,psio,memav,commit,committed,dirty,slab,load1,g1_mib,g1_util,g1_temp,g1_watt,g1_thr,g2_mib,g2_util,g2_temp,g2_watt,g2_thr" > "$OUT"
fi

ppid_of() { awk '{print $4}' "/proc/$1/stat" 2>/dev/null; }

descends() {   # $1=candidate pid  $2=ancestor pid
  local p=$1 i=0
  while [ -n "$p" ] && [ "$p" != "0" ] && [ $i -lt 12 ]; do
    [ "$p" = "$2" ] && return 0
    p=$(ppid_of "$p"); i=$((i + 1))
  done
  return 1
}

maps_n()  { [ -n "${1:-}" ] && wc -l < "/proc/$1/maps" 2>/dev/null || echo 0; }
fd_n()    { [ -n "${1:-}" ] && ls "/proc/$1/fd" 2>/dev/null | wc -l || echo 0; }
stat_f()  { [ -n "${1:-}" ] && awk "/^$2:/{print \$2}" "/proc/$1/status" 2>/dev/null || echo 0; }
stat_mb() { [ -n "${1:-}" ] && awk -v f="$2" '$1==f":"{printf "%.1f", $2/1048576}' "/proc/$1/status" 2>/dev/null || echo 0; }
mem_kb()  { awk -v f="$1" '$1==f":"{printf "%.0f", $2/1048576}' /proc/meminfo; }
nv_maps() { [ -n "${1:-}" ] && grep -c nvidia "/proc/$1/maps" 2>/dev/null || echo 0; }
psi10()   { awk '$1=="some"{for(i=1;i<=NF;i++) if($i ~ /^avg10=/){sub("avg10=","",$i); print $i; exit}}' "/proc/pressure/$1" 2>/dev/null; }

while :; do
  A=$(ss -ltnp 2>/dev/null | grep ":$PORT " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  E=""; W=""
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); do descends "$p" "$A" && E=$p && break; done
  for p in $(pgrep -f 'VLLM::Worker_TP' 2>/dev/null); do descends "$p" "$A" && W="$W $p"; done

  mw=0; fw=0; tw=0; rw=0.0; nw=0
  for p in $W; do
    mw=$((mw + $(maps_n "$p"))); fw=$((fw + $(fd_n "$p"))); tw=$((tw + $(stat_f "$p" Threads)))
    rw=$(awk -v a="$rw" -v b="$(stat_mb "$p" VmRSS)" 'BEGIN{printf "%.1f", a+b}')
    nw=$((nw + $(nv_maps "$p")))
  done

  maj=$([ -n "$E" ] && awk '{print $12}' "/proc/$E/stat" 2>/dev/null || echo 0)
  mnr=$([ -n "$E" ] && awk '{print $10}' "/proc/$E/stat" 2>/dev/null || echo 0)
  shmmax=$( (cd /dev/shm 2>/dev/null && stat -c %s * 2>/dev/null | sort -n | tail -1) || echo 0 )
  irq=$(awk '/nvidia|nvme|mlx/{for(i=2;i<=NF;i++) if($i ~ /^[0-9]+$/) s+=$i} END{print s+0}' /proc/interrupts 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null)
  g1=$(printf '%s\n' "$gpu" | awk -F', *' '$1=="1"{printf "%s,%s,%s,%s,%s",$2,$3,$4,$5,$6}')
  g2=$(printf '%s\n' "$gpu" | awk -F', *' '$1=="2"{printf "%s,%s,%s,%s,%s",$2,$3,$4,$5,$6}')

  echo "$(date -u +%H:%M:%S),${A:-0},${E:-0},$(maps_n "$A"),$(maps_n "$E"),$mw,$(fd_n "$A"),$(fd_n "$E"),$fw,$(stat_f "$A" Threads),$(stat_f "$E" Threads),$tw,$(stat_mb "$A" VmRSS),$(stat_mb "$E" VmRSS),$rw,$(stat_mb "$A" VmPin),$(stat_mb "$E" VmPin),$(stat_mb "$A" VmSwap),${maj:-0},${mnr:-0},$(ls /dev/shm 2>/dev/null | wc -l),$(du -sm /dev/shm 2>/dev/null | awk '{print $1}'),${shmmax:-0},$(nv_maps "$A"),$(nv_maps "$E"),$nw,${irq:-0},$(ss -s 2>/dev/null | awk '/^TCP:/{print $2}'),$(wc -l < /proc/net/unix),$(psi10 cpu),$(psi10 memory),$(psi10 io),$(mem_kb MemAvailable),$(mem_kb CommitLimit),$(mem_kb Committed_AS),$(mem_kb Dirty),$(mem_kb Slab),$(cut -d' ' -f1 /proc/loadavg),$g1,$g2" >> "$OUT"
  sleep "${RESMON_INTERVAL:-60}"
done
