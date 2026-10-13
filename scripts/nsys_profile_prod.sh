#!/usr/bin/env bash
# nsys_profile_prod.sh — 窗口用途:用 nsys 会话方式启动一次生产,就绪+加载流量后采 12s GPU 内核账本。
# 纪律(R38):本脚本【不含任何 kill/stop 进程逻辑】;停止一律由调用方按端口派生 PID + 进程组执行。
# nsys 生命周期(2023.1 实测):launch=注入不采集 / start=开录 / stop=落盘 rep,目标进程存活;cancel 未实现 ⇒ 禁用。
# 用法(远端,前提:8070 未监听):bash scripts/nsys_profile_prod.sh
set -uo pipefail
SCR=/home/user/lvllm-tmp/p0
[ -d "$SCR" ] || { echo "missing $SCR"; exit 1; }
[ -f "$SCR/quick.sh" ] || { echo "missing $SCR/quick.sh"; exit 1; }
command -v nsys >/dev/null || { echo "no nsys"; exit 1; }

# 占用门:8070 有监听 ⇒ 拒绝启动(绝不 kill)
if ss -ltn | grep -q ':8070'; then
  echo "REFUSE: 8070 already listening — stop production first (by owner, not by this script)"; exit 1
fi

cd /home/user/lvllm/vllm-xiaotu-moe || { echo "repo missing"; exit 1; }

# 前台:nsys launch 注入 bringup(参数/环境原样透传;launch 态无采集开销)
setsid nsys launch --session=xtu1 -t cuda --output="$SCR/nsys_prod" \
  env NSYS_WINDOW=1 bash scripts/bringup_prod_8070.sh < /dev/null > "$SCR/nsys_prod_boot.log" 2>&1 &

# 采集器(后台):等就绪(端口 + 新鲜 tag 日志含"全部形状预热完成")⇒ 起负载 ⇒ 采 12s ⇒ 落盘。
setsid bash -c '
  set -u
  P='$SCR'
  log(){ echo "[$(date +%H:%M:%S)] $*" >> "$P/nsys_prod_collect.log"; }
  for i in $(seq 1 90); do
    sleep 20
    ss -ltn 2>/dev/null | grep -q ":8070" || continue
    TL=$(readlink -f /home/user/lvllm/vllm-xiaotu-moe/dev-docs/report/tuning/logs/v41_8070.log 2>/dev/null)
    if [ -n "$TL" ] && grep -aq "全部形状预热完成" "$TL" 2>/dev/null; then
      # 新鲜度:tag 日志 20 分钟内被写过(防陈日志假就绪)
      [ "$(stat -c %Y "$TL")" -gt $(( $(date +%s) - 1200 )) ] && break
    fi
  done
  ss -ltn | grep -q ":8070" || { log "TIMEOUT waiting ready"; exit 1; }
  log "ready; starting load"
  ( cd /home/user/tmp/c4rerun_20261012 && bash quick.sh nsys > logs/nsysq.out 2>&1 ) &
  BPID=$!
  sleep 90                      # warm 轮 + 进 C1 正式轮中段
  timeout 60 nsys start --session=xtu1 >> "$P/nsys_prod_collect.log" 2>&1 || { log "nsys start FAILED"; exit 1; }
  sleep 12
  timeout 60 nsys stop  --session=xtu1 >> "$P/nsys_prod_collect.log" 2>&1 || log "nsys stop FAILED"
  wait $BPID 2>/dev/null
  log "bench done; rep should be $P/nsys_prod*.nsys-rep"
' </dev/null >/dev/null 2>&1 &

echo "armed: boot log=$SCR/nsys_prod_boot.log collect log=$SCR/nsys_prod_collect.log"
exit 0
