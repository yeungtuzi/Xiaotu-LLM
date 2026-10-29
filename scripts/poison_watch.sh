#!/usr/bin/env bash
# 中毒即留证（**只读**：不杀进程、不改服务，只往自己的输出目录写文件）
#
# 背景：非法访存是**异步**暴露的 —— CUDA context 被污染后，往往直到我们自己的清理线程
#       调 empty_cache() 才第一次报 "an illegal memory access was encountered"。
#       那一刻**事故已经发生**，但此时抓 py-spy 栈仍有机会看到"谁还在跑/卡在哪"。
#
# 用法:  bash scripts/poison_watch.sh <logfile> [输出目录]
# 行为:  tail -F 日志；首次命中以下任一模式即**一次性**采集并退出：
#          "empty_cache(cuda" + "失败"      清理线程报错（context 已污染）
#          "illegal memory access"          直接非法访存
#          "CUDA error"                     driver 级错误
#          "No available shared memory..."  worker 失去响应（hang，每次 60 s 超时）
#       采集内容：py-spy 栈(EngineCore + 各 Worker_TP) / nvidia-smi / 进程表 / 日志尾 200 行
# 设计约束：⛔ 不含 kill/重启；每条命令带 timeout；只在命中时写盘（平时零开销）。
set -uo pipefail
LOG="${1:?用法: poison_watch.sh <logfile> [outdir]}"
OUT="${2:-$HOME/lvllm/xtu-oct28-tools/logs/poison}"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d-%H%M%S)

# 【2026-10-29 扩充】worker 挂住时最先出现的是 shm 广播超时（每次 60 s），
# 此时 CUDA context 往往还没被污染 ⇒ 必须**在 hang 的当下**就抓栈，否则事后拿不到现场。
# ⚠️ 2026-10-29 修复：**启动期必定出现一条良性 shm 超时**，而本脚本是"命中即退出"⇒
# 旧版会在启动 2 分钟后自行退出，真正中毒时已无人看守（r107 的现场就是这样丢的）。
# 现在分两类：
#   · 硬命中（CUDA 错误/非法访存）⇒ 抓完证据退出
#   · 软命中（shm 广播超时）⇒ 抓证据但**继续盯**，最多软命中 GRACE_HITS 次后退出
PAT_HARD='empty_cache\(cuda.*失败|illegal memory access|CUDA error'
PAT_SOFT='No available shared memory broadcast block'
GRACE_SEC=${POISON_GRACE_SEC:-420}   # 启动后的宽限期：此期间的一切命中都只记录、不退出
GRACE_HITS=${POISON_GRACE_HITS:-3}   # 宽限期外，软命中累计这么多次后退出
HITS=0

# -F 跟随（日志被轮转也不断），命中即停
START_TS=$(date +%s)
timeout 90000 tail -n 0 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  HARD=0; SOFT=0
  printf '%s' "$line" | grep -qE "$PAT_HARD" && HARD=1
  printf '%s' "$line" | grep -qE "$PAT_SOFT" && SOFT=1
  if [ "$HARD" = 1 ] || [ "$SOFT" = 1 ]; then
    echo "[poison] $(date -Is) 命中: ${line:0:160}" | tee -a "$OUT/$STAMP.hit"
    {
      echo "=== 命中行 ==="; printf '%s\n' "$line"
      echo; echo "=== nvidia-smi ==="
      timeout 8 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1
      echo; echo "=== 进程表 ==="
      timeout 8 ps -eo pid,ppid,etime,stat,rss,args 2>/dev/null \
        | grep -E "[V]LLM::|[v]llm[.]entrypoints" | cut -c1-150
      echo; echo "=== 日志尾 200 行 ==="
      timeout 8 tail -200 "$LOG" | cut -c1-200
    } > "$OUT/$STAMP.env" 2>&1

    # 宿主侧 Python 栈：逐个 pid dump（py-spy 只读 /proc，不注入代码）
    for p in $(timeout 8 ps -eo pid,args 2>/dev/null | grep -E "[V]LLM::EngineCore|[V]LLM::Worker_TP" | awk '{print $1}'); do
      timeout 25 "$HOME/anaconda3/envs/vllm-xiaotu-moe/bin/py-spy" dump --pid "$p" \
        > "$OUT/$STAMP.pyspy.$p.txt" 2>&1 || echo "[poison] py-spy $p 失败(可能已退出)" >> "$OUT/$STAMP.hit"
    done
    echo "[poison] 证据已写入 $OUT/$STAMP.{hit,env,pyspy.*}" | tee -a "$OUT/$STAMP.hit"

    NOW=$(date +%s); AGE=$((NOW - START_TS)); HITS=$((HITS + 1))
    if [ "$HARD" = 1 ]; then
      echo "[poison] 硬命中(CUDA 错误)，抓完退出" | tee -a "$OUT/$STAMP.hit"
      exit 0
    fi
    if [ "$AGE" -lt "$GRACE_SEC" ]; then
      echo "[poison] 软命中且处于启动宽限期(${AGE}s<${GRACE_SEC}s) ⇒ 继续盯" | tee -a "$OUT/$STAMP.hit"
      STAMP=$(date +%Y%m%d-%H%M%S)
      continue
    fi
    if [ "$HITS" -ge "$GRACE_HITS" ]; then
      echo "[poison] 软命中累计 ${HITS} 次 ⇒ 退出（可能是真 hang）" | tee -a "$OUT/$STAMP.hit"
      exit 0
    fi
    echo "[poison] 软命中第 ${HITS} 次 ⇒ 继续盯" | tee -a "$OUT/$STAMP.hit"
    STAMP=$(date +%Y%m%d-%H%M%S)
  fi
done
