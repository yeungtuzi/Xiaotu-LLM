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
#       采集内容：py-spy 栈(EngineCore + 各 Worker_TP) / nvidia-smi / 进程表 / 日志尾 200 行
# 设计约束：⛔ 不含 kill/重启；每条命令带 timeout；只在命中时写盘（平时零开销）。
set -uo pipefail
LOG="${1:?用法: poison_watch.sh <logfile> [outdir]}"
OUT="${2:-$HOME/lvllm/xtu-oct28-tools/logs/poison}"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d-%H%M%S)

PAT='empty_cache\(cuda.*失败|illegal memory access|CUDA error'

# -F 跟随（日志被轮转也不断），命中即停
timeout 90000 tail -n 0 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  if printf '%s' "$line" | grep -qE "$PAT"; then
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
    exit 0
  fi
done
