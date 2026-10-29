#!/usr/bin/env bash
# ============================================================================
# imbalance_watch.sh —— 两卡"不同步"告警器（**只读**：nvidia-smi + py-spy）
#
# 来由（2026-10-29，用户观察）：有几次出错前，两张卡的占用/进度已经不平衡。
#   日志侧证据：r108 死前 14 条 CPU-MoE 进度全是 `L0`，且两个 rank 的 `call` 计数漂移
#   （235 vs 234、230 vs 229），而 r106/r107 死前是 `L0↔L20` 成对且计数同步。
#
# 判据（任一成立即"不平衡"）
#   ① 两卡已用显存差 |g1-g2| > LIMIT_MIB 连续 PERSIST 次
#   ② 两卡 GPU 利用率差 > UTIL_DIFF 连续 PERSIST 次
# 命中动作：抓 nvidia-smi 全量 + 两个 worker 的 py-spy 栈 + 日志尾，**不杀任何进程**
#
# 用法： setsid nohup bash scripts/imbalance_watch.sh <日志> [...] > out 2>&1 </dev/null &
# ============================================================================
set -uo pipefail
LOG=${1:?需给一个 vllm 日志路径}
LIMIT_MIB=${IMB_LIMIT_MIB:-1024}
UTIL_DIFF=${IMB_UTIL_DIFF:-40}
PERSIST=${IMB_PERSIST:-3}
OUT=${IMB_OUT:-$HOME/lvllm/xtu-oct28-tools/logs/poison}
mkdir -p "$OUT"
GPUS=${IMB_GPUS:-1,2}
HIT=0
STAMP=$(date +%Y%m%d-%H%M%S)

while :; do
  read -r M1 M2 <<<"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null | tr '\n' ' ')"
  read -r U1 U2 <<<"$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null | tr '\n' ' ')"
  DM=$(( ${M1:-0} - ${M2:-0} )); DM=${DM#-}
  DU=$(( ${U1:-0} - ${U2:-0} )); DU=${DU#-}

  if [ "$DM" -gt "$LIMIT_MIB" ] || [ "$DU" -gt "$UTIL_DIFF" ]; then
    HIT=$((HIT+1))
  else
    HIT=0
  fi

  if [ "$HIT" -ge "$PERSIST" ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    F="$OUT/$STAMP.imbalance"
    {
      echo "[imbalance] $(date -Is) 两卡不平衡持续 ${HIT} 次: mem=${M1}/${M2} MiB (差 ${DM}) util=${U1}/${U2}% (差 ${DU})"
      echo "=== nvidia-smi ==="
      timeout 8 nvidia-smi 2>&1 | head -30
      echo "=== 进程 ==="
      timeout 8 ps -eo pid,ppid,etime,stat,rss,args 2>/dev/null | grep -E "[V]LLM::" | cut -c1-140
      echo "=== 日志尾 120 行 ==="
      timeout 8 tail -120 "$LOG" | cut -c1-200
    } > "$F.env" 2>&1
    for p in $(timeout 8 ps -eo pid,args 2>/dev/null | grep -E "[V]LLM::Worker_TP" | awk '{print $1}'); do
      timeout 25 "$HOME/anaconda3/envs/vllm-xiaotu-moe/bin/py-spy" dump --pid "$p" \
        > "$F.pyspy.$p.txt" 2>&1 || echo "[imbalance] py-spy $p 失败" >> "$F.env"
    done
    echo "[imbalance] 证据写入 $F.{env,pyspy.*}" >> "$F.env"
    HIT=0                      # 记录后重新计数，避免狂刷
    sleep 60
  fi
  sleep 10
done
