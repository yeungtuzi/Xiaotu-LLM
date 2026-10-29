#!/usr/bin/env bash
# ============================================================================
# metricsmon.sh —— vLLM 指标采样（**只读**：GET /metrics，不干预服务）
#
# 目的：验证"累积量"假设 —— 单次超长上下文杀不死，跑几小时必死。
#       累计"做了多少事"要比"活了多久"更有判别力：
#         ① 队列积压：num_requests_waiting（及 by_reason capacity/deferred）
#         ② KV 占用：kv_cache_usage_perc（单调抬升 ⇒ 块不回收）
#         ③ 抢占：num_preemptions_total（积压时调度器开始抢占）
#         ④ 累计工作量：prompt_tokens_total / generation_tokens_total（用于归一化"活多久"）
#
# 用法： setsid nohup bash scripts/metricsmon.sh [out.csv] >/dev/null 2>&1 </dev/null &
# 口径：每分钟一行 CSV，只追加。
# ============================================================================
set -uo pipefail
OUT=${1:-$HOME/lvllm/xtu-oct28-tools/logs/resmon/metrics.csv}
URL=${METRICS_URL:-http://127.0.0.1:8700/metrics}
mkdir -p "$(dirname "$OUT")"

if [ ! -f "$OUT" ]; then
  echo "ts,running,waiting,wait_cap,wait_defer,kv_perc,preempt_total,prompt_tok_total,gen_tok_total,ttft_p50_s,e2e_p50_s,up" > "$OUT"
fi

val() {  # $1=metric regex (exact prefix), take first sample value
  printf '%s\n' "$BODY" | awk -v k="$1" 'index($0, k) == 1 {print $2; exit}'
}

while :; do
  BODY=$(curl -s -m 8 --noproxy 127.0.0.1 "$URL" 2>/dev/null || true)
  if [ -z "$BODY" ]; then
    echo "$(date -u +%H:%M:%S),,,,,,,,,,,0" >> "$OUT"
  else
    r=$(val 'vllm:num_requests_running')
    w=$(val 'vllm:num_requests_waiting{')
    wc=$(val 'vllm:num_requests_waiting_by_reason{engine="0",model_name="DeepSeek-V4.1-Flash",reason="capacity"}')
    wd=$(val 'vllm:num_requests_waiting_by_reason{engine="0",model_name="DeepSeek-V4.1-Flash",reason="deferred"}')
    kv=$(val 'vllm:kv_cache_usage_perc')
    pr=$(val 'vllm:num_preemptions_total')
    pt=$(val 'vllm:prompt_tokens_total')
    gt=$(val 'vllm:generation_tokens_total')
    echo "$(date -u +%H:%M:%S),${r:-0},${w:-0},${wc:-0},${wd:-0},${kv:-0},${pr:-0},${pt:-0},${gt:-0},,,1" >> "$OUT"
  fi
  sleep "${METRICS_INTERVAL:-60}"
done
