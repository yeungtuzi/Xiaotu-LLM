#!/usr/bin/env bash
# dl_rate_sampler.sh —— 每分钟记一次权重目录字节数（用于算真实下载速率/ETA）。
# 用法（远端）：setsid nohup bash scripts/dl_rate_sampler.sh >/dev/null 2>&1 < /dev/null &
# 停止：kill "$(cat /home/user/lvllm/run/dl_rate.pid)"
# 输出：/home/user/lvllm/logs/dl_rate.log，每行 "<epoch> <bytes>"
set -uo pipefail
DIR="${DIR:-/home/user/lvllm/models/dsv41f-1017-weights}"
LOG="${LOG:-/home/user/lvllm/logs/dl_rate.log}"
PIDF="${PIDF:-/home/user/lvllm/run/dl_rate.pid}"
INTERVAL="${INTERVAL:-60}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDF")"
echo $$ > "$PIDF"
while :; do
    printf '%s %s\n' "$(date +%s)" "$(du -sb "$DIR" 2>/dev/null | cut -f1)" >> "$LOG"
    sleep "$INTERVAL"
done
