#!/usr/bin/env bash
# dl_status.sh —— 权重下载进度一行报（含速率与 ETA），**只读**，不碰下载器与服务。
#
# 用法：bash scripts/dl_status.sh [模型目录]
# 速率用上次采样点算（采样点存在 $RUN_DIR/dl_status.state），不依赖日志解析。
set -uo pipefail
DIR="${1:-/home/user/lvllm/models/dsv41f-1017-weights}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
STATE="$RUN_DIR/dl_status.state"
TOTAL_GB="${TOTAL_GB:-352}"       # 48 片 × ≈7.39 GB（前两片更小）

now_bytes=$(du -sb "$DIR" 2>/dev/null | cut -f1)
[ -n "${now_bytes:-}" ] || { echo "⛔ 取不到 $DIR 的大小"; exit 2; }
now_s=$(date +%s)

done_shards=$(ls "$DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l)
part=$(ls "$DIR"/*.incomplete 2>/dev/null | wc -l)

rate="—"; eta="—"
if [ -f "$STATE" ]; then
    read -r prev_bytes prev_s < "$STATE" || true
    if [ -n "${prev_bytes:-}" ] && [ -n "${prev_s:-}" ] && [ "$now_s" -gt "$prev_s" ]; then
        dt=$(( now_s - prev_s ))
        db=$(( now_bytes - prev_bytes ))
        if [ "$dt" -ge 5 ] && [ "$db" -ge 0 ]; then
            rate_bps=$(( db / dt ))
            rate="$(awk -v r="$rate_bps" 'BEGIN{printf "%.1f MB/s", r/1e6}')"
            remain_gb=$(( TOTAL_GB - now_bytes / 1000000000 ))
            if [ "$rate_bps" -gt 0 ] && [ "$remain_gb" -gt 0 ]; then
                eta="$(awk -v g="$remain_gb" -v r="$rate_bps" 'BEGIN{printf "%.1f 小时", (g*1e9)/r/3600}')"
            fi
        fi
    fi
fi
printf '%s %s\n' "$now_bytes" "$now_s" > "$STATE"

pct=$(awk -v b="$now_bytes" -v t="$TOTAL_GB" 'BEGIN{printf "%.0f", (b/1e9)/t*100}')
printf '已下 %s / ~%sG (%s%%)  分片 %s 完成 + %s 在传  速率 %s  ETA %s\n' \
    "$(du -sh "$DIR" 2>/dev/null | cut -f1)" "$TOTAL_GB" "$pct" \
    "$done_shards" "$part" "$rate" "$eta"
if [ "$done_shards" -ge 48 ] && [ "$part" -eq 0 ]; then
    echo "✅ 48 个分片齐全，可以开始预检：CHECK_SHARDS=1 bash scripts/prepare_model_dir.sh $DIR"
fi
