#!/usr/bin/env bash
# weights_watch_remote.sh —— 【在远端跑】的权重就绪守望进程（会话结束后仍存活）
#
# 为什么要在远端跑：本地 harness 的后台 job 会随会话结束一起消失；
# 权重还要几小时，需要一个**不依赖会话**的守望者留下"就绪/损坏"的落盘证据（台账 #241）。
#
# 行为：每 INTERVAL 秒检查一次；48 片齐备且 0 个 .incomplete 时，
#       校验两个 101.5 GB 大分片的 sha256 + 字节数，然后写标志文件并退出。
#
# 标志文件（都在 /home/user/lvllm/logs/）：
#   WEIGHTS_READY      校验通过（内容含时间戳与总量）
#   WEIGHTS_BAD        校验不符（内容含不符项）
#   weights_watch.log  逐次进度
#
# 启动（远端，脱离会话）：
#   setsid nohup bash scripts/weights_watch_remote.sh > /home/user/lvllm/logs/weights_watch.out 2>&1 < /dev/null &
set -uo pipefail
DIR=${DIR:-/home/user/lvllm/models/dsv41f-1017-weights}
LOGDIR=${LOGDIR:-/home/user/lvllm/logs}
INTERVAL=${INTERVAL:-900}
MAX_HOURS=${MAX_HOURS:-12}
TOTAL=${TOTAL:-510300000000}
LOG="$LOGDIR/weights_watch.log"
mkdir -p "$LOGDIR"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

SHA47=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
SHA48=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd
SZ47=101535150936
SZ48=101537926640

rm -f "$LOGDIR/WEIGHTS_READY" "$LOGDIR/WEIGHTS_BAD"
rounds=$(( MAX_HOURS * 3600 / INTERVAL )); [ "$rounds" -lt 1 ] && rounds=1
log "守望启动 pid=$$ 每 ${INTERVAL}s 一次，最多 ${MAX_HOURS}h（${rounds} 轮）dir=$DIR"

for i in $(seq 1 "$rounds"); do
    n=$(ls "$DIR"/*.safetensors 2>/dev/null | wc -l)
    inc=$(ls "$DIR"/*.incomplete 2>/dev/null | wc -l)
    b=$(du -sb "$DIR" 2>/dev/null | cut -f1); b=${b:-0}
    if [ "$n" = "48" ] && [ "$inc" = "0" ]; then
        log "[$i/$rounds] 分片齐备（48/0）⇒ 开始校验 sha256"
        s47=$(sha256sum "$DIR/model-00047-of-00048.safetensors" | awk '{print $1}')
        s48=$(sha256sum "$DIR/model-00048-of-00048.safetensors" | awk '{print $1}')
        z47=$(stat -c %s "$DIR/model-00047-of-00048.safetensors")
        z48=$(stat -c %s "$DIR/model-00048-of-00048.safetensors")
        bad=""
        [ "$s47" = "$SHA47" ] || bad="$bad 47.sha256"
        [ "$s48" = "$SHA48" ] || bad="$bad 48.sha256"
        [ "$z47" = "$SZ47" ] || bad="$bad 47.size=$z47"
        [ "$z48" = "$SZ48" ] || bad="$bad 48.size=$z48"
        if [ -z "$bad" ]; then
            printf '%s\n' "sha256+size OK at $(date -u +%FT%TZ) bytes=$b" > "$LOGDIR/WEIGHTS_READY"
            log "✅ 校验通过 ⇒ WEIGHTS_READY"
            exit 0
        fi
        printf '%s\n' "校验不符：$bad" > "$LOGDIR/WEIGHTS_BAD"
        log "❌ 校验不符：$bad ⇒ WEIGHTS_BAD"
        exit 2
    fi
    log "[$i/$rounds] ${n}/48 片，${inc} 个未完成，已下 $((b/1000000000)) GB（$((b/5103000000))%），剩余 $(( (TOTAL-b)/1000000000 )) GB"
    sleep "$INTERVAL"
done
log "超时 ${MAX_HOURS}h 仍未齐备"; exit 3
