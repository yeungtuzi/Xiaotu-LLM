#!/usr/bin/env bash
# watch_weights_ready.sh —— 后台监视 DeepSeek-V4.1-Flash-1017 权重下载
#
# 用途：每 INTERVAL 秒（默认 900 = 15 分钟）查一次远端权重目录；
#       48 片齐备且 0 个 .incomplete 时，立刻做机械校验（两个 101.5 GB 大分片的
#       sha256 + 精确字节数），通过则打印 WEIGHTS_READY 并退出 0。
#
# ⚠️ 本脚本**只读**远端：不启动、不停止、不杀任何进程 ⇒ 不涉及 R38 起停审计。
# ⚠️ 日志写 /root/lvllm/logs/（持久），不用 /tmp。
#
# 退出码：0 = WEIGHTS_READY（校验通过）；2 = 分片齐了但校验不符；3 = 超时未见齐备
set -uo pipefail

SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
DIR=${DIR:-/home/user/lvllm/models/dsv41f-1017-weights}
TOTAL=${TOTAL:-510300000000}          # 510.3 GB（实测总量，见台账 #205/#206）
SHARDS=${SHARDS:-48}
INTERVAL=${INTERVAL:-900}
MAX_HOURS=${MAX_HOURS:-10}
LOG=${LOG:-/root/lvllm/logs/watch_weights_ready.log}

mkdir -p "$(dirname "$LOG")" || exit 1

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
remote() { timeout 45 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST" "$1" 2>&1; }

# 权威值（台账 #207 实测记录）

rounds=$(( MAX_HOURS * 3600 / INTERVAL ))
[ "$rounds" -lt 1 ] && rounds=1
log "启动监视：每 ${INTERVAL}s 一次，最多 ${MAX_HOURS}h（${rounds} 轮）dir=$DIR"

for i in $(seq 1 "$rounds"); do
    out=$(remote 'd='"$DIR"'; n=$(ls $d/*.safetensors 2>/dev/null | wc -l); \
                 inc=$(ls $d/*.incomplete 2>/dev/null | wc -l); \
                 b=$(du -sb $d 2>/dev/null | cut -f1); \
                 echo "COUNT $n ${inc:-0} ${b:-0}"' | awk '/^COUNT /{n=$2;i=$3;b=$4} END{if(n!="") print n, i, b}')
    read -r n inc b <<<"${out:-}"

    if [ -z "${n:-}" ]; then
        log "[$i/$rounds] ssh 查询失败，稍后重试"
    elif [ "$n" = "$SHARDS" ] && [ "${inc:-1}" = "0" ]; then
        log "[$i/$rounds] 分片齐备：${n}/${SHARDS}，无 .incomplete，总 $((b/1000000000)) GB ⇒ 开始机械校验"
        # 统一走 check_weights_ready.sh（单一真源：口径 + sha256 都校验）
        if bash "$(dirname "$(readlink -f "$0")")/check_weights_ready.sh" --sha >>"$LOG" 2>&1; then
            log "WEIGHTS_READY shards=${n} bytes=${b}"
            exit 0
        fi
        log "❌ 校验未通过 ⇒ 退出 2（需人工判断是否重下）"
        exit 2
    else
        pct=$(( b * 100 / TOTAL ))
        log "[$i/$rounds] 进行中：${n}/${SHARDS} 片，${inc} 个未完成，已下 $((b/1000000000)) GB（${pct}%），剩余 $(( (TOTAL-b)/1000000000 )) GB"
    fi

    [ "$i" -lt "$rounds" ] && sleep "$INTERVAL"
done

log "超时 ${MAX_HOURS}h 仍未齐备 ⇒ 退出 3"
exit 3
