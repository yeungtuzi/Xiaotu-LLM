#!/usr/bin/env bash
# kv_gate.sh —— 从引擎日志里读出 KV 容量并判门槛（只读日志，不碰服务）
#
# 为什么需要专门脚本：vLLM 打印的是**带千分位逗号**的
#   "GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 524,288 tokens per request: 2.35x"
# ⇒ 任何 `grep -o 'size: [0-9]*'` 都会在第一个逗号处截断，读出 1（假失败）。
#
# 用法： bash scripts/kv_gate.sh <logfile> <need_tokens>
#   退出码 0 = 达标；1 = 不达标；2 = 日志里找不到该行（服务没到那一步）
set -uo pipefail

LOG=${1:?用法: kv_gate.sh <logfile> <need_tokens>}
NEED=${2:?用法: kv_gate.sh <logfile> <need_tokens>}

raw=$(grep -a "GPU KV cache size:" "$LOG" 2>/dev/null | tail -1)
if [ -z "$raw" ]; then
    echo "⛔ 日志里没有 'GPU KV cache size:' 行（服务可能还没建池完成）"
    exit 2
fi

have=$(printf '%s\n' "$raw" | sed -n 's/.*GPU KV cache size: *\([0-9,]*\) *tokens.*/\1/p' | tr -d ',')
if [ -z "${have:-}" ]; then
    echo "⛔ 解析失败，原始行："
    printf '   %s\n' "$raw"
    exit 2
fi

# 顺带报出 max concurrency（同一行）
mc=$(printf '%s\n' "$raw" | sed -n 's/.*Maximum concurrency for [0-9,]* tokens per request: *\([0-9.]*\)x.*/\1/p')

printf '   KV 容量 = %s tokens' "$(printf '%s' "$have" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
[ -n "${mc:-}" ] && printf '   （max concurrency %.2fx）' "$mc"
printf '\n   门槛   = %s tokens\n' "$NEED"

if [ "$have" -ge "$NEED" ]; then
    echo "✅ KV 门槛通过"
    exit 0
fi
echo "❌ KV 不足：$have < $NEED ⇒ 缩小 PIN 表（pin700 → pin400 → pin320）后重启"
exit 1
