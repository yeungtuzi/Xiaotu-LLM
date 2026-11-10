#!/usr/bin/env bash
# =============================================================================
# degen_repro.sh —— 「长会话退化」8 条连发复现器（唯一入口，可复用）
#
# 背景（详见 dev-docs/REPORT_DEGEN_PREFIX_CACHE_2026-11-09.md）：
#   生产口径下，带前缀缓存时**重启后前 7 条 131K 请求正确、第 8 条起错误**；
#   本脚本就是那个 20 分钟的复现器，用来判定某个修复候选是否有效。
#
# 用法（在**远端**执行，工作目录 = 本仓库）：
#   bash scripts/degen_repro.sh [N=7430] [CNT=8] [DEPTH=0.5] [TAG=repro] [PORT=8700]
#
# 判据：
#   * 前 7 条必须全 ✓（否则环境/口径不对，本次作废）；
#   * 第 8 条起出现 ✗ ⇒ **退化复现**；
#   * CNT 条全 ✓ ⇒ 该口径**未复现**（若在测修复候选，这才算通过）。
#
# ⛔ 本脚本**不启动/不重启服务**：口径由 scripts/serve_ds41.sh 决定，重启是批处理资源。
#    先自行 `bash scripts/serve_ds41.sh restart MODEL_DIR=… PIN_CUDA_MOE=1 MARLIN=1`（需要时加 NOPC=1）。
# =============================================================================
set -uo pipefail

N="${N:-7430}"
CNT="${CNT:-8}"
DEPTH="${DEPTH:-0.5}"
TAG="${TAG:-repro}"
PORT="${PORT:-8700}"
PY="${PY:-python3}"
OUT_DIR="${OUT_DIR:-dev-docs/report/tuning/logs/degen-repro}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/${TAG}-$(date +%Y%m%d-%H%M%S).log"
cd "$(dirname "$0")/.." || exit 1

# 就绪自检：只读端口，不重启
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${PORT}/health" || true)
if [ "$code" != "200" ]; then
    echo "⛔ 服务未就绪（health=$code, port=$PORT）；先起服务，本脚本不会替你重启" | tee -a "$LOG"
    exit 2
fi

echo "=== degen_repro tag=$TAG N=$N cnt=$CNT depth=$DEPTH port=$PORT $(date -Iseconds)" | tee -a "$LOG"

ok=0; bad=0; first_bad=""
for i in $(seq 1 "$CNT"); do
    uuid=$(printf "%s%02d%010x" "${TAG:0:2}" "$i" $((RANDOM * RANDOM)))
    line=$("$PY" scripts/probe_dump_long.py --n "$N" --uuid "$uuid" --depth "$DEPTH" --maxtok 64 2>&1 | tail -1)
    echo "[$i] $(date -Iseconds) $line" | tee -a "$LOG"
    case "$line" in
        *"命中=✓"*) ok=$((ok + 1)) ;;
        *) bad=$((bad + 1)); [ -z "$first_bad" ] && first_bad="$i" ;;
    esac
done

echo "=== 汇总 ok=$ok bad=$bad 首个失败=$([ -n "$first_bad" ] && echo "#$first_bad" || echo 无)" | tee -a "$LOG"

if [ "$ok" = "$CNT" ]; then
    echo "✅ 未复现退化（$CNT/$CNT）：该口径可作为修复候选，需再跑 (A)(B)(C) 正式验收" | tee -a "$LOG"
    exit 0
fi
if [ -n "$first_bad" ] && [ "$first_bad" -ge 8 ]; then
    echo "🔴 退化已复现：第 $first_bad 条起失败（前 $((first_bad - 1)) 条正常）" | tee -a "$LOG"
    exit 1
fi
echo "⚠️ 早期失败（#$first_bad）：环境或口径异常，本次不算复现，先查 health/日志" | tee -a "$LOG"
exit 3
