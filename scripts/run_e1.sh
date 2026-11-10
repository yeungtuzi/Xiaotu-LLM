#!/usr/bin/env bash
# =============================================================================
# run_e1.sh —— E1 取证一键跑（**不启停服务、不杀进程**，只打补丁 + 发请求 + 出结论）
#
# 前置：服务已按生产口径起好（PIN_CUDA_MOE=1 MARLIN=1，PC 开，decoder-side SWA bounded replay 开）
# 用法：
#   bash scripts/run_e1.sh [PORT=8700] [CNT=8] [TAG=e1]
# 产物：
#   $RUN_DIR/<TAG>_e1.txt        8 条 131K 的原始输出
#   $RUN_DIR/<TAG>_verdict.txt   --verdict 的结论（rc=0 才算重放侧健康）
# 退出码：0=判据全过；1=有异常；2=环境不满足（服务没起/补丁失败）
# =============================================================================
set -uo pipefail
PORT="${PORT:-8700}"
CNT="${CNT:-8}"
TAG="${TAG:-e1}"                 # 复现产物的标签
INSTANCE_TAG="${INSTANCE_TAG:-prod}"  # 服务的 TAG（决定读哪份实例日志）
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
TREE="${TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DBG="$TREE/../run/dbg.patch"
TRIG="$RUN_DIR/triggers/swa_bounded_replay_dbg"
LOG="$RUN_DIR/ds41_${INSTANCE_TAG}.log"

say() { printf '[e1] %s\n' "$*"; }

# ① 服务在跑？
if ! curl -s -o /dev/null --max-time 4 "http://127.0.0.1:${PORT}/health"; then
    say "⛔ 端口 ${PORT} 没有服务（/health 不通）⇒ 先按 RUNBOOK §1 起服务"
    exit 2
fi
say "✅ 服务在 ${PORT}"

# ② 仪器补丁（幂等：已打过就跳过）
if grep -qs "xtu-swa-bounded-replay" "$TREE/vllm/models/deepseek_v41/nvidia/model_state.py"; then
    say "✅ 仪器补丁已在树里"
else
    if [ -f "$DBG" ]; then
        if (cd "$TREE" && patch -p1 < "$DBG" >/dev/null 2>&1); then
            say "✅ 仪器补丁已应用（⚠️ 需要重启服务才生效 ⇒ 本脚本不会重启，请重跑 RUNBOOK §1）"
            exit 2
        fi
        say "⛔ 仪器补丁应用失败（可能上下文不符）⇒ 手工核对"
        exit 2
    fi
    say "⛔ 找不到 $DBG（应由部署方放好）"
    exit 2
fi
mkdir -p "$(dirname "$TRIG")" && touch "$TRIG" && say "✅ 触发文件已就位：$TRIG"

# ③ 8 条 131K 连发（复用既有复现器，避免两套逻辑）
OUT="$RUN_DIR/${TAG}_e1.txt"
say "… 开始 ${CNT} 条 131K（每条约 2.3 min prefill，总计约 $((CNT * 3)) 分钟）"
# ⚠️ PORT 必须一起传：degen_repro.sh 自己也读 PORT=${PORT:-8700}，不传就会打到 8700
if ! PORT="$PORT" CNT="$CNT" TAG="$TAG" bash "$HERE/degen_repro.sh" > "$OUT" 2>&1; then
    say "⚠️ degen_repro 返回非 0（仍继续分析，看是哪几条失败）"
fi
say "原始输出：$OUT"
tail -n "$CNT" "$OUT" | sed 's/^/[e1]   /'

# ④ 判据
VD="$RUN_DIR/${TAG}_verdict.txt"
if [ ! -f "$LOG" ]; then
    say "⛔ 找不到实例日志 $LOG ⇒ 用 INSTANCE_TAG=<服务TAG> 指定（当前 $INSTANCE_TAG）"
    exit 2
fi
python3 "$HERE/analyze_replay_instrument.py" "$LOG" --verdict | tee "$VD"
rc=${PIPESTATUS[0]}
say "结论文件：$VD"
if [ "$rc" = "0" ]; then
    say "✅ 三条判据全过（hit=1 / computed==replay_start / pad_added=128）"
else
    say "❌ 判据有异常 ⇒ 按上面的指向查（见 RUNBOOK §2）"
fi
exit "$rc"
