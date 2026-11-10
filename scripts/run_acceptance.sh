#!/usr/bin/env bash
# =============================================================================
# run_acceptance.sh —— 跑验收 A / B（**不启停服务**），并给出一张汇总表
#
# 用法：
#   bash scripts/run_acceptance.sh [PORT=8700] [A_ROUNDS=10] [B_ROUNDS=3] [N=1200]
# 产物：
#   $RUN_DIR/acc_A.txt / acc_B.txt   原始输出
#   stdout                           汇总表（判据 + 结果）
# 退出码：0 = A、B 都过；1 = 有失败；2 = 环境不满足
#
# 口径（与台账一致）：
#   A = 同一 prompt 连发 N 次，输出签名必须唯一           → scripts/accept_abc.py --a-only
#   B = 四臂交错（shared/unique/tail/suffix），unique 必须 100% → scripts/probe_degen_arms.py
#       ⚠️ B 不用 accept_abc 的"前缀深度"变体（那是辅助，口径不同，见台账 #155）
# =============================================================================
set -uo pipefail
PORT="${PORT:-8700}"
A_ROUNDS="${A_ROUNDS:-10}"
B_ROUNDS="${B_ROUNDS:-3}"
N="${N:-1200}"
MAXTOK="${MAXTOK:-64}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${PY_BIN:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3

if ! curl -s -o /dev/null --max-time 4 "http://127.0.0.1:${PORT}/health"; then
    echo "[acc] ⛔ 端口 ${PORT} 没有服务 ⇒ 先起服务（RUNBOOK §1）"
    exit 2
fi

rc_a=0; rc_b=0
"$PY_BIN" "$HERE/accept_abc.py" --port "$PORT" --a-rounds "$A_ROUNDS" --max-tokens "$MAXTOK" --a-only \
    > "$RUN_DIR/acc_A.txt" 2>&1 || rc_a=$?
PORT="$PORT" ROUNDS="$B_ROUNDS" N="$N" MAXTOK="$MAXTOK" \
    "$PY_BIN" "$HERE/probe_degen_arms.py" > "$RUN_DIR/acc_B.txt" 2>&1 || rc_b=$?

a_line=$(grep -aE "^\[A\]" "$RUN_DIR/acc_A.txt" | tail -1)
uniq=$(grep -aoE "不同输出 [0-9]+ 种" "$RUN_DIR/acc_A.txt" | tail -1)
b_uniq=$(grep -aE "^unique " "$RUN_DIR/acc_B.txt" | tail -1)
armbroken=$(grep -ac "ARM-BROKEN" "$RUN_DIR/acc_B.txt" || true)

printf '\n================ 验收汇总 ================\n'
printf 'A（同一 prompt ×%s，要求签名唯一）: %s\n' "$A_ROUNDS" "${uniq:-未取得}"
printf 'B（四臂交错，要求 unique 100%%）    : %s\n' "${b_uniq:-未取得}"
printf 'ARM-BROKEN 次数（应为 0）          : %s\n' "$armbroken"
printf '原始输出: %s/acc_A.txt  %s/acc_B.txt\n' "$RUN_DIR" "$RUN_DIR"
if [ "$rc_a" = "0" ] && [ "$rc_b" = "0" ] && [ "$armbroken" = "0" ]; then
    printf '判定: ✅ A、B 全过\n'
    exit 0
fi
printf '判定: ❌ 有失败（A rc=%s, B rc=%s）\n' "$rc_a" "$rc_b"
exit 1
