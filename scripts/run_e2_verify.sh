#!/usr/bin/env bash
# =============================================================================
# run_e2_verify.sh —— E2 验证 + **修前/修后对照**（不启停服务）
#
# 前置：服务已用**带 D1 补丁**的树重启（`bash scripts/e2_apply.sh apply` 后再 start）
# 用法：
#   bash scripts/run_e2_verify.sh [PORT=8700] [CNT=8] [N=7430] [INSTANCE_TAG=prod]
# 产物：
#   $RUN_DIR/e2_e2.txt / e2_verdict.txt      本轮原始输出
#   stdout                                   修前(E1)/修后(E2) 对照表
# 退出码：0 = E2 全过（8/8 且判据 rc=0）；1 = 未过；2 = 环境不满足
# =============================================================================
set -uo pipefail
PORT="${PORT:-8700}"
CNT="${CNT:-8}"
N="${N:-7430}"
INSTANCE_TAG="${INSTANCE_TAG:-prod}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${PY_BIN:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3

say() { printf '[e2v] %s\n' "$*"; }

# ① 补丁必须在树里（否则这轮不是 E2）
TREE="${TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}"
d1=$(grep -c "^def compute_token_to_req_indices" "$TREE/vllm/v1/attention/ops/metadata.py" 2>/dev/null || true)
[ "${d1:-0}" = "1" ] || { say "⛔ 树里没有 D1（d1=${d1:-0}）⇒ 先 e2_apply.sh apply 并**重启**服务"; exit 2; }
say "✅ 树里已有 D1"

# ② 跑本轮 E1（此时是"修后"）
PORT="$PORT" CNT="$CNT" TAG=e2 INSTANCE_TAG="$INSTANCE_TAG" N="$N" bash "$HERE/run_e1.sh"
rc_e1=$?

# ③ 取修前（E1）与修后（E2）的 8 条结果
pick() {  # $1=文件 → 打印 "ok=.. bad=.."
    grep -aoE "汇总 ok=[0-9]+ bad=[0-9]+" "$1" 2>/dev/null | tail -1
}
before=$(pick "$RUN_DIR/e1_e1.txt")
after=$(pick "$RUN_DIR/e2_e2.txt")

va=0
if [ -f "$RUN_DIR/e1_verdict.txt" ]; then
    grep -aq "三条判据全满足" "$RUN_DIR/e1_verdict.txt" && va=0 || va=1
fi
if [ -f "$RUN_DIR/e2_verdict.txt" ]; then
    grep -aq "三条判据全满足" "$RUN_DIR/e2_verdict.txt" && vb=0 || vb=1
else
    vb=1
fi

printf '\n============ E2 修前/修后对照 ============\n'
printf '%-22s %-26s %-26s\n' "项" "修前（E1）" "修后（E2，D1）"
printf '%-22s %-26s %-26s\n' "8×长请求汇总" "${before:-未取得}" "${after:-未取得}"
printf '%-22s %-26s %-26s\n' "仪器三条判据" "$([ "$va" = 0 ] && echo '✅ 全满足' || echo '❌ 有异常')" \
    "$([ "$vb" = 0 ] && echo '✅ 全满足' || echo '❌ 有异常')"
printf '原始输出: %s/e1_*.txt  %s/e2_*.txt\n' "$RUN_DIR" "$RUN_DIR"

if [ "$rc_e1" = "0" ] && [ -n "$after" ] && [ "$vb" = "0" ]; then
    case "$after" in
        *"bad=0"*) say "✅ E2 通过（8/8 且判据全满足）"; exit 0 ;;
    esac
fi
say "❌ E2 未通过 ⇒ 按 RUNBOOK §4 的下一步走（叠加 F1 或转调度侧重绕）"
exit 1
