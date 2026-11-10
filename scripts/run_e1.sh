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
PY_BIN="${PY_BIN:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
# 环境路径可能随部署变化；解释器不存在就回落到 python3（否则 ③b 会以"没有命中"的形式误判缓存失效）
[ -x "$PY_BIN" ] || PY_BIN=python3
N="${N:-7430}"
DEPTH="${DEPTH:-0.5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DBG="$TREE/../run/dbg.patch"
TRIG="$RUN_DIR/triggers/swa_bounded_replay_dbg"
LOG="$RUN_DIR/ds41_${INSTANCE_TAG}.log"

say() { printf '[e1] %s\n' "$*"; }

# ⚠️ 仪器补丁里的触发文件路径是**硬编码**的；RUN_DIR 一旦被改，补丁就看不到触发文件，
#    表现为"日志里没有任何 [xtu-swa-bounded-replay] 行"，极易被误判成"重放没生效"（台账 #267）。
#    生产必须用默认 RUN_DIR；测试（SKIP_HEALTH=1）允许自定义。
TRIG_EXPECT="/home/user/lvllm/run/triggers/swa_bounded_replay_dbg"
if [ "${SKIP_HEALTH:-0}" != "1" ] && [ "$TRIG" != "$TRIG_EXPECT" ]; then
    say "⛔ RUN_DIR=$RUN_DIR ⇒ 触发文件路径 $TRIG 与仪器硬编码的 $TRIG_EXPECT 不一致"
    say "   仪器只会读后者 ⇒ 本轮拿不到任何重放证据。改用默认 RUN_DIR（不要自定义）"
    exit 2
fi

# ① 服务在跑？
# SKIP_HEALTH=1 仅用于离线回归（见 test_e1_rc_gate.sh）；生产绝不要设
if [ "${SKIP_HEALTH:-0}" != "1" ] && ! curl -s -o /dev/null --max-time 4 "http://127.0.0.1:${PORT}/health"; then
    say "⛔ 端口 ${PORT} 没有服务（/health 不通）⇒ 先按 RUNBOOK §1 起服务"
    exit 2
fi
say "✅ 服务在 ${PORT}"

# ①a 口径自证：读**进程环境**（/proc/<pid>/environ），确认 MARLIN/PIN/SWA/NVCC 真的在这个进程里。
#    不通过 ⇒ 本轮**作废**（否则 prefill 对照与"生产口径"都是假的；见台账 #251）。
if [ "${SKIP_HEALTH:-0}" != "1" ]; then
    if ! _prof=$(RUN_DIR="$RUN_DIR" bash "$HERE/assert_profile_local.sh" "$INSTANCE_TAG" 2>&1); then
        printf '%s\n' "$_prof" | sed 's/^/[e1]   /'
        say "⛔ 运行口径不符 ⇒ 本轮作废（不要用这一轮的失败/通过下任何结论）"
        exit 2
    fi
    say "✅ 运行口径符合生产期望（MARLIN/PIN/SWA/NVCC 均在进程环境里）"
fi

# ①b 口径自检：只认**引擎日志里的自报**（不是"旗标传了"）
if [ -f "$LOG" ]; then
    if grep -aq "开关开但表缺失\|建池失败" "$LOG"; then
        say "⛔ 日志里出现 'PIN 表缺失 / 建池失败' ⇒ 这不是生产口径（GPU 专家驻留没生效）"
        say "   先把 XIAOTU_PIN_TABLE 指到存在的表并重启，再跑 E1"
        exit 2
    fi
    grep -aq "Decoder SWA bounded replay: in eager prefill steps" "$LOG" \
        && say "✅ 日志确认 decoder-side SWA bounded replay 生效" \
        || say "⚠️ 日志里没看到重放生效的自报行（对照 RUNBOOK 五查第 ⑤ 条）"
    # 内存前缀缓存：该串**确实会出现**，但它来自 core.py:133 的整行 "with config: …" 巨型 dump
    # （只证明"参数传了 enable_prefix_caching=True"，不证明"真的命中过"）
    # ⇒ 仍然只用行为判据（下面 ③b 的 cached>0）；这里留一句提示，避免有人拿 dump 当证据
    grep -aq "enable_prefix_caching=True" "$LOG" \
        && say "ℹ️ 配置 dump 里有 enable_prefix_caching=True（仅证明参数，不证明命中；行为判据见 ③b）"
else
    say "⚠️ 找不到日志 $LOG ⇒ 跳过口径自检（INSTANCE_TAG=${INSTANCE_TAG}）"
fi

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
PORT="$PORT" CNT="$CNT" TAG="$TAG" bash "$HERE/degen_repro.sh" > "$OUT" 2>&1
rc_repro=$?
# rc 语义必须翻译：rc=2 是"口径不符"（例如 ptok 不足 131K）⇒ 本轮作废，不能继续拿结论（台账 #230）
# TAG=e2 视为"修后"臂，其余视为基线臂。
case "$TAG" in e2*) _mode=fix ;; *) _mode=base ;; esac
_rv_out=$(bash "$HERE/repro_rc_verdict.sh" "$rc_repro" "$_mode"); _rv=$?
printf '%s\n' "$_rv_out" | sed 's/^/[e1] /'
if [ "$_rv" = "2" ]; then
    say "⛔ 本轮口径不成立 ⇒ 中止（不产生任何「通过 / 未复现」结论）"
    exit 2
fi
say "原始输出：$OUT"
tail -n "$CNT" "$OUT" | sed 's/^/[e1]   /'

# 基线若 8/8 全过 ⇒ "没复现"，必须先对比口径，不能当成"已修好"
_sum=$(grep -aoE "汇总 ok=[0-9]+ bad=[0-9]+" "$OUT" 2>/dev/null | tail -1)
say "复现汇总：${_sum:-（未取得）}"
case "$_sum" in
    *"bad=0"*) say "⚠️ 本次 **未复现**（8 条全过）⇒ 不要当成'已修好'：先对比口径（权重/SPEC/PC/PIN/MAXLEN/MBT）与历史复现那次是否一致" ;;
    "")        say "⚠️ 没读到汇总行 ⇒ 先看上面的原始输出，可能是请求失败而非'复现'" ;;
    *)         say "✅ 出现失败（与历史'第 8 条起错'同向）⇒ 基线成立" ;;
esac

# ③b 保证"一定有缓存命中"的两次同 prompt（degen_repro 每条都用不同 uuid ⇒ cached=0，
#     只有这一步能保证仪器里出现 hit=1 的重放步；否则 --verdict 会报"没有 hit=1 的 prefill 步"）
HITOUT="$RUN_DIR/${TAG}_hits.txt"
say "… 追加 2 条**完全相同**的长 prompt（固定 uuid）以确保出现命中重放步"
for k in 1 2; do
    PORT="$PORT" "$PY_BIN" "$HERE/probe_dump_long.py" --n "${N:-7430}" --uuid "e1fixed00" \
        --depth "${DEPTH:-0.5}" --maxtok 8 2>&1 | tail -1 | sed "s/^/[e1]  hit#$k /" | tee -a "$HITOUT"
done

# ③b 的行为判据：第 2 条必须命中（cached>0），否则说明内存前缀缓存没生效
_h2=$(grep -a "hit#2" "$HITOUT" 2>/dev/null | tail -1)
_c2=$(printf '%s' "$_h2" | grep -oE "cached=[0-9]+" | head -1 | cut -d= -f2)
if [ -n "${_c2:-}" ] && [ "${_c2:-0}" -gt 0 ]; then
    say "✅ 行为判据：第 2 条同 prompt 命中 ${_c2} tokens ⇒ 内存前缀缓存确实开启"
else
    say "⛔ 第 2 条同 prompt 没有命中（cached=${_c2:-?}）⇒ 内存前缀缓存未生效，本轮结论无效"
    exit 2
fi

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
