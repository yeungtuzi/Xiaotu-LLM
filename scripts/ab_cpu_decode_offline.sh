#!/bin/bash
# ---------------------------------------------------------------------------
# 离线 A/B 驱动：CPU decode 固定开销改动（M1 / R1 / R2 / M4 / R5）
#
# 用法（在远端执行，不需要起服务、不碰任何 vllm 进程）：
#     bash scripts/ab_cpu_decode_offline.sh                # 默认矩阵
#     VARIANTS="a b" bash scripts/ab_cpu_decode_offline.sh # 指定开关组合
#
# 判据：
#   * 数值：`[hash]` 必须**逐字相同**（同一个 .so 内不同开关之间）；
#   * 性能：`ms/layer` 取 min-of-REP，且**必须记同期 load**（本机常有其它负载）。
#
# ⚠️ 本脚本**不**起停任何服务、不 kill 任何进程、不按名字匹配进程。
# ---------------------------------------------------------------------------
set -u -o pipefail

REPO="${REPO:-/home/user/lvllm/vllm-xiaotu-moe}"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master}"
CONDA="${CONDA:-/home/user/anaconda3}"
ENVNAME="${ENVNAME:-vllm-xiaotu-moe}"

THREADS="${THREADS:-60}"     # TP=2 ⇒ 每 rank 60（全局 120 = 24 CCD × 5）
DEDUP="${DEDUP:-12}"         # 真实 decode 的活跃专家数
BS="${BS:-6}"
REP="${REP:-7}"
LOG="${LOG:-/tmp/ab_cpu_decode_$(date +%H%M%S).log}"

# 负载门放在最前：本地无 conda/无 GPU 时也能先做判断，避免"跑了一半才发现机器忙"
MAXLOAD="${MAXLOAD:-12}"; ROUNDS="${ROUNDS:-3}"
_load1=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$_load1" -v m="$MAXLOAD" 'BEGIN{exit !(l>m)}' && [ "${FORCE:-0}" != "1" ]; then
    echo "[ab] 当前 load=$_load1 > MAXLOAD=$MAXLOAD ⇒ 拒绝取数（确要跑：FORCE=1）" | tee -a "$LOG"
    exit 3
fi

source "$CONDA/etc/profile.d/conda.sh"
conda activate "$ENVNAME"
cd "$REPO" || exit 1

export XIAOTU_LAYER1_NPZ="$CKPT" BS="$BS" REP="$REP" DEDUP="$DEDUP" CHECK=1

echo "[ab] repo=$REPO threads=$THREADS dedup=$DEDUP bs=$BS rep=$REP" | tee "$LOG"
echo "[ab] load=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "[ab] so=$(ls -la xiaotu_moe/build/_xiaotu_moe_C_avx512_bf16_vbmi*.so 2>/dev/null | head -1)" | tee -a "$LOG"

echo "[ab] load=$_load1 rounds=$ROUNDS" | tee -a "$LOG"

_HASH_REF=""; _HASH_BAD=0
run_one() {           # run_one <标签> <env 赋值...>：跑 ROUNDS 轮取 min，并校验哈希一致
    local tag="$1"; shift
    local best="" h="" ms r
    for r in $(seq 1 "$ROUNDS"); do
        local out
        out=$(env "$@" THREADS="$THREADS" XIAOTU_MOE_THREADS="$THREADS" \
              nice -n 19 python scripts/bench_cpu_engine.py 2>&1)
        ms=$(printf '%s\n' "$out" | awk -v b="$BS" '$1==b && NF>=5 {print $2; exit}')
        h=$(printf '%s\n' "$out" | sed -n 's/.*\[hash\] .*sha256=\([0-9a-f]*\).*/\1/p' | head -1)
        if [ -z "$h" ] && [ -z "$_HASH_REF" ]; then
            echo "[ab] $tag 没拿到 [hash] ⇒ 检查 CHECK=1 是否生效" | tee -a "$LOG"
        fi
        if [ -z "$_HASH_REF" ] && [ -n "$h" ]; then _HASH_REF="$h"; fi
        if [ -n "$h" ] && [ -n "$_HASH_REF" ] && [ "$h" != "$_HASH_REF" ]; then
            echo "[ab] !! $tag 哈希与基准不同：$h vs $_HASH_REF" | tee -a "$LOG"; _HASH_BAD=1
        fi
        if [ -n "${ms:-}" ] && awk -v a="$ms" -v b="${best:-999}" 'BEGIN{exit !(a<b)}'; then best="$ms"; fi
    done
    printf '%-22s min_ms=%-8s sha=%s load=%s\n' \
        "$tag" "${best:-ERR}" "${h:-none}" "$(cut -d' ' -f1 /proc/loadavg)" | tee -a "$LOG"
}

# ---- 矩阵 ----------------------------------------------------------------
run_one "baseline(默认)"          XIAOTU_MOE_C_WLIMIT=0 XIAOTU_PIN_HITSTAT_HOST=1
run_one "M1 C_WLIMIT=1"          XIAOTU_MOE_C_WLIMIT=1 XIAOTU_PIN_HITSTAT_HOST=1
run_one "R1 旧同步(对照)"         XIAOTU_MOE_C_WLIMIT=0 XIAOTU_PIN_HITSTAT_HOST=0

# ---- H1 预取变体（需要先把构建好的 .so 放进 xiaotu_moe/build/） ----------
for v in avx512_bf16_vbmi_pf4 avx512_bf16_vbmi_pf avx512_bf16_vbmi_pf16; do
    if compgen -G "xiaotu_moe/build/_xiaotu_moe_C_${v}.cpython-*.so" > /dev/null; then
        run_one "H1 预取 $v"      XIAOTU_MOE_C_WLIMIT=0 XIAOTU_PIN_HITSTAT_HOST=1 \
                                      XIAOTU_MOE_VARIANT="$v"
    else
        echo "[ab] skip 变体 $v（未构建：xiaotu_moe/build/_xiaotu_moe_C_${v}.*.so 不存在）" | tee -a "$LOG"
    fi
done

echo "[ab] log=$LOG"
if [ "$_HASH_BAD" = "1" ]; then echo "[ab] ✗ 存在哈希不一致的臂 ⇒ 立即回退该开关，不要看性能"; exit 2; fi
echo "[ab] ✓ 所有臂 sha256 一致；min_ms 为 ${ROUNDS} 轮最小值"
