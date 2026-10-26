#!/usr/bin/env bash
# ============================================================================
# run_bench_cached.sh —— qfn 4096-prompt GPU-prefill 基准启动器（**仓库版**）
#
# 来由：2026-10-24 收尾时这一份只在目标机 `/tmp/` 里，机器一重启就没了（HANDOFF §2/§8）。
#       ⇒ 本版进仓库，由 `scripts/musa/` 统一分发，`/tmp` 只放运行产物。
#
# 口径（HANDOFF §2，2026-10-24 实测最佳）：
#   QFN_LEN=4096 QFN_MAXLEN=8192 QFN_REPEATS=2 QFN_GPU_UTIL=0.72 QFN_MBT=4096
#   XIAOTU_GPF_GEMM=bmm XIAOTU_GPF_BMM_CHUNK=64 XIAOTU_GPF_PIN=0 XIAOTU_GPF_PHASE_LOG=1
#   期望：RUN 1/2 TTFT≈48–50 s，prefill≈83–85 tok/s（CPU 基线 261 s / 15.7）
#
# KV 指纹缓存：`kv_cache_memory_bytes` 能让 vLLM **跳过 profile**（省 2–6 min 启动）。
#   指纹 = (model,tp,maxlen,util,mbt,len) 的 md5；命中即 export QFN_KV_BYTES。
#   ⛔ 缓存**默认不填充**：值必须来自一次真实跑通后的实测，不许我拍脑袋写死。
#      填充方式：`bash run_bench_cached.sh --record-kv <bytes>`（见文末）。
#
# 用法（目标机，**后台**）：
#   nohup setsid bash run_bench_cached.sh > /tmp/bench.outer.log 2>&1 < /dev/null &
#   然后挂 musa_watch.sh 看状态；会话侧只看状态文件，**不要**前台等。
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${MUSA_CACHE_DIR:-/root/vllm-xiaotu-moe-tmp/cache}"
BENCH_PY="${MUSA_BENCH_PY:-/root/vllm-xiaotu-moe-tmp/scripts/bench_6x.py}"
PY="${MUSA_PY:-/root/venvs/vllm310/bin/python}"
# ⛔ 不许放 /tmp：/tmp 是 **tmpfs（吃内存）**，且重启即清（用户 2026-10-24 两次指出）
RUNDIR="${MUSA_RUNDIR:-/root/lvllm/tmp/bench}"
mkdir -p "$RUNDIR" || { echo "mkdir $RUNDIR failed"; exit 3; }
LOG="${BENCH_LOG:-$RUNDIR/bench.log}"
PIDFILE="${BENCH_PIDFILE:-$RUNDIR/qfn_bench.pid}"

export QFN_MODEL="${QFN_MODEL:-/root/models/Qwen3.8-Flash-Next-FP8}"
export QFN_TP="${QFN_TP:-2}"
export QFN_LEN="${QFN_LEN:-4096}"
export QFN_MAXLEN="${QFN_MAXLEN:-8192}"
export QFN_REPEATS="${QFN_REPEATS:-2}"
export QFN_GPU_UTIL="${QFN_GPU_UTIL:-0.72}"
export QFN_MBT="${QFN_MBT:-4096}"
export XIAOTU_GPF_GEMM="${XIAOTU_GPF_GEMM:-bmm}"
export XIAOTU_GPF_BMM_CHUNK="${XIAOTU_GPF_BMM_CHUNK:-64}"
export XIAOTU_GPF_PIN="${XIAOTU_GPF_PIN:-0}"
export XIAOTU_GPF_PHASE_LOG="${XIAOTU_GPF_PHASE_LOG:-1}"
export XIAOTU_GPF_W2TN="${XIAOTU_GPF_W2TN:-0}"
# ⭐⭐【2026-10-24 补】专家权重**必须留 CPU**：漏了这条 ⇒ `create_weights` 把 512 专家建成
#   显卡张量 ⇒ 建到第 N 层就 `MUSA out of memory`（实测 45.04 GiB 已分配 / 剩 345 MiB）。
#   这正是 BUG_REGISTRY **C11** 那类"复刻时漏了隐含 env"；HANDOFF_QFN_STARTUP §C 也点了这一条。
# ⭐⭐ 2026-10-25：vLLM 默认 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`，
# 单次 execute_model 超过 5 分钟就 `TimeoutError: RPC call ... timed out`
# ⇒ `EngineDeadError` + workers 空转（必须重启整机）。实测约 **1/2 的 run 会撞上**，
# 是实验成本的主要来源。调大到 30 分钟：卡死变成"慢"，我们至少能拿到数据。
# ⭐ 2026-10-25：tile 粒度实测（qfn E=512）——默认 `E//3=170` ⇒ 4 个 tile；改成 256 ⇒ **2 个 tile**
#   层时 573–628 → **562–577 ms**、TTFT 38.80 → **37.47 s**（prefill 105.6 → 109.3 tok/s）✓
#   机理：每层 Python/内核启动次数减半（环槽仍 2 个，显存 1.0 → 1.5 GB，preflight slack 仍 >3 GiB）。
#   回滚：`XIAOTU_GPF_STAGE_TILE_E=170`
: "${XIAOTU_GPF_STAGE_TILE_E:=256}"
export XIAOTU_GPF_STAGE_TILE_E

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

export VLLM_EXPERTS_LOAD_DEVICE="${VLLM_EXPERTS_LOAD_DEVICE:-cpu}"
# ⭐⭐【2026-10-24 补 · 第二个致命隐含 env】GPU 预填充的**总开关**：
#   `gpu_prefill.gpu_prefill_min_tokens()` 读 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`，
#   **默认 "0" ⇒ 整个 GPU 预填是关的**，于是 4096 prompt 会静默走 CPU（≈260 s/次，5× 慢），
#   而日志里**只有** `[xtu-pf-progress] device=cpu` 这一处线索（`GPU prefill ACTIVE` 根本不打印）。
#   历史口径（`dev-docs/CPU_MOE_PLAN.md:717`）= GPU 路径用 **4096**（qlen≥阈值才走 GPU）。
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="${VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:-4096}"
# MUSA 双卡：tp=2 必须 MCCL_P2P_DISABLE=1（P2P 直连在这台机器上不可用，BRIEF §5.4）
export MCCL_P2P_DISABLE="${MCCL_P2P_DISABLE:-1}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1}"
# 线程按 24/rank 切（tp=2 时 48 线程不压 64 核；BRIEF §7）
export XIAOTU_MOE_THREADS="${XIAOTU_MOE_THREADS:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

fp_src="v1|$QFN_MODEL|$QFN_TP|$QFN_MAXLEN|$QFN_GPU_UTIL|$QFN_MBT|$QFN_LEN"
FP=$(printf '%s' "$fp_src" | md5sum | cut -c1-16)
KVFILE="$CACHE_DIR/kv_$FP.txt"

if [ "${1:-}" = "--record-kv" ]; then
  mkdir -p "$CACHE_DIR" || exit 3
  printf '%s\n%s\n' "${2:?bytes}" "$fp_src" > "$KVFILE"
  echo "recorded kv=$2 fp=$FP -> $KVFILE"
  exit 0
fi

if [ -f "$KVFILE" ]; then
  export QFN_KV_BYTES=$(head -1 "$KVFILE")
  echo "[cached] fp=$FP 命中 ⇒ QFN_KV_BYTES=$QFN_KV_BYTES（vLLM 将跳过 profile）"
else
  echo "[cached] fp=$FP 未命中 ⇒ 走 vLLM 自身 profile（启动更慢，但口径已知可信）"
  echo "[cached] 指纹来源：$fp_src"
fi

echo "==== config ===="
for v in QFN_MODEL QFN_TP QFN_LEN QFN_MAXLEN QFN_REPEATS QFN_GPU_UTIL QFN_MBT \
         XIAOTU_GPF_GEMM XIAOTU_GPF_BMM_CHUNK XIAOTU_GPF_PIN XIAOTU_GPF_PHASE_LOG \
         XIAOTU_GPF_W2TN XIAOTU_GPF_W13TN XIAOTU_GPF_EVT XIAOTU_GPF_EVT_EVERY XIAOTU_GPF_DECODE_BLK XIAOTU_GPF_DECODE_WARPS XIAOTU_GPF_BOUNDS_PIPE XIAOTU_GPF_BOUNDS_REUSE XIAOTU_GPF_PROF XIAOTU_GPF_NOSYNC XIAOTU_GPF_STAGE_TILE_E XIAOTU_GPF_BMM_MAXROWS XIAOTU_GPF_BMM_CHUNK XIAOTU_GPF_DMA_CPU XIAOTU_GPF_STAGE_NSLOTS \
         VLLM_EXPERTS_LOAD_DEVICE VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS \
         MCCL_P2P_DISABLE MUSA_VISIBLE_DEVICES XIAOTU_MOE_THREADS \
         XIAOTU_MOE_GPU_RESIDENT_LAYERS XIAOTU_SPEC_DECODE XIAOTU_PLE_CPU; do
  printf '  %-28s= %s\n' "$v" "$(eval "printf '%s' \"\${$v:-}\"")"
done
echo "  log=$LOG  pidfile=$PIDFILE"
echo "================"

# ============================================================================
# ⭐⭐⭐【2026-10-25 关键修复】`XIAOTU_*` 开关**必须经 `XIAOTU_ENV_FILE` 递给 worker**
#
# 事实（`vllm_xiaotu_moe/__init__.py` 的头注释，已经咬过我们两次：§382 / §414）：
#   「EngineCore 的 environ 是被**重建**的（实测 2711 条 vs launcher 76 条），
#     重建时**只保留它认识的名字** ⇒ 于是"已经在环境里"的变量反而会被丢掉」
# ⇒ 本脚本 2026-10-25 实测复现：launcher 里 `XIAOTU_GPF_EVT=1`，而两个
#   `VLLM::Worker` 的 `/proc/<pid>/environ` 里**一个 XIAOTU 键都没有** ⇒
#   ⇒ 那一跑的 event 归因"设了等于没设"，且**日志里无声** ✗
#   ⚠️ 更严重的是：这会让"某开关是否生效"变成**按变量名随机**的事 ⇒
#      任何 A/B 结论都必须**先证明开关到了 worker**，否则不可信。
#
# 机制的正式用法：显式指定 `XIAOTU_ENV_FILE` ⇒ 插件**以文件为准（覆盖）**，
# 并且**逐行打印实际生效的键**（`[vllm-xtu-moe/env] K=V`）⇒ 从此"设了没生效"有迹可查 ✓
# ============================================================================
ENVFILE="${XIAOTU_ENV_FILE:-$RUNDIR/xiaotu_env}"
export XIAOTU_ENV_FILE="$ENVFILE"
: > "$ENVFILE"
# ⭐ 2026-10-25：**最小 env 集**（只有经过验证/必需的才投递）。
#   来由：env-file 修好后，`XIAOTU_MOE_THREADS=24` / `XIAOTU_GPF_PIN=0` 等**首次真正生效**，
#   同期两次"RUN1 后卡死 + workers 800% 空转"都发生在这些臂上 ⇒ 先怀疑"新到位的开关"，
#   用最小集跑通基线，再**一次加回一个**。`XIAOTU_MOE_THREADS` 故意不投递（用引擎默认）。
for v in XIAOTU_GPF_GEMM XIAOTU_GPF_PIN XIAOTU_GPF_PHASE_LOG \
         XIAOTU_GPF_W2TN XIAOTU_GPF_W13TN XIAOTU_GPF_CPT XIAOTU_GPF_EVT XIAOTU_GPF_EVT_EVERY XIAOTU_GPF_DECODE_BLK XIAOTU_GPF_DECODE_WARPS XIAOTU_GPF_BOUNDS_PIPE XIAOTU_GPF_BOUNDS_REUSE XIAOTU_GPF_PROF XIAOTU_GPF_NOSYNC XIAOTU_GPF_STAGE_TILE_E XIAOTU_GPF_BMM_MAXROWS XIAOTU_GPF_BMM_CHUNK XIAOTU_GPF_DMA_BENCH XIAOTU_GPF_DMA_CPU XIAOTU_GPF_STAGE_NSLOTS XIAOTU_GPF_DMA_DEBUG \
         XIAOTU_MOE_GPU_RESIDENT_LAYERS XIAOTU_SPEC_DECODE \
         GPU_DEVICE XTU_MUSA_CORRECTNESS; do   # decode 端到端基准需要投递层常驻
# ⭐ GPU_DEVICE 必须进 env-file：EngineCore 会重建 environ 并丢掉不认识的名字，
#   而 MUSA 路径的判定发生在 **worker 进程**（mixed_experts._musa_active）。
#   不投递 ⇒ launcher 里 `GPU_DEVICE=MUSA` 在 worker 侧等于没设（且日志无声）。
  val="$(eval "printf '%s' \"\${$v:-}\"")"
  [ -n "$val" ] && printf '%s=%s\n' "$v" "$val" >> "$ENVFILE"
done
echo "[env-file] $ENVFILE ($(grep -c . "$ENVFILE" 2>/dev/null || echo 0) 个键) —— worker 侧将以它为准"
cat "$ENVFILE" | sed 's/^/  | /'


# ⭐ 落一份**完整 env 快照**：BUG_REGISTRY C11 的根因就是"复刻时漏了隐含 env"
#    ⇒ 从此"这一跑到底带了什么 env"有物证，不靠记忆、不靠交接文档的文字。
env | LC_ALL=C sort > "$RUNDIR/env.snapshot" 2>/dev/null || true
echo "[env] snapshot -> $RUNDIR/env.snapshot"

# ============================================================================
# ⭐ 前置门（**fail-closed**：任一不满足即拒绝启动，不猜、不抢资源）
#   为什么必须加（2026-10-25 实测）：旧的 `VLLM::Worker` 在 `kill -TERM` 之后仍会
#   在 CPU 暖机里继续跑（state=R），并继续占着 **42 GiB 显存**；此时新实例照样起 ⇒
#   要么 OOM、要么两个实例抢显存，而且**日志上看起来"新实例启动了"** ✗
#   绕过请显式 `MUSA_SKIP_GATE=1`（危险，需自负其责）。
# ============================================================================
if [ "${MUSA_SKIP_GATE:-0}" != "1" ]; then
  gate_fail=0
  avail=$(free -g | awk 'NR==2{print $7}')
  need_avail="${MUSA_GATE_AVAIL_GB:-200}"
  if [ "${avail:-0}" -lt "$need_avail" ]; then
    echo "⛔ 门1 MemAvailable=${avail}G < ${need_avail}G"
    gate_fail=1
  fi
  nwork=$(ps -eo cmd --no-headers 2>/dev/null | awk '$0 ~ /VLLM::Worker/ && $0 !~ /awk/' | wc -l)
  if [ "${nwork:-0}" -gt 0 ]; then
    echo "⛔ 门2 仍有 ${nwork} 个 VLLM::Worker 在跑（先确认它们退了；不要 kill -9）"
    ps -eo pid,pgid,stat,cmd --no-headers 2>/dev/null | awk '$0 ~ /VLLM::Worker/ && $0 !~ /awk/' | sed 's/^/    /' | head -4
    gate_fail=1
  fi
  vram=$("$PY" - <<'PYEOF' 2>/dev/null
import json, subprocess
try:
    d = json.loads(subprocess.run(["mthreads-gmi", "-q", "--json"],
                                  capture_output=True, text=True).stdout)
    used = 0
    for g in d.get("GPU", []):
        u = g.get("FB Memory Usage", {}).get("Used", "0MiB")
        used += int("".join(c for c in str(u) if c.isdigit()) or 0)
    print(used)
except Exception:
    print(-1)
PYEOF
)
  if [ "${vram:--1}" -lt 0 ]; then
    echo "⛔ 门3 读不到显存（mthreads-gmi 失败）⇒ 拒绝启动（fail-closed）"
    gate_fail=1
  elif [ "$vram" -gt "${MUSA_GATE_VRAM_MIB:-512}" ]; then
    echo "⛔ 门3 显存已用 ${vram}MiB > ${MUSA_GATE_VRAM_MIB:-512}MiB（有残留，先查清来源）"
    gate_fail=1
  fi
  if [ "$gate_fail" != "0" ]; then
    echo "⛔ 前置门未过 ⇒ 不启动（MUSA_SKIP_GATE=1 可显式绕过）"
    exit 3
  fi
  echo "[gate] 通过：MemAvailable=${avail}G，VLLM::Worker=0，VRAM=${vram}MiB"
fi

echo $$ > "$PIDFILE"

cd /root/lvllm || { echo "cd /root/lvllm failed"; exit 3; }
set -o pipefail
"$PY" "$BENCH_PY" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
echo "[cached] python exited rc=$rc"
[ "$rc" -eq 0 ] || echo "[cached] ⛔ 非 0 退出；先看 $LOG 里的 Traceback / FW_PANIC / musaErrorUnknown"
rm -f "$PIDFILE"
exit "$rc"
