#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# serve_ds41.sh —— ds41f（DeepSeek-V4.1-Flash）**唯一**启动入口
#
# ⛔⛔ 单一入口纪律（2026-11-01 用户明令）：
#   「给 ds41 设立一个'单一'的启动脚本，所有的引用，不管是调试还什么的，都必须从这里开始」
#   ⇒ 任何进程若要以 ds41f 起在 GPU1/GPU2，**只能**通过本脚本的 start 子命令。
#   ⇒ ⛔ 禁止再直接 `python -m vllm.entrypoints.openai.api_server`（历史事故：
#      绕过入口直起 ⇒ `COMPILE_RANGES` 从未被翻译进命令行 ⇒ Inductor 按 token 数
#      产生 853,377 个编译产物 ⇒ 显存随形状单调上涨 ⇒ OOM）。
#
# 用法：
#   bash scripts/serve_ds41.sh start   [NAME=VAL ...] [+--flag=值 ...]
#   bash scripts/serve_ds41.sh stop    [TAG=NAME]
#   bash scripts/serve_ds41.sh status  [TAG=NAME]
#   bash scripts/serve_ds41.sh restart [NAME=VAL ...] [+--flag=值 ...]
#
# 可覆盖的 NAME=VAL（白名单外一律拒绝）：
#   TAG=prod PORT=8700 TP=2 GPU_UTILIZATION=0.90 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=8192 MAXLEN=524288
#   （旧短名 GPU_UTIL / MAXSEQS / MBT / NOPC / EXPANDABLE / CED 仍接受但弃用，见 dev-docs/GLOSSARY.md）
#   THREADS=60 IMAGE_MAX=128 COMPILE_RANGES=32,256,8192
#   PIN=1 PIN_SLOTS=15 PIN_MAXQ=16 PIN_CUDA_MOE=0 MARLIN=0 SWA_BOUNDED_REPLAY=1 WARMUP=1 KV_DTYPE=fp8_ds_mla
#   IDLE_EC=0 IDLE_EC_SECONDS=60   预热收尾即归还 torch 缓存（**默认关**；1=开）
#                                  ⚠️ 2026-11-05 回退：C 判据（≥2h 带流量）未通过 ——
#                                  实测窗口内无推理流量、且时长仅 1h50m，无证据支撑默认开启。
#                                  ⚠️ IDLE_EC_SECONDS 在本入口**不生效**：本入口总是设
#                                  XIAOTU_RELEASE_AFTER_LOG（预热日志收尾触发），此时空闲线程
#                                  主动让位、该参数用不到；仅"未设触发器的兜底臂"才用它。
#                                  ⚠️ WARMUP=0 时**不会归还**（没有预热高水位，且空闲线程已让位）。
#   SPEC=1 MODEL_DIR=<路径>
#   LOG_KEEP=3     启动时保留几份历史日志（环境变量；归档命名 <log>.<时间戳>，上限 50）
#   EXPANDABLE=1（默认）  expandable_segments:True + 追加 --disable-custom-all-reduce（碎片治理；
#                        custom AR 退回 NCCL）。=0 ⇒ expandable_segments:False，custom AR 保持开启。
#
#
# ⛔ 启动冒烟（2026-11-04 起，**独立于 WARMUP**）：start 返回后仍会在后台跑一次
#    "混合并发"冒烟（1 路短 prompt 长解码 + 1 路 ~380 token 短 prefill），
#    上限 10 分钟（正常约 6 秒）、固定占 2 个 slot。长 prefill 耐力（6.9K+/~10min）由手工压测覆盖，不进启动闸门。⇒ **测速/TTFT 前先看 `status`**：
#    出现「⏳ 启动冒烟进行中」就等它结束；失败会写 `<TAG>.smoke.fail`，
#    `status` 会以「⛔ 冒烟失败」显示（即使服务已死也能看到）。
#
# +额外参数：只允许**白名单之外**的 `--flag` / `--flag=值`；
#   会覆盖本脚本保证口径的旗标（--port/--compilation-config/--no-swa-bounded-replay…）一律拒绝。
#
# 退出码：0 成功 / 2 参数错误 / 3 前置条件不满足（端口、锁、内存门、归属冲突）
#         4 启动失败 / 5 就绪超时 / 6 前缀缓存断言失败
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/user/lvllm/vllm-xiaotu-moe}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
# ⭐ 2026-11-10：系统盘重装后系统 libstdc++ 只到 CXXABI_1.3.13，而 xiaotu_moe 扩展需要
# CXXABI_1.3.15（conda env 的 libstdc++ 有）⇒ 必须把 env 的 lib 放进 LD_LIBRARY_PATH，
# 否则 worker 子进程 import 扩展直接失败（已实测）。
# ⭐ 2026-11-10 审计 §三.2：收窄写法——只前置**该 python 的** libdir，并断言里面真有
#    CXXABI_1.3.15；否则兜底静默无效、故障会原样复发。
case "$PYTHON" in
    /*) ;;
    *) echo "[ds41] ⛔ PYTHON 必须是绝对路径（得到 '$PYTHON'）" >&2; exit 2 ;;
esac
#    候选顺序：① `sys.prefix/lib`（conda env 自己的 lib，**正确的那一个**；
#    ⚠️ 不能用 sysconfig 的 LIBDIR —— 实测它返回系统 `/usr/lib/x86_64-linux-gnu`）② 由 PYTHON 路径推
_XTU_PYLIB=""
_xtu_cands=""
if [ -x "$PYTHON" ]; then
    _xtu_cands="$("$PYTHON" -c 'import sys;print(sys.prefix+"/lib")' 2>/dev/null || true)"
fi
_xtu_cands="$_xtu_cands $(dirname "$(dirname "$PYTHON")")/lib"
_xtu_has_sym=0; command -v strings >/dev/null 2>&1 && _xtu_has_sym=1
for _c in $_xtu_cands; do
    [ -f "$_c/libstdc++.so.6" ] || continue
    if [ "$_xtu_has_sym" = "1" ]; then
        # ⛔ 不能用 `strings | grep -q`：`set -o pipefail` 下 grep -q 提前退出会让 strings
        #    吃到 SIGPIPE（141）⇒ 管道整体返回失败 ⇒ 误判"没有该符号"（本脚本踩过一次）
        _xtu_syms="$(strings "$_c/libstdc++.so.6" 2>/dev/null || true)"
        case "$_xtu_syms" in *CXXABI_1.3.15*) ;; *) continue ;; esac
    fi
    _XTU_PYLIB="$_c"; break
done
if [ -n "$_XTU_PYLIB" ]; then
    export LD_LIBRARY_PATH="$_XTU_PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[ds41] libstdc++ 兜底生效（CXXABI_1.3.15 已确认）：$_XTU_PYLIB" >&2
else
    echo "[ds41] ⚠️ 未找到含 CXXABI_1.3.15 的 libstdc++（候选：$_xtu_cands）⇒ 不前置；插件可能 import 失败" >&2
fi

MODEL_DIR_DEFAULT="/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master"
SERVED_NAME="DeepSeek-V4.1-Flash"
CACHE_DIR="/home/user/.cache/vllm/torch_compile_cache/xtu-master-tp2-vllm_compile-f03026a548-9afa2a11"
WARMUP_SCRIPT="${WARMUP_SCRIPT:-/home/user/lvllm/stress-run5/warmup_shapes.py}"
MEM_GATE="${MEM_GATE:-$REPO_DIR/scripts/prelaunch_mem_gate.sh}"
# 本机实际部署树（REPO_DIR/vllm 不存在；跨树校验按真实位置比对，不按名字猜）
VLLM_DIR="${VLLM_DIR:-/home/user/lvllm/process_data/ref/repos/vllm-mainline/vllm}"   # 注意：vpath 是包目录 dirname(vllm.__file__)

_XTU_PORT_ENV="${PORT:-}"   # ⭐ 审计 B2：必须先记住"调用者是否显式给了 PORT"，再赋默认值
TAG="${TAG:-prod}"; PORT="${PORT:-8700}"; TP="${TP:-2}"
# ⭐ 2026-11-10 审计 B1：短名/全名的**环境变量**形式都必须生效（原先这行会把 env 值覆盖掉）
GPU_UTIL="${GPU_UTIL:-0.9}"; MAXSEQS="${MAXSEQS:-4}"; MBT="${MBT:-8192}"; MAXLEN="${MAXLEN:-524288}"
THREADS=60; IMAGE_MAX=128; COMPILE_RANGES="32,256,8192"
PIN=1; PIN_SLOTS=15; PIN_MAXQ=16; PIN_CUDA_MOE=0; MARLIN=0; WARMUP=1
# ⭐⭐ 2026-11-10 复审 N1：必须在赋默认值**之前**记下"调用者是否显式给了 SWA_BOUNDED_REPLAY"，
#    否则 SWA_EXPLICIT 恒为 1 ⇒ 单独写 `CED=0` 会被误判成"与显式 SWA 冲突"（fail-closed 但挡住合法用法）。
_XTU_SWA_ENV="${SWA_BOUNDED_REPLAY:-}"
SWA_BOUNDED_REPLAY="${SWA_BOUNDED_REPLAY:-1}"
SWA_EXPLICIT=0; [ -n "$_XTU_SWA_ENV" ] && SWA_EXPLICIT=1
PORT_EXPLICIT=0; [ -n "$_XTU_PORT_ENV" ] && PORT_EXPLICIT=1
# 无歧义参数名（推荐）；旧短名为别名，见 dev-docs/GLOSSARY.md
NO_PREFIX_CACHE="${NO_PREFIX_CACHE:-${NOPC:-0}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MBT:-8192}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${MAXSEQS:-4}}"
GPU_UTILIZATION="${GPU_UTILIZATION:-${GPU_UTIL:-0.9}}"
EXPANDABLE_SEGMENTS="${EXPANDABLE_SEGMENTS:-${EXPANDABLE:-1}}"
NOPC_ALIAS=0; MBT_ALIAS=0; MAXSEQS_ALIAS=0; GPU_UTIL_ALIAS=0; EXPANDABLE_ALIAS=0
# 历史别名（弃用）：CED=0/1 —— 语义等同于 SWA_BOUNDED_REPLAY=0/1；设置它会在启动时警告
CED_ALIAS=""; CED_EXPLICIT=0
# 环境变量形式的历史别名（CED=0/1）
if [ -n "${CED:-}" ]; then CED_ALIAS="$CED"; CED_EXPLICIT=1; fi
IDLE_EC=0; IDLE_EC_SECONDS=60   # 预热收尾即归还显存缓存（实测单卡归还 10.4–10.73 GiB；默认开）
PC=0; PC_GB=64; PC_BLOCKS=16; PC_ROOT="/home/user/.cache/prefix-cache"
PC_FREE_MIN_MIB="${PC_FREE_MIN_MIB:-524288}"
PC_WRITE_WAIT="${PC_WRITE_WAIT:-900}"
ALLOC_CONF_EFFECTIVE=""
KV_DTYPE="fp8_ds_mla"; SPEC=1; MODEL_DIR="$MODEL_DIR_DEFAULT"; EXPANDABLE=1
EXTRA_ARGS=()

die() { printf '[ds41] ⛔ %s\n' "$1" >&2; exit "${2:-2}"; }
say() { printf '[ds41] %s\n' "$*"; }
# ⭐ 审计 §五.8：原先是 2,29p，把 30-33 行（MODEL_DIR / LOG_KEEP / EXPANDABLE）的帮助切掉了
usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }

# ── 参数解析 ───────────────────────────────────────────────────────────────
CMD="${1:-}"; shift || true
[ -z "$CMD" ] && { usage; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        +*) EXTRA_ARGS+=("${1#+}"); shift ;;
        *=*)
            k="${1%%=*}"; v="${1#*=}"
            case "$k" in
                TAG) TAG="$v" ;;
                PORT) PORT="$v"; PORT_EXPLICIT=1 ;;
                TP) TP="$v" ;;
                GPU_UTIL) GPU_UTIL="$v"; GPU_UTILIZATION="$v"; GPU_UTIL_ALIAS=1 ;;
                # ↑ GPU_UTIL 为历史短名，等价于 GPU_UTILIZATION（弃用）
                MAXSEQS) MAXSEQS="$v"; MAX_NUM_SEQS="$v"; MAXSEQS_ALIAS=1 ;;
                # ↑ MAXSEQS 为历史短名，等价于 MAX_NUM_SEQS（弃用）
                MBT) MBT="$v"; MAX_NUM_BATCHED_TOKENS="$v"; MBT_ALIAS=1 ;;
                # ↑ MBT 为历史短名，等价于 MAX_NUM_BATCHED_TOKENS（弃用）
                MAXLEN) MAXLEN="$v" ;;
                THREADS) THREADS="$v" ;;
                IMAGE_MAX) IMAGE_MAX="$v" ;;
                COMPILE_RANGES) COMPILE_RANGES="$v" ;;
                PIN) PIN="$v" ;;
                PIN_SLOTS) PIN_SLOTS="$v" ;;
                PIN_CUDA_MOE) PIN_CUDA_MOE="$v" ;;
                MARLIN) MARLIN="$v" ;;
                IDLE_EC) IDLE_EC="$v" ;;
                IDLE_EC_SECONDS) IDLE_EC_SECONDS="$v" ;;
                PIN_MAXQ) PIN_MAXQ="$v" ;;
                SWA_BOUNDED_REPLAY) SWA_BOUNDED_REPLAY="$v"; SWA_EXPLICIT=1 ;;
                NO_PREFIX_CACHE) NO_PREFIX_CACHE="$v"; NOPC="$v" ;;
                # ↑ NOPC 为历史短名（CLI 亦可写 NOPC=1，审计 B1 修正）
                NOPC) NOPC="$v"; NO_PREFIX_CACHE="$v"; NOPC_ALIAS=1 ;;
                MAX_NUM_BATCHED_TOKENS) MAX_NUM_BATCHED_TOKENS="$v"; MBT="$v" ;;
                MAX_NUM_SEQS) MAX_NUM_SEQS="$v"; MAXSEQS="$v" ;;
                PIN_EXPERTS_TO_GPU) PIN="$v" ;;
                PIN_MAX_PENDING_REQUESTS) PIN_MAXQ="$v" ;;
                GPU_UTILIZATION) GPU_UTILIZATION="$v"; GPU_UTIL="$v" ;;
                EXPANDABLE_SEGMENTS) EXPANDABLE_SEGMENTS="$v"; EXPANDABLE="$v" ;;
                CED) CED_ALIAS="$v"; CED_EXPLICIT=1 ;;
                # ↑ CED 是历史别名，语义 = SWA_BOUNDED_REPLAY（弃用，仅兼容旧命令行）
                WARMUP) WARMUP="$v" ;;
                PC) PC="$v" ;;
                PC_GB) PC_GB="$v" ;;
                PC_BLOCKS) PC_BLOCKS="$v" ;;
                PC_ROOT) PC_ROOT="$v" ;;
                KV_DTYPE) KV_DTYPE="$v" ;;
                SPEC) SPEC="$v" ;;
                MODEL_DIR) MODEL_DIR="$v" ;;
                EXPANDABLE) EXPANDABLE="$v"; EXPANDABLE_SEGMENTS="$v" ;;
                # ↑ EXPANDABLE 为历史短名，等价于 EXPANDABLE_SEGMENTS（弃用）
                *) die "未知参数 '$k'（可用项见文件头；新增参数请改本脚本，不要绕过入口）" ;;
            esac
            shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "无法识别的参数 '$1'（用 TAG=NAME；老写法 --tag 不再支持）" ;;
    esac
done

case "$TAG" in ''|*[!A-Za-z0-9_-]*) die "TAG 只允许 [A-Za-z0-9_-]（得到 '$TAG'）" ;; esac

PID_FILE="$RUN_DIR/ds41_${TAG}.pid"
LOG_FILE="$RUN_DIR/ds41_${TAG}.log"
LOG_KEEP="${LOG_KEEP:-3}"   # 启动时保留几份历史日志（归档命名 <log>.<时间戳>）
# ⭐⭐ 2026-11-10 审计 B2（blocking）：PORT 必须可回读。
#    `stop TAG=x` 不带 PORT 时，原先会拿默认 8700 去比归属 ⇒ 失败后**删掉在跑实例的 PID 文件**。
PORT_FILE="$RUN_DIR/ds41_${TAG}.port"
if [ "$PORT_EXPLICIT" = "0" ] && [ -f "$PORT_FILE" ]; then
    _xtu_p="$(tr -dc '0-9' < "$PORT_FILE" 2>/dev/null || true)"
    if [ -n "$_xtu_p" ]; then
        PORT="$_xtu_p"
        echo "[ds41] 端口从 $PORT_FILE 回读为 $PORT" >&2
    fi
fi
# ⭐⭐ 2026-11-10 审计 B4（blocking）：锁**按 PORT**，否则两个 TAG 抢同一端口可同时通过检查。
#    ⚠️ 复审 N2：这一行必须在**PORT 回读之后**，否则不带 PORT 的 start 会锁错端口。
LOCK_FILE="$RUN_DIR/ds41_port_${PORT}.lock"

# ══ 端口归属三态：free / owned:PID / unknown —— unknown 一律按拒绝处理 ════════
port_state() {
    local p="${1:-$PORT}" rows pid
    command -v ss >/dev/null 2>&1 || { echo unknown; return; }
    rows="$(ss -Hltnp "sport = :$p" 2>/dev/null)" || { echo unknown; return; }
    [ -z "$rows" ] && { echo free; return; }
    pid="$(printf '%s\n' "$rows" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
    if [ -n "$pid" ]; then echo "owned:$pid"; else echo unknown; fi
}

pid_alive() {   # 僵尸/死亡不算活
    local p="${1:-}" st
    case "$p" in ''|*[!0-9]*) return 1 ;; esac
    [ -r "/proc/$p/stat" ] || return 1
    st="$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)"
    [ "$st" != "Z" ] && [ "$st" != "X" ]
}

verify_owns() {   # pid 必须确实是本实例：端口归属，或【本入口那条命令行】且 --port 匹配
    local p="${1:-}" ps
    pid_alive "$p" || return 1
    ps="$(port_state "$PORT")"
    [ "$ps" = "owned:$p" ] && return 0
    # ⛔ 兜底不能只看"argv 里有 8700"：`sleep 8700` 这种会被误判成本实例（审计实测）
    tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null \
        | grep -qx -- "vllm.entrypoints.openai.api_server" || return 1
    tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null \
        | awk -v p="$PORT" 'prev=="--port" && $0==p {f=1} {prev=$0} END{exit !f}'
}

is_uint() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }

# 碎片治理（2026-11-01 用户指示）：EXPANDABLE 与 expandable_segments 必须一致。
# ⛔ 检查放在**父进程**里（子 shell 里 die 会让父空转 600s 才报"孤儿"，审计 N1）。
resolve_alloc_conf() {
    if [ "$EXPANDABLE" = "1" ]; then
        case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
            *expandable_segments:False*)
                die "环境把 expandable_segments 设为 False，与 EXPANDABLE=1 冲突；请去掉该环境变量" 2 ;;
            *expandable_segments:True*)
                ALLOC_CONF_EFFECTIVE="$PYTORCH_CUDA_ALLOC_CONF" ;;
            '')
                ALLOC_CONF_EFFECTIVE="expandable_segments:True" ;;
            *)
                # 调用者给了别的组合但没写 expandable ⇒ 补上，保证臂名与行为一致（审计 N5）
                ALLOC_CONF_EFFECTIVE="$PYTORCH_CUDA_ALLOC_CONF,expandable_segments:True" ;;
        esac
    else
        case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
            *expandable_segments:True*)
                die "环境把 expandable_segments 设为 True，但 EXPANDABLE=0；要开请显式传 EXPANDABLE=1" 2 ;;
            '')
                ALLOC_CONF_EFFECTIVE="expandable_segments:False" ;;
            *)  ALLOC_CONF_EFFECTIVE="$PYTORCH_CUDA_ALLOC_CONF,expandable_segments:False" ;;
        esac
    fi
}

# ══ 校验（fail-closed） ═══════════════════════════════════════════════════════
validate() {
    # ⭐⭐ 2026-11-10 审计 B1（blocking）：**先同步、后校验**。
    #   原先同步块在范围校验之后 ⇒ 用全长名（CLI 或环境变量）传 MAX_NUM_SEQS /
    #   MAX_NUM_BATCHED_TOKENS / GPU_UTILIZATION 时，校验读到的是旧默认值 ⇒ **非法值直接进 vllm**。
    MBT="$MAX_NUM_BATCHED_TOKENS"
    MAXSEQS="$MAX_NUM_SEQS"
    GPU_UTIL="$GPU_UTILIZATION"
    EXPANDABLE="$EXPANDABLE_SEGMENTS"
    NOPC="$NO_PREFIX_CACHE"
    # ⭐ 2026-11-10 审计 B3（blocking）：CED 不得**静默**覆盖显式 SWA_BOUNDED_REPLAY。
    if [ -n "$CED_ALIAS" ]; then
        if [ "$CED_EXPLICIT" = "1" ] && [ "$SWA_EXPLICIT" = "1" ] \
           && [ "$SWA_BOUNDED_REPLAY" != "$CED_ALIAS" ]; then
            die "CED=$CED_ALIAS 与显式 SWA_BOUNDED_REPLAY=$SWA_BOUNDED_REPLAY 冲突（CED 已弃用）：只写 SWA_BOUNDED_REPLAY" 2
        fi
        say "⚠️ 参数 CED 已弃用（它实际控制 decoder-side SWA bounded replay）；请改用 SWA_BOUNDED_REPLAY=${CED_ALIAS}"
        SWA_BOUNDED_REPLAY="$CED_ALIAS"
    fi
    if [ "$MBT_ALIAS" = "1" ]; then say "⚠️ 参数 MBT 已弃用；请改用 MAX_NUM_BATCHED_TOKENS=$MBT"; fi
    if [ "$MAXSEQS_ALIAS" = "1" ]; then say "⚠️ 参数 MAXSEQS 已弃用；请改用 MAX_NUM_SEQS=$MAXSEQS"; fi
    if [ "$GPU_UTIL_ALIAS" = "1" ]; then say "⚠️ 参数 GPU_UTIL 已弃用；请改用 GPU_UTILIZATION=$GPU_UTIL"; fi
    if [ "$EXPANDABLE_ALIAS" = "1" ]; then say "⚠️ 参数 EXPANDABLE 已弃用；请改用 EXPANDABLE_SEGMENTS=$EXPANDABLE"; fi
    if [ "$NOPC_ALIAS" = "1" ]; then say "⚠️ 参数 NOPC 已弃用；请改用 NO_PREFIX_CACHE=$NOPC"; fi
    is_uint "$PORT" && [ "$PORT" -ge 1024 ] && [ "$PORT" -le 65535 ] || die "PORT 非法：'$PORT'"
    is_uint "$TP" && [ "$TP" -ge 1 ] && [ "$TP" -le 2 ] || die "TP 只能是 1 或 2（本机只放 GPU1/2）"
    is_uint "$MAXSEQS" && [ "$MAXSEQS" -ge 1 ] && [ "$MAXSEQS" -le 256 ] || die "MAXSEQS 越界：'$MAXSEQS'"
    is_uint "$MBT" && [ "$MBT" -ge 256 ] && [ "$MBT" -le 65536 ] || die "MBT 越界：'$MBT'"
    is_uint "$MAXLEN" && [ "$MAXLEN" -ge 4096 ] && [ "$MAXLEN" -le 1048576 ] || die "MAXLEN 越界：'$MAXLEN'"
    is_uint "$THREADS" || die "THREADS 必须是正整数（得到 '$THREADS'）"
    [ $(( THREADS % 12 )) -eq 0 ] || die "THREADS=$THREADS 不是 12 的倍数（CCD=12 核）"
    [ "$THREADS" -ge 12 ] && [ "$THREADS" -le 192 ] || die "THREADS=$THREADS 超出 12..192"
    is_uint "$PIN_SLOTS" && is_uint "$PIN_MAXQ" && is_uint "$IMAGE_MAX" \
        || die "PIN_SLOTS/PIN_MAXQ/IMAGE_MAX 必须是非负整数"
    [ "$IMAGE_MAX" -le 128 ] || die "IMAGE_MAX 上限 128（得到 '$IMAGE_MAX'）"
    [ "$PIN_SLOTS" -le 64 ] || die "PIN_SLOTS 上限 64（得到 '$PIN_SLOTS'）"
    case "$PIN_CUDA_MOE" in 0|1) ;; *) die "PIN_CUDA_MOE 只能是 0/1（得到 '$PIN_CUDA_MOE'）" ;; esac
    case "$MARLIN" in 0|1) ;; *) die "MARLIN 只能是 0/1（得到 '$MARLIN'）" ;; esac
    case "$IDLE_EC" in 0|1) ;; *) die "IDLE_EC 只能是 0/1（得到 '$IDLE_EC'）" ;; esac
    # 与 PC=0 的先例一致：不用的参数不该拦启动
    if [ "$IDLE_EC" = "1" ]; then
        is_uint "$IDLE_EC_SECONDS" && [ "$IDLE_EC_SECONDS" -ge 30 ] && [ "$IDLE_EC_SECONDS" -le 3600 ] \
            || die "IDLE_EC_SECONDS 越界 30..3600（得到 '$IDLE_EC_SECONDS'）"
    fi
    [ "$PIN_MAXQ" -le 256 ] || die "PIN_MAXQ 上限 256（得到 '$PIN_MAXQ'）"
    awk -v u="$GPU_UTIL" 'BEGIN{exit !(u+0>0.05 && u+0<=1.0)}' || die "GPU_UTIL 必须在 (0.05,1.0]（得到 '$GPU_UTIL'）"
    case "$SWA_BOUNDED_REPLAY" in 0|1) ;; *) die "SWA_BOUNDED_REPLAY 只能是 0/1（旧名 CED 亦接受）" ;; esac
    case "$PIN" in 0|1) ;; *) die "PIN 只能是 0/1" ;; esac
    case "$SPEC" in 0|1) ;; *) die "SPEC 只能是 0/1" ;; esac
    case "$WARMUP" in 0|1) ;; *) die "WARMUP 只能是 0/1（得到 '$WARMUP'）" ;; esac
    # ⛔ [2026-11-05 审计 B] LOG_KEEP 是**按 glob 删文件**的旋钮，比别的数值旋钮更危险：
    #    0 ⇒ `tail -n +1` 会把**刚归档的那份也删掉**（正好以新方式丢崩溃现场）；
    #    abc/1x ⇒ 算术展开失败，bash 在中途 abort，启动没做、退出码也不是文档里的那几个。
    case "$LOG_KEEP" in ''|0*|*[!0-9]*) die "LOG_KEEP 必须是正整数（得到 '$LOG_KEEP'）" ;; esac
    [ "$LOG_KEEP" -ge 1 ] && [ "$LOG_KEEP" -le 50 ] \
        || die "LOG_KEEP 越界 1..50（得到 $LOG_KEEP）"
    case "$EXPANDABLE" in 0|1) ;; *) die "EXPANDABLE 只能是 0/1" ;; esac
    case "$NO_PREFIX_CACHE" in 0|1) ;; *) die "NO_PREFIX_CACHE 只能是 0/1（旧名 NOPC 亦接受；得到 '$NO_PREFIX_CACHE'）" ;; esac
    case "$PC" in 0|1) ;; *) die "PC 只能是 0/1（得到 '$PC'）" ;; esac
    if [ "$PC" = "0" ]; then
        :   # PC=0 时 PC_GB/PC_BLOCKS/PC_ROOT 一律不校验（不用的参数不该拦启动）
    else
    case "$PC_GB" in ''|0*|*[!0-9]*) die "PC_GB 必须是无前导零的正整数（得到 '$PC_GB'）" ;; esac
    local _shm_gib
    _shm_gib="$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{printf "%d", $2/1024}')"
    case "${_shm_gib:-}" in ''|*[!0-9]*|0) die "读不到 /dev/shm 总量（df 失败）⇒ 拒绝（PC_GB 无从校验）" 3 ;; esac
    [ "$PC_GB" -ge 8 ] || die "PC_GB 必须 >= 8（得到 $PC_GB）"
    [ "$PC_GB" -le $(( _shm_gib - 8 )) ] \
        || die "PC_GB=$PC_GB 超出 /dev/shm 可用总量（本机 ${_shm_gib}GiB，须留 8GiB）；内存门按 900+PC_GB 计" 3
    case "$PC_BLOCKS" in ''|0*|*[!0-9]*) die "PC_BLOCKS 必须是无前导零的正整数（得到 '$PC_BLOCKS'）" ;; esac
    [ "$PC_BLOCKS" -ge 1 ] && [ "$PC_BLOCKS" -le 256 ] || die "PC_BLOCKS 超出 1..256"
    PC_ROOT="${PC_ROOT%/}"
    case "$PC_ROOT" in
        /*) ;;
        *) die "PC_ROOT 必须是绝对路径（得到 '$PC_ROOT'）" ;;
    esac
    case "$PC_ROOT" in
        /|*/../*|*/..) die "PC_ROOT 不得为 / 或含 ..（得到 '$PC_ROOT'）" ;;
    esac
    case "$PC_ROOT" in
        *[!A-Za-z0-9._/-]*) die "PC_ROOT 只允许 [A-Za-z0-9._/-]（要拼进 JSON，拒绝空白/引号/反斜杠/控制字符）：'$PC_ROOT'" ;;
    esac
    fi
    # 冲突检查必须尽早（父进程侧、且在耗时/依赖文件系统的检查之前）⇒ 审计 N1
    resolve_alloc_conf
    [ -d "$MODEL_DIR" ] || die "模型目录不存在：$MODEL_DIR"
    [ -x "$PYTHON" ] || die "python 不存在：$PYTHON"
    [ -d "$REPO_DIR" ] || die "仓库目录不存在：$REPO_DIR"

    if [ -n "$COMPILE_RANGES" ]; then
        case "$COMPILE_RANGES" in
            *$'\n'*|*$'\r'*|*$'\t'*|*' '*) die "COMPILE_RANGES 不得含空白/控制字符（拒绝多行绕过）" ;;
        esac
        printf '%s' "$COMPILE_RANGES" | LC_ALL=C grep -Eqz '^[1-9][0-9]*(,[1-9][0-9]*)*$' \
            || die "COMPILE_RANGES='$COMPILE_RANGES' 非法（只允许 a,b,c 正整数）"
        local _prev=0 _e
        for _e in ${COMPILE_RANGES//,/ }; do
            [ "${#_e}" -le 9 ] || die "COMPILE_RANGES 端点过大 '$_e'（上限 999999999）"
            [ "$_e" -gt "$_prev" ] || die "COMPILE_RANGES 必须严格递增（'$_e' 在 '$_prev' 之后）"
            _prev="$_e"
        done
    fi

    # 额外参数：必须 --flag，且不得覆盖本脚本保证的口径
    local _a _k
    local deny='^(--port|--model|--served-model-name|--tensor-parallel-size|--gpu-memory-utilization|--max-model-len|--max-num-seqs|--max-num-batched-tokens|--compilation-config|--kv-cache-dtype|--limit-mm-per-prompt|--speculative-config|--kv-transfer-config|--kv-offloading-size|--kv-offloading-backend|--swa-bounded-replay|--no-swa-bounded-replay|--enforce-eager|--enable-sleep-mode|--swap-space|--num-gpu-blocks-override|--block-size|--dtype|--quantization|--load-format|--kernel-config|--mm-processor-cache-gb)$'
    for _a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
        case "$_a" in --*) ;; *) die "额外参数必须是 --flag 或 --flag=值（得到 '$_a'）" ;; esac
        _k="${_a%%=*}"
        _k="${_k//_/-}"        # vLLM 把 _ 归一成 -，先归一化再比对
        case "$_k" in *.*) die "额外参数 '$_k' 含点号子键（会被重排覆盖），拒绝" 2 ;; esac
        local _d
        for _d in --port --model --served-model-name --tensor-parallel-size \
                  --gpu-memory-utilization --max-model-len --max-num-seqs \
                  --max-num-batched-tokens --compilation-config --kv-cache-dtype \
                  --limit-mm-per-prompt --speculative-config --kv-transfer-config \
                  --kv-offloading-size --kv-offloading-backend \
                  --prefix-caching-hash-algo --enable-prefix-caching \
                  --no-enable-prefix-caching --swa-bounded-replay \
                  --no-swa-bounded-replay \
                  --disable-custom-all-reduce --enable-custom-all-reduce \
                  --no-enable-custom-all-reduce; do
            if [ "$_d" = "$_k" ] || [[ "$_d" == "$_k"* ]]; then
                die "额外参数 '$_a' 命中受保护旗标 $_d（含缩写前缀），拒绝" 2
            fi
        done
        if [[ "$_k" =~ $deny ]]; then
            # ⭐ 2026-11-10 审计后加：调试用途（如 --load-format=dummy）需要显式旁路。
            #   必须显式 ALLOW_EXTRA_OVERRIDE=1，且**大声警告** —— 这条旁路只供诊断实例，
            #   生产口径仍然只能走白名单。
            if [ "${ALLOW_EXTRA_OVERRIDE:-0}" = "1" ]; then
                say "⚠️⚠️ ALLOW_EXTRA_OVERRIDE=1：放行受保护旗标 '$_a'（**非生产口径**，报告必须标注）"
            else
                die "额外参数 '$_k' 会覆盖本脚本保证的口径/安全参数，拒绝（调试旁路须显式 ALLOW_EXTRA_OVERRIDE=1）" 2
            fi
        fi
        # ⛔ 族级硬门：EXPANDABLE=1 时 VMM 与 IPC 句柄互斥，任何 custom all-reduce 变体都不得混进来
        if [ "$EXPANDABLE" = "1" ] && [[ "$_k" =~ custom.*all-?reduce ]]; then
            die "EXPANDABLE=1 时禁止任何 custom-all-reduce 额外参数（VMM 不能导出 IPC 句柄）：'$_a'" 2
        fi
    done
}

build_compilation_config() {
    local cc="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"mode\":\"VLLM_COMPILE\",\"cache_dir\":\"$CACHE_DIR\""
    [ -n "$COMPILE_RANGES" ] && cc="$cc,\"compile_ranges_endpoints\":[$COMPILE_RANGES]"
    printf '%s}' "$cc"
}

build_args() {
    ARGS=("$PYTHON" -m vllm.entrypoints.openai.api_server
        --model "$MODEL_DIR"
        --served-model-name "$SERVED_NAME"
        --load-format auto
        --max-model-len "$MAXLEN"
        --tensor-parallel-size "$TP"
        --max-num-seqs "$MAXSEQS"
        --max-num-batched-tokens "$MBT"
        --gpu-memory-utilization "$GPU_UTIL"
        --enable-auto-tool-choice
        --tool-call-parser deepseek_v41
        --reasoning-parser deepseek_v41
        --default-chat-template-kwargs '{"thinking":true}'
        --kv-cache-dtype "$KV_DTYPE"
        --enable-prompt-tokens-details
        --compilation-config "$(build_compilation_config)"
        --trust-remote-code
        --limit-mm-per-prompt "{\"image\":$IMAGE_MAX,\"video\":0}"
        --mm-processor-device cpu
        --mm-processor-cache-gb 0
        --kernel-config '{"enable_jit_warmup": false}'
        --port "$PORT")
    [ "$SPEC" = "1" ] && ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}')
    if [ "$PC" = "1" ]; then
        # 引擎版本层（取编译缓存目录名末尾 hash）⇒ 升级引擎自动换目录；
        # 模型名/dtype/TP/KV 组等指纹由上游 FileMapper 自动追加在下一层。
        local _ver _pcroot _pcjson _eng
        _eng="$( cd "$VLLM_DIR/.." && git rev-parse --short=10 HEAD 2>/dev/null )"
        [ -n "$_eng" ] || die "取不到在役 vLLM commit（$(dirname "$VLLM_DIR")）⇒ 无法保证 PC 目录随引擎换代，拒绝启动" 2
        _ver="${CACHE_DIR##*/}"   # 编译缓存目录名（含 <vLLM commit>-<源码 sha1>）
        _pcroot="$PC_ROOT/engine-${_ver}-vllm${_eng}"
        _pcjson='{"kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","blocks_per_chunk":'"$PC_BLOCKS"',"secondary_tiers":[{"type":"fs","root_dir":"'"$_pcroot"'","n_read_threads":16,"n_write_threads":16}]}}'
        ARGS+=(--kv-offloading-size "$PC_GB" --kv-transfer-config "$_pcjson")
        say "prefix-cache=on root=$_pcroot L1=${PC_GB}GiB blocks/chunk=$PC_BLOCKS"
    fi
    [ "$EXPANDABLE" = "1" ] && ARGS+=(--disable-custom-all-reduce)
    [ "$SWA_BOUNDED_REPLAY" = "0" ] && ARGS+=(--no-swa-bounded-replay)
    # ⭐【2026-11-08 · 诊断】关**内存前缀缓存**（默认开；仅诊断用，NOPC=1 才关）
    #   目的：验证"第 8 条长请求开始错"是否由前缀缓存块累积（每请求 131K token 留在池里）触发。
    [ "${NOPC:-0}" = "1" ] && { ARGS+=(--no-enable-prefix-caching); say "⚠️ NOPC=1：内存前缀缓存已关闭（诊断臂）"; }
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then ARGS+=("${EXTRA_ARGS[@]}"); fi
    return 0
}

probe() {   # http_code；000 = 连不上/无 curl（--noproxy 防代理冒充）
    command -v curl >/dev/null 2>&1 || { echo 000; return; }
    { curl -sS --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 3 \
         "http://127.0.0.1:$1/health" 2>/dev/null || echo 000; } | head -1
}

eff_compilation_config() {   # 生效（最后一个）--compilation-config
    tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null \
        | awk '$0=="--compilation-config"{getline v; last=v} END{print last}'
}

start() {
    validate
    mkdir -p "$RUN_DIR" || die "RUN_DIR 不可创建：$RUN_DIR" 4

    case "$(port_state "$PORT")" in
        free) ;;
        owned:*) die "端口 $PORT 已被 pid=$(port_state "$PORT" | sed 's/^owned://') 占用；先 stop 或换 PORT=" 3 ;;
        *) die "无法判定端口 $PORT 归属（ss 缺失/取不到 pid）⇒ 缺省拒绝启动" 3 ;;
    esac

    exec 9>"$LOCK_FILE" || die "锁文件不可写：$LOCK_FILE" 4
    flock -n 9 || die "TAG=$TAG 已有 start/stop 在进行（锁 $LOCK_FILE）" 3

    if [ -f "$PID_FILE" ]; then
        local old; old="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)"
        if verify_owns "$old"; then die "PID 文件存在且归属校验通过(pid=$old)；先 stop TAG=$TAG" 3; fi
        say "清理陈旧 PID 文件（pid=${old:-空} 已不在或不归属本实例）"
        rm -f "$PID_FILE"
    fi

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "1,2" ]; then
        die "本入口固定 GPU1,GPU2；检测到 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}（调试请用其它方式）" 2
    fi

    if [ -f "$MEM_GATE" ]; then
        local _need=900
        [ "$PC" = "1" ] && _need=$(( 900 + PC_GB ))
        bash "$MEM_GATE" --need-gib "$_need" --gpus 1,2 --tag "$TAG" \
            || die "内存/GPU 门未通过（need=${_need}GiB，含 PC_GB=${PC_GB}）" 3
    else
        [ "${ALLOW_NO_MEM_GATE:-0}" = "1" ] \
            || die "未找到内存门脚本 $MEM_GATE ⇒ 拒绝启动（要跳过须显式 ALLOW_NO_MEM_GATE=1 并记录）" 3
    fi

    local vpath
    vpath="$( cd "$REPO_DIR" && "$PYTHON" -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null )"
    [ -n "$vpath" ] || die "无法解析 vllm 包位置（python 导入失败）" 2
    [ "$(readlink -f "$vpath")" = "$(readlink -f "$VLLM_DIR")" ] \
        || die "vllm 解析到 $(readlink -f "$vpath" 2>/dev/null || echo "${vpath:-未知}")，与 VLLM_DIR=$VLLM_DIR 不一致 ⇒ 跨树/安装错误" 2

    build_args
    say "启动 TAG=$TAG PORT=$PORT TP=$TP util=$GPU_UTIL seqs=$MAXSEQS MBT=$MBT maxlen=$MAXLEN"
    say "线程=$THREADS PIN=$PIN/$PIN_SLOTS MAXQ=$PIN_MAXQ CUDA_MOE=$PIN_CUDA_MOE MARLIN=$MARLIN IDLE_EC=$IDLE_EC/$IDLE_EC_SECONDS SWA_BOUNDED_REPLAY=$SWA_BOUNDED_REPLAY SPEC=$SPEC image=$IMAGE_MAX"
    # ⛔ [2026-11-05 审计 B · M] 小 M 内核**只走 PIN 常驻槽位**（gpu_prefill.py:3017
    #    `slot is not None and _moe4_ok`）；`PIN=0` 时 slot 恒为 None ⇒ 内核永不生效。
    #    不提示的话会白烧一次启动，且 A/B 结论会被误读成"内核无效"。
    if [ "$PIN_CUDA_MOE" = "1" ] && [ "$PIN" != "1" ]; then
        say "⚠️ PIN_CUDA_MOE=1 但 PIN=0 ⇒ 小 M 内核不会生效（需 PIN 常驻槽）"
    fi
    # ⛔ [2026-11-05 审计 A · A] MARLIN=1 是**两段门**：先要 PIN=1 且 PIN_CUDA_MOE=1
    #    才进 moe4_smallm.moe_layer，再由 MARLIN 选内核。缺任一 ⇒ 融合内核静默不生效，
    #    而 status/check 仍会显示"已请求 MARLIN=1"，是**假口径**。这里必须显式告警。
    if [ "$MARLIN" = "1" ] && { [ "$PIN" != "1" ] || [ "$PIN_CUDA_MOE" != "1" ]; }; then
        say "⚠️ MARLIN=1 但 PIN=$PIN / PIN_CUDA_MOE=$PIN_CUDA_MOE ⇒ 融合内核**不会生效**（需 PIN=1 且 PIN_CUDA_MOE=1）"
    fi
    if [ -n "$COMPILE_RANGES" ]; then
        say "compile_ranges=[$COMPILE_RANGES] ⇒ 要求生效配置含 compile_ranges_endpoints"
    else
        say "compile_ranges=空（本次不传端点；这是非生产口径，报告中必须注明）"
    fi
    if [ "$EXPANDABLE" = "1" ]; then
        say "EXPANDABLE=1：expandable_segments:True + --disable-custom-all-reduce（custom AR 退回 NCCL，用于治碎片）"
    else
        say "EXPANDABLE=0：expandable_segments:False（历史设置；碎片靠其它手段）"
    fi
    [ "$SWA_BOUNDED_REPLAY" = "0" ] && say "⚠️ SWA_BOUNDED_REPLAY=0：这是基线臂（关闭 decoder-side SWA bounded replay），报告中必须标注"
    [ "$PC" = "0" ] && say "⚠️ PC=0（稳定口径）：不启用磁盘前缀缓存；如需持久缓存实验臂用 PC=1，须加 ALLOW_PC1=1"

    if [ "$PC" = "1" ]; then
        mkdir -p "$PC_ROOT" 2>/dev/null || die "PC_ROOT 无法创建：$PC_ROOT" 3
        [ -w "$PC_ROOT" ] || die "PC_ROOT 不可写：$PC_ROOT（PC=1 时引擎会在权重加载后才发现）" 3
        ( : > "$PC_ROOT/.probe.$$" ) 2>/dev/null || die "PC_ROOT 写入探针失败：$PC_ROOT" 3
        rm -f "$PC_ROOT/.probe.$$"
        local _shm_avail _pc_free
        _shm_avail="$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{print $4}')"
        case "${_shm_avail:-}" in
            ''|*[!0-9]*) die "读不到 /dev/shm 余量（df 失败/格式异常，值='${_shm_avail:-}'）⇒ 缺省拒绝" 3 ;;
        esac
        [ "$_shm_avail" -ge $(( PC_GB * 1024 + 8192 )) ] \
            || die "/dev/shm 余量 ${_shm_avail}MiB < 需要 $((PC_GB*1024+8192))MiB（PC_GB=$PC_GB）" 3
        _pc_free="$(df -Pm "$PC_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
        case "${_pc_free:-}" in ''|*[!0-9]*) die "读不到 $PC_ROOT 所在盘余量 ⇒ 缺省拒绝" 3 ;; esac
        [ "$_pc_free" -ge 524288 ] \
            || die "PC_ROOT 所在盘余量 ${_pc_free}MiB < 512GiB（fs 层无配额，会写满）" 3
        local _pc_bins
        _pc_bins="$(find "$PC_ROOT" -name '*.bin' 2>/dev/null | wc -l)"
        say "PC 预检 ✓ root=$PC_ROOT 可写、/dev/shm 余量 ${_shm_avail:-?}MiB、已有块文件 ${_pc_bins} 个"
        say "⚠️ fs 层无淘汰无配额（实测 ≈115 GB/日）⇒ 已挂运行时磁盘守护，阈值 ${PC_FREE_MIN_MIB}MiB"
    fi
    # ⛔ 2026-11-05 事故教训（台账 #396）：原来这里 `: > "$LOG_FILE"` **直接截断**旧日志，
    #    于是 11-04 20:06 那次 Xid31 崩溃后一重启，7579 行现场（含 188 个 [xtu-vram] 快照
    #    与完整栈）**全部丢失**，无法复盘。
    #    ⇒ 启动前先归档，保留最近 LOG_KEEP 份（默认 3）。
    if [ -s "$LOG_FILE" ]; then
        _arch="$LOG_FILE.$(date +%Y%m%d-%H%M%S)"
        if mv "$LOG_FILE" "$_arch" 2>/dev/null; then
            say "旧日志已归档：$_arch"
            # 只保留最近 LOG_KEEP 份，防无限增长
            ls -1t "$LOG_FILE".* 2>/dev/null | tail -n +"$(( LOG_KEEP + 1 ))" \
                | while read -r _old; do rm -f "$_old"; done
        else
            say "⚠️ 旧日志归档失败（继续，但会覆盖 $LOG_FILE）"
        fi
    fi
    : > "$LOG_FILE" || die "日志不可写：$LOG_FILE" 4
    (
        cd "$REPO_DIR" || exit 97
        export CUDA_VISIBLE_DEVICES=1,2
        export XIAOTU_PIN="$PIN" XIAOTU_PIN_MAX_PER_LAYER="$PIN_SLOTS" XIAOTU_PIN_MAXQ="$PIN_MAXQ"
        export XIAOTU_MOE_THREADS="$THREADS" OMP_NUM_THREADS=1
        export XIAOTU_PIN_CUDA_MOE="$PIN_CUDA_MOE" XIAOTU_MOE4_MARLIN="$MARLIN" XIAOTU_ENGRAM_LAST=1
        if [ "$IDLE_EC" = "1" ]; then
            export XIAOTU_IDLE_EMPTY_CACHE=1 XIAOTU_IDLE_EC_SECONDS="$IDLE_EC_SECONDS"
            # ⭐ 预热收尾即归还：实测 10.8 GiB 高水位 100% 来自形状预热，
            #    且"空闲 ≥60 s"窗口在持续流量下不存在 ⇒ 以**预热日志收尾**为触发源。
            export XIAOTU_RELEASE_AFTER_LOG="$RUN_DIR/ds41_${TAG}_warmup.log"
            # 插件里 marker **优先于** log：父环境若残留 XIAOTU_RELEASE_MARKER 且指向
            # 任一已存在文件，就会在启动结束瞬间触发、盖过日志触发源 ⇒ 显式清掉。
            unset XIAOTU_RELEASE_MARKER 2>/dev/null || true
        else
            unset XIAOTU_IDLE_EMPTY_CACHE 2>/dev/null || true
        fi
        # ⛔ 2026-11-04 用户：**GPU0 独立 MTP（DSpark remote）已废除**。
        #    这里显式关死，防父环境里残留 XIAOTU_DSPARK_REMOTE=1 把已废除的路径复活
        #    （它会 connect GPU0 socket，失败后回落 identity draft ⇒ 启动即崩）。
        export XIAOTU_DSPARK_REMOTE=0
        export GPU_UTIL="$GPU_UTIL" MAXSEQS="$MAXSEQS" MBT="$MBT" MAXLEN="$MAXLEN"
        export COMPILE_RANGES="$COMPILE_RANGES"
        # 碎片治理（2026-11-01 用户指示，默认关、显式开）：
        #   expandable_segments 用 CUDA VMM 分配，VMM 显存**不能导出 IPC 句柄**
        #   ⇒ custom_all_reduce 会在 KV profiling 阶段报 invalid argument、Worker 被杀
        #   （2026-10-27 两次实测）。所以 EXPANDABLE=1 时**必须同时**关掉它。
        export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF_EFFECTIVE"
        export EXPANDABLE="$EXPANDABLE"      # status 靠它读口径（此前从未 export ⇒ status 恒缺）
        export VLLM_LOGGING_LEVEL=INFO        # PC 生效断言依赖 INFO 级日志（tiering/spec.py:379）
        export VLLM_USE_SIMPLE_KV_OFFLOAD=0   # 防父环境漏入 ⇒ SimpleCPUOffloadConnector 会让 fs 层静默消失
        # ⛔ 生产 env 必须复刻：缺 VLLM_EXPERTS_LOAD_DEVICE=cpu 时缺省为 gpu ⇒ 476GB 专家上卡必 OOM
        export VLLM_EXPERTS_LOAD_DEVICE="${VLLM_EXPERTS_LOAD_DEVICE:-cpu}"
        export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
        export VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-120}"
        export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
        export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="${VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:-2560}"
        export XIAOTU_PIN_TABLE="${XIAOTU_PIN_TABLE:-/home/user/lvllm/tmp/pin_1017/expert_pin_pin700.bin}"
        export XIAOTU_RELEASE_SOURCE="${XIAOTU_RELEASE_SOURCE:-1}"
        export XIAOTU_MOE_NSLICE_SMALL="${XIAOTU_MOE_NSLICE_SMALL:-0}"
        export XIAOTU_MOE_ASYNC="${XIAOTU_MOE_ASYNC:-0}"
        export XIAOTU_MOE_SPIN_IDLE_US="${XIAOTU_MOE_SPIN_IDLE_US:-300}"
        export XIAOTU_GPF_STAGE_TILE_E="${XIAOTU_GPF_STAGE_TILE_E:-48}"
        export XIAOTU_MOE_MBT="$MBT"   # 必须跟随 --max-num-batched-tokens，否则 staging 预分配错配
        # ⭐ 审计 B4：TAG nonce（端口派生 PID 的归属判据）+ 口径变量（供 status 读出）
        export XTU_DS41_TAG="$TAG"
        export SWA_BOUNDED_REPLAY NO_PREFIX_CACHE MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS GPU_UTILIZATION
        export DS41_MAXLEN="$MAXLEN" DS41_TP="$TP" DS41_SPEC="$SPEC" DS41_KV_DTYPE="$KV_DTYPE"
        export XIAOTU_PIN_MAX_PER_LAYER="$PIN_SLOTS" XIAOTU_IMAGE_MAX="$IMAGE_MAX"
        # ⭐ 复审 N4：旁路必须留痕（事后只看 cmdline 是发现不了的）
        export DS41_EXTRA_ARGS="${EXTRA_ARGS[*]:-}" DS41_ALLOW_EXTRA_OVERRIDE="${ALLOW_EXTRA_OVERRIDE:-0}"
        exec 9>&-      # 别把锁 fd 继承给服务（否则服务在跑时 start 会误报"已有 start/stop"）
        exec setsid "${ARGS[@]}"
    ) >> "$LOG_FILE" 2>&1 < /dev/null &
    local lp=$! pid="" i
    for i in $(seq 1 10); do
        if pid_alive "$lp" && tr '\0' '\n' < "/proc/$lp/cmdline" 2>/dev/null | grep -qx -- "vllm.entrypoints.openai.api_server"; then
            pid="$lp"; break
        fi
        sleep 1
    done
    if [ -z "$pid" ]; then
        say "launcher pid=$lp 不是服务本体（setsid 已 fork），改按端口归属取 PID…"
        for i in $(seq 1 120); do
            pid="$(port_state "$PORT" | sed -n 's/^owned://p')"
            if [ -n "$pid" ]; then
                # ⭐⭐ 审计 B4：必须确认端口属主**就是本次启动**（按 TAG nonce 比对），
                #    否则会把别人的实例写进本 TAG 的 PID 文件 ⇒ `stop TAG=B` 停掉 A。
                if ! tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
                     | grep -qx "XTU_DS41_TAG=$TAG"; then
                    die "端口 $PORT 的属主 pid=$pid 不是本次启动（缺 XTU_DS41_TAG=$TAG）⇒ 拒绝采纳；可能同端口被别人占用" 3
                fi
                break
            fi
            if ! pid_alive "$lp" && [ "${_xtu_warned:-0}" = "0" ]; then
                _xtu_warned=1
                say "launcher pid=$lp 已退出（setsid fork 后属正常），继续按端口等待；有错请看 $LOG_FILE"
            fi
            sleep 5
        done
    fi
    if [ -z "$pid" ]; then
        say "⚠️ 已启动但未取得 PID；端口现状=$(port_state "$PORT")；日志 $LOG_FILE"
        die "启动 10 分钟内既认不出 launcher 也取不到端口 $PORT 归属（可能是孤儿，需人工确认）" 4
    fi
    printf '%s\n' "$pid" > "$PID_FILE" || die "PID 文件写入失败：$PID_FILE（已启动 pid=$pid）" 4
    printf '%s\n' "$PORT" > "$PORT_FILE" 2>/dev/null || say "⚠️ PORT 文件写入失败：$PORT_FILE"
    say "已启动 pid=$pid 日志=$LOG_FILE"

    local ok=0
    for i in $(seq 1 90); do
        if [ "$(probe "$PORT")" = "200" ]; then
            case "$(port_state "$PORT")" in
                "owned:$pid") ok=1; break ;;
                *) die "端口 $PORT 返回 200 但归属不是本次 pid=$pid（$(port_state "$PORT")）⇒ 拒绝判定就绪" 3 ;;
            esac
        fi
        pid_alive "$pid" || die "进程已退出（看 $LOG_FILE 末尾）" 4
        sleep 10
    done
    [ "$ok" = "1" ] || die "就绪超时（约 15 分钟）；进程仍在，看 $LOG_FILE" 5

    local kvs kvp cc
    kvs="$(grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$LOG_FILE" | tail -1)"
    kvp="$(grep -a "Available KV cache memory" "$LOG_FILE" | tail -1 | grep -oE "[0-9.]+ GiB")"
    say "就绪 ✓  ${kvp:-?} 可用 KV  ${kvs:-?}"

    local _pc_base=""
    [ "$PC" = "1" ] && _pc_base="$(find "$PC_ROOT" -name '*.bin' 2>/dev/null | wc -l)"
    cc="$(eff_compilation_config "$pid")"
    if [ -n "$COMPILE_RANGES" ]; then
        printf '%s' "$cc" | grep -q '"compile_ranges_endpoints"' \
            || die "生效的 --compilation-config 里没有 compile_ranges_endpoints ⇒ 拒绝宣称启动成功（$cc）" 4
        say "compile_ranges_endpoints 已在【生效】配置中 ✓"
    fi
    if [ "$PC" = "1" ]; then
        local _ev=0
        grep -aqE 'Created secondary tier #[0-9]+ \(fs\)' "$LOG_FILE" && _ev=$((_ev+1))
        local _mtr
        _mtr="$(curl -s --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/metrics" 2>/dev/null || true)"
        printf '%s' "$_mtr" | grep -qE "kv_offloading_size=\"(${PC_GB}|$(printf '%.1f' "$PC_GB"))\"" && _ev=$((_ev+1))
        [ "$_ev" -ge 1 ] \
            || die "PC=1 但日志与 /metrics 均无 fs 层证据 ⇒ 磁盘前缀缓存未生效（勿当成功，先查 $LOG_FILE）" 6
        say "磁盘前缀缓存已生效（证据 ${_ev}/2）✓"
    fi
    # B1：fs 层无淘汰/无配额（实测 45min 写 3.5GB ≈115GB/日）⇒ 必须有运行时磁盘守护；
    #     ⛔ 只告警不删除（删 KV 缓存只造成 miss，但属破坏性动作，需用户当次点头）
    if [ "$PC" = "1" ] && [ "${PC_DISK_GUARD:-1}" = "1" ]; then
        PC_ROOT="$PC_ROOT" PC_GUARD_LOG="$RUN_DIR/ds41_${TAG}.pcdisk.log" \
        PC_FREE_MIN_MIB="$PC_FREE_MIN_MIB" \
        setsid bash -c 'while :; do
            f=$(df -Pm "$PC_ROOT" 2>/dev/null | awk "NR==2{print \$4}")
            case "${f:-}" in ""|*[!0-9]*) f=0;; esac
            if [ "$f" -lt "$PC_FREE_MIN_MIB" ]; then
                printf "%s PC_ROOT=%s free=%sMiB < %sMiB ⇒ 磁盘前缀缓存有写满风险（停 PC 或清理）\n" \
                    "$(date -Is)" "$PC_ROOT" "$f" "$PC_FREE_MIN_MIB" >> "$PC_GUARD_LOG"
            fi
            sleep 600
        done' < /dev/null > /dev/null 2>&1 &
        local _gpid=$!
        sleep 1
        if pid_alive "$_gpid"; then
            printf '%s\n' "$_gpid" > "$RUN_DIR/ds41_${TAG}.pcdisk.pid"
        else
            say "⚠️ PC 磁盘守护启动失败（pid=$_gpid 已退出）⇒ 磁盘无运行时保护，请检查 setsid/df"
        fi
        say "PC 磁盘守护已起（每 600s，阈值 ${PC_FREE_MIN_MIB}MiB，日志 $RUN_DIR/ds41_${TAG}.pcdisk.log）"
        # ⛔ 审计 M2：日志只能证明"层建好"，不证明"写成" ⇒ 必须在预热后实测块文件增量
        PC_ROOT="$PC_ROOT" PC_BASE="$_pc_base" PC_LOG="$LOG_FILE" \
        PC_WAIT="${PC_WRITE_WAIT:-900}" \
        setsid bash -c 'base="${PC_BASE:-0}"; end=$(( SECONDS + PC_WAIT )); new=0
            while [ "$SECONDS" -lt "$end" ]; do
                new=$(find "$PC_ROOT" -name "*.bin" 2>/dev/null | wc -l)
                [ "$new" -gt "$base" ] && break
                sleep 15
            done
            if [ "$new" -gt "$base" ]; then
                echo "[ds41] [PC写入] ✓ 新增块文件 $((new-base)) 个（基线 $base → 现 $new）⇒ 磁盘层确实在写" >> "$PC_LOG"
            else
                echo "[ds41] [PC写入] ⚠️ $PC_WAIT 秒内无新增块文件（基线 $base）⇒ 只有层构造函数成功、无写入证据；查 O_DIRECT/权限/余量" >> "$PC_LOG"
            fi' < /dev/null > /dev/null 2>&1 &
        say "PC 写入证据守护已起（最长 ${PC_WRITE_WAIT:-900}s，结果写 $LOG_FILE 的 [PC写入] 行）"
    fi
    if [ "$SWA_BOUNDED_REPLAY" = "1" ] && grep -aq "SWA bounded replay needs model runner V2" "$LOG_FILE"; then
        die "SWA_BOUNDED_REPLAY=1 但日志显示未生效（needs model runner V2）" 4
    fi
    if [ "$EXPANDABLE" = "1" ] && tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
            | grep -qE 'enable-custom-all-reduce'; then
        die "EXPANDABLE=1 但生效命令行里出现 enable-custom-all-reduce ⇒ 绑定被破坏" 4
    fi
    { echo "[ds41] [口径] CUDA_MOE=$PIN_CUDA_MOE IDLE_EC=$IDLE_EC/$IDLE_EC_SECONDS EXPANDABLE=$EXPANDABLE PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF_EFFECTIVE:-unset} custom-AR=$([ "$EXPANDABLE" = "1" ] && echo off || echo on) PC=$PC PC_GB=$PC_GB PC_BLOCKS=$PC_BLOCKS PC_ROOT=$PC_ROOT GPU_UTIL=$GPU_UTIL PC_FILES_BASE=${_pc_base:-na}"; } >> "$LOG_FILE"
    say "显存: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

    # ⛔ 2026-11-04 审计 B 后重写：启动健康闸。
    #   事故：14:17 服务被错误断言打崩（一 prefill + 一 decode 并发），8 小时无人发现。
    #   本闸**独立于 WARMUP**（调试臂 WARMUP=0 同样需要它），两路**非对称**：
    #     B 路：短 prompt + 长解码（先起，占住 decode）
    #     A 路：短 prompt（~380 token）+ 只解 1 token（让"1 decode + 1 prefill"同处一个 batch step）
    #   失败时同时落到：smoke.log、服务日志 $LOG_FILE、标记文件 <TAG>.smoke.fail
    #   （`status` 会 tail 标记文件；只写 stderr 等于没人看得见）。
    (
        exec 9>&- >/dev/null 2>>"$RUN_DIR/ds41_${TAG}_smoke.err"
        sm() { printf '%s\n' "$*" >> "$RUN_DIR/ds41_${TAG}_smoke.log"; }
        # 清理必须在任何早退之前（审计 B · M-3：否则无 curl 时旧失败标记会一直挂着）
        rm -f "$RUN_DIR/ds41_${TAG}_smoke_"*.code "$RUN_DIR/ds41_${TAG}_smoke_"*.rc \
              "$RUN_DIR/ds41_${TAG}_smoke_"*.req "$RUN_DIR/ds41_${TAG}.smoke.fail" 2>/dev/null
        if ! command -v curl >/dev/null 2>&1; then
            sm "[$(date +%H:%M:%S)] ⚠️ 冒烟跳过：无 curl"
            exit 0
        fi
        # A 路的填充串：30 × 46 ≈ 1.4K 字符 ≈ 380 token。
        #   ⛔ 2026-11-04 实测教训：填到 6.9K token 时，与 decode 并发会让 GPU prefill 的
        #   slack 不足 ⇒ 落到 **CPU prefill**（`[xtu-pf-progress] device=cpu`，~8s/层），
        #   单次冒烟要十几分钟、还占着 2 个 slot ⇒ 冒烟本身成了负担。
        #   本冒烟要覆盖的是"**同一 batch 里既有 prefill 又有 decode**"（14:17 事故形态），
        #   380 token 已足够形成混合批；不追求把 prompt 堆到 GPU prefill 门槛之上。
        # ⛔ 前缀缓存默认开启 ⇒ 固定填充串从第二次 start 起会被**整段命中**，
        #    A 路的"长 prefill"直接消失，冒烟退化成两路 decode 却仍显示 ok=2（比原版更隐蔽的假绿）。
        #    ⇒ 每轮拼一个 nonce 进 content，保证每次都是冷 prefill。
        _rid="$(date +%s)_$$"
        _pad="run ${_rid} $(awk 'BEGIN{for(i=0;i<30;i++)printf "alpha beta gamma delta epsilon zeta eta theta "}')"
                printf '{"model":"%s","messages":[{"role":"user","content":"probe b"}],"max_tokens":256,"temperature":0}' \
            "$SERVED_NAME" > "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_1.req"
        printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":1,"temperature":0}' \
            "$SERVED_NAME" "$_pad" > "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_2.req"
        printf 'started %s rid=%s\n' "$(date '+%H:%M:%S')" "$_rid" > "$RUN_DIR/ds41_${TAG}.smoke.busy" 2>/dev/null
        _url="http://127.0.0.1:${PORT}/v1/chat/completions"
        _fire() {  # $1=序号 $2=请求体文件
            { curl -s -o /dev/null --noproxy '*' --max-time 600 -w '%{http_code}' \
                  -H 'Content-Type: application/json' --data-binary "@$2" "$_url" \
                  > "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_$1.code" 2>/dev/null
              echo $? > "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_$1.rc"; } &
        }
        _fire 1 "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_1.req"     # 先起 B（短 prompt，长解码）
        sleep 0.3
        _fire 2 "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_2.req"     # 再起 A（长 prefill）
        wait
        _ok=0; _bad=""
        for _n in 1 2; do
            _c="$(cat "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_${_n}.code" 2>/dev/null)"
            _r="$(cat "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_${_n}.rc" 2>/dev/null)"
            # 判据：http=200 **且** curl 退出码 0（只有 200 会漏「头发完、body 断流」= rc 18）
            if [ "$_c" = "200" ] && [ "$_r" = "0" ]; then
                _ok=$((_ok + 1))
            else
                _bad="${_bad} ${_n}(http=${_c:-none},rc=${_r:-none})"
            fi
        done
        _line="[$(date +%H:%M:%S)] 混合并发冒烟(1=短解/2=长prefill) ok=${_ok} bad=[${_bad# }]"
        sm "$_line"
        if [ -n "$_bad" ]; then
            printf '[ds41] ⛔ 启动冒烟失败:%s ⇒ 详见 %s\n' "$_line" "$RUN_DIR/ds41_${TAG}_smoke.log" >> "$LOG_FILE"
            printf '[ds41] ⛔ 启动冒烟失败:%s\n' "$_line" > "$RUN_DIR/ds41_${TAG}.smoke.fail"
        fi
        rm -f "$RUN_DIR/ds41_${TAG}_smoke_${_rid}_"*.req "$RUN_DIR/ds41_${TAG}.smoke.busy" 2>/dev/null
    ) &
    say "启动冒烟已在后台发起（混合并发；结果 $RUN_DIR/ds41_${TAG}_smoke.log，失败标记 $RUN_DIR/ds41_${TAG}.smoke.fail）"

    if [ "$WARMUP" = "1" ]; then
        if [ -f "$WARMUP_SCRIPT" ]; then
            say "形状预热开始（文本多长度 + 1/4 图 + 并发 2）…"
            ( exec 9>&-; cd "$(dirname "$WARMUP_SCRIPT")" && setsid "$PYTHON" -u "$WARMUP_SCRIPT" --port "$PORT" \
                >> "$RUN_DIR/ds41_${TAG}_warmup.log" 2>&1 < /dev/null & )
            say "预热在后台跑，日志 $RUN_DIR/ds41_${TAG}_warmup.log"
        else
            die "WARMUP=1 但 $WARMUP_SCRIPT 不存在 ⇒ 拒绝以「已预热」口径继续（改 WARMUP=0 或修路径）" 3
        fi
    fi
}

stop() {
    local pid="" ps
    [ -f "$PID_FILE" ] && pid="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)"
    ps="$(port_state "$PORT")"

    if ! verify_owns "$pid"; then
        case "$ps" in
            owned:*) die "PID 文件=${pid:-无} 与端口 $PORT 归属($ps)不符；拒绝按 PID 文件杀，请人工确认" 3 ;;
            unknown) die "无法判定端口 $PORT 归属（ss 不可用/取不到 pid）⇒ 缺省拒绝" 3 ;;
            *)
                # ⭐ 审计 B2：只有进程**确实不在**时才删 PID 文件（否则 B2 事故复发）
                if pid_alive "${pid:-}"; then
                    die "PID 文件里的 pid=$pid 仍存活，但端口 $PORT 归属是 '$ps' ⇒ 拒绝删除 PID 文件（TAG/PORT 可能不匹配，检查 PORT=）" 3
                fi
                say "TAG=$TAG 未在运行（归属校验通过）"; rm -f "$PID_FILE"; return 0 ;;
        esac
    fi

    say "停止 TAG=$TAG pid=$pid（先 TERM）"
    if [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ]; then
        kill -TERM "-$pid" 2>/dev/null
    else
        kill -TERM "$pid" 2>/dev/null
    fi
    local i
    for i in $(seq 1 30); do pid_alive "$pid" || break; sleep 2; done
    if pid_alive "$pid"; then
        say "未在 60s 内退出 ⇒ KILL pid=$pid"
        if [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ]; then
            kill -KILL "-$pid" 2>/dev/null
        else
            kill -KILL "$pid" 2>/dev/null
        fi
        sleep 3
    fi
    pid_alive "$pid" && die "仍在运行 pid=$pid（人工介入）" 4

    for i in $(seq 1 30); do
        case "$(port_state "$PORT")" in free) break ;; esac
        sleep 2
    done
    case "$(port_state "$PORT")" in
        free) ;;
        *) die "stop 未完成：端口 $PORT 仍被 $(port_state "$PORT") 持有" 4 ;;
    esac

    local _gpid
    if [ -f "$RUN_DIR/ds41_${TAG}.pcdisk.pid" ]; then
        _gpid="$(tr -dc '0-9' < "$RUN_DIR/ds41_${TAG}.pcdisk.pid" 2>/dev/null || true)"
        if pid_alive "$_gpid"; then kill -TERM "$_gpid" 2>/dev/null || true; fi
        rm -f "$RUN_DIR/ds41_${TAG}.pcdisk.pid"
        say "已回收 PC 磁盘守护 pid=${_gpid:-?}"
    fi
    rm -f "$PID_FILE"
    say "等待显存释放 20s…"
    sleep 20
    say "GPU 现状: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}

status() {
    local pid="" ps cc
    [ -f "$PID_FILE" ] && pid="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)"
    ps="$(port_state "$PORT")"
    say "TAG=$TAG PID文件=${pid:-无} 端口归属=${ps}"
    # ⛔ 启动冒烟状态（审计 B · H3/M-4）：**必须**在下面的"未运行"早退之前，
    #    否则服务已经死掉时永远看不到冒烟失败标记 —— 而那正是最需要它的场景。
    if [ -s "$RUN_DIR/ds41_${TAG}.smoke.fail" ]; then
        say "⛔ 冒烟失败: $(tr -d '\n' < "$RUN_DIR/ds41_${TAG}.smoke.fail" | cut -c1-160)"
    fi
    if [ -f "$RUN_DIR/ds41_${TAG}.smoke.busy" ]; then
        say "⏳ 启动冒烟进行中（占 2 槽；测速请等它结束）: $(tr -d '\n' < "$RUN_DIR/ds41_${TAG}.smoke.busy" | cut -c1-80)"
    fi
    if ! verify_owns "$pid"; then
        case "$ps" in
            owned:*) say "⚠️ PID 文件与端口归属不一致（疑似陈旧 PID 文件）；端口实际拥有者=${ps#owned:}" ;;
            *) say "未运行" ;;
        esac
        return 0
    fi
    say "health=$(probe "$PORT")"
    say "GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
    say "实际口径:"
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
        | grep -E '^(XIAOTU_MOE_THREADS|XIAOTU_PIN|XIAOTU_PIN_CUDA_MOE|XIAOTU_PIN_MAX_PER_LAYER|XIAOTU_IMAGE_MAX|XIAOTU_MOE4_MARLIN|XIAOTU_IDLE_EMPTY_CACHE|XIAOTU_IDLE_EC_SECONDS|XIAOTU_PIN_MAXQ|GPU_UTIL|MAXSEQS|COMPILE_RANGES|PYTORCH_CUDA_ALLOC_CONF|EXPANDABLE|SWA_BOUNDED_REPLAY|NO_PREFIX_CACHE|MAX_NUM_BATCHED_TOKENS|MAX_NUM_SEQS|GPU_UTILIZATION|XIAOTU_MOE_MBT|DS41_MAXLEN|DS41_TP|DS41_SPEC|DS41_KV_DTYPE)=' \
        | sed 's/^/    /'
    cc="$(eff_compilation_config "$pid")"
    say "    [生效] --compilation-config $cc"
    local _kvo _pcdir
    _kvo="$(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '--kv-offloading-size [0-9]+' | head -1)"
    if [ -n "$_kvo" ]; then
        _pcdir="$(find "$PC_ROOT" -maxdepth 1 -type d -name 'engine-*' 2>/dev/null | tail -1)"
        say "    PC: ${_kvo} 目录=${_pcdir:-?} 占用=$(du -sh "$PC_ROOT" 2>/dev/null | awk '{print $1}') 块文件=$(find "$PC_ROOT" -name '*.bin' 2>/dev/null | wc -l)"
    else
        say "    PC: 未开（无 --kv-offloading-size）"
    fi
}

case "$CMD" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    # ⛔ [2026-11-05 审计 A · B] 先 validate 再 stop：否则非法参数会**先把生产停掉**
    #    再在 start 里被拒，留下无服务可用的现场。validate 无副作用，前置安全。
    restart) validate; stop; start ;;
    *) usage; exit 2 ;;
esac
