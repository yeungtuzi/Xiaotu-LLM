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
#   TAG=prod PORT=8700 TP=2 GPU_UTIL=0.90 MAXSEQS=4 MBT=8192 MAXLEN=524288
#   THREADS=60 IMAGE_MAX=128 COMPILE_RANGES=32,256,8192
#   PIN=1 PIN_SLOTS=15 PIN_MAXQ=16 CED=1 WARMUP=1 KV_DTYPE=fp8_ds_mla
#   SPEC=1 MODEL_DIR=<路径>
#
# +额外参数：只允许**白名单之外**的 `--flag` / `--flag=值`；
#   会覆盖本脚本保证口径的旗标（--port/--compilation-config/--no-swa-bounded-replay…）一律拒绝。
#
# 退出码：0 成功 / 2 参数错误 / 3 前置条件不满足（端口、锁、内存门、归属冲突）
#         4 启动失败 / 5 就绪超时
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/user/lvllm/vllm-xiaotu-moe}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
MODEL_DIR_DEFAULT="/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master"
SERVED_NAME="DeepSeek-V4.1-Flash"
CACHE_DIR="/home/user/.cache/vllm/torch_compile_cache/xtu-master-tp2-vllm_compile-f03026a548-9afa2a11"
WARMUP_SCRIPT="${WARMUP_SCRIPT:-/home/user/lvllm/stress-run5/warmup_shapes.py}"
MEM_GATE="${MEM_GATE:-$REPO_DIR/scripts/prelaunch_mem_gate.sh}"
# 本机实际部署树（REPO_DIR/vllm 不存在；跨树校验按真实位置比对，不按名字猜）
VLLM_DIR="${VLLM_DIR:-/home/user/lvllm/process_data/ref/repos/vllm-mainline/vllm}"   # 注意：vpath 是包目录 dirname(vllm.__file__)

TAG="prod"; PORT=8700; TP=2; GPU_UTIL=0.90; MAXSEQS=4; MBT=8192; MAXLEN=524288
THREADS=60; IMAGE_MAX=128; COMPILE_RANGES="32,256,8192"
PIN=1; PIN_SLOTS=15; PIN_MAXQ=16; CED=1; WARMUP=1
PC=1; PC_GB=64; PC_BLOCKS=16; PC_ROOT="/home/user/.cache/prefix-cache"
KV_DTYPE="fp8_ds_mla"; SPEC=1; MODEL_DIR="$MODEL_DIR_DEFAULT"
EXTRA_ARGS=()

die() { printf '[ds41] ⛔ %s\n' "$1" >&2; exit "${2:-2}"; }
say() { printf '[ds41] %s\n' "$*"; }
usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

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
                PORT) PORT="$v" ;;
                TP) TP="$v" ;;
                GPU_UTIL) GPU_UTIL="$v" ;;
                MAXSEQS) MAXSEQS="$v" ;;
                MBT) MBT="$v" ;;
                MAXLEN) MAXLEN="$v" ;;
                THREADS) THREADS="$v" ;;
                IMAGE_MAX) IMAGE_MAX="$v" ;;
                COMPILE_RANGES) COMPILE_RANGES="$v" ;;
                PIN) PIN="$v" ;;
                PIN_SLOTS) PIN_SLOTS="$v" ;;
                PIN_MAXQ) PIN_MAXQ="$v" ;;
                CED) CED="$v" ;;
                WARMUP) WARMUP="$v" ;;
                PC) PC="$v" ;;
                PC_GB) PC_GB="$v" ;;
                PC_BLOCKS) PC_BLOCKS="$v" ;;
                PC_ROOT) PC_ROOT="$v" ;;
                KV_DTYPE) KV_DTYPE="$v" ;;
                SPEC) SPEC="$v" ;;
                MODEL_DIR) MODEL_DIR="$v" ;;
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
LOCK_FILE="$RUN_DIR/ds41_${TAG}.lock"

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

verify_owns() {   # pid 必须确实是本实例（端口归属 或 cmdline 的 --port）
    local p="${1:-}" ps
    pid_alive "$p" || return 1
    ps="$(port_state "$PORT")"
    [ "$ps" = "owned:$p" ] && return 0
    tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -qx -- "$PORT"
}

is_uint() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }

# ══ 校验（fail-closed） ═══════════════════════════════════════════════════════
validate() {
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
    awk -v u="$GPU_UTIL" 'BEGIN{exit !(u+0>0.05 && u+0<=1.0)}' || die "GPU_UTIL 必须在 (0.05,1.0]（得到 '$GPU_UTIL'）"
    case "$CED" in 0|1) ;; *) die "CED 只能是 0/1" ;; esac
    case "$PIN" in 0|1) ;; *) die "PIN 只能是 0/1" ;; esac
    case "$SPEC" in 0|1) ;; *) die "SPEC 只能是 0/1" ;; esac
    case "$WARMUP" in 0|1) ;; *) die "WARMUP 只能是 0/1（得到 '$WARMUP'）" ;; esac
    case "$PC" in 0|1) ;; *) die "PC 只能是 0/1（得到 '$PC'）" ;; esac
    if [ "$PC" = "0" ]; then
        :   # PC=0 时 PC_GB/PC_BLOCKS/PC_ROOT 一律不校验（不用的参数不该拦启动）
    else
    case "$PC_GB" in ''|0*|*[!0-9]*) die "PC_GB 必须是无前导零的正整数（得到 '$PC_GB'）" ;; esac
    [ "$PC_GB" -ge 8 ] && [ "$PC_GB" -le 512 ] || die "PC_GB 超出 8..512（/dev/shm 实为 756 GiB）"
    case "$PC_BLOCKS" in ''|0*|*[!0-9]*) die "PC_BLOCKS 必须是无前导零的正整数（得到 '$PC_BLOCKS'）" ;; esac
    [ "$PC_BLOCKS" -ge 1 ] && [ "$PC_BLOCKS" -le 256 ] || die "PC_BLOCKS 超出 1..256"
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
    local deny='^(--port|--model|--served-model-name|--tensor-parallel-size|--gpu-memory-utilization|--max-model-len|--max-num-seqs|--max-num-batched-tokens|--compilation-config|--kv-cache-dtype|--limit-mm-per-prompt|--speculative-config|--kv-transfer-config|--kv-offloading-size|--kv-offloading-backend|--swa-bounded-replay|--no-swa-bounded-replay)$'
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
                  --no-swa-bounded-replay; do
            if [ "$_d" = "$_k" ] || [[ "$_d" == "$_k"* ]]; then
                die "额外参数 '$_a' 命中受保护旗标 $_d（含缩写前缀），拒绝" 2
            fi
        done
        if [[ "$_k" =~ $deny ]]; then
            die "额外参数 '$_k' 会覆盖本脚本保证的口径/安全参数，拒绝（要改请改本脚本白名单）" 2
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
        local _ver _pcroot _pcjson
        _ver="${CACHE_DIR##*/}"   # 保留 <vLLM commit>-<源码 sha1> 两段（尾段只是我们源码指纹）
        _pcroot="$PC_ROOT/engine-$_ver"
        _pcjson='{"kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","blocks_per_chunk":'"$PC_BLOCKS"',"secondary_tiers":[{"type":"fs","root_dir":"'"$_pcroot"'","n_read_threads":16,"n_write_threads":16}]}}'
        ARGS+=(--kv-offloading-size "$PC_GB" --kv-transfer-config "$_pcjson")
        say "prefix-cache=on root=$_pcroot L1=${PC_GB}GiB blocks/chunk=$PC_BLOCKS"
    fi
    [ "$CED" = "0" ] && ARGS+=(--no-swa-bounded-replay)
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then ARGS+=("${EXTRA_ARGS[@]}"); fi
    return 0
}

probe() {   # http_code；000 = 连不上/无 curl（--noproxy 防代理冒充）
    command -v curl >/dev/null 2>&1 || { echo 000; return; }
    curl -sS --noproxy '*' -o /dev/null -w '%{http_code}' --max-time 3 \
         "http://127.0.0.1:$1/health" 2>/dev/null || echo 000
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
    say "线程=$THREADS PIN=$PIN/$PIN_SLOTS MAXQ=$PIN_MAXQ CED=$CED SPEC=$SPEC image=$IMAGE_MAX"
    if [ -n "$COMPILE_RANGES" ]; then
        say "compile_ranges=[$COMPILE_RANGES] ⇒ 要求生效配置含 compile_ranges_endpoints"
    else
        say "compile_ranges=空（本次不传端点；这是非生产口径，报告中必须注明）"
    fi
    [ "$CED" = "0" ] && say "⚠️ CED=0：这是基线臂，报告中必须标注"

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
        say "PC 预检 ✓ root=$PC_ROOT 可写、/dev/shm 余量 ${_shm_avail:-?}MiB"
    fi
    : > "$LOG_FILE" || die "日志不可写：$LOG_FILE" 4
    (
        cd "$REPO_DIR" || exit 97
        export CUDA_VISIBLE_DEVICES=1,2
        export XIAOTU_PIN="$PIN" XIAOTU_PIN_MAX_PER_LAYER="$PIN_SLOTS" XIAOTU_PIN_MAXQ="$PIN_MAXQ"
        export XIAOTU_MOE_THREADS="$THREADS" OMP_NUM_THREADS=1
        export XIAOTU_PIN_CUDA_MOE=0 XIAOTU_ENGRAM_LAST=1
        export GPU_UTIL="$GPU_UTIL" MAXSEQS="$MAXSEQS" MBT="$MBT" MAXLEN="$MAXLEN"
        export COMPILE_RANGES="$COMPILE_RANGES"
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
        export XIAOTU_MOE_MBT="${XIAOTU_MOE_MBT:-8192}"
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
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
            [ -n "$pid" ] && break
            sleep 5
        done
    fi
    if [ -z "$pid" ]; then
        say "⚠️ 已启动但未取得 PID；端口现状=$(port_state "$PORT")；日志 $LOG_FILE"
        die "启动 10 分钟内既认不出 launcher 也取不到端口 $PORT 归属（可能是孤儿，需人工确认）" 4
    fi
    printf '%s\n' "$pid" > "$PID_FILE" || die "PID 文件写入失败：$PID_FILE（已启动 pid=$pid）" 4
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

    cc="$(eff_compilation_config "$pid")"
    if [ -n "$COMPILE_RANGES" ]; then
        printf '%s' "$cc" | grep -q '"compile_ranges_endpoints"' \
            || die "生效的 --compilation-config 里没有 compile_ranges_endpoints ⇒ 拒绝宣称启动成功（$cc）" 4
        say "compile_ranges_endpoints 已在【生效】配置中 ✓"
    fi
    if [ "$PC" = "1" ]; then
        local _ev=0
        grep -aqE 'Created secondary tier #[0-9]+ \(fs\)' "$LOG_FILE" && _ev=$((_ev+1))
        curl -s --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
            | grep -q "kv_offloading_size=\"$PC_GB\"" && _ev=$((_ev+1))
        ls -d "$PC_ROOT"/engine-*/*_* >/dev/null 2>&1 && _ev=$((_ev+1))
        if [ "$_ev" -eq 0 ]; then
            say "⚠️ PC=1 但三条证据全无（日志/metrics/目录）⇒ 磁盘缓存可能未生效，先查 $LOG_FILE"
        else
            say "磁盘前缀缓存证据 ${_ev}/3 ✓"
        fi
    fi
    if [ "$CED" = "1" ] && grep -aq "SWA bounded replay needs model runner V2" "$LOG_FILE"; then
        die "CED=1 但日志显示 CED 未生效（needs model runner V2）" 4
    fi
    say "显存: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

    if [ "$WARMUP" = "1" ]; then
        if [ -f "$WARMUP_SCRIPT" ]; then
            say "形状预热开始（文本多长度 + 1/4 图 + 并发 2）…"
            ( cd "$(dirname "$WARMUP_SCRIPT")" && setsid "$PYTHON" -u "$WARMUP_SCRIPT" --port "$PORT" \
                >> "$RUN_DIR/ds41_${TAG}_warmup.log" 2>&1 < /dev/null & )
            say "预热在后台跑，日志 $RUN_DIR/ds41_${TAG}_warmup.log"
        else
            say "⚠️ WARMUP=1 但 $WARMUP_SCRIPT 不存在 ⇒ 预热未执行（口径缺失，需记录）"
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
            *) say "TAG=$TAG 未在运行（归属校验通过）"; rm -f "$PID_FILE"; return 0 ;;
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
        | grep -E '^(XIAOTU_MOE_THREADS|XIAOTU_PIN|XIAOTU_PIN_MAXQ|GPU_UTIL|MAXSEQS|COMPILE_RANGES)=' \
        | sed 's/^/    /'
    cc="$(eff_compilation_config "$pid")"
    say "    [生效] --compilation-config $cc"
}

case "$CMD" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) stop; start ;;
    *) usage; exit 2 ;;
esac
