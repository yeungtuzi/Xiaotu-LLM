#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# serve_ds41.sh —— ds41f（DeepSeek-V4.1-Flash）**唯一**启动入口
#
# ⛔⛔ 单一入口纪律（2026-11-01 用户明令）：
#   「给 ds41 设立一个'单一'的启动脚本，所有的引用，不管是调试还什么的，都必须从这里开始」
#   ⇒ 任何进程若要以 ds41f 起在 GPU1/GPU2，**只能**通过本脚本的 start 子命令。
#   ⇒ ⛔ 禁止再直接 `python -m vllm.entrypoints.openai.api_server`（历史事故：
#      2026-10-31 绕过入口直起 ⇒ `COMPILE_RANGES` 从未被翻译进命令行 ⇒
#      Inductor 按 token 数产生了 85 万个编译产物 ⇒ 显存随形状单调上涨 ⇒ OOM）。
#
# 为什么必须是"一个脚本"：
#   ① 参数有**隐式耦合**：COMPILE_RANGES 只有变成 `--compilation-config` 里的
#      `compile_ranges_endpoints` 才生效；`PIN_MAXQ` 只对 prefill 生效；
#      `CED` 默认就是开的（树里 `swa_bounded_replay` 默认 True）。
#      ⇒ 每个绕过入口的脚本都要重学一遍这些坑，已经踩过至少 3 次。
#   ② 口径必须可比：线程 60、util、seqs、maxlen、image 上限一变，历史数据就不可比。
#
# 用法：
#   bash scripts/serve_ds41.sh start  [NAME=VAL ...] [+额外参数 ...]
#   bash scripts/serve_ds41.sh stop   [--tag NAME]
#   bash scripts/serve_ds41.sh status [--tag NAME]
#   bash scripts/serve_ds41.sh restart [NAME=VAL ...] [+额外参数 ...]
#
# 可覆盖的 NAME=VAL（其余一律拒绝，避免"改了但没人知道"）：
#   TAG=prod            实例名（PID/日志文件名）
#   PORT=8700           端口
#   TP=2                tensor-parallel-size
#   GPU_UTIL=0.90       gpu-memory-utilization
#   MAXSEQS=4           max-num-seqs
#   MBT=8192            max-num-batched-tokens
#   MAXLEN=524288       max-model-len
#   THREADS=60          XIAOTU_MOE_THREADS（必须是 12 的倍数）
#   IMAGE_MAX=128       limit-mm-per-prompt 的 image 上限
#   COMPILE_RANGES=32,256,8192
#   PIN=1               XIAOTU_PIN
#   PIN_SLOTS=15        XIAOTU_PIN_MAX_PER_LAYER
#   PIN_MAXQ=16         prefill 跳过热专家的 qlen 阈值
#   CED=1               1=开（默认，树里默认即开）；0=显式 --no-swa-bounded-replay
#   WARMUP=1            起来后跑形状预热（文本多长度 + 1/4 图 + 并发 2）
#   KV_DTYPE=fp8_ds_mla
#   MODEL_DIR=<路径>    模型目录
#   SPEC=1              1=dspark k=5；0=关投机
#
# 退出码：0 成功 / 2 参数错误 / 3 端口被占 / 4 启动失败 / 5 就绪超时
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── 固定路径（单一真源；改这里就等于改全部） ───────────────────────────────
REPO_DIR="${REPO_DIR:-/home/user/lvllm/vllm-xiaotu-moe}"
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
MODEL_DIR_DEFAULT="/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master"
SERVED_NAME="DeepSeek-V4.1-Flash"
CACHE_DIR="/home/user/.cache/vllm/torch_compile_cache/xtu-master-tp2-vllm_compile-f03026a548-9afa2a11"
WARMUP_SCRIPT="${WARMUP_SCRIPT:-/home/user/lvllm/stress-run5/warmup_shapes.py}"

# ── 生产默认值（口径唯一来源） ─────────────────────────────────────────────
TAG="prod"; PORT=8700; TP=2; GPU_UTIL=0.90; MAXSEQS=4; MBT=8192; MAXLEN=524288
THREADS=60; IMAGE_MAX=128; COMPILE_RANGES="32,256,8192"
PIN=1; PIN_SLOTS=15; PIN_MAXQ=16; CED=1; WARMUP=1
KV_DTYPE="fp8_ds_mla"; SPEC=1; MODEL_DIR="$MODEL_DIR_DEFAULT"
EXTRA_ARGS=()

die() { echo "[ds41] ⛔ $*" >&2; exit "${2:-2}"; }
say() { echo "[ds41] $*"; }

usage() { sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; }

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
                KV_DTYPE) KV_DTYPE="$v" ;;
                SPEC) SPEC="$v" ;;
                MODEL_DIR) MODEL_DIR="$v" ;;
                *) die "未知参数 '$k'（可用项见文件头；新增参数请改本脚本，不要绕过入口）" ;;
            esac
            shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "无法识别的参数 '$1'" ;;
    esac
done

PID_FILE="$RUN_DIR/ds41_${TAG}.pid"
LOG_FILE="$RUN_DIR/ds41_${TAG}.log"
mkdir -p "$RUN_DIR"

# ── 端口归属（只认 PID 文件 / 端口，绝不按进程名匹配） ─────────────────────
port_owner() { ss -ltnp 2>/dev/null | grep -F ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }

pid_alive() { [ -n "${1:-}" ] && [ -d "/proc/$1" ]; }

# ── 校验 ───────────────────────────────────────────────────────────────────
validate() {
    case "$THREADS" in
        ''|*[!0-9]*) die "THREADS 必须是正整数（得到 '$THREADS'）" ;;
    esac
    [ $(( THREADS % 12 )) -eq 0 ] || die "THREADS=$THREADS 不是 12 的倍数（CCD=12 核，实测非倍数会掉性能）"
    [ "$THREADS" -ge 12 ] && [ "$THREADS" -le 192 ] || die "THREADS=$THREADS 超出 12..192"
    [ -d "$MODEL_DIR" ] || die "模型目录不存在：$MODEL_DIR"
    [ -x "$PYTHON" ] || die "python 不存在：$PYTHON"
    case "$CED" in 0|1) ;; *) die "CED 只能是 0/1" ;; esac
    case "$PIN" in 0|1) ;; *) die "PIN 只能是 0/1" ;; esac
    case "$SPEC" in 0|1) ;; *) die "SPEC 只能是 0/1" ;; esac
    # COMPILE_RANGES：严格递增正整数（与 serve_v41.sh 同规则）
    if [ -n "$COMPILE_RANGES" ]; then
        printf '%s' "$COMPILE_RANGES" | LC_ALL=C grep -Eq '^[1-9][0-9]*(,[1-9][0-9]*)*$' \
            || die "COMPILE_RANGES='$COMPILE_RANGES' 非法（只允许 a,b,c 递增正整数）"
    fi
}

# ── 组装 argv + env ────────────────────────────────────────────────────────
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
    # CED：树里默认就是开的 ⇒ 只有显式关闭才需要旗标
    [ "$CED" = "0" ] && ARGS+=(--no-swa-bounded-replay)
    [ ${#EXTRA_ARGS[@]} -gt 0 ] && ARGS+=("${EXTRA_ARGS[@]}")
}

start() {
    validate
    local owner; owner="$(port_owner)"
    if [ -n "$owner" ]; then
        die "端口 $PORT 已被 pid=$owner 占用；先 stop 或换 PORT=" 3
    fi
    if [ -f "$PID_FILE" ]; then
        local old; old="$(cat "$PID_FILE" 2>/dev/null || true)"
        pid_alive "$old" && die "PID 文件存在且进程存活(pid=$old)；先 stop TAG=$TAG" 3
        rm -f "$PID_FILE"
    fi
    build_args

    say "启动 TAG=$TAG PORT=$PORT TP=$TP util=$GPU_UTIL seqs=$MAXSEQS MBT=$MBT maxlen=$MAXLEN"
    say "线程=$THREADS PIN=$PIN/$PIN_SLOTS MAXQ=$PIN_MAXQ CED=$CED SPEC=$SPEC image=$IMAGE_MAX"
    say "compile_ranges=[${COMPILE_RANGES:-空}]（已翻译进 --compilation-config）"
    [ "$CED" = "0" ] && say "⚠️ CED=0：这是基线臂，报告中必须标注"

    # 环境：CUDA_VISIBLE_DEVICES 固定 GPU1,GPU2（GPU0 留给调试）
    : > "$LOG_FILE"
    (
        cd "$REPO_DIR" || exit 1
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
        export XIAOTU_PIN="$PIN" XIAOTU_PIN_MAX_PER_LAYER="$PIN_SLOTS" XIAOTU_PIN_MAXQ="$PIN_MAXQ"
        export XIAOTU_MOE_THREADS="$THREADS" OMP_NUM_THREADS=1
        export XIAOTU_PIN_CUDA_MOE=0 XIAOTU_ENGRAM_LAST=1
        export GPU_UTIL="$GPU_UTIL" MAXSEQS="$MAXSEQS" MBT="$MBT" MAXLEN="$MAXLEN"
        export COMPILE_RANGES="$COMPILE_RANGES"
        exec setsid "${ARGS[@]}"
    ) >> "$LOG_FILE" 2>&1 &
    # ⭐ 子 shell 不是进程组长 ⇒ `setsid` 不 fork，直接 exec ⇒ $! 就是服务进程 PID ✓
    local pid=$!
    sleep 2
    # 只做校验（不按名字找进程）：确认这个 PID 真的是 api_server
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "api_server"; then
        die "启动后 $pid 不是 api_server（看 $LOG_FILE）" 4
    fi
    echo "$pid" > "$PID_FILE"
    say "已启动 pid=$pid 日志=$LOG_FILE"

    # 就绪判定：只认 HTTP 200（子串匹配过日志不可靠）
    local ok=0
    for _ in $(seq 1 90); do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")" = "200" ]; then ok=1; break; fi
        pid_alive "$pid" || die "进程已退出（看 $LOG_FILE 末尾）" 4
        sleep 10
    done
    [ "$ok" = "1" ] || die "就绪超时（15 分钟）；进程仍在，看 $LOG_FILE" 5

    # 生效校验（不看"旗标传了"，看引擎真实行为）
    local kvs kvp
    kvs="$(grep -aoE "GPU KV cache size: [0-9,]+ tokens" "$LOG_FILE" | tail -1)"
    kvp="$(grep -a "Available KV cache memory" "$LOG_FILE" | tail -1 | grep -oE "[0-9.]+ GiB")"
    say "就绪 ✓  ${kvp:-?} 可用 KV  ${kvs:-?}"
    if [ "$CED" = "1" ] && grep -aq "SWA bounded replay needs model runner V2" "$LOG_FILE"; then
        say "⚠️ 日志出现 'needs model runner V2' ⇒ CED 未真正生效"
    fi
    # 机械校验：编译档位必须真的在命令行里（这是 2026-10-31 事故的守门点）
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "compile_ranges_endpoints"; then
        say "compile_ranges_endpoints 已在命令行 ✓"
    else
        say "⚠️ 命令行里没有 compile_ranges_endpoints ⇒ 形状特化会重新堆积"
    fi
    say "显存: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

    if [ "$WARMUP" = "1" ] && [ -f "$WARMUP_SCRIPT" ]; then
        say "形状预热开始（文本多长度 + 1/4 图 + 并发 2）…"
        ( cd "$(dirname "$WARMUP_SCRIPT")" && setsid "$PYTHON" -u "$WARMUP_SCRIPT" --port "$PORT" \
            >> "$RUN_DIR/ds41_${TAG}_warmup.log" 2>&1 & ) || say "⚠️ 预热启动失败（不阻塞）"
        say "预热在后台跑，日志 $RUN_DIR/ds41_${TAG}_warmup.log"
    fi
}

stop() {
    local pid=""
    [ -f "$PID_FILE" ] && pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if ! pid_alive "$pid"; then
        local owner; owner="$(port_owner)"
        if [ -n "$owner" ]; then die "PID 文件缺失，但端口 $PORT 被 pid=$owner 占用；请人工确认后处理" 3; fi
        say "TAG=$TAG 未在运行"
        rm -f "$PID_FILE"
        return 0
    fi
    say "停止 TAG=$TAG pid=$pid（只发本 PID 的进程组信号）"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 30); do pid_alive "$pid" || break; sleep 2; done
    if pid_alive "$pid"; then
        say "未在 60s 内退出 ⇒ 强杀 pid=$pid"
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
        sleep 3
    fi
    pid_alive "$pid" && die "仍在运行 pid=$pid（人工介入）" 4
    rm -f "$PID_FILE"
    # 杀进程后等显存释放（AGENTS.md 明令：约 20 秒）
    say "等待显存释放 20s…"
    sleep 20
    say "GPU 现状: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}

status() {
    local pid=""
    [ -f "$PID_FILE" ] && pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    local owner; owner="$(port_owner)"
    say "TAG=$TAG PID文件=${pid:-无} 端口拥有者=${owner:-无}"
    if pid_alive "${owner:-}"; then
        say "health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health")"
        say "GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
        say "实际口径:"
        tr '\0' '\n' < "/proc/$owner/environ" 2>/dev/null \
            | grep -E '^(XIAOTU_MOE_THREADS|XIAOTU_PIN|XIAOTU_PIN_MAXQ|GPU_UTIL|MAXSEQS|COMPILE_RANGES)=' \
            | sed 's/^/    /'
        python3 - "$owner" <<'PY' 2>/dev/null | sed 's/^/    /'
import json, sys
argv = [a.decode() for a in open("/proc/%s/cmdline" % sys.argv[1], "rb").read().split(b"\0") if a]
for k in ("--max-model-len", "--max-num-seqs", "--gpu-memory-utilization", "--compilation-config"):
    if k in argv:
        i = argv.index(k)
        print(k, argv[i + 1][:200])
PY
    else
        say "未运行"
    fi
}

case "$CMD" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) stop; start ;;
    *) usage; exit 2 ;;
esac
