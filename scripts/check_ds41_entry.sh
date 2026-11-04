#!/usr/bin/env bash
# check_ds41_entry.sh —— 机械检查：ds41f 服务是否**经由唯一入口** `serve_ds41.sh` 启动
#
# 来由（2026-11-01 用户明令）：
#   ① ds41f 只能走 `serve_ds41.sh` 启动；
#   ② **判活不能只看 8700 端口**（加载期还没 bind，会被误判成"未运行"）；
#   ③ 重复启动必须给出**明确提示**（不是静默失败）。
#
# 状态机（判据顺序：PID 文件 → 进程 → 端口 → health）：
#   RUNNING   进程活、端口归属它、health==200      ⇒ 跑全部口径检查
#   STARTING  进程活、但端口未 bind 或 health!=200 ⇒ 打印已运行秒数，提示勿重复启动
#   BYPASS    端口有主，但该 pid 不在任何 ds41_*.pid 里 ⇒ 违规（疑似绕过入口）
#   DUPLICATE 同时存在 >1 个 api_server 进程         ⇒ 违规
#   STOPPED   无存活进程、端口空闲                  ⇒ 未运行
#
# 退出码：0=RUNNING 且合规 / 1=违规（BYPASS 或 DUPLICATE 或口径不符）
#         2=未运行 / 3=启动中（合规，但未就绪）
#
# 用法：bash scripts/check_ds41_entry.sh [--port 8700] [--run-dir <dir>]
# 环境：ALLOW_PC1=1 显式允许非稳定口径（PC=1，实验臂用）

set -u
PORT=8700
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?}"; shift 2 ;;
        --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "未知参数: $1（ALLOW_PC1=1 是环境变量，不是参数）" >&2; exit 64 ;;
    esac
done

alive() {
    local p="${1:-}" st
    case "$p" in ''|*[!0-9]*) return 1 ;; esac
    [ -r "/proc/$p/stat" ] || return 1
    st="$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)"
    [ "$st" != "Z" ] && [ "$st" != "X" ]
}
is_api_server() {
    local p="${1:-}"
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "vllm.entrypoints.openai.api_server"
}
port_owner() {
    ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2
}
health() {
    local h
    h="$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)"
    case "${h:-}" in ''|*[!0-9]*) echo 000 ;; *) echo "$h" ;; esac
}

# ── 收集 PID 文件 ────────────────────────────────────────────────────────────
declared=()   # "pid|file"
for f in "$RUN_DIR"/ds41_*.pid; do
    [ -e "$f" ] || continue
    p="$(tr -dc '0-9' < "$f" 2>/dev/null || true)"
    [ -n "$p" ] && declared+=("$p|$f")
done

owner="$(port_owner)"
running_pid=""
for rec in ${declared[@]+"${declared[@]}"}; do
    p="${rec%%|*}"
    if alive "$p" && is_api_server "$p"; then running_pid="$p"; declared_file="${rec#*|}"; fi
done

api_count="$(ps -eo args 2>/dev/null | grep -c "[v]llm.entrypoints.openai.api_server")"

# ── DUPLICATE ───────────────────────────────────────────────────────────────
if [ "${api_count:-0}" -gt 1 ]; then
    echo "✗ 同时存在 ${api_count} 个 api_server 进程 ⇒ 重复启动" >&2
    ps -eo pid,etimes,args | grep "[a]pi_server" | awk '{printf "    pid=%s 已运行%s 秒\n", $1, $2}'
    echo "    处置：用 PID 文件确认哪一个受控（$RUN_DIR/ds41_*.pid），其余按"唯一入口 stop"或人工确认后停" >&2
    exit 1
fi

# ── BYPASS：端口有主但不受控 ────────────────────────────────────────────────
if [ -n "$owner" ] && [ "$owner" != "${running_pid:-}" ]; then
    echo "✗ 端口 $PORT 由 pid=$owner 持有，但它不在 $RUN_DIR/ds41_*.pid 中" >&2
    echo "    该实例**不是**由 serve_ds41.sh 启动 ⇒ 编译档位/CED/线程口径都不可信" >&2
    echo "    处置：停掉它，改用 bash scripts/serve_ds41.sh start" >&2
    exit 1
fi

# ── STOPPED ─────────────────────────────────────────────────────────────────
if [ -z "$running_pid" ]; then
    echo "未运行：无存活的 api_server 进程、端口 $PORT 空闲（退出码 2）"
    exit 2
fi

# ── STARTING：进程活但未就绪（**不按端口判定就绪**）────────────────────────
elapsed="$(ps -o etimes= -p "$running_pid" 2>/dev/null | tr -d ' ')"
h="$(health)"
if [ -z "$owner" ] || [ "$h" != "200" ]; then
    echo "启动中：pid=$running_pid 已运行 ${elapsed:-?} 秒，端口 ${owner:+已归属 pid=$owner}${owner:-尚未监听}，health=$h（退出码 3）"
    echo "  ⇒ 请勿重复启动（此状态下再执行 start 会被唯一入口拒绝并给出提示）"
    echo "  ⇒ 日志：$RUN_DIR/$(basename "${declared_file%.pid}").log"
    exit 3
fi

# ── RUNNING：跑全部口径检查 ─────────────────────────────────────────────────
echo "运行中：pid=$running_pid（$declared_file）已运行 ${elapsed:-?} 秒，health=200"
rc=0
cmdline="$(tr '\0' ' ' < "/proc/$running_pid/cmdline")"
environ="$(tr '\0' '\n' < "/proc/$running_pid/environ" 2>/dev/null)"

case "$cmdline" in
    *compile_ranges_endpoints*) echo "✓ 命令行含 compile_ranges_endpoints" ;;
    *) echo "✗ 缺 compile_ranges_endpoints ⇒ 形状特化会重新堆积（2026-10-31 事故同款）" >&2; rc=1 ;;
esac

case "$cmdline" in
    *--no-swa-bounded-replay*) echo "⚠️ CED 关闭（基线臂，报告须标注）" ;;
    *) echo "✓ CED 未关闭（树默认开启）" ;;
esac

threads="$(printf '%s\n' "$environ" | grep '^XIAOTU_MOE_THREADS=' | cut -d= -f2)"
if [ "${threads:-}" = "60" ]; then
    echo "✓ XIAOTU_MOE_THREADS=60"
else
    echo "⚠️ XIAOTU_MOE_THREADS=${threads:-未设}（非 60，性能数据须标注口径）"
fi

alloc="$(printf '%s\n' "$environ" | grep '^PYTORCH_CUDA_ALLOC_CONF=' | cut -d= -f2-)"
case "${alloc:-}" in
    *expandable_segments:True*)
        if printf '%s' "$cmdline" | grep -q -- '--disable-custom-all-reduce'; then
            echo "✓ expandable_segments:True + custom-AR 已关（VMM/IPC 绑定完整）"
        else
            echo "✗ expandable_segments:True 但缺 --disable-custom-all-reduce ⇒ KV profiling 必崩" >&2
            rc=1
        fi ;;
    *expandable_segments:False*) echo "  expandable_segments:False（碎片不回收，frag 会累积）" ;;
    *) echo "  PYTORCH_CUDA_ALLOC_CONF=${alloc:-未设}" ;;
esac

# ⛔ 2026-11-04 用户定：**稳定口径 = PC=0 + EXPANDABLE=1**（控显存乱飞优先，放弃磁盘前缀缓存）。
#    PC 与 expandable_segments 在 vLLM 里硬互斥（vllm/config/vllm.py:1214-1258）。
#    以前这里反过来（PC=1 才是"默认"），导致实例可以静默用 PC=1+EXPANDABLE=0 跑一整天
#    而没人发现（台账 #391）。现在改为**机械校验**，两侧都要对上。
kvoff="$(printf '%s' "$cmdline" | grep -oE -- '--kv-offloading-size [0-9]+' | head -1)"
expd="$(printf '%s' "$cmdline" | grep -oE 'expandable_segments:(True|False)' | head -1)"
if [ "${ALLOW_PC1:-0}" = "1" ]; then
    echo "  PC=1（已用 ALLOW_PC1=1 显式放行；报告须注明这是非稳定口径）"
else
    if [ -n "$kvoff" ]; then
        echo "✗ 发现 $kvoff（PC 开）⇒ 与稳定口径 PC=0 不符；实验臂请加 ALLOW_PC1=1" >&2
        rc=1
    else
        echo "✓ PC=0（稳定口径）"
    fi
fi
if [ "${ALLOW_PC1:-0}" = "1" ]; then
    :
elif [ "$expd" = "expandable_segments:True" ]; then
    echo "✓ expandable_segments:True（稳定口径）"
else
    echo "✗ expandable_segments 非 True（实测 '${expd:-未取到}'）⇒ 碎片治理未启用" >&2
    rc=1
fi

[ "$rc" = "0" ] && echo "合规：ds41f 由唯一入口启动且口径自洽" || echo "违规：口径不符" >&2
exit "$rc"
