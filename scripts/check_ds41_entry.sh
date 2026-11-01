#!/usr/bin/env bash
# check_ds41_entry.sh —— 机械检查：ds41f 服务是否**经由唯一入口** `serve_ds41.sh` 启动
#
# 来由（2026-11-01 用户明令）：ds41f 必须只有一个启动入口，其他 agent / 调试脚本
#   一律从这里开始。历史事故：绕过入口直起 ⇒ `COMPILE_RANGES` 从未被翻译进
#   `--compilation-config` ⇒ Inductor 按 token 数产生 85 万个编译产物 ⇒ 显存单调上涨 ⇒ OOM。
#
# 检查项（全部只读）：
#   ① 端口 8700 的拥有者 PID，是否等于 `run/ds41_*.pid` 里登记的 PID；
#   ② 命令行是否含 `--compilation-config` 且其中有 `compile_ranges_endpoints`；
#   ③ 命令行是否含 `--no-swa-bounded-replay`（= CED 被关，视为基线臂，仅告警）；
#   ④ 环境里 `XIAOTU_MOE_THREADS` 是否为 60（不是 60 只告警，需在报告中标注口径）。
#
# 退出码：0=合规；1=违规（存在 vLLM 不知道/未受控的启动）；2=未找到服务
#
# 用法：bash scripts/check_ds41_entry.sh [--port 8700] [--run-dir <dir>]

set -u
PORT=8700
RUN_DIR="${RUN_DIR:-/home/user/lvllm/run}"

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?}"; shift 2 ;;
        --run-dir) RUN_DIR="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 64 ;;
    esac
done

owner="$(ss -ltnp 2>/dev/null | grep -F ":$PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
if [ -z "$owner" ]; then
    echo "未找到监听 $PORT 的进程 ⇒ 服务未运行（退出码 2）"
    exit 2
fi

echo "端口 $PORT 拥有者 pid=$owner"
rc=0

# ① 是否与 PID 文件登记一致
matched=""
for f in "$RUN_DIR"/ds41_*.pid; do
    [ -e "$f" ] || continue
    if [ "$(cat "$f" 2>/dev/null)" = "$owner" ]; then matched="$f"; fi
done
if [ -n "$matched" ]; then
    echo "✓ PID 与 $matched 一致（由 serve_ds41.sh 启动）"
else
    echo "✗ 该 PID 不在 $RUN_DIR/ds41_*.pid 中 ⇒ 疑似绕过唯一入口直起" >&2
    rc=1
fi

# ② 编译档位必须进命令行
cmdline="$(tr '\0' ' ' < "/proc/$owner/cmdline" 2>/dev/null)"
case "$cmdline" in
    *compile_ranges_endpoints*) echo "✓ 命令行含 compile_ranges_endpoints" ;;
    *) echo "✗ 命令行缺 compile_ranges_endpoints ⇒ 形状特化会重新堆积（2026-10-31 事故同款）" >&2; rc=1 ;;
esac

# ③ CED（默认开；显式关闭只告警）
case "$cmdline" in
    *--no-swa-bounded-replay*) echo "⚠️ 命令行含 --no-swa-bounded-replay ⇒ CED 关闭（基线臂，报告须标注）" ;;
    *) echo "✓ CED 未关闭（树默认开启）" ;;
esac

# ④ 线程口径
threads="$(tr '\0' '\n' < "/proc/$owner/environ" 2>/dev/null | grep '^XIAOTU_MOE_THREADS=' | cut -d= -f2)"
if [ "${threads:-}" = "60" ]; then
    echo "✓ XIAOTU_MOE_THREADS=60"
else
    echo "⚠️ XIAOTU_MOE_THREADS=${threads:-未设}（非 60，性能数据须标注口径）"
fi

# ⑤ 碎片/分配器口径（2026-11-01 新增轴：EXPANDABLE 与 custom AR 必须成对出现）
alloc="$(tr '\0' '\n' < "/proc/$owner/environ" 2>/dev/null | grep '^PYTORCH_CUDA_ALLOC_CONF=' | cut -d= -f2-)"
exp_flag="$(printf '%s' "$cmdline" | grep -o -- '--disable-custom-all-reduce' | head -1)"
case "${alloc:-}" in
    *expandable_segments:True*)
        if [ -n "$exp_flag" ]; then
            echo "✓ expandable_segments:True + custom-AR 已关（VMM/IPC 绑定完整）"
        else
            echo "✗ expandable_segments:True 但命令行缺 --disable-custom-all-reduce ⇒ 必然在 KV profiling 崩（2026-10-27 事故）" >&2
            rc=1
        fi ;;
    *expandable_segments:False*) echo "  expandable_segments:False（碎片不回收，frag 会累积）" ;;
    *) echo "  PYTORCH_CUDA_ALLOC_CONF=${alloc:-未设}" ;;
esac

# ⑥ 磁盘前缀缓存口径（PC=1 时命令行必须带 L1 大小与 transfer-config）
kvoff="$(printf '%s' "$cmdline" | grep -oE -- '--kv-offloading-size [0-9]+' | head -1)"
if [ -n "$kvoff" ]; then
    if printf '%s' "$cmdline" | grep -q -- 'TieringOffloadingSpec'; then
        echo "✓ 前缀缓存已接线（$kvoff，tiering offload）"
    else
        echo "✗ 有 --kv-offloading-size 但缺 TieringOffloadingSpec ⇒ 磁盘层不会生效" >&2
        rc=1
    fi
else
    echo "  前缀缓存未开（无 --kv-offloading-size）"
fi


[ "$rc" = "0" ] && echo "合规：ds41f 由唯一入口启动" || echo "违规：存在不受控启动" >&2
exit "$rc"
