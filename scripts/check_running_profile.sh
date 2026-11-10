#!/usr/bin/env bash
# check_running_profile.sh —— 读出**正在运行的那个实例**的真实环境（不是命令行里的意图）
#
# 为什么需要（台账 #251）：E1 与 E2 之间要重启一次，prefill 对照就建立在"两次口径完全相同"之上。
# 而环境变量是**进程级**的：`NVCC=` 忘了内联、`MARLIN` 打错、`PC` 被默认值覆盖……
# 都会让两次运行的口径悄悄不同，而 `serve_ds41.sh status` 只显示"请求了什么"。
# 本脚本直接读 `/proc/<pid>/environ` —— 那是**内核记录的、进程真正拿到的**值。
#
# 【本地工具】内部 ssh；需要在远端有该实例的 PID 文件。
# 用法： bash scripts/check_running_profile.sh [TAG]
# 退出码：0 = 口径符合期望；1 = 有偏差；2 = 服务没在跑/读不到
set -uo pipefail
TAG=${1:-prod}
SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
REPO=${REPO:-/home/user/lvllm/vllm-xiaotu-moe}

SSH=(timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST")
"${SSH[@]}" true >/dev/null 2>&1 || { echo "⛔ 连不上远端 ⇒ 本脚本必须在【本地 harness】执行"; exit 2; }

out=$("${SSH[@]}" "cd $REPO || exit 9
pidf=\$(ls run/ds41_${TAG}.pid 2>/dev/null || true)
[ -n \"\$pidf\" ] || { echo 'NOPID'; exit 0; }
pid=\$(tr -dc '0-9' < \"\$pidf\")
[ -n \"\$pid\" ] || { echo 'NOPID'; exit 0; }
[ -d /proc/\$pid ] || { echo 'DEAD'; exit 0; }
echo PIDS \$pid
tr '\\0' '\\n' < /proc/\$pid/environ | grep -E '^(XIAOTU_PIN_CUDA_MOE|XIAOTU_MOE4_MARLIN|XIAOTU_PIN|XIAOTU_PIN_TABLE|NVCC|SWA_BOUNDED_REPLAY|NO_PREFIX_CACHE|PC|EXPANDABLE|XIAOTU_MOE_THREADS|MAX_NUM_SEQS|GPU_UTILIZATION|MAX_NUM_BATCHED_TOKENS)=' | sort" 2>&1)

case "$out" in
    NOPID) echo "⛔ 没有 PID 文件 run/ds41_${TAG}.pid ⇒ 服务未运行（或不是用 serve_ds41.sh 起的）"; exit 2 ;;
    DEAD)  echo "⛔ PID 文件存在但进程已退出 ⇒ 服务未运行"; exit 2 ;;
esac

echo "$out" | grep -E '^PIDS ' | sed 's/^/   /'
env_lines=$(echo "$out" | grep -vE '^PIDS ')
echo "$env_lines" | sed 's/^/   /'

get() { echo "$env_lines" | awk -F= -v k="$1" '$1==k{print $2}' | tail -1; }
bad=0
chk() { # 名称 期望
    local v; v=$(get "$1")
    if [ "$v" = "$2" ]; then printf '✅ %-24s = %s\n' "$1" "$v"
    else printf '❌ %-24s = %s（期望 %s）\n' "$1" "${v:-<未设置>}" "$2"; bad=1; fi
}
echo
chk XIAOTU_PIN_CUDA_MOE 1
chk XIAOTU_MOE4_MARLIN 1
chk XIAOTU_PIN 1
chk SWA_BOUNDED_REPLAY 1
chk NO_PREFIX_CACHE 0
echo "ℹ️ NVCC = $(get NVCC)  （MARLIN 需要它；未设置时 _build() 会退回 /usr/local/cuda/bin/nvcc ⇒ 回退 smallm）"
[ -n "$(get NVCC)" ] || bad=1
echo
[ "$bad" = 0 ] && { echo "RESULT: ✅ 运行口径符合生产期望"; exit 0; }
echo "RESULT: ❌ 运行口径有偏差（见上）"; exit 1
