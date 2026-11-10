#!/usr/bin/env bash
# assert_profile_local.sh —— 【在远端跑】断言"当前实例的进程环境"符合生产口径
#
# 与 check_running_profile.sh 的关系：那个是**本地工具**（ssh 出去查），供人/总闸用；
# 本脚本是**远端侧**版本，供 run_e1 / run_e2_verify 在开跑前**自证口径**（否则本轮的 prefill 对照无效）。
#
# 用法： bash scripts/assert_profile_local.sh [TAG]
# 退出码：0 = 符合；1 = 有偏差；2 = 服务没在跑
set -uo pipefail
TAG=${1:-prod}
RUN_DIR=${RUN_DIR:-/home/user/lvllm/run}
pidf="$RUN_DIR/ds41_${TAG}.pid"
[ -f "$pidf" ] || { echo "⛔ 没有 PID 文件 $pidf ⇒ 服务未运行"; exit 2; }
pid=$(tr -dc '0-9' < "$pidf")
[ -d "/proc/$pid" ] || { echo "⛔ PID=$pid 已退出 ⇒ 服务未运行"; exit 2; }

envf=$(mktemp); trap 'rm -f "$envf"' EXIT
tr '\0' '\n' < "/proc/$pid/environ" > "$envf" 2>/dev/null || { echo "⛔ 读不到 /proc/$pid/environ"; exit 2; }
get() { awk -F= -v k="$1" '$1==k{print $2}' "$envf" | tail -1; }

bad=0
chk() { local v; v=$(get "$1"); if [ "$v" = "$2" ]; then echo "  ✅ $1=$v"; else echo "  ❌ $1=${v:-<未设置>}（期望 $2）"; bad=1; fi; }
echo "[口径] 实例 TAG=$TAG pid=$pid"
chk XIAOTU_PIN_CUDA_MOE 1
chk XIAOTU_MOE4_MARLIN 1
chk XIAOTU_PIN 1
chk SWA_BOUNDED_REPLAY 1
chk NO_PREFIX_CACHE 0
nvcc=$(get NVCC); echo "  ℹ️ NVCC=${nvcc:-<未设置>}"
[ -n "$nvcc" ] || { echo "  ❌ NVCC 未设置 ⇒ marlin 会回退 smallm（口径不符）"; bad=1; }

[ "$bad" = 0 ] && { echo "[口径] ✅ 符合生产期望"; exit 0; }
echo "[口径] ❌ 有偏差 ⇒ **本轮作废**（先修口径再跑，别把数据记成结论）"
exit 1
