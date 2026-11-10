#!/usr/bin/env bash
# check_marlin_ready.sh —— MXFP4 融合内核（MARLIN=1）**是否真的能生效**（本地工具，内部 ssh）
#
# 为什么需要（台账 #246）：`vllm_xiaotu_moe/moe4_marlin.py` 的内核是**运行时用 nvcc 编译**的
# （`-arch=sm_80`，产物缓存在 ~/.cache/xtu_moe4_marlin/）。机器上**没有 nvcc** 时它 fail-closed：
# 打印 "[xtu-marlin] 编译失败，回退原路径" 并回退 `moe4_smallm` —— 服务照常起，
# 但 `MARLIN=1` 就是**假口径**（跑的不是融合内核）。
#
# 本脚本只做一件事：在远端按生产开关跑一次 `available()`（会编译两个 .cu，约 1–2 分钟），
# 用 GPU0、不需要权重、不需要起服务（符合"局部验证纪律"）。
#
# 用法： bash scripts/check_marlin_ready.sh [--no-build]
#   --no-build  只查"缓存里有没有现成 .so"，不触发编译（秒级）
# 退出码：0 = 可用；1 = 不可用（已回退）；2 = 环境问题
set -uo pipefail
SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
REPO=${REPO:-/home/user/lvllm/vllm-xiaotu-moe}
PY=${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}
BUILD=1
[ "${1:-}" = "--no-build" ] && BUILD=0

SSH=(timeout 300 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST")
"${SSH[@]}" true >/dev/null 2>&1 || { echo "⛔ 连不上远端（$HOST）⇒ 本脚本必须在【本地 harness】执行"; exit 2; }

echo "== 1) nvcc 是否存在"
nvcc=$("${SSH[@]}" 'ls /usr/local/cuda/bin/nvcc 2>/dev/null; command -v nvcc 2>/dev/null; echo "NVCC_ENV=${NVCC:-<unset>}"' | tail -3)
echo "$nvcc" | sed 's/^/   /'
if ! echo "$nvcc" | grep -q "/nvcc\|NVCC_ENV=/"; then
    echo "   ⚠️ 没找到 nvcc（NVCC 环境变量也没设）⇒ 融合内核无法编译"
fi

echo "== 2) 编译缓存"
"${SSH[@]}" 'ls -l ~/.cache/xtu_moe4_marlin/*.so 2>/dev/null | tail -3 || echo "   （无缓存）"' | sed 's/^/   /'

if [ "$BUILD" = 0 ]; then
    echo; echo "（--no-build：到此为止）"; exit 0
fi

echo "== 3) 实跑 available()（会编译，约 1–2 分钟；用 GPU0，不加载权重）"
out=$("${SSH[@]}" "cd $REPO && CUDA_VISIBLE_DEVICES=0 XIAOTU_MOE4_MARLIN=1 $PY - <<'PY' 2>&1
import vllm_xiaotu_moe.moe4_marlin as m
print('ENABLED', m.enabled())
print('AVAILABLE', m.available())
print('READY', m.ready())
PY")
rc=$?
echo "$out" | sed 's/^/   /'
echo
if echo "$out" | grep -q "AVAILABLE True"; then
    echo "RESULT: ✅ MARLIN 融合内核可用（编译成功并已加载）"
    exit 0
fi
echo "RESULT: ❌ MARLIN 不可用 ⇒ MARLIN=1 会是假口径（服务会回退 moe4_smallm）"
echo "        先装 nvcc（见 RESUME v2 第 2.5 步），再重跑本脚本"
exit 1
