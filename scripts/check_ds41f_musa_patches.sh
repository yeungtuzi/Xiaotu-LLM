#!/bin/bash
# scripts/check_ds41f_musa_patches.sh —— 核对 ds41f/MUSA 的四个补丁是否在树上
#
# 为什么需要: MUSA 树 third_party/vllm **不是 git 仓库**(208 个 MUSA 补丁后的纯副本)
#   ⇒ 没有 git diff 可查; 而 `patch -p1 --forward` 在已打过时会静默跳过(返回 0)
#   ⇒ "patch 返回 0" **不能** 证明补丁在树上。
#   ⇒ 唯一机械判据 = 每个补丁留在代码里的唯一**标记字符串**是否还在。
#
# 用法:
#   bash scripts/check_ds41f_musa_patches.sh [容器名]
# 退出码: 0 = 四个全在; 1 = 有缺失(会列出); 2 = 容器/路径不可达
set -u
CONTAINER="${1:-qfn-tp1}"
V="${VLLM_TREE:-/home/mccuseradmin/s5000/vllm-musa-032/third_party/vllm}/vllm"

# 自动适配运行位置:
#   * 在宿主上跑  ⇒ 用 docker exec 进容器
#   * 已在容器内跑(没有 docker 或容器名 == 本机 hostname) ⇒ 直接读文件
_inside=0
if ! command -v docker >/dev/null 2>&1; then _inside=1; fi
if [ "$(hostname 2>/dev/null)" = "$CONTAINER" ]; then _inside=1; fi

_incontainer() {  # 在"目标文件系统"上执行一段 shell
    if [ "$_inside" -eq 1 ]; then
        bash -c "$1"
    else
        docker exec "$CONTAINER" bash -c "$1"
    fi
}

probe() {  # $1=标签 $2=标记 $3=相对路径
    local n
    n=$(_incontainer "grep -c '$2' '$V/$3' 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r' | tail -1)
    [[ "$n" =~ ^[0-9]+$ ]] || n=0
    if [ "$n" -ge 1 ]; then
        printf '  ✓ %-6s %-24s %s\n' "$1" "$2" "$3"
        return 0
    fi
    printf '  ✗ %-6s %-24s %s  (标记缺失!)\n' "$1" "$2" "$3"
    return 1
}

if ! _incontainer "test -d '$V'" 2>/dev/null; then
    echo "✗ 容器 '$CONTAINER' 里找不到 vllm 树: $V" >&2
    echo "  提示: 可用 VLLM_TREE=<含 vllm/ 的目录> 覆盖" >&2
    exit 2
fi

echo "== ds41f/MUSA 补丁完整性核对 (container=$CONTAINER) =="
rc=0
probe 0024 xtu-musa-v41          model_executor/layers/quantization/__init__.py      || rc=1
probe 0023 xtu-musa-oot          model_executor/kernels/linear/__init__.py           || rc=1
probe 0025 xtu-musa-bmm          model_executor/kernels/linear/__init__.py           || rc=1
probe 0026 xtu-musa-rope-quantize models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py || rc=1

if [ "$rc" -eq 0 ]; then
    echo "== 全部在树上 ✓ =="
else
    echo "== ⛔ 有补丁缺失, 当前跑的**不是**修好的代码 ==" >&2
fi
exit "$rc"
