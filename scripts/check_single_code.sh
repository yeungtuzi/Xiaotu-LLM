#!/usr/bin/env bash
# 单一代码线守卫：一份源码同时支持 CUDA / MUSA，平台差异只允许走 GPU_DEVICE 门控。
#
# 设计见 dev-docs/MERGE_MUSA_TO_MAINLINE_PLAN.md §9。
# 判据（全部满足才算通过）：
#   1. GPU_DEVICE 只允许在 musa_correctness.py 读取（单点判定）
#   2. 不允许平台专用实现副本（*_musa*.py，musa_correctness.py 除外）
#   3. 不允许顶层 import torch_musa / torch.musa
#   4. 不允许平台专用依赖文件
#   5. 不允许残留合并冲突标记
#   6. vllm 可导入时，默认路径（GPU_DEVICE 未设）必须能 import 本包
#
# 用法：bash scripts/check_single_code.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

fail=0
ok()   { printf '  ✅ %s\n' "$1"; }
bad()  { printf '  ❌ %s\n' "$1"; fail=$((fail+1)); }

echo "[single-code] 1) GPU_DEVICE 单点读取"
hits=$(grep -rn 'environ.get("GPU_DEVICE"\|environ\["GPU_DEVICE"\]' vllm_xiaotu_moe/ 2>/dev/null \
       | grep -v 'musa_correctness.py' | grep -v '__pycache__' || true)
if [ -z "$hits" ]; then ok "只有 musa_correctness.py 读取 GPU_DEVICE"
else bad "GPU_DEVICE 被多处读取:"; printf '%s\n' "$hits"; fi

echo "[single-code] 2) 无平台专用实现副本"
copies=$(find vllm_xiaotu_moe -maxdepth 1 -name '*_musa*.py' ! -name 'musa_correctness.py' 2>/dev/null)
if [ -z "$copies" ]; then ok "无 *_musa*.py 实现副本"
else bad "出现平台专用实现文件:"; printf '%s\n' "$copies"; fi

echo "[single-code] 3) 无顶层 torch_musa import"
musa_imports=$(grep -rn '^import torch_musa\|^from torch_musa\|^import torch\.musa\|^from torch\.musa' \
               vllm_xiaotu_moe/ 2>/dev/null | grep -v '__pycache__' || true)
if [ -z "$musa_imports" ]; then ok "无顶层 torch_musa import"
else bad "顶层 import torch_musa:"; printf '%s\n' "$musa_imports"; fi

echo "[single-code] 4) 无平台专用依赖文件"
deps=$(find . -maxdepth 1 -iname 'requirements*musa*' 2>/dev/null)
if [ -z "$deps" ]; then ok "无平台专用依赖文件"
else bad "出现平台专用依赖:"; printf '%s\n' "$deps"; fi

echo "[single-code] 5) 无合并冲突标记"
markers=$(grep -rn '^<<<<<<< \|^>>>>>>> ' --include='*.py' --include='*.sh' --include='*.cpp' \
          --include='*.hpp' --include='*.md' vllm_xiaotu_moe xiaotu_moe scripts docs tests AGENTS.md 2>/dev/null | head -20 || true)
if [ -z "$markers" ]; then ok "无冲突标记"
else bad "残留冲突标记:"; printf '%s\n' "$markers"; fi

echo "[single-code] 6) 默认路径可导入（需要 vllm；本机无则跳过）"
if python3 -c 'import vllm' >/dev/null 2>&1; then
  if env -u GPU_DEVICE python3 -c 'import vllm_xiaotu_moe' >/dev/null 2>&1; then
    ok "GPU_DEVICE 未设时 import 成功（CUDA 默认路径）"
  else
    bad "GPU_DEVICE 未设时 import 失败"
  fi
else
  ok "跳过（本机未安装 vllm）"
fi

echo
if [ "$fail" -eq 0 ]; then echo "ALL PASS — 单代码线检查通过"; exit 0
else echo "FAIL ($fail 项) — 见上"; exit 1; fi
