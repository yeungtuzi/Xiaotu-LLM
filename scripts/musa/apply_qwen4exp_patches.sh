#!/usr/bin/env bash
# Apply / check the candidate qwen4_exp MUSA patches.
# Usage: apply_qwen4exp_patches.sh /path/to/vllm [--check]
set -euo pipefail
ROOT=${1:?usage: $0 /path/to/vllm [--check]}
MODE=${2:-apply}
PATCH_DIR="$(cd "$(dirname "$0")/../../patches/musa" && pwd)"
PATCHES=(
  "$PATCH_DIR/0001-qwen4exp-musa-select-triton.patch"
  "$PATCH_DIR/0002-qwen4exp-musa-disable-pdl.patch"
  "$PATCH_DIR/0003-qwen4exp-musa-ple-pdl.patch"
  "$PATCH_DIR/0004-qwen4exp-musa-qsa-cache-pdl.patch"
  "$PATCH_DIR/0005-qwen4exp-musa-qsa-topk-fallback.patch"
  "$PATCH_DIR/0006-qwen4exp-musa-qsa-partial-stages.patch"
)
cd "$ROOT"
for p in "${PATCHES[@]}"; do
  echo "[check] $(basename "$p")"
  git apply --check "$p"
done
if [ "$MODE" = "--check" ]; then
  echo "all patches check clean"
  exit 0
fi
for p in "${PATCHES[@]}"; do
  echo "[apply] $(basename "$p")"
  git apply "$p"
done
echo "all patches applied"
