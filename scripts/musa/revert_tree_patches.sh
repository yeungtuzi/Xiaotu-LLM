#!/usr/bin/env bash
# 把"我们项目在 vLLM 树内留的 4 处补丁"还原成上游/fork 原样（HANDOFF §34/§35.4）。
#
# 为什么要脚本：还原要**精确**（md5 必须等于 .bak / git HEAD），
# 手写 `cp`/`git checkout` 很容易漏一个或覆盖错文件（本项目有前科）。
#
# 用法：
#   bash scripts/musa/revert_tree_patches.sh                 # 只体检（默认，不动任何文件）
#   bash scripts/musa/revert_tree_patches.sh --apply         # 真的还原
#   bash scripts/musa/revert_tree_patches.sh --apply --only utils   # 只还原某一项
#
# 判据（脚本自己检查，失败即退出码非 0）：
#   1. 还原前：文件存在且**确实是改过的**（与 bak/HEAD 不同）
#   2. 还原后：`md5sum` == bak 的 md5（或 git status 对该文件已干净）
#   3. 全流程结束：4 处都干净 ⇒ 打印 OK
set -u -o pipefail

VLLM_TREE="${VLLM_TREE:-/root/vllm-musa/third_party/vllm}"
APPLY=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --only) shift; ONLY="${1:-}" ;;
    --tree) shift; VLLM_TREE="${1:-}" ;;
    -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

[ -d "$VLLM_TREE" ] || { echo "⛔ 树不存在: $VLLM_TREE" >&2; exit 2; }

# 项：<短名> <相对路径> <bak 文件或 "-"> <期望 md5（"-" = 用 git 判定）>
ITEMS=(
  "linear|vllm/model_executor/layers/linear.py|vllm/model_executor/layers/linear.py.bak_20261025-093546|9c026f7d15dcb9200281013576e8eb82"
  "utils|vllm/model_executor/models/utils.py|vllm/model_executor/models/utils.py.bak_20261025-094809|80341ce6d680311a754de3d0185b9153"
  "emul|vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py|vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py.bak_20261025-105714|36b821dff39b0a6d3fb4ec567edc0a38"
  "oracle|vllm/model_executor/layers/fused_moe/oracle/nvfp4.py|-|-"
)

fail=0
pending=0
for it in "${ITEMS[@]}"; do
  IFS='|' read -r name rel bak want <<<"$it"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then continue; fi
  f="$VLLM_TREE/$rel"
  if [ ! -f "$f" ]; then
    echo "⛔ [$name] 文件不存在: $f"; fail=1; continue
  fi
  cur=$(md5sum "$f" | awk '{print $1}')
  if [ "$bak" != "-" ]; then
    bf="$VLLM_TREE/$bak"
    [ -f "$bf" ] || { echo "⛔ [$name] 备份不存在: $bf"; fail=1; continue; }
    bmd5=$(md5sum "$bf" | awk '{print $1}')
    [ "$bmd5" = "$want" ] || { echo "⛔ [$name] 备份 md5 与期望不符: $bmd5 != $want"; fail=1; continue; }
    if [ "$cur" = "$bmd5" ]; then
      echo "✓ [$name] 已经是原样（md5=$cur）"
      continue
    fi
    pending=$((pending+1))
    echo "→ [$name] 需要还原: 当前 $cur → 备份 $bmd5"
    if [ "$APPLY" = "1" ]; then
      cp -f "$bf" "$f" || { echo "⛔ [$name] cp 失败"; fail=1; continue; }
      now=$(md5sum "$f" | awk '{print $1}')
      if [ "$now" = "$bmd5" ]; then
        echo "✓ [$name] 还原完成（md5=$now）"
      else
        echo "⛔ [$name] 还原后 md5 不符: $now != $bmd5"; fail=1
      fi
    fi
  else
    # 无 bak：用 git 判定
    if git -C "$VLLM_TREE" diff --quiet -- "$rel"; then
      echo "✓ [$name] git 已干净（无需还原）"
      continue
    fi
    pending=$((pending+1))
    echo "→ [$name] git 有改动，需要 checkout（当前 $cur）"
    if [ "$APPLY" = "1" ]; then
      git -C "$VLLM_TREE" checkout -- "$rel" || { echo "⛔ [$name] checkout 失败"; fail=1; continue; }
      if git -C "$VLLM_TREE" diff --quiet -- "$rel"; then
        echo "✓ [$name] 还原完成"
        # 额外判据：诊断日志必须消失
        n=$(grep -c "_xtu-nvfp4-diag" "$f" 2>/dev/null || true)
        [ "${n:-0}" = "0" ] || { echo "⛔ [$name] 仍残留 _xtu-nvfp4-diag（$n 处）"; fail=1; }
      else
        echo "⛔ [$name] checkout 后仍有改动"; fail=1
      fi
    fi
  fi
done

if [ "$APPLY" != "1" ]; then
  echo "（体检模式：$pending 处待还原；加 --apply 才会真正动文件）"
  [ "$pending" = "0" ] || exit 0
fi
if [ "$fail" != "0" ]; then
  echo "⛔ 有失败项" >&2
  exit 1
fi
if [ "$APPLY" = "1" ]; then echo "✅ 4 处已全部还原为原样"; else echo "✅ 4 处均为原样（无需还原）"; fi
