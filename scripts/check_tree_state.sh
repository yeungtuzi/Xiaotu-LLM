#!/usr/bin/env bash
# check_tree_state.sh —— 【远端运行】报告 vLLM 树的两项状态：仪器补丁标记数 / D1 标记数
#
# 单独成脚本的原因（台账 #273）：把这两条 grep 内联进 preflight 的 ssh 单引号字符串里，
# 只要里面再出现引号就会导致 "unexpected EOF while looking for matching"，
# 而那种报错**看起来像判据失败**（其实是脚本自己的语法错）。抽出来以后：
#   * 远端只管跑、打印两个数字；
#   * 本地只管解析数字并判定。
#
# 输出（固定两行，便于 awk 解析）：
#   INSTRUMENT <n>
#   D1 <n>
set -uo pipefail
TREE=${TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}
i=$(grep -c "xtu-swa-bounded-replay" "$TREE/vllm/models/deepseek_v41/nvidia/model_state.py" 2>/dev/null || true)
d=$(grep -c "Recompute the validity mask" "$TREE/vllm/v1/attention/backends/mla/sparse_swa.py" 2>/dev/null || true)
echo "INSTRUMENT ${i:-0}"
echo "D1 ${d:-0}"
