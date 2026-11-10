#!/usr/bin/env bash
# check_d1_state.sh —— 纯函数判据：D1 是否已应用，与本轮所处阶段是否一致
#
# 为什么（台账 #268）：E1 是**未修基线**，若 D1 已经打进树里，E1 测的就不是基线，整轮作废；
# 反过来 E2 阶段若 D1 没打，E2 测的就不是修复。两种错都会产出"看起来正常"的数据。
#
# 用法： bash scripts/check_d1_state.sh <已应用标记数> <expect: 0|1>
# 退出码：0 = 与阶段一致；1 = 不一致；2 = 用法错误
set -uo pipefail
n=${1:-}; expect=${2:-}
case "$n" in ''|*[!0-9]*) echo "⛔ 用法：check_d1_state.sh <标记数> <0|1>（得到 '$n'）"; exit 2 ;; esac
case "$expect" in 0|1) ;; *) echo "⛔ expect 只能是 0 或 1（得到 '$expect'）"; exit 2 ;; esac

if [ "$expect" = "1" ]; then
    if [ "$n" -ge 1 ]; then echo "✅ D1 已应用（$n 处标记）⇒ 符合 E2 阶段"; exit 0; fi
    echo "❌ E2 阶段要求 D1 已应用，但标记数为 0 ⇒ 先 e2_apply.sh apply 并重启"; exit 1
fi
if [ "$n" = "0" ]; then echo "✅ D1 未应用 ⇒ 符合基线（E1）阶段"; exit 0; fi
echo "❌ 基线阶段 D1 不该已应用（$n 处标记）⇒ 先回退（patch -R）再起基线"; exit 1
