#!/usr/bin/env bash
# repro_rc_verdict.sh —— 把 degen_repro.sh 的退出码翻译成 run_e1 的动作
#
# 为什么要单独一层：degen_repro 的退出码**语义各不相同**，而 run_e1 原来只做了
# "非 0 就打一句 ⚠️ 然后继续"，于是 rc=2（口径不符，例如 ptok 不足 131K）会被当成普通失败吞掉，
# 后面若仪器判据恰好过，run_e1 仍可能 exit 0 ⇒ 拿"没测对口径"的一轮当"基线成立"（台账 #230）。
#
# 用法：bash scripts/repro_rc_verdict.sh <rc> <base|fix>
#   退出码：0 = 可以继续（结果有意义）；2 = 必须中止（口径不成立）
set -uo pipefail
rc=${1:?用法: repro_rc_verdict.sh <rc> <base|fix>}
mode=${2:-base}

case "$rc:$mode" in
    2:*)
        printf '⛔ degen_repro 报"口径不符"（rc=2，通常是 ptok 不足 131K）⇒ 本轮作废，先修口径\n'
        exit 2 ;;
    1:base)
        printf '✅ 基线成立：退化已复现（rc=1）\n'; exit 0 ;;
    1:fix)
        printf '❌ 修后仍有失败（rc=1）⇒ 修复未生效\n'; exit 0 ;;   # 继续走判据，最终由调用方判定
    0:base)
        printf '⚠️ 基线**未复现**（8/8 全过，rc=0）⇒ 不要当成"已修好"，先对比口径\n'; exit 0 ;;
    0:fix)
        printf '✅ 修后 8/8 全过（rc=0）\n'; exit 0 ;;
    *)
        printf '⚠️ degen_repro 退出码未知（rc=%s）⇒ 读原始输出人工判断\n' "$rc"; exit 0 ;;
esac
