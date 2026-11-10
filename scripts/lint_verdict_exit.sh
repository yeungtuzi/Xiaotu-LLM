#!/usr/bin/env bash
# lint_verdict_exit.sh —— 审计"**会给出验收结论**"的脚本：失败路径必须变成非零退出码
#
# 为什么（台账 #259/#260）：同一个坑踩了两次 —— 判据打印得很清楚，但 `return 0` / `exit 0` 恒定，
# 于是"未通过"在自动化里等于"通过"，人看日志完全看不出来。
#
# 与"全仓扫描"的区别：全仓扫会被注释/文档字符串里的 ⛔ 大量误报（实测 29 个假命中），
# 没人会天天看一份 29 条噪声的报告。⇒ 只审**验收链上**的那几个脚本，宁少勿滥。
#
# 用法： bash scripts/lint_verdict_exit.sh [--verbose]
# 退出码：0 = 全部有非零退出路径；1 = 有脚本恒定成功
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERBOSE=0; [ "${1:-}" = "--verbose" ] && VERBOSE=1

# 验收链上的脚本（新增验收脚本请加到这里；这是白名单也是责任清单）
FILES=(
    accept_abc.py                 # 验收 A/B
    probe_degen_arms.py           # 验收 B 四臂
    long_uptime_needle.py         # 验收 C 探针 + 聚合
    analyze_replay_instrument.py  # E1/E2 三条判据
    prefill_perf.py               # prefill 对照
    repro_rc_verdict.sh           # 复现 rc 语义
    kv_gate.sh                    # KV 预算门
    ptok_check.sh                 # 长度门
    check_weights_ready.sh        # 权重就绪
    check_running_profile.sh      # 运行口径
    preflight_go.sh               # 起服务总闸
)

hits=0
for b in "${FILES[@]}"; do
    f="$HERE/$b"
    if [ ! -f "$f" ]; then echo "⚠️  缺少 $b（清单过期？）"; hits=$((hits+1)); continue; fi
    # 非零退出路径：return/exit/sys.exit 后跟非 0 数字，或 die/SystemExit，或 `exit "$rc"`
    # 注意：`raise SystemExit(0)` / `return 0` 都**不算**失败路径（这一点曾误判过，见 test 的反例）
    if grep -qE '(return|exit|sys\.exit)\s*\(?[1-9]|raise SystemExit\([1-9]|exit "\$rc"|die ' "$f"; then
        [ "$VERBOSE" = 1 ] && echo "✅ $b"
    else
        echo "❌ $b：没有任何非零退出路径（结论不会进退出码）"
        grep -nE '未通过|❌|⛔|FAIL' "$f" | head -2 | sed 's/^/      /'
        hits=$((hits+1))
    fi
done

echo
if [ "$hits" -eq 0 ]; then echo "✅ 验收链上 ${#FILES[@]} 个脚本都有非零退出路径"; exit 0; fi
echo "❌ $hits 个脚本需要修（把结论变成退出码）"; exit 1
