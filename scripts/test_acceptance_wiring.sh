#!/usr/bin/env bash
# test_acceptance_wiring.sh —— 断言 run_acceptance.sh 真的把"长上下文口径"传给了 A 臂。
#
# 背景（台账 #228）：A 的漏洞（#224）修在 accept_abc.py 里，但**编排脚本** run_acceptance.sh
# 原来既没传 --len 也没传 --require-long-tokens ⇒ 默认口径又退回 ≈5K，修复形同虚设。
# 这类"修在组件里、漏在编排层"的问题只能靠断言命令行来防。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out=$(DRY_RUN=1 bash "$HERE/run_acceptance.sh" 2>&1)
rc=$?
fails=0

[ "$rc" = 0 ] || { echo "❌ DRY_RUN 退出码 $rc（应为 0）"; fails=$((fails+1)); }

echo "$out" | grep -q -- "--require-long-tokens 120000" \
    && echo "✅ A 臂带了 --require-long-tokens 120000" \
    || { echo "❌ A 臂缺 --require-long-tokens"; fails=$((fails+1)); }

# 断言"带 --len 且值与默认一致"（数值改过多次，别把具体数字写死；见台账 #255）
want_len=$(grep -oP '(?<=^LEN="\$\{LEN:-)[0-9]+' "$HERE/run_acceptance.sh" | head -1)
want_len=${want_len:-80000}
echo "$out" | grep -q -- "--len $want_len" \
    && echo "✅ A 臂带了 --len $want_len（真 tokenizer 实测 125,044 tokens）" \
    || { echo "❌ A 臂缺 --len（期望 $want_len）"; fails=$((fails+1)); }

echo "$out" | grep -q -- "--a-rounds 10" \
    && echo "✅ A 臂 10 轮" \
    || { echo "❌ A 臂轮数不对"; fails=$((fails+1)); }

echo "$out" | grep -q "probe_degen_arms.py" \
    && echo "✅ B 臂走正式的四臂探针" \
    || { echo "❌ B 臂不是 probe_degen_arms"; fails=$((fails+1)); }

echo
if [ "$fails" -ne 0 ]; then echo "❌ $fails 处不符合预期"; exit 1; fi
echo "✅ 4/4 断言符合预期"
