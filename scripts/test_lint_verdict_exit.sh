#!/usr/bin/env bash
# test_lint_verdict_exit.sh —— 确认这条 lint **真的会报错**（不是永远绿灯）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fails=0

# ① 真实仓库：应当通过
out=$(bash "$HERE/lint_verdict_exit.sh" 2>&1); rc=$?
[ "$rc" = 0 ] && echo "✅ 真实仓库：通过" || { echo "❌ 真实仓库：应通过，实得 rc=$rc"; printf '%s\n' "$out"|tail -5; fails=$((fails+1)); }

# ② 造一个"打印失败但恒 return 0"的脚本，放进临时目录，lint 必须命中
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp "$HERE/lint_verdict_exit.sh" "$TMP/"
cat > "$TMP/accept_abc.py" <<'PY'
print("❌ 未通过：unique 命中率 90%")
raise SystemExit(0)
PY
# 其余 10 个文件用"有非零退出"的替身（保证只让第 1 个成为唯一失败点）
for f in probe_degen_arms.py long_uptime_needle.py analyze_replay_instrument.py prefill_perf.py; do
    printf 'import sys\nprint("❌ x")\nsys.exit(1)\n' > "$TMP/$f"
done
for f in repro_rc_verdict.sh kv_gate.sh ptok_check.sh check_weights_ready.sh check_running_profile.sh preflight_go.sh; do
    printf '#!/usr/bin/env bash\necho "❌ x"; exit 1\n' > "$TMP/$f"
done
out2=$(bash "$TMP/lint_verdict_exit.sh" 2>&1); rc2=$?
if [ "$rc2" = 1 ] && printf '%s' "$out2" | grep -q "accept_abc.py"; then
    echo "✅ 造假的 accept_abc.py 被点名（rc=1）"
else
    echo "❌ lint 没抓到造假的脚本：rc=$rc2"; printf '%s\n' "$out2" | tail -6; fails=$((fails+1))
fi

# ③ 缺失文件也要报（清单过期保护）
rm -f "$TMP/preflight_go.sh"
out3=$(bash "$TMP/lint_verdict_exit.sh" 2>&1); rc3=$?
[ "$rc3" = 1 ] && echo "✅ 清单里的文件缺失时报错（rc=1）" || { echo "❌ 缺文件应报错，实得 rc=$rc3"; fails=$((fails+1)); }

echo
[ "$fails" -eq 0 ] && { echo "✅ 3/3 用例符合预期"; exit 0; }
echo "❌ $fails 个用例不符合预期"; exit 1
