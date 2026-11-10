#!/usr/bin/env bash
# test_repro_guard.sh —— 用假探针验证 degen_repro.sh 的三件事：
#   ① 全 131K + 全命中        ⇒ ok=8 rc=0（正常通过）
#   ② 全 90K  + 全命中        ⇒ **不得**报通过，rc=2（口径不符）  ← 台账 #226 的漏洞
#   ③ 前 7 条对、第 8 条错     ⇒ rc=1 且提示"第 8 条起失败"
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts"
cp "$HERE/degen_repro.sh" "$HERE/ptok_check.sh" "$TMP/scripts/"

cat > "$TMP/scripts/probe_dump_long.py" <<'PY'
import os, sys
mode = os.environ.get("FAKE_MODE", "ok")
ptok = os.environ.get("FAKE_PTOK", "131774")
cnt_file = os.path.join(os.path.dirname(__file__), "_cnt")
n = int(open(cnt_file).read()) + 1 if os.path.exists(cnt_file) else 1
open(cnt_file, "w").write(str(n))
hit = "✓"
if mode == "eighth" and n >= 8:
    hit = "✗"
print("cached=0 ptok=%s 90.0s 命中=%s 期望=1 实得=[]" % (ptok, hit))
PY

run_case() { # 名称 期望rc FAKE_MODE FAKE_PTOK
    rm -f "$TMP/scripts/_cnt"
    out=$(cd "$TMP" && SKIP_HEALTH=1 FAKE_MODE="$3" FAKE_PTOK="$4" OUT_DIR="$TMP/out" bash scripts/degen_repro.sh 2>&1)
    rc=$?
    printf '%s\n' "$out" > "$TMP/last.out"
    if [ "$rc" = "$2" ]; then printf '✅ %s（rc=%s）\n' "$1" "$rc"
    else printf '❌ %s：期望 rc=%s 实得 %s\n%s\n' "$1" "$2" "$rc" "$out"; return 1; fi
}

fails=0
run_case "全 131K + 全命中 ⇒ 通过" 0 ok 131774 || fails=$((fails+1))
run_case "全 90K + 全命中 ⇒ 必须不通过（rc=2）" 2 ok 90000 || fails=$((fails+1))
grep -q "不能当作 131K 的结论" "$TMP/last.out" || { echo "❌ 90K 情形缺少口径警告"; fails=$((fails+1)); }
run_case "第 8 条起错 ⇒ rc=1" 1 eighth 131774 || fails=$((fails+1))
grep -q "第 8 条起失败" "$TMP/last.out" || { echo "❌ 未提示第 8 条起失败"; fails=$((fails+1)); }

echo
if [ "$fails" -ne 0 ]; then echo "❌ $fails 处不符合预期"; exit 1; fi
echo "✅ 3 个场景 + 2 处文案检查 全部符合预期"
