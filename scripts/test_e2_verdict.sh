#!/usr/bin/env bash
# test_e2_verdict.sh —— E2 对照的三件事（台账 #229）：
#   ① "修前/修后"文件必须**显式**（原来 `ls -t` 会把修后当修前 ⇒ 自己比自己）
#   ② prefill 掉到 <95% 时，E2 必须**失败**（原来 pp 算了但没进判定）
#   ③ 缺文件时也必须失败（无法证明"prefill 未牺牲"）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/run"

mk() { # 文件 每条秒数(ptok 固定 131774)
    : > "$1"
    for s in "$@"; do :; done
    shift_file="$1"; shift
    for sec in "$@"; do
        printf 'cached=0 ptok=131774 %ss 命中=✗\n' "$sec" >> "$shift_file"
    done
}

run_case() { # 名称 期望rc
    out=$(SKIP_HEALTH=1 MODE=compare ROOT="$ROOT" RUN_DIR="$TMP/run" bash "$HERE/run_e2_verify.sh" 2>&1)
    rc=$?
    printf '%s\n' "$out" > "$TMP/last.out"
    [ "$rc" = "$2" ] && { printf '✅ %s（rc=%s）\n' "$1" "$rc"; return 0; }
    printf '❌ %s：期望 rc=%s 实得 %s\n%s\n' "$1" "$2" "$rc" "$out"; return 1
}

fails=0

# ① 正常：修前 120s、修后 124s（≈97%）
mk "$TMP/run/e1_e1.txt" 120.0 122.0
mk "$TMP/run/e2_e1.txt" 124.0 125.0
# 真实的 e2_e1.txt 里含 degen_repro 的汇总行（run_e1 会把它带过来）⇒ 测试也要还原这一点
printf '=== 汇总 ok=8 bad=0 口径不符=0 首个失败=无\n' >> "$TMP/run/e2_e1.txt"
printf '三条判据全满足\n' > "$TMP/run/e2_verdict.txt"
run_case "prefill ≈97% ⇒ 通过" 0 || fails=$((fails+1))

# ② prefill 掉 25% ⇒ 必须失败
mk "$TMP/run/e2_e1.txt" 160.0 162.0
printf '=== 汇总 ok=8 bad=0 口径不符=0 首个失败=无\n' >> "$TMP/run/e2_e1.txt"
run_case "prefill ≈75% ⇒ 必须失败" 1 || fails=$((fails+1))
grep -q "prefill 对照未通过" "$TMP/last.out" || { echo "❌ 缺少 prefill 未通过提示"; fails=$((fails+1)); }

# ③ 缺修前文件 ⇒ 必须失败（不能静默跳过）
rm -f "$TMP/run/e1_e1.txt"
run_case "缺修前文件 ⇒ 必须失败" 1 || fails=$((fails+1))
grep -q "无法证明 prefill 未牺牲" "$TMP/last.out" || { echo "❌ 缺少'无法证明'提示"; fails=$((fails+1)); }


# 【2026-11-10 新增 · 台账 #252】口径自证：服务未运行（无 PID 文件）⇒ 必须 rc=2 作废，而不是继续算对照
out=$(MODE=compare ROOT="$ROOT" RUN_DIR="$TMP/run_nosvc" bash "$HERE/run_e2_verify.sh" 2>&1); rc=$?
if [ "$rc" = 2 ] && printf '%s' "$out" | grep -q "运行口径不符"; then
    echo "✅ 服务未运行 ⇒ 口径自证让本轮作废（rc=2）"
else
    echo "❌ 口径自证未生效：rc=$rc"; printf '%s\n' "$out" | tail -6; fails=$((fails+1))
fi

echo
if [ "$fails" -ne 0 ]; then echo "❌ $fails 处不符合预期"; exit 1; fi
echo "✅ 4 个场景 + 2 处文案检查 全部符合预期（含口径自证反例）"
