#!/usr/bin/env bash
# run_harness_selftests.sh —— 一条命令跑完本地 harness 的全部自检
#
# 用途：在"起服务前"确认判据体系本身没坏（语法 / 离线单测 / 端到端桩测）。
# 只跑本地、不连远端、不启停任何服务。
#
# 用法： bash scripts/run_harness_selftests.sh [--quiet]
# 退出码：0 = 全过；1 = 有失败
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1
fails=0
pass=0

say() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }
ok()  { pass=$((pass+1)); say "  ✅ $*"; }
bad() { fails=$((fails+1)); printf '  ❌ %s\n' "$*"; }

say "== 1) bash 语法（scripts/*.sh）"
for f in "$HERE"/*.sh; do
    if bash -n "$f" 2>/dev/null; then ok "$(basename "$f")"; else bad "$(basename "$f") 语法错误"; fi
done

say "== 2) python 语法（scripts/*.py，按 **3.10** 语法检查）"
# ⚠️ 本地 python 可能是 3.12+（本机是 3.13），而**远端是 3.10**。
#    3.12 起 PEP 701 允许 f-string 内复用同种引号 ⇒ 本地能解析、远端 SyntaxError。
#    这类"本地通过、远端炸"必须在本地就拦住，故显式用 feature_version=(3,10)。
if command -v python3 >/dev/null 2>&1; then
    for f in "$HERE"/*.py; do
        if python3 - "$f" <<'PYCHK' 2>/dev/null
import ast, sys
ast.parse(open(sys.argv[1], encoding="utf-8").read(), feature_version=(3, 10))
PYCHK
        then ok "$(basename "$f")"
        else bad "$(basename "$f") 语法错误（按 3.10 判；远端是 3.10）"; fi
    done
else
    say "  ⚠️ 没有 python3，跳过"
fi

say "== 3) 离线单测 / 桩测"
# 每个测试都在自己的目录下跑，避免相对路径问题
for t in test_verdict_branches.py test_token_mapping_semantics.py test_b_pass_guard.py \
         test_a_long_gate.py test_arms_no_evidence.py test_prefill_len_gate.py \
         test_ptok_check.sh test_repro_guard.sh test_repro_rc_verdict.sh \
         test_e1_rc_gate.sh test_e2_verdict.sh test_acceptance_wiring.sh \
         test_report_check.sh test_pc_expandable.sh test_preflight_go.sh \
         test_needle_ratio.py test_arms_accuracy.py test_needle_aggregate.py; do
    f="$HERE/$t"
    [ -f "$f" ] || { say "  ⏭  跳过（不存在）$t"; continue; }
    # ⛔ 防递归：preflight_go.sh 会调用本套件，而 test_preflight_go.sh 又会调用 preflight_go.sh。
    #    preflight 里跑自检时置 PREFLIGHT_ACTIVE=1，这里就跳过那个测试（否则无限递归）。
    if [ "$t" = "test_preflight_go.sh" ] && [ "${PREFLIGHT_ACTIVE:-0}" = "1" ]; then
        say "  ⏭  跳过（PREFLIGHT_ACTIVE=1，防递归）$t"; continue
    fi
    case "$t" in
        *.py) out=$(cd "$ROOT" && python3 "$f" 2>&1); rc=$? ;;
        *.sh) out=$(cd "$ROOT" && bash "$f" 2>&1); rc=$? ;;
    esac
    if [ "$rc" = 0 ]; then ok "$t"; else bad "$t（rc=$rc）"; printf '%s\n' "$out" | tail -12 | sed 's/^/      /'; fi
done

printf '\n== 汇总：%s 项通过，%s 项失败\n' "$pass" "$fails"
if [ "$fails" -ne 0 ]; then echo "❌ 自检未全过 ⇒ 先修判据体系，再去起服务"; exit 1; fi
echo "✅ 全部通过"
