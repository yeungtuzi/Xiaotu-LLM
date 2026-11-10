#!/usr/bin/env bash
# test_e1_rc_gate.sh —— run_e1 必须把 degen_repro 的 rc=2（口径不符）升级为"本轮作废并中止"。
#
# 台账 #230：原来 `if ! bash degen_repro ...; then say "⚠️ 非 0（仍继续分析）"; fi`
# ⇒ rc=2 被当成普通失败吞掉，若后续仪器判据恰好通过，run_e1 会 exit 0，
#    于是"口径没测对"的一轮被记成"基线成立/修复有效"。
#
# 做法：在临时目录里放 run_e1.sh + 桩（degen_repro / probe_dump_long / analyze_replay_instrument），
#      用 SKIP_HEALTH=1 跳过健康检查，断言三种场景的退出码。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/run"

cp "$HERE/run_e1.sh" "$HERE/repro_rc_verdict.sh" "$TMP/scripts/"

# 桩：degen_repro 的行为由 FAKE_REPRO_RC 控制；输出里带一行汇总（真实脚本也有）
cat > "$TMP/scripts/degen_repro.sh" <<'PY'
#!/usr/bin/env bash
echo "=== degen_repro tag=$TAG N=$N cnt=$CNT"
echo "[1] cached=0 ptok=131774 90.0s 命中=✗"
echo "=== 汇总 ok=0 bad=8 口径不符=0 首个失败=#1"
exit "${FAKE_REPRO_RC:-1}"
PY

# 桩：探针第 2 条命中（满足 ③b 行为判据）
cat > "$TMP/scripts/probe_dump_long.py" <<'PY'
import os
n = int(open(os.path.join(os.path.dirname(__file__), "_hits")).read()) if os.path.exists(os.path.join(os.path.dirname(__file__), "_hits")) else 0
open(os.path.join(os.path.dirname(__file__), "_hits"), "w").write(str(n + 1))
print("cached=%d ptok=131774 1.0s 命中=✗" % (0 if n == 0 else 6400))
PY

# 桩：仪器判据全过
cat > "$TMP/scripts/analyze_replay_instrument.py" <<'PY'
print("✅ 三条判据全满足：hit=1 / computed == replay_start / pad_added=128")
raise SystemExit(0)
PY
chmod +x "$TMP/scripts/"*.sh

# 假 vLLM 树：只要带上仪器补丁标记，run_e1 就会认为"补丁已在树里"而不再要求 dbg.patch
mkdir -p "$TMP/tree/vllm/models/deepseek_v41/nvidia"
echo "# xtu-swa-bounded-replay (fake)" > "$TMP/tree/vllm/models/deepseek_v41/nvidia/model_state.py"

run_case() { # 名称 期望rc FAKE_REPRO_RC
    rm -f "$TMP/scripts/_hits"
    : > "$TMP/run/ds41_prod.log"
    out=$(cd "$TMP" && SKIP_HEALTH=1 TREE="$TMP/tree" RUN_DIR="$TMP/run" INSTANCE_TAG=prod PORT=8700 \
          N=7430 CNT=8 TAG=e1 FAKE_REPRO_RC="$3" bash scripts/run_e1.sh 2>&1)
    rc=$?
    printf '%s\n' "$out" > "$TMP/last.out"
    if [ "$rc" = "$2" ]; then printf '✅ %s（rc=%s）\n' "$1" "$rc"
    else printf '❌ %s：期望 rc=%s 实得 %s\n%s\n' "$1" "$2" "$rc" "$out"; return 1; fi
}

fails=0
run_case "rc=2（口径不符）⇒ run_e1 必须 rc=2" 2 2 || fails=$((fails+1))
grep -q "口径不成立" "$TMP/last.out" || { echo "❌ 缺少'口径不成立'提示"; fails=$((fails+1)); }

run_case "rc=1（基线已复现）⇒ 走完判据 rc=0" 0 1 || fails=$((fails+1))
grep -q "基线成立" "$TMP/last.out" || { echo "❌ 缺少'基线成立'提示"; fails=$((fails+1)); }

run_case "rc=0（未复现）⇒ 仍走判据但给警告" 0 0 || fails=$((fails+1))
grep -q "未复现" "$TMP/last.out" || { echo "❌ 缺少'未复现'警告"; fails=$((fails+1)); }


# 【2026-11-10 新增 · 台账 #267】RUN_DIR 被改成非默认值 ⇒ 触发文件路径与仪器硬编码不一致 ⇒ 必须 rc=2
rm -f "$TMP/scripts/_hits"; : > "$TMP/run/ds41_prod.log"
out=$(cd "$TMP" && RUN_DIR="$TMP/elsewhere" TREE="$TMP/tree" INSTANCE_TAG=prod PORT=8700 \
      N=7430 CNT=8 TAG=e1 FAKE_REPRO_RC=1 bash scripts/run_e1.sh 2>&1); rc=$?
if [ "$rc" = 2 ] && printf '%s' "$out" | grep -q "与仪器硬编码的"; then
    echo "✅ 自定义 RUN_DIR ⇒ rc=2（触发文件路径与仪器硬编码不一致）"
else
    echo "❌ 自定义 RUN_DIR 未被拦：rc=$rc"; printf '%s\n' "$out" | tail -5; fails=$((fails+1))
fi
echo
[ "$fails" -eq 0 ] && { echo "✅ 4 个场景 + 3 处文案 全部符合预期"; exit 0; }
echo "❌ $fails 处不符合预期"; exit 1
