#!/usr/bin/env bash
# ============================================================================
# test_doc_write_guards.sh —— `scripts/doc_write.sh` 的【守卫回归测试】(2026-10-17)
#
# 为什么存在(实证,不是担心):
#   2026-10-17 实测发现 R36 的机械执行点 `doc_write.sh` 有【两处守卫失效】:
#     ① 撞号检查 与 `--append-auto` **只认三井号 `### #NN`** ✗,而 `USER_QA_LEDGER.md`
#        等台账实际写的是【两井号】`## #NN` ⇒ 检查对它们**完全失效** ✗
#        ⇒ 后果:真的撞出 `#255` / `#256` **各两条**(两个会话各写一条,脚本一声不吭 ✓)
#     ② `--replace` 遇到 **old == new** 时**报"成功"却什么都没改** ✗
#        成因:调用方用 bash `${v/pat/rep}` 构造 new,而 pat **以 `#` 开头** ⇒
#        bash 把那个 `#` 当成【锚定在开头】的标记 ⇒ **静默不替换**(实测 ✓)
#   ⇒ 本脚本把这两处修复钉成**可跑回归**:任何人改了 `doc_write.sh` 后跑一遍即可 ✓
#
# 用法:`bash scripts/test_doc_write_guards.sh`   (秒级;不碰任何真文档;不需要起服务 ✓)
# 判据:末行 `ALL PASS (10/10)` 且 **exit 0** ✓
#      (2026-10-24 新增 ⑥/⑦:调用方编号恰为"最大+1"时曾被**静默拒绝**,见脚本内注释)
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
DW=scripts/doc_write.sh
[ -f "$DW" ] || { echo "⛔ 找不到 $DW"; exit 2; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
pass=0; fail=0
chk_exit(){ # chk_exit <名字> <期望 exit> <实际 exit>
  if [ "$2" = "$3" ]; then echo "  ✅ $1 (exit=$3)"; pass=$((pass+1))
  else echo "  ⛔ $1 —— 期望 exit=$2,实得 exit=$3"; fail=$((fail+1)); fi
}
chk_text(){ # chk_text <名字> <期望整行> <实际整行>
  if [ "$2" = "$3" ]; then echo "  ✅ $1"; pass=$((pass+1))
  else echo "  ⛔ $1 —— 期望 [$2],实得 [$3]"; fail=$((fail+1)); fi
}

F="$T/ledger.md"; printf '## #4(2026-01-01) a\n\n## #5(2026-01-02) b\n' > "$F"

# ① 【两井号】撞号必须被拒(这是 2026-10-17 失效的那条)
printf '## #5(2026-01-03) dup\n' | bash "$DW" "$F" --append >/dev/null 2>&1
chk_exit "① 两井号撞号 ⇒ 拒写(exit 4)" 4 $?

# ② --append-auto 对【两井号】可用,且**保留两井号**
printf '## #0(2026-01-04) auto\n' | bash "$DW" "$F" --append-auto >/dev/null 2>&1
chk_exit "② --append-auto 对两井号可用(exit 0)" 0 $?
chk_text "②b 取号 = 现有最大+1 且保留 '##'" '## #6(2026-01-04) auto' "$(tail -1 "$F")"

# ③ 三井号行为不变(向后兼容)
printf '### #0(2026-01-05) three\n' | bash "$DW" "$F" --append-auto >/dev/null 2>&1
chk_text "③ 三井号仍可用且保留 '###'" '### #7(2026-01-05) three' "$(tail -1 "$F")"

# ④ old == new 的空操作必须【当失败】(这是 2026-10-17 第二处失效)
bash "$DW" "$F" --replace '## #4(2026-01-01) a' '## #4(2026-01-01) a' >/dev/null 2>&1
chk_exit "④ replace 的 old==new ⇒ 空操作,拒写(exit 2)" 2 $?

# ⑤ 真改动必须成功(防止上面的守卫误伤正常替换)
bash "$DW" "$F" --replace '## #4(2026-01-01) a' '## #40(2026-01-01) a' >/dev/null 2>&1
chk_exit "⑤ replace 真改动 ⇒ 成功(exit 0)" 0 $?
chk_text "⑤b 内容确实被改" '## #40(2026-01-01) a' "$(head -1 "$F")"

# ⑥ (2026-10-24) 调用方编号**恰好等于 现有最大+1** 时必须仍能取号
#    —— 曾是"静默拒绝":判据写成 `body2 == body`,而替换后文本与原文字节相同 ⇒ 误判"没有编号" ✗
#    (实测:台账最大 #298 时写 `## #299` 的条目,连续两次被拒,报的却是"首行没有编号可分配")
printf '## #41(2026-10-24) exact\n' | bash "$DW" "$F" --append-auto >/dev/null 2>&1
chk_exit "⑥ 编号恰为 最大+1 ⇒ 仍成功(exit 0)" 0 $?
chk_text "⑥b 该条确实被追加" '## #41(2026-10-24) exact' "$(tail -1 "$F")"

# ⑦ 真·没有编号时仍必须拒(防止 ⑥ 的修法把守卫改成恒放行)
printf '没有编号的一行\n' | bash "$DW" "$F" --append-auto >/dev/null 2>&1
chk_exit "⑦ 无编号 ⇒ 仍拒写(exit 4)" 4 $?

echo "ALL $([ "$fail" -eq 0 ] && echo PASS || echo FAIL) ($pass/$((pass+fail)))"
[ "$fail" -eq 0 ]
