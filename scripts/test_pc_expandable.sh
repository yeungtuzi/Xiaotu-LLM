#!/usr/bin/env bash
# test_pc_expandable.sh —— 四种组合的判据
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fails=0
chk() { out=$(bash "$HERE/check_pc_expandable.sh" "$2" "$3" 2>&1); rc=$?
    if [ "$rc" = "$4" ]; then printf '✅ %s（rc=%s）\n' "$1" "$rc"; else printf '❌ %s：期望 %s 实得 %s\n%s\n' "$1" "$4" "$rc" "$out"; fails=$((fails+1)); fi; }
chk "生产口径：PC=0 + expandable" \
    "python -m vllm ... --enable-prefix-caching" "expandable_segments:True" 0
chk "磁盘前缀缓存 + expandable ⇒ 冲突" \
    "python -m vllm ... --kv-transfer-config {}" "expandable_segments:True" 2
chk "磁盘前缀缓存 + 非 expandable ⇒ 相容" \
    "python -m vllm ... --kv-transfer-config {}" "expandable_segments:False" 0
chk "两者都无 ⇒ 相容" "python -m vllm ..." "" 0
echo
[ "$fails" -eq 0 ] && { echo "✅ 4/4 用例符合预期"; exit 0; }
echo "❌ $fails 个用例不符合预期"; exit 1
