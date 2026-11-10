#!/usr/bin/env bash
# test_pin_table_check.sh —— PIN 表判据必须"会红"（台账 #265）
# 用【本地临时文件】替身，不连远端：把 SSH 指向一个把命令原样执行的 stub。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fails=0

# ② 三张表：好的 / 魔数错的 / 空的
python3 - "$TMP" <<'PY'
import sys, os
d=sys.argv[1]
open(os.path.join(d,"good.bin"),"wb").write(b"XTUP" + (0).to_bytes(4,"little")*8)
open(os.path.join(d,"badmagic.bin"),"wb").write(b"XXXX" + b"\x00"*16)
open(os.path.join(d,"empty.bin"),"wb").write(b"")
PY

run() { # 表路径 期望rc
    out=$(LOCAL=1 bash "$HERE/check_pin_table.sh" "$1" 2>&1); rc=$?
    if [ "$rc" = "$2" ]; then printf '✅ %-16s rc=%s\n' "$(basename "$1")" "$rc"
    else printf '❌ %-16s 期望 rc=%s 实得 %s\n%s\n' "$(basename "$1")" "$2" "$rc" "$out"; fails=$((fails+1)); fi
}

# 该脚本用 ssh 连远端，stub 后即可本地跑
run "$TMP/good.bin" 0
run "$TMP/badmagic.bin" 1
run "$TMP/empty.bin" 1
run "$TMP/nonexistent.bin" 1

echo
[ "$fails" -eq 0 ] && { echo "✅ 4/4 用例符合预期"; exit 0; }
echo "❌ $fails 个用例不符合预期"; exit 1
