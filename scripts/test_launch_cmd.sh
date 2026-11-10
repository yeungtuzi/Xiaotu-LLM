#!/usr/bin/env bash
# test_launch_cmd.sh —— 启动命令白名单比对必须有"会红"的证明（台账 #264）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fails=0

# ① 真实文档 + 真实脚本：应通过
out=$(bash "$HERE/check_launch_cmd.sh" 2>&1); rc=$?
[ "$rc" = 0 ] && echo "✅ 真实文档：通过" || { echo "❌ 真实文档：应通过 rc=$rc"; printf '%s\n' "$out"|tail -4; fails=$((fails+1)); }

# ② 造一份"含白名单外参数"的文档 ⇒ 必须报错并点名
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts" "$TMP/dev-docs"
cp "$HERE/serve_ds41.sh" "$TMP/scripts/"
cat > "$TMP/dev-docs/RESUME_AFTER_WEIGHTS_2026-11-10.md" <<'MD'
# 【v2 · 测试】
## 3.【远端】起服务
```
FOO_BAR=1 TAG=prod PORT=8700 bash scripts/serve_ds41.sh start
```
## 4.【远端】下一步
```
```
MD
out2=$(bash "$TMP/scripts/../scripts/check_launch_cmd.sh" 2>&1 || true)
# 用真实脚本 + 假文档（把 ROOT 指向临时目录）
cp "$HERE/check_launch_cmd.sh" "$TMP/scripts/"
out2=$(bash "$TMP/scripts/check_launch_cmd.sh" 2>&1); rc2=$?
if [ "$rc2" = 1 ] && printf '%s' "$out2" | grep -q "FOO_BAR"; then
    echo "✅ 白名单外参数被点名（rc=1）"
else
    echo "❌ 未抓到白名单外参数：rc=$rc2"; printf '%s\n' "$out2"|tail -6; fails=$((fails+1))
fi

echo
[ "$fails" -eq 0 ] && { echo "✅ 2/2 用例符合预期"; exit 0; }
echo "❌ $fails 个用例不符合预期"; exit 1
