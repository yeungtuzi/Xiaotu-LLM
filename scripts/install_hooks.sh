#!/usr/bin/env bash
# 安装 git hooks（.git/hooks 不在版本控制里，fresh clone 不会有 ⇒ 用本脚本重建）。
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
cat > .git/hooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
# 提交前跑内核资源门禁 + 自检（自检里的源码反回归能抓旧 bug 复活）。
# ⚠️ fail-closed：缺少 python3 时**拒绝提交**，而不是静默跳过。
set -o pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ 找不到 python3，无法执行门禁 ⇒ 拒绝提交（fail-closed）" >&2
  echo "   如确需跳过：git commit --no-verify" >&2
  exit 1
fi
echo "[pre-commit] 内核资源门禁…"
python3 scripts/check_kernel_alloc_rules.py || { echo "❌ 内核资源门禁未通过，提交被拒" >&2; exit 1; }
echo "[pre-commit] 门禁自检…"
python3 scripts/test_kernel_alloc_rules.py || { echo "❌ 门禁自检未通过，提交被拒" >&2; exit 1; }
HOOK
chmod +x .git/hooks/pre-commit
echo "✓ 已安装 .git/hooks/pre-commit（fail-closed）"
