#!/usr/bin/env bash
# =============================================================================
# check_terms.sh —— 术语误用机械检查（落实 dev-docs/GLOSSARY.md）
#
# 来由：2026-11-09 会话里 CED / SWA / SWA bounded replay 被反复混用，
#       用户连续纠正三次（USER_QA_LEDGER #415、#416）。
#       ⇒ 把"别乱用缩写"落成一条能跑的检查，而不是只写在文档里。
#
# 用法：
#   bash scripts/check_terms.sh              # 扫默认集合（AGENTS.md + dev-docs/ + docs/）
#   bash scripts/check_terms.sh <path> ...   # 扫指定文件/目录
#
# 退出码：0 = 无命中；1 = 命中（有误用）；2 = 参数/环境问题。
# ⚠️ 命中不一定都是错的：本条注释本身、GLOSSARY 的"常见误用"段、以及引用旧文的勘误会命中，
#    因此脚本只列出 文件:行号:片段，由人判定；**文档里若确属勘误，请在该行标注"❌ 常见误用/勘误"以豁免**。
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

if [ "$#" -gt 0 ]; then
    targets=("$@")
else
    # 默认集合 = "当前材料"：AGENTS.md + GLOSSARY + 最近 7 天改过的 dev-docs 文档（排除 archive）
    # ⇒ 历史文件（>7 天）默认只作提示，不阻断；用 --all 才全量扫
    targets=(AGENTS.md dev-docs/GLOSSARY.md)
    while IFS= read -r f; do targets+=("$f"); done < <(
        find dev-docs -maxdepth 1 -type f -name '*.md' -mtime -7 ! -path '*archive*' 2>/dev/null | sort
    )
fi

# 误用模式（正则，PCRE）
patterns=(
    'CED *[=＝] *SWA'                                        # CED 直接等于 SWA
    'CED *(就是|即|也就是) *SWA'                              # 中文同义混用
    'CED *(与|/|、) *`?--swa-bounded-replay'                 # 把两者并列成同义词
    'CED *(开启|默认|关闭|未生效|生效)'                       # 把 CED 当开关/特性（架构没有开关）
    '\bSBR\b|SWA-BR|\bBR\b'                               # 禁止缩写 SWA bounded replay
)

hits=0
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

for pat in "${patterns[@]}"; do
    # -r 递归、-n 行号、-I 跳过二进制；排除 GLOSSARY 自身的"常见误用"示范与勘误行
    grep -rnIP --binary-files=without-match "$pat" "${targets[@]}" 2>/dev/null \
        | grep -v 'GLOSSARY\.md' \
        | grep -vE '勘误|常见误用|❌|禁止任何缩写|等缩写|SWA-BR / SBR|误称|历史标题|一律以本条为准' \
        >> "$tmp" || true
done

if [ -s "$tmp" ]; then
    echo "⛔ 术语误用（CED / SWA / SWA bounded replay 混用）——见 dev-docs/GLOSSARY.md："
    sort -u "$tmp" | sed 's/^/  /'
    hits=1
else
    echo "✅ 未发现已知术语误用模式"
fi

exit "$hits"
