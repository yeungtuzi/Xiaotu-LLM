#!/usr/bin/env bash
# 研究前置查证（强制）—— 在任何“第三方项目/性能/架构”结论或实验之前运行。
# 用法: bash scripts/research_preflight.sh strata sglang lsglang lvllm lk_moe
# 输出: 本地权威文档命中（dev-docs/docs），以及必须优先阅读的文件清单。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "$#" -eq 0 ]; then
  echo "usage: $0 <keyword> [keyword...]" >&2
  exit 2
fi
echo "=== research preflight: local authoritative docs ==="
for kw in "$@"; do
  echo
  echo "## $kw"
  # 只搜文本，排除 PDF/二进制；按命中文件聚合，最多 25 行
  grep -rIn --include='*.md' --include='*.txt' --include='*.sh' --include='*.py' \
    -i -- "$kw" docs dev-docs AGENTS.md README.md 2>/dev/null \
    | head -25 || true
done
echo
echo "=== 必读优先级（存在则先读） ==="
for f in \
  dev-docs/REFERENCE_RADAR.md \
  dev-docs/EXPERT_RESIDENCY_REFERENCES.md \
  dev-docs/AB_LK_VS_XIAOTU.md \
  dev-docs/STRATA_ENGINE_LESSONS.md \
  dev-docs/CPU_MOE_PLAN.md \
  dev-docs/PERFORMANCE_OPTIMIZATION.md \
  dev-docs/USER_QA_LEDGER.md \
  dev-docs/BUG_REGISTRY.md \
  dev-docs/report/tuning/NOTES.md \
  docs/EXPERIMENTS.md; do
  [ -f "$f" ] && echo "  - $f"
done
echo
echo "=== 规则: 未读上述命中/未写来源，不得给结论或开实验 ==="
