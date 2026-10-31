#!/usr/bin/env bash
# 提建议/动手前查"已否决路线总账"。
#
# 用法:
#   scripts/check_rejected_routes.sh [关键词...]
#   scripts/check_rejected_routes.sh            # 不带词:打印目录(各节标题)与用法
#
# 退出码: 0 = 没命中(可以继续验证前提);1 = 命中(先读命中行,确认前提是否变了);2 = 文件缺失
#
# 说明:只读,不改任何文件、不碰进程。命中不代表"永远不许",只代表"必须先说明前提变了"。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/dev-docs/REJECTED_ROUTES.md"

if [ ! -f "$DOC" ]; then
  echo "⛔ 找不到 $DOC" >&2
  exit 2
fi

if [ "$#" -eq 0 ]; then
  echo "== 已否决路线总账: $DOC =="
  grep -nE '^## ' "$DOC"
  echo
  echo "用法: $0 <关键词> [关键词...]"
  echo "例:   $0 barrier 屏障      # 查同步类"
  echo "      $0 prefetch 预取       # 查预取类"
  echo "      $0 nvme numa 线程      # 多词(任一命中即返回)"
  exit 0
fi

rc=1
for kw in "$@"; do
  hits="$(grep -niE -- "$kw" "$DOC" | grep -vE '^[0-9]+:# ' || true)"
  if [ -n "$hits" ]; then
    rc=0
    echo "== 命中「$kw」=="
    printf '%s\n' "$hits" | cut -c1-220
    echo
  fi
done

if [ "$rc" -eq 0 ]; then
  echo "⚠️ 上面这些已否决过。要重开必须写清:哪条前提变了 + 新证据(服务级/同协议/≥2% 判据/口径)。"
else
  echo "未命中:本文件里没有直接相关的否决条目(仍建议先读 §0 的机械三步)。"
fi
exit "$rc"
