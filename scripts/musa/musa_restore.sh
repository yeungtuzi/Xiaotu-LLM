#!/usr/bin/env bash
# ============================================================================
# musa_restore.sh —— 一键回滚（配合 musa_edit_guard.sh 的备份 + ROLLBACK_INDEX.tsv）
#
# ⛔ 不起停任何进程、不 kill（只做文件还原）。
# ⚠️ 还原**不会**自动重启引擎：文件换了要重起才生效，而重起按纪律要攒批。
#
# 用法（目标机）：
#   bash musa_restore.sh --list                      # 列最近的备份
#   bash musa_restore.sh <backup-file> [--yes]       # 还原（会先打印将被覆盖的目标）
#
# 判据：还原后打印新 md5，并到 ROLLBACK_INDEX.tsv 里查该备份登记的旧 md5 ⇒ 两者必须相等。
# 退出码：0 还原并校验一致 / 2 参数问题 / 3 校验失败 / 4 需要 --yes 确认
# ============================================================================
set -uo pipefail

BACKUPS="${MUSA_BACKUPS:-/root/vllm-xiaotu-moe-tmp/backups}"
IDX="$BACKUPS/ROLLBACK_INDEX.tsv"

if [ "${1:-}" = "--list" ] || [ $# -eq 0 ]; then
  if [ -f "$IDX" ]; then
    echo "==== ROLLBACK_INDEX（最近 20 条）===="
    tail -20 "$IDX" | awk -F'\t' '{printf "%s  %s\n   target=%s\n   backup=%s\n   md5=%s reason=%s\n", $1, $2, $3, $4, $5, $7}'
  else
    echo "（还没有任何备份记录：$IDX 不存在）"
  fi
  exit 0
fi

BK="$1"; shift || true
CONFIRM="${1:-}"
[ -f "$BK" ] || { echo "no such backup: $BK" >&2; exit 2; }

# 从索引里找回"这个备份属于哪个目标文件 + 它当时的 md5"
row=$(grep -F "	$BK	" "$IDX" 2>/dev/null | tail -1)
if [ -n "$row" ]; then
  TGT=$(printf '%s' "$row" | awk -F'\t' '{print $3}')
  OLDMD5=$(printf '%s' "$row" | awk -F'\t' '{print $5}')
else
  echo "⚠ 索引里找不到该备份（可能是手工复制的备份）⇒ 必须显式给出目标：" >&2
  echo "   TARGET_FILE=<path> bash musa_restore.sh $BK --yes" >&2
  TGT="${TARGET_FILE:-}"
  [ -n "$TGT" ] || exit 2
  OLDMD5=$(md5sum "$BK" | awk '{print $1}')
fi

echo "将还原：$BK"
echo "   -> $TGT"
echo "   还原后应为 md5=$OLDMD5"
if [ -f "$TGT" ]; then
  echo "   当前目标 md5=$(md5sum "$TGT" | awk '{print $1}')"
fi
if [ "$CONFIRM" != "--yes" ]; then
  echo "（未加 --yes，未做任何改动）"
  exit 4
fi

cp -p "$BK" "$TGT" || { echo "restore copy failed" >&2; exit 3; }
new=$(md5sum "$TGT" | awk '{print $1}')
if [ "$new" = "$OLDMD5" ]; then
  echo "✓ 还原完成且 md5 一致：$new"
  echo "  ⚠ 引擎正在跑的话，新文件要重起才生效（按纪律攒批再重起）"
  printf '%s\t%s\tRESTORE\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(hostname)" "$TGT" "$BK" "$new" "restored" >> "$IDX"
  exit 0
fi
echo "✗ md5 不一致：期望 $OLDMD5，实际 $new" >&2
exit 3
