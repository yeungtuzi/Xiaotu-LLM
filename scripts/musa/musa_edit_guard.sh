#!/usr/bin/env bash
# ============================================================================
# musa_edit_guard.sh —— 改目标机文件【前】的强制备份（2026-10-24 用户要求"破坏性操作可回滚"）
#
# 它只做三件事（⛔ 不起停任何进程、不 kill ⇒ 不属于 R38 起停脚本类别）：
#   1) 算 md5 + 行数 + 大小；
#   2) 复制到 BACKUPS/<name>_<reason>_<YYYYmmdd-HHMM>.py；
#   3) 往 BACKUPS/ROLLBACK_INDEX.tsv 追加一行，并打印**可直接复制的还原命令**。
#
# 用法（在**目标机**上、改文件之前跑）：
#   bash musa_edit_guard.sh /root/lvllm/vllm_xiaotu_moe/gpu_prefill_fp8.py w2tn
# 输出示例：
#   BACKUP=/root/vllm-xiaotu-moe-tmp/backups/gpu_prefill_fp8_w2tn_20261024-2215.py
#   MD5=<旧 md5>
#   RESTORE=bash musa_restore.sh <BACKUP>
#
# 退出码：0 已备份 / 2 参数或文件问题 / 3 备份校验失败
# ============================================================================
set -uo pipefail

SRC="${1:-}"
REASON="${2:-edit}"
BACKUPS="${MUSA_BACKUPS:-/root/vllm-xiaotu-moe-tmp/backups}"

[ -n "$SRC" ] || { echo "usage: musa_edit_guard.sh <file> [reason]" >&2; exit 2; }
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 2; }

mkdir -p "$BACKUPS" || exit 3
base=$(basename "$SRC")
stem="${base%.*}"
ext="${base##*.}"
[ "$ext" = "$base" ] && ext="bin"
stamp=$(date '+%Y%m%d-%H%M')
reason=$(printf '%s' "$REASON" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40)
dst="$BACKUPS/${stem}_${reason}_${stamp}.${ext}"

md5_before=$(md5sum "$SRC" | awk '{print $1}')
lines=$(wc -l < "$SRC")
bytes=$(stat -c %s "$SRC")

cp -p "$SRC" "$dst" || { echo "copy failed" >&2; exit 3; }
md5_after=$(md5sum "$dst" | awk '{print $1}')
if [ "$md5_before" != "$md5_after" ]; then
  echo "⚠ 备份 md5 不一致：$md5_before vs $md5_after" >&2
  exit 3
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date '+%F %T')" "$(hostname)" "$SRC" "$dst" "$md5_before" "$bytes" "$reason" \
  >> "$BACKUPS/ROLLBACK_INDEX.tsv"

echo "BACKUP=$dst"
echo "MD5=$md5_before"
echo "LINES=$lines BYTES=$bytes"
echo "TARGET=$SRC"
echo "RESTORE=bash $(dirname "$0")/musa_restore.sh $dst"
