#!/bin/bash
set -uo pipefail
OUT="${1:-${OUT:-$HOME/lvllm/stress-20261026}}"
echo "dir: $OUT"
for name in monitor driver; do
  f="$OUT/$name.pid"
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
    echo "  $name: RUNNING pid $(cat "$f")  ($(ps -o etime= -p "$(cat "$f")" 2>/dev/null | tr -d ' '))"
  else
    echo "  $name: not running"
  fi
done
[ -f "$OUT/alerts.log" ] && { echo "-- 最近告警 --"; tail -5 "$OUT/alerts.log"; }
[ -f "$OUT/driver.out" ] && { echo "-- 回放尾 --"; tail -3 "$OUT/driver.out"; }
[ -f "$OUT/monitor.csv" ] && { echo "-- 监控尾 --"; tail -2 "$OUT/monitor.csv"; }
