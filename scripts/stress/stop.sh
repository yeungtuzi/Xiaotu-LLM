#!/bin/bash
# 只按 PID 文件停，绝不按名字匹配。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-${OUT:-$HOME/lvllm/stress-20261026}}"
for name in driver monitor; do
  f="$OUT/$name.pid"
  if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
      echo "stopped $name (pid $pid)"
    else
      echo "$name 未在运行 (stale pid $pid)"
    fi
    rm -f "$f"
  else
    echo "$name: no pid file"
  fi
done
