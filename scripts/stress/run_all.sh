#!/bin/bash
# 启动 A100 长稳压力测试（回放 + 监控）。只在本机跑，只打 127.0.0.1。
# 用法: bash run_all.sh [--port <port>] [--hours 72] [--mode transcript] [--max-ctx-tokens 200000] [--max-outage-min 360]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HOME/lvllm/stress-20261026}"
PORT=8070
HOURS=72
MODE=transcript
MAXCTX=200000
MAXOUT=30
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --hours) HOURS="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --max-ctx-tokens) MAXCTX="$2"; shift 2;;
    --max-outage-min) MAXOUT="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

mkdir -p "$OUT"
cd "$HERE"

# ⛔ 不依赖外部网络：清掉一切 proxy 变量，并只在 loopback 上说话
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY ftp_proxy FTP_PROXY || true
export no_proxy="127.0.0.1,localhost" NO_PROXY="127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
umask 022

echo "== 预检 =="
curl -s -o /dev/null --max-time 8 -w "health($PORT): %{http_code}\n" "http://127.0.0.1:$PORT/health" || true
if ! curl -sf -o /dev/null --max-time 8 "http://127.0.0.1:$PORT/health"; then
  echo "⛔ 端口 $PORT 的 /health 不通，拒绝启动。"
  echo "   请用 --port <实际端口> 重跑（端口见 dev-docs/ENV_TWO_MACHINES.md）。"
  exit 2
fi
FREE_GIB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
if [ "${FREE_GIB:-0}" -lt 20 ]; then echo "⛔ $OUT 剩余 ${FREE_GIB}G < 20G"; exit 2; fi
for f in driver.pid monitor.pid; do
  if [ -f "$OUT/$f" ] && kill -0 "$(cat "$OUT/$f")" 2>/dev/null; then
    echo "⛔ 已有进程在跑（$OUT/$f = $(cat "$OUT/$f")），先 stop.sh"; exit 2
  fi
done

echo "== 启动监控（只读） =="
setsid nohup /usr/bin/python3 -u "$HERE/monitor.py" \
  --port "$PORT" --hours "$HOURS" --out-dir "$OUT" \
  > "$OUT/monitor.out" 2>&1 < /dev/null &
echo $! > "$OUT/monitor.pid"

sleep 2
echo "== 启动回放（dry-run，不执行工具） =="
setsid nohup /usr/bin/python3 -u "$HERE/replay_driver.py" \
  --port "$PORT" --hours "$HOURS" --mode "$MODE" --max-ctx-tokens "$MAXCTX" --max-outage-min "$MAXOUT" \
  --out-dir "$OUT" \
  > "$OUT/driver.out" 2>&1 < /dev/null &
echo $! > "$OUT/driver.pid"

sleep 5
echo "== 状态 =="
bash "$HERE/status.sh" "$OUT"
cat <<EOF

日志目录: $OUT
  driver.out      回放过程（每轮一行）
  replay.jsonl    每次请求的延迟/token/重复率
  monitor.csv     5 秒一采样的健康/显存/内存/fd
  alerts.log      只写告警（DOWN / RESTART / LEAK_SUSPECT / OOM_KILL）
停止: bash $HERE/stop.sh   状态: bash $HERE/status.sh
EOF
