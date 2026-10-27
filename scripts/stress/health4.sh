#!/usr/bin/env bash
# health4.sh —— 开发 agent 指定的四条判据（压测期间随时可跑）
#   ① 持久缓冲是否恒等 1.776 GiB
#   ② GPU 占用（37.3/40 GB 为基线，持续爬升 = 有未记账分配）
#   ③ 宿主 RSS（单 rank ~253 GB 正常；单实例 >900 GiB 查泄漏）
#   ④ 致命信号计数（应恒为 0）
# 用法: bash health4.sh [日志路径]      # 缺省取 logs 下最新的 .log
PORT="${PORT:-8000}"
# 跟随【当前】服务：端口 → pid → 它的 stdout(日志)；找不到再退回最新 .log
resolve_log() {
  local pid t fd
  pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
  [ -n "$pid" ] || return 1
  for fd in 1 2; do
    t=$(readlink "/proc/$pid/fd/$fd" 2>/dev/null) || continue
    case "$t" in *.log) echo "$t"; return 0;; esac
  done
  return 1
}
L="${1:-$(resolve_log)}"
[ -n "$L" ] || L=$(ls -t "$HOME"/lvllm/*/dev-docs/report/tuning/logs/*.log 2>/dev/null | head -1)
SERVER_PID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
AGE=$(ps -o etime= -p "${SERVER_PID:-0}" 2>/dev/null | tr -d " ")
echo "=== $(date -u '+%F %T UTC')  port=$PORT pid=${SERVER_PID:-?} age=${AGE:-?} log=$(basename "${L:-none}")"

echo "--- ① 持久缓冲 steady（应恒等 1.776 GiB）"
out=$(grep -oE "xtu-vram. steady: 持久缓冲合计 [0-9.]+ GiB" "$L" 2>/dev/null | tail -5)
if [ -n "$out" ]; then
  echo "$out"
else
  echo "  (未匹配到 'xtu-vram. steady' 行；回退取最后 3 条含「持久缓冲」的记录)"
  grep -E "持久缓冲" "$L" 2>/dev/null | tail -3 | cut -c1-200 || true
fi

echo "--- ② GPU 占用 (MiB)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

echo "--- ③ 宿主 RSS"
ps -eo pid,rss,args | grep -E "[V]LLM::(Worker|EngineCore)" \
  | awk '{printf "  pid=%s rss=%.1f GiB\n", $1, $2/1048576}'
ps -eo rss,args | grep -E "[V]LLM::(Worker|EngineCore)" | awk '{s+=$1} END{printf "  合计=%.1f GiB\n", s/1048576}'

echo "--- ⑤ 显存「谷值漂移」（判泄漏的真指标：抖动正常，谷值逐窗抬升 = 未记账增长）"
CSV="${2:-$HOME/lvllm/stress-run4/monitor.csv}"
if [ -f "$CSV" ]; then
  python3 - "$CSV" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
g = [(r["ts"], int(r["gpu_mem_max"])) for r in rows if r.get("gpu_mem_max")]
if not g:
    print("  (无数据)"); raise SystemExit
win = 120  # 10 分钟 @5s
buckets = [g[i:i+win] for i in range(0, len(g), win)]
print("  窗口(10min)      谷      峰      末   样本")
for b in buckets[-6:]:
    print("  %s  %6d  %6d  %6d  %4d" % (b[0][0][11:16], min(x[1] for x in b), max(x[1] for x in b), b[-1][1], len(b)))
mins = [min(x[1] for x in b) for b in buckets if len(b) >= 12]
if len(mins) >= 2:
    print("  谷值序列: %s" % " → ".join(str(m) for m in mins))
    print("  谷值变化: %+d MiB（正=地板抬升=可疑；负/零=有归还）" % (mins[-1] - mins[0]))
print("  全程: n=%d 首=%d 末=%d 峰=%d 谷=%d" % (len(g), g[0][1], g[-1][1], max(x[1] for x in g), min(x[1] for x in g)))
PYEOF
else
  echo "  ($CSV 不存在)"
fi

echo "--- ④ 致命信号（应 0）"
n=$(grep -icE "Xid|illegal memory access|out of memory" "$L" 2>/dev/null || true)
echo "  命中行数=${n:-0}"
[ "${n:-0}" -gt 0 ] && grep -inE "Xid|illegal memory access|out of memory" "$L" | tail -3 | cut -c1-200

echo "--- ⑥ xtu-mem-cleaner（r79+：每 300 s 后台回收，跳过 CUDA Graph 捕获期）"
_cl=$(grep -ohE 'xtu-guard.{0,150}|empty_cache 已执行.{0,60}|软上限.{0,80}|硬上限.{0,80}|xtu-mem-cleaner.{0,120}' "$L" 2>/dev/null | tail -6)
if [ -n "$_cl" ]; then
  echo "$_cl"
  echo "  次数: guard=$(grep -c 'xtu-guard' "$L" 2>/dev/null || echo 0) empty_cache=$(grep -c 'empty_cache 已执行' "$L" 2>/dev/null || echo 0)"
else
  echo "  (本日志暂无 cleaner 行：可能未满首个 300 s，或该版本无此埋点)"
fi

echo "--- ⑦ 智力体检（每 10 分钟一题，最近 5 条）"
P="$HOME/lvllm/stress-run4/intel_probe.jsonl"
if [ -f "$P" ]; then
  python3 - "$P" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip().startswith("{")]
for r in rows[-5:]:
    print(f"  {r.get('ts')} {r.get('item'):<16} {'PASS' if r.get('ok') else 'FAIL'} "
          f"{r.get('latency_s')}s {str(r.get('answer_head') or r.get('error'))[:44]}")
ok = sum(1 for r in rows if r.get("ok"))
print(f"  累计 {ok}/{len(rows)} 通过")
PY
else
  echo "  (尚无数据)"
fi
