#!/usr/bin/env bash
# 生产实例死亡取证（**只读**：不杀进程、不改配置、不写服务目录）
#
# 用法:  bash scripts/triage_prod.sh [日志文件]
#        （不给日志文件 ⇒ 自动取 dev-docs/report/tuning/logs/ 下最新的 dsv41_prod_r*.log）
#
# 输出分区：① 服务健康 ② GPU/进程 ③ 日志错误分类 ④ 内核侧(Xid/OOM) ⑤ 内存历史 ⑥ 进程栈(可选)
# 设计约束：
#   * 每条外部命令都带 timeout，避免取证把前台拖住；
#   * 拿不到的项（如 dmesg 受限）**明确标注**，不静默跳过；
#   * ⛔ 不含任何 kill / 重启逻辑 —— 重启由调用者按 PID 单独执行（见 AGENTS 第 0 条）。
set -uo pipefail
PORT="${PORT:-8700}"
TREE="${XTU_TREE:-$HOME/lvllm/prod_20261027}"
LOGDIR="$TREE/dev-docs/report/tuning/logs"

LOG="${1:-}"
if [ -z "$LOG" ]; then
  LOG=$(ls -t "$LOGDIR"/dsv41_prod_r*.log 2>/dev/null | head -1)
fi

echo "================ TRIAGE $(date -Is) ================"
echo "LOG=$LOG"
[ -n "${LOG:-}" ] && [ -f "$LOG" ] || echo "⚠️ 没找到日志文件（继续采集其它项）"

echo
echo "--- ① 服务健康 ---"
code=$(timeout 5 curl -s -m 4 --noproxy 127.0.0.1 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "curl失败")
echo "health=$code  (200=活着, 000=不可用)"

echo
echo "--- ② GPU / 进程 ---"
timeout 8 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi 失败"
echo "-- 计算进程 --"
timeout 8 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
echo "-- vllm 相关进程（只列，不杀） --"
timeout 8 ps -eo pid,ppid,etime,stat,rss,args 2>/dev/null \
  | grep -E "[V]LLM::|[v]llm[.]entrypoints|[s]erve_v41" \
  | cut -c1-160

echo
echo "--- ③ 日志错误分类 ---"
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
  for pat in "illegal memory" "Xid" "OutOfMemory" "cancelled" "Traceback" "empty_cache.*失败" "CUDA error" "NCCL"; do
    n=$(timeout 8 grep -cE "$pat" "$LOG" 2>/dev/null || echo 0)
    printf '  %-22s %s\n' "$pat" "$n"
  done
  echo "-- 首次出现的错误行（带行号） --"
  timeout 8 grep -nE "illegal memory|OutOfMemory|cancelled|CUDA error|Traceback" "$LOG" 2>/dev/null | head -6 | cut -c1-200
  echo "-- 日志尾部 40 行 --"
  timeout 8 tail -40 "$LOG" | cut -c1-200
fi

echo
echo "--- ④ 内核侧（Xid / OOM killer） ---"
if timeout 6 dmesg 2>/dev/null | tail -400 > /tmp/xtu_triage_dmesg.$$ 2>/dev/null; then
  hits=$(grep -cE "Xid|Out of memory|oom-kill|Killed process" /tmp/xtu_triage_dmesg.$$ 2>/dev/null || echo 0)
  echo "最近 400 行内核日志中命中 $hits 条"
  grep -E "Xid|Out of memory|oom-kill|Killed process" /tmp/xtu_triage_dmesg.$$ 2>/dev/null | tail -8 | cut -c1-200
  rm -f /tmp/xtu_triage_dmesg.$$
else
  echo "⚠️ dmesg 不可读（kernel.dmesg_restrict=1 或权限不足）⇒ 该项缺失，需 root 才能补"
fi

echo
echo "--- ⑤ 内存历史（steady 行，最后 10 条） ---"
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
  timeout 8 grep -oE "steady cuda:[01] allocated=[0-9.]+ reserved=[0-9.]+ max_alloc=[0-9.]+ frag=[0-9.]+ free=[0-9.]+" "$LOG" \
    | grep -v "allocated=0.000" | tail -10
  echo "-- guard 兜底次数 --"
  timeout 8 grep -c "xtu-guard" "$LOG" 2>/dev/null || echo 0
fi

echo
echo "--- ⑥ 宿主内存 ---"
timeout 6 free -g 2>/dev/null | head -3

echo
echo "--- ⑦ 结论提示（人工判读） ---"
echo "  illegal memory access  ⇒ 历史主因(3h20m–4h)；查 §33.10–§33.17，需精确定位时用 GPU0+8层裁剪模型跑 sanitizer"
echo "  OutOfMemory            ⇒ 查 reserved/free 与 guard 计数，按 §33.10 内存配置"
echo "  cancelled              ⇒ 看 timeout 与对端 rank 的退出码"
echo "  Xid                    ⇒ GPU 硬件/驱动级错误，需记下 Xid 号再判"
echo "================ TRIAGE END ================"
