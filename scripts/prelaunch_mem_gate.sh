#!/usr/bin/env bash
# prelaunch_mem_gate.sh — 起测试实例前的【宿主内存 / 进程 / 显存】门禁(2026-10-19 用户明令)。
#
# 为什么要有它:2026-10-19 远端被两个调试实例各 ~472 GiB anon-rss 拖到 OOM 死机。
# ⇒ 以后**每次**起实例前先跑本脚本,退出码 0 才许启动。**缺省拒绝**(fail-closed)。
#
# 用法(必须显式给出 need 与 gpus,脚本不给默认值 ⇒ 缺参数直接拒绝):
#   bash scripts/prelaunch_mem_gate.sh --need-gib 500 --gpus 1,2 --tag qfn_tp2_pin16
#   bash scripts/prelaunch_mem_gate.sh --need-gib 900 --gpus 1,2 \
#        --allow-pid 12345            # 仅放行**明确列出的** PID(生产实例须这样定向排除)
#
# 判据(全部满足才 rc=0):
#   ① MemAvailable >= need_gib
#   ② 除 --allow-pid 列出的以外,零 vllm/EngineCore/Worker 进程
#   ③ 指定 GPU 的 used MiB <= gpu_max_mib(默认 512)
#   ④ 同名 <tag>.pid 不是活进程
#
# 退出码:0 通过 / 3 内存不足 / 4 有实例在跑 / 5 显存不干净 / 2 参数或环境错
set -euo pipefail

NEED_GIB=""
GPUS=""
GPU_MAX_MIB=512
TAG=""
ALLOW_PIDS=""

die() { local rc="$1"; shift; echo "[gate] ⛔ $*" >&2; exit "$rc"; }
say() { echo "[gate] $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --need-gib) NEED_GIB="${2:-}"; shift 2 || die 2 "--need-gib 缺值" ;;
    --gpus) GPUS="${2:-}"; shift 2 || die 2 "--gpus 缺值" ;;
    --gpu-max-mib) GPU_MAX_MIB="${2:-}"; shift 2 || die 2 "--gpu-max-mib 缺值" ;;
    --tag) TAG="${2:-}"; shift 2 || die 2 "--tag 缺值" ;;
    --allow-pid) ALLOW_PIDS="${2:-}"; shift 2 || die 2 "--allow-pid 缺值" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die 2 "未知参数: $1" ;;
  esac
done

# ---- 参数消毒:缺参数/非整数一律拒绝(绝不 fail-open)----
[ -n "$NEED_GIB" ] || die 2 "必须显式给 --need-gib(算法:预期宿主×1.5+200GiB)"
[ -n "$GPUS" ] || die 2 "必须显式给 --gpus(例如 1,2 或 0)"
case "$NEED_GIB" in ''|*[!0-9]*) die 2 "--need-gib 必须是整数('$NEED_GIB')" ;; esac
case "$NEED_GIB" in 0|0*) die 2 "--need-gib 必须 >0('$NEED_GIB')" ;; esac
case "$GPU_MAX_MIB" in ''|*[!0-9]*) die 2 "--gpu-max-mib 必须是整数('$GPU_MAX_MIB')" ;; esac
case "$GPUS" in ''|,*|*,|*,,*|*[!0-9,]*) die 2 "--gpus 非法('$GPUS';只接受逗号分隔的非负整数)" ;; esac
if [ -n "$ALLOW_PIDS" ]; then
  case "$ALLOW_PIDS" in ''|,*|*,|*,,*|*[!0-9,]*) die 2 "--allow-pid 非法('$ALLOW_PIDS')" ;; esac
  [ -n "$TAG" ] || die 2 "使用 --allow-pid 时必须同时给 --tag(便于审计)"
fi

# 自己的 PID 与其祖先链:即使 pgrep -f 误命中,也必须排除
_self_chain=" $$ "
_p=$$
for _ in $(seq 1 32); do
  _p="$(awk '{print $4}' "/proc/$_p/stat" 2>/dev/null || true)"
  case "${_p:-}" in ''|*[!0-9]*) break ;; esac
  [ "$_p" = "0" ] && break
  _self_chain="$_self_chain$_p "
done

# ---- ① 可用内存 ----
_avail="$(LC_ALL=C awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || true)"
case "${_avail:-}" in ''|*[!0-9]*) die 2 "读不到 MemAvailable(值='${_avail:-}')" ;; esac
say "MemAvailable=${_avail} GiB  need=${NEED_GIB} GiB"
[ "$_avail" -ge "$NEED_GIB" ] || die 3 "可用内存不足:${_avail} < ${NEED_GIB} GiB"

# ---- ② 进程(只判有无;--allow-pid 之外一律拒绝)----
# ⚠️ pgrep 在"零匹配"时 rc=1 ⇒ 在 pipefail 下会直接终止脚本(审计 P0)⇒ 必须 `|| true` 先捕获
# ⚠️ 必须区分 pgrep 的 rc:0=有匹配 / 1=无匹配 / >=2=枚举失败或 pgrep 缺失
#    ⇒ 后者绝不能当成"0 个进程"放行(fail-open)✗
set +e
_pg="$(pgrep -af 'vllm\.entrypoints\.openai\.api_server|VLLM::' 2>/dev/null)"
_pg_rc=$?
set -e
[ "$_pg_rc" -le 1 ] || die 2 "pgrep 枚举失败(rc=${_pg_rc})⇒ 无法确认是否有实例,拒绝放行"
_n_all=0
_bad=""
if [ -n "$_pg" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    pid="${line%% *}"
    case "${pid:-}" in ''|*[!0-9]*) continue ;; esac
    case "$_self_chain" in *" $pid "*) continue ;; esac
    _n_all=$(( _n_all + 1 ))
    if [ -n "$ALLOW_PIDS" ]; then
      case ",$ALLOW_PIDS," in *",$pid,"*) continue ;; esac
    fi
    _bad="$_bad$line"$'\n'
  done <<< "$_pg"
fi
if [ -n "$_bad" ]; then
  echo "[gate] 以下进程仍在运行(需先按 PID 逐个确认并停掉):" >&2
  printf '%s' "$_bad" | sed 's/^/  /' >&2
  die 4 "检测到未放行的 vllm 相关进程(${_n_all} 个总数)⇒ 拒绝启动第二个实例"
fi
say "vllm 相关进程: ${_n_all}(allow_pids='${ALLOW_PIDS}')"

# ---- ③ 目标 GPU 显存(必须真的检查到 ≥1 张)----
command -v nvidia-smi >/dev/null 2>&1 || die 2 "找不到 nvidia-smi ⇒ 无法确认显存,拒绝放行"
_gpu_tbl="$(timeout 5 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
[ -n "$_gpu_tbl" ] || die 2 "nvidia-smi 无输出 ⇒ 无法确认显存,拒绝放行"
_checked=0
for g in ${GPUS//,/ }; do
  case "$g" in ''|*[!0-9]*) die 2 "GPU 编号非法('$g')" ;; esac
  _used="$(printf '%s\n' "$_gpu_tbl" | awk -F', *' -v i="$g" '$1==i{print $2; exit}')"
  case "${_used:-}" in ''|*[!0-9]*) die 2 "读不到 GPU${g} 的显存用量(表='$_gpu_tbl')" ;; esac
  say "GPU${g} used=${_used} MiB (上限 ${GPU_MAX_MIB})"
  [ "$_used" -le "$GPU_MAX_MIB" ] || die 5 "GPU${g} 仍有 ${_used} MiB 被占用(>${GPU_MAX_MIB})"
  _checked=$(( _checked + 1 ))
done
[ "$_checked" -gt 0 ] || die 2 "没有检查任何 GPU(--gpus='$GPUS')"

# ---- ④ 同名 PID 文件(活着的同名实例 = 冲突;僵尸/陈旧只告警)----
if [ -n "$TAG" ]; then
  case "$TAG" in */*|*' '*) die 2 "TAG 非法('$TAG')" ;; esac
  _pf="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dev-docs/report/tuning/logs/${TAG}.pid"
  if [ -f "$_pf" ]; then
    _pid="$(cat "$_pf" 2>/dev/null || true)"
    case "${_pid:-}" in
      ''|*[!0-9]*) say "⚠️ ${TAG}.pid 内容非法('${_pid:-}')⇒ 视为陈旧" ;;
      *)
        if [ -d "/proc/${_pid}" ]; then
          _st="$(awk '{print $3}' "/proc/${_pid}/stat" 2>/dev/null || true)"
          if [ "${_st:-}" = "Z" ]; then say "⚠️ ${TAG}.pid 指向僵尸(${_pid})⇒ 视为陈旧";
          else die 4 "同名实例 ${TAG} 仍活着(pid=${_pid})"; fi
        else say "⚠️ ${TAG}.pid 陈旧(pid=${_pid} 已不在)"; fi ;;
    esac
  fi
fi

say "✅ 门禁通过(${NEED_GIB} GiB / GPUS=${GPUS})"
