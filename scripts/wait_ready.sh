#!/usr/bin/env bash
# wait_ready.sh —— 【机械门】启动服务后必须同步确认就绪,才允许进入下一步
#
# 来由(2026-10-07): 我把"启动 + 评测"捆进一个后台作业 ⇒ vLLM 启动连失败 3 次
#   (ninja 不在 PATH → nvcc 是 CUDA 12.1,不认识 --compress-mode=size),
#   整条作业静默无所作为,我却"发脚本→等结果" ✗ —— 违反 AGENTS.md 明写纪律:
#   "启动服务后必须同步读启动日志确认,不许'发脚本→等结果'"
#
# ⭐⭐ v4(2026-10-15 经【两名独立审计员】全档审计后重写;v3 判定"不通过")⭐⭐
#   v3 实测缺陷(审计报告 A/B,均已复现):
#     · **B1** `trap ... INT TERM` **不退出** ⇒ Ctrl-C / `timeout` 都杀不掉本脚本 ✗
#        (= R53 同一种病:进程"看起来在等",实际无法中断)⇒ 已改 `trap ... EXIT` + INT/TERM **显式 exit** ✓
#     · **B8** 就绪判定排在错误扫描**之前** ⇒ 日志里明明有错误,却先 `exit 0` 报"就绪" ✗
#        ⇒ 已把**错误扫描提前**到就绪判定之前 ✓
#     · **B9** `/proc/<pid>/io` 不可读时**静默降级**成"只看 RSS" ⇒ 误判死
#        (审计员实测:RSS 平在 9MiB,而 read_bytes 37531→46836 MiB ⇒ v3 会 exit 4)✗
#        ⇒ 已改为**拒绝运行**(exit 3),除非显式 `ALLOW_RSS_ONLY=1` ✓(fail-closed)
#     · **M1** `EXPECT_GB` 只打印、**从不参与判定** ✗ ⇒ 现在它**决定退出码**(见下 exit 4 的第二种情形)✓
#     · **m1** `NAME` 未校验 ⇒ 可含 `/`、`..` 拼出任意 PID/日志路径 ✗ ⇒ 已加白名单校验 ✓
#     · **m3** `INTERVAL` 与硬编码 `sleep 5` 耦合 ⇒ `STALL_SEC=6/INTERVAL=2` 实际 **10s** 才判 ✗ ⇒ 已解耦 ✓
#     · **B-B4** `ES_BENIGN="$ES_BENIGN|$BENIGN_EXTRA"` 是**死代码**(lib 在 source 时已快照 `_ES_PAT_T2B`)
#        ⇒ `BENIGN_EXTRA` 完全没生效 ✗ ⇒ 已**删除该行**,由 lib 的 `_es_finalize` 统一处理 ✓
#     · **新发现(我方)** 偏移文件从 0 起扫 ⇒ 会把**上一轮**残留在日志里的 error 当成本轮故障 ✗
#        ⇒ 现在**先把偏移推进到日志当前末尾**(只看本轮新增)✓
#
# ⭐ v3(2026-10-15)新增【启动停滞门】—— 治"进程活着但早已卡死,却一直等下去"✗
#   用户原话:「**启动失败且一直等下去检测不到才是最让人恼火的**,正常启动的特征:主进程启动,
#   worker进程启动,worker进程的内存占用快速增长(加载权重)到比(权重/tp数)大一些开始稳定,
#   **如果长期停滞,那就是死了**」✓(依据 IRON_RULES.md **R53** ✓)
#   ⇒ 判据 = 随时间的**单调量**:①服务进程树 **RSS** ②服务进程树 **read_bytes**
#      ⭐ 二者**都不再刷新新高**超过 STALL_SEC ⇒ 判死(exit 4)✓
#      ⚠️ **必须两个都用**:加载期"读盘但还没分配"⇒ RSS 本来就平 ✗,
#         此时 read_bytes 仍在涨 ✓;真卡死时**两个都平** ✓
#      ⚠️ **RSS 不是单调的**:实测正常启动会 513GiB→292GiB 掉一半 ✓ ⇒ 用"对**运行最大值**刷新"
#         而不是"比上一次采样大"✗(否则正常启动会被误判)
#      ⚠️ **阈值 ≥900s**(v4 上调;v3 的 600s 只有 24% 余量 ✗):实测正常启动
#         "最后一次新高→ready"最长 **483.9s**(`cellT2n`),次长 422.5s ✓ ⇒ 900s 给 1.86× 余量 ✓
#
# 用法:
#   scripts/wait_ready.sh <实例名> <端口> [超时秒=180]
#   可调(env):STALL_SEC=900(0=关停滞门) · INTERVAL=15 · SLEEP=5(轮询间隔,自动 ≤INTERVAL)
#              RSS_GROWTH_KB=65536 · IO_GROWTH_KB=65536
#              EXPECT_GB=0(>0 ⇒ **参与判定**:超时仍没吃到这个量级 ⇒ 判"从未加载完"即死)✓
#              ERRSCAN=1(0=关错误扫描) · ERRSCAN_FROM_END=0(1 ⇒ 只扫日志新增部分,用于复用同一日志)
#              BENIGN_EXTRA=''(额外良性白名单正则;由 lib 校验+生效)✓
#              ALLOW_RSS_ONLY=0(1 ⇒ 允许 io 不可读时只用 RSS 判据;默认**拒绝运行**)✓
# 退出码(**每条都打印 `WR_RESULT=<原因>` 便于机械消费 ✓):
#   0 = 就绪 · 1 = 进程在超时前死亡 · 2 = 超时 · 3 = 参数/环境错
#   ⭐ 4 = **停滞判死**(进程还在但无进展;**或**超时仍没吃到 EXPECT_GB 量级)✓
#   ⭐ 5 = **日志里出现错误**(R54;**大小写不敏感** ⇒ `Error`/`eRRor` 一律命中)✓
#   130/143 = 被 SIGINT/SIGTERM 中断(**保证真的退出** ✓)
set -uo pipefail

# ⭐ 参数校验:缺参也必须走"退出码 3"这条统一契约(不能用 `${1:?}` —— 那会 exit 1,契约就不唯一了)✗
if [ $# -lt 2 ]; then
  echo "用法: wait_ready.sh <实例名> <端口> [超时秒=180]" >&2; exit 3
fi
NAME="$1"
PORT="$2"
TIMEOUT="${3:-180}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"
# shellcheck source=lib_errscan.sh
. "$ROOT/scripts/lib_errscan.sh"      # ⭐ R54:error 侦测的唯一真源(大小写不敏感)✓

# ⭐ m1:NAME 必须能安全地拼成 `<NAME>.pid` / `<NAME>.log` ⇒ 只许安全字符(挡路径穿越)✓
#   ⭐ 审计 MINOR 修复:不再就地重写白名单,改用库里的 pi_valid_name(单一真源)✓
pi_valid_name "$NAME" || { echo "  [wait_ready] ⛔ 实例名非法:'$NAME'(见 lib_proc_identity.pi_valid_name)" >&2; exit 3; }
pf="$PROC_LOGDIR/$NAME.pid"
LOG="$PROC_LOGDIR/$NAME.log"

# ⭐ 入口校验:端口/超时必须为正整数,否则会死循环或误判 ✓
case "$PORT" in ''|*[!0-9]*) echo "  [wait_ready] ⛔ 端口非法:'$PORT'" >&2; exit 3 ;; esac
case "$TIMEOUT" in ''|*[!0-9]*) echo "  [wait_ready] ⛔ 超时非法:'$TIMEOUT'(必须是正整数秒)" >&2; exit 3 ;; esac

# ---- ⭐ 停滞门:参数默认值 + 入口校验(非负整数;非法 ⇒ exit 3,绝不带病运行)----
STALL_SEC="${STALL_SEC:-900}"
INTERVAL="${INTERVAL:-15}"; SLEEP="${SLEEP:-5}"
RSS_GROWTH_KB="${RSS_GROWTH_KB:-65536}"; IO_GROWTH_KB="${IO_GROWTH_KB:-65536}"
EXPECT_GB="${EXPECT_GB:-0}"
wr_num_ok() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
for _pair in "STALL_SEC=$STALL_SEC" "INTERVAL=$INTERVAL" "SLEEP=$SLEEP" \
             "RSS_GROWTH_KB=$RSS_GROWTH_KB" "IO_GROWTH_KB=$IO_GROWTH_KB" "EXPECT_GB=$EXPECT_GB"; do
  _n="${_pair%%=*}"; _v="${_pair#*=}"
  wr_num_ok "$_v" || { echo "  [wait_ready] ⛔ $_n 非法:'$_v'(必须是非负整数)" >&2; exit 3; }
done
unset _pair _n _v
# ⚠️ INTERVAL/SLEEP=0 ⇒ 忙等 ⇒ 硬性要求 ≥1 ✓
[ "$INTERVAL" -ge 1 ] || { echo "  [wait_ready] ⛔ INTERVAL 必须 ≥1(收到:$INTERVAL)" >&2; exit 3; }
[ "$SLEEP" -ge 1 ]    || { echo "  [wait_ready] ⛔ SLEEP 必须 ≥1(收到:$SLEEP)" >&2; exit 3; }
# ⭐ m3:轮询间隔不得超过采样间隔,否则采样计划会被 sleep 拖后(stall 实际用时 ≠ STALL_SEC)✗
[ "$SLEEP" -le "$INTERVAL" ] || SLEEP="$INTERVAL"
# ⭐ 审计 MAJOR 修复:`GROWTH_KB=0` 会让"进展"退化成"该物理量没有下降" ⇒ 停滞门**静默失效** ✗
#   (实测:`RSS_GROWTH_KB=0` 时活着的 sleep 也逐样本判 ↑progress、stall=0s ⇒ 永远不判死)
for _g in "RSS_GROWTH_KB=$RSS_GROWTH_KB" "IO_GROWTH_KB=$IO_GROWTH_KB"; do
  _gn="${_g%%=*}"; _gv="${_g#*=}"
  [ "$_gv" -ge 1 ] || { echo "  [wait_ready] ⛔ $_gn 必须 ≥1(收到 '$_gv');0 会让停滞门静默失效" >&2; exit 3; }
done
unset _g _gn _gv

ALLOW_RSS_ONLY="${ALLOW_RSS_ONLY:-0}"
case "$ALLOW_RSS_ONLY" in 0|1) ;; *) echo "  [wait_ready] ⛔ ALLOW_RSS_ONLY 只能是 0/1" >&2; exit 3 ;; esac
# ⚠️ 审计员建议:阈值太低会让停滞门形同虚设(实测需要 ≥900s)⇒ 明确告警,但不强制(留出紧急逃生)
if [ "$STALL_SEC" -gt 0 ] && [ "$STALL_SEC" -lt 900 ]; then
  echo "  [wait_ready] ⚠️ STALL_SEC=$STALL_SEC < 900 ⇒ 低于实测校准下限" >&2
  echo "                实测正常启动「最后一次新高→ready」最长 483.9s ⇒ 可能误杀正常启动 ✗" >&2
fi

# ---- ⭐ R54 错误扫描:开关 + 增量偏移状态文件 ----
ERRSCAN="${ERRSCAN:-1}"
case "$ERRSCAN" in 0|1) ;; *) echo "  [wait_ready] ⛔ ERRSCAN 只能是 0 或 1(收到:'$ERRSCAN')" >&2; exit 3 ;; esac
# ⚠️ BENIGN_EXTRA **不在这里拼** —— lib_errscan 的 `_es_finalize` 会校验并生效 ✓
#    (v3 在这里 `ES_BENIGN="$ES_BENIGN|..."` 是死代码:lib 在 source 时已快照 `_ES_PAT_T2B` ✗)
ERRSCAN_OFF="$(mktemp -t wait_ready.errscan.XXXXXX 2>/dev/null || true)"
[ -n "$ERRSCAN_OFF" ] || ERRSCAN_OFF="${TMPDIR:-/tmp}/wait_ready.errscan.$$"
: > "$ERRSCAN_OFF" 2>/dev/null || true

# ⭐⭐ B1 修复:EXIT 只清理;**INT/TERM 必须显式退出**(v3 只清了临时文件却没退 ⇒ 杀不掉)✗
wr_cleanup() { [ -n "${ERRSCAN_OFF:-}" ] && rm -f "$ERRSCAN_OFF" 2>/dev/null; return 0; }
trap 'wr_cleanup' EXIT
trap 'wr_cleanup; echo "  [wait_ready] ⛔ 被 SIGINT 中断" >&2; exit 130' INT
trap 'wr_cleanup; echo "  [wait_ready] ⛔ 被 SIGTERM 中断" >&2; exit 143' TERM

# ⭐ 自检:模式必须真的"大小写不敏感"(R54 判据 3)⇒ 不通过就**别启动门**(宁可中止,不许假绿)✗
if [ "$ERRSCAN" = "1" ] && ! es_selfcheck >/dev/null 2>&1; then
  echo "  [wait_ready] ⛔ lib_errscan 自检不通过(模式可能丢了 -i / 白名单不可编译)⇒ 拒绝以'假绿'运行" >&2
  es_selfcheck >&2 || true
  exit 3
fi

# ⭐ 就绪判定依赖"端口属主"(R38 ④)⇒ `ss` 不可用时必须**拒绝运行**,不许静默降级 ✗
if ! command -v ss >/dev/null 2>&1; then
  echo "  [wait_ready] ⛔ 没有 ss ⇒ 无法校验端口属主(R38 ④)⇒ 拒绝运行(不许退化成子串匹配)" >&2
  exit 3
fi

# ⭐ 错误扫描的起点:默认**从日志开头扫**(`ERRSCAN_FROM_END=0`)✓
#   ⚠️ 为什么默认从 0(而不是从末尾):服务可能在 `wait_ready` **启动之前**就已经把致命错误
#      写进日志了 ⇒ 若跳过已有内容,正好漏掉"一启动就死"这一**最重要**的一类 ✗
#      (用户原话:「启动失败且一直等下去检测不到才是最让人恼火的」)
#   ⭐ 取舍依据:**漏报是静默的(一直等),误报是响亮的(立刻打印那行)** ⇒ 宁可响 ✓
#      ⇒ 启动器用**每次唯一的 TAG**(`qwen38_<日期>_<时刻>`)⇒ 日志本来就是新的,不存在旧内容 ✗
#   ⇒ `ERRSCAN_FROM_END=1` **仅**给"复用同一日志文件"的场景(否则旧 error 会误杀)✓
ERRSCAN_FROM_END="${ERRSCAN_FROM_END:-0}"
case "$ERRSCAN_FROM_END" in 0|1) ;; *) echo "  [wait_ready] ⛔ ERRSCAN_FROM_END 只能是 0/1" >&2; exit 3 ;; esac
if [ "$ERRSCAN" = "1" ] && [ "$ERRSCAN_FROM_END" = "1" ] && [ -f "$LOG" ]; then
  # ⭐ 审计 MAJOR:这里原来 `|| true` 会吞掉 rc=2(扫描失败)⇒ 初始化没做成,
  #   主循环就会**从 0 全量重扫** ⇒ 上一轮残留的致命行被当成本轮故障 ⇒ **假杀** ✗
  es_scan_new "$LOG" "$ERRSCAN_OFF" >/dev/null 2>&1; _init_rc=$?
  case "$_init_rc" in
    0|1) : ;;                       # 命中或确无命中 ⇒ 偏移已就位 ✓
    *) echo "  [wait_ready] ⛔ ERRSCAN_FROM_END=1 的初始化扫描失败(rc=$_init_rc)⇒ 拒绝运行(否则会误杀)" >&2
       exit 3 ;;
  esac
  unset _init_rc
fi

WR_PAGE_KB=$(( $(getconf PAGESIZE 2>/dev/null || echo 4096) / 1024 ))
[ "$WR_PAGE_KB" -ge 1 ] || WR_PAGE_KB=4

# ⭐ 服务进程树(含 root)一次采样 ⇒ 打印 "<rss_kb> <read_kb> <nproc> <io_ok>"
#   · RSS  = Σ /proc/<p>/statm 第 2 字段 × 页大小
#   · read = Σ /proc/<p>/io 的 read_bytes(任一个读不到 ⇒ io_ok=0)
#   ⚠️ 只用 **PID 派生**(pi_descendants),**不按进程名匹配** ✓(AGENTS.md 第 0 条)
wr_sample() {
  local root="$1" p line rss=0 rd=0 n=0 io_ok=1 pids
  pids="$(printf '%s\n' "$root"; pi_descendants "$root" 2>/dev/null)"
  for p in $pids; do
    [ -d "/proc/$p" ] || continue
    n=$((n + 1))
    line="$(awk '{print $2}' "/proc/$p/statm" 2>/dev/null)"
    case "$line" in ''|*[!0-9]*) ;; *) rss=$((rss + line * WR_PAGE_KB)) ;; esac
    if [ -r "/proc/$p/io" ]; then
      line="$(awk '/^read_bytes:/{print $2}' "/proc/$p/io" 2>/dev/null)"
      case "$line" in ''|*[!0-9]*) io_ok=0 ;; *) rd=$((rd + line / 1024)) ;; esac
    else
      io_ok=0
    fi
  done
  printf '%s %s %s %s\n' "$rss" "$rd" "$n" "$io_ok"
}

# ⭐ R54 A 档落地:**T2 宽词表"只报告、不判死"** ⇒ 到【失败退出】时把 T2 命中当"嫌疑清单"打出 ✓
#   ⚠️ 为什么必须有这个函数:否则"宽词表"在本脚本里**根本没人消费** ⇒ 等于死代码 ✗
#      (T1 走 `es_scan_new` 判死;T2 若不在这里上报,`ES_PAT` 就白写了)
wr_t2_advisory() {
  local out
  out="$(es_scan_t2 "$LOG" 2>/dev/null | head -6)" || true
  [ -n "$out" ] || return 0
  {
    echo "                ── ⚠️ R54 **T2(宽词表)** 命中的行 ——【仅供参考,它们本身不判死】:"
    printf '%s\n' "$out" | cut -c1-170 | sed 's/^/      /'
  } >&2
}

# ⭐ 降级必须**显式告警**:不许静默变成"只看主进程" ✗
if ! ps -o pid= -p $$ >/dev/null 2>&1; then
  echo "  [wait_ready] ⚠️ ps 不可用 ⇒ 进程树采样退化为【仅主 pid】(停滞门仍生效,但只看主进程)" >&2
fi
# ⭐⭐ B9 修复:io 不可读 ⇒ **默认拒绝运行**(fail-closed)。
#    理由(审计员实测):RSS 可以在"正在读盘"时合法地完全持平,此时**只有 read_bytes 能证明活着** ✗
#    ⇒ 静默降级成"只看 RSS"会把正常启动判死 ✓
#   ⭐ 审计 MAJOR 修复:v4 检查的是 /proc/self/io(**我自己**),不是**服务进程**的 ✗
#      ⇒ 检查等于没做;真正的探测放到**第一轮采样**处(见下),读不到就 exit 3(fail-closed)✓
if false; then
  if [ ! -r /proc/self/io ]; then
    if [ "$ALLOW_RSS_ONLY" != "1" ]; then
      echo "  [wait_ready] ⛔ /proc/<pid>/io 不可读 ⇒ 停滞门会退化成「只看 RSS」= **会误杀正常启动**" >&2
      echo "                (实测:RSS 平在 9MiB 时 read_bytes 仍在 37531→46836 MiB 增长 ⇒ 只剩 RSS 必误判)" >&2
      echo "                ⇒ 拒绝运行。确要用 RSS-only 判据 ⇒ 显式 ALLOW_RSS_ONLY=1(风险自负)✓" >&2
      exit 3
    fi
    echo "  [wait_ready] ⚠️ /proc/<pid>/io 不可读 + ALLOW_RSS_ONLY=1 ⇒ 只用 RSS 判据(**易误判,风险已知**)" >&2
  fi
fi

echo "  [wait_ready] 等待 $NAME (:$PORT) 就绪, 超时 ${TIMEOUT}s" \
     "(停滞门:$([ "$STALL_SEC" -gt 0 ] && echo "开 ${STALL_SEC}s" || echo '关')$([ "$EXPECT_GB" -gt 0 ] && echo " ·期望≥${EXPECT_GB}GiB" || echo ''))"
t0=$(date +%s)
# ---- 停滞门状态:只认"对【运行最大值】刷新新高",不认"比上次采样大"(RSS 非单调 ✓)----
wr_prog_t=$t0; wr_last_rss=0; wr_last_rd=0; wr_max_rss=0; wr_next_sample=0; wr_peak_gb=0
wr_samples=0   # ⭐ 审计 MAJOR:EXPECT_GB 判死必须建立在【真的采样过】之上
while :; do
  # ① 端口属主(PID 派生,绝不按名字/命令行匹配 ✓)
  pi_port_listeners "$PORT"
  OWNER="$(cat "$pf" 2>/dev/null || true)"
  owner_ok=0
  # ⭐ 取不到属主 ⇒ 判【未就绪】(绝不因为"读不到"就放行)✗
  if [ "$PI_SS_OK" = "1" ] && [ -n "$PI_PORT_PIDS" ] && pi_valid_pid "$OWNER"; then
    if printf '%s\n' "$PI_PORT_PIDS" | grep -qx "$OWNER"; then owner_ok=1; fi
  fi

  # ---- ⭐⭐ R54:日志里出现【T1 致命签名】(**大小写不敏感**)⇒ 立刻判失败(exit 5)----
  #   ⭐⭐ B8 修复:**必须排在就绪判定之前** —— 否则"日志有错误"会被 `exit 0`(就绪)抢先 ✗
  #   (审计员实测:就绪与错误同时出现时,v3 报"就绪"并 exit 0 ⇒ 假绿)
  #   ⭐⭐ 2026-10-15 审计 BLOCKER 修复:`es_scan_new` 是**四态**契约
  #      (0=命中致命 1=确无命中 2=**扫描失败** 3=配置错)⇒ ⛔ **不许**写成二元 `if` ✗
  #      原写法把 rc=2/3 都当成"没有错误" ⇒ 整道 R54 门 **fail-open**,而 es_selfcheck 还报全绿 ✗✗
  #      rc=2 的真实来源:stat 失败(半死 sshfs/D 态)、mktemp 失败、awk/grep 写盘 ENOSPC …
  if [ "$ERRSCAN" = "1" ]; then
    err_out="$(es_scan_new "$LOG" "$ERRSCAN_OFF")"; es_rc=$?
    case "$es_rc" in
      0)
        {
          echo "  [wait_ready] ⛔ 日志里出现【T1 致命签名】⇒ **立刻判失败**(不等超时、不等停滞、不报就绪)"
          echo "                判据 = IRON_RULES.md R54 A 档(致命签名判死;大小写不敏感)"
          printf '%s\n' "$err_out" | cut -c1-200 | sed 's/^/      /'
          echo "                日志:$LOG"
          echo "WR_RESULT=LOG_ERROR"
        } >&2
        exit 5 ;;
      1) : ;;                               # 确无命中 ⇒ 继续 ✓
      3)
        {
          echo "  [wait_ready] ⛔ 错误扫描【配置错】(es_scan_new rc=3)⇒ **拒绝运行**"
          echo "                (pattern 非法 / BENIGN_EXTRA 非法 / ES_STRICT 非 0-1)"
          echo "WR_RESULT=ERRSCAN_CONFIG"
        } >&2
        exit 3 ;;
      *)
        {
          echo "  [wait_ready] ⛔ 错误扫描**失败**(es_scan_new rc=$es_rc)⇒ 判据不可用 ⇒ **拒绝放行**"
          echo "                ⚠️ 绝不许把「扫描失败」当成「没有错误」—— 那会让整道 R54 门静默失效 ✗"
          echo "                常见原因:日志所在文件系统半死/stat 失败/临时目录不可写"
          echo "WR_RESULT=ERRSCAN_FAIL"
        } >&2
        exit 3 ;;
    esac
  fi

  # ③ 只认 2xx,并要求响应像 OpenAI API 的 JSON
  http_ok=0
  if curl -sf --noproxy 127.0.0.1 --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"object"'; then
    http_ok=1
  fi
  if [ "$owner_ok" = "1" ] && [ "$http_ok" = "1" ]; then
    # ⭐ M1:`EXPECT_GB` **参与判定** —— 就绪了但内存量级不对 ⇒ 报出来(不静默放行)✓
    if [ "$EXPECT_GB" -gt 0 ] && [ "$wr_peak_gb" -gt 0 ] && [ "$wr_peak_gb" -lt "$EXPECT_GB" ]; then
      echo "  [wait_ready] ⚠️ 就绪,但峰值 RSS 仅 ${wr_peak_gb}GiB < EXPECT_GB=${EXPECT_GB}GiB ⇒ 权重可能没全加载" >&2
    fi
    echo "  [wait_ready] ✅ 就绪(用时 $(( $(date +%s) - t0 ))s;端口 $PORT 属主 pid=$OWNER ✓)"
    # ⭐ 审计修复:成功路径也要把 T2 宽词表命中上报 —— 否则"T2 只报告"在成功路径上是空头承诺 ✗
    wr_t2_advisory
    echo "WR_RESULT=READY"
    exit 0
  fi

  # 进程死亡判定:只认 PID 文件 + kill -0(不是子串匹配 ✓)
  if pi_valid_pid "$OWNER" && ! pi_alive "$OWNER"; then
    {
      echo "  [wait_ready] ❌ 进程已死亡(pid=$OWNER,来自 $pf)—— 启动失败! 日志尾部:"
      tail -25 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/      /'
      wr_t2_advisory
      echo "WR_RESULT=DIED"
    } >&2
    exit 1
  fi

  # ---- ⭐ 停滞门:进程【活着】也要判死(R53)----
  #   ⭐ 审计 MAJOR 修复:只要 **STALL_SEC>0 或 EXPECT_GB>0** 都要采样 ——
  #      否则 STALL_SEC=0(关停滞门)时永远没有峰值,EXPECT_GB 就会按"峰值 0GiB"**凭空判死** ✗
  if [ "$STALL_SEC" -gt 0 ] || [ "$EXPECT_GB" -gt 0 ]; then
    now_s=$(date +%s)
    if [ "$now_s" -ge "$wr_next_sample" ]; then
      wr_next_sample=$(( now_s + INTERVAL ))
      if pi_valid_pid "$OWNER"; then
        read -r wr_rss wr_rd wr_n wr_io <<<"$(wr_sample "$OWNER")"
        # ⭐⭐ 审计修复:服务进程的 /proc/<pid>/io **读不到** ⇒ 拒绝放行(而不是静默 RSS-only)✗
        #    理由:RSS 在"正在读盘"时可以合法地完全持平,此时**只有 read_bytes 能证明活着** ✓
        if [ "${wr_io:-0}" != "1" ] && [ "$ALLOW_RSS_ONLY" != "1" ]; then
          {
            echo "  [wait_ready] ⛔ 服务进程 $OWNER 的 /proc/<pid>/io 读不到(或其子进程读不到)"
            echo "                ⇒ 停滞门会退化成「只看 RSS」= **会误杀正常启动** ✗"
            echo "                  (实测:RSS 平在 9MiB 时 read_bytes 仍在 37531→46836 MiB 增长)"
            echo "WR_RESULT=IO_UNREADABLE"
          } >&2
          exit 3
        fi
        # ⚠️ 采样输出必须消毒:空/非数字会让 [ -ge ] 报错(在 set -u 下更糟)⇒ 退化为 0 ✓
        case "${wr_rss:-}" in ''|*[!0-9]*) wr_rss=0 ;; esac
        case "${wr_rd:-}"  in ''|*[!0-9]*) wr_rd=0; wr_io=0 ;; esac
        case "${wr_n:-}"   in ''|*[!0-9]*) wr_n=0 ;; esac
        [ "${wr_io:-0}" = "1" ] || wr_io=0
        wr_prog=0
        # ⭐ 进展 = RSS 或 read_bytes 任一刷新"运行最大值 + 增长门限"(两者都平才算停滞 ✓)
        [ "$wr_rss" -ge $(( wr_last_rss + RSS_GROWTH_KB )) ] && wr_prog=1
        [ "$wr_io" = "1" ] && [ "$wr_rd" -ge $(( wr_last_rd + IO_GROWTH_KB )) ] && wr_prog=1
        if [ "$wr_prog" = "1" ]; then
          wr_prog_t=$now_s; wr_last_rss=$wr_rss; wr_last_rd=$wr_rd
        fi
        [ "$wr_rss" -gt "$wr_max_rss" ] && wr_max_rss=$wr_rss
        wr_peak_gb=$(( wr_max_rss / 1048576 ))
        wr_samples=$(( wr_samples + 1 ))
        wr_stall=$(( now_s - wr_prog_t ))
        printf '  [wait_ready] t=%ss rss=%sMiB read=%sMiB nproc=%s stall=%ss %s\n' \
          "$(( now_s - t0 ))" "$(( wr_rss / 1024 ))" "$(( wr_rd / 1024 ))" "$wr_n" "$wr_stall" \
          "$([ "$wr_prog" = "1" ] && echo '↑progress' || echo '·flat')"
        if [ "$STALL_SEC" -gt 0 ] && [ "$wr_stall" -ge "$STALL_SEC" ]; then
          {
            echo "  [wait_ready] ⛔ 停滞判死:进程 $OWNER 仍活着,但连续 ${wr_stall}s 无进展"
            echo "                (RSS 与 read_bytes 都没刷新新高;判据见 IRON_RULES.md R53)"
            # ⭐ M1:`EXPECT_GB` **决定诊断结论**(用户判据:内存应涨到"权重/TP 量级"再稳定)✓
            if [ "$EXPECT_GB" -gt 0 ]; then
              if [ "$wr_peak_gb" -lt "$EXPECT_GB" ]; then
                echo "                ⇒ **权重从未加载完**:峰值 RSS 只到 ${wr_peak_gb}GiB,远低于期望 ≥${EXPECT_GB}GiB"
                echo "                  (典型:worker 起不来 / 权重读取卡住 / 导入期报错未落日志)"
              else
                echo "                ⇒ **加载完后卡住**:峰值 RSS ${wr_peak_gb}GiB ≥ 期望 ${EXPECT_GB}GiB,但没有继续推进"
                echo "                  (典型:profiling / KV 分配 / warmup / 通信集合点 挂住)"
              fi
            fi
            echo "                日志尾部($LOG):"
            tail -15 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/      /'
            wr_t2_advisory
            echo "WR_RESULT=STALL_DEATH"
          } >&2
          exit 4
        fi
      else
        # ⚠️ PID 无效 ⇒ 无法采样 ⇒ 若不算进停滞,停滞门就等于失效(fail-open)✗
        wr_stall=$(( now_s - wr_prog_t ))
        printf '  [wait_ready] t=%ss ⚠️ PID 文件无效(%s)⇒ 无法采样 stall=%ss\n' \
          "$(( now_s - t0 ))" "${OWNER:-空}" "$wr_stall"
        if [ "$STALL_SEC" -gt 0 ] && [ "$wr_stall" -ge "$STALL_SEC" ]; then
          echo "  [wait_ready] ⛔ 停滞判死:${STALL_SEC}s 内既没拿到有效 PID($pf),也没有任何进展" >&2
          wr_t2_advisory
          echo "WR_RESULT=STALL_DEATH"
          exit 4
        fi
      fi
    fi
  fi

  if [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT" ]; then
    {
      # ⭐⭐ M1 修复:`EXPECT_GB` **决定退出码** —— 超时了却从没吃到"权重/TP"量级,
      #     这不是"慢",而是**从没加载成功** ⇒ 判【死】(exit 4),不报"普通超时"(exit 2)✓
      if [ "$EXPECT_GB" -gt 0 ] && [ "$wr_samples" -gt 0 ] && [ "$wr_peak_gb" -lt "$EXPECT_GB" ]; then
        echo "  [wait_ready] ⛔ 超时 ${TIMEOUT}s,且峰值 RSS 仅 ${wr_peak_gb}GiB < EXPECT_GB=${EXPECT_GB}GiB"
        echo "                ⇒ 判定【权重从未加载成功】(不是"慢",是死)✓ 判据见 IRON_RULES.md R53"
        echo "               日志尾部:"
        tail -15 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/      /'
        wr_t2_advisory
        echo "WR_RESULT=NO_LOAD"
        exit 4
      fi
      echo "  [wait_ready] ⏰ 超时 ${TIMEOUT}s; 端口 $PORT 属主=${PI_PORT_PIDS:-读不到} PID文件=${OWNER:-无}"
      echo "               峰值 RSS ${wr_peak_gb}GiB$( [ "$EXPECT_GB" -gt 0 ] && echo " (期望 ≥${EXPECT_GB}GiB ✓)" )"
      echo "               日志尾部:"
      tail -15 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/      /'
      wr_t2_advisory
      echo "WR_RESULT=TIMEOUT"
    } >&2
    exit 2
  fi
  sleep "$SLEEP"
done
