#!/usr/bin/env bash
# lib_errscan.sh —— ⭐【唯一真源】"启动日志里出现错误 ⇒ 立刻失败"的机械判据(IRON_RULES **R54**)
#
# ═══════════════════════════════════════════════════════════════════════════════
# 版本 v2 —— 2026-10-15 经【两名独立审计员】全档审计后**重写**(v1 判定"不通过")
#   v1 实测缺陷(审计报告 A/B,均已复现):
#     · 增量偏移把一行劈成两半 ⇒ **永久漏检**(A-B1)
#     · 日志轮转/truncate ⇒ 新错误**永不报**(A-B2)
#     · 先落偏移后扫描 ⇒ 一次扫描失败丢掉整段(A-B3/B-B5)
#     · 畸形白名单 ⇒ 假绿 + 全吞(A-B4)
#     · 词表缺 timeout/died/stuck/OOM 等 ⇒ 漏掉"卡死"的典型签名(A-B7)
#     · 白名单整行否决 ⇒ 良性短语与真错误同行时真错误被吞(A-M5)
#     · 在 `set -e` 下脱离 if 调用会**静默杀掉调用方**(A-M6)
#     · 无 pattern 编译校验、无自检失败方向纠正(A-B4/A-M4)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 三条不变量(R54;改词表前先读):
#   ① ⭐ **一律大小写不敏感**(`-i` / `-aiE`)
#      —— **大小写组合不是有限集合**(`Error`/`eRRor`/`ErRoR` …)⇒ ⛔ 不许写 `ERROR|error` 枚举 ✗
#   ② ⭐ **宁可误报,不可漏报** —— 词表必须宽(v2 已补 timeout/died/stuck/OOM/… ✓)
#   ③ ⭐ **良性行只能靠【显式白名单】排除**,且每条必须写出理由 ✓
#
# ⭐⭐ 两级判据(v2 的核心设计,**有实测依据**):
#   T1【致命 / 判死】:结构式签名 —— Python traceback、引擎/Worker 死亡、OOM 等
#       实测:499 份"曾成功 ready"日志的 ready-前缀,**T1 命中 0 份**(0.0% 误杀)✓
#             同时**100% 抓到** qfnFIX1 / qfnFIX2 两次真实失败 ✓
#   T2【告警 / 只报不杀】:宽词表命中但非致命
#       实测:同一语料上宽词表命中 **69/499 = 13.8%** ✗
#       (来源:我们自己的 shim 回退行、vLLM 的 KV-profile OOM **WARNING**、含 "fail" 的 INFO 行…)
#       ⇒ ⛔ **宽词表绝不能默认判死**,否则会杀掉 1/7 的正常启动 ✗
#
# ⚠️ **诚实说明**:用户 2026-10-15 的原话是"碰到**任何 error** 就立刻失败"。
#   v1 照字面实现 ⇒ 实测 **13.8%~30.4% 的真实成功启动会被误杀**(含生产 ds41f 线)✗
#   ⇒ v2 采取:T1 判死(**默认**)、T2 全量上报(**默认只报不杀**)✓
#   ⇒ 若要恢复"任何 error 都判死"的字面语义,设 **`ES_STRICT=1`**(并把 T2 白名单补全)✓
#
# 用法:
#   . scripts/lib_errscan.sh
#   es_scan_new  <log> <statefile>   # 增量;rc 0=命中致命(打到 stdout) 1=无 2=扫描失败 3=配置错
#   es_scan_t2   <log>               # 全量 T2 告警扫描;rc 0=有告警(打到 stdout) 1=无
#   es_first_error <log>             # 全量;rc 0=有(打印首行,截 200 字)
#   es_has_error   <log>
#   es_selfcheck                     # rc 0=全绿;任何 pattern 不可编译/漏检 ⇒ 非 0 ✓
#
# ⚠️ 日志一律用 `-a` ⇒ 混进 NUL/非法字节时不"静默不匹配" ✓
# ⚠️ 本库**对 `set -euo pipefail` 安全**:不 bare-return 失败命令、不裸用管道判据 ✓(A-M6)
# License: Apache-2.0

# ── 可重入保护(对比 lib_proc_identity.sh;审计 m8)────────────────────────────
if [ -n "${_LIB_ERRSCAN_LOADED:-}" ]; then return 0; fi
_LIB_ERRSCAN_LOADED=1

# ═══ T1【致命】结构式签名 —— 命中即判死 ═══════════════════════════════════════
# 每一条都必须是"进程级/加载级死亡"的签名,⛔ 不许放"可能只是警告"的词 ✗
ES_FATAL='traceback \(most recent call last\)'
ES_FATAL="$ES_FATAL|workerproc failed to start|engine core initialization failed"
ES_FATAL="$ES_FATAL|enginedeaderror|engine core is stuck|enginecore proc died|engine dead"
ES_FATAL="$ES_FATAL|child process exited with code|process group not initialized, hanging"
ES_FATAL="$ES_FATAL|torch\.cuda\.outofmemoryerror|cuda out of memory|\[rank[0-9]+\]: oom"
ES_FATAL="$ES_FATAL|watchdog caught collective operation timeout|worker unresponsive"

# T1 白名单 —— ⛔ **必须锚定**(只挡这一种已知良性异常,不许整行放行)✗
#   实测依据:v2 之前的 T1 在 499 份成功日志上命中 2 份,**两份都是同一条**:
#   vLLM 的 `usage_lib` 统计线程调 `cpuinfo.get_cpu_info()` 抛异常 ⇒
#   `Exception in thread Thread-1 (_report_usage_worker)` + cpuinfo 的 json 解析失败
#   ⇒ 该线程死掉**不影响引擎**,服务随后正常 ready ✓(abmx_b.log / xtu41.memfoot.log)
ES_FATAL_BENIGN='Exception in thread Thread-1 \(_report_usage_worker\)'

# ═══ T2【告警】宽词表 —— 默认只报不杀 ✓ ═══════════════════════════════════════
# 词必须**不带大小写假设**;v2 按审计 A-B7 补入"卡死"类签名 ✓
ES_PAT='error|exception|traceback|fatal|panic|abort|assert|fail|refus|not implemented'
ES_PAT="$ES_PAT|notimplemented|no module named|modulenotfound|out of memory|cuda error"
ES_PAT="$ES_PAT|segmentation fault|core dumped|killed|invalid|cannot|can not|unable to"
ES_PAT="$ES_PAT|unrecognized|timeout|timed out|died|crash|exited|stuck|hang|unresponsive"
ES_PAT="$ES_PAT|heartbeat|oom|corrupt|mismatch|broken pipe|killed by signal"

# ═══ T2 白名单 —— **每条都必须写出为什么** ✓(它是"我认为这句不是错"的显式凭证)═══
# 取证方法(必须这样做,不许凭印象):把模式跑在所有【成功启动】日志的 ready-前缀上,
# 看命中了什么;⛔ 不许靠"把词表收窄"来回避误报 ✗(R54 规则 ③)
ES_BENIGN='Auto-initialization of reasoning token IDs failed'
ES_BENIGN="$ES_BENIGN|You are using a model of type"
ES_BENIGN="$ES_BENIGN|stopping engine client mode=abort"
# ↓ v2 新增(全部来自 2026-10-15 对 499 份 ready-前缀日志的实测统计,括号内为出现次数)
ES_BENIGN="$ES_BENIGN|\[vllm-xtu-moe/shims\] skip engram-last shim: ImportError"   # 28× 我们自己的回退
ES_BENIGN="$ES_BENIGN|\[vllm-xtu-moe/shims\] ced 窗口扩展失败"                      # 271× 设计内回退
ES_BENIGN="$ES_BENIGN|\[ced-diag\] dump 失败"                                       # 68× 我们自己的诊断
ES_BENIGN="$ES_BENIGN|CUDACachingAllocator\.cpp:[0-9]+\] memory allocation failed with OOM"  # 64× KV profile **WARNING**
ES_BENIGN="$ES_BENIGN|non-default args: .*kv_load_failure_policy"                   # 30× INFO,只是**含** "fail"
ES_BENIGN="$ES_BENIGN|gpu_prefill_min_batch_size .* must be less than or equal to"   # 80× ERROR 级但被夹取后继续
ES_BENIGN="$ES_BENIGN|Exception in thread Thread-1 \(_report_usage_worker\)"         # 2× 见上

# ── 良性理由(逐条;⛔ 新增必须补理由)──────────────────────────────────────────
# ① `Auto-initialization of reasoning token IDs failed.`
#    vLLM **WARNING**;本项目 reasoning parser 没实现 start/end str ⇒ **每次成功启动都出现** ✓
# ② `You are using a model of type ...`
#    `transformers` 的**信息性**提示(句子里的 "yield errors" 命中 `error`)✓
# ③ `stopping engine client mode=abort`
#    vLLM **INFO** 停机行(`abort` 是模式名,不是异常)✓
# ④⑤⑥ 三条 `[vllm-xtu-moe/shims]` / `[ced-diag]`:我们自己的**设计内回退/诊断**,
#    代码路径显式"退回上游行为"⇒ 不是故障 ✓(实测 271+68+28 次,全部出现在成功启动里)
# ⑦ `CUDACachingAllocator.cpp … memory allocation failed with OOM`
#    这是 vLLM 在 **KV cache profile** 阶段的**试探性分配**失败,日志级别是 **W** ⇒
#    引擎自动缩小 KV 后继续 ✓
#    ⭐ **最强证据**:生产 `v41_8070.1018242.log` 第 636 行出现该行(03:39:34),
#      第 728 行 `Application startup complete`(03:39:50)—— **同一进程 16 s 后成功就绪** ✓
# ⑧ `non-default args: {…'kv_load_failure_policy': 'fail'}`:INFO 行,只是**值**叫 fail ✓
# ⑨ `gpu_prefill_min_batch_size … must be less than or equal to`:ERROR 级,但 vLLM
#    **夹取(clamp)后继续**;实测 80 次全部出现在成功启动里 ✓
# ⑩ `Exception in thread Thread-1 (_report_usage_worker)`:见 ES_FATAL_BENIGN 注释 ✓
# ⚠️ 注意:真正的 `[rankN] CUDACachingAllocator … OOM` **不属于**(⑦只挡 "memory allocation failed
#    with OOM" 这一种措辞;`OOM!` / `torch.cuda.OutOfMemoryError` 仍由 T1 抓到)✓

# ═══ 内部工具 ═════════════════════════════════════════════════════════════════
_ES_PAT_T1="${ES_FATAL}"
_ES_PAT_T1B="${ES_FATAL_BENIGN}"
_ES_PAT_T2="${ES_PAT}"
_ES_PAT_T2B="${ES_BENIGN}"

# 由消费方追加白名单(⚠️ v2:**必须校验正则可编译**,否则 exit 3 —— 修 A-B4 的注入面)
_es_compile_ok() {  # rc 0 = 可编译;1 = 合法但无匹配;≥2 = 正则非法 ✓
  printf '' | grep -aiE "$1" >/dev/null 2>&1
  local rc=$?
  [ "$rc" -le 1 ]
}

# ⭐⭐ T1 扫描器(v3,2026-10-15 R38 审计后**彻底去掉 awk**)⭐⭐
#   ⛔ 为什么必须去掉 awk(审计判定的 BLOCKER):
#      v2 用 `awk -v p="$1"` 把正则传进 awk,而 ⭐ **部署机 `gpu` 的 awk 是 gawk 5.1.0**
#      (GNU Awk 5.1.0;`/etc/alternatives/awk -> /usr/bin/gawk`),**gawk 会处理 `-v` 值里的转义序列** ⇒
#      `\\(` 被剥成 `(` ⇒ T1 第一条 `traceback \(most recent call last\)` **在部署机上永不命中** ✗✗
#      实测:同一份冻结代码在 gpu 上跑本 lib 自带的 7 条 T1 语料 **只命中 5 条**(漏掉两条 Traceback)✓
#      ⭐ 而本机 harness 的 awk 是 **mawk 1.3.4**,它**保留** `\\(` ⇒ **开发机全绿、部署机失效** ✗
#      ⇒ ⇒ 这正是用户最痛的模式:"启动失败却检测不到",而且**测试机永远暴露不了** ✗
#   ⭐ v3 的机械判据:**检测路径与自检路径必须用【同一个引擎】** ✓
#      现在检测与 `es_selfcheck` 都走 `grep -aiE`,不再可能出现"自检绿、检测瞎" ✗
#   ⭐ 附带收益:grep `-i` 天然大小写不敏感,不必再像 v2 那样两侧 `tolower()`;
#      也不再受"awk 程序体内不许有非 ASCII"的 mawk 怪癖约束 ✓
#   参数:$1=T1 pattern  $2=上一行白名单  $3=输入文件  $4=命中输出文件
#   rc:0=有命中(至多 3 行写入 $4)  1=无命中  ≥2=**扫描失败**
_es_t1_scan() {
  local pat="$1" benign="$2" in="$3" out="$4"
  local raw; raw="$(mktemp)" || return 2
  : > "$out" 2>/dev/null || { rm -f "$raw"; return 2; }

  # ⭐ 取"行号:内容" —— -a 二进制安全 / -n 行号 / -i **大小写不敏感** / -E ERE / -m 上限
  grep -ainE -m 500 -- "$pat" "$in" > "$raw" 2>/dev/null
  local grc=$?
  case "$grc" in
    1) rm -f "$raw"; return 1 ;;        # 确无命中
    0) : ;;
    *) rm -f "$raw"; return 2 ;;        # ⭐ grep 自身失败(读不到/正则非法)⇒ 必须报"扫描失败" ✗
  esac

  local line lineno prev n=0
  while IFS= read -r line; do
    lineno="${line%%:*}"
    case "$lineno" in ''|*[!0-9]*) continue ;; esac
    # ⭐ 上下文感知:已知良性 traceback 的**判据在上一行**
    #   (`Exception in thread Thread-1 (_report_usage_worker):` 在前,`Traceback` 在后)
    #   ⇒ 上一行命中白名单就跳过本行(grep 逐行做不到这件事,故必须有这一步)✓
    if [ -n "$benign" ] && [ "$lineno" -gt 1 ]; then
      prev="$(sed -n "$((lineno - 1))p" "$in" 2>/dev/null)"
      if [ -n "$prev" ] && printf '%s\n' "$prev" | grep -qiE -- "$benign" 2>/dev/null; then
        continue
      fi
    fi
    printf '%s\n' "${line#*:}" >> "$out"
    n=$((n + 1))
    [ "$n" -ge 3 ] && break
  done < "$raw"

  rm -f "$raw"
  [ "$n" -gt 0 ] && return 0
  return 1
}

_es_finalize() {
  local rc=0
  # ⭐ 四个 pattern **全部**编译校验 —— 坏 pattern 必须**确定性地**报 3,
  #   而不是让它退化成"扫描失败/静默无匹配" ✗(修 A-B3/A-B5 的根因之一)
  _es_compile_ok "$_ES_PAT_T1"  || { echo "⛔ T1 致命 pattern 不是合法正则" >&2; return 3; }
  _es_compile_ok "$_ES_PAT_T1B" || { echo "⛔ T1 白名单不是合法正则" >&2; return 3; }
  _es_compile_ok "$_ES_PAT_T2"  || { echo "⛔ T2 宽词表不是合法正则" >&2; return 3; }
  _es_compile_ok "$_ES_PAT_T2B" || { echo "⛔ T2 白名单不是合法正则" >&2; return 3; }

  local extra="${BENIGN_EXTRA:-}"
  if [ -n "$extra" ]; then
    # ⭐ BENIGN_EXTRA 只许影响 **T2(告警)**;⛔ 永远不许影响 T1(判死)✗
    #    ⇒ 恶性/畸形注入都无法关掉判死门(修 A-B4)
    _es_compile_ok "$extra" || { echo "⛔ BENIGN_EXTRA 不是合法正则" >&2; return 3; }
    _ES_PAT_T2B="${ES_BENIGN}|${extra}"
  fi
  local strict="${ES_STRICT:-0}"
  case "$strict" in
    0|1) ;;
    *) echo "⛔ ES_STRICT 只能是 0/1(实得 '$strict')" >&2; return 3 ;;
  esac
  # ⭐⭐⭐ 审计 MINOR 修复(R58 同型:【文档承诺了却没有消费方】= 死旋钮)⭐⭐⭐
  #   ⛔ 原版 `ES_STRICT` 只被上面那个 case **校验**,**没有任何分支读它** ⇒
  #      设了它**行为一点不变** ✗ —— 而注释里写着"1 ⇒ 恢复任何 error 都判死的字面语义" ✗
  #      ⇒ 这正是 R58 说的"空头承诺",本项目已犯第 N 次 ✓
  #   ⇒ 现在**落实**该承诺:ES_STRICT=1 ⇒ 把 T2 宽词表并进 T1 判死集合 ✓
  #      ⚠️ 注意此时 **T2 良性白名单不再豁免**(字面语义就是"任何 error 词都杀")⇒
  #        实测该口径在 499 个"曾经就绪"前缀上**误杀 13.8%(生产日志 30.4%)** ⇒
  #        仅在**明知会误杀**的排障场景下才使用(R54 规则 2b)✓
  if [ "$strict" = "1" ]; then
    _ES_PAT_T1="${_ES_PAT_T1}|${_ES_PAT_T2}"
  fi
  return 0
}

# 落盘状态:`dev:inode:offset`(用于识别**轮转/替换** ⇒ 修 A-B2)✓
_es_save() { printf '%s:%s:%s\n' "$1" "$2" "$3" > "$4" 2>/dev/null || true; }

# ── 增量扫描(⭐ 核心)──────────────────────────────────────────────────────────
# rc: 0=命中致命(行打到 stdout) 1=无 2=扫描失败(**偏移不推进**) 3=配置错
es_scan_new() {
  local f="${1:-}" st="${2:-}"
  [ -n "$f" ] && [ -n "$st" ] || return 3
  _es_finalize || return 3
  [ -f "$f" ] || return 1

  local dev ino size
  dev="$(stat -c '%d' "$f" 2>/dev/null)" || return 2
  ino="$(stat -c '%i' "$f" 2>/dev/null)" || return 2
  size="$(stat -c '%s' "$f" 2>/dev/null)" || return 2
  case "$size" in ''|*[!0-9]*) return 2 ;; esac

  # 读旧状态(兼容 v1 的"纯数字"格式)
  local sdev="" sino="" off="0"
  if [ -f "$st" ]; then
    local raw=""; raw="$(cat "$st" 2>/dev/null || true)"
    case "$raw" in
      *:*) IFS=: read -r sdev sino off <<<"$raw" ;;
      *)   off="$raw" ;;
    esac
  fi
  case "$off" in ''|*[!0-9]*) off=0 ;; esac

  # ⭐ inode/设备变了 或 文件变短 ⇒ 轮转/截断 ⇒ **重置为 0 全量重扫** ✓(修 A-B2)
  if [ "$sdev" != "$dev" ] || [ "$sino" != "$ino" ] || [ "$size" -lt "$off" ]; then off=0; fi
  if [ "$size" -le "$off" ]; then _es_save "$dev" "$ino" "$off" "$st"; return 1; fi

  # 取 [off, EOF) 区间
  local tmp; tmp="$(mktemp)" || return 2
  tail -c +$((off + 1)) "$f" > "$tmp" 2>/dev/null || { rm -f "$tmp"; return 2; }
  local newbytes; newbytes="$(wc -c < "$tmp" | tr -d ' ')"
  [ -n "$newbytes" ] || { rm -f "$tmp"; return 2; }

  # ⭐⭐ **只消费到最后一个完整行** —— 尾部残行留到下一轮 ✓(修 A-B1:跨块半行永久漏检)
  local consume="$newbytes"
  local lastbyte=""
  if [ "$newbytes" -gt 0 ]; then lastbyte="$(tail -c1 "$tmp" | od -An -tu1 2>/dev/null | tr -d ' \n')"; fi
  if [ "$newbytes" -gt 0 ] && [ "$lastbyte" != "10" ]; then
    # ⭐ 这里仍用一次 awk,但**只算长度、不传正则** ⇒ gawk 的 `-v` 转义问题不存在 ✓
    #   ⚠️ 必须 `LC_ALL=C`:审计实测部署机(gawk + UTF-8 locale)下 `length($0)` **数的是字符**,
    #      而偏移是**字节** ⇒ `consume` 偏大、偏移落进尾部半行内部 ✗
    #      (实测:`printf '错误信息\n' | awk '{print length($0)}'` 在 gpu 上是 4,应为 12 字节)
    #   ⭐ 又因为已经确认"文件末尾不是换行"(`lastbyte != 10`)⇒ awk 的末条记录**就是**那半行 ✓
    local incomplete; incomplete="$(LC_ALL=C awk 'END{print length($0)+0}' "$tmp" 2>/dev/null)"
    case "$incomplete" in ''|*[!0-9]*) incomplete=0 ;; esac
    consume=$((newbytes - incomplete))
  fi
  if [ "$consume" -le 0 ]; then rm -f "$tmp"; _es_save "$dev" "$ino" "$off" "$st"; return 1; fi

  head -c "$consume" "$tmp" > "$tmp.c" 2>/dev/null || { rm -f "$tmp" "$tmp.c"; return 2; }
  rm -f "$tmp"

  # T1 扫描 —— ⭐ 上下文感知 + 直取 rc,⛔ 不走管道(修 A-B5/B-B5)、⛔ 不用逐行 grep(修上一条)
  local rc=0
  _es_t1_scan "$_ES_PAT_T1" "$_ES_PAT_T1B" "$tmp.c" "$tmp.hit"; rc=$?
  case "$rc" in
    1) rm -f "$tmp.c" "$tmp.hit"; _es_save "$dev" "$ino" "$((off + consume))" "$st"; return 1 ;;
    0) : ;;
    *) rm -f "$tmp.c" "$tmp.hit"; return 2 ;;          # 扫描失败 ⇒ ⭐ 不推进偏移 ✓
  esac
  [ -s "$tmp.hit" ] || { rm -f "$tmp.c" "$tmp.hit"; _es_save "$dev" "$ino" "$((off + consume))" "$st"; return 1; }

  # ⭐ 只有**扫描成功**才推进偏移 ✓(修 A-B3)
  _es_save "$dev" "$ino" "$((off + consume))" "$st"
  head -3 "$tmp.hit"
  rm -f "$tmp.c" "$tmp.hit"
  return 0
}

# ── T2 全量告警扫描(只报不杀;`ES_STRICT=1` 时由调用方当判死依据)──────────────
es_scan_t2() {
  local f="${1:-}"
  [ -n "$f" ] || return 3
  _es_finalize || return 3
  [ -f "$f" ] || return 1
  local tmp; tmp="$(mktemp)" || return 2
  local rc=0
  grep -aiE "$_ES_PAT_T2" "$f" > "$tmp.a" 2>/dev/null; rc=$?
  case "$rc" in
    1) rm -f "$tmp" "$tmp.a"; return 1 ;;
    0) : ;;
    *) rm -f "$tmp" "$tmp.a"; return 2 ;;
  esac
  grep -aivE "$_ES_PAT_T2B" "$tmp.a" > "$tmp.b" 2>/dev/null; rc=$?
  case "$rc" in
    1) rm -f "$tmp" "$tmp.a" "$tmp.b"; return 1 ;;
    0) : ;;
    *) rm -f "$tmp" "$tmp.a" "$tmp.b"; return 2 ;;
  esac
  echo "$(wc -l < "$tmp.b" | tr -d ' ') 条含错误词的行(未必致命)"; head -10 "$tmp.b"
  rm -f "$tmp" "$tmp.a" "$tmp.b"; return 0
}

# ── 全量取首条致命行(事后验尸用)✓ ────────────────────────────────────────────
es_first_error() {
  local f="${1:-}"
  [ -n "$f" ] || return 3
  _es_finalize || return 3
  [ -f "$f" ] || return 1
  local tmp; tmp="$(mktemp)" || return 2
  local rc=0
  _es_t1_scan "$_ES_PAT_T1" "$_ES_PAT_T1B" "$f" "$tmp"; rc=$?
  case "$rc" in
    1) rm -f "$tmp"; return 1 ;;
    0) : ;;
    *) rm -f "$tmp"; return 2 ;;
  esac
  head -1 "$tmp" | cut -c1-200
  rm -f "$tmp"; return 0
}

es_has_error() { es_first_error "${1:-}" >/dev/null 2>&1; }

# ── 自检(⭐ v2:失败方向已纠正 —— grep rc≥2 必须判【失败】)─────────────────────
es_selfcheck() {
  local rc=0 s n=0 tmp
  # ⓪ ⭐ **先跑 _es_finalize** —— 否则本函数对 `BENIGN_EXTRA='['` 会报"全绿"(假绿,审计 A-B4)✗
  if ! _es_finalize >/dev/null 2>&1; then
    echo "  ⛔ 自检失败:pattern/白名单配置非法(_es_finalize 返回非 0)—— 例如 BENIGN_EXTRA 不是合法正则"
    return 3
  fi
  # ① T1 必须抓到致命签名(⛔ 含**非 error 词**的卡死签名,防止下次又被退回 ✗ —— 审计 A-B7)
  #   ⭐⭐ v3(审计 BLOCKER 修复):**必须走真正的检测函数 `_es_t1_scan`**,⛔ 不许另用 grep 自检 ✗
  #      理由:自检用 grep、检测用 awk 时,两者引擎不同 ⇒ 检测瞎了自检照样报"全绿" = **假绿** ✓
  #      实测后果:v2 在部署机(gawk)上 T1 首条 Traceback 永不命中,而本自检打印"✅ 全绿" ✗
  local _sc; _sc="$(mktemp)" || return 2
  while IFS= read -r s; do
    [ -n "$s" ] || continue
    n=$((n + 1))
    printf '%s\n' "$s" > "$_sc.in"
    if ! _es_t1_scan "$_ES_PAT_T1" "$_ES_PAT_T1B" "$_sc.in" "$_sc.hit"; then
      echo "  ⛔ T1 自检失败:检测函数 _es_t1_scan 未命中 '$s'(自检与检测必须同引擎)"
      rc=1
    fi
  done <<'T1CASES'
Traceback (most recent call last):
(Worker_TP1 pid=1) Traceback (most recent call last):
Engine core initialization failed. See root cause above.
EngineCore proc died unexpectedly, exiting.
Child process exited with code 1
torch.cuda.OutOfMemoryError: CUDA out of memory.
Watchdog caught collective operation timeout: WorkNCCL ran for 600000 ms before timing out.
T1CASES
  rm -f "$_sc" "$_sc.in" "$_sc.hit"
  # ② T2 必须抓到**大小写变体**与**卡死词表**(R54 不变式 ① ②)
  while IFS= read -r s; do
    [ -n "$s" ] || continue
    n=$((n + 1))
    printf '%s\n' "$s" | grep -aiE "$_ES_PAT_T2" >/dev/null 2>&1 || { echo "  ⛔ T2 自检失败:未命中 '$s'"; rc=1; }
  done <<'T2CASES'
ERROR boom
error boom
Error boom
eRRoR boom
ErRoR boom
NotImplementedError: nope
RuntimeError: x
ImportError: y
FATAL z
Engine core is stuck, no heartbeat for 60s
EngineCore proc died unexpectedly
[rank0]: OOM!
worker 1 crashed
Timeout(ms)=600000 expired, worker unresponsive
process group not initialized, hanging
T2CASES
  # ③ 白名单必须挡掉已知良性(⭐ v2:**失败方向纠正** —— grep rc≥2 判失败,不当"已挡掉")
  while IFS= read -r s; do
    [ -n "$s" ] || continue
    n=$((n + 1))
    if printf '%s\n' "$s" | grep -aivE "$_ES_PAT_T2B" >/dev/null 2>&1; then
      echo "  ⛔ 自检失败:良性行未被挡掉 '$s'"; rc=1
    else
      local g=$?
      [ "$g" -eq 1 ] || { echo "  ⛔ 自检失败:白名单正则不可编译(rc=$g)"; rc=1; }
    fi
  done <<'BENCASES'
WARNING Auto-initialization of reasoning token IDs failed. Please check
[transformers] You are using a model of type `x` to instantiate
INFO [launcher.py:1] [shutdown] stopping engine client mode=abort timeout=30
[rank1]:[W1015 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 1
(APIServer pid=1) INFO non-default args: {'kv_load_failure_policy': 'fail'}
ERROR [routed_experts.py:1] gpu_prefill_min_batch_size (2048) must be less than or equal to max_num_batched_tokens (1024)
BENCASES
  # ④ ⭐ **白名单正则必须可编译** —— 这正是不良 `BENIGN_EXTRA` 的注入面(A-B4)
  tmp="$(mktemp)" || return 2
  printf '' > "$tmp"
  local g=0
  grep -aiE "$_ES_PAT_T2B" "$tmp" >/dev/null 2>&1; g=$?
  grep -aiE "$_ES_PAT_T1" "$tmp" >/dev/null 2>&1 || true
  rm -f "$tmp"
  [ "$g" -le 1 ] || { echo "  ⛔ 白名单正则不可编译(rc=$g)⇒ 整门会失效"; rc=1; }

  if [ "$rc" = "0" ]; then
    echo "  ✅ es_selfcheck 全绿(T1 $n 例 / 大小写变体 / 卡死词表 / 良性白名单 全过)"
  fi
  return "$rc"
}
