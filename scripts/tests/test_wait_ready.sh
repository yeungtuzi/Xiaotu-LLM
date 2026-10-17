#!/usr/bin/env bash
# G9 —— `scripts/wait_ready.sh` 的【行为测试】(2026-10-07)
#
# 为什么需要它(审计判定 wait_ready.sh 的 4 处硬伤):
#   ① `curl` 没有 `-f` ⇒ 404/500 也判"就绪" ✗
#   ② **取不到端口属主【即放行】**(事故 F 未堵死)✗ ⇒ 必须 fail-closed
#   ③ awk 正则未锚定 ⇒ `:814` 会匹配 `:8143` ✗
#   ④ `TIMEOUT` 非整数 ⇒ `[ -ge ]` 报错 ⇒ **死循环** ✗
#
# ⭐ 本测试用【假 endpoint + 临时 LOGDIR】,不碰生产、不碰 8070/8071 ✓
#   假服务通过 `proc.sh spawn` 起(顺带复用它的 PID 文件与存活保证)✓
#
# 用法: bash scripts/tests/test_wait_ready.sh
# 退出码:0 = 全绿 · 1 = 有用例不符 · 2 = 用法/内部错
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL_LOGDIR="$ROOT/dev-docs/report/tuning/logs"
# shellcheck source=../lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"
# ⭐ 8b 反向用例要直接调 `es_scan_t2` 验证"T2 行没被静默丢弃" ⇒ 必须 source 这个 lib ✓
#   (⚠️ 本测试自身曾因没 source 它而误报"失败" —— 那是测试的缺陷,不是被测脚本的 ✗)
# shellcheck source=../lib_errscan.sh
. "$ROOT/scripts/lib_errscan.sh"
WAIT="$ROOT/scripts/wait_ready.sh"

TMP="$(mktemp -d)" || exit 2
export LOGDIR="$TMP"
case "$(readlink -f "$TMP")/" in
  "$(readlink -f "$REAL_LOGDIR")"/*) echo "⛔ 拒绝在真 LOGDIR 上运行" >&2; exit 2 ;;
esac

PASS=0; FAIL=0; SKIP=0
chk() {   # chk <描述> <期望 rc> <实得 rc>
  if [ "$2" = "$3" ]; then echo "  ✅ $1  (期望=$2 实得=$3)"; PASS=$((PASS+1))
  else echo "  ❌ $1  (期望=$2 实得=$3)"; FAIL=$((FAIL+1)); fi
}
skip() { echo "  ⏭  SKIP: $*"; SKIP=$((SKIP+1)); }
# ⭐ 2026-10-07 独立审计:核心用例"跳过"不得等于通过(除非显式 --allow-skip=<理由>)✗
ALLOW_SKIP_REASON=""
for _a in "$@"; do case "$_a" in --allow-skip=*) ALLOW_SKIP_REASON="${_a#--allow-skip=}" ;; esac; done
skipf() {
  echo "  ❌ 核心用例未执行:$*"
  if [ -n "$ALLOW_SKIP_REASON" ]; then
    echo "     (显式 --allow-skip=$ALLOW_SKIP_REASON ⇒ 降级为 SKIP)"; SKIP=$((SKIP+1))
  else FAIL=$((FAIL+1)); fi
}

# 假 endpoint:GET 任意路径都回指定状态码与 body(只为测 curl -f 与就绪判据)✓
cat > "$TMP/fakeapi.py" <<'PY'
import sys, http.server, socketserver
port = int(sys.argv[1]); code = int(sys.argv[2]); body = sys.argv[3].encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(code); self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', port), H) as s:
    s.serve_forever()
PY

_free_port() { python3 - <<'PY'
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()
PY
}

_cleanup() {
  for n in srvA srvB srvD srvE srvF srvG; do
    [ -f "$TMP/$n.pid" ] || continue
    # ⭐ G5 要求 stop 调用点必须"留下"退出码:这里用命令替换捕获(收尾清理,rc 只作记录)
    _o="$(bash "$ROOT/scripts/proc.sh" stop "$n" --force --reason=g9-cleanup 2>&1)"; : "$_o"
  done
  rm -rf "$TMP"
}
trap _cleanup EXIT

echo "G9 wait_ready.sh 行为测试(临时 LOGDIR=$TMP)"
echo

# ── 起两个假服务:A=200 + JSON; B=404 ─────────────────────────────────────────
PA="$(_free_port)"; PB="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvA python3 "$TMP/fakeapi.py" "$PA" 200 '{"object":"list"}' >/dev/null 2>&1
bash "$ROOT/scripts/proc.sh" spawn srvB python3 "$TMP/fakeapi.py" "$PB" 404 '{"error":"nope"}' >/dev/null 2>&1
sleep 1
PA_PID="$(cat "$TMP/srvA.pid" 2>/dev/null || true)"
PB_PID="$(cat "$TMP/srvB.pid" 2>/dev/null || true)"
if ! pi_valid_pid "$PA_PID" || ! pi_alive "$PA_PID"; then
  skip "假服务 A 没起来(port=$PA)⇒ 无法测试"; echo; echo "G9 结果:通过 $PASS / 失败 $FAIL / 跳过 $SKIP"; exit 1
fi

echo "  假服务 A: pid=$PA_PID port=$PA (200+JSON) · 假服务 B: pid=${PB_PID:-?} port=$PB (404)"
echo

# ── 1) 就绪:2xx + JSON + 端口属主==PID文件里的 pid ⇒ 0 ───────────────────────
note() { echo "  ── $*"; }
note "1) 就绪路径"
bash "$WAIT" srvA "$PA" 8 >/dev/null 2>&1; chk "2xx + 属主匹配 ⇒ 0" 0 "$?"

# ── 2) 404 不得判就绪(审计 fix #1)───────────────────────────────────────────
note "2) 非 2xx 不得判就绪"
bash "$WAIT" srvB "$PB" 4 >/dev/null 2>&1; chk "404 ⇒ 超时(2),不得放行" 2 "$?"

# ── 3) 端口属主不是本实例 ⇒ 必须拒绝(审计 fix #2,事故 F)───────────────────
note "3) 端口属主不匹配 ⇒ fail-closed"
# 3a) ⚠️ 必须是【另一个】活进程登记成 nC:端口 PA 的属主是 srvA,而 nC.pid 指向 srvB
#     (若把 nC.pid 指向 srvA 的同一个 pid,那就是"属主匹配",本来就该判就绪 —— 别写错用例 ✗)
if pi_valid_pid "$PB_PID" && pi_alive "$PB_PID"; then
  echo "$PB_PID" > "$TMP/nC.pid"
  bash "$WAIT" nC "$PA" 4 >/dev/null 2>&1; chk "PID 文件 pid(srvB) ≠ 端口属主(srvA)⇒ 超时(2)" 2 "$?"
  rm -f "$TMP/nC.pid"
else
  skipf "假服务 B 不在,无法构造属主不匹配用例"
fi
# 3b) 完全没有 PID 文件时,即使 2xx 也不得放行
bash "$WAIT" nosuchinst "$PA" 4 >/dev/null 2>&1; chk "无 PID 文件(取不到属主)⇒ 超时(2)" 2 "$?"

# ── 4) 端口子串误匹配(审计 fix #3)──────────────────────────────────────────
note "4) 端口必须整段比对(:814 ≠ :8143)"
PREFIX="${PA:0:${#PA}-1}"     # 例如 34567 ⇒ 3456
[ -n "$PREFIX" ] && [ "$PREFIX" != "$PA" ] || skip "端口太短,无法构造前缀用例"
if [ -n "$PREFIX" ] && [ "$PREFIX" != "$PA" ]; then
  bash "$WAIT" srvA "$PREFIX" 4 >/dev/null 2>&1; chk "查询 ${PREFIX}(服务其实在 ${PA})⇒ 超时(2)" 2 "$?"
fi

# ── 5) 进程已死 ⇒ 1(且不该等满超时)────────────────────────────────────────
note "5) 进程死亡判定"
if kill -0 999999 2>/dev/null; then skip "999999 竟然活着"; else
  echo 999999 > "$TMP/dead.pid"
  t0=$(date +%s); bash "$WAIT" dead 1 6 >/dev/null 2>&1; rc=$?; dt=$(( $(date +%s) - t0 ))
  chk "PID 文件里的进程已死 ⇒ 1" 1 "$rc"
  if [ "$dt" -le 5 ]; then chk "死亡判定是立即的(未等满超时,$dt s)" 0 0; else chk "死亡判定是立即的(未等满超时,$dt s)" 0 1; fi
  rm -f "$TMP/dead.pid"
fi

# ── 6) 参数校验:非整数超时/端口不得致死循环(审计 fix #4)────────────────────
note "6) 参数校验(防死循环)"
bash "$WAIT" srvA "$PA" abc >/dev/null 2>&1; chk "TIMEOUT 非整数 ⇒ 3(不死循环)" 3 "$?"
bash "$WAIT" srvA abc 5     >/dev/null 2>&1; chk "PORT 非整数 ⇒ 3" 3 "$?"
bash "$WAIT" srvA           >/dev/null 2>&1; chk "缺参数 ⇒ 3(统一契约)" 3 "$?"
bash "$WAIT"                >/dev/null 2>&1; chk "无参数 ⇒ 3" 3 "$?"

# ── 7) 服务中途死掉 ⇒ 1(而不是一直等到超时)───────────────────────────────
note "7) 启动中死亡 ⇒ 立即 1"
PF="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvD python3 "$TMP/fakeapi.py" "$PF" 200 '{"object":"list"}' >/dev/null 2>&1
sleep 1
DPID="$(cat "$TMP/srvD.pid" 2>/dev/null || true)"
if pi_valid_pid "$DPID"; then
  # 直接杀掉它 ⇒ wait_ready 应在下一轮判"已死亡"
  kill -KILL "$DPID" 2>/dev/null || true
  sleep 0.5
  t0=$(date +%s); bash "$WAIT" srvD "$PF" 20 >/dev/null 2>&1; rc=$?; dt=$(( $(date +%s) - t0 ))
  chk "实例死亡 ⇒ 1" 1 "$rc"
  if [ "$dt" -le 12 ]; then chk "死亡后快速返回(未等满 20s,$dt s)" 0 0; else chk "死亡后快速返回(未等满 20s,$dt s)" 0 1; fi
else
  skipf "srvD 没起来(死亡判定未被验证)"
fi

# ── 8) ⭐⭐ R54:日志里出现【T1 致命签名】⇒ **立刻** 5,且**大小写不敏感** ──────
#   用户 2026-10-15 原话:「碰到**任何 error** 就会立刻失败,**任何大小写组合都是**,
#   难道只检测了 `ERROR` 和 `error`,**没检测 `Error`**?」
#   ⭐ 同日用户裁定 **A 档**:实现为两级 —— **T1 致命签名判死**,**T2 宽词表只报告**
#      (依据:499 个"曾经就绪"前缀上,字面"任何 error 词"误杀 13.8%/生产 30.4%;T1 为 0.0%)✓
#   ⇒ 本用例**只写混合大小写 T1 签名**(全行没有任何大写 `TRACEBACK`)
#      ⇒ 模式一旦大小写敏感,必漏检 ✗
note "8) 日志 T1 致命签名 ⇒ 立刻 5(大小写不敏感)"
cat > "$TMP/errsvc.sh" <<'SH'
#!/usr/bin/env bash
printf 'tRaCeBaCk (MoSt ReCeNt CaLl LaSt):\n'
printf '  File "x.py", line 1, in <module>\n'
exec sleep 300
SH
PE="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvE bash "$TMP/errsvc.sh" >/dev/null 2>&1
sleep 1
EE_PID="$(cat "$TMP/srvE.pid" 2>/dev/null || true)"
if pi_valid_pid "$EE_PID" && pi_alive "$EE_PID"; then
  t0=$(date +%s); bash "$WAIT" srvE "$PE" 30 >/dev/null 2>&1; rc=$?; dt=$(( $(date +%s) - t0 ))
  chk "混合大小写 T1 签名 'tRaCeBaCk' ⇒ 5" 5 "$rc"
  if [ "$dt" -le 15 ]; then chk "T1 是【立刻】报的(未等满 30s,$dt s)" 0 0
  else chk "T1 是【立刻】报的(未等满 30s,$dt s)" 0 1; fi
else
  skipf "srvE 没起来(R54 大小写不敏感未被验证)"
fi

# ── 8b) ⭐⭐ R54 A 档【反向用例】:裸 `error` 词(T2)⇒ **不得**判死 ──────────────
#   ⚠️ 这条是**防退化闸**:将来若有人"顺手把 T2 也接成退出码",本用例立刻变红 ✗
#   依据:R54 规则 2b —— 字面"任何 error 词"在真实成功启动上误杀 13.8%(生产 30.4%)✓
note "8b) 裸 error 词(T2)不得判死"
cat > "$TMP/t2svc.sh" <<'SH'
#!/usr/bin/env bash
printf 'eRRoR: 这行是 T2 宽词表命中,但【不是】致命签名 ⇒ 不得判死\n'
exec sleep 300
SH
PT2="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvT2 bash "$TMP/t2svc.sh" >/dev/null 2>&1
sleep 1
T2_PID="$(cat "$TMP/srvT2.pid" 2>/dev/null || true)"
if pi_valid_pid "$T2_PID" && pi_alive "$T2_PID"; then
  bash "$WAIT" srvT2 "$PT2" 8 >/dev/null 2>&1; rc=$?
  chk "裸 'eRRoR'(T2)⇒ 不得是 5(应为超时 2)" 2 "$rc"
  # ⭐ 同时确认"没丢掉":T2 必须仍能被 lib 看见(只是不判死)
  if es_scan_t2 "$TMP/srvT2.log" >/dev/null 2>&1; then chk "T2 仍看得见该行(未被静默丢弃)" 0 0
  else chk "T2 仍看得见该行(未被静默丢弃)" 0 1; fi
else
  skipf "srvT2 没起来(R54 A 档反向用例未执行)"
fi

# ── 9) ⭐ R54 规则5:良性白名单**不得**误杀(否则每次成功启动都被判失败)────────
note "9) 良性白名单不得误杀"
cat > "$TMP/benignsvc.sh" <<'SH'
#!/usr/bin/env bash
# ⭐ 这条实测在 60 份**成功**日志里出现 80 次(每次成功启动都有)⇒ 必须被白名单挡掉 ✓
printf 'WARNING 10-15 00:00:00 [vllm.py:1] Auto-initialization of reasoning token IDs failed. Please check whether your reasoning parser has implemented the `reasoning_start_str` and `reasoning_end_str`.\n'
exec sleep 300
SH
PB2="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvG bash "$TMP/benignsvc.sh" >/dev/null 2>&1
sleep 1
EG_PID="$(cat "$TMP/srvG.pid" 2>/dev/null || true)"
if pi_valid_pid "$EG_PID" && pi_alive "$EG_PID"; then
  # 关掉停滞门,只留错误扫描 ⇒ 期望是【超时 2】,绝不能是 5 ✗
  STALL_SEC=0 bash "$WAIT" srvG "$PB2" 6 >/dev/null 2>&1; chk "良性行不得判错 ⇒ 超时(2) 而非 5" 2 "$?"
else
  skipf "srvG 没起来(白名单未被验证)"
fi

# ── 10) ⭐ R53 停滞门:进程活着但【无任何进展】⇒ 4(且远早于超时)──────────────
note "10) 停滞门 ⇒ 4(进程活着)"
PS2="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvF sleep 300 >/dev/null 2>&1
sleep 1
EF_PID="$(cat "$TMP/srvF.pid" 2>/dev/null || true)"
if pi_valid_pid "$EF_PID" && pi_alive "$EF_PID"; then
  t0=$(date +%s)
  STALL_SEC=6 INTERVAL=2 bash "$WAIT" srvF "$PS2" 60 >/dev/null 2>&1; rc=$?; dt=$(( $(date +%s) - t0 ))
  chk "活着但 RSS/read 都无进展 ⇒ 4" 4 "$rc"
  if [ "$dt" -le 30 ]; then chk "停滞判死早于超时($dt s < 60s)" 0 0
  else chk "停滞判死早于超时($dt s < 60s)" 0 1; fi
else
  skipf "srvF 没起来(停滞门未被验证)"
fi

# ── 11) ⭐ R53:STALL_SEC=0 必须能显式关掉(逃生阀,防止误杀正常慢启动)────────
note "11) STALL_SEC=0 关闭停滞门 ⇒ 走超时(2)"
if pi_valid_pid "$EF_PID" && pi_alive "$EF_PID"; then
  STALL_SEC=0 bash "$WAIT" srvF "$PS2" 5 >/dev/null 2>&1; chk "STALL_SEC=0 ⇒ 超时(2),不是 4" 2 "$?"
else
  skipf "srvF 不在,无法验证逃生阀"
fi

# ── 12) ⭐⭐ R38 审计 BLOCKER 修复的反向用例(fail-closed 必须真的 fail)────────
note "12) 审计修复的反向用例:配置错必须【拒绝运行】而不是静默放行"
for bad in "RSS_GROWTH_KB=0" "IO_GROWTH_KB=0" "RSS_GROWTH_KB=abc"; do
  # shellcheck disable=SC2086
  env $bad bash "$WAIT" srvA "$PA" 3 >/dev/null 2>&1; chk "$bad ⇒ 3(拒绝运行)" 3 "$?"
done
etmp="$(mktemp -d)"
env LOGDIR="$etmp" BENIGN_EXTRA='[' bash "$WAIT" srvZ 58998 3 >/dev/null 2>&1
chk "BENIGN_EXTRA='[' ⇒ 3(fail-closed,不许静默关掉判死门)" 3 "$?"
rm -rf "$etmp"

# ── 13) ⭐⭐ 宿主 awk 无关性:检测路径里【不许出现 awk】────────────────────────
note "13) 检测路径必须与宿主 awk 解耦(审计 BLOCKER 1:gawk 会剥 -v 里的 \\( )"
#   ⭐ 静态断言:真正做 T1 命中的函数体里不许有 awk ✗
body="$(sed -n '/^_es_t1_scan()/,/^}/p' "$ROOT/scripts/lib_errscan.sh")"
n_awk="$(printf '%s\n' "$body" | grep -c 'awk' || true)"
chk "_es_t1_scan 体内 awk 出现 0 次(实得 $n_awk)" 0 "$([ "$n_awk" -eq 0 ] && echo 0 || echo 1)"
#   ⭐ 动态断言:走真实的检测函数,Windows-1252 之外的非 ASCII 与括号转义都要命中
awkprobe="$(mktemp)"
printf 'Traceback (most recent call last):\n' > "$awkprobe"
. "$ROOT/scripts/lib_errscan.sh"
if _es_t1_scan "$_ES_PAT_T1" "$_ES_PAT_T1B" "$awkprobe" "$awkprobe.hit"; then rc=0; else rc=1; fi
chk "真实检测函数命中 'Traceback (most recent call last):' ⇒ 0" 0 "$rc"
rm -f "$awkprobe" "$awkprobe.hit"

# ── 14) ⭐⭐ 启动器必须拒绝踩生产资源(默认 fail-closed)──────────────────────
note "14) serve_qwen38.sh 默认必须【拒绝】踩生产 GPU/端口"
serve="$ROOT/scripts/serve_qwen38.sh"
# ⚠️ 必须喂一个**假的 CKPT**(含 config.json),否则脚本会**早于**资源守卫就以 rc=2 退出
#    ⇒ 断言"rc=3"会**假绿**:看着像守卫生效,其实根本没走到守卫 ✗(自查抓到的测试缺陷)
mkdir -p "$TMP/fakeckpt"; echo '{}' > "$TMP/fakeckpt/config.json"
# ⭐ 所以除 rc 外,**必须断言那句守卫的原文** ✓ —— 这才是"守卫真的生效"的证据
runserve() {   # runserve <期望rc> <期望消息子串> <描述> [env...]
  local erc="$1" emsg="$2" desc="$3"; shift 3
  local out rc
  out="$(env CKPT="$TMP/fakeckpt" OUTDIR="$TMP" "$@" bash "$serve" 2>&1)"; rc=$?
  chk "$desc ⇒ rc=$erc" "$erc" "$rc"
  if printf '%s' "$out" | grep -qF "$emsg"; then chk "$desc ⇒ 守卫原文生效(含『$emsg』)" 0 0
  else chk "$desc ⇒ 守卫原文生效(含『$emsg』);实得:$(printf '%s' "$out" | tail -1)" 0 1; fi
}
runserve 3 "生产在用"        "GPUS=2(生产卡)"            GPUS=2
runserve 3 "生产在用"        "GPUS=1(生产卡)"            GPUS=1
runserve 3 "拒绝启动:端口"    "PORT=8070(生产端口)"        PORT=8070
runserve 3 "生产 PID 文件"    "TAG 撞生产 PID 文件名"      TAG=vllm_prod_8070
runserve 3 "实例名非法"       "TAG 含空格(非法名)"         TAG='a b'

# ── 15) ⭐⭐ 审计 F10:【默认值】必须被断言(否则把默认改回 2 也不会变红)──────────
note "15) 默认必须是 GPU0(只测 GPUS=1/2 被拒 ⇒ 假绿)"
dout="$(DRYRUN=1 CKPT="$TMP/fakeckpt" OUTDIR="$TMP" bash "$serve" 2>&1 || true)"
if printf '%s' "$dout" | grep -q 'CUDA_VISIBLE_DEVICES=0'; then chk "默认 CUDA_VISIBLE_DEVICES=0" 0 0
else chk "默认 CUDA_VISIBLE_DEVICES=0(实得:$(printf '%s' "$dout" | grep -o 'CUDA_VISIBLE_DEVICES=[0-9,]*' | head -1))" 0 1; fi
if printf '%s' "$dout" | grep -qE 'gpus=0[, ]'; then chk "默认 gpus=0" 0 0
else chk "默认 gpus=0(实得:$(printf '%s' "$dout" | grep -oE 'gpus=[0-9,]+' | head -1))" 0 1; fi

# ── 16) ⭐⭐ 审计 F2:同一个 TAG 不允许两个实例(否则互相改写 pid/日志)───────────
note "16) 同一 TAG 已被活实例占用 ⇒ 必须拒绝(不许改写它的 pid/日志)"
sleep 120 & victim=$!
mkdir -p "$TMP/duptag.lock"
printf '%s\n' "$victim" > "$TMP/duptag.pid"
# ⚠️ 测试自身缺陷(已犯第二次):`$(... || true)` 会把真实 rc 换成 `true` 的 0 ✗
#    ⇒ 必须**不带 `|| true`** 地赋值,再取 $?(本测试文件没有 set -e ✓)
dout="$(env CKPT="$TMP/fakeckpt" OUTDIR="$TMP" GPUS=0 PORT=8155 TAG=duptag ALLOW_LOW_MEM=1 bash "$serve" 2>&1)"; drc=$?
chk "同 TAG 已被占用 ⇒ 3" 3 "$drc"
if printf '%s' "$dout" | grep -q '已被一个活着的实例占用'; then chk "报出占用原文" 0 0
else chk "报出占用原文(实得:$(printf '%s' "$dout" | tail -1))" 0 1; fi
kill -TERM "$victim" 2>/dev/null || true
rm -rf "$TMP/duptag.lock" "$TMP/duptag.pid"

# ── 17) ⭐⭐ 结构不变式:防止那两条最危险的写法被改回来 ────────────────────────
note "17) 结构不变式(静态断言):不许组杀 / 归属必须来自 \$! / 门必须后台+wait"
#   ⚠️ 必须 `grep -v '^\s*#'` 去掉注释 —— 注释里正写了这条反面示例,否则断言会**误报** ✗
if grep -vE '^[[:space:]]*#' "$serve" | grep -qE 'kill +-[A-Z]+ +"-' ; then chk "⛔ 不许把 pid 当进程组号做组杀" 0 1
else chk "无组杀(kill -TERM \"-\$PID\")⇒ 不会杀掉调用方整个进程组" 0 0; fi
if grep -q 'SVC_PID="\$!"' "$serve"; then chk "归属来自 \$!(不读 pid 文件判归属)" 0 0
else chk "归属来自 \$!" 0 1; fi
if grep -q 'wait "\$WR_PID"' "$serve"; then chk "就绪门后台化 + wait(否则 trap 会被推迟到门结束)" 0 0
else chk "就绪门后台化 + wait" 0 1; fi
if grep -q '_gw_cleanup 1' "$serve"; then chk "失败出口会清理服务(不留孤儿)" 0 0
else chk "失败出口会清理服务" 0 1; fi

echo
echo "G9 结果:通过 $PASS / 失败 $FAIL / 跳过 $SKIP"
[ "$FAIL" -eq 0 ] && echo "G9 ✅ 全绿" || echo "G9 ⛔ 不通过"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
