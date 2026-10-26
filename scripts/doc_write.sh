#!/usr/bin/env bash
# ============================================================================
# doc_write.sh —— 共享文档的【独占写入】
#   落实 AGENTS.md【最高规则 · 2026-10-06】/ IRON_RULES **R36**
#
# 用户原话(定为最高规则):
#   "写入文档要【先用独占写入方式打开(先检查是否失败)】,写完释放,防止重复写入"
#
# ⭐⭐ 语义边界(最容易搞反的一条):
#   · 锁只排他【写者 vs 写者】;⭐⭐【读永远不加锁、不受任何影响】✓
#   · 锁加在【独立锁文件】<file>.lock 上 ⇒ 连正文的 inode 都不碰 ⇒ 更不干扰读 ✓
#   · `flock` 是 advisory:不调用它的进程(read/grep/cat/tail/编辑器…)完全不会被阻塞 ✓
#
# 用法:
#   bash scripts/doc_write.sh <file> --append              <<'EOF'  ...  EOF
#   bash scripts/doc_write.sh <file> --append-auto         <<'EOF'  ...  EOF
#        ⭐ 两者差在【编号】:--append 要求调用方自己写的 `## #NN` 不撞号(撞 ⇒ exit 4,不写);
#           --append-auto 由【脚本在锁内】把首行 `## #NN` 改成 `<同井号数> #<现有最大+1>` ✓
#           ⭐ 井号个数【两或三个都认】(`## #NN` / `### #NN`;重写时保留调用方原样 ✓)
#           ⇒ ⭐ 多会话共写台账时**一律用 --append-auto**(R36 第 4 条的机械执行点 ✓)
#   bash scripts/doc_write.sh <file> --replace '<old>' '<new>'
#        ⚠️⚠️ 调用方警示(2026-10-17 实测踩过):bash 的 `${v/pat/rep}` 里,若 **pat 以 `#` 开头**,
#           那个 `#` 会被 bash 当成**【锚定在开头】的标记** ⇒ 对 `## #255` 这类文本**静默不替换** ✗
#           (实测:`${C/#255/#257}` 不动,`${D//#/X}` 却能动 ⇒ 极易误判为"已改") 
#           ⇒ 构造 new 时用 `sed`/python,并**断言 new != old**(脚本已内置该断言 ⇒ 空操作直接 exit 2 ✓)
#   bash scripts/doc_write.sh <file> --truncate-at '<marker>'   # 自 marker 删到文末(撤销误记)
#   bash scripts/doc_write.sh <file> --check     # 只查锁是否空闲(不写)
#
# 退出码:0 成功 / 2 用法或输入错 / 3 拿不到锁(⭐ 一个字节都没写)
#         / 4 编号撞号 或 待替换文本不唯一 或 找不到标记(⭐ 均未写)
# ============================================================================
set -uo pipefail

usage() {
  sed -n '2,20p' "$0" >&2
  exit 2
}

[ $# -ge 2 ] || usage
F="$1"; MODE="$2"; shift 2
[ -f "$F" ] || { echo "⛔ doc_write: 目标文件不存在: $F" >&2; exit 2; }
LOCK="${F}.lock"

# ---- ① 独占打开 + ② 检查是否失败(非阻塞;失败即【不写】)-------------------
exec 9>>"$LOCK" || { echo "⛔ doc_write: 无法打开锁文件 $LOCK" >&2; exit 3; }
if ! flock -n 9; then
  echo "⛔ doc_write: 独占写入失败 —— 另一端正在写 $F(锁被占: $LOCK)" >&2
  echo "   ⇒ 按最高规则【不要写】;请退避后重试(先确认对方已释放)" >&2
  exit 3
fi
# 拿到锁 ⇒ 进入临界区;无论如何退出都要释放
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
trap release EXIT

case "$MODE" in
  --check)
    echo "✓ doc_write: 锁空闲($LOCK)⇒ 现在可以写"
    exit 0
    ;;

  --append|--append-auto)
    tmp="$(mktemp)"; cat > "$tmp"
    if [ ! -s "$tmp" ]; then
      echo "⛔ doc_write: 输入为空,未写" >&2; rm -f "$tmp"; exit 2
    fi
    # ⭐ 编号在【锁内】分配 / 校验(落实 R36 第 4 条):--append 撞号即拒写;--append-auto 由脚本取号
    AUTO=0; [ "$MODE" = "--append-auto" ] && AUTO=1
    python3 - "$F" "$tmp" "$AUTO" <<'PY'
import re, sys
f, tmp, auto = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = open(f, encoding='utf-8').read()
body = open(tmp, encoding='utf-8').read()
mx = max([int(m) for m in re.findall(r'^#{2,3} #(\d+)', src, re.M)] or [0])
# ⭐ 2026-10-17 修正:下面两处原为 '^### #\d+' / '^### #(\d+)'(【只认三井号】),而
#    `USER_QA_LEDGER.md` 等台账实际写的是【两井号】`## #NN` ⇒ 撞号检查对它们【完全失效】✗
#    实测:两次追加都打印 "本条 -"(即 ids 为空、检查根本没跑)⇒ 已真的撞出 #255/#256 各两条 ✗
#    ⇒ 统一按 '^(#{2,3}) #NN' 匹配,并【保留调用方的井号个数】✓
if auto:
    # ⭐ 2026-10-24 修正(实测踩到):原判据是 `body2 == body` ⇒ 当调用方自己写的编号
    #    恰好等于 mx+1 时,替换结果与原文字节相同 ⇒ 被误判为"首行没有编号"而 exit 4 ✗
    #    (实测:`#298` 存在时写 `## #299` 的条目连续两次被拒;脚本报的是"没有编号可分配",
    #     而真实原因是"分配的编号与你写的一样"。)
    #    ⇒ 改为用 subn 的**替换次数**判据:count==0 才是"没有编号"✓
    body2, n_sub = re.subn(r'^(#{2,3}) #\d+', lambda m: f"{m.group(1)} #{mx+1}", body, count=1, flags=re.M)
    if n_sub == 0:
        print("⛔ --append-auto: 首行没有 '## #NN' / '### #NN' 可分配 ⇒ 未写", file=sys.stderr)
        sys.exit(4)
    body, tag = body2, f"#{mx+1}"
else:
    ids = [int(x) for x in re.findall(r'^#{2,3} #(\d+)', body, re.M)]
    bad = [i for i in ids if i <= mx]
    if bad:
        print(f"⛔ doc_write: 撞号 —— 本条要写 {bad},而现有最大是 #{mx}"
              f" ⇒ ⭐ 未写(改用 --append-auto 让脚本在锁内取号)", file=sys.stderr)
        sys.exit(4)
    tag = f"#{ids[-1]}" if ids else "-"
open(f, 'a', encoding='utf-8').write(body)
print(f"✓ doc_write: 已【独占】追加 {len(body.encode())} 字节 → {f}(现有最大 #{mx};本条 {tag})")
PY
    rc=$?; rm -f "$tmp"; [ "$rc" -eq 0 ] || exit "$rc"
    ;;

  --replace)
    [ $# -eq 2 ] || { echo "usage: doc_write.sh <file> --replace <old> <new>" >&2; exit 2; }
    OLD="$1"; NEW="$2"
    python3 - "$F" "$OLD" "$NEW" <<'PY'
import os, sys, tempfile
f, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
if old == new:
    print("⛔ doc_write: --replace 的 old 与 new 【完全相同】⇒ 空操作,未写 ✗"
          "(常见成因:调用方用 bash `${v/pat/rep}` 构造 new,而 pat 以 `#` 开头 ⇒"
          " bash 把它当成\"锚定开头\"的标记而【静默不替换】⇒ 用 sed/python 构造,并确认 new != old)",
          file=sys.stderr)
    sys.exit(2)
src = open(f, encoding='utf-8').read()
cnt = src.count(old)
if cnt != 1:
    print(f"⛔ doc_write: 待替换文本出现 {cnt} 次(要求恰好 1 次)⇒ 未写", file=sys.stderr)
    sys.exit(4)
out = src.replace(old, new)
d = os.path.dirname(os.path.abspath(f)) or '.'
fd, t = tempfile.mkstemp(dir=d, prefix='.docwrite.')
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(out)
    os.replace(t, f)          # 原子替换:读者要么看到旧版、要么看到新版,不会看到半截 ✓
except BaseException:
    try: os.unlink(t)
    except OSError: pass
    raise
print(f"✓ doc_write: 已【独占】定点替换 → {f}")
PY
    ;;

  --truncate-at)
    # 从 <marker> 起删到文末(含紧邻其前的分隔线)—— 用于【撤销一次误记】
    [ $# -eq 1 ] || { echo "usage: doc_write.sh <file> --truncate-at <marker>" >&2; exit 2; }
    python3 - "$F" "$1" <<'PY'
import os, sys, tempfile
f, marker = sys.argv[1], sys.argv[2]
src = open(f, encoding='utf-8').read()
i = src.find(marker)
if i < 0:
    print(f"⛔ doc_write: 找不到标记 {marker!r} ⇒ 未写", file=sys.stderr)
    sys.exit(4)
head = src[:i].rstrip()
if head.endswith('---'):
    head = head[:-3].rstrip()
head += '\n'
d = os.path.dirname(os.path.abspath(f)) or '.'
fd, t = tempfile.mkstemp(dir=d, prefix='.docwrite.')
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(head)
    os.replace(t, f)
except BaseException:
    try: os.unlink(t)
    except OSError: pass
    raise
print(f"✓ doc_write: 已【独占】撤销 {len(src)-len(head)} 字节(自 {marker!r} 起至文末)→ {f}")
PY
    ;;

  *) usage ;;
esac
