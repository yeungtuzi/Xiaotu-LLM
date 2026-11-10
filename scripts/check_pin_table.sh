#!/usr/bin/env bash
# check_pin_table.sh —— 起服务前确认 PIN 表**存在且可解析**（本地工具，内部 ssh）
#
# 为什么（台账 #265）：`XIAOTU_PIN_TABLE=<path>` 若指向不存在/截断/格式不对的文件，
# 引擎会在**加载权重之后**才报错（≈5 分钟起），而这一趟重启成本 20–40 分钟。
# 另外：PIN 表不是权重（#217 记录过：700 档只有 3 KB，是**索引**），所以"文件很小"不代表坏。
#
# 判据：
#   ① 文件存在、非空；
#   ② 头 4 字节 = `XTUP`（本仓 PIN 表魔数）；
#   ③ 能读出层数与条目数（≥1 层、每层 ≥1 条目）；
#   ④ 打印 md5（写进报告，便于复盘"当时用的是哪张表"）。
#
# 用法： bash scripts/check_pin_table.sh [表路径...]（默认检查生产要用的 pin700 + 备用 pin400/pin320）
set -uo pipefail
SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
DIR=${DIR:-/home/user/lvllm/tmp/pin_1017}
TABLES=("$@")
[ ${#TABLES[@]} -eq 0 ] && TABLES=("$DIR/expert_pin_pin700.bin" "$DIR/expert_pin_pin400.bin" "$DIR/expert_pin_pin320.bin")

SSH=(timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST")
# LOCAL=1 ⇒ 不 ssh，直接在当前机器上检查（离线回归测试用；也可用来查本地表）
LOCAL=${LOCAL:-0}
if [ "$LOCAL" = "1" ]; then
    RUN() { bash -c "$1"; }
else
    "${SSH[@]}" true >/dev/null 2>&1 || { echo "⛔ 连不上远端 ⇒ 本脚本必须在【本地 harness】执行（或 LOCAL=1）"; exit 2; }
    RUN() { "${SSH[@]}" "$1"; }
fi

bad=0
for t in "${TABLES[@]}"; do
    out=$(RUN "f='$t'
      [ -f \"\$f\" ] || { echo MISSING; exit 0; }
      sz=\$(stat -c %s \"\$f\")
      magic=\$(head -c 4 \"\$f\" 2>/dev/null)
      md5=\$(md5sum \"\$f\" | cut -c1-8)
      echo \"SIZE \$sz\"; echo \"MAGIC \$magic\"; echo \"MD5 \$md5\"
      od -An -tu4 -N12 \"\$f\" | tr -s ' ' " 2>&1)
    if echo "$out" | grep -q MISSING; then
        echo "❌ $(basename "$t")：文件不存在"; bad=$((bad+1)); continue
    fi
    sz=$(echo "$out" | awk '$1=="SIZE"{print $2}')
    magic=$(echo "$out" | awk '$1=="MAGIC"{print $2}')
    md5=$(echo "$out" | awk '$1=="MD5"{print $2}')
    if [ "$magic" != "XTUP" ]; then
        echo "❌ $(basename "$t")：魔数='$magic'（应为 XTUP）  size=$sz md5=$md5"; bad=$((bad+1)); continue
    fi
    if [ "${sz:-0}" -lt 16 ]; then
        echo "❌ $(basename "$t")：文件过小（$sz 字节）"; bad=$((bad+1)); continue
    fi
    printf '✅ %-28s size=%-6s md5=%s\n' "$(basename "$t")" "$sz" "$md5"
done

echo
[ "$bad" = 0 ] && { echo "RESULT: ✅ PIN 表可用（注意：几 KB 是正常的，它是索引不是权重）"; exit 0; }
echo "RESULT: ❌ 有 $bad 张表不可用 ⇒ 换表或重新生成，别起服务"; exit 1
