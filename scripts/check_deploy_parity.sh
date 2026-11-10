#!/usr/bin/env bash
# check_deploy_parity.sh —— 比对"本地真源"与"远端部署副本"的 scripts/ 是否一致
#
# 为什么需要（台账 #234）：所有验收脚本都从远端副本执行。若本地改过、忘了 scp，
# 远端跑的就是**旧判据**——而且它照样会打印"通过"，只是通过的是旧标准。
# 这类漂移在"改一处判据 → 跑一轮 → 看结果"的循环里极容易发生。
#
# 【必须在本地 harness 执行】（本脚本内部 ssh 到远端读 md5）
# 用法： bash scripts/check_deploy_parity.sh [-v]
# 退出码：0 = 一致；1 = 有漂移；2 = 连不上远端
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
REMOTE_DIR=${REMOTE_DIR:-/home/user/lvllm/vllm-xiaotu-moe/scripts}
VERBOSE=0
[ "${1:-}" = "-v" ] && VERBOSE=1

SSH=(timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST")
if ! "${SSH[@]}" true >/dev/null 2>&1; then
    echo "⛔ 连不上远端（$HOST）⇒ 本脚本必须在【本地 harness】执行"; exit 2
fi

local_list=$(cd "$ROOT/scripts" && md5sum *.sh *.py 2>/dev/null | sort -k2)
remote_list=$("${SSH[@]}" "cd $REMOTE_DIR && md5sum *.sh *.py 2>/dev/null | sort -k2")

drift=0; missing=0; extra=0
while read -r sum name; do
    [ -z "${name:-}" ] && continue
    r=$(printf '%s\n' "$remote_list" | awk -v n="$name" '$2==n{print $1}')
    if [ -z "$r" ]; then
        [ "$VERBOSE" = 1 ] && echo "⬇ 远端缺: $name"
        missing=$((missing+1)); drift=$((drift+1))
    elif [ "$r" != "$sum" ]; then
        echo "⚠️ 漂移: $name  本地 ${sum:0:8} / 远端 ${r:0:8}"
        drift=$((drift+1))
    elif [ "$VERBOSE" = 1 ]; then
        echo "✅ $name"
    fi
done <<< "$local_list"

while read -r sum name; do
    [ -z "${name:-}" ] && continue
    printf '%s\n' "$local_list" | awk -v n="$name" '$2==n{f=1} END{exit !f}' || {
        [ "$VERBOSE" = 1 ] && echo "⬆ 远端多出: $name"
        extra=$((extra+1))
    }
done <<< "$remote_list"

echo
echo "本地文件 $(printf '%s\n' "$local_list" | grep -c . ) 个 | 远端 $(printf '%s\n' "$remote_list" | grep -c . ) 个 | 缺 $missing | 多 $extra"
if [ "$drift" = 0 ]; then echo "✅ 部署一致（判据与本地真源相同）"; exit 0; fi
echo "❌ 有 $drift 处漂移 ⇒ 远端跑的是旧判据，先同步（scp 缺失/变化的文件）再跑验收"
exit 1
