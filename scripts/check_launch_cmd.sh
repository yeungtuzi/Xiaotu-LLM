#!/usr/bin/env bash
# check_launch_cmd.sh —— 校验 **文档里那条生产启动命令** 的参数名是否都还在 `serve_ds41.sh` 的白名单里
#
# 为什么（台账 #264）：起服务的命令是"手抄"的，而 `serve_ds41.sh` 会把**白名单外**的参数直接拒绝。
# 一旦文档与脚本漂移（改名、删参数），那一刻才发现 = 又要多等一次 20–40 分钟的重启。
# 本脚本只读、不起服务：把 RESUME §3 的命令块与脚本白名单做机械比对。
#
# 用法： bash scripts/check_launch_cmd.sh
# 退出码：0 = 全部参数都在白名单；1 = 有不在白名单的（会被拒绝启动）
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/dev-docs/RESUME_AFTER_WEIGHTS_2026-11-10.md"
SERVE="$ROOT/scripts/serve_ds41.sh"
[ -f "$DOC" ] || { echo "⛔ 找不到文档：$DOC"; exit 1; }
[ -f "$SERVE" ] || { echo "⛔ 找不到启动脚本：$SERVE"; exit 1; }

# 抓 §3 段落里"大写 KEY=" 形式（环境变量形式也算：它们由 serve_ds41.sh 读环境）
# ⚠️ 文档里有 v1/v2 两套步骤，先切到【v2】段，再取其中的 §3（否则会抓到 v1 的 E1 命令，误报 CNT/INSTANCE_TAG）
keys=$(awk '/【v2/{v2=1} v2' "$DOC" | awk '/^## 3\./,/^## 4\./' \
       | grep -oE '\b[A-Z][A-Z0-9_]{2,}=' | tr -d '=' | sort -u)

# serve_ds41.sh 白名单：case 分支里的字面量标签，以及环境变量读取处 ${KEY:-...}
whitelist=$( { grep -oE '^\s+[A-Z][A-Z0-9_]{2,}\)' "$SERVE" | tr -d ' )';
              grep -oE '\$\{[A-Z][A-Z0-9_]{2,}:-' "$SERVE" | sed 's/^\${//; s/:-$//'; } | sort -u )

bad=0; n=0
for k in $keys; do
    n=$((n+1))
    # 忽略明显不是启动参数的（路径/命令里的词）
    case "$k" in CUDA_VISIBLE_DEVICES|NVCC|XIAOTU_PIN_TABLE) ok=1 ;; *) ok=0 ;; esac
    if [ "$ok" = 1 ]; then
        printf '✅ %-22s（环境变量，由引擎消费）\n' "$k"; continue
    fi
    if printf '%s\n' "$whitelist" | grep -qx "$k"; then
        printf '✅ %-22s\n' "$k"
    else
        printf '❌ %-22s 不在 serve_ds41.sh 白名单里 ⇒ 启动会被拒绝\n' "$k"
        bad=$((bad+1))
    fi
done

echo
echo "共检查 $n 个参数名"
[ "$bad" = 0 ] && { echo "✅ 文档里的启动命令与脚本白名单一致"; exit 0; }
echo "❌ $bad 个参数名不匹配（先改文档或脚本，再去起服务）"; exit 1
