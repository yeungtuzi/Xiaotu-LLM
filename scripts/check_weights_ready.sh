#!/usr/bin/env bash
# check_weights_ready.sh —— 权重目录"就绪"的机械校验（不需要加载权重、不需要起服务）
#
# 只读：不启动/不停止任何进程 ⇒ 不涉及 R38 起停审计。
#
# 检查项（任一不过 ⇒ 退出码 1，并打印 ❌ 行）：
#   1. .safetensors 文件数 == 48，且 .incomplete 文件数 == 0
#   2. index.json 声明的分片数 == 48
#   3. text_config: num_hidden_layers=40 / num_nextn_predict_layers=3 /
#      sliding_window=128 / index_topk=512 / max_position_embeddings=1048576
#   4. MTP 权重键是 mtp.0/1/2（**不是** layers.40..42），最大 layers=39
#   5. （可选 --sha）两个 101.5 GB 大分片 sha256 + 精确字节数
#
# 用法： bash scripts/check_weights_ready.sh [--sha]
set -uo pipefail

SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
DIR=${DIR:-/home/user/lvllm/models/dsv41f-1017-weights}
PROBE=$(dirname "$(readlink -f "$0")")/_weights_probe.py
DO_SHA=0
[ "${1:-}" = "--sha" ] && DO_SHA=1

SSH=(timeout 60 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST")
bad=0

# ⚠️ 本脚本要在【本地 harness】跑：它不读本地模型目录，而是 ssh 到远端去读。
#    若在远端直接跑，它会尝试 ssh 到自己的别名（通常失败）⇒ 明确提示，而不是吐一串 ?。
if ! "${SSH[@]}" true >/dev/null 2>&1; then
    echo "⛔ 连不上远端（$HOST）⇒ 本脚本必须在【本地 harness】执行。"
    echo "   若当前就在远端，请直接跑："
    echo "     WPROBE_DIR=$DIR python3 scripts/_weights_probe.py"
    echo "     ls $DIR/*.safetensors | wc -l ; ls $DIR/*.incomplete 2>/dev/null | wc -l"
    exit 2
fi
fail() { echo "❌ $*"; bad=1; }
ok()   { echo "✅ $*"; }

counts=$("${SSH[@]}" "d=$DIR; echo \"SAT \$(ls \$d/*.safetensors 2>/dev/null | wc -l) INC \$(ls \$d/*.incomplete 2>/dev/null | wc -l)\"" 2>&1)
sat=$(printf '%s\n' "$counts" | awk '/^SAT /{print $2}')
inc=$(printf '%s\n' "$counts" | awk '/^SAT /{print $4}')
[ "${sat:-0}" = "48" ] && ok "分片文件 48 个" || fail "分片文件 ${sat:-?} 个（期望 48）"
[ "${inc:-1}" = "0" ]  && ok "无 .incomplete" || fail "仍有 ${inc:-?} 个 .incomplete"

facts=$("${SSH[@]}" "WPROBE_DIR=$DIR python3 -" <"$PROBE" 2>&1 | awk '/^[FM] /')
printf '%s\n' "$facts" | sed 's/^/   /'
F=$(printf '%s\n' "$facts" | awk '/^F /{print $2,$3,$4,$5,$6}')
M=$(printf '%s\n' "$facts" | awk '/^M /{print $2,$3,$4}')
L=$(printf '%s' "$F" | awk '{print $1}'); N=$(printf '%s' "$F" | awk '{print $2}')
W=$(printf '%s' "$F" | awk '{print $3}'); K=$(printf '%s' "$F" | awk '{print $4}')
P=$(printf '%s' "$F" | awk '{print $5}')
[ "$L" = "40" ]      && ok "num_hidden_layers = 40"            || fail "num_hidden_layers = ${L:-?}（期望 40）"
[ "$N" = "3" ]       && ok "num_nextn_predict_layers = 3"      || fail "num_nextn_predict_layers = ${N:-?}（期望 3）"
[ "$W" = "128" ]     && ok "sliding_window = 128"              || fail "sliding_window = ${W:-?}（期望 128）"
[ "$K" = "512" ]     && ok "index_topk = 512"                  || fail "index_topk = ${K:-?}（期望 512）"
[ "$P" = "1048576" ] && ok "max_position_embeddings = 1048576" || fail "max_position_embeddings = ${P:-?}（期望 1048576）"

MT=$(printf '%s' "$M" | awk '{print $1}')
ML=$(printf '%s' "$M" | awk '{print $2}')
MS=$(printf '%s' "$M" | awk '{print $3}')
[ "$MT" = "0,1,2" ] && ok "MTP 权重键 = mtp.0/1/2（不是 layers.40..42）" || fail "MTP 权重键 = ${MT:-?}（期望 0,1,2）"
[ "$ML" = "39" ]    && ok "最大 layers = 39（共 40 层 0..39）"           || fail "最大 layers = ${ML:-?}（期望 39）"
[ "$MS" = "48" ]    && ok "index 声明分片 = 48"                          || fail "index 声明分片 = ${MS:-?}（期望 48）"

if [ "$DO_SHA" = 1 ]; then
    # ⛔ 2026-11-10 修（台账 #272）：原来复用 `SSH`（timeout 60）⇒ 对 101 GB 分片做 sha256
    #    必然超时 ⇒ 输出为空 ⇒ 判成"分片不符"，差点让人以为权重下载坏了、要重下 200 GB。
    #    ⇒ sha 单独用长超时（默认 1800 s），可用 SHA_TIMEOUT 覆盖。
    SHA_TIMEOUT=${SHA_TIMEOUT:-1800}
    got=$(timeout "$SHA_TIMEOUT" ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST" \
          "cd $DIR && sha256sum model-00047-of-00048.safetensors model-00048-of-00048.safetensors; \
           stat -c '%s %n' model-00047-of-00048.safetensors model-00048-of-00048.safetensors" 2>&1)
    s47=$(printf '%s\n' "$got" | awk '$2 ~ /00047/ && length($1)==64 {print $1}')
    s48=$(printf '%s\n' "$got" | awk '$2 ~ /00048/ && length($1)==64 {print $1}')
    z47=$(printf '%s\n' "$got" | awk '$2 ~ /00047/ && $1 ~ /^[0-9]+$/ {print $1}')
    z48=$(printf '%s\n' "$got" | awk '$2 ~ /00048/ && $1 ~ /^[0-9]+$/ {print $1}')
    if [ "$s47" = "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5" ] && [ "$z47" = "101535150936" ]; then
        ok "47 号分片 sha256 + 字节数一致"
    else
        fail "47 号分片不符：sha=${s47:-空} size=${z47:-空}"
    fi
    if [ "$s48" = "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd" ] && [ "$z48" = "101537926640" ]; then
        ok "48 号分片 sha256 + 字节数一致"
    else
        fail "48 号分片不符：sha=${s48:-空} size=${z48:-空}"
    fi
fi

echo
if [ "$bad" = 0 ]; then
    echo "RESULT: ✅ 权重目录就绪（可进入 RESUME 第 1 步预检）"
    exit 0
fi
echo "RESULT: ❌ 未就绪/不符（见上方 ❌ 行）"
exit 1
