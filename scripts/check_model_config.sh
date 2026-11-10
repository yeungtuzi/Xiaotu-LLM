#!/usr/bin/env bash
# check_model_config.sh —— 用 vLLM 自己的 get_config 解析远端模型目录（**本地工具**，内部 ssh 到远端）
#
# 为什么值得单独一步（台账 #244）：配置解析失败会在**引擎初始化后**才炸，
# 而那时已经花了 20–40 分钟的加载/编译/预热。这个检查 30 秒就能给出同样信息。
#
# 用法： bash scripts/check_model_config.sh
# 退出码：0 = 可解析且口径符合；1 = 口径不符；2 = 环境问题（连不上/找不到树/导入失败）
set -uo pipefail
SSH_CFG=${SSH_CFG:-/root/lvllm/tmp/sshcfg}
HOST=${HOST:-gpu}
TREE=${TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}
DIR=${DIR:-/home/user/lvllm/models/dsv41f-1017-weights}
PY=${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}

SSH=(timeout 240 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "$SSH_CFG" "$HOST")
"${SSH[@]}" true >/dev/null 2>&1 || { echo "⛔ 连不上远端（$HOST）⇒ 本脚本必须在【本地 harness】执行"; exit 2; }

out=$("${SSH[@]}" "cd $TREE || exit 9; CUDA_VISIBLE_DEVICES='' $PY - <<PY 2>&1
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
from vllm.transformers_utils.config import get_config
c = get_config('$DIR', trust_remote_code=True)
print('CLASS', type(c).__name__)
for k in ('num_hidden_layers','num_nextn_predict_layers','sliding_window','index_topk','max_position_embeddings'):
    print('KV', k, getattr(c,k,None))
qc = getattr(c,'quantization_config',None)
print('QC', 'yes' if qc else 'no', (','.join(sorted(qc.keys())) if isinstance(qc,dict) else ''))
print('ARCH', ','.join(getattr(c,'architectures',[]) or []))
PY")
rc=$?
echo "$out" | grep -E '^(CLASS|KV|QC|ARCH) ' | sed 's/^/   /'

[ "$rc" = 0 ] || { echo "⛔ 解析失败（rc=$rc）⇒ 先看上面的报错"; exit 2; }

bad=0
chk() { # 名称 实际 期望
    if [ "$2" = "$3" ]; then printf '✅ %s = %s\n' "$1" "$2"
    else printf '❌ %s = %s（期望 %s）\n' "$1" "$2" "$3"; bad=1; fi
}
cls=$(echo "$out" | awk '/^CLASS /{print $2}')
chk "config 类" "${cls:-}" "DeepseekV41Config"
for pair in "num_hidden_layers:40" "num_nextn_predict_layers:3" "sliding_window:128" \
            "index_topk:512" "max_position_embeddings:1048576"; do
    k=${pair%%:*}; want=${pair##*:}
    got=$(echo "$out" | awk -v k="$k" '$1=="KV" && $2==k {print $3}')
    chk "$k" "${got:-?}" "$want"
done
qc=$(echo "$out" | awk '/^QC /{print $2}')
[ "$qc" = "yes" ] && echo "✅ quantization_config 存在（量化路径会被启用）" || { echo "❌ 没有 quantization_config"; bad=1; }
arch=$(echo "$out" | awk '/^ARCH /{print $2}')
chk "architectures" "${arch:-}" "DeepseekV41ForCausalLM"

echo
[ "$bad" = 0 ] && { echo "RESULT: ✅ 配置可解析、口径符合（可以进入预检/起服务）"; exit 0; }
echo "RESULT: ❌ 口径不符（见上）"; exit 1
