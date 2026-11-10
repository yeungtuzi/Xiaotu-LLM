#!/usr/bin/env bash
# =============================================================================
# prepare_model_dir.sh —— 真权重目录的**起服务前预检**（只读 + 仅补缺失的索引）
#
# 背景（2026-11-10 实测）：DeepSeek-V4.1-Flash-1017 的官方权重仓**不含** tokenizer，
# 也不保证带 `model.safetensors.index.json`；而 vLLM 需要 tokenizer 才能起服务。
# 本脚本在起服务之前把这些缺失项查出来（缺 tokenizer 就从本机已有的同族切片目录复制），
# 不启动服务、不占 GPU。
#
# 用法：
#   bash scripts/prepare_model_dir.sh <模型目录> [--tokenizer-from <目录>] [--write-index]
# 退出码：0=齐全（或已补齐）；1=缺东西且未能补；2=用法错误
# =============================================================================
set -uo pipefail

MODEL_DIR="${1:-}"; shift || true
TOK_FROM="${TOK_FROM:-/home/user/lvllm/models/ds41f-l8}"
WRITE_INDEX=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tokenizer-from) TOK_FROM="${2:?}"; shift 2 ;;
        --write-index) WRITE_INDEX=1; shift ;;
        *) echo "[prep] ⛔ 未知参数 $1" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ] || { echo "[prep] ⛔ 用法：prepare_model_dir.sh <模型目录>" >&2; exit 2; }

fail=0
say() { printf '[prep] %s\n' "$*"; }

# ⚠️ 生成索引需要 `safetensors` 模块 —— 系统 python 通常没有，必须用 vLLM 环境的 python
PYTHON="${PYTHON:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

# ① config.json
[ -f "$MODEL_DIR/config.json" ] && say "✅ config.json" || { say "❌ 缺 config.json"; fail=1; }

# ② tokenizer（必须有 tokenizer.json；tokenizer_config.json 可选但强烈建议）
if [ -f "$MODEL_DIR/tokenizer.json" ]; then
    say "✅ tokenizer.json"
else
    if [ -d "$TOK_FROM" ] && [ -f "$TOK_FROM/tokenizer.json" ]; then
        cp -n "$TOK_FROM/tokenizer.json" "$MODEL_DIR/" 2>/dev/null
        [ -f "$TOK_FROM/tokenizer_config.json" ] && cp -n "$TOK_FROM/tokenizer_config.json" "$MODEL_DIR/" 2>/dev/null
        [ -f "$MODEL_DIR/tokenizer.json" ] && say "✅ tokenizer.json（已从 $TOK_FROM 复制）" || { say "❌ tokenizer 复制失败"; fail=1; }
    else
        say "❌ 缺 tokenizer.json，且 $TOK_FROM 也没有"; fail=1
    fi
fi

# ③ chat template：官方仓**没有** ⇒ 只能用 /v1/completions（脚本会提示）
if grep -qs '"chat_template"' "$MODEL_DIR/tokenizer_config.json"; then
    say "✅ chat template 存在（可用 /v1/chat/completions）"
else
    say "⚠️ 没有 chat template ⇒ 评测/复现脚本必须用 /v1/completions（本仓已按此改）"
fi

# ④ safetensors 索引
shards=$(ls "$MODEL_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l)
inc=$(ls "$MODEL_DIR"/*.incomplete 2>/dev/null | wc -l)
say "分片：完整 $shards 个，未完成 $inc 个"
# ⛔ 有 .incomplete ⇒ 下载未完成 ⇒ 必须 fail-closed（否则会得出"可以起服务"的错误结论，实测踩过）
if [ "$inc" -gt 0 ]; then
    say "❌ 还有 $inc 个分片未下载完（.incomplete）⇒ 不能起服务；等下载完成后再跑本脚本"
    fail=1
fi
if [ "$shards" -lt 48 ]; then
    say "⚠️ 完整分片 $shards < 48 ⇒ 权重不齐（若该模型确实不是 48 片，请忽略此行）"
fi
if [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    say "✅ model.safetensors.index.json"
elif [ "$WRITE_INDEX" = "1" ]; then
    say "… 生成索引（只读各分片 header）"
    "$PYTHON" - "$MODEL_DIR" <<'PY'
import json, sys, glob, os
d = sys.argv[1]
try:
    from safetensors import safe_open
except ImportError:
    print("[prep] ❌ 需要 safetensors 才能生成索引"); raise SystemExit(1)
wm, total = {}, 0
for f in sorted(glob.glob(os.path.join(d, "model-*-of-*.safetensors"))):
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            wm[k] = os.path.basename(f)
out = os.path.join(d, "model.safetensors.index.json")
with open(out, "w") as fh:
    json.dump({"metadata": {"total_size": total}, "weight_map": wm}, fh, indent=1)
print(f"[prep] ✅ 已写 {out}（{len(wm)} 个张量）")
PY
    [ -f "$MODEL_DIR/model.safetensors.index.json" ] || { say "❌ 索引生成失败"; fail=1; }
else
    say "⚠️ 缺 model.safetensors.index.json（有些加载器可按文件名排序兜底；要生成加 --write-index）"
fi

# ⑤ 分片完整性（只读 header）：起服务前先确认没有半截文件
if [ "${CHECK_SHARDS:-0}" = "1" ]; then
    say "… 校验完整分片的 header（只读，不解权重）"
    if "$PYTHON" - "$MODEL_DIR" <<'PYEOF'
import glob, os, sys
try:
    from safetensors import safe_open
except ImportError:
    print("[prep] 无 safetensors，跳过 header 校验")
    raise SystemExit(0)
d = sys.argv[1]
bad, tot = 0, 0
files = sorted(glob.glob(os.path.join(d, "model-*-of-*.safetensors")))
for f in files:
    try:
        with safe_open(f, framework="pt") as fh:
            n = len(list(fh.keys()))
        tot += n
        if n == 0:
            print("[prep] 异常：" + os.path.basename(f) + " 没有张量")
            bad += 1
    except Exception as exc:
        print("[prep] 异常：" + os.path.basename(f) + " header 读不了：" + repr(exc))
        bad += 1
print("[prep] 分片 header 校验：%d 个文件 / %d 个张量 / %d 个异常" % (len(files), tot, bad))
raise SystemExit(1 if bad else 0)
PYEOF
    then
        :
    else
        fail=1
    fi
fi

if [ "$fail" = "0" ]; then
    say "结论：可以起服务（记得 PC 开、decoder-side SWA bounded replay 开）"
    exit 0
fi
say "结论：❌ 还有缺失项，先补齐再起服务"
exit 1
