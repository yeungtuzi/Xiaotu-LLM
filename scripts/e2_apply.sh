#!/usr/bin/env bash
# =============================================================================
# e2_apply.sh —— 把 D1+F1 组合补丁应用到**远端部署树**，并验证"能 import"（**不启停服务**）
#
# 为什么单独成脚本：E2 的时间都在"重启 20 分钟"上，应用补丁这一步必须**零思考、可回退**。
# 用法：
#   bash scripts/e2_apply.sh [apply|revert|check]
# 退出码：0=成功；1=失败（已在日志里打印原因）；2=用法/环境问题
# =============================================================================
set -uo pipefail
TREE="${TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}"
# 默认只上 **D1**（上游同款修复），F1 是后备假设：只有 D1 单独失败时才叠
# 需要 D1+F1 时显式 PATCH=/home/user/lvllm/run/combo.patch
PATCH="${PATCH:-/home/user/lvllm/run/d1.patch}"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
ACTION="${1:-check}"

say() { printf '[e2] %s\n' "$*"; }
[ -d "$TREE/vllm" ] || { say "⛔ 找不到 vLLM 树：$TREE"; exit 2; }

_patched() {
    # 两条判据：D1 的函数是否只有 1 个定义；F1 的标记是否存在
    local d1 f1
    # ⚠️ `grep -c` 无匹配时**输出 0 且退出码 1** ⇒ 不能写 `|| echo 0`（会得到 "0 0"）
    d1=$(grep -c "^def compute_token_to_req_indices" "$TREE/vllm/v1/attention/ops/metadata.py" 2>/dev/null || true)
    f1=$(grep -c "xtu-F1" "$TREE/vllm/models/deepseek_v41/nvidia/model.py" 2>/dev/null || true)
    d1=${d1:-0}; f1=${f1:-0}
    echo "$d1 $f1"
}

case "$ACTION" in
check)
    read -r d1 f1 <<<"$(_patched)"
    say "D1 定义数=$d1（期望 1）F1 标记数=$f1（期望 2 或 0）"
    # 合法状态只有三种：(0,0) 未应用 / (1,0) 只 D1 / (1,2) D1+F1
    case "$d1:$f1" in
        0:0) say "ℹ️ 未应用"; exit 0 ;;
        1:0) say "ℹ️ 只应用了 D1（E2 的默认臂）"; exit 0 ;;
        1:2) say "✅ 已应用 D1+F1"; exit 0 ;;
        *)   say "⚠️ 非法/半应用状态（d1=$d1 f1=$f1）⇒ 先修好再动"; exit 1 ;;
    esac
    ;;
apply)
    read -r d1 f1 <<<"$(_patched)"
    if [ "$d1" = "1" ] && { [ "$f1" = "0" ] || [ "$f1" = "2" ]; }; then say "✅ 已应用（d1=$d1 f1=$f1），跳过"; exit 0; fi
    if [ "$d1" != "0" ] || [ "$f1" != "0" ]; then say "⛔ 非法状态（d1=$d1 f1=$f1），拒绝叠加"; exit 1; fi
    [ -f "$PATCH" ] || { say "⛔ 找不到补丁 $PATCH（先 scp 过来）"; exit 2; }
    # ⚠️ md5 白名单：远端 run/ 里同时躺着 d1.patch 与 **旧版 D1.patch / D1_step_mask_recompute.patch**
    #    （md5 1cb78e4e…，2907 B，与本地权威 74378ee2… 不同）⇒ 按名字拿到旧版会白验一轮。
    _md5=$(md5sum "$PATCH" | awk '{print $1}')
    case "$_md5" in
        74378ee29e7d31975c23f7a49594a7ce) say "补丁 = d1.patch（D1 单独，权威）" ;;
        8a02b436a8db80e7d1859e3562202015) say "补丁 = combo.patch（D1+F1）" ;;
        5a7600e98fa1b9eadbe8f7633902f173) say "补丁 = f1.patch（仅 F1，非常规）" ;;
        *) say "⛔ 补丁 md5 不在白名单：$_md5 ($PATCH)"; \
           say "   已知陷阱：旧版 D1.patch/D1_step_mask_recompute.patch = 1cb78e4e72b25d443a48bf9d3f202415（不要用）"; \
           say "   权威文件见本地 patches/replay/D1_step_mask_recompute.patch = 74378ee2…"; exit 1 ;;
    esac
    if ! (cd "$TREE" && patch -p1 --dry-run < "$PATCH" >/dev/null 2>&1); then
        say "⛔ dry-run 失败（上下文不符）⇒ 不要强行应用"; exit 1
    fi
    (cd "$TREE" && patch -p1 < "$PATCH" >/dev/null) || { say "⛔ 应用失败"; exit 1; }
    read -r d1 f1 <<<"$(_patched)"
    [ "$d1" = "1" ] || { say "⛔ 应用后校验不通过（d1=$d1 f1=$f1；D1 必须生效）"; exit 1; }
    case "$f1" in 0|2) ;; *) say "⛔ F1 标记异常（f1=$f1）"; exit 1 ;; esac
    if ! (cd "$TREE" && "$PY" -c "import vllm.models.deepseek_v41" >/dev/null 2>&1); then
        say "⛔ import 失败 ⇒ 回退"; (cd "$TREE" && patch -p1 -R < "$PATCH" >/dev/null 2>&1); exit 1
    fi
    say "✅ 已应用 + import 通过（⚠️ 需重启服务才生效；本脚本不重启）"
    ;;
revert)
    [ -f "$PATCH" ] || { say "⛔ 找不到补丁 $PATCH，无法按单回退"; exit 2; }
    (cd "$TREE" && patch -p1 -R < "$PATCH" >/dev/null 2>&1)
    read -r d1 f1 <<<"$(_patched)"
    if [ "$d1" = "0" ] && [ "$f1" = "0" ]; then say "✅ 已回退干净 + import 检查：$( cd "$TREE" && "$PY" -c "import vllm.models.deepseek_v41" >/dev/null 2>&1 && echo OK || echo FAIL )"; exit 0; fi
    say "⛔ 回退后仍残留（d1=$d1 f1=$f1）"; exit 1
    ;;
*)
    say "用法：e2_apply.sh [apply|revert|check]"; exit 2
    ;;
esac
