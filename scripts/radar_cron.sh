#!/usr/bin/env bash
# 系统 cron 入口：定期调用 DSH headless 刷新参考项目雷达。
# 不把 API key 写进本文件；运行时从 root-only 的 DSH credential 文件读取。
set -euo pipefail

ROOT="/root/lvllm/repos/vllm-xiaotu-moe"
LOG_DIR="$ROOT/dev-docs/report/tuning/logs"
LOG="$LOG_DIR/radar_cron.log"
LOCK="/tmp/vllm_xtu_radar_cron.lock"
PROMPT='你是“参考项目雷达”的定期维护任务，不需要询问用户。工作目录固定为 /root/lvllm/repos/vllm-xiaotu-moe。

步骤：
1) 先运行 `bash scripts/research_preflight.sh`，并阅读命中的本地台账；不要凭记忆下结论。
2) 检查 DeepSeek 官方 org 的新仓库/release（DeepInfer、FlashMLA、DeepGEMM、DeepEP、DeepSelect、TileKernels、EPLB），以及 vLLM、SGLang、TensorRT-LLM、LMDeploy、KTransformers、PowerInfer、llama.cpp 的最新 release；并检查 lovedheart/sglang 的 qsa-on-main / rebase/qsa-on-main 分支（qfn MXFP4 / NVFP4 专用补丁）与两个模型卡 revision。
3) 用 GitHub 搜索 `moe inference`、`expert parallel`、`cpu offload llm`、`numa moe`，按 stars/updated 扫一遍；只关注有具体机制、能落到我们瓶颈（GPU 62% / CPU 36% / 热专家 / 长上下文）且有可验证判据的项目。
4) 只更新 `dev-docs/REFERENCE_RADAR.md`：新增项目写“链接 / 相关性 / 读什么 / 状态 / 核对日期”；旧结论标“历史，已被 X 取代”；不修改其他文件，不 commit，不 push。
5) 输出本次新增/更新摘要；若没有值得加入的新内容，明确写“本期无新增”，不要硬凑。若网络失败，记录失败原因，不要编造结果。'

# 仅用于人工烟测：设置 RADAR_CRON_PROMPT 可临时替换任务文本，cron 不设置它。
PROMPT="${RADAR_CRON_PROMPT:-$PROMPT}"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] skip: previous radar_cron still running" >> "$LOG"
  exit 0
fi

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="/root"
export DSH_HOME="${DSH_HOME:-/root/.dsh}"
export DEEPSEEK_API_KEY="$(
  python3 - <<'PY'
import re
p = "/root/.dsh/.credentials.yaml"
s = open(p, encoding="utf-8").read()
m = re.search(r"^\s*DEEPSEEK_API_KEY:\s*(\S+)", s, re.M)
print(m.group(1) if m else "")
PY
)"
if [ -z "${DEEPSEEK_API_KEY}" ]; then
  echo "[$(date -Is)] ERROR: DEEPSEEK_API_KEY not found in $DSH_HOME/.credentials.yaml" >> "$LOG"
  exit 1
fi

cd "$ROOT"
RADAR="dev-docs/REFERENCE_RADAR.md"
PENDING="dev-docs/RADAR_PENDING.md"
before_mtime="$(stat -c %Y "$RADAR" 2>/dev/null || echo 0)"
{
  echo
  echo "===== $(date -Is) radar_cron start ====="
} >> "$LOG"

set +e
timeout 3600 dsh --profile headless "$PROMPT" >> "$LOG" 2>&1
rc=$?
set -e

after_mtime="$(stat -c %Y "$RADAR" 2>/dev/null || echo 0)"
if [ "$rc" -eq 0 ] && [ "$after_mtime" -gt "$before_mtime" ]; then
  mkdir -p dev-docs/archive
  cat > "$PENDING" <<EOF
# 参考项目雷达待办（一次性提醒）

- 生成时间: $(date -Is)
- 触发: \`scripts/radar_cron.sh\` 检测到 \`dev-docs/REFERENCE_RADAR.md\` 有更新
- 请求: 在新会话里向用户**只报告一次**“参考外部资源库已更新，可启动一次调研比对”
- 之后: 把本文件移动到 \`dev-docs/archive/RADAR_PENDING_$(date +%Y%m%d).ack.md\`，避免重复提醒
- 建议: 用户同意后再读 \`REFERENCE_RADAR.md\` 的变更记录，不要自动展开调研
EOF
  echo "[$(date -Is)] radar updated → wrote $PENDING" >> "$LOG"
fi

echo "===== $(date -Is) radar_cron end rc=$rc =====" >> "$LOG"
exit "$rc"
