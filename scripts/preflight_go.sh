#!/usr/bin/env bash
# preflight_go.sh —— 起服务前的**唯一总闸**：把所有前置判据按顺序跑一遍，给出 GO / NO-GO
#
# 设计原则：
#   * 只做"调用 + 汇总"，判据本身都在各自脚本里（避免第二套真理）；
#   * 任何一项失败 ⇒ 立刻 NO-GO（fail-closed），并打印**下一步该做什么**；
#   * 默认**不启停任何服务**（DRY_RUN=1 可完全离线自测）。
#
# 用法：
#   bash scripts/preflight_go.sh                # 完整前置检查（本地工具，内部 ssh）
#   DRY_RUN=1 bash scripts/preflight_go.sh      # 只跑本地项（自检/补丁 md5），跳过远端项
#   SKIP_MARLIN=1 bash scripts/preflight_go.sh  # 明示本轮不用 MARLIN（会打印警告）
#
# 退出码：0 = GO；1 = NO-GO
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=${DRY_RUN:-0}
SKIP_MARLIN=${SKIP_MARLIN:-0}
fails=0
step=0

run() { # 标题 命令...
    local title="$1"; shift
    step=$((step+1))
    printf '\n── [%d] %s\n' "$step" "$title"
    if "$@"; then printf '   → 通过\n'; else printf '   → ❌ 失败\n'; fails=$((fails+1)); fi
}

skip() { step=$((step+1)); printf '\n── [%d] %s\n   → 跳过（%s）\n' "$step" "$1" "${2:-DRY_RUN}"; }

# 1) 判据体系自检（纯本地）
#    PREFLIGHT_ACTIVE=1 告诉自检套件"跳过 test_preflight_go.sh"，避免 preflight ↔ 测试 递归。
export PREFLIGHT_ACTIVE=1
run "判据体系自检（语法 3.10 + 离线单测）" bash "$ROOT/scripts/run_harness_selftests.sh" --quiet

# 2) 启动补丁 md5（纯本地；按 md5 认人）
run "仪器补丁存在且 md5 正确" bash -c '
  p="$1/patches/diag/decoder_swa_bounded_replay_trigger_dbg.patch"
  [ -f "$p" ] || { echo "   缺文件：$p"; exit 1; }
  want=d0f31200b3077fffb663902e0a9e5072; got=$(md5sum "$p" | cut -d" " -f1)
  [ "$got" = "$want" ] || { echo "   md5=$got（期望 $want）"; exit 1; }
  echo "   md5=$got ✓"' _ "$ROOT"

# 3) D1 补丁：**按 md5 认人，不按文件名**（远端 run/ 里有同名的旧版，见 #218）
run "D1 补丁 md5 白名单（不依赖文件名）" bash -c '
  want=74378ee2
  hit=""
  for f in "$1"/patches/replay/*.patch; do
      [ -f "$f" ] || continue
      got=$(md5sum "$f" | cut -c1-8)
      [ "$got" = "$want" ] && hit="$f"
  done
  [ -n "$hit" ] || { echo "   patches/replay/ 下没有任何文件 md5=$want…"; ls -1 "$1"/patches/replay/ 2>/dev/null | sed "s/^/     /"; exit 1; }
  echo "   $(basename "$hit")  md5=$want… ✓"' _ "$ROOT"

# 3b) 文档里的生产启动命令 vs 脚本白名单（纯本地；防止临场抄错参数被拒）
run "文档启动命令与 serve_ds41 白名单一致" bash "$ROOT/scripts/check_launch_cmd.sh"

if [ "$DRY" = 1 ]; then
    skip "部署一致性（远端副本 vs 本地真源）"
    skip "权重就绪 + sha256"
    skip "模型配置可解析（vLLM get_config）"
    skip "MARLIN 融合内核可用（AVAILABLE True）"
    skip "GPU/内存门 + 端口/PID 归属"
else
    run "部署一致性（远端副本 vs 本地真源）" bash "$ROOT/scripts/check_deploy_parity.sh"
    # 4b) 仪器补丁**是否已打进远端那棵树**（只看本地 md5 不够：文件在 ≠ 已应用）
    #     不通过就别起服务 —— 否则 E1 会缺 `[xtu-swa-bounded-replay]` 证据行，白烧 20–40 分钟。
    run "仪器补丁已在远端 vLLM 树里" bash -c '
      n=$(timeout 30 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "${SSH_CFG:-/root/lvllm/tmp/sshcfg}" "${HOST:-gpu}" \
          "grep -c xtu-swa-bounded-replay /home/user/lvllm/process_data/ref/repos/vllm-mainline/vllm/models/deepseek_v41/nvidia/model_state.py 2>/dev/null || echo 0")
      echo "   命中行数=$n"
      [ "${n:-0}" -ge 2 ] || { echo "   未应用 ⇒ 先按 RESUME 2.6 打补丁（$n 行 < 2）"; exit 1; }
      # 6b) 基线轮里 **D1 必须还没打**：若已应用，E1 测的就不是"未修基线"，整轮作废。
      #     第二阶段（E2 前）用 EXPECT_D1=1 跑同一条总闸。
      d1=$(timeout 30 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "${SSH_CFG:-/root/lvllm/tmp/sshcfg}" "${HOST:-gpu}" \
           "grep -c 'Recompute the validity mask' /home/user/lvllm/process_data/ref/repos/vllm-mainline/vllm/v1/attention/backends/mla/sparse_swa.py 2>/dev/null || echo 0")
      echo "   D1 已应用标记数=$d1（EXPECT_D1=${EXPECT_D1:-0}）"
      bash "$ROOT/scripts/check_d1_state.sh" "$d1" "${EXPECT_D1:-0}" || exit 1' _
    # 注意：这里**不**跑 --sha（对两个 101 GB 分片做 sha256 要 ~10 分钟）；
    # sha 已由远端常驻守望 `weights_watch_remote.sh` 在落盘 WEIGHTS_READY 前算过（见 #241）。
    run "权重就绪（分片齐备 + 模型口径）" bash "$ROOT/scripts/check_weights_ready.sh"
    run "远端已落盘 WEIGHTS_READY（含 sha256 校验结论）" bash -c '
      out=$(timeout 30 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "${SSH_CFG:-/root/lvllm/tmp/sshcfg}" "${HOST:-gpu}" \
            "cat /home/user/lvllm/logs/WEIGHTS_READY 2>/dev/null || echo __MISSING__")
      case "$out" in __MISSING__|"") echo "   没有 WEIGHTS_READY（守望还没通过 sha 校验）"; exit 1;; esac
      echo "   $out"' _
    run "PIN 表可用（存在 / XTUP 魔数 / 大小合理）" bash "$ROOT/scripts/check_pin_table.sh"
    run "模型配置可解析（vLLM get_config）" bash "$ROOT/scripts/check_model_config.sh"
    if [ "$SKIP_MARLIN" = 1 ]; then
        skip "MARLIN 融合内核可用" "SKIP_MARLIN=1（⚠️ 本轮 MARLIN=1 将是假口径，报告须注明）"
    else
        run "MARLIN 融合内核可用（AVAILABLE True）" bash "$ROOT/scripts/check_marlin_ready.sh"
    fi
    run "GPU 显存 / 宿主内存 / 端口 / PID 归属（起服务门）" bash -c '
      timeout 120 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "${SSH_CFG:-/root/lvllm/tmp/sshcfg}" "${HOST:-gpu}" \
        "cd /home/user/lvllm/vllm-xiaotu-moe && bash scripts/prelaunch_mem_gate.sh --need-gib 900 --gpus 1,2 --tag prod" 2>&1 | tail -6
      rc=${PIPESTATUS[0]:-1}
      [ "$rc" = 0 ] || { echo "   内存门 rc=$rc（0 才许启动）"; exit 1; }
      # 磁盘：权重 510 GB + 日志/转储。少于 50 GB 就别起（避免跑一半写满盘）。
      avail=$(timeout 30 ssh -o ConnectTimeout=8 -o BatchMode=yes -F "${SSH_CFG:-/root/lvllm/tmp/sshcfg}" "${HOST:-gpu}" \
              "df -BG --output=avail /home | tail -1 | tr -dc 0-9")
      echo "   /home 可用 ${avail:-?} GB"
      [ "${avail:-0}" -ge 50 ] || { echo "   磁盘不足 50 GB（当前 ${avail:-?} GB）"; exit 1; }' _
fi

printf '\n════════ 结论：'
if [ "$fails" -eq 0 ]; then
    echo "GO（$step 项全部通过）════════"
    echo "下一步：按 RESUME v2 第 3 步起服务（生产口径），随后 kv_gate.sh ≥300000 再跑 E1。"
    exit 0
fi
echo "NO-GO（$fails 项失败）════════"
echo "先修上面 ❌ 的项；**不要**带缺口起服务（一次重启 20–40 分钟）。"
exit 1
