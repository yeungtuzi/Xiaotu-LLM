#!/usr/bin/env bash
# GLM-5.3-Flash on mainline vLLM + the `vllm-xiaotu-moe` plugin (A100 / SM80).
#
# What this configuration depends on (all verified in dev-docs/GLM53_SM80_PLAN.md):
#   * The 11 sparse-MLA (DSA) layers need the SM8x Triton backend, which the
#     model binds only when `sm8x_sparse_mla_enabled()` is true -> this requires
#     **--kv-cache-dtype bfloat16** (the SM8x route is bf16-only; fp8/fp4 KV
#     stays on the upstream SM90+ pool and fails closed).
#   * Routed expert weights live on the host and are computed by the xiaotu
#     engine: VLLM_EXPERTS_LOAD_DEVICE=cpu + the plugin's CPU backend swap.
#     The native FP8 checkpoint's block-128 experts take the FP8 kernel.
#   * The DSA (full-attention) projections are FP8 in the checkpoint but the
#     model keeps them BF16 and dequantizes on load (glm5next model.py).
#
# Usage:
#   bash scripts/serve_glm53_mainline.sh
#   TAG=glm53_t1 PORT=8073 MAXLEN=8192 MBT=2048 SEQS=8 bash scripts/serve_glm53_mainline.sh
#   COMPILE=1 ...   # add a compilation config (slow first start; off for bring-up)
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/ZhipuAI--GLM-5.3-Flash/snapshots/master}"

TAG="${TAG:-glm53_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8070}"          # 生产服务端口(2026-09-19 起从 8073 迁到 8070)
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
# 【§601,2026-09-19】0.90 → **0.85**。原因是一次真实生产事故:util 0.90 时启动后只剩
# ~6.8 GiB 空闲,而 GPU 预填充的 FP8 staging 缓冲是**进程级持久**的(~4.2 GiB/rank),
# 它吃掉的是 vLLM 拿来定 KV 的那份"激活峰预留"(启动日志 `peak activation: 2.9 GiB`)
# ⇒ 28,553-token 的真实请求在 `chunk_kda_with_fused_gate` 里只差 52 MiB 就把
# **整个服务打崩**(GPU 0/1 各剩 45 MiB,见 logs/glm53_prod.log 11:10:56)。
# 0.85 让"staging + 激活预留"有 ~1 GiB 余量;预填充预检(见 gpu_prefill.fits_device)
# 现在也把这份激活预留算进去,腾不出来就**优雅退回 CPU 预填充**而不是崩服务。
# 代价:KV 池 988,081 → ~814,000 token(2 路 256K 仍占 64%,3.1x 并发)。
# 要换回更大的 KV 池,先量 `GPU prefill ACTIVE ... slack` 那行的余量再动。
# ⛔⛔ 【2026-10-15:以下 2026-09-20 的"util 高 + 必须显式封顶 KV"原则**已过时**(与 §5.8 同因)】
#   原文保留存档 ⇒ **原则:留给 staging+激活的空间 = util×VRAM − 非专家 − KV池**
#     ⇒ util 尽量高、KV 池按需最小(用 KV_CACHE_BYTES 封顶);
#     ⚠️ 只提 util 不封顶 ⇒ 池会填满预算、staging 被逐层拒、长 prompt 激活 OOM;
#     ⇒ 顺序:先定 cap,再提 util。
#   **机制(⚠️ 本仓证据;vLLM 侧待启动日志核实 —— 审计员指出:vLLM 源码在本机为坏挂载,无法直接验证)**:该原则的前提是"vLLM 定 KV 尺寸时**看不见**我们的 staging/激活"。
#   而插件**已经在 `profile_run` 期把 staging 如实占位**,于是 vLLM 的账本(`total_consumed` /
#   `transient_peak_headroom`)里**含**它,并据此给 KV 定容 —— 证据在本仓:
#     * `vllm_xiaotu_moe/gpu_prefill.py:753`「而 **profile_run 期必须放行**(只有真分配 staging,
#       vLLM 的账本才含它)」;`:886-902` 用 shim 包住 `GPUModelRunner.profile_run` 并在其中
#       `_reserve_staging_from_model(self)`;`:815-820` profile 结束即释放;
#     * `vllm_xiaotu_moe/mainline_shims.py:1930` 已装载 `_install_gpu_prefill_profile_guard`。
#   ⇒ 所以"先定 cap 再提 util"不再是唯一安全路径:**不封顶(vLLM 统一管理)时,池不会再"填满预算"**,
#     因为预算里已经扣掉了 staging。`GPU_UTIL` 默认仍是 0.90,保持不变。
#   ⚠️ **首次以新默认启动时必须核对日志判据**(见 `dev-docs/report/tuning/vllm_vram_experiment_2026-10-09.md`
#      H1/H2):须出现 ①`profile 期占位` ②`profile 结束 ⇒ 释放 staging 占位` ③vLLM 自己的
#      `Available KV cache memory: …`;若这三类行缺失、或"起来之后才 OOM" ⇒ 前提不成立 ⇒ 回退 `KV_AUTO=0`。
#   ⚠️ 仍然有效的一半(与本原则无关):`util` 不能高到"非专家 + 池 + staging"都放不下 —— 那种情形
#      **应当启动即失败**(vLLM 会自己拒绝),而不是运行中崩。
GPU_UTIL="${GPU_UTIL:-0.90}"
# Context length. The 4096 this script used to default to was a *test* setting, not a
# hardware limit: measured on 2xA100-40GB at util 0.88 (bf16 KV, TP=2), the KV pool and
# the largest max-model-len that starts are
#     256K -> 919,520 tok pool, 3.51x concurrency   (comfortable)
#     512K -> 843,055 tok pool, 1.61x concurrency
#     704K -> 763,177 tok pool, 1.06x concurrency   (single sequence, no margin)
#     768K -> does not start (vLLM reports a 733,312 ceiling); 1M needs 11.57 GiB of KV
#             against 6.87 GiB available -> needs an fp8 KV cache, see MODEL_GUIDES 2.5
# The pool SHRINKS as maxlen grows because the KDA state pools scale with it, so the
# per-request limit and the total token capacity trade against each other. 256K is the
# safe default; raise it deliberately.
# 【2026-09-21 用户定的生产口径】**512K 上下文 + MBT=4096**:
#   要 512K 就必须把 `MBT` 压到 4096 —— 激活工作区 ∝ MBT,而 KV 池与之争同一份显存。
#   代价是 prefill 变慢(256K 下实测 MBT 4096 比 12288 慢 34%,见 TUNING_GUIDE §8),
#   换来的是"512K 一定起得来"。要速度优先的格子再显式传 `MBT=12288 MAXLEN=262144`。
# ⭐【2026-10-15 用户定案】默认上下文 512K → **256K**:512K 下 KV 池连一路都装不满
#   (实测池 416,228 token < 524,288 ⇒ 0.79x);256K 下同一池有 1.59x,再叠加其它显存优化后目标是 **3 路 256K**(SEQS 仍 4)✓
MAXLEN="${MAXLEN:-262144}"
MBT="${MBT:-8192}"  # ⭐ 0.2.9 实测最优(与 qfn/ds41f 对齐;旧 4096 作废)
# Admission limit. 256K context x 2 concurrent sequences = 524,288 tokens, and the
# KV pool at this maxlen is 919,520 tokens (measured, util 0.88) => two full-length
# sequences are guaranteed by construction, with ~40% of the pool left as headroom
# for the activation workspace. (The pool itself reports 3.51x concurrency for
# 262,144-token requests, so 3 would also be admitted; 2 is the deliberate choice.)
SEQS="${SEQS:-4}"   # 用户 2026-09-22 定:所有模型默认 seqs=4(原为 2)
THREADS="${THREADS:-60}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
# 投机解码(MTP,2026-09-20)。GLM-5.3-Flash 的检查点自带 **1 层 MTP**
# (`model.language_model.layers.45`,见 docs/MODEL_GUIDES.md §2.6)。
#   **SPEC_K=0(默认,关闭)**;要尝鲜再 `SPEC_K=1..4`。
#   单层 MTP 通过复用 hidden state 支持 K>1(`index_share_for_mtp_iteration=True`),
#   但**实测 K>1 明显更差**(K=4 时 accept 只到 1.63、吞吐掉到 11.2 tok/s)。
#
# 【2026-09-20 改默认:关】用户裁定。实测的收益/代价严重不成比例:
#   * **收益太小**:·256/128 C=1 只把 decode 从 21.9 抬到 22.6 tok/s(≈+3%,噪声内);
#     ShareGPT 上更小(21.9 → 22.3)。接受率只有 **1.46**(p0≈0.46),逐位衰减极快。
#   * **代价三项都是硬的**:
#       ① draft 第 45 层**强制常驻 GPU**(约 **3.38 GiB/rank**);
#       ② profile 把激活峰从 2.9 GiB 低估到 0.84 GiB ⇒ **KV 池 915,487 → 666,366(−27%)**,
#          256K 并发 3.49× → 2.54×;
#       ③ 单 token 间隔**脉冲化**(46 → 64 ms)。
#   * **而且它是长 prompt OOM 的元凶**:util 0.82 下
#     `非专家 ~7.5 + KV 6 + staging 8.35 + draft 3.38 ≈ 25.2 GiB`,留给**激活工作区**
#     的只剩 ~7 GiB,长 prefill 的激活会超过它 ⇒ **实测 CUDA OOM / EngineCore 死**
#     (dev-docs/archive/handoff/HANDOFF_PERF_TOPN.md §10)。
#     ⇒ GLM 上"GPU 预填充 + 投机 + 长上下文"**不可兼得**,而投机的 +3% 最不值得保。
#   * MiMo 不受此限:draft 是 **dense 3 层**(很小),GPU 预填充与投机共存 ——
#     实测 4,148-token × C=4 拿到 **1,845 tok/s** prefill 且投机开着。
SPEC_K="${SPEC_K:-0}"
SPEC_MODEL="${SPEC_MODEL:-$CKPT}"
# EAGER=1 ⇒ --enforce-eager。MTP 的 draft 图捕获若在插件路径下有问题,退这一档。
EAGER="${EAGER:-0}"
# GPU-prefill threshold ⚠️ 【2026-10-09 更正(R38 复审指出事实错误)】插件的真实默认值是
# **0 = 关闭**(见 `vllm_xiaotu_moe/gpu_prefill.py` 的 `gpu_prefill_min_tokens()`);
# 旧注释里写的 **4096** 其实是**另一个旋钮** `XIAOTU_GPU_PREFILL_SWITCH_TOKENS`
# (见 `vllm_xiaotu_moe/vram_policy.py`),两者曾被混为一谈 ✗。
# 本脚本**总是显式**给 `GPU_PREFILL_MIN`(默认 1500),所以这个默认值对本脚本无影响 ✓。
# 为什么取 1500:GLM-5.3 的 KDA(mamba-like)状态只在 `block_size = 2176` 边界写入 ⇒
# scheduler 会把每个 chunk 裁到 2176 的倍数(与 --max-num-batched-tokens 无关);
# 而实测(side-stream 装配重叠下)break-even 约 1300 tok/chunk
# (1580 tok: CPU 10.9 s vs GPU 10.2 s;2176 tok: 1.2x;~4.1k: 1.26x)⇒ 1500 是本机甜点。
# 只有实测过才往下调。
GPU_PREFILL_MIN="${GPU_PREFILL_MIN:-1500}"
# GPU_PREFILL=0 彻底关掉 GPU 流式预填充(走 CPU 引擎)。
# 为什么需要这个开关(2026-09-20 实测):staging 是**进程级持久**的 ~7.59 GiB/rank,
# 会把 vLLM 的激活峰挤掉;而开 MTP 后第 45 层 draft 又要额外 3.38 GiB/rank ⇒
# 32k 预填充 / 两路并发会 OOM 并把整个引擎打死。关掉它只损失长 prompt 的 TTFT,
# 换来稳定(见 docs/KNOWN_LIMITATIONS.md)。
GP_PREFILL="${GP_PREFILL:-1}"
if [ "$GP_PREFILL" = "0" ]; then
  # ⭐ 2026-10-09:原来靠 `XIAOTU_GP_ACT_RESERVE_GIB=99` 让**我们的预检**不通过,从而间接
  #   关掉 GPU 预填 ✗ —— 那个自建预检门已删除(QA #152)⇒ 改用**用户的正式旋钮** ✓
  # ⚠️ 必须改 `GPU_PREFILL_MIN`(不能只 export):本脚本 :193 会把该值写进**权威 env 文件**、
  #   :251 又直接放进子进程 env,而 env 桥对显式文件是"文件覆盖 os.environ" ⇒ 只 export 会被覆盖 ✗
  #   (2026-10-09 独立审计 **MF-1** 查出;DRYRUN 实测确认)
  GPU_PREFILL_MIN=0
  export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=0
fi
# 工具调用 + 思考解析。不加这两项时:
#   * 客户端带 tools + tool_choice="auto" 会被 vLLM 直接 400:
#     '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
#     (DSH 等 agent 框架默认就会发 tools,所以生产服务必须开);
#   * 不开 reasoning parser 时,思考内容会混在 content 里(模型模板会输出思考块),
#     客户端拿不到 reasoning_content、也无法按思考等级区分。
# GLM 系在 vLLM 里的注册名是 glm47_moe(别名 glm47);本检查点模板消费 reasoning_effort,
# 只区分 low / high,其余取值一律按 max 处理(见 chat_template.jinja)。
# 前缀缓存命中率上报。vLLM 默认 **不上报** `usage.prompt_tokens_details`
# (`enable_prompt_tokens_details` 默认 False,不开时 `_make_prompt_tokens_details()` 直接返回 None
# ⇒ 客户端只能看到 `cached_tokens: null`)。前端(DSH 等)要显示命中率就必须开这一项。
PROMPT_TOKENS_DETAILS="${PROMPT_TOKENS_DETAILS:-1}"
TOOL_PARSER="${TOOL_PARSER:-glm47}"     # vLLM 注册名:glm45 / glm47(实现是 glm47_moe_tool_parser)
REASONING_PARSER="${REASONING_PARSER:-glm47}"
INTERLEAVE="${INTERLEAVE:-1}"
COMPILE="${COMPILE:-0}"
OUTDIR="${OUTDIR:-$ROOT/logs}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"

ARGS=(
  --model "$CKPT"
  --served-model-name GLM-5.3-Flash
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --kv-cache-dtype "$KV_DTYPE"
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MBT"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-chunked-prefill
  --disable-custom-all-reduce
)
# 工具调用:必须同时给 --enable-auto-tool-choice 与 --tool-call-parser,否则
# 客户端发 tool_choice="auto" 会 400(见上面注释)。
[ -n "$TOOL_PARSER" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
# 让客户端能读到前缀缓存命中数(usage.prompt_tokens_details.cached_tokens)
[ "$PROMPT_TOKENS_DETAILS" = "1" ] && ARGS+=(--enable-prompt-tokens-details)
# ⛔⛔ 【已过时 · 2026-10-15 用户明令】原 RUNBOOK §5.8「开 GPU 预填充时**必须**显式封顶 KV 池」
#   —— 用户原话:「**这个铁律在 vLLM 统一管理显存的时候不存在了,标记为"已过时"并注明原因**」。
#   **过时原因(三条,缺一不可)**:
#     ① 原铁律的前提是"**vLLM 看不见我们的 staging / 投机 / 长 prefill 激活**" ⇒ 它才只能靠我们把
#        KV 池手动压小、把余量留出来。⇒ 前提错在"看不见"这件事本身,而不是"余量不该留"。
#     ② vLLM 自己会算 **`transient_peak_headroom`**(`gpu_worker.py:697`),但它**只在跑真实
#        `profile_run` 时才算**;而我们以前**显式传 `--kv-cache-memory` 正好把这次剖析跳过了** ✗
#        (本仓 `dev-docs/HANDOFF_DEGEN_AND_MBT_2026-10-06.md:50` 已自证此点;
#          `dev-docs/SPEC_mbt_budget.md` 改动① 因此把 ds41f 的这条传参**整条删除**)。
#        ⇒ **不是 vLLM 管不了,是我们以前没让它管。**
#     ③ v0.2.7 又把预填 staging 由「2×整层」缩成「N×子层块环」:**7.56 → 1.89 GiB/rank**
#        (TP=2 共省 11 GB)⇒ 即便有瞬时峰,vLLM 的留白也吸收得下。
#   ⇒ **新默认**:不传 `--kv-cache-memory`,**由 vLLM 统一管理**(与 ds41f 现行做法一致)。
#     逃生门两条:显式 `KV_CACHE_BYTES=<字节>` 仍会封顶;`KV_AUTO=0` 可回退旧的"按 maxlen 自动推算"。
#   ⚠️ 保留的**唯一有效警戒**(与原铁律无关):`MAXLEN × SEQS × 1.10` 那种写法会一次性要 23 GiB,
#      无论谁管显存都会 OOM —— 那是**算术错误**,不是"该不该封顶"的问题。
# 历史默认值标尺(仅在 KV_AUTO=0 回退时使用):6 GiB ↔ 293,651 token ⇒ **48,942 token/GiB**。
# ⛔⛔ 【M1 修 · R38 复审(独立审计员)查出】**输入校验必须先做**:下面 `:181/:189` 用
#   `$(( … / 1073741824 ))` 打印 GiB;若 `KV_CACHE_BYTES` 是**非纯整数**(如 `12G`),该算术会报错
#   ⇒ **整块 if 被中止** ⇒ `ARGS+=(…)` 被跳过 ⇒ **脚本照常启动且根本没封顶** ✗
#   (改前的旧代码把 `ARGS+=(…)` 放在 if **之外**、无条件执行 ⇒ 会把 `12G` 交给 vLLM ⇒ argparse 报错 ⇒ fail-closed)。
#   ⇒ 即"以为封顶了、其实没封"的**静默误解**(与 `serve_v41.sh:125` 的既有教训同型,且违反 R38 ⑦「缺省拒绝」)。
#   ⇒ 故此处**显式拒非法值**:非纯数字 ⇒ 直接 exit 2(宁可不启动,不静默放行)。
if [ -n "${KV_CACHE_BYTES:-}" ]; then
  # ⛔【M1-residual 第三轮修 · 独立审计员】两处加固,互相兜底:
  #   ① 守卫用**显式字符集** `[!0123456789]`,**不用范围** `[!0-9]` ——
  #      范围在非 C locale(本机 `LANG=en_HK.UTF-8`)下会把**全角/阿拉伯-印度数字**当"数字"放行,
  #      于是 `KV_CACHE_BYTES=０８` 通过守卫、`10#０８` 却报 `invalid integer constant` ⇒ 整块被中止
  #      ⇒ **rc=0 且完全不封顶地启动** ✗(CJK 输入法下敲全角数字就能触发,比 `08` 更现实)。
  #      ⚠️ 字符集不受 collation 加宽,实测对 `０８`/`٠٨` 一律 REJECT ✓
  #   ② `ARGS+=(…)` 提到**所有算术之前**(argv 保留原始字符串;ASCII 数字 vLLM 按十进制解析 ⇒ 安全),
  #      这样**任何**算术(归一化或显示)再出问题都不可能造成"静默不封顶"。
  case "$KV_CACHE_BYTES" in
    ''|*[!0123456789]*)
      echo "⛔ KV_CACHE_BYTES 必须是**纯 ASCII 十进制整数(字节)**,收到 '${KV_CACHE_BYTES}'" >&2
      echo "   ⇒ 拒绝启动(R38 ⑦:缺省拒绝,不做静默放行;例如 6 GiB 请写 6442450944)" >&2
      exit 2 ;;
  esac
  ARGS+=(--kv-cache-memory "$KV_CACHE_BYTES")
  KV_CACHE_BYTES=$(( 10#$KV_CACHE_BYTES ))   # 仅用于下面显示(消灭前导零/八进制陷阱)
  echo "[glm53] 显式封顶:--kv-cache-memory ${KV_CACHE_BYTES} bytes(~$(( KV_CACHE_BYTES / 1073741824 )) GiB)"
elif [ "${KV_AUTO:-1}" = "1" ]; then
  echo "[glm53] ⭐ 不传 --kv-cache-memory ⇒ **显存交由 vLLM profile 统一管理**"\
       "(会算出 transient_peak_headroom;原 RUNBOOK §5.8 铁律已于 2026-10-15 标记过时)"
else
  # ⛔【同型】回退分支的算术吃 MAXLEN ⇒ 同样用显式字符集守卫,并**显式**处理算术失败(fail-closed)。
  case "${MAXLEN:-}" in
    ''|*[!0123456789]*)
      echo "⛔ KV_AUTO=0 回退分支要求 MAXLEN 为**纯 ASCII 十进制整数**(token 数);收到 '${MAXLEN:-}' ⇒ 拒绝启动" >&2
      exit 2 ;;
  esac
  KV_CACHE_BYTES=$(( 10#$MAXLEN * 1073741824 / 48942 )) \
    || { echo "⛔ 计算 KV 上限失败(MAXLEN='${MAXLEN}')⇒ 拒绝启动,不做静默放行" >&2; exit 2; }
  ARGS+=(--kv-cache-memory "$KV_CACHE_BYTES")
  echo "[glm53] KV_AUTO=0 ⇒ 回退旧行为:按 maxlen 自动封顶 ${KV_CACHE_BYTES} bytes"\
       "(~$(( KV_CACHE_BYTES / 1073741824 )) GiB;不乘 seqs=${SEQS})"
fi
if [ -n "${SPEC_CONFIG:-}" ]; then
  # Escape hatch: any other vLLM speculative method (ngram / eagle / medusa /
  # draft_model / suffix), passed through verbatim as JSON. Overrides SPEC_K.
  ARGS+=(--speculative-config "$SPEC_CONFIG")
elif [ "$SPEC_K" != "0" ]; then
  ARGS+=(--speculative-config \
    "{\"method\":\"mtp\",\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$SPEC_K}")
fi
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
if [ "$COMPILE" = "1" ]; then
  ARGS+=(--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}')
fi

# per-TAG env bridge (see vllm_xiaotu_moe/__init__.py): the EngineCore child does
# not inherit every XIAOTU_* var, so write the ones we need to a TAG-local file.
XTU_ENV_FILE="${XTU_ENV_FILE_OVERRIDE:-$OUTDIR/$TAG.envfile}"
{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${SPIN_IDLE_US:-300}"
  # Only the FILE is authoritative (XIAOTU_ENV_FILE is set explicitly below),
  # so the threshold has to be written here to reach the EngineCore child.
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GPU_PREFILL_MIN"
  # 专家层常驻 GPU(逗号+区间,如 "20-21";空=不常驻)。用户 2026-09-22 要求按需驻留。
  [ -n "${RESIDENT_LAYERS:-}" ] && echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=$RESIDENT_LAYERS"
  [ -n "${XIAOTU_MOE_NSLICE_SMALL:-}" ] && echo "XIAOTU_MOE_NSLICE_SMALL=$XIAOTU_MOE_NSLICE_SMALL"
  [ -n "${XIAOTU_MOE_ASYNC:-}" ] && echo "XIAOTU_MOE_ASYNC=$XIAOTU_MOE_ASYNC"
  [ -n "${XIAOTU_MOE_RANK_SPLIT:-}" ] && echo "XIAOTU_MOE_RANK_SPLIT=$XIAOTU_MOE_RANK_SPLIT"
  [ -n "${XIAOTU_MOE_NOSHARD:-}" ] && echo "XIAOTU_MOE_NOSHARD=$XIAOTU_MOE_NOSHARD"
  [ -n "${XIAOTU_MOE_GEMM_FP8_SCALE:-}" ] && echo "XIAOTU_MOE_GEMM_FP8_SCALE=$XIAOTU_MOE_GEMM_FP8_SCALE"
  # 引擎子进程可能丢 --speculative-config;显式钉进 env 文件(文件是权威)。
  [ "$SPEC_K" != "0" ] && echo "XIAOTU_SPEC_DECODE=1"
} > "$XTU_ENV_FILE"
# Pass through every other XIAOTU_*/VLLM_XIAOTU_* variable from the caller's
# environment. `XIAOTU_ENV_FILE` is set explicitly below, so the plugin takes the
# file as authoritative and *overwrites* os.environ with it -- anything not listed
# above is therefore dropped on the floor for the EngineCore child. That silently
# broke runtime A/Bs: exporting VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE here and
# expecting the worker to see it measured the same path twice (the same class of
# bug the plugin's own comment at `gpu_prefill_min_tokens` warns about).
while IFS='=' read -r _k _v; do
  case "$_k" in
    XIAOTU_ENV_FILE) continue ;;
    XIAOTU_*|VLLM_XIAOTU_*) ;;
    *) continue ;;
  esac
  grep -q "^${_k}=" "$XTU_ENV_FILE" 2>/dev/null || echo "${_k}=${_v}"
done < <(env) >> "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE="$XTU_ENV_FILE"
echo "[glm53] env-bridge -> $XTU_ENV_FILE"

if [ "$INTERLEAVE" = "1" ] && command -v numactl >/dev/null 2>&1; then
  NCTL=(numactl --interleave=all)
else
  NCTL=()
fi
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "[glm53] tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN mbt=$MBT seqs=$SEQS"
echo "[glm53] kv-cache-dtype=$KV_DTYPE (SM8x sparse-MLA is bf16-only)"
echo "[glm53] tool-call-parser=${TOOL_PARSER:-off} reasoning-parser=${REASONING_PARSER:-off}"
echo "[glm53] prompt-tokens-details=${PROMPT_TOKENS_DETAILS} (cached_tokens reporting)"
echo "[glm53] speculative=${SPEC_K} (0=off; MTP method=mtp, draft=layers.45) eager=${EAGER}"
echo "[glm53] log=$LOG"

# 【§601】PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" —— 事故现场是"PyTorch 已分配
# 37.90 GiB、另有 **695 MiB reserved-but-unallocated**",PyTorch 自己的 OOM 提示就是这一项。
# 本服务没有 KV connector,所以 vLLM 那条 "kv connector 与 expandable_segments 不兼容" 的
# 检查不适用。想要旧行为就显式 `PYTORCH_CUDA_ALLOC_CONF=` 传空。
ENVPREFIX=(
  PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}"
  HF_HUB_OFFLINE=1
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
  VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-60}"
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
  VLLM_USE_FLASHINFER_SAMPLER=0
  FLASHINFER_DISABLE_VERSION_CHECK=1
  VLLM_EXPERTS_LOAD_DEVICE=cpu
  XIAOTU_MOE_THREADS="$THREADS"
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GPU_PREFILL_MIN"
  XIAOTU_MOE_SPIN_IDLE_US="${SPIN_IDLE_US:-300}"
  OMP_NUM_THREADS=1
)
CMD=("${NCTL[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}")

# `DRYRUN=1` 只打印最终命令行(给 systemd 单元/排错用),不碰 GPU。
if [ "${DRYRUN:-0}" = "1" ]; then
  printf '[glm53] DRYRUN port=%s gpus=%s\n  ' "$PORT" "$GPUS"
  printf '%q ' env "${ENVPREFIX[@]}" "${CMD[@]}"
  printf '\n'
  exit 0
fi

# `FOREGROUND=1` 不 fork、不写 pidfile、不轮询就绪,直接把进程交给调用者
# (systemd 单元用这一档:进程即服务,日志进 journal,崩溃由 Restart= 拉起)。
# 手动后台模式(默认)才是 nohup + 日志文件 + 就绪轮询。
if [ "${FOREGROUND:-0}" = "1" ]; then
  echo "[glm53] FOREGROUND: exec 服务进程(日志由调用者/supervisor 接管,本脚本不再轮询就绪)"
  exec env "${ENVPREFIX[@]}" "${CMD[@]}"
fi

# ⚠️ 这一行是 `>` ⇒ **每次启动都会截断旧日志**。要保留历史请显式给带时间戳的 TAG
# (`TAG=glm53_$(date +%m%d_%H%M%S)`),或用 systemd 档(journal 自动留存)。
nohup env "${ENVPREFIX[@]}" "${CMD[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[glm53] pid=$(cat "$OUTDIR/$TAG.pid")"

for _ in $(seq 1 1080); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[glm53] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[glm53] server exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[glm53] timeout; tail:"; tail -40 "$LOG"; exit 1
