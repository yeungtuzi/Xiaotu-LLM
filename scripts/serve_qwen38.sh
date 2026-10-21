#!/usr/bin/env bash
# Qwen3.8-Flash-Next(官方 FP8)在单卡 SM80 上的调试服务 —— **默认 GPU2 / TP=1**。
#
# 纪律(AGENTS.md「自服务环境纪律」,用户 2026-09-29):
#   * 8070 是本 agent 自身推理后端 ⇒ **本脚本默认不碰 GPU0/1、不用 8070**;
#   * 调试需要 GPU 时用 **GPU2 + TP=1**;要动 GPU0/1 必须显式获得用户许可。
#   本脚本默认 `GPUS=0 TP=1 PORT=8140 MAXLEN=8192` —— 全部是"单卡调试"口径(GPU0;生产占 GPU1+GPU2)。
#
# 为什么官方 FP8 可行(依据 dev-docs/patches/probe_official_fp8_gate.py 的实测):
#   * 检查点在本地 HF 缓存(173 GiB,131 分片),`quant_method: fp8` ⇒ vLLM 原生 `Fp8Config`;
#     本插件按 `FusedMoEMethodBase` **泛化打 shim** ⇒ `Fp8MoEMethod` 自动走 CPU 专家;
#   * 权重分布(按张量头部精确统计):路由专家 **114.9 GiB**(留 CPU)、
#     **PLE/ngram 表 47.8 GiB**(必须 `XIAOTU_PLE_CPU=1` 走 UVA,否则单卡必 OOM,
#     见 vllm_xiaotu_moe/ple_offload.py 头部注释)、其余非专家仅 ~10 GiB;
#   * ⇒ PLE offload 后 GPU 侧非专家 ~10 GiB,40GB 单卡放得下。
#
# 用法:
#   bash scripts/serve_qwen38.sh                       # dummy? 不,默认真权重
#   LOAD=dummy bash scripts/serve_qwen38.sh            # 只验架构/显存,不读 173GiB(首选第一步)
#   MAXLEN=4096 GPUS=0 TP=1 bash scripts/serve_qwen38.sh
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ⭐ 审计修复:必须用【库】的身份判据(pi_valid_name / PROD_PIDFILES_RE / PROD_PORTS),
#   ⛔ 不许在本脚本里另写一份 —— 本仓已因"两份实现分叉"出过事故(见 AGENTS.md 第 0 条)✗
# shellcheck source=lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"
# ⭐ R54:error 侦测的**唯一真源** = `scripts/lib_errscan.sh`,但**本脚本不再直接引用它** ——
#   ⛔ 不许在本脚本里另写 `ERROR|error` 这类**大小写敏感**的封闭词表 ✗
#   ⭐ 2026-10-15:error 扫描**整体移入就绪门**(`scripts/wait_ready.sh` 会 source 那个 lib)✓
#     ⇒ 这样"扫描"与"判死"在**同一处**、同一时刻发生 ⇒ 不会再出现
#       "就绪判定排在错误扫描之前"这种时序漏洞(审计 B8)✗
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
# 官方 FP8 检查点(本地 HF 缓存快照;可用 CKPT= 覆盖)
#   ⚠️ 审计 MAJOR:原写法在 set -e 下,若 glob 不匹配 ⇒ 整条脚本 **静默 rc=2** ✗(`|| true` 必需)
CKPT="${CKPT:-$(ls -d /home/user/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/*/ 2>/dev/null | head -1 || true)}"
[ -n "$CKPT" ] || { echo "[qwen38] ⛔ 找不到 Qwen3.8-Flash-Next-FP8 快照目录 ⇒ 请显式 CKPT=<路径>" >&2; exit 3; }

TAG="${TAG:-qwen38_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8140}"        # ⚠️ 不能是 8070(生产)
#   ⛔⛔ BLOCKER 修复(审计):原默认是 **2**,而**生产 ds41f = TP=2 正好用 GPU1+GPU2** ✗✗
#      ⇒ 默认值等于"把调试实例放到生产卡上";且 R19/2026-10-09 明令:**实验只能 GPU0 + TP=1** ✓
#      ⇒ 默认改 **0**;并显式拒绝踩生产的 1/2(除非当次显式 ALLOW_PROD_GPU=1)✗
GPUS="${GPUS:-0}"           # 【纪律】调试只用 GPU0(生产占 GPU1+GPU2)
TP="${TP:-1}"               # 【纪律】TP=1
MAXLEN="${MAXLEN:-262144}"  # ⭐ 0.2.9 实测最优:256K 上下文(KV 足够大)
MBT="${MBT:-8192}"   # ⭐ 0.2.9 实测最优
SEQS="${SEQS:-4}"     # ⭐ 0.2.9 实测最优(ds/glm 同为 4)
GPU_UTIL="${GPU_UTIL:-0.90}"
THREADS="${THREADS:-60}"    # ⭐⭐【2026-10-19 用户裁定】60 = 最优解,不许上调;旧 48/120 口径均作废 ✗
LOAD="${LOAD:-auto}"        # auto=读真权重;dummy=不读盘,只验架构与显存
EAGER="${EAGER:-0}"         # ⭐ 0.2.9 实测最优:必须上图(R64;eager 只有图上 1/4~1/2)
PLE_CPU="${XIAOTU_PLE_CPU:-1}"   # 必须有:PLE 47.8GiB 走主机内存(UVA)
GPU_PREFILL_MIN="${GPU_PREFILL_MIN:-2048}"
# ⭐ 2026-10-19【最优默认】12 层精选常驻(按 CPU 时间/活跃专家数选出;B1053/B1055)
#   实测 15K 自然结束:decode 46.0/43.8/44.0(C=1/2/4),比无常驻的 45.6 更高、步时更低 ✓
#   ⛔ 不要再叠加热专家池:12 层 + 64 槽会挤爆 KV(实测 `No available memory for the cache blocks`)
XIAOTU_MOE_GPU_RESIDENT_LAYERS="${XIAOTU_MOE_GPU_RESIDENT_LAYERS:-15,9,11,16,31,3,39,25,7,23,24,13}"  # ⭐ 0.2.9 实测最优:>0 才启用 tile 环(0=关!)
# MUSA correctness profile: opt-in, conservative; never touches the CUDA defaults above.
if [ "${XTU_MUSA_CORRECTNESS:-0}" = "1" ]; then
  echo "[qwen38] MUSA correctness profile: CPU experts, no resident, no GPU prefill, eager, 8K"
  export VLLM_EXPERTS_LOAD_DEVICE=cpu
  export XIAOTU_MOE_GPU_RESIDENT_LAYERS=""
  export XIAOTU_PLE_CPU=0
  export XIAOTU_SPEC_DECODE=0
  PLE_CPU=0
  GPU_PREFILL_MIN=0
  EAGER=1
  MAXLEN="${MAXLEN:-8192}"
  SEQS="${SEQS:-1}"
  MBT="${MBT:-8192}"
fi
KV_DTYPE="${KV_DTYPE:-auto}"
TOOL_PARSER="${TOOL_PARSER:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
OUTDIR="${OUTDIR:-$ROOT/dev-docs/report/tuning/logs}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
# env 桥写【仓库内】,绝不覆盖生产用的 /tmp/xiaotu_env
XTU_ENV_FILE="${XTU_ENV_FILE_OVERRIDE:-$OUTDIR/$TAG.envfile}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "[qwen38] 检查点未找到: '$CKPT'(可显式 CKPT=<snapshot 目录>)" >&2; exit 2
fi
# 【2026-10-15 用户明令】不准 `--interleave=all`(依据 IRON_RULES.md:418/420、:875 ✗):
#   "权重缓冲不许依赖进程级 numactl --interleave=all …当前实现是显式 mbind" ✓
# 默认**不套** numactl ⇒ 让插件的显式 mbind 按 socket 绑定(每 rank 绑自己 socket 的 4 个 node ✓)。
# ⚠️ 只借用生产(serve_v41.sh:435)的"可覆盖"idiom;**缺省值故意与之不同** ✗ ——
#    生产缺省仍是 interleave ✗,本脚本缺省必须**不套** ✗;不要"对齐生产缺省"✗
# 需要时显式覆盖,例如:NUMA_CMD='numactl --cpunodebind=0 --membind=0' ✓
NUMA_CMD="${NUMA_CMD-}"
NCTL=(); [ -n "$NUMA_CMD" ] && NCTL=($NUMA_CMD)

{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${SPIN_IDLE_US:-300}"
  echo "VLLM_EXPERTS_LOAD_DEVICE=${VLLM_EXPERTS_LOAD_DEVICE:-gpu}"
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GPU_PREFILL_MIN"
  echo "XIAOTU_PLE_CPU=$PLE_CPU"
  echo "XIAOTU_RELEASE_SOURCE=${XIAOTU_RELEASE_SOURCE:-1}"
  # 实验变量透传(未设=不写行,set -e 安全):XIAOTU_PIN 分流实验用
  if [ -n "${XIAOTU_PIN:-}" ]; then echo "XIAOTU_PIN=$XIAOTU_PIN"; fi
  if [ -n "${XIAOTU_PIN_TABLE:-}" ]; then echo "XIAOTU_PIN_TABLE=$XIAOTU_PIN_TABLE"; fi
  # ⭐ 2026-10-15:动态换入需要窗口阈值才可能触发(expert_pool.maybe_adapt:
  #    `_win < XIAOTU_PIN_WINDOW` ⇒ 直接 return 0)⇒ 缺这几项时**只能静态驻留** ✗
  #    ⚠️ 与上面两行**同一 idiom**:未设 ⇒ 不写行 ⇒ 引擎侧取默认 ⇒ **行为逐字不变** ✓
  if [ -n "${XIAOTU_PIN_WINDOW:-}" ]; then echo "XIAOTU_PIN_WINDOW=$XIAOTU_PIN_WINDOW"; fi
  if [ -n "${XIAOTU_PIN_HYST:-}" ]; then echo "XIAOTU_PIN_HYST=$XIAOTU_PIN_HYST"; fi
  if [ -n "${XIAOTU_PIN_HITSTAT:-}" ]; then echo "XIAOTU_PIN_HITSTAT=$XIAOTU_PIN_HITSTAT"; fi
  # ⭐ 2026-10-17:路由 dump 与换入限流也必须经 env 桥 —— EngineCore 会**重建 environ**
  #    ⇒ 只 export 到 shell 的变量会被丢掉(实测:XIAOTU_ROUTING_DUMP 未生效,dump 文件 0 字节)✗
  #    同一 idiom:未设 ⇒ 不写行 ⇒ 行为逐字不变 ✓
  if [ -n "${XIAOTU_ROUTING_DUMP:-}" ]; then echo "XIAOTU_ROUTING_DUMP=$XIAOTU_ROUTING_DUMP"; fi
  if [ -n "${XIAOTU_PIN_MAXSWAP:-}" ]; then echo "XIAOTU_PIN_MAXSWAP=$XIAOTU_PIN_MAXSWAP"; fi
  # ⭐ 2026-10-19:异步 CPU 解码是"每层一次 D2H 同步"的既有解药(ds41f: 6.53→1.80 ms/token),
  #    同样必须经 env 桥(EngineCore 重建 environ)。未设 ⇒ 不写行 ⇒ 行为逐字不变 ✓
  if [ -n "${XIAOTU_MOE_ASYNC:-}" ]; then echo "XIAOTU_MOE_ASYNC=$XIAOTU_MOE_ASYNC"; fi
  # ⭐ 2026-10-19:第一步=静态驻留(启动期装载,decode 期零传输)⇒ 该开关必须能被透传
  if [ -n "${XIAOTU_PIN_STATIC:-}" ]; then echo "XIAOTU_PIN_STATIC=$XIAOTU_PIN_STATIC"; fi
  if [ -n "${XIAOTU_MOE_GPU_RESIDENT_LAYERS:-}" ]; then echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=$XIAOTU_MOE_GPU_RESIDENT_LAYERS"; fi
  # ⭐ 2026-10-19:逐层 CPU 计时(选常驻层要用)与池的逐层集中度统计
  if [ -n "${XIAOTU_CD_TIMING:-}" ]; then echo "XIAOTU_CD_TIMING=$XIAOTU_CD_TIMING"; fi
  if [ -n "${XIAOTU_CD_TIMING_EVERY:-}" ]; then echo "XIAOTU_CD_TIMING_EVERY=$XIAOTU_CD_TIMING_EVERY"; fi
  if [ -n "${XIAOTU_PIN_DEBUG:-}" ]; then echo "XIAOTU_PIN_DEBUG=$XIAOTU_PIN_DEBUG"; fi
} > "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE

ARGS=(
  --model "$CKPT"
  --served-model-name Qwen3.8-Flash-Next
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MBT"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --load-format "$LOAD"
  --trust-remote-code
  --enable-chunked-prefill
  --disable-custom-all-reduce
)
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
[ -n "$TOOL_PARSER" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ "${PROMPT_TOKENS_DETAILS:-1}" = "1" ] && ARGS+=(--enable-prompt-tokens-details)
# ⭐ 2026-10-04:MXFP4 变体的 quantization_config 用的是 compressed-tensors 的 config_groups,
#   其 group_2 正则写的是【检查点侧】的 `in_proj_qkv` / `in_proj_z`,而新版模型代码把两者
#   融合成 `in_proj_qkvz` ⇒ 没有规则命中 ⇒ EngineCore 抛
#   "No compressed-tensors compatible scheme was found for … in_proj_qkvz" ✗
#   ⇒ 用 --hf-overrides 把融合名补进 targets(不改 vLLM 代码)。见 EXPERIMENTS B307。
[ -n "${HF_OVERRIDES:-}" ] && ARGS+=(--hf-overrides "$HF_OVERRIDES")
# 通用透传(测投机/图模式等一次性实验用;生产口径请写进本脚本而不是靠这个)
# ⭐ 2026-10-04【显式 KV 预算,照 DS-4.1 的做法】:给了 --kv-cache-memory 之后 vLLM 会
#   【跳过显存剖析】⇒ --gpu-memory-utilization 失效(R24 第 8 条)⇒ 剩下的显存留给
#   GPU-prefill 的 staging。QFN 实测 ≈72,368 B/token ⇒ 预算 = 目标 token 数 × 该值。
[ -n "${KV_CACHE_BYTES:-}" ] && ARGS+=(--kv-cache-memory "$KV_CACHE_BYTES")
# ⭐⭐ 0.2.9 实测最优:全解码图(FULL_DECODE_ONLY)+ 编译模式;
#   默认 `FULL_AND_PIECEWISE` 在 qfn 上会多抓一份无用图并挤占显存 ✓
# ⚠️ EAGER=1 是"图捕获崩溃时的回退档" ⇒ 此时**不能**再塞 compilation-config(否则两项自相矛盾)
if [ "$EAGER" != "1" ] && [[ "${EXTRA_ARGS:-}" != *compilation-config* ]]; then
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}')
elif [ "$EAGER" = "1" ]; then
  echo "[qwen38] EAGER=1 ⇒ 不注入 compilation-config(回退档保持纯 eager)"
fi
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)

echo "[qwen38] tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN mbt=$MBT load=$LOAD"
echo "[qwen38] PLE_CPU=$PLE_CPU(必须=1,否则单卡 OOM)  eager=$EAGER  gpu_prefill_min=$GPU_PREFILL_MIN"
echo "[qwen38] ckpt=$CKPT"
echo "[qwen38] log=$LOG   envfile=$XTU_ENV_FILE"

if [ "${DRYRUN:-0}" = "1" ]; then
  printf '[qwen38] DRYRUN gpus=%s tp=%s: ' "$GPUS" "$TP"
  printf '%q ' env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}" \
    VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_ENV_FILE="$XTU_ENV_FILE" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
  printf '\n'; exit 0
fi

cd "$OUTDIR"; mkdir -p run; cd run
export CUDA_VISIBLE_DEVICES="$GPUS"

# ⭐ 2026-10-04【同机多实例纪律,AGENTS.md §4】起第二个实例前的四项自查:
#   ① 结构相似的替代模型(QFN ✓,不是巨模型)  ② MemAvailable 够
#   ③ oom_score_adj=+800(让内核【先杀自己】,别去挑生产)  ④ THREADS=n_ccd×5(=120)+ taskset(见下方 2026-10-06 段 ✓)
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-800}"
# ⭐ 2026-10-06【qfn 服务默认值,用户授权】线程数默认改为 **n_ccd × 5**(本机 24 CCD ⇒ 120)。
#   依据(实测):
#     · 引擎自身默认就是 n_ccd×5 = 120(`serve_v41.sh` 注释:*"未设时插件按 n_ccd×5 自动调优"*);
#       本脚本此前钉成 60 是"先跑通"的保守值,`serve_v41.sh` 明确写着 **60 两头都亏**:
#       60 → 386.8 ms/层,120 → 219.2,192 → 170.9 ⇒ 120 比 60 快 **1.76×**。
#     ⛔ 2026-10-19 用户裁定:**THREADS=60 就是最优解,不许用更多核**;
#        旧文"120 比 60 快 1.62×"的说法**已作废**(那条测量口径不可信)✗。
#   ⚠️ 代价(用户已知并接受):生产 8070 自身也吃 CPU;用户说明**当前引擎实现的核心利用率
#      本就吃不满**,故按 120 设定。若生产转为满载,应把 THREADS_MAX 调回 64 ✗。
#   ⚠️ 本项与 GPU 预填无关 —— 本次任务不关注 GPU 预填性能,只要求行为正确 ✓。
# ⛔⛔【2026-10-07 用户明令 · 上面 120 那条【已作废】】✗✗
#   用户原话:"直接设置 qfn 的线程数为 **48**，**不准手工绑定核心**" ✓
#   ⇒ 默认改为 **48**(= n_ccd×2),并**取消手工核绑定**(见下方 TASKSET 段)✓
#   ⇒ 上面那段"120 = n_ccd×5"的实测依据**仍然成立**(那是"120 比 60 快 1.62×"的历史事实),
#     但它**不再是本脚本的默认**;要复现旧口径请显式传 `THREADS=120` ✓
THREADS_MAX="${THREADS_MAX:-192}"
THREADS="$(awk -v t="$THREADS" -v m="$THREADS_MAX" 'BEGIN{print (t+0>m)?m:(t+0)}')"
# ⛔ 手工绑核【默认不做】(用户 2026-10-07 明令:"不准手工绑定核心")✓
#   ⚠️ 语义(务必分清,否则会悄悄改变别人的实例 ✗):
#     · 取值 **`0-191`**(默认)= 全部 192 个在线核 ⇒ **等价于不绑**,既有调用方行为**不变** ✓
#     · 显式传【空字符串】`TASKSET=` ⇒ 连 `taskset` 包装都不加 ✓
#       ⭐ 核分配交给引擎自身(它按 CCD/NUMA 拓扑自己 pin `cores_`,见 numa_pool.hpp)✓
#   ⚠️ 故这里用 `${TASKSET-0-191}`【不带冒号】—— 带冒号会把"显式空"当未设 ⇒ 退回 0-191 ✗
TASKSET="${TASKSET-0-191}"
TS_PREFIX=(); [ -n "$TASKSET" ] && TS_PREFIX=(taskset -c "$TASKSET")
# ⭐⭐ 审计 BLOCKER/MAJOR 修复:启动前**显式拒绝**踩生产资源(默认 fail-closed)✓
case ",${GPUS}," in
  *,1,*|*,2,*) if [ "${ALLOW_PROD_GPU:-0}" != "1" ]; then
      echo "[qwen38] ⛔ 拒绝启动:GPUS=$GPUS 含 **生产在用** 的 GPU1/GPU2(ds41f = TP=2)✗" >&2
      echo "[qwen38]   R19 与 2026-10-09 用户明令:实验只能 GPU0 + TP=1" >&2
      echo "[qwen38]   确需占用生产卡 ⇒ 先取得用户当次许可,再显式 ALLOW_PROD_GPU=1" >&2
      exit 3
    fi ;;
esac
# ⛔⛔ 自查抓出的【死代码】(R58 第四次同型错误):
#   v1 我写的是 `pi_prod_ports_of` + 遍历 `$PI_PROD_PORTS` —— 但该函数**需要一个 PID 参数**,
#   它回答的是「**这个进程**监听了哪些生产端口」,⛔ 不是"静态的生产端口清单" ✗
#   ⇒ 无参调用下 `PI_PROD_PORTS` **恒为空** ⇒ 循环体永不执行 ⇒ **端口守卫是死代码** ✗✗
#   ⭐ 正确来源是库里的**静态**常量 `PROD_PORTS`(库 :35)✓
for _pp in $PROD_PORTS; do
  if [ "$PORT" = "$_pp" ]; then
    echo "[qwen38] ⛔ 拒绝启动:端口 $PORT 是**生产/同生命周期服务**端口(${PROD_PORTS// /\/})✗" >&2
    exit 3
  fi
done
unset _pp
pi_valid_name "$TAG" || { echo "[qwen38] ⛔ 实例名非法:'$TAG'(见 lib_proc_identity.pi_valid_name)" >&2; exit 3; }
if printf '%s\n' "${TAG}.pid" | grep -qE "$PROD_PIDFILES_RE"; then
  echo "[qwen38] ⛔ 拒绝启动:实例名 '$TAG' 会覆盖**生产 PID 文件**(${TAG}.pid)✗" >&2
  exit 3
fi

MEM_GATE_GIB="${MEM_GATE_GIB:-500}"  # ⭐ AGENTS 2026-10-19:A.2 qfn TP=2 = 500 GiB(旧 250 过松,作废)
_avail="$(LC_ALL=C awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || true)"
# ⭐ 审计 MAJOR 修复:两个数都必须校验成整数 —— 否则 `[ "$_avail" -lt "$MEM_GATE_GIB" ]` 报错,
#   在 set -e 下**直接退出**或(视写位置)被吞掉 ⇒ 共存内存门 **fail-open** ✗
for _pair in "MemAvailable=$_avail" "MEM_GATE_GIB=$MEM_GATE_GIB"; do
  _n="${_pair%%=*}"; _v="${_pair#*=}"
  case "$_v" in ''|*[!0-9]*) echo "[qwen38] ⛔ $_n 不是整数('$_v')⇒ 拒绝启动(共存内存门不可用)" >&2; exit 3 ;; esac
done
unset _pair _n _v
echo "[qwen38] 资源纪律: THREADS=$THREADS taskset=${TASKSET:-<不加taskset>} nice=19 oom_score_adj=$OOM_SCORE_ADJ MemAvailable=${_avail}GiB"
if [ "${ALLOW_LOW_MEM:-0}" != "1" ] && [ "$_avail" -lt "$MEM_GATE_GIB" ]; then
  echo "[qwen38] ✗ 拒绝启动:MemAvailable=${_avail}GiB < MEM_GATE_GIB=${MEM_GATE_GIB}GiB" >&2
  echo "[qwen38]   同机很可能已有生产在跑 ⇒ 确需放行请显式 ALLOW_LOW_MEM=1(自担风险)" >&2
  exit 3
fi

# ⭐⭐ BLOCKER 修复(审计 A):**启动前必须清掉同名 PID 文件** ——
#   否则"读到只要非空就 break"会读进【上一轮/别人的 PID】,随即把 oom_score_adj=800 写进去 ✗✗
#   审计实测:预置一个外来活 PID ⇒ **8/8 次**被当成自己的实例;远端 uid=1000 可写
#   ⇒ 若该 PID 是**生产 worker** ⇒ 等于**把生产标成 OOM 首杀目标** ✗(与 2026-10-04 事故同型)
# ⭐⭐⭐ 审计 MAJOR 修复(第二轮,F2):**同一 TAG 必须互斥** ——
#   ⛔ TAG 默认是 `qwen38_<秒级时刻>` ⇒ **同一秒启动两次即同名**;第二实例会 `rm -f` 掉
#      第一实例正在用的 `<TAG>.pid` / `<TAG>.log` 符号链接并重指到自己的日志 ⇒
#      实测:第一实例**读到了第二实例的日志**,被别人的 Traceback 判成 T1 致命签名(rc=5)✗
#   ⇒ 用 `mkdir`(原子操作)抢占 `<TAG>.lock`;抢不到就**拒绝启动**(fail-closed)✓
GW_LOCK="$OUTDIR/$TAG.lock"
if ! mkdir "$GW_LOCK" 2>/dev/null; then
  _oldpid="$(cat "$OUTDIR/$TAG.pid" 2>/dev/null || true)"
  if [ -n "$_oldpid" ] && pi_valid_pid "$_oldpid" && pi_alive "$_oldpid"; then
    echo "[qwen38] ⛔ 拒绝启动:实例名 '$TAG' **已被一个活着的实例占用**(pid=$_oldpid)✗" >&2
    echo "[qwen38]    并发/同秒重启请换 TAG(例如加后缀),或用 scripts/proc.sh stop $TAG 先停掉它" >&2
    exit 3
  fi
  # 旧实例已不在 ⇒ 锁是陈旧的 ⇒ 清掉重抢(这里只删锁目录,不动任何进程 ✓)
  echo "[qwen38] ⚠️ 发现陈旧锁 $GW_LOCK(登记的 pid 已不在)⇒ 清理后继续" >&2
  rmdir "$GW_LOCK" 2>/dev/null || true
  mkdir "$GW_LOCK" 2>/dev/null || { echo "[qwen38] ⛔ 无法取得实例锁 $GW_LOCK ⇒ 拒绝启动" >&2; exit 3; }
fi
unset _oldpid

# ⭐⭐⭐ 审计 MAJOR 修复(第二轮,F3/F4/F5):trap 必须**在派生之前**就装好 ——
#   原版装在第 322 行(派生之后)⇒ 派生到装 trap 之间有实测 **0.72s 无保护窗口** ✗
GW_DONE=0
_GW_IN_CLEANUP=0
_gw_cleanup() {   # $1=1 ⇒ 强制清理(失败出口用)
  [ "${_GW_IN_CLEANUP:-0}" = "1" ] && return 0          # ⭐ 重入保护(审计实测曾自我重入 1947 次 ✗)
  if [ "${GW_DONE:-0}" = "1" ] && [ "${1:-0}" != "1" ]; then return 0; fi
  _GW_IN_CLEANUP=1
  if [ -n "${SVC_PID:-}" ] && pi_valid_pid "$SVC_PID"; then
    # ⛔⛔ 审计 MAJOR:原版 `kill -TERM "-$SVC_PID"` 把 pid 文件里的值当【进程组号】,
    #      且不做归属判断 ⇒ 实测**杀掉调用方整个进程组**、并因组内含自己而**反复重入 trap** ✗
    #      (AGENTS 第 0 条明令:禁止任何形式的"按组/按模式"批量 kill ✗)
    #   ⭐ 现在两条硬门:①它必须是**我亲生的子进程** ②只会**按 PID 逐个**发信号(含其后代)✓
    if [ "$(pi_ppid "$SVC_PID" 2>/dev/null || echo x)" = "$$" ]; then
      echo "[qwen38] ⛔ 停掉本次启动的服务 pid=$SVC_PID(失败/中断出口,不留孤儿)" >&2
      local _p
      if pi_descendants "$SVC_PID" >/dev/null 2>&1; then
        for _p in $PI_DESC_SEEN; do
          pi_valid_pid "$_p" || continue
          [ "$_p" = "$$" ] && continue
          kill -TERM "$_p" 2>/dev/null || true
        done
      fi
      kill -TERM "$SVC_PID" 2>/dev/null || true
      sleep 1
      if pi_descendants "$SVC_PID" >/dev/null 2>&1; then
        for _p in $PI_DESC_SEEN; do
          pi_valid_pid "$_p" || continue
          [ "$_p" = "$$" ] && continue
          kill -KILL "$_p" 2>/dev/null || true
        done
      fi
      kill -KILL "$SVC_PID" 2>/dev/null || true
    else
      echo "[qwen38] ⚠️ 清理被跳过:pid=${SVC_PID:-} 不是本脚本($$)的子进程 ⇒ 一律不动它 ✗" >&2
    fi
  fi
  rmdir "$GW_LOCK" 2>/dev/null || true
  return 0
}
trap '_gw_cleanup; exit 143' TERM
trap '_gw_cleanup; exit 130' INT
trap '_gw_cleanup' EXIT

rm -f "$OUTDIR/$TAG.pid" 2>/dev/null || true
# ⭐ 同样要清掉**陈旧的日志符号链接** —— 否则 `wait_ready` 会去扫**上一轮**的日志,
#   那里残留的 T1 行会被当成本轮故障 ⇒ **假杀** ✗(子壳会立刻重建该链接 ✓)
rm -f "$LOG" 2>/dev/null || true

# ⭐ 日志按【真 PID】命名(与 serve_v41.sh 同规):旧的 `> "$LOG"` 固定名会互相截断 ✗
(
  LOGFILE="$OUTDIR/$TAG.$BASHPID.log"
  ln -sfn "$(basename "$LOGFILE")" "$LOG" 2>/dev/null || true
  echo "$BASHPID" > "$OUTDIR/$TAG.pid"
  exec > "$LOGFILE" 2>&1
  echo "[qwen38] service log=$LOGFILE pid=$BASHPID"
  exec env \
    PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}" \
    HF_HUB_OFFLINE=1 \
    VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}" \
    VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-120}" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_EXPERTS_LOAD_DEVICE=cpu \
    XIAOTU_ENV_FILE="$XTU_ENV_FILE" \
    XIAOTU_PLE_CPU="$PLE_CPU" \
    VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GPU_PREFILL_MIN" \
    XIAOTU_MOE_THREADS="$THREADS" \
    OMP_NUM_THREADS=1 \
    "${NCTL[@]}" "${TS_PREFIX[@]}" nice -n 19 \
    "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
) &
# ⭐⭐⭐ BLOCKER 修复(第二轮复审,审计员 A)⭐⭐⭐
#   ⛔ 上一版"四重校验"校验的是【看起来像不像】,不是【归属】✗ ——
#      实测:pid 文件在 `rm -f` **之后**被写入任何外部活 PID ⇒ 四条判据(合法数/存活/mtime/不在生产名单)
#      **全部通过**,**6/6 次被采纳** ⇒ 照样把 oom_score_adj 注进去、TERM 时把它杀掉 ✗
#   ⛔ 更根本:**生产的 worker 子进程从来不写进任何 <prod>.pid 文件** ⇒
#      "生产名单"这条判据对**真正的生产进程永远不命中** ✗
#   ⭐⭐ 结构性修法:**根本不要读 pid 文件来判归属** ——
#      后台作业的 PID **父壳本来就知道**:`$!` ✓;子壳 `exec env … python` 后 **PID 不变** ✓
#      ⇒ 归属判据从"查名单"升级为"**它就是我亲生的子进程**" ⇒ **不可能被外部伪造** ✓
SVC_PID="$!"
_gw_own_ok=0
if pi_valid_pid "$SVC_PID" && pi_alive "$SVC_PID" \
   && [ "$(pi_ppid "$SVC_PID" 2>/dev/null || echo x)" = "$$" ]; then
  _gw_own_ok=1
fi
if [ "$_gw_own_ok" != "1" ]; then
  _gw_pp="?"; [ -n "${SVC_PID:-}" ] && _gw_pp="$(pi_ppid "$SVC_PID" 2>/dev/null || echo '?')"
  echo "[qwen38] ⛔ 后台子壳未成为本脚本(pid=$$)的【直接子进程】(pid='${SVC_PID:-}' ppid='$_gw_pp')⇒ 拒绝继续" >&2
  { tail -20 "$LOG" 2>/dev/null || true; } | sed 's/^/      /' >&2 || true
  exit 3
fi
unset _gw_own_ok _gw_pp
if echo "$OOM_SCORE_ADJ" > "/proc/$SVC_PID/oom_score_adj" 2>/dev/null; then
  echo "[qwen38] oom_score_adj=$OOM_SCORE_ADJ 已注入 pid=$SVC_PID ✓(OOM 时内核先杀本实例)"
else
  echo "[qwen38] ⚠️ oom_score_adj 注入失败(pid=$SVC_PID)⇒ 内存告急时【生产有被误杀的风险】"
  echo "[qwen38]    ⚠️ 审计 MINOR:此路默认 **fail-open**(只告警);要严格请 REQUIRE_OOM_ADJ=1"
  if [ "${REQUIRE_OOM_ADJ:-0}" = "1" ]; then
    echo "[qwen38] ⛔ REQUIRE_OOM_ADJ=1 且注入失败 ⇒ 停掉刚起的服务并失败退出" >&2
    _gw_cleanup 1; exit 3
  fi
fi
echo "[qwen38] pid=$SVC_PID(归属判据 = 本脚本 $$ 的**直接子进程** ✓,不依赖 pid 文件内容)"
echo "[qwen38] 日志(按 PID 命名)= $OUTDIR/$TAG.$SVC_PID.log;稳定符号链接 $LOG → $TAG.$SVC_PID.log"

# ⭐⭐ 就绪 / 错误 / 停滞【三重门】—— 唯一实现 = `scripts/wait_ready.sh` ✓
#   ⛔ **不再在本脚本里手写循环**!理由(2026-10-15 两名独立审计员对 v3 的判定):
#     · 手写版就绪判定是 **子串匹配** `Application startup complete` ⇒ 违反 **R38 ④**(必须校验端口属主)✗
#     · 手写版**没有停滞门** ⇒ 进程活着但卡死时**永远等下去** ✗(R53;这正是用户最痛的点)
#     · 手写版把"就绪"排在"错误扫描"**之前** ⇒ 就绪与错误同时出现时误报 READY ✗(审计 B8)
#     · 手写版 `ERRSCAN` 未校验 ⇒ `ERRSCAN=banana` 会**静默关掉**错误扫描 ✗(fail-open)
#   ⇒ 委派后三重门齐备:①R38 ④ 端口属主 ②R53 停滞(RSS + read_bytes 双单调量)③R54 T1 致命签名 ✓
LOGDIR="$OUTDIR"                          # ⭐ 让 wait_ready 找到同目录的 <TAG>.pid / <TAG>.log ✓
READY_TIMEOUT="${READY_TIMEOUT:-3600}"    # 总超时(停滞门会**先于**它触发)✓
STALL_SEC="${STALL_SEC:-900}"             # R53:实测校准下限(最长"最后新高→ready" = 483.9s)✓
export LOGDIR READY_TIMEOUT STALL_SEC
export INTERVAL="${INTERVAL:-15}" ERRSCAN="${ERRSCAN:-1}" \
       ERRSCAN_FROM_END="${ERRSCAN_FROM_END:-0}" ALLOW_RSS_ONLY="${ALLOW_RSS_ONLY:-0}" \
       BENIGN_EXTRA="${BENIGN_EXTRA:-}" EXPECT_GB="${EXPECT_GB:-0}"

echo "[qwen38] 就绪门 → wait_ready.sh $TAG :$PORT(总超时 ${READY_TIMEOUT}s / 停滞门 ${STALL_SEC}s)"
# ⭐⭐ 审计 MAJOR 修复(第二轮,F3):原来 `bash wait_ready.sh …` 跑在【前台】⇒
#    bash 会把 trap **推迟到前台命令结束** ⇒ 只把 TERM/INT 发给启动器时,
#    实测 t≈33s 时**启动器 / 门 / 服务三者全都活着**,一直耗到 READY_TIMEOUT(默认 3600s)✗
#    ⇒ 改成**后台作业 + `wait`**:`wait` 是内建,遇到被 trap 的信号会**立即返回** ✓
bash "$ROOT/scripts/wait_ready.sh" "$TAG" "$PORT" "$READY_TIMEOUT" &
WR_PID=$!
rc=0
wait "$WR_PID" || rc=$?

# ⭐⭐ 审计 MAJOR 修复(第二轮,F5):⛔ 原来 rc=1/2/3/4/5 的失败出口**只 echo 就 exit**,
#   服务继续活着(实测 rc=2 后服务仍在,占着显存/内存)✗
#   ⇒ 除 rc=0(真的 READY)外,**一律停掉刚起的服务**;确需保留请显式 KEEP_ON_FAIL=1 ✓
if [ "$rc" != "0" ] && [ "${KEEP_ON_FAIL:-0}" = "1" ]; then
  echo "[qwen38] ⚠️ KEEP_ON_FAIL=1 ⇒ 即使判失败也【保留】服务 pid=$SVC_PID(请自行 proc.sh adopt/stop)" >&2
else
  case "$rc" in 0) : ;; *) _gw_cleanup 1 ;; esac
fi
case "$rc" in
  0) GW_DONE=1
     echo "[qwen38] ✅ READY tag=$TAG(端口 $PORT);实例锁 $GW_LOCK 保留(同 TAG 重启会被拒,换 TAG 或先停)"
     exit 0 ;;
  1) echo "[qwen38] ⛔ 进程在就绪前【死亡】⇒ 启动失败(tag=$TAG)" >&2; exit 1 ;;
  2) echo "[qwen38] ⛔ 就绪【超时】${READY_TIMEOUT}s(tag=$TAG)" >&2; exit 2 ;;
  3) echo "[qwen38] ⛔ 就绪门【拒绝运行】(参数/环境错,见上)⇒ 本次启动未获判定" >&2; exit 3 ;;
  4) echo "[qwen38] ⛔ 【停滞判死】(tag=$TAG;R53 —— 进程活着,但它已经死了)" >&2; exit 4 ;;
  5) echo "[qwen38] ⛔ 日志出现【致命签名】(tag=$TAG;R54 T1)⇒ 立刻判失败" >&2; exit 5 ;;
  130|143) echo "[qwen38] ⛔ 就绪门被信号中断(rc=$rc)" >&2; exit "$rc" ;;
  *) echo "[qwen38] ⛔ 就绪门返回未知码 $rc" >&2; exit 6 ;;
esac
