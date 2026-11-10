#!/bin/bash
# qfn MUSA prefill A/B wrapper —— 两臂**除 HOST_KM 外逐字相同**。
#
# 用法：  HOSTKM=0 bash run_qfn_ab_hostkm.sh    # 基线臂（默认路径）
#         HOSTKM=1 bash run_qfn_ab_hostkm.sh    # 新路径（整块连续写 ring）
#
# 纪律符合性：
#   * 12 层常驻列表与 gpupf40 **完全一致**（未改动）
#   * grouped 正确性修复未触碰（BMM_MODE=grouped）
#   * 去掉所有诊断 flag（SHAPE_DBG / DMA_BENCH / DMA_CPP_LOG）⇒ 不污染基线
#   * HOST_KM 默认 0 ⇒ 新代码默认关闭
set -u
HOSTKM="${HOSTKM:-0}"
# EVT=1：采样 GPU event，输出 [gpf-evt] 分层归因（含本轮新增 c_decode）。
# EVT_EVERY=1：每层都采（默认 8）。两者默认关闭 ⇒ 不影响吞吐口径。
EVT="${EVT:-0}"
EVTEVERY="${EVTEVERY:-1}"
# TLAYER=1：每层打印无歧义分段计时 [gpf-tl]（tiles/kmajor/bounds/loop/total）。默认关。
TLAYER="${TLAYER:-0}"
# W13TN/W2TN：环存 N-major ⇒ `_host_km_fill` 的 CPU 装配变成大块 memcpy。
#   实测(scripts/test_host_km_fill_tn.py)：K-major 205.7ms/tile -> N-major 74.7ms/tile，
#   2.75×，且 `torch.equal` 逐字节相等。默认 0（保持原 K-major 行为）。
W13TN="${W13TN:-0}"
W2TN="${W2TN:-0}"
# BOUNDS_EV=1：给 seg_start 记窄事件，侧流只等 cumsum 而非整条主流。
#   替代原先无效的 BOUNDS_PIPE(wait_stream 等价于全同步，见 §9.13)。默认 0。
BOUNDS_EV="${BOUNDS_EV:-0}"
# HK_COUNT=1：`_host_km_fill` 逐层记录**调用次数与字节数**，用来给 event 口径定案。默认 0。
HK_COUNT="${HK_COUNT:-0}"
# CPT=1：DMA 由独立 producer 线程发（历史默认，wrapper 一直强制 1）。
# CPT=0：主线程发 DMA（源码缺省）。用于 A/B 定位主线程序列点。
CPT="${CPT:-1}"
# STACK_WD=<秒>：主线程在 tolist 里阻塞超过该秒数就自动 dump 全线程栈（0=关）。
STACK_WD="${STACK_WD:-0}"
# BSPLIT=1：把 `tolist()` 拆成 `.cpu()` 与列表构建两段分别计时（语义不变）。
BSPLIT="${BSPLIT:-0}"
# PROF=1：打开 torch profiler，抓逐 kernel CUDA 时间表（出 37s 在哪）。
#   CALLS 必须 **≤ 实际 MoE 调用次数**，否则 profiler 永不 flush、零输出（历史踩过）。
PROF="${PROF:-0}"
PROF_CALLS="${PROF_CALLS:-40}"
# ALLOC=<字符串>：直接覆盖 PYTORCH_MUSA_ALLOC_CONF。默认保留 expandable_segments:True。
#   验假设：`bounds` 的数秒停顿是否来自 expandable_segments 的 segment 增长/remap（设备级同步）。
#   传 ALLOC=off 表示**不设**该变量（用分配器默认行为）。
ALLOC="${ALLOC-expandable_segments:True}"
# EAGER=1：关掉图捕获（--enforce-eager）。用来验"分段 CUDA graph 重捕获"是否是
#   `bounds` 数秒尖峰的来源。注意 eager 整体会慢 2~4×（serve_qwen38.sh R64），
#   本轮只看**尖峰是否消失**，不看吞吐。
EAGER="${EAGER:-0}"
ALLOC_EVN=()
if [ "$ALLOC" != "off" ]; then
  ALLOC_EVN=(-e "PYTORCH_MUSA_ALLOC_CONF=$ALLOC")
fi
PROF_EVN=()
if [ "$PROF" = "1" ]; then
  PROF_EVN=(-e XIAOTU_TORCH_PROFILE=/tmp/xtprof -e "XIAOTU_TORCH_PROFILE_CALLS=$PROF_CALLS")
fi
TAG="qfn_ab_hostkm${HOSTKM}_evt${EVT}_tl${TLAYER}_tn${W13TN}${W2TN}_bev${BOUNDS_EV}_hc${HK_COUNT}_cpt${CPT}_sw${STACK_WD}_bs${BSPLIT}_pf${PROF}_al${ALLOC//[:,\/]/_}_eg${EAGER}"
docker rm -f qfn-tp1 2>/dev/null || true
# 杀容器后等 20 s 让 GPU 释放显存（AGENTS.md 明令）
sleep 20
docker run -d --name qfn-tp1 --security-opt seccomp=unconfined --ipc host --network host --shm-size 32g \
  -e MUSA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_PLUGINS=musa,vllm_xiaotu_moe -e GPU_DEVICE=MUSA -e VLLM_EXPERTS_LOAD_DEVICE=cpu \
  -e EAGER="$EAGER" -e ENV=/home/mccuseradmin/s5000/venv032 \
  -e XTU_TREE=/home/mccuseradmin/s5000/xiaotu-llm \
  -e CKPT=/data/models/Qwen/Qwen3.8-Flash-Next-FP8 -e TAG="$TAG" \
  -e PORT=8140 -e GPUS=0 -e TP=1 -e MAXLEN=65536 -e SEQS=4 -e MBT=16384 \
  -e GPU_PREFILL_MIN=2048 -e XIAOTU_GPF_TT=0 -e XIAOTU_GPF_BMM_MODE=grouped \
  -e XIAOTU_GPF_CPT="$CPT" -e XIAOTU_GPF_BOUNDS_REUSE=0 \
  -e XIAOTU_GPF_W13TN="$W13TN" -e XIAOTU_GPF_W2TN="$W2TN" \
  -e XIAOTU_GPF_HOST_KM="$HOSTKM" \
  -e XIAOTU_GPF_EVT="$EVT" -e XIAOTU_GPF_EVT_EVERY="$EVTEVERY" \
  -e XIAOTU_GPF_TLAYER="$TLAYER" -e XIAOTU_GPF_HK_COUNT="$HK_COUNT" \
  -e XIAOTU_GPF_STACK_WD="$STACK_WD" -e XIAOTU_GPF_BOUNDS_SPLIT="$BSPLIT" \
  ${PROF_EVN[@]+"${PROF_EVN[@]}"} \
  ${ALLOC_EVN[@]+"${ALLOC_EVN[@]}"} \
  -e GPU_UTIL=0.90 -e MEM_GATE_GIB=500 -e THREADS=80 \
  -e XIAOTU_MOE_GPU_RESIDENT_LAYERS=15,9,11,16,31,3,39,25,7,23,24,13 \
  -e EXTRA_ARGS='--compilation-config {"cudagraph_mode":"PIECEWISE","mode":"VLLM_COMPILE"}' \
  -v /home/mccuseradmin/s5000/cache/root:/root/.cache -v /data/models:/data/models:ro \
  -v /home/mccuseradmin/s5000:/home/mccuseradmin/s5000 \
  -w /home/mccuseradmin/s5000/xiaotu-llm \
  --entrypoint bash registry.mthreads.com/mcconline/musa-pytorch-release-public:rc5.2.0-v2.9.1.post1-S5000-py310 \
  -lc "export LD_LIBRARY_PATH=/home/mccuseradmin/s5000/venv/lib/python3.10/site-packages/z3/lib:/opt/intel/oneapi/mkl/lib/intel64:\$LD_LIBRARY_PATH; bash scripts/serve_qwen38.sh; echo LAUNCHER_DONE; sleep infinity"
echo RUN_RC=$? TAG=$TAG HOSTKM=$HOSTKM
