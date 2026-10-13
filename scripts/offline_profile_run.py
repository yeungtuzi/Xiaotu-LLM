#!/usr/bin/env python3
"""GPU0 离线剖析:ds41f TP=1 + torch profiler ⇒ 拿 verify 步的 GPU 内核账本。
用法(远端 GPU0 上跑):  CUDA_VISIBLE_DEVICES=0 XIAOTU_MOE_THREADS=48 python3 offline_profile_run.py
设计:起 in-process 引擎(无任何进程管理/kill 逻辑),载真实转录 prompt,
warm 若干步后 start_profile → 生成 256 tok → stop_profile,trace 落盘。"""
import json, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 脚本内钉死:只用 GPU0(实验卡),不信外层环境
os.environ.setdefault("XIAOTU_OOT_OVERRIDE", "0")
os.environ.setdefault("VLLM_EXPERTS_LOAD_DEVICE", "cpu")
os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")   # 生产同款:关自动 breakable(离线自动开=capture 炸的隐藏 diff)

os.kill_pid = None
try:
    with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
        f.write("800")   # 只调高(自我牺牲优先),非 root 合法
except Exception as e:
    print("oom adj failed:", e)

MODEL = "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master"
DATA = os.environ.get("PROMPT_FILE", "/home/user/tmp/c4rerun_20261012/f_r11.jsonl")
TRACE_DIR = os.environ.get("TRACE_DIR", "/home/user/lvllm-tmp/p0/proftraces")
os.makedirs(TRACE_DIR, exist_ok=True)

prompt = None
for line in open(DATA):
    d = json.loads(line)
    prompt = d.get("prompt") or d.get("messages") or d.get("input")
    if prompt:
        break
assert prompt, "no prompt found"

import torch as _tc
print("[chk] visible cuda devices:", _tc.cuda.device_count(), flush=True)
assert _tc.cuda.device_count() == 1, "必须恰好 1 张可见卡(=GPU0)"
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    max_model_len=32768,
    max_num_batched_tokens=8192,
    gpu_memory_utilization=float(os.environ.get("GMI", "0.85")),
    kv_cache_dtype="fp8_ds_mla",
    trust_remote_code=True,
    speculative_config={"method": "dspark", "num_speculative_tokens": 5,
                        "draft_sample_method": "probabilistic"},
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "mode": "VLLM_COMPILE"},
    profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=TRACE_DIR),
    kernel_config={"enable_jit_warmup": True},   # 生产同款 True(False=capture 期现编译 triton ⇒ capture invalidated)
)
sp = SamplingParams(temperature=0.0, max_tokens=48)
print("[warm] generating 48 tokens…", flush=True)
t0 = time.time()
llm.generate([{"prompt": prompt} if isinstance(prompt, str) else {"messages": prompt}], sp)
print(f"[warm] done {time.time()-t0:.1f}s", flush=True)

eng = llm.llm_engine
sp2 = SamplingParams(temperature=0.0, max_tokens=256)
if int(os.environ.get("DUTY_SECONDS", "0")):
    # 占空比模式(B393 追问用):不碰 profiler,持续 decode 供 dmon 采样(TP=1 对照组)
    t_end = time.time() + int(os.environ["DUTY_SECONDS"])
    n_tok = 0
    while time.time() < t_end:
        spq = SamplingParams(temperature=0.0, max_tokens=200)
        outs = llm.generate([{"prompt": prompt} if isinstance(prompt, str) else {"messages": prompt}], spq)
        n_tok += len(outs[0].outputs[0].token_ids)
        print(f"[duty] tok~{n_tok} left~{t_end-time.time():.0f}s", flush=True)
    sys.exit(0)
print("[prof] start_profile + 256 tokens…", flush=True)
eng.start_profile()
t0 = time.time()
llm.generate([{"prompt": prompt} if isinstance(prompt, str) else {"messages": prompt}], sp2)
eng.stop_profile()
print(f"[prof] stopped after {time.time()-t0:.1f}s; traces in {TRACE_DIR}", flush=True)
