#!/usr/bin/env python3
"""GPU0 离线剖析:ds41f TP=1 + torch profiler ⇒ 拿 verify 步的 GPU 内核账本。
用法(远端 GPU0 上跑):  CUDA_VISIBLE_DEVICES=0 XIAOTU_MOE_THREADS=48 python3 offline_profile_run.py
设计:起 in-process 引擎(无任何进程管理/kill 逻辑),载真实转录 prompt,
warm 若干步后 start_profile → 生成 256 tok → stop_profile,trace 落盘。"""
import json, os, sys, time

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

from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    max_model_len=32768,
    max_num_batched_tokens=8192,
    gpu_memory_utilization=0.85,
    kv_cache_dtype="fp8_ds_mla",
    trust_remote_code=True,
    speculative_config={"method": "dspark", "num_speculative_tokens": 5,
                        "draft_sample_method": "probabilistic"},
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "mode": "VLLM_COMPILE"},
    profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=TRACE_DIR),
    kernel_config={"enable_jit_warmup": False},
)
sp = SamplingParams(temperature=0.0, max_tokens=48)
print("[warm] generating 48 tokens…", flush=True)
t0 = time.time()
llm.generate([{"prompt": prompt} if isinstance(prompt, str) else {"messages": prompt}], sp)
print(f"[warm] done {time.time()-t0:.1f}s", flush=True)

eng = llm.llm_engine
sp2 = SamplingParams(temperature=0.0, max_tokens=256)
print("[prof] start_profile + 256 tokens…", flush=True)
eng.start_profile()
t0 = time.time()
llm.generate([{"prompt": prompt} if isinstance(prompt, str) else {"messages": prompt}], sp2)
eng.stop_profile()
print(f"[prof] stopped after {time.time()-t0:.1f}s; traces in {TRACE_DIR}", flush=True)
