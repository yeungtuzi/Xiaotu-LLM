"""端到端基准：**15K prompt × C=1,2,4**，报 ①prefill 速度 ②TTFT ③decode 速度 ④解码步长。

口径（用户 2026-10-25 定稿，见 `dev-docs/MUSA_BRIEF_2026-10-24.md` §7）：
  官方 qfn / 当前 MBT / TP=2 / **MTP off** / **常驻 12 层** / GPU prefill on / prompt 15K。

为什么单独写（而不是复用 `bench_6x.py`）：
  `bench_6x.py` 只测 **prefill**（每次只出 1 token，C=1）；`bench_llm.py` 只测 **decode**
  （短 prompt、C 默认 32）。本基准要**同一实例内**给齐四项指标 × 三种并发 ✓

用法（由 launcher 起，别单独跑）：
  MUSA_BENCH_PY=.../bench_e2e_qfn.py \
  XIAOTU_MOE_GPU_RESIDENT_LAYERS="15,9,11,16,31,3,39,25,7,23,24,13" \
  bash run_bench_cached.v17.sh
"""
import hashlib
import json
import os
import time


_COPY_STATS = {"h2d": 0, "d2h": 0, "cpu": 0, "to": 0}


def _install_copy_counter():
    """`XIAOTU_COPY_COUNT=1`：统计 **H2D / D2H 次数**（"纯 GPU 模式"的机械证据）。

    做法：包住 `torch.Tensor.to` 与 `torch.Tensor.cpu`，只**计数**不改变行为。
    判据：稳态 decode 期间 `d2h/h2d` 应为 **0**（PLE 关闭、权重全驻留时）。
    """
    import torch
    if getattr(torch.Tensor, "_xtu_copy_counted", False):
        return
    _orig_to = torch.Tensor.to
    _orig_cpu = torch.Tensor.cpu

    def _dst_is_cpu(args, kwargs):  # noqa: ANN001
        dev = kwargs.get("device", None)
        if dev is None and args:
            dev = args[0]
        if dev is None:
            return None
        try:
            return torch.device(dev).type == "cpu"
        except Exception:  # noqa: BLE001
            return None

    def to(self, *args, **kwargs):  # noqa: ANN001
        _COPY_STATS["to"] += 1
        tgt = _dst_is_cpu(args, kwargs)
        if tgt is True and self.is_cuda:
            _COPY_STATS["d2h"] += 1
        elif tgt is False and not self.is_cuda:
            _COPY_STATS["h2d"] += 1
        return _orig_to(self, *args, **kwargs)

    def cpu(self, *args, **kwargs):  # noqa: ANN001
        _COPY_STATS["to"] += 1
        if self.is_cuda:
            _COPY_STATS["d2h"] += 1
        else:
            _COPY_STATS["cpu"] += 1
        return _orig_cpu(self, *args, **kwargs)

    torch.Tensor.to = to
    torch.Tensor.cpu = cpu
    torch.Tensor._xtu_copy_counted = True


def main():
    if os.environ.get("XIAOTU_COPY_COUNT", "0") == "1":
        _install_copy_counter()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    n = int(os.environ.get("QFN_LEN", "15360"))
    out_tok = int(os.environ.get("QFN_OUT", "64"))
    concs = [int(c) for c in os.environ.get("QFN_CONC", "1,2,4").split(",")]
    reps = int(os.environ.get("QFN_REPEATS", "2"))
    model = os.environ.get("QFN_MODEL", "/root/models/Qwen3.8-Flash-Next-FP8")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    _kv = os.environ.get("QFN_KV_BYTES")
    _extra = {"kv_cache_memory_bytes": int(_kv)} if _kv else {}
    # ⭐ `QFN_MAXSEQS`（2026-10-26）：**全 GPU 驻留**时 `max_num_seqs` 直接决定
    #   GDN/mamba 状态组的显存（实测默认值把"上传后剩的 7.6 GiB"吃光 ⇒ KV 定容期 OOM）。
    #   只在 >0 时传，默认不传 ⇒ 与旧口径逐字一致 ✓
    _mns = int(os.environ.get("QFN_MAXSEQS", "0") or 0)
    if _mns > 0:
        _extra["max_num_seqs"] = _mns
    # ⭐ `QFN_CAP_SIZES`（2026-10-26）：**只捕获指定 token 尺寸**的图。
    #   动机：MUSA 上每张 PIECEWISE 图约 1.25 GiB，48 层全驻留后只剩 ~2.2 GiB
    #   ⇒ 只容得下 2 张。默认 vLLM 按 [1,2,4,8,...] 连续捕获 ⇒ 捕到第 3 张就 OOM；
    #   这里允许显式给 "1,8" 这样的稀疏集合，把有限的图池花在**最需要的两个尺寸**上。
    _caps = os.environ.get("QFN_CAP_SIZES", "").strip()
    if _caps:
        _extra["compilation_config"] = {
            "cudagraph_capture_sizes": [int(x) for x in _caps.split(",") if x.strip()]
        }
    llm = LLM(
        model=model,
        tensor_parallel_size=int(os.environ.get("QFN_TP", "2")),
        max_model_len=int(os.environ.get("QFN_MAXLEN", "17408")),
        gpu_memory_utilization=float(os.environ.get("QFN_GPU_UTIL", "0.72")),
        # ⭐ `QFN_EAGER`（2026-10-26）：默认 1 = 保持旧的 eager 口径。
        #   置 0 ⇒ 允许 vLLM 做 CUDA 图捕获 —— 用来量"decode 是不是被每步上千个
        #   小 kernel 的发射开销拖住"（驻留后 decode 578 ms/步 vs prefill 51 ms/token）
        enforce_eager=(os.environ.get("QFN_EAGER", "1") != "0"),
        trust_remote_code=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        max_num_batched_tokens=int(os.environ.get("QFN_MBT", "4096")),
        **_extra,
    )

    text = "The following is a detailed technical report about GPU kernels. " * 3000
    ids = tok.encode(text)[:n]
    prompt = tok.decode(ids)
    real = len(tok.encode(prompt))
    resident = os.environ.get("XIAOTU_MOE_GPU_RESIDENT_LAYERS", "")
    print(f"[e2e] prompt_tokens={real} out_tokens={out_tok} concs={concs} "
          f"mbt={os.environ.get('QFN_MBT', '4096')} mtp=off "
          f"resident={resident or '(none)'}", flush=True)

    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    spN = SamplingParams(temperature=0.0, max_tokens=out_tok)

    # ⚠️ 预热**必须用短 prompt**：预热期 `in_profile_run()`/首次请求会走 **CPU prefill**
    #    （`mixed_experts.py:1897` 明确在 startup 期不放行 GPU 预填），
    #    若用 15K prompt 预热 ⇒ 36 个非驻留层在 CPU 上算 15K ⇒ **要多花 ~10 分钟** ✗
    #    短 prompt 同样能把 decode 路径与内核都编译/热身一遍 ✓
    t0 = time.perf_counter()
    llm.generate(["hello"], spN)         # 预热（短 prompt；丢弃）
    print(f"[e2e] warmup(short) {time.perf_counter() - t0:.1f}s", flush=True)

    # ⭐ 2026-10-26【质量探针】`QFN_PROBE=1` 时先做一道**看得见文本**的贪心问答。
    #   为什么必须有：NVFP4 接线的失败模式是"能跑但数全错"（例如组尺度被数值转换毁掉），
    #   只看 tok/s 分不出对错 ✗。默认关闭 ⇒ 现有 FP8 口径零影响 ✓
    # ⭐ `QFN_DUMP=<path>`：落 **贪心 token id**，供"合并前后逐 token 对齐"用。
    #   只记录、不改生成参数 ⇒ 未设置时零影响 ✓
    _dump_path = os.environ.get("QFN_DUMP", "").strip()
    _dump = {"model": model, "prompt_tokens": real, "out_tokens": out_tok,
             "concs": concs, "resident": resident, "probes": [], "decode": []}

    if os.environ.get("QFN_PROBE", "0").strip() == "1":
        _qs = ["What is 2 + 2? Answer with a single number.",
               "Repeat exactly this sentence: the quick brown fox jumps over the lazy dog."]
        for _q in _qs:
            _o = llm.generate([_q], SamplingParams(temperature=0.0, max_tokens=48))
            _txt = _o[0].outputs[0].text.replace("\n", "\\n")
            print(f"[probe] Q={_q!r} -> {_txt!r}", flush=True)
            _dump["probes"].append({"q": _q, "text": _o[0].outputs[0].text,
                                    "token_ids": list(_o[0].outputs[0].token_ids)})

    rows = []
    for C in concs:
        prompts = [prompt] * C
        # ── ① TTFT + ② prefill 速度：只出 1 token，取多次最好值 ──
        best_ttft = 1e9
        for _i in range(reps):
            t0 = time.perf_counter()
            llm.generate(prompts, sp1)
            _dt = time.perf_counter() - t0
            best_ttft = min(best_ttft, _dt)
            # ⭐ 逐次落日志：即使后续轮次失败/被杀，也能拿到部分数据 ✓
            print("[e2e-run] C=%d phase=ttft it=%d/%d elapsed=%.1fs prefill=%.1f tok/s"
                  % (C, _i + 1, reps, _dt, C * real / _dt), flush=True)
        # ── ③ decode 速度 + ④ 步长：出 out_tok，扣掉 TTFT ──
        best_dec = 1e9
        for _i in range(reps):
            # ⭐ `XIAOTU_STEP_CPU=1`：同时记录 **CPU 时间**（`process_time`，含全部线程）
            #    与墙钟 ⇒ 用来证明"纯 GPU 模式"下 CPU 几乎不干活：
            #    `cpu_ratio = cpu / wall` 接近 0 ⇒ CPU 在等 GPU（图回放），不是它在算 ✓
            _cc0 = dict(_COPY_STATS) if os.environ.get("XIAOTU_COPY_COUNT", "0") == "1" else None
            _c0 = time.process_time()
            t0 = time.perf_counter()
            _res = llm.generate(prompts, spN)
            tot = time.perf_counter() - t0
            if _dump_path and not _dump["decode"]:
                _dump["decode"] = [
                    {"C": C, "idx": _i, "text": _r.outputs[0].text,
                     "token_ids": list(_r.outputs[0].token_ids)}
                    for _i, _r in enumerate(_res)]
            _cpu = time.process_time() - _c0
            _dec = max(tot - best_ttft, 1e-9)
            best_dec = min(best_dec, _dec)
            print("[e2e-run] C=%d phase=decode it=%d/%d total=%.1fs decode=%.1f tok/s step=%.2f ms"
                  % (C, _i + 1, reps, tot, C * (out_tok - 1) / _dec, _dec / max(out_tok - 1, 1) * 1000),
                  flush=True)
            if _cc0 is not None:
                _d = {k: _COPY_STATS[k] - _cc0[k] for k in _COPY_STATS}
                print("[e2e-copy] C=%d H2D=%d D2H=%d（.to/.cpu 调用 %d 次；"
                      "纯 GPU 模式判据：D2H=0 且 H2D=0）"
                      % (C, _d["h2d"], _d["d2h"], _d["to"]), flush=True)
            if os.environ.get("XIAOTU_STEP_CPU", "0") == "1":
                _nstep = max(out_tok - 1, 1)
                print("[e2e-cpu] C=%d cpu=%.3fs wall=%.3fs cpu/wall=%.3f "
                      "cpu_ms/step=%.2f（CPU 若≈0 ⇒ 纯 GPU 回放）"
                      % (C, _cpu, tot, _cpu / max(tot, 1e-9), _cpu / _nstep * 1000),
                      flush=True)
        step_ms = best_dec / max(out_tok - 1, 1) * 1000
        rows.append((C, best_ttft * 1000, C * real / best_ttft, C * (out_tok - 1) / best_dec, step_ms))
        print("[e2e] C=%d TTFT=%7.0f ms  prefill=%7.1f tok/s  decode=%6.1f tok/s  步长=%6.2f ms"
              % rows[-1], flush=True)

    if _dump_path:
        _blob = json.dumps(_dump, ensure_ascii=False).encode()
        with open(_dump_path, "wb") as _fh:
            _fh.write(_blob)
        print(f"[e2e-dump] {_dump_path} sha256={hashlib.sha256(_blob).hexdigest()}",
              flush=True)
    print("[e2e] ---- 汇总（口径：15K prompt / TP2 / MTP off / 常驻 %s 层 / GPU prefill on）----"
          % (len(resident.split(",")) if resident else 0), flush=True)
    print("[e2e] %3s %10s %12s %11s %10s" % ("C", "TTFT(ms)", "prefill(tok/s)", "decode(tok/s)", "步长(ms)"),
          flush=True)
    for C, ttft, pf, dec, step in rows:
        print("[e2e] %3d %10.0f %12.1f %11.1f %10.2f" % (C, ttft, pf, dec, step), flush=True)


if __name__ == "__main__":
    main()
