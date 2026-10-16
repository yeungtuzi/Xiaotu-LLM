# vllm-xtu-moe

[![Version](https://img.shields.io/github/v/tag/yeungtuzi/vllm-xtu-moe?label=version&color=blue)](https://github.com/yeungtuzi/vllm-xtu-moe/releases)
[![License](https://img.shields.io/github/license/yeungtuzi/vllm-xtu-moe?color=lightgrey)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-plugin-blueviolet)](https://github.com/vllm-project/vllm)
[![CPU](https://img.shields.io/badge/CPU-AVX512--VNNI%20%C2%B7%20BF16%20%C2%B7%20MXFP4%20%C2%B7%20FP8-orange)](#project-highlights)
[![Platform](https://img.shields.io/badge/Platform-2%C3%97%20EPYC%209654%20%2B%203%C3%97%20A100-76b900)](#measurement-host)
[![Models](https://img.shields.io/badge/models-DeepSeek--V4.1--Flash%20%C2%B7%20GLM--5.3--Flash%20%C2%B7%20Qwen3.8-green)](docs/MODEL_GUIDES.md)

> **XTU = X Transformers Unity** (pronounced *"xiao tu"*, Chinese for "little rabbit") —
> a **high-performance MoE inference acceleration layer**.
>
> It is a **vLLM plugin for very large MoE models**. The core idea is to split the work:
>
> * the **GPU** handles the **non-expert** parts — attention, KV cache, embeddings;
> * the **CPU** holds the **expert weights** and does the **expert compute**;
> * for **long prompts**, part of the expert compute is **streamed onto the GPU**.
>
> In other words: even when the **expert weights do not fit in VRAM**, a large MoE model
> still runs — while **changing mainline vLLM as little as possible**.

[中文](README.md) · English (default)

> **📌 Current release: v0.2.8** (2026-10-15) — Hot experts resident on the GPU: the GPU and the CPU decode together
> [Release notes](RELEASE_NOTES_v0.2.8.md) · [Benchmark protocol](docs/BENCH_PROTOCOL.md)

---

## Project highlights

1. **Runs very large MoE models on a single GPU (>40 GB) with enough system RAM** (verified on DeepSeek-V4.1-Flash, GLM-5.3-Flash, MiMo-V2.6-Flash-RL and Qwen3.8-Flash-Next); other MoE models are presumed to work.
2. **Automatic CPU/GPU prefill split**: for long prompts (threshold configurable) the weights are streamed into VRAM with DMA double buffering, **saturating the PCIe link** (verified on 2x Tesla A100 40GB: both PCIe 4.0 x16 lanes reach the **25 GB/s** ceiling), keeping GPU-side prefill performance as high as possible.
3. **Supports older GPUs such as SM80**, and **NVFP4 / MXFP4** quantized weights; where the hardware has no native support, the weights are **dequantized and computed on the best available path**.
4. Supports **AVX-512 VNNI and other ISAs**, picking the best CPU compute backend automatically.
5. **Hardware-aware NUMA slicing**: the process is spread evenly across cores according to the NUMA topology, and the MoE weights are sharded to match, so **every node's compute only touches local memory** (no cross-socket traffic) while **total memory stays at a single copy of the weights**.

## Measurement host

Every performance number below was taken on this machine:

| Item | Configuration |
|---|---|
| CPU | 2× AMD EPYC 9654 (192 physical cores / 384 threads, 8 NUMA nodes) |
| RAM | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| OS | Ubuntu 22.04 · conda env `lvllm` |

---

## Performance

* The **DeepSeek-V4.1-Flash row is the 2026-10-12 measurement on the current production config** (the authoritative numbers in this table); **GLM-5.3 and MiMo-V2.6 rows are legacy** v0.2.4-v0.2.6 numbers (random dataset, aggregate metric), pending re-measurement.
* Metrics (**changed in v0.2.8; the authoritative definition is [`docs/BENCH_PROTOCOL.md`](docs/BENCH_PROTOCOL.md)**): **prefill** = `sum(prefill tokens) / (last decode start)`, measured **cold** (always count from server `usage`); **decode** = the **`sum(output tokens) / sum(decode windows)`** measured under a **fixed output length** (`max_tokens=min_tokens` + `ignore_eos`, every request emits exactly N tokens) with **C requests kept in flight**, then **x C** for the aggregate. Fixing the output length removes the acceptance-length sampling noise (+-10%) that plagued the old metric (**B416/B418**); verifier: `scripts/bench_protocol_check.py`. **`deg` (8-gram copy rate) is a mandatory health column** for these numbers.
* Config details, criteria and the `MBT` trade-off: `docs/TUNING_GUIDE.md` section 8 and `docs/MODEL_GUIDES.md` section 0.1.

| Model (optimal config) | load | actual tokens | conc. | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.90 · **MAXLEN 262144**<br>MBT 8192 · **MTP k=1** · **GPU prefill on** (v0.2.8 config) | genuine DSH payload‡ | ≈8,900 | 1 | **901** | **23.1** |
| | genuine DSH payload‡ | ≈8,900 | 2 | **≈901**§ | **27.1** |
| | genuine DSH payload‡ | ≈8,900 | 4 | **≈901**§ | **32.2** |
| **DeepSeek-V4.1-Flash**<br>TP=2 · util 0.90 · **MAXLEN 524288**<br>**MBT 8192** · **seqs 4** · **DSpark k=5** · **PIN hot-expert residency** (v0.2.8 config) | genuine DSH payload‡ | ≈9,340 | 1 | **1,041** | **47.5** |
| | genuine DSH payload‡ | ≈9,340 | 2 | **1,045** | **81.5** |
| | genuine DSH payload‡ | ≈9,340 | 4 | **1,156** | **101.7** |
| **MiMo-V2.6-Flash-RL**<br>TP=2 · util 0.85 · **MAXLEN 131072**<br>MBT 8192 · **seqs 4** · **MTP k=1** · **GPU prefill on (threshold 4096)** | short (random) | 128 | 1 | **232.0** | **30.5** |
| | short (random) | 128 | 2 | **367.5** | **40.7** |
| | long (random) | 16,384 | 1 | **811.3** | **27.3** |
| | long (random) | 16,384 | 2 | **1455.3** | **39.6** |

‡ Load = **genuine DSH session payloads** (pre-assembled `<|im_start|>` template, i.e. the real HTTP shape; distinct, naturally long output). **The decode column is aggregate throughput** (per-request decode window x C). Method: greedy, **fixed output length 1024** (`max_tokens=min_tokens` + `ignore_eos`; a natural stop would make the output-token count a random variable and inject +-10% noise), **C requests kept in flight** (measured slot utilisation 0.99), 16 (DS41F) / 8 (GLM) requests per level. **`deg` (8-gram copy rate) is a mandatory health column**: forced long output drifts into copy loops which inflate tok/s (measured median `deg` 0.37-0.72) -- to judge whether the *engine* got faster, read the **step time** in the release notes. **Prefill**: measured cold (see `docs/BENCH_PROTOCOL.md`); § prefills run serially at C>=2, so aggregate prefill stays close to C=1. Gain over non-speculative **≈2.9x / 2.1x / 1.7x** (the old set was inflated by forced continuation and is retired). The old metric (per-request x C = 48.7/70.4/95.2; GLM short load 115.2/22.0) is archived in [`docs/BENCH_HISTORY.md`](docs/BENCH_HISTORY.md). Full data: [v0.2.8 release notes](RELEASE_NOTES_v0.2.8.md).

## Quick start

```bash
# 1) upstream vLLM (this project is a plugin, no fork needed; verified versions and commits: docs/INSTALL_MAINLINE.md)
pip install vllm

# 2) this plugin (distribution name vllm-xtu-moe, NOT on PyPI) -- pick one
# (a) install the wheel attached to the GitHub Release (all 7 ISA variants included):
pip install ./vllm_xtu_moe-0.2.7-cp312-cp312-manylinux_2_34_x86_64.whl
# (b) or install from source (needs a local compiler):
# CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .

# 3) serve a MoE model whose experts do not fit in VRAM
VLLM_EXPERTS_LOAD_DEVICE=cpu \
vllm serve <MODEL_DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 65536 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90
```

Per-model recipes, memory budgeting, self-checks and troubleshooting →
**[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** and **[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**.

---

## Changelog (brief)

| Version | Theme |
|---|---|
| **v0.2.8** (2026-10-15) | **Hot experts resident on the GPU: the GPU and the CPU decode together** — hot experts (from a real routing trace) are pinned in VRAM and skipped by the CPU kernels, so one step computes part of the experts on the GPU and the rest on the CPU **in parallel**: decode step time is **-1.9% / -4.4% / -14.1% at C=1/2/4** (all three CIs exclude zero; at C=4 both step time and decode throughput, **+20.5%**, are significant); switch `XIAOTU_PIN` (off by default, zero-cost rollback). Same release: the benchmark protocol moves to the **industry-standard fixed-length** metric ([BENCH_PROTOCOL](docs/BENCH_PROTOCOL.md)), and GLM enables MTP on a unified load — see [release notes](RELEASE_NOTES_v0.2.8.md) |
| **v0.2.7** (2026-10-09) | **Engine optimisation: CPU→GPU DMA buffers cut** — per-prefill weight residency goes from **2 × whole layer** to an **N × sub-layer block ring**: **7.56 → 1.89 GiB/rank** (**11 GB** saved in total at TP=2), no performance loss; switch `XIAOTU_GPF_STAGE_TILE_E` (`0` = old behaviour, instantly revertible) — see [release notes](RELEASE_NOTES_v0.2.7.md) |
| **v0.2.6** | **Bootstrapped development: INT8 activation x FP4 expert weights** — MXFP4 experts gain an **INT8 activation** path (the fp4 code set fits int8 exactly, so the **weight side is free**): kernel level **24.9 -> 15.1-15.7 ms/layer (1.54-1.62x)**; end to end (qfn, MXFP4-FP8) GSM8K **193/200 on both arms**, 194/200 per-question (net 0), 256K four-pin 4/4 + `stop` on both arms, Vision 23 agrees 18/23; **fallback-path regression fixed** (16-22% slower -> +1.49% instructions); **MTP on qfn (2.44x)**; qfn default threads **120 (5 cores/CCD)**; path off by default (`XIAOTU_MOE_INT8=1`) |
| **v0.2.5** | **GPU prefill memory rework** — the GPU operator consumes NUMA-sharded experts directly, dropping one layer of weight staging (**6.33 GiB saved for DeepSeek-V4.1-Flash, 3.16 GiB per card at TP=2**; per-layer staging 10.73 → **7.56 GiB/rank**; byte-identical results); **1M context + GPU prefill** (CED: KV 5437 → 2106 B/token, so 1M needs ~2.2 GiB); **Engram pinned over-allocation fixed (−75 GiB host)**; rebasing onto upstream restores **CED** (16k prefill 527.8 → **982.4** tok/s); **TP=2 uses GPU1+GPU2**; service logs named by PID |
| **v0.2.4** | **MiMo-V2.6-Flash-RL support + performance work** — MXFP4 experts over V4.1's engine path (TP=2 / 1M context / multimodal / MTP k=1 all measured); the **GPU prefill "zero means off" trap fixed** (long prefill 313 → **811** tok/s); **`--max-num-seqs` now defaults to 4**; decode metric switched to median ITL |
| **v0.2.3** | **Upstream tracking + GLM/MiMo MTP** — patch stack rebased onto upstream `133b71e0b` (11 patches / 40 files); **GLM-5.3-Flash MTP wired up, ON by default** (`SPEC_K=1`, decode 21.9 → 22.6 tok/s at the cost of a 27% smaller KV pool); **MiMo-V2.5 (310B/15B) runs end-to-end on one A100-40GB** with MTP k=1 at +10% decode; GLM memory contract re-calibrated (`GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **GLM-5.3-Flash support** — FP8 GPU prefill wired up (4K-prompt TTFT 29.3 → 22.8 s), delivered as 256K × 2 concurrent; fixes an e4m3 subnormal-decode defect and adds an all-codeword gate |
| **v0.2.1** | **Major CPU-engine performance work** — the CPU MoE engine now **beats `lk_moe` on every real shape**; DeepSeek-V4-Flash benefits too |
| v0.2 | DeepSeek-V4.1-Flash end-to-end (1M context + GPU prefill + speculative decoding) plus CPU-prefill path optimisation |
| v0.1.0 | First public release: hybrid mode (CPU experts + GPU rest), AVX2 / AVX-512 multi-ISA, DeepSeek-V4 family |

Details, performance comparisons and parameter changes:
[**v0.2.6**](RELEASE_NOTES_v0.2.6.md) · [**v0.2.5**](RELEASE_NOTES_v0.2.5.md) · [**v0.2.4**](RELEASE_NOTES_v0.2.4.md) · [**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
(archived: [v0.2pre](RELEASE_NOTES_v0.2pre.md))

---

## Documentation

**For users — [`docs/`](docs/)**

| Document | Contents |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | **Runbook**: install, launch flags, VRAM budgeting, GPU-prefill recipe, JIT cache, release process |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | Per-model resource needs, launch commands and performance |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Measured data and how to reproduce it |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | Known limitations and unsupported combinations |
| [`docs/INSTALL_MAINLINE.md`](docs/INSTALL_MAINLINE.md) | Preparing an upstream vLLM environment |
| [`docs/FP4_INT8_HARDWARE_OUTLOOK.md`](docs/FP4_INT8_HARDWARE_OUTLOOK.md) | **Public hardware outlook on 4-bit inference formats** (FP4 vs INT4, with sources) and what it implies for **CPU-first MoE serving** |
| [`docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md`](docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md) | **INT8 activation x MXFP4 expert weights** — what the path solves, kernel-level gains, the end-to-end semantic acceptance protocol, how to enable it, and known limitations |

**For developers** — architecture and upstream integration, the GPU-prefill implementation,
upstream drift, tuning logs and all internal reports are **internal development documents and are
not published with this repository** (they exist only in the local working copy).

---

## About the author

Independent researcher. Chief designer and operator of a **10-PFLOP/s (double-precision) supercomputer**, with a background in high-performance computing.

That is where this project's priorities come from: **NUMA topology and memory layout, bandwidth-bound kernels, and host-side data movement** - which is exactly where running giant MoE models on ordinary servers is decided today.

## Acknowledgements

This project was inspired by **KTransformers** and **Lvllm**. In particular the compute engine
**`xiaotu-moe`** borrows heavily from ideas in **lk-moe**; **all `xiaotu-moe` code is independently written**.
Our sincere thanks.

* KTransformers — <https://github.com/kvcache-ai/ktransformers>
* Lvllm (and its CPU MoE engine **lk-moe**; the `lk_moe` 2.4.2 build used as our baseline comes from this repo) — <https://github.com/guqiong96/Lvllmds4-x>
* Thanks also to **vLLM** (<https://github.com/vllm-project/vllm>) for the plugin extension points
  that let this project integrate without a fork.

---

## License

Apache-2.0. Third-party components and notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`NOTICE`](NOTICE).
