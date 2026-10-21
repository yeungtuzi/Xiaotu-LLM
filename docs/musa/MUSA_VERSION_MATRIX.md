# MUSA 版本/仓库锁定（2026-10-21）

> 只锁**正确性验证**需要的版本；设备上线后以 `musaInfo` / `vllm_collect_env` 实测为准。

## 0. 权重目标

| 权重 | 大小 | 优先级 |
|---|---:|---|
| `Qwen/Qwen3.8-Flash-Next-FP8` | 185.56 GB | **P0（先跑）** |
| `Qwen/Qwen3.8-Flash-Next` BF16 | 360.02 GB | P2（256GB host 放不下，需 NVMe/mmap） |
| `lovedheart/...NVFP4...` | 128.95 GB | P2（后续） |

---

## 1. vllm-musa 目标

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/MooreThreads/vllm-musa |
| 分支 | `v0.28.0-dev` |
| commit | `d85ef82d15f5f96132611a0755945c2f8ed9132d`（2026-10-19） |
| 包版本 | `vllm-musa 0.1.28` |
| 上游 vLLM | `v0.28.0` @ `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| 引擎 | vLLM V1；支持时 runner V2 |

**其他相关分支（不要混用）**

| 分支 | HEAD | 上游 vLLM | 备注 |
|---|---|---|---|
| `v0.30.0-dev` | `47184da2` | v0.30.0 | 中间版本 |
| `v0.32.0-dev` | `93183864` | v0.32.0 @ `5df63a72` | patch 里已出现 `qwen4_exp_qsa_triton_warmup`；更接近主线但和我们的 0.29 不同 |

先固定 `v0.28.0-dev` 做正确性，因为它有 qfn 专用镜像线索；`v0.32.0-dev` 作为后续 port 目标。

## 2. 运行时 pin（来自 requirements）

```text
torch==2.11.0.post2+musa5.2.0
torch_musa==2.11.0.post2+musa5.2.0
torchvision==0.26.0.post2+musa5.2.0
torchaudio==2.11.0.post2+musa5.2.0
mate==0.2.7
mate-mubin==0.2.7
flash_attn_3==0.2.7+musa
flash_mla==0.2.7+musa
deep-gemm==0.2.7+musa
flashinfer-python==0.2.7+musa
sageattention==0.2.7+musa
deep_ep==1.1.0+musa5.2.0torch2.11.0.post2
tilelang_musa==0.1.12+musa.2
triton==3.2.0
torchada==0.1.95
```

## 3. 相关仓库（HEAD 记录）

| 仓库 | HEAD/分支 | 用途 |
|---|---|---|
| [torch_musa](https://github.com/MooreThreads/torch_musa) | `2838e77d` main | PyTorch MUSA |
| [torchada](https://github.com/MooreThreads/torchada) | `50501cfd` main | CUDA 兼容层 |
| [MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA) | `c564db4a` master | MLA |
| [mutlass](https://github.com/MooreThreads/mutlass) | `78b34951` main | MUSA CUTLASS |
| [Model-Optimizer-musa](https://github.com/MooreThreads/Model-Optimizer-musa) | `90ba9fbc` main | 量化 |
| [MT-DeepEP](https://github.com/MooreThreads/MT-DeepEP) | `ca030a01` HEAD / `eb56e64e` main | EP 通信 |
| [mccl](https://github.com/MooreThreads/mccl) | `23edec5a` HEAD | 集合通信 |
| [mtshmem](https://github.com/MooreThreads/mtshmem) | `c1ea1f79` HEAD | 共享内存 |
| [tilelang_musa](https://github.com/MooreThreads/tilelang_musa) | `4b59fea1` main | TileLang |
| [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py) | `b594a7db` main | 监控 |
| [mate](https://github.com/MooreThreads/mate) | `3c5b5929` main | MATE kernels |

## 4. 与主线差距

| 项 | 我们 | MUSA 参考 | 处理 |
|---|---|---|---|
| vLLM | 0.29.0 目标 | 0.28.0 pin | 先用 0.28 跑正确性 |
| qfn 模型 | 我们 qfn 插件 | qfn 专用镜像 | 先取镜像/补丁 |
| 量化 | NVFP4 W4A16 + FP8_PB_WO | MUSA quant patch | 单层对拍 |
| CPU 引擎 | x86 `.so` | 同一 x86 | AVX2 优先 |
| GPU prefill | Triton tile ring | 不需要 | 关闭 |
| Graph | 可选 | 先 eager | 关闭 |
