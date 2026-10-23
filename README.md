# vllm-xtu-moe

[![Version](https://img.shields.io/github/v/tag/yeungtuzi/vllm-xtu-moe?label=version&color=blue)](https://github.com/yeungtuzi/vllm-xtu-moe/releases)
[![License](https://img.shields.io/github/license/yeungtuzi/vllm-xtu-moe?color=lightgrey)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-plugin-blueviolet)](https://github.com/vllm-project/vllm)
[![CPU](https://img.shields.io/badge/CPU-AVX512--VNNI%20%C2%B7%20BF16%20%C2%B7%20MXFP4%20%C2%B7%20FP8-orange)](#项目特色)
[![Platform](https://img.shields.io/badge/Platform-2%C3%97%20EPYC%209654%20%2B%203%C3%97%20A100-76b900)](#实测平台)
[![Models](https://img.shields.io/badge/models-DeepSeek--V4.1--Flash%20%C2%B7%20GLM--5.3--Flash%20%C2%B7%20Qwen3.8-green)](docs/MODEL_GUIDES.md)

> **v0.2.9**:针对 **Qwen3.8-Flash-Next** 优化 —— 新增**专家层常驻**(12 层精选,decode 46.3/92.9/127.3 tok/s @C=1/2/4),并修复 CUDA Graph 捕获崩溃。详见 [发行说明](RELEASE_NOTES_v0.2.9.md)。

> **XTU = X Transformers Unity**(读音:汉语「小兔」)—— 一个**高性能 MoE 推理加速层**。
>
> 这是一个面向**超大 MoE 模型**的 **vLLM 插件**。核心思路是**数据分层放置**,
> 再由一个统一调度器在层间换入、预取和重叠:
>
> * **GPU 常驻**:attention/KV、dense、drafter 等**每步都要用**的部分;
> * **GPU 动态换入**:热专家池/热层,按路由与预算**按需驻留**;
> * **CPU 内存**:全量专家权重,以及 Engram/PLE 等大表;
> * **NVMe(可选)**:Engram/PLE 等冷数据的设计位,默认关闭。
>
> **prefill 和 decode 两个阶段,GPU 与 CPU 都共同参与计算**:GPU 承担 attention/KV/dense
> 与驻留专家,CPU 承担冷专家;调度器按路由与预算切分专家,并让**传输与两侧计算重叠**。
>
> 结果:即使**显存只够放权重的一个零头**,仍然能对超大 MoE 做**高效计算**——
> 不是"把专家永久丢给 CPU",而是让**数据在正确的层级、以正确的时机**参与计算。
> 而且**尽量不改主线 vLLM**。

[**English**](README_EN.md) · 中文(默认)

> **📌 当前版本：v0.2.9**（2026-10-19）—— 针对 Qwen3.8-Flash-Next 的专家层常驻；修复 CUDA Graph 捕获崩溃
> [发行说明](RELEASE_NOTES_v0.2.9.md) · [性能口径](docs/BENCH_PROTOCOL.md)

---

## 项目特色

1. **支持在单张显卡（>40 GB）且内存充足的设备上运行超大 MoE 模型**（已验证 DeepSeek-V4.1-Flash、GLM-5.3-Flash、MiMo-V2.6-Flash-RL 与 Qwen3.8-Flash-Next），其他 MoE 模型推定支持。
2. **自动 CPU/GPU Prefill 分流**：对长 prompt（阈值可设）使用 DMA 双缓冲把权重流式传入显存，**吃满 PCIe 带宽**（已在 2× Tesla A100 40 GB 上验证：两组 PCIe 4.0 x16 均达到 **25 GB/s** 的上限），最大限度保证 GPU 侧的 Prefill 性能。
3. **支持 SM80 等较老的显卡**，支持 **NVFP4 / MXFP4** 量化权重；硬件不支持时，权重会**反量化后按最优路径计算**。
4. 支持 **AVX-512 VNNI 等多种 ISA**，自动选择最优的 CPU 计算后端。
5. **硬件感知的 NUMA 切片**：按 NUMA 拓扑把进程均匀分布到核心，并把 MoE 权重做对应分片，**每个 node 的计算只访问本地内存**，避免跨 socket 访存；同时**内存总占用仍为单份权重**。
6. **支持 GPU 常驻热专家，并按策略执行动态换入**，实现 **GPU + CPU 协同解码**，显著提升 **DeepSeek-V4.1-Flash** 的解码性能（自 **v0.2.8** 起）；**Qwen3.8-Flash-Next 自 v0.2.9 起支持「专家层常驻」**（12 层精选，见 [v0.2.9 发行说明](RELEASE_NOTES_v0.2.9.md)）；**暂不支持 GLM-5.3-Flash**（显存限制）。

## 实测平台

下文所有性能数字都在这台机器上取得:

| 项 | 配置 |
|---|---|
| CPU | 2× AMD EPYC 9654(192 物理核 / 384 线程,8 NUMA node) |
| 内存 | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| 系统 | Ubuntu 22.04 · conda env `lvllm` |

---

## 性能

* 本表 **DeepSeek-V4.1-Flash(1017)行为 2026-10-19 实测**(本表唯一权威数据;按 `docs/BENCH_PROTOCOL.md` §6:真实 DSH 15K v2、**自然结束**、decode 用 token 计数窗口、每并发档 16 请求、充分预热);**GLM-5.3-Flash 行为 2026-10-17 实测**(按 §2/§3,15K 载荷、每档 16 请求、`--outlen 1024`、充分预热);**MiMo-V2.6 行为旧数据**(v0.2.4~v0.2.6,random 数据集、聚合口径,待复测)。
* 口径(**v0.2.8 起变更,唯一权威定义见 [`docs/BENCH_PROTOCOL.md`](docs/BENCH_PROTOCOL.md)**):**prefill** = `Σ预填 token ÷ 最后一路解码开始时刻`,从**冷缓存**测(计数一律取服务端 `usage`);**decode** = ⭐ **固定输出长度**(`max_tokens=min_tokens` + `ignore_eos`,每请求恰好 N 个 token)、**满 C 连续灌入**下的 **`Σ输出token ÷ Σ(解码段)`**,再 **× C** 得聚合值 —— 固定输出长度使"分子"不再是随机变量,消除了旧口径里"接受长度抽样"带来的 ±10% 级抖动(**B416/B418**);校验脚本 `scripts/bench_protocol_check.py`。⭐ **`deg`(复制循环体检)是这些数字的必要标注**。⚠️ **1017 行是例外**:按 `BENCH_PROTOCOL` §6 用**自然结束 + token 计数窗口**,不是固定 1024;跨行不可直接比较。
* 配置细节、判据与 `MBT` 取舍见 `docs/TUNING_GUIDE.md` §8、`docs/MODEL_GUIDES.md` §0.1。

| 模型(最优配置) | 负载 | 实际 token | 并发 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.90 · **MAXLEN 262144**<br>MBT 8192 · **MTP k=4** · **GPU 预填开**(v0.2.8 配置) | 真实 DSH 载荷‡ | ≈14,500 | 1 | **1,477** | **29.8** |
| | 真实 DSH 载荷‡ | ≈14,500 | 2 | **1,441** | **31.6** |
| | 真实 DSH 载荷‡ | ≈14,500 | 4 | **1,394** | **37.2** |
| **DeepSeek-V4.1-Flash (1017)**<br>TP=2 · util 0.70 · **MAXLEN 524288**<br>**MBT 8192** · **seqs 4** · **DSpark k=5** · **PIN off** | 真实 DSH 15K v2† | ≈15,000 | 1 | **1,190** | **68.6**‡‡ |
| | 真实 DSH 15K v2† | ≈15,000 | 2 | **1,197** | **70.8**‡‡ |
| | 真实 DSH 15K v2† | ≈15,000 | 4 | **1,206** | **79.6**‡‡ |
| **DeepSeek-V4.1-Flash (1017) + GPU0 侧车**<br>同上 + `GPUS=1,2,0` · `XTU_SIDECAR=gpu0` · **1900 槽** | 真实 DSH 15K v2† | ≈15,000 | 4 | **1,206** | **106.8**§ |
| **MiMo-V2.6-Flash-RL**<br>TP=2 · util 0.85 · **MAXLEN 131072**<br>MBT 8192 · **seqs 4** · **MTP k=1** · **GPU 预填开(门槛 4096)** | 短(随机) | 128 | 1 | **232.0** | **30.5** |
| | 短(随机) | 128 | 2 | **367.5** | **40.7** |
| | 长(随机) | 16,384 | 1 | **811.3** | **27.3** |
| | 长(随机) | 16,384 | 2 | **1455.3** | **39.6** |
| **Qwen3.8-Flash-Next(qfn)**<br>TP=2 · util 0.90 · **MAXLEN 262144**<br>**MBT 8192** · **MTP 关** · **12 层精选常驻** · `FULL_DECODE_ONLY` | 真实 DSH 15K v2† | ≈15,000 | 1 | **1,517** | **46.3** |
| | 真实 DSH 15K v2† | ≈15,000 | 2 | **1,538** | **92.9** |
| | 真实 DSH 15K v2† | ≈15,000 | 4 | **1,529** | **127.3** |

‡ 负载 = **真实 DSH 会话载荷**(预装配 `<|im_start|>` 模板,即真实 HTTP 形状;互异、天然长输出)。**decode 列 = 聚合吞吐**(每请求解码段 × C)。口径:greedy、**固定输出长度 1024**(`max_tokens=min_tokens` + `ignore_eos`,⛔ 自然结束会让"输出 token 数"变成随机变量而引入 ±10% 级抖动)、**满 C 连续灌入**(槽位利用率实测 0.99)、每档 16(DS41F)/ 8(GLM)条请求。⚠️ **`deg`(8-gram 复制率)是必要标注**:强制长输出会落进复制循环而抬高 tok/s(实测 `deg` 中位 0.37~0.72)⇒ 判"引擎是否更快"请看发行说明里的**步时**。**prefill**:冷测(§ 见 `docs/BENCH_PROTOCOL.md`);§ C≥2 时预填串行 ⇒ 聚合 prefill 与 C=1 接近。**相对无投机的收益 ≈2.9× / 2.1× / 1.7×**(旧 "2.4×/2.4×/1.8×" 受强制续写污染,已作废)。旧口径(每请求解码段 × C 的 48.7/70.4/95.2、GLM 短负载 115.2/22.0)已归档到 [`docs/BENCH_HISTORY.md`](docs/BENCH_HISTORY.md)。全部数据与口径演变见 [v0.2.8 发行说明](RELEASE_NOTES_v0.2.8.md)。⚠️ **1017 行的 decode 为自然结束的 token 计数窗口,与 GLM/MiMo/qfn 的固定 1024 口径不同,跨行不可直接比较**;1017 在 524K + 700 槽 PIN 下仍会触发 `aten::new_empty` 谷底 OOM(见 `dev-docs/BUG_REGISTRY.md` A68),故本行用 **PIN off + util 0.70** 的稳定配置。



§ 侧车行（2026-10-24，DSpark 开）：C=4 三次 **106.8 / 108.5 / 81.1** tok/s —— **tok/s 随投机接受率浮动**（83%→65%→40%）；**步时收益三次稳定 −34%**（115.6ms vs 197.4ms）。明细见 [`docs/SIDECAR_GPU0_POSTMORTEM.md`](docs/SIDECAR_GPU0_POSTMORTEM.md)。

‡‡ 2026-10-24 按同一官方口径复测（`dsh_15k_v2` + `bench_decode_window.py --tokenizer`、每档 2×C 预热）：decode **68.6 / 70.8 / 79.6** tok/s；同档 fixed-16 为 43.7 / 62.9 / 73.8。明细见 [`docs/SIDECAR_GPU0_POSTMORTEM.md`](docs/SIDECAR_GPU0_POSTMORTEM.md)。

† **qfn 行口径(2026-10-19 用户统一口径,`docs/BENCH_PROTOCOL.md` §6/§6.1)**:15K prompt、`MBT=8192`、TP=2、GPU prefill on;
**prefill** = 15K 冷预填(16 条/档);**decode = 稳态解码窗口**(C 条 15K 请求**全部产出首 token 之后**才计时,排除预填竞争;步时 21.6 / 21.5 / 31.4 ms)。
自然结束口径,**不要与上表固定长度行直接比较**;⛔ 不得用"连续灌入"下混入预填等待的聚合数回答扩展性(§6.1 有反例)。

## 快速开始

```bash
# 1) 主线 vLLM(本插件是插件,不需要 fork;已验证的版本与提交见 docs/INSTALL_MAINLINE.md)
pip install vllm

# 2) 本插件(发行名 vllm-xtu-moe,**不发 PyPI**;二选一)
# (a) 从 GitHub Release 附件装:wheel 里已含 7 个 ISA 变体,无需本地编译器
pip install ./vllm_xtu_moe-0.2.7-cp312-cp312-manylinux_2_34_x86_64.whl
# (b) 源码安装(需要本地编译器):
# CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .

# 3) 跑一个专家权重放不进显存的 MoE 模型
VLLM_EXPERTS_LOAD_DEVICE=cpu \
vllm serve <MODEL_DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 65536 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90
```

逐模型配方、显存核算、自检与排错 → **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**、
**[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**。

---

## 版本变更(简)

| 版本 | 主题 |
|---|---|
| **v0.2.9**（2026-10-19） | **针对 Qwen3.8-Flash-Next 的专家层常驻** —— 12 层精选整层常驻（decode 46.3/92.9/127.3 tok/s @C=1/2/4）；新增 decode 专用 MoE 内核与零权重 pair 跳过；修复 CUDA Graph 捕获崩溃 —— 详见[发行说明](RELEASE_NOTES_v0.2.9.md) |
| **v0.2.8**（2026-10-15） | **引入 GPU 常驻热专家:GPU 与 CPU 共同解码** —— 按真实路由轨迹把热专家常驻显存、CPU 内核跳过它们,同一步里 GPU 与 CPU **并行**解码:**解码步时 C=1/C=2/C=4 分别 −1.9% / −4.4% / −14.1%**(三级 CI 排除 0,C=4 与"解码吞吐 +20.5%"双重显著);开关 `XIAOTU_PIN`(默认关,回退零成本);**同版**:性能口径改为**行业标准固定长度口径**([BENCH_PROTOCOL](docs/BENCH_PROTOCOL.md))+ GLM 开启 MTP 并统一负载 —— 详见[发行说明](RELEASE_NOTES_v0.2.8.md) |
| **v0.2.7**（2026-10-09） | **优化引擎：削减 CPU→GPU DMA 缓冲区** —— 预填期的权重驻留量由 **2 × 整层** 改为 **N × 子层块环**：**7.56 → 1.89 GiB/rank**（TP=2 共省 **11 GB**），性能无损失；开关 `XIAOTU_GPF_STAGE_TILE_E`（`0` = 旧行为，可即时回滚）—— 详见[发行说明](RELEASE_NOTES_v0.2.7.md)（**2026-10-10 勘误**：性能表 DSpark 组 decode 列已更正，见发行说明） |
| **v0.2.6** | **自举开发：INT8 激活 × FP4 专家权重 的计算路径** —— 为 **MXFP4** 专家权重增加 **INT8 激活**路径（fp4 码值全含于 int8 ⇒ **权重侧零代价**）：内核级 **24.9 → 15.1–15.7 ms/层（1.54–1.62×）**；端到端（qfn·MXFP4-FP8）GSM8K **两臂 193/200**、逐题一致 194/200（净 0）、256K 四针两臂 4/4 + `stop`、Vision 23 逐项一致 18/23；修复**回退路径回归**（慢 16–22% ⇒ +1.49% 指令）；打通 **qfn 的 MTP**（**2.44×**）；qfn 默认线程数 **120（5 核/CCD）**；路径默认关（`XIAOTU_MOE_INT8=1`）|
| **v0.2.5** | **优化 GPU prefill 内存机制** —— **为 GPU 算子增加对 NUMA 切片权重的处理路径,节省了一层权重空间**(对 DeepSeek-V4.1-Flash **节约 6.33 GiB**;2 个 GPU 时**每卡节约 3.16 GiB**):单层 staging 10.73→**7.56 GiB/rank**、离线同层 364.91→**341.31 ms**、数值逐字节相同;**支持 1M 上下文 + GPU 预填充**(CED 让每 token KV 5437→2106 B,1M 只需 KV ~2.2 GiB);**Engram pinned 浪费修复**(宿主内存 **−75 GiB**);rebase 到上游后 **CED 恢复生效**(16k prefill 527.8→**982.4** tok/s);**TP=2 一律用 GPU1+GPU2**(GPU0 只有 x8);服务日志按 **PID** 命名 |
| **v0.2.4** | **支持 MiMo-V2.6-Flash-RL + 性能优化** —— 新模型接入（MXFP4/TP=2/1M/多模态/MTP k=1，**层内数值门禁 94/94 层通过**、312K 针测试命中、多模态真图通过）；**GPU 预填充「0=关闭」陷阱修复**（长 prefill 313→**811** tok/s）；**`--max-num-seqs` 默认统一为 4**（短 C=2 prefill 236→**367**）；**decode 口径改为 median ITL**（真实争用 1.13–1.77×）；修好 MXFP4 上的层内数值门禁 |
| **v0.2.3** | **跟进上游 + GLM/MiMo 的 MTP** —— 补丁栈 rebase 到上游 `133b71e0b`(11 补丁/40 文件);**GLM-5.3-Flash 的 MTP 落地**(draft 层识别 + GPU 常驻,`SPEC_K=1..4`);**MiMo-V2.5(310B/15B)单卡端到端支持**,MTP k=1 decode +10%;显存契约重标定(GLM `GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **支持 GLM-5.3-Flash** —— FP8 GPU 预填充接线(4K prompt TTFT 29.3 → 22.8 s)、256K × 2 路并发交付配置;修掉 e4m3 次正规数解码缺陷 + 新增全码字门禁 |
| **v0.2.1** | **针对引擎的显著性能优化** —— CPU MoE 引擎在全部真实形状上**反超 `lk_moe`**;DeepSeek-V4-Flash 同步受益 |
| v0.2 | DeepSeek-V4.1-Flash 全链路可用(1M 上下文 + GPU 预填充 + 投机解码)+ CPU 预填充路径优化 |
| v0.1.0 | 首个公开版:混合模式(CPU 专家 + GPU 其余)、AVX2 / AVX-512 多 ISA、DeepSeek-V4 系列 |

改动清单、性能对照与运行参数变更:
[**v0.2.6**](RELEASE_NOTES_v0.2.6.md) · [**v0.2.5**](RELEASE_NOTES_v0.2.5.md) · [**v0.2.4**](RELEASE_NOTES_v0.2.4.md) · [**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
(归档:[v0.2pre](RELEASE_NOTES_v0.2pre.md))

---

## 文档

**面向使用者 —— [`docs/`](docs/)**

| 文档 | 内容 |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | **运行手册**:安装、启动参数、显存核算、GPU 预填充配方、JIT 缓存、发布流程 |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | 逐模型的资源需求、启动命令与性能 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 实测数据与复现方式 |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | 已知限制与不支持的组合 |
| [`docs/INSTALL_MAINLINE.md`](docs/INSTALL_MAINLINE.md) | 主线 vLLM 环境准备 |
| [`docs/FP4_INT8_HARDWARE_OUTLOOK.md`](docs/FP4_INT8_HARDWARE_OUTLOOK.md) | **4-bit 推理格式的公开硬件路线图**(FP4 vs INT4;含出处与"已宣告/传闻"标注)与 **CPU 优先 MoE 服务的设计依据** |
| [`docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md`](docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md) | **INT8 激活 × MXFP4 专家权重** —— 这条计算路径解决什么问题、内核级收益、端到端语义验收口径、启用方法与已知限制 |
| [`docs/AUXILIARY_MODEL_EVALUATION.md`](docs/AUXILIARY_MODEL_EVALUATION.md) | **辅助压缩模型评测(负结果)** —— 为什么不采用小模型做长会话压缩;含 0.6B/1.7B/4B 的真实速度、精度、崩坏率与每 token KV 开销的客观数据 |

**面向开发者** —— 架构与主线集成、GPU 预填充实现、上游漂移、调优记录与全部内部报告,
**属于内部开发文档,不随本仓库发布**(只保留在本地工作副本里)。

---

## 关于作者

独立研究者。曾担任一台 **10 PFLOP/s(双精度)超级计算机**的主要设计师和运营者,此前长期从事高性能计算方向的工作。

这正是本项目关注点的来源:**NUMA 拓扑与内存布局、带宽受限内核、主机侧数据搬运** —— 也正是"把巨型 MoE 模型放到常规服务器上跑"这件事今天真正的战场。

## 致谢

本项目受到 **KTransformers** 和 **Lvllm** 项目启发,尤其是计算引擎 **`xiaotu-moe`** 深度借鉴了 **lk-moe** 思路,
**`xiaotu-moe` 全部代码均为独立编写**,特此致谢。

* KTransformers —— <https://github.com/kvcache-ai/ktransformers>
* Lvllm(及其 CPU MoE 引擎 **lk-moe**;本项目对照的 `lk_moe` 2.4.2 即来自此仓库)—— <https://github.com/guqiong96/Lvllmds4-x>
* 同时感谢 **vLLM**(<https://github.com/vllm-project/vllm>)提供的插件式扩展点,使本项目无需 fork 即可接入。

---

## 许可

Apache-2.0。第三方组件与致谢清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`NOTICE`](NOTICE)。