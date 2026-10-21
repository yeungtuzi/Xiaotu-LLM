# MUSA / S4000 正确性验证计划（2026-10-21）

> **目标平台**：Moore Threads MTT S4000 + 海光 x86 + 256GB host  
> **目标**：先验证 **官方 Qwen3.8-Flash-Next FP8 混合推理在 MUSA 上的正确性**；NVFP4 后续再挂。不追求性能。  
> **分支**：`feature/musa-correctness`（从 `main` 切出；验证通过后再合并）。  
> **状态**：本机无 MUSA 硬件；先完成离线调研、版本锁定、移植映射和 preflight，设备上线后按 Phase 1 开工。

---

## 0.1 权重口径定稿（2026-10-21）

**第一目标：官方 `Qwen/Qwen3.8-Flash-Next-FP8`。**

原因很直接：

| 权重 | 大小 | 256GB host 是否可行 |
|---|---:|---|
| 官方 BF16 | **360.02 GB** | ❌ 放不下 |
| 官方 FP8 | **185.56 GB** | ✅ 可以 |
| 社区 NVFP4 | 128.95 GB | ✅，但属于后续目标 |

所以：

- **BF16 不作为第一目标**（360GB > 256GB；除非做 NVMe/mmap 流式）；
- 先跑 **官方 FP8**，它也是官方发布、qfn 架构、CPU 专家路径可以复用；
- **社区 NVFP4 放到 Phase 4/P2**，不阻塞官方 FP8 的正确性验证；
- 仍然不依赖厂商专用镜像作为运行路径；镜像只作参考。

---

## 1. 范围

### 做

- **官方 Qwen3.8-Flash-Next FP8** 在 MUSA 上的**加载/前向正确性**；
- 后续：qfn NVFP4（`W4A16_NVFP4` + `FP8_PB_WO`）的加载/反量化/前向；
- CPU 专家 + GPU attention/KV 的**混合模式**；
- 单卡优先，随后多 S4000；
- 8K/32K 上下文，batch 1，MTP 关；
- 与 CPU golden / A100 / 参考实现逐 token/logits 对拍。

### 不做

- 不追求 tok/s、不比较 S4000 vs A100；
- 不验证 CUDA Graph / Marlin / CUTLASS / NCCL 的 CUDA 行为；
- 不做全驻留 3 卡 A100 的替代；
- 不把 S4000 当生产部署卡；
- 不直接改主线；所有提交留在 `feature/musa-correctness`。

---

## 2. 目标栈（先锁定，再动手）

| 组件 | 目标 | 说明 |
|---|---|---|
| GPU | MTT S4000 | 48GB/卡；S5000 是 vllm-musa 当前主要验证卡，S4000 需自己验 |
| CPU | 海光 x86 | AVX2 起步；AVX512 需现场确认 |
| host | 256GB | qfn experts 67.99GB + PLE 51.2GB 可放 host |
| MUSA | Toolkit 5.2.0 系列 | 以 vllm-musa `requirements/musa_private.txt` 为准 |
| PyTorch | 2.11.0.post2+musa5.2.0 | torch_musa 同版本 |
| vLLM | **0.28.0**（先用） | vllm-musa 当前 pin：`2cf0a691...`；我们主线目标 0.29.0，先别混 |
| vllm-musa | `v0.28.0-dev` @ `d85ef82d` | 包版本 0.1.28；2026-10-19；另有 v0.30/v0.32-dev 分支，先不混 |
| torchada | 0.1.95 | CUDA→MUSA 兼容层 |
| MATE | 0.2.7 | MUSA attention/tensor kernels |
| Triton | 3.2.0 | MUSA 后端 |
| flash_mla | 0.2.7+musa | 官方 MUSA 包 |
| deep_ep | 1.1.0+musa5.2.0 | 多卡 EP 时再启用 |
| tilelang_musa | 0.1.12+musa.2 | 可选 |

**qfn 专用镜像的定位（2026-10-21 用户澄清）**

vllm-musa README 提到：
`registry.mthreads.com/mcconline/inference/vllm/vllm-openai:qwen38-flash-next`

但**它不是我们的运行路径，也不是交付依赖**：

- 只作 **oracle / 补丁参考**，用来确认 qwen4_exp/attention 的期望行为；
- 正式目标：**我们的 vLLM-MUSA 分支直接加载权重**，不要求用户安装厂商专用镜像；
- 如果该镜像只支持官方 BF16/FP8 qfn，不支持我们的 NVFP4 社区量化，则它不能作为 NVFP4 的 oracle。

**权重口径必须先确认**

- “官方权重”通常指 `Qwen/Qwen3.8-Flash-Next` 的 BF16/FP8；
- `lovedheart/...NVFP4-W4A16...` 是**社区量化**，不是 Qwen 官方发布；
- 两者不是同一个 checkpoint，也不是同一套量化 kernel；
- 在动手前必须确认 MUSA 验证目标是：
  - A. 官方 BF16/FP8 qfn；
  - B. 社区 NVFP4 qfn；
  - C. 两者都要（先官方、后 NVFP4）。

---

## 3. 分支与合并纪律

```bash
git switch main
git switch -c feature/musa-correctness
# 只提交 MUSA 相关文件；不把工作区其它未提交改动混进来
git push -u origin feature/musa-correctness
```

- 分支只包含 MUSA 计划/脚本/移植代码/测试；
- 不直接改 `main`；
- 合并门槛见 §6；
- 与 qfn 3×A100、ds41f GPU0 两条主线隔离。

---

## 4. 阶段

### Phase 0：离线准备（当前）

- [x] 建分支 `feature/musa-correctness`；
- [x] 锁定 vllm-musa / torch_musa / vLLM / MUSA 版本；
- [x] 建端口映射（见 [MUSA_PORT_MAP.md](MUSA_PORT_MAP.md)）；
- [x] preflight 脚本（无硬件也能跑）；
- [ ] 确认“官方权重”具体 checkpoint（BF16/FP8 vs 社区 NVFP4）；
- [ ] 从镜像/分支提取 `qwen4_exp` 补丁作为**参考**，不把镜像作为运行依赖；
- [ ] 准备官方 FP8 qfn 的 CPU golden / 单层 shape 测试；
- [ ] 准备 `musaInfo`/`vllm_collect_env` 采集脚本。

### Phase 1：设备上线，栈冒烟（0.5–1 天）

1. 装 MUSA Toolkit + 驱动 + torch_musa；
2. `torch_musa` matmul / attention 小算子；
3. vllm-musa 起一个小 dense 模型；
4. 采集版本、拓扑、显存、MUSA 可用算子；
5. **不做 qfn**，先证明栈可用。

### Phase 2：官方 FP8 qfn 单层/短上下文（1–3 天）

1. 官方 FP8 权重加载；
2. 单层 dequant + MoE，与 CPU golden 对拍；
3. qfn 8K、batch 1、MTP 关、文本 only；
4. GPU 0 层专家，CPU 全部专家；
5. 检查 NaN、路由、partial sum、PLE host。

### Phase 3：端到端正确性（2–4 天）

- 官方 FP8 qfn 端到端；
- 固定 prompt 集，比较 token/logits；
- 工具调用/多轮；
- 32K 上下文；
- 多 S4000 只做 TP 通信冒烟，不追性能。

### Phase 4：NVFP4（P2）与收口

- 官方 FP8 通过后，再挂社区 NVFP4；
- 记录 MUSA 与 CUDA 的数值差异；
- 记录未支持算子/回退；
- 通过后合并到 main；
- 不通过则保留分支和报告，不阻塞主线。

---

## 5. 最小垂直切片

```text
qfn 单层
  → NVFP4 W4A16 dequant
  → grouped MoE
  → CPU 专家 partial sum
  → 与 CPU/A100 golden 对拍
```

先打通这一层，再谈整模型。

---

## 6. 合并门槛

- [ ] `musa_preflight.sh` 全绿；
- [ ] qfn NVFP4 单层数值通过；
- [ ] 8K 端到端无 NaN/OOM；
- [ ] 固定 prompt 的 token/logits 与参考一致或在容差内；
- [ ] 不修改主线默认路径；
- [ ] 报告写清 MUSA 不支持/回退的算子；
- [ ] 用户确认后再合并。

---

## 7. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| S4000 非 vllm-musa 主验证卡 | 可能有未覆盖算子 | 先小模型/单层；记录 `musaInfo` |
| vllm-musa pin vLLM 0.28 vs 我们 0.29 | 补丁不匹配 | 先用 0.28 验证；再评估 0.29 port |
| qwen4_exp 只在专用镜像 | 源码分支不完整 | 先拉镜像/分支；提取补丁 |
| NVFP4 无原生 MUSA 支持 | 需 dequant | 先 CPU golden；再 MUSA dequant |
| 多卡 MCCL 不熟 | TP 失败 | 单卡优先，多卡后置 |
| Hygon AVX512 不确定 | CPU 引擎变慢/不支持 | preflight 记录 CPU flags；AVX2 回退 |
