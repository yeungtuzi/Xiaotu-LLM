# S5000 / MUSA 能力缺口与错误实现台账

用途：S4000 时期为大量缺失能力编写了 torch fallback。S5000 上硬件更强，但**软件栈并不自动等价于 CUDA**。
本台账只记录“S5000 实际探测到”的缺口，避免把三类完全不同的情况混在一起：

| 级别 | 含义 | 处理原则 |
|---|---|---|
| **M** | CUDA/上游有，S5000 当前栈**没有** | 允许有明确边界的 fallback，必须记录探测方式 |
| **U** | 符号**声明/注册了**，但 MUSA 后端**没有实现** | 不能 `hasattr` 就认为可用；必须试跑并捕获 `NotImplementedError` |
| **W** | **声明实现了，但结果错误/静默错**（最恶劣） | 禁止使用；必须逐元素对拍，记录失败尺寸 |
| **B** | 本项目构建/配置缺口，不是厂商 API 缺口 | 修构建，不写进厂商问题清单 |

> 规则：新增任何 torch/手搓 fallback 前，先归入 M/U/W/B 并在本文件登记；修复后把状态改为 `fixed` 并写回归证据。

## 当前环境

- 机器：MTT S5000
- 驱动：`5.2.0-server`
- Triton：`3.2.0`（MUSA 容器内）
- vLLM-MUSA：`0.32`
- 日期：2026-11-03

## 查证顺序（怀疑能力缺失/实现错误时）

1. **先跑本机探针**：确认是符号不存在、注册但无后端，还是“返回成功但结果错”。
2. **查厂商 SDK release notes**：例如 MUSA SDK 5.2.0 文档写明 Triton-MUSA 支持 3.2.0 与 3.6.0。
3. **查厂商 Docker 镜像**：⚠️ 最新能力有时**只在镜像里**，不在独立发布包；必须记录镜像 tag / digest / 实际容器内版本。
4. **查社区/上游**：Triton、vLLM、torch_musa 的 issue/PR，确认是否已修复、修在哪个版本。
5. **最后才写 fallback**：写入本台账，标 M/U/W/B，并附探针与回归证据。

> 判据：不能仅凭本机容器版本断言“厂商不支持”；也不能仅凭 release note 断言“镜像里已有”。

---

## S5000-001 · Triton 无 `tl.gather`

- **级别**：M
- **探测**：

```python
import triton, triton.language as tl
assert triton.__version__ == "3.2.0"
hasattr(tl, "gather")  # False
```

- **CUDA/上游**：较新的 Triton 已有 `tl.gather`（上游 PR #5262）；vLLM 自身在 `vllm/model_executor/layers/fla/ops/op.py` 对旧 Triton 有“返回 None”的占位，不能用于真实 QSA metadata。
- **S5000 现状**：没有；不是 S4000 专属问题。
- **本项目 fallback**：`qwen4_exp/common/qsa_cache.py` 选择 `_build_qsa_metadata_torch`。
- **状态**：open；等 Triton-MUSA 升级到含 `tl.gather` 的版本后再切回 Triton builder。
- **版本错位**：MUSA SDK 5.2.0 的 release note 写明 Triton-MUSA 支持 3.2.0 与 **3.6.0**；但当前镜像 `rc5.2.0-v2.9.1.post1-S5000-py310` 内实际是 **3.2.0**，属于“发布包/镜像内容不一致”，需要取更新的厂商镜像验证。
- **备注**：torch builder 当前没有 GPU→CPU 同步，数值正确；代价是每步 launch/alloc 偏多。

## S5000-002 · `_C::persistent_topk` 声明存在但无 MUSA 实现

- **级别**：U
- **探测/现象**：

```python
hasattr(torch.ops._C, "persistent_topk")  # True
# 实际调用：
torch.ops._C.persistent_topk(...)
# NotImplementedError: Could not run '_C::persistent_topk' with arguments
# from the 'musa' backend
```

- **为什么危险**：只看 `hasattr` 会误判“可用”，必须实际调用。
- **CUDA/上游**：CUDA 有对应 kernel；`cooperative_topk` 同理。
- **本项目 fallback**：`qwen4_exp/nvidia/ops/qsa_indexer.py::_topk` 在 MUSA 上走 `torch.topk` + visible mask，并用 `k=min(block_topk, columns)` 处理短 prefill。
- **状态**：open；等 MUSA 侧提供真实 topk kernel 后再切回。
- **回归**：短 prefill（512 token）不能触发 `selected index k out of range`。

## S5000-003 · `get_scheduler_metadata` 缺失

- **级别**：M
- **探测**：`vllm_musa.v1.attention.backends.fa_utils` 没有该符号；CUDA 路径有。
- **S5000 现象**：`flash_attn.py` 调用处 `NameError`。
- **本项目 fallback**：`_get_scheduler_metadata` 在 `NameError` 时返回 `None`，跳过 AOT 预排程。
- **影响**：正确性不受影响；只影响 FA3 的 AOT/性能/图捕获。
- **状态**：open。

## S5000-004 · `reshape_and_cache_flash` 可用（非缺口）

- **级别**：N（已确认可用）
- **路径**：`vllm_musa.v1.attention.backends.fa_utils.reshape_and_cache_flash`
- **结论**：不是手搓；布局与上游 NHD 调用点兼容。保留官方 MUSA 导入。

## S5000-005 · 静态引擎缺少 DMA binding（项目构建缺口）

- **级别**：B
- **现象**：

```text
[gpf-tile] _dma_hostbuf_range(w13) returned None (node=0)
[gpf-tile] _issue(0) failed -> give up (layer)
```

- **根因**：xiaotu engine 以 `XIAOTU_MOE_NO_CUDA` 构建，`copy_hostbuf_to_device*` 被预处理掉，不是 MUSA 厂商 API 缺失。
- **S5000 MUSA 头文件确认**：存在 `musaMemcpyAsync` / `musaMemcpy2DAsync`。
- **修复中**：给 `binding.cpp` / `moe_v2.hpp` 增加 `XIAOTU_MOE_MUSA` 分支，用 `musaMemcpy*` + `musaHostRegister`。
- **状态**：`XIAOTU_MOE_MUSA` 分支已落地；`musaMemcpy*` 对连续区间走线性 copy（避免 h=1 的 2D 慢路径），`pin_hostbufs` 已实测 6/6；待 15K 运行验证 H2D 与 tile ring。

---

## S4000 历史 W 级（仅供参考，禁止在 S5000 重演）

以下问题在 S4000 上实际发生过，保留作为“声明实现了但结果是错”的判例：

1. **`fp8(e4m3) → bf16` 部分尺寸静默错**
   - 失败尺寸：256 / 1024 / 2048 / 4096 / 16384 / 32768；约一半元素错，误差 ~0.5。
   - 通过尺寸：512 / 768 / 1000 / 1280 / 8192 / 65536，呈**尺寸非单调**。
   - `muDNN ERROR# NOT_SUPPORTED in Tensor::SetType` **不抛错，静默返回无效值**。
   - 结论：任何 fp8 反量化必须逐元素对拍，不能仅凭“API 返回成功”签收。

2. **图捕获中途失效且无 traceback**
   - 表现为 replay 静默走错路径；必须用端到端数值探针验证，不只看 `health=200`。

3. **能力探测误判**
   - `has_device_capability(90)` 在 MUSA 恒 False ⇒ 静默退化空实现；不能把 CUDA 能力判断直接套到 MUSA。

---

## 记录模板

```markdown
## S5000-XXX · 标题

- 级别：M / U / W / B
- 探测命令/代码：
- 期望（CUDA/上游）：
- S5000 实际：
- 本项目 workaround：
- 正确性证据：
- 状态：open / fixed
```
