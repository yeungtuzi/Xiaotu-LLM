# MUSA / MTT S4000 问题清单（厂商提交版）

> 环境：MTT S4000（`mp_22`，`__MUSA_ARCH__=220`）、MUSA Toolkit 5.2.0（mcc/clang-14）、
> torch 2.11 + torch_musa、Ubuntu 22.04、vLLM 0.32.0（Qwen3.8-Flash-Next FP8 推理）。
> 每条都附**最小复现**与**我们的临时规避**，便于定位。

---

## 1. 【严重】fp8(e4m3) → bf16 转换在部分尺寸下**静默返回错误数据**

### 现象

| numel | 不一致元素 | 最大误差 | 数据 absmax |
|---|---|---|---|
| **256** | 128 / 256 | 0.449 | 0.247 |
| 512 / 768 / 1000 / 1280 | 0 | — | 0.25 |
| **1024** | 511 / 1024 | 0.480 | 0.249 |
| **2048 / 4096** | 128 / 1152 | 0.44 / 0.48 | 0.25 |
| 8192 | 0 | — | 0.25 |
| **16384 / 32768** | 7417 / 16358 | 0.50 / 0.50 | 0.25 |
| 65536 / 131072 / 262144 | 0 | — | 0.25 |

- 失败时**恰好约一半元素错**（128/256、512/1024）⇒ 疑似向量化内核只处理对了一半 lane；
- 误差 ~0.5 而数据 absmax 仅 0.25 ⇒ **不是舍入误差，是读到了错误的数据**；
- 失败与否随尺寸**非单调** ⇒ 按尺寸/对齐选了不同内核变体，其中一个是坏的；
- 同期日志：`muDNN(v3400) ERROR# NOT_SUPPORTED in Tensor::SetType, Reason: The data type is not supported on current GPU architecture`
  —— **库知道不支持，但没有报错，静默返回无效值**。

### 最小复现

```python
import torch, torch_musa
x = (torch.rand(1024, device="musa") - 0.5) * 0.5
x8 = x.to(torch.float8_e4m3fn)
a = x8.to(torch.bfloat16)                     # 错：约 512/1024 元素错
b = x8.to(torch.float32).to(torch.bfloat16)   # 对
print(int((a != b).sum()))                    # 期望 0，实际 512
```

> 参照物：`fp8 → float32` 在**所有尺寸**都正确（16 … 1048576 全部验证过）。

### 影响

任何做 FP8 权重/激活的模型都会拿到**看似正常、实则错误**的数值，且极难排查
（我们为此花了数小时，最终靠逐层 absmax 才定位到是这一层）。

### 我们的规避

所有 fp8 反量化统一走 `.float()` 中转：`(x8.float() * scale).to(bf16)`。

### 建议

1. 修复该 bf16 转换内核；
2. **在未支持时抛错**，不要静默返回垃圾。

---

## 2. 【严重】torch_musa 未接上 **graph / stream** 能力（SDK 有，绑定缺失）

| 检查 | 结果 |
|---|---|
| SDK `musaGraphCreate` / `musaStreamBeginCapture` / `musaGraphInstantiate` | ✅ 存在（`include/cuda_runtime.h`） |
| `torch.musa.CUDAGraph` | ❌ 不存在 |
| `torch.musa.graph` | ⚠️ 空 class（`dir()` 无任何 Graph 成员） |
| `torch.musa.graph_pool_handle()` | ⚠️ 返回假值 `(0, 1)` |
| `torch.musa.is_current_stream_capturing()` | ⚠️ 恒 `False` |
| `torch.musa.Stream()` | ❌ `AttributeError: module 'torch._C' has no attribute '_cuda_getCurrentStream'` |
| `torch.cuda.Stream()` | ❌ `RuntimeError: torch.cuda.Stream requires CUDA support` |
| `torch.cuda.CUDAGraph` | ⚠️ 类存在，但没有可用 Stream 就无从 capture |

### 最小复现

```python
import torch, torch_musa
torch.musa.Stream()          # AttributeError: _cuda_getCurrentStream
```

### 影响

CUDA-graph 不可用 ⇒ vLLM 只能 `enforce_eager=True`，每步都有 kernel 启动与 Python 开销，
长序列/高并发性能损失明显。**这是本卡跑大模型的主要性能天花板之一。**

### 我们的规避

`enforce_eager=True`（接受性能损失）。

### 建议

补齐 `torch_musa` 的 Stream / CUDAGraph 绑定（GPU 侧 API 已具备，缺的是 PyTorch 后端）。

---

## 3. 【严重】`_C` 部分算子未注册 **MUSA(PrivateUse1)** dispatch

```
RuntimeError: Could not run '_C::persistent_topk' with arguments from the 'musa' backend
RuntimeError: Could not run '_C::top_k_per_row_decode' with arguments from the 'musa' backend
```

`torch._C._dispatch_has_kernel_for_dispatch_key(op, PrivateUse1)` 返回 **False**。

### 影响

我们只能退回 Python 版 `torch.topk`（并额外做 padding / index clamp），性能与稳定性都受影响。

### 建议

为这些算子注册 `PrivateUse1` 实现，或明确文档说明「仅 CUDA 可用」。

---

## 4. 【中】Triton-MUSA 与上游差距（三个具体问题）

### 4.1 `tl.gather` 不存在

```
AttributeError: module 'triton.language' has no attribute 'gather'
```

上游 Triton 早已有 `tl.gather`；我们只能把 gather 手写展开成 load/store。

### 4.2 共享内存上限 **72 KiB**（S4000）

```
triton.runtime.errors.OutOfResources: shared memory, Required: 131072, limit: 73728
```

同样的 kernel 在 H100（228 KiB）上不用改。我们只能把 `TILE_R=16`、`K_TILES=8`、`STAGES=1` 硬调小。
**建议**：文档里明确写出各代卡的 shared memory 上限，并在编译期给出更友好的提示。

### 4.3 `tl.arange` 参数必须是 `tl.constexpr`

```
ValueError: arange's arguments must be of type tl.constexpr
```

上游对 `HEAD_DIM` 这种常量表达式更宽松；我们改成 `BLOCK_D = 16 if HEAD_DIM < 16 else HEAD_DIM` 才好。

---

## 5. 【中】Triton 内核签名里 `uint8` 指针被错拼成 `'u8>'`

```
KeyError: 'u8>'
```

通过打印 JIT 函数的 signature 定位到 `weight_ptr: 'u8>'`：`tl.pointer_type(tl.uint8)`
作参数时前端把签名拼坏了。我们把张量先 `.view(torch.uint8)` 传进去绕过。

**建议**：修正签名拼装（应生成 `*u8` 之类合法类型串）。

---

## 6. 【中】`masked_fill` 未实现 Float8_e4m3fn

```
NotImplementedError: masked_fill not implemented for Float8_e4m3fn
```

规避：先在 float32 上做 mask 再转回。

---

## 7. 【中】TCE flash-attention 内核要求 **MP31**，S4000 用不了

```
run_bf16tce_flash_atten_512_256x128x128_causal_persistence requires MP31 or above
```

（`flash_attn_varlen_func` 与 `mate_flash_attn_varlen_func` 都一样。）

**建议**：要么给 mp_22 提供等价内核，要么在 API 层直接报「该内核需要 MP31」，别等到运行期。

---

## 8. 【中】MATE `causal_conv1d_update` 非法访存

```
Triton Error [MUSA]: an illegal memory access was encountered
  at causal_conv1d_update (GDN decode)
```

我们只能关掉 MATE GDN decode（`VLLM_MUSA_ENABLE_MATE_GDN_DECODE=0`），等于放弃这项优化。

---

## 9. 【中】`deep_ep_cpp` 扩展 ABI 不匹配

```
ImportError: deep_ep_cpp.cpython-310-x86_64-linux-gnu.so: undefined symbol: mcclA
```

（该 .so 由厂商包提供，与当前 MCCL 版本不匹配。）

---

## 10. 【低】UVA 相关：语义容易误解

- `musaHostGetDevicePointer` 在 S4000 上返回的是**映射后的设备地址**，
  与宿主 `data_ptr()` **不相等**（实测 host `94301804397120` vs device `4608491648`）；
  建议文档明确写「这不是 alias，是映射」，否则很容易写出错误的验收条件。
- `torch.ops._C.get_cuda_view_from_cpu_tensor` 需要先 `import vllm._C` 才会注册 CPU
  dispatch key —— 虽然属于上层库行为，但「op 已存在却报 Could not run ... 'CPU' backend」
  的报错非常误导。

---

## 附：我们已验证 **bf16 在 MP22 是原生可用**（供参考）

- 官方规格书：FP16/BF16 **100 TFLOPS**、INT8 **200 TOPS**；
- SDK：`include/musa_bf16.h` 的整段 bf16 API 由 `#if __MUSA_ARCH__ >= 220` 保护；
- 实测：`mcc -arch mp_22` 编译后设备端 `__MUSA_ARCH__ == 220`；
- 我们的模型全程 bf16 前向，逐层数值正常。

⇒ 所以第 1 条是**转换内核的实现缺陷**，不是硬件不支持 bf16。

---

## 1bis. 【严重 · 补充】`fp8(e4m3) → float32` **也不总是正确**（推翻我最初结论）

第 1 条里我写过"`fp8 → float32` 在所有尺寸都正确"——**这个结论是错的**，
当时的扫描只用了**一维连续**张量。实际运行中发现反例：

### 现象

一个 `torch.float8_e4m3fn`、形状 `(7, 2560)`（17920 元素）的设备张量，其数据由
"从锁页主机表按行 gather 后逐字节拷入"得到。对该张量执行：

```python
t.float().abs().max()
```

返回 **`1.49514e+32`** —— 而 `float8_e4m3fn` 的**数值上限只有 448**，
**任何单字节都不可能表示出 1.5e32** ⇒ 这是**转换错误**，不是数据错误。

### 关键判据（可用于你们复现）

* 同一份数据，**搬到 CPU 后再 `.float()`** 得到的是正常值（absmax ≈ 1.9e-1 量级）；
* 换算一致性：`1.49514e32 × scale(1.993e-4) = 2.98653e28` —— 后续所有异常数值都是它的等比放大；
* 张量 `is_pinned()` 为 **False**（所以与"是否锁页"无关），
  更像是**形状/对齐触发的转换内核缺陷**（与第 1 条 `fp8→bf16` 的**非单调按尺寸失败**同族）。

### 影响

**静默**给出 1e32 量级的值，一路污染到最终 logits（表现为输出恒定、无意义）。

### 我们的规避（已验证有效）

反量化**无条件搬到主机侧**做：

```python
out = (embeddings.cpu().float() * float(scale)).to(dtype).to(device)
```

张量很小（tokens × hidden），代价可忽略。修复后 layer1 的 absmax 从 `3.37e27` 回到 `3.83e-2`，
模型开始输出有意义文本。

### 建议

1. 排查 fp8 转换内核在**非一维/特定对齐**输入上的实现（怀疑向量化路径的对齐假设）；
2. **同样建议在转换结果明显超出该 dtype 值域时报错**，而不是静默返回。

---

## 11. 【严重】同一进程的**第二个请求**会永久阻塞在 `muLaunchKernel`

### 现象

单进程连续处理两个请求：

* 第 1 个请求：**正常完成**（我们拿到 `'\n\nThe sum of 1 and 1 is 2. This is is'`）；
* 第 2 个请求：一进入 `prepare_inputs`（框架侧准备输入）就**永久 hang**。

### 证据（py-spy 原生栈）

```
Thread (active): "MainThread"
    prepare_inputs (vllm/v1/worker/gpu/model_runner.py)
    muLaunchKernel        (libmusa.so.5.2.0)
    musaLaunchKernel      (libmusart.so.5.2.0)
    musa::dnn::_FillRun   (libmudnn_tensor.so.3.4.0.0)
    musa::dnn::Fill::Run  (libmudnn.so.3.4.0.0)
```

* 线程状态 **active**（不是 sleep）；
* 但**整进程 0% CPU**，**GPU 0% 利用率**，显存仍被占用；
* 我们逐个排除了自己的代码：换成"折 0 / 折末槽 / 纯设备端 clamp / 关掉 padding fill"，
  **四种改法下 hang 位置只在小范围内平移**（`fill_` → `async_tensor_h2d`），
  ⇒ **任何一次设备操作都会阻塞**，与具体算子无关。

### 影响

**任何多请求 / 多轮会话服务都不可用**；单请求推理正常。

### 我们的环境

`MUSA_LAUNCH_BLOCKING=1`、eager（无 CUDA graph）、`torch 2.11.0.post1` + `torch_musa 2.11.0.post1+b431668`、
MTT S4000（`mp_22`）、驱动 5.2.0。

### 建议

1. 排查"同一 context 内第二次 `muLaunchKernel` 永久阻塞"的原因（疑似上一次请求结束时
   有未回收的异步状态 / 事件）；
2. 若能提供 `MUSA_LAUNCH_BLOCKING` 相关已知问题说明，对我们帮助很大
   （我们正在测这个开关是否相关）。

---

## 12. 【严重】MCCL 在本机**完全不可用**（单卡模式也失败）⇒ tp≥2 无法启动

### 现象

最小复现（`probes/test_mccl.py`，两卡 all-reduce）：

```
Rank 0/1: RuntimeError: MCCL error in: .../MCCLUtils.cpp:73,
          unhandled system error, MCCL version 2.30.4
```

带 `MCCL_DEBUG=INFO` 后：

```
MCCL INFO ENV/Plugin: Could not find: libmccl-env.so
MCCL INFO Bootstrap: Using enp33s0f0np0:192.168.100.248
MCCL INFO musaDriverVersion 50200
MCCL WARN misc/musawrap.cc:126  Musa failure 101 'invalid device ordinal'
```

关键复测：

| 场景 | 结果 |
|---|---|
| 2 rank all-reduce | ❌ `unhandled system error` |
| 1 rank（`--nproc_per_node=1`）all-reduce | ❌ **同样失败** |
| 去掉 `MUSA_VISIBLE_DEVICES` | ❌ 同样失败 |
| 纯双卡计算（zeros/add/sync，不走通信） | ✅ 正常 |

⇒ **问题在 MCCL 初始化本身，与多卡传输/可见性无关**。

### 环境

* `libmccl.so.2.11.4`（34 MB）存在；
* MCCL 报 `version 2.30.4`，`musaDriverVersion 50200`（驱动 5.2.0）；
* 缺 `libmccl-env.so`（ENV/Plugin，官方说可选）。

### 影响

* **vLLM 的 tensor_parallel_size ≥ 2 无法启动**（每层都要 all-reduce）；
* 4 卡全驻留方案同样被阻塞；
* 我们的 `tp=2` 尝试最终报 `buffer_utils.py:160 torch.zeros` 的
  `illegal memory access`，那是 MCCL 初始化失败后**上下文被污染**的次生现象。

### 建议

1. 确认 MCCL 2.30.4 与驱动 5.2.0 的兼容矩阵；
2. `invalid device ordinal` 提示 MCCL 在取设备序号时用的口径与 torch 不一致
   （我们已试过 `MUSA_VISIBLE_DEVICES` 有/无两种情况）；
3. 是否需要在容器/特权模式下运行 MCCL（IPC 共享内存/设备节点权限）？

---

## 12 条【勘误 2026-10-24】：**不是厂商问题，是本机 ld.so 配置**

### 我当时的结论（错误）

曾把「MCCL 单 rank 也失败、`invalid device ordinal`」记为厂商缺陷。

### 实际根因（由另一位 agent 定位并修复，已复核）

`/etc/ld.so.conf.d/00-mccl-s5000.conf` 把 **`/opt/mccl-s5000/lib`** 排在搜索路径最前：

```
libmccl.so.2 => /opt/mccl-s5000/lib/libmccl.so.2      # 2.30.4pre —— S5000 版
libmccl.so.2 => /usr/local/musa/lib/libmccl.so.2      # 2.11.4    —— S4000 版
```

本机**只有 S4000**，却加载了 S5000 的 MCCL ⇒ 必然
`Musa failure 101 'invalid device ordinal'` +
`mcclMaxSharedMem 40960 exceeds device/fn maxSharedMem 22656`。

即：`MCCL version 2.30.4pre` 这行就是证据（我一直看到它，但没意识到 2.30.4 是 S5000 版本）。

### 修复后复核（2026-10-24 00:15）

```
$ mpirun --version        → mpirun (Open MPI) 4.1.2
$ ldconfig -p | grep libmccl.so.2
    libmccl.so.2 => /usr/local/musa/lib/libmccl.so.2     ← S4000 ✓
$ dpkg --audit            → 干净
$ all_reduce_perf（单卡 8MB）→ 146 GB/s ✓
```

详细修复步骤见 `dev-docs/ENV_OPENMPI_MCCL_FIX_2026-10-24.md`（含回滚）。

### 教训（写给自己）

1. **看到 `MCCL version X` 要先确认它属于哪张卡**（S4000 应为 `2.11.4`，S5000 是 `2.30.4`）；
2. 报厂商之前**先 `ldconfig -p` 查库解析路径**——多卡 SDK 并存时极易抢错；
3. 本条保留在文档里**作为勘误**，不再作为厂商待办。

---

## 13. 【严重 · 阻塞特性】MUSA 构建下**引擎完全没有 H2D DMA 通道**（GPU 流式预填无法工作）

> 记录时间：2026-10-24 08:55 · 复现机：192.168.100.248（2× MTT S4000 / mp_22）·
> 组件：本仓 `xiaotu_moe`（CPU 专家引擎）+ `vllm_xiaotu_moe`（vLLM 侧插件）

### 13.1 现象

`② GPU prefill`（tile 环 / 微 ping-pong）在 MUSA 上**永远走不到**：

```
[gpf-tile] _dma_hostbuf_range(w13) returned None (node=0)
[gpf-tile] _issue(0) failed -> give up (layer)
→ 退回"源张量兜底"分支
→ ValueError: Pointer argument (at 4) cannot be accessed from Triton (cpu tensor?)
```

### 13.2 根因（不是配置，是**构建裁剪**）

1. `vllm_xiaotu_moe/gpu_prefill.py::_dma_hostbuf_range()` 在
   `engine.copy_hostbuf_to_device_2d` **不存在**时返回 `None`；
2. 该绑定在 `xiaotu_moe/csrc/python_binding/binding.cpp:1106` 处被
   **`#ifndef XIAOTU_MOE_NO_CUDA`** 包住，实现直接调用 `cudaMemcpyAsync` /
   `cudaMemcpy2DAsync` / `cudaHostRegister`；
3. **MUSA 上没有 cudart / libcuda**，本仓构建脚本 `scripts/build_engine_variants.sh`
   在这种情况下**必须**加 `-DXIAOTU_MOE_NO_CUDA`（否则脚本直接 exit 2）；
   ⇒ 这段 DMA 代码**整段被编译掉**；
4. 实测旧二进制：
   ```
   $ strings _xiaotu_moe_C_avx512_vnni.cpython-310-*.so | grep -c copy_hostbuf_to_device
   0            # 连非 2D 版本都没有
   $ strings ... | grep -c pin_hostbufs
   1            # 而同段的 pin_hostbufs 在（说明是同一处被裁剪）
   ```

⇒ **MUSA 端的 GPU 流式预填从设计上就没有可用的主机→设备拷贝通道**，
必须有 MUSA 版实现才能工作。

### 13.3 我们做的 MUSA 端口（workaround，已在本机验证可用）

在 `binding.cpp` 增加 `#if defined(XIAOTU_MOE_MUSA)` 分支，用 MUSA runtime API
补上**同签名**的两个方法：

```cpp
musaMemcpyAsync (void* dst, const void* src, size_t n,
                 musaMemcpyHostToDevice, musaStream_t stream);
musaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                  size_t width, size_t height,
                  musaMemcpyHostToDevice, musaStream_t stream);
```

构建（**这是 MUSA 侧需要的新增步骤**）：

```bash
NO_CUDA=1 PYTHON=/root/venvs/vllm310/bin/python \
EXTRA_FLAGS='-DXIAOTU_MOE_MUSA=1 -I/usr/local/musa/include \
             -L/usr/local/musa/lib -lmusart -Wl,-rpath,/usr/local/musa/lib' \
bash scripts/build_engine_variants.sh
```

验证：8 个 ISA 变体的 `.so` 全部含 `copy_hostbuf_to_device_2d`。

### 13.4 给厂商的两个具体问题

1. **`cudaMemcpy2DAsync` 在 MUSA 的对应物是 `musaMemcpy2DAsync`**，
   但**驱动头 `<cuda.h>` 只暴露 `muMemcpy2DAsync`、运行时头才暴露 `musaMemcpy2DAsync`**，
   且枚举名是 `musaMemcpyHostToDevice`（不是 `muMemcpyHostToDevice`）——
   跨头文件混用会直接编译失败。建议在文档里给出**一张对照表**
   （CUDA 名 → MUSA runtime 名 → MUSA driver 名）；
2. **是否有 CUDA 兼容层**（把 `cudaMemcpy*` 映射到 `muMemcpy*`）？
   若存在，很多"CUDA-only 的 host→device 快路径"就能零改动移植到 S4000；
   若不存在，建议官方提供 **`cuda_runtime_api.h` 兼容 shim**（哪怕只覆盖
   Memcpy/HostRegister/Stream 这几个最常用的），能显著降低移植成本。

### 13.5 定性

| 项 | 结论 |
|---|---|
| 是否厂商 bug | **不是 bug，是能力缺口（无 CUDA 兼容层）** |
| 影响 | 任何依赖 `cudaMemcpy*` 的 GPU 流式/预取路径在 MUSA 上**全部不可用** |
| 我们的状态 | 已用 `musaMemcpy*` 端口绕过，MUSA 端可编译、符号齐备 |
| 待厂商回答 | 13.4 的两个问题（API 对照表 / 是否提供兼容层） |

---

## 14. 【严重】一次 H2D DMA 失败导致 **GPU 固件 panic（CSW_TIMEOUT）**，而且显存**不回收**（`-r 0` 无效，只有 `-r 2` 能救）

> 记录时间：2026-10-24 09:35 · 机器：192.168.100.248 · 2× MTT S4000（mp_22）·
> 驱动/MUSA Toolkit 5.2.0 · 复现路径：`musaMemcpy2DAsync`（我们自己移植的 H2D 通道）

### 14.1 时间线（dmesg 实录）

```
01:21:02  mtgpu 0000:3d:00.0: comm = python, pid = 2962033, force destroy app memory context
01:22:42  submission response FAULT: pid(2962082), comm(python), token(...), cmd(GPU_CMP)
01:22:42  FAULT DUMP [WR/RD 0/0]: reason 8 ...
01:22:42  FW_PANIC: DMUP_REASON 0x8, FAULT_ID 0x0 false:/tmp/mtgpu-fw.9DgH5S/so...
01:22:42  Fault Reason : CSW_TIMEOUT
01:22:43  （用户态）musaMemcpy2DAsync 返回 999 musaErrorUnknown
```

**顺序很关键**：固件 panic 在前，`999` 在后 ⇒ 那个 999 **不是**"2D 拷贝参数非法"，
而是"设备已经处于 panic 状态"的表现。

### 14.2 后果：显存**不随进程退出回收**

* 进程已全部退出（`ps` 无 python/EngineCore/VLLM 进程）；
* `lsof /dev/mtgpu*` 只剩 `btop`（监控工具，不占显存）；
* 但 `torch.musa.mem_get_info()` 报 **used = 44.3 / 45.0 GiB**，
  连 `torch.empty(8GiB)` 都直接 `MUSA out of memory`；
* `mthreads-gmi -q -d MEMORY` 也报 `Used 42604MiB / 43950MiB`（与 btop 一致）。

### 14.3 恢复方法（实测有效，务必按这个顺序）

```bash
mthreads-gmi -i 0 -r 0     # ❌ 报告 success，但显存**没回来**
mthreads-gmi -i 0 -r 2     # ✅ 整设备复位
mthreads-gmi -i 1 -r 2
```

复位后：

```
dev0: free=45.8GiB used=2.1GiB
dev1: free=45.8GiB used=2.1GiB
```

⇒ **运维判据**：GPU 出现 `submission response FAULT` / `FW_PANIC` 之后，
**不要**指望进程退出能释放显存；直接 `mthreads-gmi -r 2`（整设备），
`-r 0` 不够。

### 14.4 给厂商的三个问题

1. **用户态一次错误的 2D 拷贝（或错误 stream 句柄）为什么会把 GPU 固件打到
   `CSW_TIMEOUT` panic？** 预期应当返回明确错误（如 `musaErrorInvalidHandle` /
   `musaErrorInvalidValue`），而不是 panic 整个设备；
2. **panic 之后为什么显存不随进程销毁回收？** 日志里 01:21:02 有
   `force destroy app memory context`（另一次退出），但 01:22:42 panic 之后
   没有对应的回收路径；
3. **`mthreads-gmi -r 0` 报 success 但实际没有复位显存** —— 这是误导性行为，
   建议 `-r 0` 在检测到 FW panic 时返回失败或自动升级为整设备复位。

### 14.5 附带更正（我们自己文档里的错误结论）

我们一度以为"`musaMemcpy2DAsync` 在 `height==1` 时不支持（返回 999）"。
独立探针（`probes/musa_memcpy2d_probe.py`）证明：**h=1、pitch==width、
宽度到 200 MiB 全部 OK**；锁页/未锁页也都 OK（`probes/musa_pin_probe.py`）。
⇒ 999 的真实含义是"固件已 panic"，与形状无关。

---

## 15. 【严重 · 已确认历史多次发生】GPU 固件 panic 的两类形态，且**厂商仓库无更新可用**

> 记录时间：2026-10-24 09:55 · 机器：192.168.100.248 · S4000 ×2 · MUSA 5.2.0

### 15.1 journald 全量统计（不是偶发）

```
$ journalctl -k | grep -cE 'CSW_TIMEOUT'      → 3
$ journalctl -k | grep -E 'FW_PANIC'          → 多次，最早 2026-10-21 19:05:50
```

| 时间 | 形态 | 说明 |
|---|---|---|
| 2026-10-21 19:05 / 19:12 / …（共 6+ 次） | `FW_PANIC: DMUP_REASON 0x7` + `Fault Reason: PAGE_FAULT` | 用户态 python 进程（`cmd(GPU_CM...)`）触发 |
| 2026-10-23 16:46 | `CSW_TIMEOUT` | tp=2 相关实验期间 |
| 2026-10-24 01:22（两张卡各一次） | `CSW_TIMEOUT` + `FW_PANIC 0x8` | FP8 GPU prefill 的 H2D DMA 期间 |

⇒ **两类**：`0x7 PAGE_FAULT` 与 `0x8 CSW_TIMEOUT`。
用户侧记忆"之前也碰到过一次"被证实——实际是**多次**。

### 15.2 驱动/固件**没有可用更新**

```
$ apt-cache madison mthreads-dkms
mthreads-dkms | 5.2.0 | https://dl.mthreads.com/repo/repository/ubuntu2204 jammy/main
mthreads-dkms | 5.1.0 | 同上
$ apt-cache policy mthreads-dkms → Installed 5.2.0 / Candidate 5.2.0
```

⇒ 厂商公开源里 **5.2.0 就是最新**（另有 5.1.0 更旧）。无法靠升级驱动规避。

### 15.3 影响（对我们）

* panic 后**显存不回收**（见第 14 条），必须 `mthreads-gmi -r 2`；
* 在 panic 之后继续调用任何 MUSA API 都只得到 `999 musaErrorUnknown`，
  **错误信息会把人引向错误方向**（我们曾误判为"2D 拷贝不支持 h=1"）；
* 任何依赖长时间 GPU 流式（prefill 流水）的路径都可能踩到，属**阻塞级**风险。

### 15.4 复现脚本（已入库，供厂商直接跑）

`probes/repro_fw_panic.py`：

```bash
GPU_DEVICE=MUSA /root/venvs/vllm310/bin/python \
  /root/vllm-xiaotu-moe-tmp/probes/repro_fw_panic.py 50 2d
# 参数：<迭代次数> <mode: 2d|mix|triton>；REPRO_WIDTH 可改搬运宽度
# 每次迭代后自动检查 journald 是否新增 FW_PANIC/CSW_TIMEOUT
```

复现步骤（受控）：
1. 先 `mthreads-gmi -r 2` 让设备处于干净状态；
2. 跑上面的脚本，记录**第几次迭代**触发；
3. 触发后 `dmesg -T | tail -40` 存证；
4. `mthreads-gmi -i 0 -r 2` 恢复。

### 15.5 请厂商回答

1. `0x7 PAGE_FAULT` 与 `0x8 CSW_TIMEOUT` 分别对应固件里什么检查？用户态如何避免？
2. 是否有**内部/热修版驱动**（比 5.2.0 新）可提供？公开源没有。
3. panic 后显存不回收 + `-r 0` 假成功，是否会在后续版本修复？

---

## 16. 固件 panic 问题的**完整环境指纹**（报障/复现必带）

> 本节固定内容，后续每次报此问题都引用本节，避免版本信息散落。

### 16.1 硬件

| 项 | 值 |
|---|---|
| 卡型号 | **MTT S4000** ×2（`Device ID 0x0327`，`Subsystem 0x0327`）|
| 序列号 | GPU0 `MY1VE5023CJ00469` · GPU1 `MY1VE5023CJ00370` |
| 插槽 | `3(BaseBoard_Slot03)` / PCI `0000:3d:00.0`、`0000:bd:00.0` |
| 显存 | 48 GiB/卡（`Total 49152MiB`）|
| 主机 | Hygon C86 7385 32-core ×2 socket（64 线程），Ubuntu **22.04.4 LTS**，kernel **5.15.0-105-generic** |

### 16.2 软件栈

| 项 | 值 |
|---|---|
| 驱动包 | `mthreads-dkms 5.2.0`、`mthreads-common 5.2.0`、`libdrm-mtgpu 5.2.0` |
| 驱动版本（gmi） | **`5.2.0-server`** |
| 内核模块 `mtgpu` srcversion | `64FE6927D4B6BBC62CCFB40` |
| 内核模块内部版本串 | `Driver Version: [3df6cad-dirty] Shared-Inc: [mtub-sysroot]` |
| **GPU 固件** | `Version : Gen 2. **Jun 24 2026 21:11:55**, Code: **517dfc6-dirty** ABI: <mtfw_main.c:112>` |
| MUSA Toolkit | **5.2.0**（`MUSA_VERSION 50200`，`libmthreads-compute 5.2.0`）|
| 其它 | `libmthreads-mtml 2.4.2`、`mthreads-gmi 2.4.2`、`libmudnn3 3.4.0.0`、`libmupp 1.13.0` |

⚠️ 注意驱动与固件都带 **`-dirty`** 后缀（`3df6cad-dirty` / `517dfc6-dirty`）
—— 说明这是**本地改动过的构建**，请厂商确认是否为可发布版本。

### 16.3 公开源版本情况

```
$ apt-cache madison mthreads-dkms
mthreads-dkms | 5.2.0 | https://dl.mthreads.com/repo/repository/ubuntu2204 jammy/main
mthreads-dkms | 5.1.0 | 同上
```

⇒ **5.2.0 即公开最新**，无更新可升；若需修复只能靠厂商内部版本。

### 16.4 报障一句话

> MTT S4000（驱动 5.2.0-server / 固件 `517dfc6-dirty` Jun 24 2026、MUSA 5.2.0，
> Ubuntu 22.04 + kernel 5.15）在**用户态 H2D DMA + GPU 计算**负载下多次触发
> GPU 固件 panic：`DMUP_REASON 0x7 PAGE_FAULT`（10-21 多次）与
> `0x8 CSW_TIMEOUT`（10-23/10-24），panic 后显存不回收且 `mthreads-gmi -r 0` 假成功。

---

## 17. 第三类 panic：`DMUP_REASON 0x11` / `KILL_TIMEOUT` —— **SIGKILL 正在跑内核的进程就会触发**

> 记录时间：2026-10-24 11:20 · 同机同栈（见 §16 环境指纹）

### 17.1 触发方式（我们无意中复现）

为了换参数，我们对正在做 15K prefill 的两个 worker 执行了 `kill -9`：

```
03:18:1x  mtgpu 0000:3d:00.0: submission response FAULT: pid(3226322), comm(python), cmd(GPU_CMP)
03:18:1x  MTGPU: FW_PANIC: DMUP_REASON 0x11, FAULT_ID 0x0
03:18:1x  MTGPU: Fault Reason     : KILL_TIMEOUT        ← 两张卡各一次
```

⇒ **用户态 SIGKILL 一个仍有在飞内核的进程，会让 GPU 固件进入 KILL_TIMEOUT panic**。

### 17.2 三类 panic 汇总（截至 2026-10-24）

| 编号 | `DMUP_REASON` | `Fault Reason` | 触发场景 |
|---|---|---|---|
| 0x7 | 7 | `PAGE_FAULT` | 2026-10-21 多次，MATE/GDN 实验期 |
| 0x8 | 8 | `CSW_TIMEOUT` | 2026-10-24 01:22，FP8 GPU prefill 的 H2D + 大内核 |
| **0x11** | **17** | **`KILL_TIMEOUT`** | **2026-10-24 03:18，我们 `kill -9` worker** |

### 17.3 给厂商的问题

1. **进程被 SIGKILL 时，驱动/固件为何不是"取消在飞任务"而是 panic？**
   CUDA 的行为是：上下文销毁，内核被终止，设备保持可用（最多报 `cudaErrorLaunchFailure` 给该上下文）。
   建议对齐：SIGKILL ⇒ 终止该进程的队列 + 回收其显存，**不要** panic 整卡；
2. 若确实无法优雅取消，至少应在 panic 后**自动复位并回收显存**（当前 `-r 0` 还假成功）；
3. 是否有 `ioctl` 可以让用户态"先冲刷队列再退出"，把 KILL_TIMEOUT 变成可控路径？

### 17.4 我们的运维 SOP（已写进 BUILD_PITFALLS P11）

* 停引擎：先 `kill -TERM` + 等 10–15 s（让它自己收尾），**绝不要直接 `kill -9`**；
* 若必须强杀：杀完立刻查 `journalctl -k | grep -E "FW_PANIC|CSW_TIMEOUT"`，
  有 panic 就 `mthreads-gmi -i N -r 2` 复位再起。

---

## 18. 【严重 · 性能致命】MUSA 的 Triton **`tl.dot` 不走张量核**（比 muDNN 慢 143 倍）

> 记录时间：2026-10-24 12:56 · 机器 192.168.100.248 · S4000 ×2 · MUSA 5.2.0

### 18.1 实测（同形状、同 bf16、同卡）

```
M=N=768..2048, K=2560, bf16, reps=10
torch.mm   (muDNN, 张量核) : 43.15 TFLOPS
triton tl.dot (BM=BN=BK=128, num_warps=8) : 0.301 TFLOPS     ← 143× 慢
```

`tf32`/`allow_tf32` 无关：两边都是 bf16 输入。

### 18.2 后果（这是我们 ② GPU prefill 长期慢的真正原因）

* 我们整套 GPU prefill 的专家 GEMM 都是 Triton 内核（`gate_up_kernel_fp8` / `down_kernel_fp8`）；
* 它们的天花板就是 **0.3 TFLOPS**，而 S4000 的 fp16 峰值约 100 TFLOPS ⇒ **上限被锁在 0.3%**；
* 4096 prompt 理论需要 9.3 TFLOP ÷ 100 TFLOPS = **93 ms**，在 Triton 路径下**物理上不可能**低于
  9.3/0.3 ≈ **31 秒**（实测 3–5 分钟，还叠加了不规则 tile 与解码开销）。

⇒ 之前我把瓶颈归到 `tl.exp2` 解码，是**次因**；主因是 **Triton 在 MUSA 上不生成张量核指令**。

### 18.3 请厂商回答

1. MUSA 的 Triton 后端（triton-musa / `musa_triton`）**什么时候支持 tensor core MMA**？
   当前 `tl.dot` 是否确定退化成了 FMA 循环？
2. 是否有推荐写法/编译开关（如 `tl.dot(..., input_precision=...)`、特定的 BLOCK 形状、
   `num_stages`、`matrix_instr_nonkdim`）能让 MUSA Triton 走到 MMA 指令？
3. 若不支持，官方推荐的高性能路径是什么（muDNN GEMM API / muBLAS / MUTLASS 类模板）？

### 18.4 我们的对策（不依赖厂商修复）

* 专家 GEMM 改用 **muDNN / `torch.bmm`**（已验证 43 TFLOPS）：
  按专家分组把 token 行 gather 成 `[E_tile, R_max, H]`，一次 `bmm` 算完一个 tile，
  再做 SwiGLU / down；
* 权重在**转置那一遍**就解码成 bf16（GPU 上、只做一次），环里存 bf16，
  这样 `bmm` 直接吃 bf16，无需任何内核内解码；
* Triton 只保留给"搬运/转置/规约"这类非 GEMM 的活。

---

## 19. fp8 GEMM 在 muDNN 上**数值不可用**（输出仍是 fp8；bf16/fp16 out 直接失败；无 `_scaled_mm`）

**发现时间**：2026-10-24 20:30（S4000 / driver 5.2.0 / MUSA 5.2.0）

### 19.1 现象

```python
a = torch.randn(4, 32, 128).to(torch.float8_e4m3fn)
b = torch.randn(4, 128, 64).to(torch.float8_e4m3fn)
o = torch.bmm(a, b)          # → dtype = torch.float8_e4m3fn  ❗ 输出仍是 fp8
```

| 调用 | 结果 |
|---|---|
| `torch.bmm(a8, b8)` | **成功，但输出 dtype = fp8**（累加在 fp8 里） |
| `torch.bmm(a8, b8, out=bf16)` | ❌ `Run MUDNN failed in: mudnnBatchMatmulGetWorkspace` |
| `torch.bmm(a8, b8, out=fp16)` | ❌ 同上 |
| `torch.bmm(a8, b8, out=fp32)` | ⚠️ "成功" 但 **rel err = 1.53（153%）** ⇒ 数值完全不正确 |
| `torch._scaled_mm(...)` | ❌ `scaled_gemm is only supported for MUSA_ARCH >= 31` |

### 19.2 影响

**无法用 fp8 直接做专家 GEMM**。这意味着：

* 不能省掉 fp8→bf16 解码那一趟（每层 1.41 GiB 读 + 2.82 GiB 写）；
* 也就无法把每层显存流量从 ~11 GiB 降到 ~2.8 GiB；
* 因此 prefill 想接近 H2D 带宽墙（2.84 s / 4096 token），只能靠
  ①提高 DMA 速率、②热层常驻跳过 DMA、③把解码与转置融合成一趟。

### 19.3 建议

向厂商确认：
1. `mudnnBatchMatmul` 是否支持 **fp8 输入 + bf16/fp32 累加输出**（当前 workspace 获取失败）；
2. S4000（MUSA_ARCH < 31）是否有 `scaled_gemm` 的替代接口（例如 per-block scale 的 GEMM）。

**在拿到支持前，不要基于 fp8 GEMM 设计任何性能方案。**

---

### 19.4 【更正】S4000 **没有 fp8 硬件支持**（与产品 spec 一致）

上文 19.1 的标题"muDNN 支持 fp8 GEMM"是**误读**，必须更正：

* S4000 规格上**无 fp8 tensor core**；
* `torch.bmm(fp8, fp8)` 之所以"不报错"，是因为走了 emulation/饱和路径，
  **输出仍是 fp8**、`out=bf16/fp16` 直接失败、`out=fp32` 数值错误（rel=1.53）；
* ⇒ 正确结论是：**fp8 GEMM 在本硬件上不可用**（静默错误，比报错更危险）。

### 19.5 由此确定的设计约束（重要）

qfn 是 **FP8 权重**模型，而本硬件只能做 **bf16** 矩阵乘 ⇒
**fp8→bf16 解码这一趟是物理上不可避免的**，不能通过"用 fp8 GEMM"绕开。

因此 4096 token prefill 的**现实最优**应按下式估算（每层）：

| 步骤 | 流量 |
|---|---|
| DMA fp8 上卡 | 1.41 GiB |
| 解码 + 转置（**融合成一趟**）| 1.41 读 + 2.82 写 |
| GEMM 读 bf16 权重 | 2.82 GiB |
| **合计** | **~8.5 GiB/层** |

按有效带宽 ~100 GB/s 估 ⇒ ~85 ms/层 ⇒ 48 层 ≈ **4 s**（4096 token ≈ 1000 tok/s）。
这才是本硬件上"算法最优"的目标，而 2.84 s 那个纯 DMA 下限**不可达**（因为要多走解码）。

⇒ 优化方向明确为：
1. **解码与转置融合成一趟**（当前是两趟：转置 1.41+1.41 + 解码 1.41+2.82）；
2. **DMA 提到 20+ GB/s**（当前 ~9）；
3. **热层常驻**（整层跳过 1.41 GiB DMA 与解码）。

---

## 20. 【严重】int8 GEMM **在软件栈中完全不可用**（硬件规格宣称支持 int8 / 200 TOPS）

> 记录日期：2026-10-25；环境同文首（MUSA 5.2 / torch 2.11.0.post1 / torch_musa 2.11.0.post1+b431668 / MTT S4000×2）

### 现象

**任何路径都无法执行 int8 矩阵乘**（int8×int8→int32，即量化 GEMM 的标准形态）：

| # | 调用 | 结果 |
|---|---|---|
| 1 | `torch.bmm(int8, int8)` | `RuntimeError: BmmCall MUDNN failed in: Run` |
| 2 | `torch.mm(int8, int8)` | `RuntimeError: MmCall MUDNN failed in: Run` |
| 3 | `torch.matmul(int8, int8)` | 同上 |
| 4 | `torch._int_mm(int8, int8)` | 同上，且 muDNN 打印 **`ERROR# NOT_SUPPORTED in BatchMatMul::RunLt`** |
| 5 | `mublasGemmEx(A=i8,B=i8,C=i32,computeType=MUBLAS_COMPUTE_32I)`（ctypes 直调 `libmublas.so`）| **`rc=8`**（`MUBLAS_STATUS_NOT_SUPPORTED`）|

### 最小复现

```python
import torch, torch_musa
a = torch.randint(-127, 127, (256, 2560), dtype=torch.int8, device="musa:0")
b = torch.randint(-127, 127, (2560, 768), dtype=torch.int8, device="musa:0")
torch._int_mm(a, b)     # ⇒ NOT_SUPPORTED in BatchMatMul::RunLt
```

### 我们的定位（代码级证据，供厂商直接定位）

1. muDNN 的 BatchMatMul **最终走 `RunLt`**（错误信息给出了函数名）⇒ 依赖 muBLASLt 的 int8 内核；
2. `nm -D /usr/local/musa/lib/libmublasLt.so` 的内核族**只有**：
   `bf16 / f16 / f16f32 / tf32 / fp8(e4m3e4m3fp16, e5m2e4m3fp16, e4m3e5m2fp16 …)`
   —— **没有任何 int8 内核**；
3. `libmublas.so` 里 981 个含 `i8/s8` 的符号**全部是 TRSM 的模板参数**，不是 int8 GEMM 内核；
4. `MUDNN_DATA_INT8 = 3` 类型枚举**确实存在**（`mudnn_graph.h:235`），
   `MUBLAS_COMPUTE_32I = 72`（`internal/mublas_types.h:229`）也存在
   ⇒ **"类型/枚举存在" ≠ "GEMM 支持"**；推测 int8 目前只覆盖 **卷积/NPU** 路径。

### 期望（请厂商明确回答）

1. **int8 的支持范围**：是否仅限卷积？GEMM（含 batched）有没有计划/已支持但入口不同？
2. 若有：请给 **≥ 5.3 SDK 的 int8 GEMM 调用示例**（`mublasGemmEx` 或 `mudnnBatchMatMulBias` 均可），
   我们愿意立刻验证并回填结果；
3. 若硬件确实不支持 int8 GEMM，**请在规格文档中写明"int8 仅限卷积"**，避免"规格有 int8、实际只有卷积"的误解。

### 对我们的影响（量化）

* MoE prefill 的权重是 FP8；**因无 int8/fp8 GEMM，必须在设备上把权重解码成 bf16**：
  **每层多 62 ms**（层时 570 ms 的 **11%**），并多写 **2.82 GB/层** 的 bf16 中转；
* 若 int8 GEMM 可用，可同时获得：**免解码 + int8 张量核 + 8-bit 存储不变**。

> ⚠️ 我们不认为这是"规格虚标"：更可能是 **int8 只覆盖卷积**，或需要更新 SDK。
> 本条与第 **19** 条（fp8 GEMM 数值不可用）是同一族的"GEMM 数据类型支持"问题，建议一并答复。

---

## 21. 【低 · 待自查】H2D 实测带宽只有链路理论的 50–67%

### 现象

* 链路：`LnkCap 32GT/s x16`（PCIe 5.0 能力），`LnkSta 16GT/s x16`（实际 4.0 x16，`downgraded`）
  ⇒ 单向理论 **≈31.5 GB/s**；
* 实测 pinned H2D：**15.6–22.6 GB/s**（`musaMemcpy2DAsync`；1D/2D、默认流/新建流/并发、
  `musaHostRegister` 均试过，都封顶在这个区间）。

### 请厂商确认

1. 是否有**更快的 H2D 路径**（专用 DMA 引擎 / 其它 flag / 批量提交语义）？
2. 是否与 **IOMMU / NUMA 绑定**有关（我们尚未把 pinned 内存绑到 GPU 所在 NUMA 节点做对照）？
3. 两张卡**并发** H2D 时聚合带宽的**设计值**是多少（我们怀疑共享上游，正在验证）？

> 状态：**先自查 NUMA/IOMMU 与并发**，拿到可信数据后再提交，避免误报。

---

## 22. 【已知问题 · 决定不追修】**S4000 的 Triton `tl.dot` 不走张量核**（见 §18）

> 记录日期：2026-10-25（用户决策）

### 决策
* **不在 S4000 上追这个问题的修复**，仅作为**已知限制**记录（§18 的 143× 慢）；
* 理由：①厂商当前主推 **S5000**，交付与软件支持压力大，**预期不会回头修 S4000 的 Triton**；
  ②我们后续会**升级到 S5000**；③为绕开它去写定制内核，在换代前是**负资产**。

### ⭐ 端到端量化（2026-10-25 实测补充，可直接用于会议）

| 项 | A100 主线 | **S4000（实测）** | 差距 |
|---|---|---|---|
| 官方 qfn prefill | **1,517 tok/s**（TP=2 · MBT 8192 · 12 层常驻）| **101.1 tok/s**（TP=2 · MBT 4096 · 无驻留）| **15.0×** |
| 每层每 1K token | ≈13.7 ms | **≈141 ms** | **10.3×**（原写 12.1× 已作废：分母误用 `e2e7` 的块 3072，基线 `e2e8` 实为 4096）|
| 15K 预填 | ≈10.1 s | **151.9 s** | 15.0× |
| 15K 起 TTFT | ≈10.1 s | **151.9 s** | 15.0× |
| decode @C=1 | **46.3 tok/s** | **1.8 tok/s**（无驻留）| 25.7× |
| decode @C=2 | 92.9 tok/s | **3.0 tok/s**（无驻留）| 31.0× |

⚠️ **口径说明（必须同时给出，否则数字会被误用）**：
主线的准确口径（`README.md:69`，已核对原文）：
`TP=2 · util 0.90 · MAXLEN 262144 · MBT 8192 · MTP 关 · 12 层精选常驻 · FULL_DECODE_ONLY`
⇒ **与 MUSA 行有三处口径差**，逐条标出：

| 差异 | A100 主线 | MUSA 实测 | 对可比性的影响 |
|---|---|---|---|
| **MBT** | 8192 | 4096（**实测切块 4096**）| ⛔ **原写"切块 3072"有误** ⇒ 两边块一致，**无切块混淆**；MBT 8192 补测后 prefill 仅 +14.7%（分块只解释 1.15×）（补 MBT=8192 可分离）|
| **`gpu_memory_utilization`** | 0.90 | **0.72**（无驻留时的可用值）| 只影响 KV 容量，**不影响 prefill 速度** ⇒ 对 15× **无影响** ✓ |
| **层常驻** | **12 层** | **无** | prefill 是否受影响**尚未测准**（旧结论已撤回，QA #321）⇒ 暂按"未知"处理，不计入 15× ✓ |

⇒ ⭐ 因此 **15× 中"确定属于平台能力"的部分 = 每层每 token 计算效率（10.3×）**，
剩余 1.24× 来自分块差异（MBT 4096→8192 可测）；**常驻的影响未知，暂不归因** ✓
* ⭐ **可确证的部分**：两者 prefill **聚合吞吐都不随并发提升**（C=1→C=2 分别为 101.1→100.8 与
  主线的"预填串行"），说明**瓶颈同为每 expert 的 GEMV 计算效率**，即 `tl.dot` 不走张量核的后果 ✓

### 对我们的约束（写给后续会话）
* ⛔ **不为 S4000 做深度专用优化**：定制 Triton/MUSA 内核、只对 S4000 有意义的私有格式、
  依赖 `tl.dot` 修复才成立的方案 —— 一律不做；
* ✅ **只做可移植改动**：流水/调度结构、布局与 ring 设计、主机侧去同步、搬运次数与流量、
  专家分块策略 —— 这些在 S5000 上同样成立；
* 📌 若将来在 S5000 上发现 `tl.dot` 已走张量核，**届时再重估**"融合 dequant+GEMM"这条路线
  （理论上能把每层 7.55 GB 权重流量降到 1.51 GB）。

---

## 23. 【会议澄清清单】需与厂商当面确认的问题（**我方不再自行排查**）

### 23.0 ⭐ 一页纸：**四个**已量化的平台问题（可直接投屏）

> 全部为 **2× MTT S4000 / TP=2 / vLLM 0.32.0 / torch_musa 2.11.0** 实测，
> 复现命令见 `dev-docs/MUSA_S4000_BENCH_REPORT_2026-10-25.md` §8.1。
> 我们**不要求厂商改我们的代码**，只请确认这三条是否在 MP31 上改善。

| # | 现象 | 我们的实测 | 参照值 | 我们想要什么 |
|---|---|---|---|---|
| **1** | **Triton 字节级 stride 转置极慢** | `ktranspose_into` 转 K-major：**6.5 GB/s**（每层 230.9 ms）| 我们自己在 A100 上同内核 **~228 GB/s**（代码注释记录）⇒ **差 35×** | ① 确认是否为 MP22 Triton 的已知限制 ② MP31 是否改善 ③ 若有推荐的 byte-transpose 写法/内置算子 |
| **2** | **H2D 有效带宽远低于峰值** | 每层搬 0.755 GB，`dma` 计时 126.7 ms ⇒ **6.0 GB/s**；单笔大块实测峰值 **~20 GB/s** | PCIe 4.0 x16 理论 ~25 GB/s | ① 小批量/多笔传输的固定开销有多大 ② 是否有 pinned-host 直读 / RDMA 替代通道 ③ MP31 是否改善 |
| **3** | **Triton `tl.dot` 不走张量核** | 该问题**仍存在，但已不是主项**：GPU 计算只占每层 **4%**（29.2 ms / 740 ms）| A100 同口径每层 ≈105 ms（推算）| 不必优先回答；**但若能在 MP31 修正，请告知** —— 它会影响我们把"融合 dequant+GEMM"作为长期路线 |

| **4** | **FlashAttention-3 在此架构不可用，回退 SDPA** | 日志：`FA3 unavailable on this arch, falling back to SDPA (InternalError: Check failed: (dprops.major > ...))`；MUSA 上**每卡都出现** | 期望：官方支持的 FA 版本 | ① 官方**支持哪个 FA 版本**（FA2/FA3/自研）② SDPA 回退对 15K 长 prefill 的注意力部分损失多少 ③ MP31 是否支持 FA3 |

**⚠️ 我们对这三条的诚实边界**
* 第 1 条的 228 GB/s 是**代码注释里的 A100 历史值**，我们未在 MUSA 上复测同一基准 ⇒ 只能作量级参照；
* 第 2 条的 0.755 GB/层是**按全部专家估算的上界**，实际按 tile 分批搬，字节数未逐笔核对；
* 第 3 条我们**主动降级**了它的重要性（原以为是主因，实测只占 4%）⇒ 不希望厂商把资源投在这里；
* 第 4 条（FA3 回退 SDPA）**尚未量化其损失**，只作为"请确认官方支持哪个 FA 版本"的提问项；
* 四项之外还有 **48% 的每层耗时未能归因**（我们的插桩不够细）⇒ 这一项**是我们自己的待办**，不推给厂商 ✓

> 2026-10-25 用户决定：**S4000（MP22）线以文档收尾**；下列问题**不再由我方逆向排查**，
> 统一拿到与厂商的会议上澄清。整理原则：每条给出**我方已掌握的实测证据**，便于对方一次定位。
> 关键前提：**MP22 上"该有却没有"的能力，很可能在 MP31（S5000）已实现** —— 会议目标是**确认版本差异**，
> 而不是要求对方为 MP22 补功能。

### A. 计算能力（GEMM / 数据类型）

| # | 问题 | 我方证据 | 我们要的答复 |
|---|---|---|---|
| A1 | **int8 GEMM 的支持范围**：只支持卷积？还是 GEMM 也有入口？ | `torch.bmm/mm/matmul/_int_mm(int8)` 全 `NOT_SUPPORTED`；`mublasGemmEx(int8,compute=32I)` `rc=8`；muDNN 日志 `NOT_SUPPORTED in BatchMatMul::RunLt`；`libmublasLt.so` 无 int8 内核（见 §20）| MP22 / MP31 各支持到哪一步？若 MP31 支持，给调用示例 |
| A2 | **"200 TOPS int8" 的口径**：是卷积峰值还是 GEMM 峰值？ | 同上；`MUDNN_DATA_INT8` 类型存在但 GEMM 不可用 | 明确指标口径，避免误读 |
| A3 | **fp8 计算**：为什么库里有 `e4m3e4m3fp16` 等内核符号，运行期却报"当前 GPU 架构不支持"？ | `mublasGemmEx(E4M3→F16)` 触发 `NOT_SUPPORTED ... The data type is not supported on current GPU architecture`；但 `libmublasLt.so` 有 `gemmex_..._e4m3e4m3fp16_tce_...`（见 §19/§22）| 那些符号是给其它型号编译的吗？S4000 有无任何 fp8 计算单元？MP31 呢？ |
| A4 | **Triton `tl.dot` 不走张量核**（慢 143×，§18） | 已实测 | MP31 的 Triton 是否已走张量核？预计哪个版本修？ |

### B. 数据搬运（H2D）

| # | 问题 | 我方证据 | 我们要的答复 |
|---|---|---|---|
| B1 | **H2D 只跑到链路理论的 50–67%** | 链路 `LnkSta 16GT/s x16`（理论 31.5 GB/s）；实测 pinned 15.6–22.6 GB/s，各种 API/流形态都封顶（§21）| 是否还有更快的 H2D 路径？是否与 IOMMU/NUMA 绑定相关？ |
| B2 | **两卡并发 H2D 的聚合带宽设计值** | 怀疑共享 PCIe 上游（待测，我方已有探针）| 给出设计值与推荐拓扑（是否应插到不同 root port） |
| B3 | **芯片间/卡间互联**（若要做 TP=2 的 AllToAll/AllGather） | 我方当前仅在单机内 TP=2 | 推荐路径与实测带宽 |

### C. 稳定性与运行时

| # | 问题 | 我方证据 | 我们要的答复 |
|---|---|---|---|
| C1 | **固件 panic 三类形态**（`CSW_TIMEOUT` / `DMUP_REASON 0x11` / `KILL_TIMEOUT`） | 已在 §14/§15/§17 记录多种触发条件（H2D 失败、SIGKILL 正在跑内核的进程等）| 哪些是已知 bug、哪个版本修？SIGKILL 触发 panic 是否可避免？ |
| C2 | **`torch_musa` 未接上 graph / stream 能力**（SDK 有、绑定缺失，§2） | 已在 §2 记录 | MP31 的 torch_musa 是否补齐？ |
| C3 | **`_C` 部分算子未注册 MUSA(PrivateUse1) dispatch**（§3） | 已在 §3 记录 | 补齐计划？ |

### D. 我们要在会议上拿到的三样东西

1. **MP31/S5000 的能力对照表**：上面 A/B/C 各项在 MP22 vs MP31 的状态（支持 / 不支持 / 计划版本）；
2. **int8 GEMM 与 fp8 GEMM 的官方调用示例**（若有）；
3. **推荐部署拓扑**（PCIe 插法、NUMA 绑定、TP 通信路径）与**已知 bug 的版本修复计划**。

> ⛔ 我方行动约束（2026-10-25 用户明令）：**不再自行逆向排查上述库行为**；
> 后续只做"能跑通 + 文档 + 会议澄清"这条线。

### 补充：decode 吞吐不随并发提升（2026-10-25 实测，可供会议讨论）

无层常驻、GPU prefill on、15K prompt、greedy、固定输出 64 token：

| C | 解码步长 (ms) | 聚合 decode (tok/s) | 单步理论 H2D（按 10 专家/层估算）|
|---|---|---|---|
| 1 | **568.69** | 1.8 | ≈66 ms |
| 2 | **658.66** | 3.0 | ≈132 ms |
| 4 | **1398.06** | 2.9 | ≈264 ms |

**现象**：步长随并发近似线性上升（568 → 659 → 1398 ms），所以 **C=4 的聚合解码吞吐（2.9 tok/s）与 C=1（1.8）几乎无改善**。

**我们的解读（待厂商确认）**：无驻留时每个解码步要为 48 层搬 top-10 专家；批内请求越多、命中的不同专家越多（10 → 最多 40），
H2D 流量随 C 近线性增长，而 HBM 有效带宽（实测 15.6–22.6 GB/s）不足以吸收，形成**逐 token 的串行搬运瓶颈**。

### 请厂商确认
1. S4000 上 `P2P`/小批量 H2D 的实际可达带宽随**传输次数**下降多少？我们实测单笔大块约 20 GB/s、大量 4–64 KB 小传输明显更低。
2. 是否有 **RDMA/GPUDirect 或 pinned host 直读** 的替代通道，能绕开逐笔 `cudaMemcpyAsync` 的固定开销？
3. 该瓶颈在 **MP31** 上是否有改善（如更低的 launch 延迟、批量 copy 引擎）？

**参考数据（同一台机、同一口径，n=2238 配对样本）**：GPU 计算 `gpf-ph total` 中位 24.9 ms，H2D DMA 中位 106.5 ms ⇒ H2D 约为计算 4.3×。
decoding 步长随并发单调变长：C=1 568.69 ms / C=2 658.66 ms / C=4 1398.06 ms（每多一个请求就多一份选中专家的 H2D 流量）。

### 23.5 ⚠️ NVFP4 加载探针的**误读风险**（跑之前必须先分清两种失败）
**模型体积实测**（`/root/models/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4`）
| 部分 | 大小 |
|---|---|
| `model-plefp8-*.safetensors`（10 分片，**PLE/ngram 表**）| **48 GB** |
| 其余（专家 NVFP4 + attention FP8 + MTP）| **73 GB** |
| 合计 | **121 GB** |

**两种失败的判据（必须分开写，否则结论会错）**
| 现象 | 含义 | 结论 |
|---|---|---|
| `LOADED OK` + 输出非空 | vLLM 的 modelopt 路径能吃这个 checkpoint | ✅ **NVFP4 在 MUSA 可加载** |
| OOM 且 `Tried to allocate` ≈ **48 GB** 或分配点在 embedding/PLE | **PLE 表被当成普通权重上卡**（我们的插件被禁用后无人把它放宿主）| ⛔ **与 NVFP4 无关**，是**显存布局问题** ✗ |
| OOM 且分配点在 MoE 专家 | 专家 NVFP4 反量化膨胀 | ⚠️ 需看具体算子 |
| 其它异常（dtype/量化方法不认识）| **真正的 NVFP4 不支持** | ⛔ 记 ISSUE |

**为什么这条必须写清**
探针**故意禁用我们的插件**（让 MoE 走 vLLM 自己的 modelopt 路径，避免我们的钩子按 FP8 几何去接 NVFP4
而掩盖真因）—— 但副作用是：**插件里"PLE 表放宿主"的逻辑也一并失效** ✓
⇒ 于是 48 GB 的表可能被尝试放上卡（每卡 48 GiB，util 0.90 预算 43.1 GiB）⇒ **必 OOM** ✗
⇒ 若不做这个区分，就会把"显存布局问题"误报成"厂商 NVFP4 不支持" —— 那是**对平台的不公评价** ✗
