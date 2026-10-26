# MUSA 构建陷阱清单（PITFALLS）

> 记录踩过的、**不查二进制就发现不了**的构建/移植坑。每条都要有：现象、根因、诊断命令、预防。
> 事实来源一律是实机（`<musa-host>`，MTT S4000，MUSA Toolkit 5.2.0）。

---

## P1. **源码/声明里有，编译产物里没有**：op 有 schema、没有 kernel

### 现象（2026-10-22）

qfn 端到端在**第一个 decode step** 崩，报错是 Triton 的：

```text
vllm/v1/worker/gpu/block_table.py:162  gather_block_tables
RuntimeError: Triton Error [MUSA]: an illegal memory access was encountered
```

追下去发现 `_gather_block_tables_kernel` 用 `num_blocks`（UVA 主存张量）决定读多少行；
GPU 侧读到的是垃圾 ⇒ 越界读。而 UVA 的设备视图来自：

```python
torch.ops._C.get_cuda_view_from_cpu_tensor(pinned_cpu_tensor)
```

这个调用报的是：

```text
NotImplementedError: Could not run '_C::get_cuda_view_from_cpu_tensor'
  with arguments from the 'CPU' backend.
```

⚠️ **关键区别**：这不是"没有这个算子"（那会报 `AttributeError`），而是
**schema 存在、kernel 没注册**。也就是说**代码写了、也编了，但注册没进二进制**。

### 根因

| 文件 | 作用 | 实际情况 |
|---|---|---|
| `csrc/musa_cuda_view.cu`（76 行） | 实现：`musaHostGetDevicePointer()` 取主机内存的设备指针 | 在 `vllm_musa/setup.py` 的 `VLLM_CSRC_SOURCES` 里 ✅ |
| `csrc/musa_torch_bindings.cpp`（15 行） | 注册：`TORCH_LIBRARY_FRAGMENT(_C, mcv) { mcv.impl("get_cuda_view_from_cpu_tensor", torch::kCPU, …) }` | 同上一行也列了 ✅ |
| 已安装的 `vllm_musa/_C.cpython-310-*.so` | 应该包含上面两者 | `strings` 里**只有** `_C_musa_ops`，**没有** `get_cuda_view_from_cpu_tensor` ❌ |

⇒ **setup.py 里列了源文件 ≠ 装到 site-packages 里的 `.so` 里有它**。
典型成因：`.so` 是加源文件**之前**构建的；或者构建走了另一条路径（另一个 target），
或者增量构建复用了旧对象文件。

### 诊断命令（三条，缺一不可）

```bash
# 1) 区分"没有算子" vs "有 schema 没 kernel"
python -c "
import torch, vllm
t = torch.zeros(8, dtype=torch.int32, pin_memory=True)
try: torch.ops._C.get_cuda_view_from_cpu_tensor(t); print('OK')
except AttributeError as e: print('NO SCHEMA:', e)
except NotImplementedError as e: print('NO KERNEL:', e)"

# 2) 二进制里到底有没有这个 op 名/注册符号（最可靠）
strings -a <path>/_C*.so | grep -c get_cuda_view_from_cpu_tensor
nm -D --defined-only <path>/_C*.so | grep -i get_cuda_view

# 3) 这个 .so 是哪个 target 编出来的（名字/命名空间会对不上）
ls -l <path>/_C*.so        # 时间戳 vs 源文件时间戳
strings -a <path>/_C*.so | grep -E "^_C"
```

⚠️ **不要用 `dir(torch.ops._C)` 判断**：torch 的 `_OpNamespace` 是惰性的，
`dir()` 只返回 `['name']`，无论注册了多少算子。必须用 `hasattr`/实际调用。

### 预防（可机械化，别靠记性）

1. **构建后必须跑一次"算子覆盖率扫描"**：扫源码里所有 `torch.ops._C.<name>` 引用，
   逐个 `hasattr` 验证，输出 MISSING 清单（脚本：`scripts/musa/scan_ops.py`）。
   源码引用数 > 0 而注册数 = 0 的情况，光看构建日志是发现不了的。
2. **验收必须验"内存 alias"，不能只验"不报错"**：

   ```python
   t = torch.zeros(8, dtype=torch.int32, pin_memory=True); t[0] = 7
   v = torch.ops._C.get_cuda_view_from_cpu_tensor(t)
   assert v.data_ptr() == t.data_ptr() and int(v[0]) == 7   # 同一块内存才算真 UVA
   ```

3. **改了 `setup.py` 的源文件列表后，先删构建缓存再编**（`build/`、`*.egg-info`、
   `~/.cache/torch_extensions/*`），否则增量构建会悄悄复用旧对象。

### 泛化判据

* **"代码里有"和"进程里有"是两件事**：凡是"注册/绑定/插件"这类机制，
  验收点必须在**运行时注册表**，不在源码、不在编译日志、不在 `nm` 输出。
* 报错信息要读**类型**：`AttributeError`=没有 schema（没编进去/没链接），
  `NotImplementedError ... with arguments from the 'X' backend`=**有 schema 没该后端的 kernel**
  （注册了但 dispatch key 不对，或注册代码没进二进制）。这两条指向完全不同的修法。
* 这类坑的代价极高：它伪装成"GPU 硬件非法访存"，把人往内核/驱动方向带，
  实际是**构建产物不完整**。以后遇到"内核读到垃圾指针"，先怀疑**指针来源的注册是否完整**。

### P1 已修复（2026-10-22 11:26）

`cd /root/vllm-musa && python setup.py build_ext --inplace` 重建后，`vllm._C` 与
`third_party/vllm/vllm/_C` 两个 `.so` 都更新（`strings` 里该 op 名 13 处），并且：

```text
>>> t = torch.zeros(8, dtype=torch.int32, pin_memory=True); t[0] = 7; t[1] = 42
>>> v = torch.ops._C.get_cuda_view_from_cpu_tensor(t)
OK device= musa:0 | alias= True | v0= 7 v1= 42      # 同一块内存 ⇒ 真 UVA
```

⚠️ 两个必须记住的细节：
1. **必须先 `import vllm._C`**（`import vllm` 不会隐式注册）；引擎里由 vLLM 自己触发；
2. `dir(torch.ops._C)` 在**注册成功之后仍然只返回 `['name']`** ⇒ 再次证明不能用它判断。

---

## P2. mcc 默认挑**最新** gcc 的头 ⇒ gcc-15 的 `<array>` 在 `__host__ __device__` 里报错

### 现象

```
/usr/lib/gcc/x86_64-linux-gnu/15/../../../../include/c++/15/array:210:2:
  error: reference to __host__ function '__glibcxx_assert_fail' in __host__ __device__ function
```

触发链：`torch/headeronly/core/DeviceType.h` → `<functional>` → `<array>`（gcc-15 版）。
gcc-15 在 `_GLIBCXX_ASSERTIONS` 下把 `__glibcxx_assert_fail()` 用在 `__host__ __device__` 语境里。

### 试过但**无效**的招

* `NVCC_PREPEND_FLAGS="-U_GLIBCXX_ASSERTIONS -D_GLIBCXX_ASSERTIONS=0"` —— mcc 不认这个 env（命令行里 0 次出现）；
* 直接把 `-U_GLIBCXX_ASSERTIONS -D_GLIBCXX_ASSERTIONS=0` 加进 mcc 参数 —— **仍报同样 9 个错**（该宏在头里，命令行改不动）；
* `--gcc-toolchain=/usr` —— 仍解析到最新（15）；
* `--gcc-install-dir=...` —— 这个 MTCC 5.2（clang 14 分支）**不支持**。

### 有效做法：给 mcc 一个**只含 gcc-11 的伪前缀**

```bash
mkdir -p /root/gcc11/lib/gcc/x86_64-linux-gnu /root/gcc11/include/c++ /root/gcc11/include/x86_64-linux-gnu/c++
ln -sfn /usr/lib/gcc/x86_64-linux-gnu/11          /root/gcc11/lib/gcc/x86_64-linux-gnu/11
ln -sfn /usr/include/c++/11                        /root/gcc11/include/c++/11
ln -sfn /usr/include/x86_64-linux-gnu/c++/11       /root/gcc11/include/x86_64-linux-gnu/c++/11
```

然后在 `setup.py` 的 `MCC_FLAGS` 里加 `--gcc-toolchain=/root/gcc11`
（已实现为 `XTU_MUSA_GCC_TOOLCHAIN`，默认 `/root/gcc11`）。

验证：同一个失败文件单编 **rc=0、0 error**；全量重建 **0 error**。

**判据**：`--gcc-toolchain` 的取值是**前缀**（含 `lib/gcc/<triple>/<ver>`），
不是版本目录本身；而且 clang 会在该前缀下挑**版本号最大**的那个 ——
所以想让 11 生效，前缀里就**不能有** 14/15。

---

## P3. 「点名」结果：509 个引用里 **494 个在 MUSA 上不存在**

`scripts/musa/verify_c_extensions.py` 扫描 `vllm/**/*.py` 里全部 `torch.ops.<ns>.<name>` 引用，
逐个运行时验证（原始输出：`dev-docs/musa_opscan_2026-10-22.log`）：

```text
python refs: 509   csrc registrations: 255   .so scanned: 3
❌ MISSING（Python 引用了，运行时没有）: 494
⚠️  ORPHAN（源码注册了，二进制里没找到）: 1
```

按来源文件聚类（前几名）：

| 缺失数 | 文件 | 判断 |
|---:|---|---|
| 113 | `_custom_ops.py` | 大量是 CUDA/ROCm 专用（`awq_dequantize`、`wvSplitK*`、`paged_attention`…），走平台分支才会调 ⇒ **未必可达** |
| 15 | `model_executor/layers/activation.py` | ⚠️ **需要逐条确认**：激活函数可能在热路径 |
| 9 | `utils/cpu_triton_utils.py` | CPU 路径 |
| 8 | `_xpu_ops.py` | XPU 专用，不可达 |
| 8 | `models/kimi_k3/...` | 非本模型，不可达 |
| 7 | `distributed/device_communicators/cpu_communicator.py` | CPU 通信 |

⇒ **结论**：`_C` 在 MUSA 上是个**极薄的子集**（15/509）。这不是立刻要修完的清单，
而是**风险地图**：真正致命的是"**热路径上调到缺失 op**"——它的报错会是
`AttributeError: '_OpNamespace' '_C' object has no attribute 'xxx'`，与 P1 同族但更早暴露。

**用法（每次重建后必跑）**：

```bash
python scripts/musa/verify_c_extensions.py --vllm-root /root/vllm-musa/third_party/vllm
# 退出码 3 = 有 MISSING；有新增 MISSING 就说明构建又漏了东西
```

**去噪办法**：把"已知不可达"（ROCm/XPU/CPU/kimi 等平台分支）加进白名单，
让 MISSING 收敛到"我们这条线真的可能调到的"；这笔账等 qfn 跑通后按实际 traceback 逐条销。

---

## P4. `setup.py build_ext --inplace` 会触发一次 **pip 卸载+重装**，把之前装好的兄弟扩展一起删掉

### 现象

重建 UVA 之后，引擎启动到加载权重时报：

```text
NotImplementedError: Qwen4Exp QSA requires FlashAttention
```

而重建之前同一个用例能加载到 48 层。检查发现 **FlashAttention 扩展消失了**：

```text
import vllm.vllm_flash_attn
ImportError: vllm.vllm_flash_attn requires the CUDA flash attention extensions
             (_vllm_fa2_C or _vllm_fa3_C)
find site-packages -name '_vllm_fa2_C*'  → 空
```

### 根因

`/root/vllm-musa/setup.py` 在构建末尾会走一次包安装流程（日志里可见）：

```text
Removing file or directory /root/venvs/vllm310/lib/python3.10/site-packages/vllm
Successfully uninstalled vllm-0.32.0
Successfully installed vllm-0.32.0
```

这次重装只装**本仓 `EXT_MODULES` 里那 4 个扩展**
（`vllm._C` / `vllm._C_stable_libtorch` / `vllm_musa._C` / `vllm._moe_C`），
而之前**手工/预编译装进去的 FlashAttention 二进制**（`_vllm_fa2_C`、`_vllm_fa3_C`）不在其中 ⇒ 被清掉。

### 教训（重要）

* **"重建一个扩展"不是局部操作**：如果构建脚本顺带装了包，它会以"本仓能构建的东西"为准，
  覆盖 site-packages 里**更完整**的既有安装 ⇒ 重建前要**确认清单**，重建后要**逐项点名校验**（见 P3）。
* 顺序建议：**先备份 `site-packages/vllm` 下所有 `*.so` 清单**（`ls -1 > /tmp/so_before.txt`），
  重建后 `diff`；少了的必须补回来，否则会在很远的地方（这里是 QSA→FA）才炸。
* 具体到这台机器：重建后必须确认 `_vllm_fa2_C`/`_vllm_fa3_C` 仍在，否则
  `Qwen4Exp QSA requires FlashAttention` 会挡住整个 qfn 端到端。

### P3.1 为什么"MISSING 494"不等于"坏了 494 个"

三条硬证据（本机实测）：

1. **`vllm._C` 在 MUSA 上只由 2 个源文件构成**（`vllm_musa/setup.py` 的 `VLLM_CSRC_SOURCES`）：

   ```text
   csrc/musa_cuda_view.cu        # 就一个 CPU->device 视图
   csrc/musa_torch_bindings.cpp  # 它的注册
   ```

   ⇒ `_C` 在这里**本来就是一层薄壳**。上游早把绝大多数算子搬到了
   `_C_stable_libtorch`（另一个扩展）和 Python/Triton 实现里。

2. **引用点大多在平台分支里**：`_custom_ops.py` 有 **12 处** `is_rocm/is_musa/is_xpu/is_cuda` 分支，
   `activation.py` 有 **21 处**。MUSA 不会走进 CUDA/ROCm/XPU 那几支 ⇒ 那些名字**永远不会被解析**。

3. **判定"缺"只看了一个命名空间**：扫描器查的是 `torch.ops._C.<name>`；
   而同一批算子可能在 `torch.ops._C_stable_libtorch`、`torch.ops._moe_C` 或 `torch.ops.vllm` 下。

⇒ 所以正确的读法是：**494 是"名字表差异"，不是"494 个功能坏了"**。
真正的风险面 = **热路径上、且不在任何平台分支里、且其他命名空间也没有** 的那些（预计个位数）。

### P3.2 去噪规则（下一步要做的）

1. 扫描器扩展成**跨命名空间**：对每个 `torch.ops._C.X`，依次尝试
   `_C` / `_C_stable_libtorch` / `_moe_C` / `vllm`，任一命中即视为"有"；
2. 对仍 MISSING 的，检查**引用行是否在 `if is_cuda/is_rocm/is_xpu(...)` 分支内**
   （按缩进 + 最近的平台判断即可，粗糙但足够），在分支内 ⇒ 记为 `UNREACHABLE`，不进告警；
3. 剩下的是 `RISK`，只有它会真正炸；把它压到个位数后再逐条销账。

### P3.3 这类坑的通用记法（"随手记"的清单）

遇到**"代码里有、运行时没有"**的任何东西，都按下面四问记录，别只记结论：

| 问 | 本次答案 |
|---|---|
| 谁声明/注册的？ | `csrc/musa_torch_bindings.cpp`（15 行） |
| 编进去了吗？ | 源码在 `VLLM_CSRC_SOURCES`，但**构建产物里没有**（P1 的真因） |
| 运行时从哪查？ | `torch.ops._C.<name>`（**不能**用 `dir()`，它恒为 `['name']`） |
| 怎么防复发？ | 构建后跑 `scripts/musa/verify_c_extensions.py`（退出码 3 = 有缺失） |

同类已记录：**P4**（重建触发 pip 重装，把 FlashAttention 扩展删了）也是这条的变体 ——
**"我以为只重建了一个模块，实际动了一整包"**。

### P4.1 现场（2026-10-22 12:00）与恢复线索

* 报错点：`vllm/models/qwen4_exp/amd/qsa.py:107`
  `if not is_flash_attn_varlen_func_available(): raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")`
  —— **不是"没有 FA 后端"，而是"varlen 函数不可用"**。
* 树内 `vllm/vllm_flash_attn/flash_attn_interface.py` 通过
  `from . import _vllm_fa2_C / _vllm_fa3_C` 判定 `FA2_AVAILABLE/FA3_AVAILABLE`；这两个 `.so` 现在都不存在。
* 但**厂商的 MUSA FA 包还在**：`pip list` 有
  `flash_attn_3 0.2.7+musa`、`flash_mla 0.2.7+musa`、`flashinfer-python 0.2.7+musa`
  （来自 MTHREADS 私有源，见 `/root/build_vllm310.sh` 的 `requirements/musa_private.txt`）。
* 环境构建方式是 `pip install -e . --no-build-isolation`（`/root/build_vllm310.sh:10`）
  ⇒ 本仓 `setup.py` **只构建 4 个扩展，不含 FA**。

⇒ 恢复方向（按代价排序）：
1. **让 shim 指向厂商包**：把 `vllm/vllm_flash_attn` 的可用性判定改为"能用 `flash_attn_3` 即可"
   （最小改动，且符合"MUSA 用厂商 FA"的现实）；
2. 或找回/重建 `_vllm_fa2_C`／`_vllm_fa3_C`（需要 FA 源码树，本机**没有**）。
3. ⚠️ 无论哪条，**改完必须重跑 `verify_c_extensions.py`** 并确认 QSA 那行不再抛。

---

## P5. UVA 打通后的下一个坑：Triton `str_to_ty` 认不了带字节序的 dtype 串（`KeyError: 'u8>'`）

时间线（2026-10-22 12:17）：

1. FA 可用性修好后（P4.1 方案 1：认厂商 `flash_attn_3`，`varlen_available: True`），
   引擎能过 QSA 前置检查、开始加载；
2. 随后崩在 **UVA 查表内核**：

```text
File ".../vllm/models/qwen4_exp/common/ngram_embedding.py", line 494, in _lookup
  _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
    self._uva_weight.view(torch.uint8), ...)
File ".../triton/runtime/jit.py", line 640, in run
File ".../triton/compiler/compiler.py", line 101, in make_ir
File ".../triton/language/__init__.py", line 294, in str_to_ty
KeyError: 'u8>'
```

**判读**：`'u8>'` 是"无符号 8 位 + 字节序后缀 `>`"的字符串，Triton 的 `str_to_ty`
只认 `u8`/`i8`/`fp16` 这类裸名 ⇒ 传入的 dtype 描述带了字节序标记。
在本机这条路上，`self._uva_weight` 是 **设备视图**（`from_blob`/stable 视图），
其 `dtype` 字符串可能保留了 numpy 风格的字节序。

**下一步（未做）**：
1. 打印 `self._uva_weight.dtype`、`.view(torch.uint8).dtype` 与 Triton 实际收到的注解字符串，
   确认 `>` 从哪来；
2. 修法二选一：给内核传**裸 u8 名字**（显式 `tl.uint8`/`"u8"`），或把视图张量的 dtype
   规整成不带字节序的形式；
3. 修完必须复跑 5-token（判据 `text="\n\nThe sum 2"`、ids `[271,760,2542,220,17]`）。

### P5.1 最小复现**没有**重现 ⇒ 不是"UVA 视图 dtype 本身"的锅

在真机上做了两组对照（`/root/repro_u8.py`、`/root/repro_u8b.py`）：

| 用例 | 结果 |
|---|---|
| 锁页 uint8 → UVA 视图 → Triton kernel | ✅ OK |
| 锁页 fp8 → UVA 视图 → `.view(torch.uint8)` → Triton kernel | ✅ OK（`uva dtype: torch.float8_e4m3fn` → `after view: torch.uint8`） |
| fp8 设备张量 → `.view(torch.uint8)` → Triton kernel | ✅ OK |

⇒ `'u8>'` **不是** UVA 视图的通病。差异只剩真实调用的**参数组合**：

```python
_lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
    self._uva_weight.view(torch.uint8),   # uint8 视图
    flat_ids,                             # input_ids.reshape(-1).long()
    _out_u8,                              # output.view(torch.uint8)
    self._row_bytes,                      # int
    self.shard_indices.org_vocab_start_index,
    self.shard_indices.org_vocab_end_index,
    BLOCK_D=self._block_d,                # tl.constexpr
)
```

下一步按"逐个参数消去法"缩小：
1. 在真机上用同样的 7 个参数（含 `BLOCK_D` 与两个 vocab 边界）复现一次；
2. 若复现，逐个替换成常量，定位是哪个参数让 Triton 把注解拼成 `'u8>'`
   （最可疑：`flat_ids` 的 `int64` 与 `BLOCK_D` 的取值组合，或 `_row_bytes` 极大时的特化）；
3. 找到后改成显式 `tl.uint8` / 常量传参，避免依赖字符串推断。

### P5.2 完整 7 参数复现也**没**重现（含真实尺寸）

| 用例 | 参数 | 结果 |
|---|---|---|
| 完整 7 参数、小尺寸 | `row_bytes=64, BLOCK_D=64, ids=int64, w=uint8 视图` | ✅ OK |
| 完整 7 参数、**真实尺寸** | `row_bytes=3584, BLOCK_D=4096` | ✅ OK |

⇒ 复现脚本（`/root/repro_p5.py`、`/root/repro_p5b.py`）与真实调用的差别只剩：
1. 真实 `_uva_weight` 是 **2D**（`num_embeddings × embedding_dim`）的大表，而脚本用的是 1D 小表；
2. 真实 `output` 形状是 `(*input_ids.shape, embedding_dim)`（多维），脚本用 1D；
3. 真实表的 `_uva_weight` 来自**插件/稳定 ABI 的视图**，脚本用 `torch.utils.torch_utils` 的同一函数 —— 理论上一致。

⇒ 下一步（已缩小到两个方向）：
1. 用 **2D 大表 + 多维 output** 复现一次（最可能，因为 2D fp8 `.view(torch.uint8)` 会改变 shape）；
2. 若 2D 复现，检查 Triton 对该 2D 指针参数的**特化字符串**（`'u8>'` 里那个 `>` 很可能是 Triton 对
   "带对齐/多维"指针的修饰符，而不是 dtype 的字节序），再决定是传 `.reshape(-1)` 后的一维视图，
   还是显式给内核 `tl.uint8` 的指针类型标注。

### P5.3 三种合成复现全部失败 ⇒ 停止猜测，改为**在真实调用点插桩**

| 复现尝试 | 结果 |
|---|---|
| 1D 小表、完整 7 参数 | ✅ OK |
| 1D 真实尺寸（`row_bytes=3584`） | ✅ OK |
| **2D 表 + 多维 ids + 多维 output**（与真实同构） | ✅ OK（2D 与 1D 视图都过） |

⇒ 合成参数**复现不了** `KeyError: 'u8>'`。继续猜参数没有意义，
**下一步改为在真实调用点插桩**（`vllm/models/qwen4_exp/common/ngram_embedding.py:_lookup`，
内核启动前打印一次）：

```python
print(f"[ple-u8] w={w.dtype}/{tuple(w.shape)}/{w.device} "
      f"ids={flat_ids.dtype}/{tuple(flat_ids.shape)}/{flat_ids.device} "
      f"out={_out_u8.dtype}/{tuple(_out_u8.shape)} "
      f"row_bytes={type(self._row_bytes).__name__}:{self._row_bytes} "
      f"lo={type(self.shard_indices.org_vocab_start_index).__name__}:{self.shard_indices.org_vocab_start_index} "
      f"hi={type(self.shard_indices.org_vocab_end_index).__name__} "
      f"BLOCK_D={self._block_d}", flush=True)
```

**判据**：谁的类型/形状与合成用例不同，谁就是元凶（重点看两个 vocab 边界是否为**张量**而非 int，
以及 `_block_d` 是否异常）。拿到这一行之后再改，不再盲试。

**教训（写进四问模板的补充）**：合成复现连续失败 ≥2 次就该**回到真实调用点插桩**，
不要继续加变量 —— 合成用例永远缺一个你没想到的真实条件。

### P5.4 根因（已修）：内核收 **fp8 张量** 就崩，收 `uint8` 字节视图就通过

决定性对比（`/root/repro_p5d.py`，与真实调用同构：2D fp8 表 + 多维 ids/output + 完整 7 参数）：

| 传入内核的 weight / output | 结果 |
|---|---|
| `float8_e4m3fn` 张量（**真实代码就是这样**） | ❌ `CompilationError`（真实链路表现为 `KeyError: 'u8>'`） |
| `.view(torch.uint8)` 字节视图 | ✅ OK（此前三组复现都是这个形态，所以一直复现不出来） |

**为什么等价**：这个内核只做**原始字节搬运**（`weight_ptr + local_idx*row_bytes + offsets`），
与元素类型无关；改用 uint8 视图不改变任何语义。

**修法**（已打在 `vllm/models/qwen4_exp/common/ngram_embedding.py:_lookup`）：

```python
_w_u8 = self._uva_weight if self._uva_weight.dtype == torch.uint8 else self._uva_weight.view(torch.uint8)
_o_u8 = output if output.dtype == torch.uint8 else output.view(torch.uint8)
_lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](_w_u8, flat_ids, _o_u8, ...)
```

**教训（补进四问模板）**：*"合成复现跑通了"不等于"线上没问题"* ——
我最初三次复现都**顺手加了 `.view(torch.uint8)`**，恰好绕过了真实缺陷。
⇒ 合成复现必须**逐字照抄真实调用**（包括所有 dtype 转换），否则你验证的是自己的改写版。

---

## P6 【2026-10-22】`_C` 的 PrivateUse1 注册块被 `#if !defined(USE_MUSA)` 整体排除

### 现象

```
NotImplementedError: Could not run '_C::top_k_per_row_decode' with arguments
from the 'musa' backend. This could be because the operator doesn't exist for
this backend, or was omitted during the selective/custom build process.
```

换用 `_C::persistent_topk` / `cooperative_topk` 也一样。
**注意措辞**：op **存在**（schema 有）、二进制里符号**也在**，缺的是 **dispatch kernel**。
这与 P4（`_C` 根本没编进去）是**不同**的坑，诊断手法不同。

### 诊断（三步，30 秒）

```python
import torch, vllm._C_stable_libtorch
torch.ops._C.top_k_per_row_decode          # ① schema 存在？
print(torch._C._dispatch_dump_table("_C::top_k_per_row_decode"))   # ② 表里有没有 PrivateUse1
print(torch._C._dispatch_has_kernel_for_dispatch_key(              # ③ 权威判据
    "_C::top_k_per_row_decode", torch._C.DispatchKey.PrivateUse1))
```

本次结果：① 存在 ② 表里只有 `AutogradPrivateUse1`（那是 fallback，**不是** kernel）
③ `False`。

### 根因

`csrc/torch_bindings.cpp:759`：

```cpp
#if !defined(USE_MUSA)
STABLE_TORCH_LIBRARY_IMPL(_C, PrivateUse1, ops) {
  ...
  ops.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));   // ← 876 行
  ...
}
#endif
```

整个 PrivateUse1 注册块在 MUSA 上**不参与编译** ⇒ `_C` 的这些 op 在 MUSA 上只有
schema、没有 kernel。而实现（`sampler.cu` 的 `top_k_per_row_decode`）**本来就是
MUSA 版**（用 `musaStream_t` / `get_current_cuda_stream()` 写的）。
⇒ 又是"源码有、二进制没有"，与 P1 里 `_C` 只编 2 个文件同一族。

⚠️ **`nm` 能看到符号 ≠ 能调用**：符号在 `.so` 里，但 dispatch 表没登记，运行时报
"Could not run ... for this backend"。**判据只能是 `_dispatch_has_kernel_...`**。

### 修法

在同一个文件里补一个 MUSA 专用块（`/root/vllm-xiaotu-moe-tmp/patches/patch_musa_topk_reg.py`）：

```cpp
#if defined(USE_MUSA)
STABLE_TORCH_LIBRARY_IMPL(_C, PrivateUse1, ops) {
  ops.impl("top_k_per_row_prefill", TORCH_BOX(&top_k_per_row_prefill));
  ops.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));
}
#endif
```

然后 `python setup.py build_ext --inplace`（约 10–20 min，`MAX_JOBS=32`）。

### 泛化（四问第 4 问：怎么让这类坑**只能犯一次**）

1. **`#if !defined(USE_MUSA)` 是本仓 MUSA 移植的主要手法** ⇒ 每次遇到
   `Could not run '_C::xxx' with arguments from the 'musa' backend`，
   **第一件事就是** `grep -n "USE_MUSA" csrc/torch_bindings.cpp`，看这个 op 是不是在
   被排除的那一块里 ✓
2. **MUSA 侧的正确姿势不是"在 Python 里兜底"**：先确认实现是否已经 MUSA 化
   （`musaStream_t` / `get_current_cuda_stream` / `#include "torch_musa/..."`）——
   若是，只需要**补注册**，不要写 torch 回退（回退慢且可能语义不同）✓
3. 与 P5 对照：P5 是 **Triton 编译器**层的问题（uint8 指针被错拼成 `'u8>'`），
   P6 是**注册层**的问题。两者的共同教训是：
   **"报错信息说的是 A，真正的原因往往在 A 的上游一层"** ✓

---

## P8. MUSA 上 **CUDA graph 不可用**（当前 torch_musa），只能 eager

### P8.1 三层证据

| 层 | 检查 | 结果 |
|---|---|---|
| **SDK** | `musaGraphCreate` / `musaStreamBeginCapture` / `musaGraphInstantiate` | ✅ **有**（`/usr/local/musa/include/cuda_runtime.h`） |
| **torch_musa 接口** | `torch.musa.CUDAGraph` | ❌ 不存在 |
| | `torch.musa.graph` | ⚠️ 是个空 class（`dir()` 里没有任何 Graph 成员） |
| | `torch.musa.graph_pool_handle()` | ⚠️ 返回假值 `(0, 1)`（stub） |
| | `torch.musa.is_current_stream_capturing()` | ⚠️ 恒返回 `False`（stub） |
| **实际能不能跑** | `torch.musa.Stream()` | ❌ `AttributeError: module 'torch._C' has no attribute '_cuda_getCurrentStream'` |
| | `torch.cuda.Stream()` | ❌ `RuntimeError: torch.cuda.Stream requires CUDA support` |
| | `torch.cuda.CUDAGraph` | ⚠️ 类存在（`torch.cuda.graphs.CUDAGraph`），但**没有可用的 stream 就无从 capture** |

### P8.2 结论

* **SDK 层具备 graph 能力，但 PyTorch 后端没有接上**（连 `Stream` 的构造函数都走不通）；
* vLLM 的 graph 路径（`torch.cuda.CUDAGraph` + `torch.cuda.graph`）在本环境**不可用**
  ⇒ **只能 eager**；
* 我们的探针本来就用 `enforce_eager=True`，与此一致 ✓

### P8.3 影响与后续

* **性能**：graph 缺失意味着每步都有 kernel 启动开销与 Python 开销，
  长序列/高并发场景会有明显损失 —— 这是**已知天花板**，不是我们代码的问题；
* **如果要 graph**：需要先补齐 `torch_musa` 的 stream/graph 绑定（上游工作），
  不是改 vLLM 能解决的；
* **注意**：这也解释了为什么"UVA 轮询 + graph capture 捕获旧指针"这类假设不成立
  （根本没有 graph），排查台账里已排除。

---

## P8 勘误（2026-10-23 16:30）：**`Stream` 是可用的；graph 仍不可用，但原因不同**

### 我之前的错误结论

P8 原文写「`torch.musa.Stream()` ❌ `AttributeError: _cuda_getCurrentStream`」，
并据此推断"我们的环境 stream/graph 绑定不全"。

### 实测更正

用**同一份包**（`torch 2.11.0.post1` / `torch_musa 2.11.0.post1+b431668`，
与厂商镜像里的 `_MUSAC.so` 同为 33872 字节）复测：

```
Stream OK torch_musa.Stream device=musa:0 musa_stream=0x563ed580b850
```

⇒ **`Stream` 完全可用**；先前那次 `AttributeError` 是**那次探测的误判/瞬态**，
不是版本差异，也不是环境缺件。

### graph 依然不可用，但根因是"未实现"

```
torch.musa.CUDAGraph = False        # 属性不存在
torch.cuda.CUDAGraph = True         # 类存在，但……
capture FAIL RuntimeError: Tried to instantiate dummy base class CUDAGraph
```

⇒ `CUDAGraph` 只是个 **dummy base class**，capture 未实现
⇒ **vLLM 的 graph 路径在 MUSA 上不可用**（结论不变，理由不同）。

### 修正后的判据

| 能力 | 状态 | 判据 |
|---|---|---|
| `torch.musa.Stream()` | ✅ **可用** | 构造成功并返回有效 handle |
| `torch.musa.CUDAGraph` | ❌ 无 | `hasattr` 为 False |
| `torch.cuda.CUDAGraph` | ⚠️ 有类但不可实例化 | `Tried to instantiate dummy base class` |
| 图捕获/回放 | ❌ 不可用 | 上述异常 |

⇒ 结论：**只能用 eager**；但**不要再把原因写成"stream 绑定缺失"**。

---

## P7 扩展（2026-10-24）：UVA 的两条硬约束 —— **只能"主机写、设备读"**

在 MUSA 上，`torch.musa` 的 UVA（`musaHostGetDevicePointer` 映射的锁页主机内存）
**只支持"主机写入、设备读取"**。任何**设备侧写入**都会失败，且两种失败的形态不同：

| # | 违反方式 | 实测表现 |
|---|---|---|
| **P7-a** | 对 UVA 张量做**原地设备端写**（`masked_fill_` / `clamp_` / 内核写回） | `RuntimeError: MUSA error: operation not permitted`；<br>在 `MUSA_LAUNCH_BLOCKING=1` 下表现为**永久 hang**（native 栈停在 `muLaunchKernel`）|
| **P7-b** | `copy_` **进** UVA 张量（即使源张量已 `.contiguous()`） | `RuntimeError: mtgpu_impl_copy_d2d MUDNN failed in: Run Copy by Identity` |

### 我们踩过的实例

| 位置 | 违反 | 症状 |
|---|---|---|
| `mamba_attn.py` 块表清洗 `_bt.masked_fill_(_bad_bt, 0)` | P7-a | 第二个请求永久 hang（§五十）|
| `ngram_embedding.py` `finalize_prefetch` 的 `output.copy_(...)`（output 来自 `buffer.new_empty`，是 UVA）| P7-b | MTP 初始化失败（§六十九）|

### 规则（写代码时照此执行）

1. **需要设备写的结果张量，一律分配普通显存**（`torch.empty(..., device="musa")`），
   不要用 `uva_buffer.new_empty(...)`；
2. 若确实要更新 UVA 内容，**在宿主侧改**（`.cpu()` / `numpy` 视图），再让设备读；
3. 对 UVA 张量做**只读**算子（`gather` / 索引 / 加进点积）完全没问题。

### 关于 muDNN 拷贝的另一个表现

同一错误 `mtgpu_impl_copy_d2d ... Run Copy by Identity` 还曾出现在
`max_num_batched_tokens=256` 的初始化里（§六十二 附近），也是**同族**：
muDNN 的 Identity 拷贝对内存位置/形状有额外要求，遇到时优先怀疑"目标是否 UVA"。

---

## P9. MUSA 上编译 `xiaotu_moe` C++ 引擎：`NO_CUDA=1` + `XIAOTU_MOE_MUSA` + MUSA 头/库

**背景**：`scripts/build_engine_variants.sh` 在没有 cudart/libcuda 的机器上**必须**
`NO_CUDA=1`，否则 `exit 2`（"CUDA runtime not found"）。而 `NO_CUDA=1` 会把
`binding.cpp` 里 `#ifndef XIAOTU_MOE_NO_CUDA` 包住的**全部 H2D DMA 绑定**编译掉
（详见 `MUSA_VENDOR_ISSUES.md` 第 13 条）。

**正确编译命令（本机实测可用）**：

```bash
cd /root/lvllm
NO_CUDA=1 PYTHON=/root/venvs/vllm310/bin/python \
EXTRA_FLAGS='-DXIAOTU_MOE_MUSA=1 -I/usr/local/musa/include \
             -L/usr/local/musa/lib -lmusart -Wl,-rpath,/usr/local/musa/lib' \
bash scripts/build_engine_variants.sh
```

**三个必踩的坑**：

| 坑 | 现象 | 正解 |
|---|---|---|
| `muMemcpyAsync` 未声明 | `error: 'muMemcpyHostToDevice' was not declared` | 运行时 API 是 **`musaMemcpyAsync` / `musaMemcpy2DAsync`**（在 `musa_runtime_api.h`）；`mu*` 是**驱动** API（在 `cuda.h`），混用会失败 |
| 枚举名 | `'muMemcpyHostToDevice' not declared` | 用 **`musaMemcpyHostToDevice`** |
| 链接 | `undefined symbol: musaMemcpy2DAsync` | `EXTRA_FLAGS` 里加 `-lmusart -Wl,-rpath,/usr/local/musa/lib`（`NO_CUDA=1` 分支默认不链任何 GPU 库） |

**验证（必须做）**：

```bash
for f in xiaotu_moe/build/*310*.so; do
  printf '%-70s ' "$(basename $f)"; strings -a $f | grep -c copy_hostbuf_to_device_2d
done      # 期望每个都是 1
```

**回滚**：`backups/xiaotu_moe_build_*.tgz`（整包 build 目录）+ `backups/binding_cpp_*.cpp`。

---

## P10. GPU 固件 panic 后**显存不回收**：必须 `mthreads-gmi -r 2`（`-r 0` 无效）

**症状**：worker 崩了，但 `torch.musa.mem_get_info()` / `btop` / `mthreads-gmi -q` 都显示
显存仍被占（实测 42.6 / 43.9 GiB），而 `ps`/`lsof` 找不到持有者（只有 btop 监控进程）。
此时任何新进程连 8 GiB 都分配不出来。

**先看内核日志确认是不是固件 panic**：

```bash
dmesg -T | grep -iE 'submission response FAULT|FW_PANIC|CSW_TIMEOUT|Fault Reason' | tail
```

若出现 `Fault Reason : CSW_TIMEOUT` + `FW_PANIC` ⇒ 设备已进坏状态，
**进程退出不会释放显存**，必须复位：

```bash
mthreads-gmi -i 0 -r 0     # ❌ 报 success 但显存不回来（实测）
mthreads-gmi -i 0 -r 2     # ✅ 整设备复位
mthreads-gmi -i 1 -r 2
# 复核（应回到 ~2 GiB used）
python -c "import torch,torch_musa; print(torch.musa.mem_get_info(0))"
```

**门禁脚本必须检查"真实可用显存"而不是"有没有进程"**：
崩溃后进程可能已经没了，显存却还在占 —— 只看 `ps` 会误判为干净。

**读数的坑**：`mthreads-gmi` 的 Used 列可能被 `cut` 截断（`42604MiB` 只剩 `4`），
判断时不要用固定列宽裁剪；用 `mthreads-gmi -q -d MEMORY` 或
`torch.musa.mem_get_info()` 交叉验证（两者一致才可信）。

---

## P11. **不要 `kill -9` 正在跑 GPU 内核的 worker**（会触发固件 panic）

**实测（2026-10-24 03:18）**：对正在做 15K GPU prefill 的两个 worker 直接 `kill -9`：

```
submission response FAULT ... cmd(GPU_CMP)
FW_PANIC: DMUP_REASON 0x11, FAULT_ID 0x0
Fault Reason : KILL_TIMEOUT      ← 两卡各一次
```

**正确停引擎顺序**：

```bash
pkill -TERM -f bench_6x.py       # 或发给 EngineCore
sleep 15                          # 让它自己收尾（flush 队列 + 释放显存）
pkill -TERM -f EngineCore
sleep 10
# 只有还活着才强杀
pkill -KILL -f 'bench_6x|EngineCore|Worker_TP'
# 强杀后必须检查
journalctl -k --since '-3min' | grep -E 'FW_PANIC|CSW_TIMEOUT|KILL_TIMEOUT'
# 有 panic ⇒ 复位
mthreads-gmi -i 0 -r 2; mthreads-gmi -i 1 -r 2
```

**判据**：`kill -9` 之后若 `mthreads-gmi -q -d MEMORY` 显示显存未回收，
不要重试启动，**先 `-r 2`**（`-r 0` 无效，见 P10）。

---

## P11b. 【2026-10-24】`musaHostRegister` 锁页的内存在 `kill -9` 后**不释放**，且 `-r 2` 也回收不了

**现象**（S4000 / driver 5.2.0 / MUSA 5.2.0, 503 GiB 机器）：

```
VRAM:     116 MiB / 82 MiB        ← 显存已回收
进程:      已退出
Mem used: 144 GB                  ← 主机内存不回收
MemFree 288 / Cached 70 / AnonPages 3.5 / Shmem 0 / Slab 1.2
```

`free` 报的 used(145 GB) **既不在进程 RSS（合计 6.2 GB）、也不在 shmem/slab/cache**，
属于"驱动持有、内核不归类"的 pinned 页。

**触发链**：

1. `pin_hostbufs()` 调 `musaHostRegister` 锁住引擎 host 分片（本机实测 6/6 成功）；
2. worker 被 **`kill -9`**（本会话为提速反复这么做，违反 P11）；
3. 驱动**不释放**这些 pinned 页 ⇒ 每轮泄漏数 GB；
4. **`mthreads-gmi -i N -r 2` 只复位显存，回收不了 host pinned 页**。

**回收方式**（按代价排序）：

1. 卸载/重载内核模块（`rmmod mthreads && modprobe mthreads`，会中断所有 GPU 使用者）；
2. 重启（最稳）；
3. 预防：
   * **绝不对 worker 用 `kill -9`**（用 `TERM` 并等待退出，P11 已有明文）；
   * 若必须频繁"起-停"，临时 `XIAOTU_GPF_PIN=0` 关闭锁页（牺牲 DMA 带宽换不泄漏）；
   * 锁页只在**长期存活**的进程里做一次，不要"每轮重新锁"。

**判定方法**（下次怀疑泄漏时一条命令）：

```bash
free -g | head -2; ps -eo rss --sort=-rss | head -5
# 若 used 远大于 ΣRSS 且 shmem 很小 ⇒ 就是本类泄漏
```

---

## P11c. 【更正 P11b】卸载 `mtgpu` 模块**也没有**回收那 141 GB ⇒ 泄漏源不是 `musaHostRegister`

**2026-10-24 21:20 实测（按顺序做，每步都量了内存）**

| 操作 | used | 说明 |
|---|---|---|
| 起始 | 144 GB | 进程已全退 |
| `kill -TERM` 掉 btop（它占着 `/dev/mtgpu.*`，refcnt=8） | 141 GB | |
| `rmmod mtgpu` + `modprobe mtgpu`（成功，GPU 重新枚举） | **141 GB** | **没回收** |
| `echo 3 > /proc/sys/vm/drop_caches` | **141 GB** | 只把 Cached 93→24，used 不变 |

### 关键事实

```
Σ 所有进程 RSS   = 0.3 GiB      (awk /proc/*/status 求和)
AnonPages        = 0.1 GiB
Shmem            = 23.8 GiB
Slab             = 0.8 GiB
Cached(掉缓存后) = 23.9 GiB
--------------------------------
可解释合计        ≈ 24.7 GiB
free 报 used      = 141 GB      ⇒ 约 117 GB 无法归类
```

### 结论（更正）

1. **不是我加的 `musaHostRegister` 造成的**——卸载驱动模块后内存未回；
2. 也不是 page cache（drop_caches 后不变）、不是 shmem、不是 slab、不是任何进程；
3. 这 117 GB 属于"**被移出 buddy allocator 且未归入任何 meminfo 字段**"的页，
   典型来源是某些驱动/服务自己 `alloc_pages` 后 `remap_pfn_range`（或 DPDK 预留），
   但**当前已无相关进程**；
4. 唯一可靠回收方式：**重启**（`rmmod` 无效）。

### 教训（对 P11b 的修正）

* **先隔离变量再下结论**：我在没有逐步测量的情况下就断言"是 kill -9 + 锁页导致的"，
  这是错的——正确做法是"停进程 → 量、卸模块 → 量、drop cache → 量"，我这次补做了；
* 但 P11b 里"**绝不对 worker 用 kill -9**"这条本身仍然成立（P11 原有纪律），保留。

---

## P11d. 那 ~117 GB 的"无主内存"：四种回收手段全部无效（含 sysctl/大页）

**2026-10-24 21:35，按"ktransformers 重启需 sysctl 释放"的提示逐项验证**

| 手段 | 结果 |
|---|---|
| `kill -TERM` 掉占用 `/dev/mtgpu.*` 的 btop | 无变化 |
| `rmmod mtgpu` + `modprobe mtgpu`（成功，GPU 重枚举） | **无变化** |
| `sysctl -w vm.drop_caches=3` / `echo 3 > drop_caches` | Cached 93→24 GB，**used 不变** |
| `sysctl -w vm.nr_hugepages=0`（各尺寸 nr 本就是 0）+ `compact_memory` | **无变化** |

**旁证排除**：

```
Committed_AS = 6.7 GiB   CommitLimit = 259.9 GiB   overcommit_memory = 0   ⇒ 不是 overcommit
HugePages(2M/1G) nr=0 free=0                                              ⇒ 不是大页池
Σ 全机进程 RSS = 0.3 GiB    AnonPages = 0.1 GiB   Shmem = 23.8 GiB         ⇒ 不是进程/shmem
```

**结论**：这 ~117 GB 属于"已从 buddy allocator 移出、未归入任何 meminfo 字段"的页，
**用户态手段无法回收**，只能重启。当前 `MemAvailable = 358 GiB`，**不构成运行风险**，
但会持续占据机器 1/4 内存。

**运维建议**：下次计划内重启窗口顺手回收；在此之前不要把它误判成"某进程泄漏"而反复杀进程。
