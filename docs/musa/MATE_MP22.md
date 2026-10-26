# MATE 在 MP22（MTT S4000）上的补齐：GDN decode

> 状态：**GDN decode 单内核已在 MP22 上跑通并与 torch 参考对拍通过**（2026-10-22）。
> 端到端（vLLM 层）**仍被 block table 垃圾独立阻塞**，见 §5。
> 相关：`patches/mate/0001-gdn-decode-mp22-plain-copy-staging.patch`、`scripts/gdn_mp22_check.py`。

---

## 1. 为什么 MP22 跑不了 MATE 的 GDN decode

MATE 的 GDN decode 内核（`mate/gdn_kernels/tilelang/gdn_decode.py`）用**显式 TMA** 做 state 的
load/store：`T.tma_copy(...)` + `T.mbarrier_arrive` / `T.mbarrier_wait_parity` / `T.tma_store_wait`。

TMA 是 **MP31（S5000）起**才有的硬件能力，MP22（S4000）没有。于是在 MP22 上编译期就报：

```text
InternalError: Check failed: (result.supported) is false:
T.tma_copy() requires TMA-capable target and global<->shared copy pattern,
but TMA is not available for src=state, dst=state_load_stage
```

⚠️ 只加 `TL_DISABLE_TMA_LOWER=True` **无效** —— 那个 pass 只影响编译器自动插入的 TMA 拷贝，
管不到源码里**显式写的** `T.tma_copy`。这正是上一轮"改了 pass 仍然 illegal"的原因。

## 2. 改法（`patches/mate/0001-*`）

在 `_build_decode_fp32_vk_kernel_factory` 增加编译期参数 `use_tma`，用**静态分支**给出两条路径：

| 位置 | TMA 路径（MP31+，保持原样） | 非 TMA 路径（MP22，新增） |
|---|---|---|
| prologue 预取 | `T.tma_copy(...)` + `T.mbarrier_arrive` | `T.copy(state[a:b, 0:dim_k], stage)` + `T.sync_threads()` |
| 循环内 store | `T.tma_store_wait()` + `T.tma_copy(stage → state)` | `T.copy(stage, state[a:b, 0:dim_k])` |
| 循环内 wait | `T.mbarrier_wait_parity(...)` | 无（由前一次 `sync_threads` 保证） |
| 循环内 refill | `T.tma_copy` + `mbarrier_arrive` | `T.copy` + `T.sync_threads()` |
| epilogue | `T.tma_copy` + `T.tma_store_wait()` | `T.copy` |

`T.alloc_barrier(...)` 只在 `use_tma=True` 时分配。

**选路是自动的**，不改任何调用方：

```python
_TMA_MIN_ARCH = (3, 1)          # MP31 起才有 TMA
target_supports_tma() -> bool   # 读 torch_musa.get_device_properties().major/minor
```

* 结果（按 device index）用 `lru_cache` 记住，热路径上只是一次 dict 查表；
* 环境变量 **`MATE_GDN_DECODE_TMA=1/0`** 可强制选路（用于 A/B 与在 MP31 上验非 TMA 路径）；
* 取不到设备信息时**退回原 TMA 路径**（fail-safe：不改变既有平台行为）。

`use_tma` 同时进了 kernel 的 `lru_cache` key，避免 MP22/MP31 混跑时拿到错的内核。

## 3. 验证（S4000，MP22，torch 2.11.0.post1 / mate 0.2.7）

`scripts/gdn_mp22_check.py`（不依赖 pytest；自带 float32 参考实现，bfloat16 激活、fp32 state）：

```text
device=MTT S4000 arch=mp22 tma_available=False
[case B=  1 pool=0] out max|err|=3.897e-03 (scale 2.22) state max|err|=5.364e-07 (scale 2.56)
[case B=  1 pool=1] out max|err|=3.897e-03 ... state max|err|=5.364e-07
[case B=  2 pool=0/1] out max|err|=3.900e-03 ... state max|err|=5.960e-07
[case B=  4 pool=0/1] out max|err|=7.378e-03 ... state max|err|=6.407e-07
[case B= 16 pool=0/1] out max|err|=7.462e-03 ... state max|err|=7.153e-07
RESULT: PASS
```

* 输出相对误差 **1.8e-3 ~ 3.3e-3** ⇒ 就是 bf16 舍入量级；
* state 误差 **~2e-7 ~ 3e-7**（fp32 累加，几乎逐位相同）；
* **pool（`state_indices`）与非 pool 两条路径都对** —— vLLM 用的是 pool 那条。

反向对照（证明分支确实是关键）：`MATE_GDN_DECODE_TMA=1` 时 8 个 case 全部复现 §1 的报错。

## 4. 尚未覆盖的 TMA 使用者

同一套 `T.tma_copy` 还出现在（`grep -l tma_copy mate/`）：

```text
gdn_mtp.py, gdn_prefill.py, gdn_kkt_solve.py,
sparse_mla/tilelang/*.py, flash_attention/tilelang/_flash_attention_bwd_split_dq.py
```

⇒ MP22 上这些路径**仍会编译失败**。`target_supports_tma()` 的放置位置目前还在 `gdn_decode.py`，
下一步应上提到共享模块（如 `mate/jit/env.py`）供这些内核复用。

## 5. 端到端现状（vLLM）

vLLM 侧 `[xtu-bt] host fallback groups=6` 说明 6 个 KV group 的 block table 走的是 host fallback。
崩溃点已从模糊的 `MUSA error: unknown error` 定位到：

```text
qsa.py:935   pages = torch.from_numpy(bt_np[req].astype("int64")).to(dev)[ixv // BS]
IndexError: index 1042049980 is out of bounds for axis 0 with size 1
```

**block table 里是垃圾值**（与 `HANDOFF.md` 的 `1046632475` 同族）。
单独测 `_apply_write_kernel` 在 MUSA 上**是对的**，所以嫌疑落在写入路径的周边，不在内核本身。

### 5.1 已定位到「`side_metadata.token_to_req` 从未被写入」（2026-10-22）

崩溃点在 QSA 的 CPU 回退 `qsa_sparse_paged_attention`（不是 block table 内核）：

```text
qsa.py:945  pages = torch.from_numpy(bt_np[req].astype("int64")).to(dev)[ixv // BS]
IndexError: index 1042049980 is out of bounds for axis 0 with size 1
```

同一个 layer 里，更早的 `qsa_select_paged_tokens` 读到的 `token_to_req` 是 **正确的** `[0]*7`，
而 `forward_qsa → qsa_sparse_paged_attention` 读到的 `req_np[0]` 是 **1042049980**（`0x3E1B…`，浮点位模式）。
两者来自**不同的 QSA metadata 对象**：

```text
main_metadata = metadata[layer_name]                      # FlashAttentionMetadata
side_metadata = metadata[indexer.raw_key_cache.prefix]    # QSAForwardMetadata（raw/压缩各一份）
forward_qsa(..., token_to_req=side_metadata.token_to_req) # ← 这里读到垃圾
```

⇒ 推断：**side（raw）那一份的 `token_to_req_buffer` 是用 `torch.empty` 分配、但从未被
`build_qsa_metadata` 写入**，读回来的就是分配器里残留的浮点激活数据。

**已排除的候选**（都在 MP22 上单独验过，结果正确）：

| 候选 | 结论 |
|---|---|
| `_apply_write_kernel`（block table staged write） | PASS（直接构造 3 次写入比对） |
| `compute_token_to_req_indices`（设备端二分查找） | PASS（decode/prefill/mixed 三种 batch） |
| fused `_build_qsa_metadata_kernel`（triton） vs `_build_qsa_metadata_torch` | 两者输出一致 |
| 显式 `torch.musa.synchronize()` 后再读 | **无效**，不是单纯的同步/时序问题 |

⚠️ 本轮我曾上报"host fallback 抛 IndexError 被 `except` 吞掉"——那是**我自己插入的调试打印**
（`bt.gpu[1]` 在只有 1 行的表上越界）造成的，已撤回。

