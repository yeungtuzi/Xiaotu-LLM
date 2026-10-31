# 显存守卫与分配器行为（对 vLLM 默认行为的改变）

> 本文档面向**使用者**：安装/运行本引擎后，进程的显存行为与"原生 vLLM + 原生 PyTorch"
> **不完全相同**。凡与默认行为不同的地方，都在本文逐条列出，并给出关闭方法。

适用范围：`vllm_xiaotu_moe` 插件（PyTorch ≥ 2.x，CUDA 12.x）。默认值即生产值，**无需任何环境变量**。

---

## 1. 为什么要改变默认行为

原生组合（vLLM + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`）在**长上下文 + 变长 chunk**
负载下有一种可复现的退化：

| 现象 | 实测（A100-40GB，39.49 GiB 可用，TP=2） |
|---|---|
| `torch.cuda.memory_allocated()` | 基本恒定（23.3 → 23.9 GiB）⇒ **不是泄漏** |
| `torch.cuda.memory_reserved()` | 单调上涨（33.4 → 36.7 GiB） |
| `reserved − allocated`（碎片） | 同步上涨（10.1 → 12.8 GiB） |
| 设备空闲（`mem_get_info` free） | 被吃到 **0.55–1.3 GiB** |
| 结果 | 一次 512 MiB 连续分配失败 ⇒ 引擎退出 |

根因是 **caching allocator 的缓存/碎片**：变长请求制造出大量**新尺寸**，`expandable_segments:False`
下每个新尺寸都要新开段，而缓存的块**永不归还驱动** ⇒ `reserved` 只增不减，直到没有足够大的连续块。

> `expandable_segments:True` **不能**作为解法：在本引擎的初始化路径上会导致 Worker 进程在 KV
> 初始化阶段死亡（已实测，记录于内部 BUG_REGISTRY A67）。

因此本引擎内置一个**显存守卫**：在不改变任何计算路径的前提下，主动回收"当前用不到"的缓存，
并在真正贴近底线时**给出明确诊断后退出**，而不是让进程在半死状态下卡住。

---

## 2. 行为变化一览（与原生默认的差异）

| # | 原生默认 | 本引擎默认 | 关闭方法 |
|---|---|---|---|
| 1 | 显存守卫：无 | **常驻**，每 60 s 检查一次 | `XIAOTU_MEM_GUARD=0` |
| 2 | 空闲块只增不减，需手动 `empty_cache()` | **低于软限时自动回收**（先选择性淘汰，再兜底 `empty_cache()`） | 软限设为 0：`XIAOTU_MEM_SOFT_FREE_FRAC=0` |
| 3 | 周期性 `empty_cache()`：无 | **每 300 s 一次，在独立后台线程执行** | `XIAOTU_EMPTY_CACHE_EVERY_S=0` |
| 4 | OOM 时只抛异常 | OOM 时**先落完整现场**（遥测 + 空闲块尺寸统计 + 分配快照）再走原逻辑 | — （只增日志，不改控制流） |
| 5 | 无显存遥测 | 每 60 s 打印 `[xtu-vram] steady ...` | `XIAOTU_VRAM_REPORT=0` |
| 6 | 无逐次分配记录 | `XIAOTU_MEM_TRACE=1` 时记录每次 alloc/free 的 Python 栈并定时落快照 | 默认已关 |

---

## 3. 三条机制

### 3.1 选择性清理（不是"全清"）

插件内每个缓存条目都记录**最近使用步号**；清理时只淘汰**空闲 ≥ N 步**的条目：

* 正在用的条目**一个都不动**；
* 清理前执行 `torch.cuda.synchronize()` —— 这些缓冲可能刚被侧流写入，先释放会造成
  use-after-free（历史事故形态），因此同步是**必要**的，不是保守。

N 由 `XIAOTU_MEM_IDLE_STEPS` 控制（默认 **32** 步）。

### 3.2 软上限（立即清理）

判据用 **设备空闲**（`torch.cuda.mem_get_info`），**按比例**给出，与显卡总量无关：

```
slack = (1 − util) × total        # vLLM 计划自身留下的余量
软限  free < max(1.5 GiB, 0.40 × slack)
```

跌破软限 ⇒ 立刻执行 3.1；若仍低于软限 ⇒ 兜底 `empty_cache()`（全清空闲块，最后手段）。

> ⚠️ **2026-10-27 更正**：早先记录的"周期清理使 `reserved` 36.7→24.8 / 碎片 12.8→1.5 / 空闲 1.08→13.6"
> 这组对照**已作废** —— 当时的调用写成了 `empty_cache(dev)`，而本 PyTorch 版本 `empty_cache()`
> **不接受参数**，异常被 `except: pass` 吞掉 ⇒ 清理其实从未执行；那组差异来自负载变化。
> 修复后（改为 `with torch.cuda.device(i): torch.cuda.empty_cache()`，且异常会被打印）**已重测**
> （2026-10-27，r83，util 0.82，300 s 间隔）：
>
> | 第几次 | 耗时 | 清理后 free |
> |---|---|---|
> | 1 | **4470.8 ms** | 7.219 GiB |
> | 2 | 6.6 ms | 6.502 GiB |
> | 3 | 20.4 ms | **10.912 GiB** |
>
> 效果：`free` 由基线 5.8 → 7.2 → 10.9 GiB；稳态 `reserved 29.3 / 碎片 5.4 / free 8.5`，
> 明显优于未清理时的 `reserved 35.1 / 碎片 11.2 / free 2.7`。
>
> ⚠️ **那 4470.8 ms 是 `cudaFree` 的隐式同步在等一个在飞的 prefill chunk**（实测 chunk ≈ 4.5 s），
> 它发生在后台线程，**不阻塞计算线程** —— 这正是 §3.4 把它移出 forward 路径的原因。
> 常态代价 7–20 ms；按 300 s 间隔摊销 < 0.1%。

### 3.3 硬上限（明确诊断后优雅退出）

```
硬限  free < max(0.8 GiB, 0.15 × slack)
```

跌破硬限 ⇒ **依次**执行：

1. 打印 `[xtu-guard] ⛔ 硬上限触发` 及完整遥测（`allocated / reserved / frag / free`）；
2. 落一份逐次分配快照（`XIAOTU_MEM_TRACE_DIR`，默认 `/tmp/xtu_mem`）；
3. `faulthandler.dump_traceback(all_threads=True)` 打印**全部线程栈**；
4. `flush` 后以**退出码 13** 退出。

> 为什么是"退出"而不是"继续跑"：CUDA 没有内核抢占；一旦内核或分配进入不可用状态，
> 进程内没有任何手段安全恢复。**提前带现场退出 + 由外部看门狗重启** 比楔死几小时后
> 只留下一个超时错误更有诊断价值。退出码 13 可用于看门狗区分"显存触底"与其它故障。

---

## 3.5 三条"永不 free"的不变式（宁可浪费电，不要乱 free）

> 2026-10-27 裁定：**宁可浪费电，不要乱 free**。凡是"可能被异步消费者持有"的缓冲，
> 一律**不归还**——归还错一次就是 illegal memory access，代价远高于让它们常驻。

| # | 不变式 | 违反后的症状 | 代价 |
|---|---|---|---|
| 1 | **捕获开始后禁止 `cudaFreeHost`**：pinned 缓冲扩容时旧块只进退休表、永不释放 | 已捕获 CUDA Graph 的节点仍记录旧 VA ⇒ 图重放把 D2H/H2D 指向已解映射主机内存 ⇒ `cudaErrorIllegalAddress`（**GPU 显存完全健康**） | 宿主 pinned **≤ +252 MB**（最坏：解码 8192 token）；GPU **0** |
| 2 | **`data_ptr` 交给异步消费者 ⇒ 必须同时进 keepalive 且 `record_stream(消费流)`** | 块回到分配池后被提前复用 ⇒ 迟发的非法访存（与流量正相关） | **0 字节**（仅推迟几 KB 张量的重用一代） |
| 3 | **staging 一律 never-free**（tile 环 / landing / slots / 段缓冲永不淘汰） | 淘汰判据是"Python 侧最近使用"，而**图重放不经过 Python** ⇒ 被图引用的缓冲被判空闲、释放、复用 ⇒ 重放写已释放内存 | **0 增量**：这些是启动期预分配，实测 `持久缓冲合计 1.776 GiB`（`tile=48 nslots=2` 占 0.84 GiB） |

**副作用（必须知道）**：`reserved` 不会再因淘汰而下降，只剩 `empty_cache()` 能归还"整段空闲"的段
⇒ 判据应从"`reserved` 只降不升"改为"**在 29–34 GiB 区间往复**"。

**可执行断言**：`XIAOTU_MEM_EVICT` 默认 `0`（不淘汰）；`_SEG_BUF` 永久不在淘汰名单；
软限路径也不再淘汰任何缓冲，只做 `empty_cache()` 兜底。

## 4. 配置项

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `XIAOTU_MEM_GUARD` | `1` | 守卫总开关 |
| `XIAOTU_MEM_SOFT_FREE_FRAC` | `0`（不启用） | 软限（占总量比例；**默认不用它**，见下） |
| `XIAOTU_MEM_HARD_FREE_FRAC` | `0`（不启用） | 硬限（占总量比例；默认不用它） |
| `XIAOTU_MEM_SOFT_FREE_GIB` / `XIAOTU_MEM_HARD_FREE_GIB` | `0`（不启用） | 以绝对值覆盖比例（调试用） |
| `XIAOTU_MEM_IDLE_STEPS` | `32` | 选择性清理的"空闲步数"阈值 |
| `XIAOTU_EMPTY_CACHE_EVERY_S` | `300` | 周期 `empty_cache()` 间隔；`0` 关闭。**在独立后台线程执行**（见下） |
| `XIAOTU_VRAM_REPORT` | `1` | 60 s 遥测开关 |
| `XIAOTU_MEM_TRACE` | `0` | 逐次分配记录开关（诊断用，占宿主内存） |
| `XIAOTU_MEM_TRACE_EVERY_S` | `600` | 快照落盘间隔 |

**为什么阈值不按总量百分比**：`--gpu-memory-utilization` 的语义就是"让进程占到
`total × util`"，它自身必然把 free 压到 `(1 − util) × total`。若按总量的固定百分比设软限，
`util=0.85` 时 free 会**天然停在软限上** ⇒ 每 60 s 触发一次清理。因此默认按
**vLLM 计划留下的 slack** 计算（`util` 取自 `GPU_UTIL` / `VLLM_GPU_MEMORY_UTILIZATION`，缺省 0.85）：

| util | slack | 软限 | 硬限 |
|---|---|---|---|
| 0.82 | 7.11 GiB | 2.84 GiB | 1.07 GiB |
| 0.85 | 5.92 GiB | 2.37 GiB | 0.89 GiB |
| 0.90 | 3.95 GiB | 1.58 GiB | 0.80 GiB |

### 3.4 为什么清理放在后台线程

`torch.cuda.empty_cache()` 内部是 `cudaFree`，**会隐式同步设备**。若在模型 forward 路径上调用，
它会等当前在飞的 kernel 排空 —— 实测一次 prefill chunk（8192 token）约 **32 s**（prompt 吞吐 255 tok/s），
最坏情况下这一步会被完全阻塞（摊销损失可达 27%）。

因此周期性清理与守卫都运行在 **`xtu-mem-cleaner` 后台线程**：它自己等待同步，
计算线程仍可继续提交 kernel。理论摊销损失降为 `T_free / 间隔`：

| 场景 | 单次代价 | 300 s 间隔下的损失 |
|---|---|---|
| 空闲/解码期 | ~40 ms | ≈0.013% |
| 有在飞任务 | 0.5–2 s | 0.17%–0.7% |

并且清理线程会跳过 **CUDA Graph 捕获**期间（`capture_in_progress()`），避免破坏捕获。

守卫**只在服务进入 `steady` 阶段后武装**：加载权重、KV 初始化、CUDA Graph 捕获期间完全不动作，
因此不影响启动。

---

## 5. 你应该看到的日志

正常（每 60 s 一行；没有第三行说明守卫没触发，这是**期望状态**）：

```
[xtu-vram] steady cuda:0 allocated=23.357 reserved=24.842 max_alloc=28.925 frag=1.485 free=13.597 total=39.493 GiB
[xtu-vram] empty_cache 已执行 ⇒ free=13.596 GiB
```

软限触发：

```
[xtu-guard] 软上限触发(steady): free=5.812 < soft=5.924 (reserved=32.623) ⇒ 淘汰空闲条目 0.0 MiB，reserved=32.6 GiB
[xtu-guard] 仍超软限 ⇒ empty_cache 兜底，free=13.5 GiB
```

硬限触发（随后进程以 13 退出）：

```
[xtu-guard] ⛔ 硬上限触发: free 低于 1.580 GiB (reserved=36.2 GiB)
[xtu-vram] guard-hard cuda:0 allocated=... reserved=... frag=... free=...
[xtu-mem] 快照已落盘: /tmp/xtu_mem/snap_..._guard-hard.pickle
```

`allocated` 与 `reserved` 的**差值**是关键判据：

* `allocated` 涨 ⇒ 真实需求增长（或泄漏），要查持有者；
* `reserved` 涨而 `allocated` 平 ⇒ 缓存/碎片，本守卫处理的正是这一类。

---

## 6. 诊断工具

`ops/a100-stress/mem_snapshot_diff.py` —— 比对两份分配快照，按**调用栈**列出"谁在长"：

```bash
python ops/a100-stress/mem_snapshot_diff.py snapA.pickle snapB.pickle
```

输出形如（示例为启动期真实数据）：

```
    0.843 GiB  gpu_prefill.py:2750 <genexpr> <- _tile_ring <- prealloc_tile_ring <- prealloc_staging
    0.420 GiB  gpu_prefill.py:2719 prealloc_tile_ring <- mixed_experts.py:1560 prealloc_staging
    0.141 GiB  gpu_prefill.py:610 _reuse_moe <- mixed_experts.py:1555 prealloc_staging
```

## 5. 与 custom allreduce 的互斥：`expandable_segments` 必须搭配 `--disable-custom-all-reduce`

**结论（2026-10-27 实测，A100 / TP=2 / vLLM V1）**

> **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 vLLM 的 IPC 版 custom allreduce
> 不可共存。要前者，就必须 `--disable-custom-all-reduce`（本仓 `serve_v41.sh` 的
> `NO_CUSTOM_AR=1` 即为此开关，默认关）。**

### 为什么（机制）

1. `expandable_segments:True` 让 PyTorch 改用 **CUDA VMM 接口**（`cuMemCreate`/`cuMemMap`）分配显存；
2. vLLM 的 `custom_all_reduce` 需要在 TP 各 rank 之间**交换显存块的 IPC 句柄**建 P2P 通路；
3. **VMM 分配的显存不能导出 IPC 句柄** ⇒ 注册阶段返回 `invalid argument`：

```
Capturing CUDA graphs (FULL): 100%|██████████| 2/2
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid argument'
Worker proc VllmWorker-0 died unexpectedly (exit code: None)
EngineCore: RuntimeError: cancelled   ← 卡在 determine_available_memory
```

4. 该失败发生在 **KV profiling 之后、图捕获完成时**，是**秒级**失败，不是运行期问题。

### 为什么仍然值得开 `expandable_segments`

不开它时（`expandable_segments:False`），碎片形态是"**127 个段各差一点点满**"
（实测：段利用率中位 **0.937**，其中 72/127 段 >90% 满），而 `empty_cache()` 的判定是**段级**的
—— 段里只要还有 1 个活块，整段都还不掉。实测对照：

| 配置 | 周期清理的效果 | `reserved` 走势 |
|---|---|---|
| `expandable_segments:False` | **释放 0 字节**（30 次清理白做） | 32 → **37.3** GiB 单向爬到墙 |
| **`expandable_segments:True`（+ 关 custom AR）** | **真的降**：`reserved 32.9→25.7`、碎片 `10.1→2.3`、`free 5.5→12.1` | **25.7 ↔ 28.8** 往复 |

### 代价（实测，可接受）

| 代价 | 数值 |
|---|---|
| 每层 allreduce 由 custom AR 改为 NCCL | **0–13 µs**（10 KiB: −0.9 µs；120 KiB: +13.1 µs；多为 1.8–6 µs） |
| KV 容量 | 1,221,977 → **1,070,006 tokens**（仍 **2.04×** 512K） |

### 历史印证

`scripts/serve_glm53_mainline.sh` 早前就默认"`expandable_segments:True` + `--disable-custom-all-reduce`"，
其 `:299` 注释记录的同型事故是："PyTorch 已分配 **37.90 GiB**、另有 **695 MiB reserved-but-unallocated**"
—— 与本手册描述的碎片形态一致。此次是在 A100 生产口径上把**机制**补全：不是 KV connector 的问题，
而是 **VMM 分配与 IPC 句柄互斥**。

### 如何验证 / 关闭

```bash
# 启用（推荐）：碎片可回收 + NCCL allreduce
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
NO_CUSTOM_AR=1 bash scripts/serve_v41.sh

# 回到旧行为（custom AR + 不可回收的碎片）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False NO_CUSTOM_AR=0 bash scripts/serve_v41.sh
```

## ⛔ 2026-10-31 勘误：显存守卫已**整体删除**

本节以上所有关于"软限/硬限/淘汰空闲条目"的描述**已作废**，仅作历史留档。

* **删除内容**：`vllm_xiaotu_moe/mem_guard.py`（整个模块）及其两处调用点
  （`gpu_prefill.py` 的 60 s 遥测 tick、`prefetch_slot.alloc` 的 OOM 取证）；
  `XIAOTU_MEM_GUARD` / `XIAOTU_MEM_SOFT_*` / `XIAOTU_MEM_HARD_*` / `XIAOTU_MEM_IDLE_STEPS` /
  `XIAOTU_MEM_MIN_FRAG_GIB` / `XIAOTU_MEM_HARD_EXIT` 一并废弃，**不再读取**。
* **原因**：守卫的兜底路径会调用 `torch.cuda.empty_cache()`，实测**会让仍在执行的 kernel 访存错误**；
  显存兜底属于引擎职责，**由 vLLM 自己处理**，插件不再介入。
* **影响**：`GPU_UTIL` 仍按原语义给 vLLM 划预算；启动期余量薄时**不再有插件侧 gc 兜底**，
  若显存不足，表现为 vLLM 自己的 OOM 报错（信息量更大、不会污染正在执行的计算）。
