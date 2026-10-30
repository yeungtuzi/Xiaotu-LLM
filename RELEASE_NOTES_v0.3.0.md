# v0.3.0 —— 稳定性修复：通过 4 小时满载压力测试；初步支持摩尔线程 MUSA（MP_22）

> 主题：**修掉一个潜伏已久、导致服务每隔 1–4 小时必崩的显存管理缺陷**，
> 并用 **4 小时满载压测**给出可复现的验证；同时把 **摩尔线程 MUSA（MTT S4000 ×2，MP_22）**
> 纳入同一份代码的**初步支持**范围。
>
> 说明：v0.3.0 这一版本号在 2026-10-24 曾用于"挎斗（辅助 GPU）"功能并整体撤回
> （tag 与 Release 均已删除）。本版为**重新以该号发布**，内容完全不同；
> 撤回记录保留在 [`RELEASE_NOTES_v0.3.0-sidecar-WITHDRAWN.md`](RELEASE_NOTES_v0.3.0-sidecar-WITHDRAWN.md)。

---

## 一、稳定性修复（本版核心）

### 1.1 现象

生产配置（DeepSeek-V4.1-Flash / TP=2 / A100 ×2）在满载压测下**必然在 24 分钟至 3 小时 41 分之间崩溃**，
连续 8 个版本无一例外。崩溃时的表象五花八门：

| 表象（谁先报错） | 出现次数 |
|---|---|
| `[xtu-vram] empty_cache 失败: illegal memory access` | 5 |
| Triton `_init_handles` / `launch_metadata` 抛 `Triton Error [CUDA]` | 3 |
| 分配器 `insert_events`（`~TensorImpl`）抛 `cudaErrorIllegalAddress` | 1 |
| 无报错，但 worker 挂住（`shm_broadcast` 60 s 超时） | 1 |

> 这些表象**都不是根因**。CUDA 的非法访存是**异步且粘性**的：肇事 kernel 早已返回，
> 错误被记在 context 上，等**下一个** CUDA API 调用才爆出来 —— 所以栈顶永远只是"下一个碰到错误的调用"。

### 1.2 定位

关键证据来自**内核日志**（`/var/log/kern.log`，`adm` 组可读；`dmesg` 需 root）：

```
NVRM: Xid (PCI:0000:c1:00): 31, pid=..., name=VLLM::Worker, channel 0x...
  MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_T1_0 faulted @ 0x12c_74000000
  Fault is of type FAULT_PDE  ACCESS_TYPE_VIRT_READ
```

* `FAULT_PDE` = 页目录项不存在；`ACCESS_TYPE_VIRT_READ` = **读**；
* 故障地址落在 80–330 GB 的虚拟地址区间 —— 正是 `expandable_segments:True` 的 **CUDA VMM** 预留区；
* 8 次崩溃对应 8 条 `Xid 31`，形态完全一致；其中两次的故障地址**一字不差**（`0x12c_74000000`）。

⇒ 真正发生的事：**某个 kernel 去读一块已经被解除映射（unmapped）的显存**。

### 1.3 根因

在 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 下，`torch.cuda.empty_cache()`
做的不再是 CUDA 传统语义的"把空闲块还回缓存池"，而是**拆除空闲段的 VMM 页表**。
本项目存在**裸指针缓冲**（不在 PyTorch allocator 的流追踪范围内），而当时的后台清理线程
采用"`synchronize()` 之后再 `empty_cache()`"的写法：

* `synchronize()` **不是锁** —— 它返回后，计算线程可以立刻提交新的 kernel；
* 实测单次 `synchronize()` 耗时可达 **4.4 秒**，窗口极大；
* 一旦解映射与"仍有 kernel 在读该段"重叠 ⇒ `Xid 31` ⇒ context 粘性中毒 ⇒ 服务不可用。

这也解释了此前所有反常观察：**单次超长上下文（150K / 333K / 40 万 token）压不死**（需要清理线程与在飞 kernel 撞窗口），
而**跑够时间必然崩**（碎片越多、被解映射的段越多、撞上的概率越高）。

### 1.4 修复

**运行期不再自主回收显存，显存管理交给 vLLM / PyTorch allocator 的按需路径**
（分配无法满足时 allocator 自己会释放缓存段，且发生在知道"确实需要内存"的线程里）。

| 位置 | 变更 |
|---|---|
| `vllm_xiaotu_moe/gpu_prefill.py` 后台清理线程 | `XIAOTU_EMPTY_CACHE_EVERY_S` **默认 300 → 0**（默认不启动） |
| `vllm_xiaotu_moe/mem_guard.py` 兜底 | 收进 `XIAOTU_GUARD_EMPTY_CACHE=1`，**默认关** |
| `vllm_xiaotu_moe/mixed_experts.py` 预取 OOM 分支 | 不再 `empty_cache()`，直接跳过本轮预取 |
| `vllm_xiaotu_moe/gpu_prefill.py` 启动期释放 staging 占位 | **保留**（此时无请求在飞、无裸指针引用，属"释放自己的占位"） |

改动是**默认更安全**的方向：即使启动时不传任何环境变量，也不会再开清理线程。

### 1.5 验证（4 小时满载）

同一负载、同一配置，唯一差异 = 关闭自主回收：

| 实例 | 自主回收 | 存活 | `Xid 31` 新增 |
|---|---|---|---|
| r108 | 开 | 24 分钟 | 1 |
| r103 / r104 | 开 | 43 / 47 分钟 | 1 / 1 |
| r107 | 开 | 1 小时 12 分 | 1 |
| r102 | 开 | 2 小时 30 分 | 1 |
| r106 | 开 | 2 小时 53 分 | 1 |
| r105 | 开 | 3 小时 18 分（挂住） | 1 |
| r101 | 开 | 3 小时 41 分 | 1 |
| **r110（本版）** | **关** | **4 小时 38 分以上，全程无崩溃** | **0（内核计数保持不变）** |

压测条件：请求长度按真实分布随机（60% 1K–8K / 25% 8K–40K / 15% 40K–300K token），
`max_tokens` 随机 1–512，10% 轮次双路并发，累计 **2100+ 轮**全部成功；
两卡显存始终对称（差值 ≤ 2 MiB），温度 42–48 °C，无降频、无 Xid、无 ECC 错误。

### 1.6 配套的工具与看门狗修复

* `scripts/poison_watch.sh`：**启动宽限期**（默认 420 s）+ **软命中不退出** ——
  旧版会被启动期那条良性的 `shm_broadcast` 超时骗退，导致真正出事的时刻无人看守；
  新增对 worker **挂住**（`No available shared memory broadcast block`）的当场抓栈。
* `scripts/imbalance_watch.sh`（新增）：两卡**显存/利用率不平衡**持续超阈值时抓
  `nvidia-smi` + 两个 worker 的 py-spy 栈（只读，不杀进程）。
* `scripts/metricsmon.sh`（新增）：按分钟采样 `num_requests_waiting`、`kv_cache_usage_perc`、
  `num_preemptions_total`、累计 prompt/generation tokens。
* `scripts/resmon.sh`（新增）：48 列资源采样（VMA / fd / 线程 / pinned / nvidia 映射 /
  `/dev/shm` / 中断 / `CommitLimit` vs `Committed_AS` / PSI / 每卡显存·功耗·温度·降频）。
* 诊断脚本：`spec_sweep.py`（特化扫描）、`state_probe.py`（形状×占用）、
  `tail_chunk_probe.py`（小尾块+长上下文）、`vary_soak.py`（高多样性压测）。

---

## 二、摩尔线程 MUSA（MP_22）初步支持

同一份代码同时支持 CUDA 与 MUSA，平台判定**单点**收敛在 `musa_correctness.py` 读取 `GPU_DEVICE`，
不保留平台专用实现副本、不允许平台专用依赖（`scripts/check_single_code.sh` 可持续检查）。

**支持范围（初步）**：**QFN 官方量化权重**在 **MTT S4000 双卡（MP_22）+ 海光处理器**上可完整跑通，
混合推理路径（专家权重驻留主机内存、prefill 走 tile-ring + GPU、decode 走 CPU 引擎）功能与数值正确。

**限定说明（重要）**：本版对 MUSA 的定位是「能完整跑通、数值与流程正确」，**不承诺性能**；
实测性能仍明显低于 roofline，瓶颈已定量但尚未定位。

## 三、安装

```bash
pip install ./xiaotu_llm-0.3.0-cp312-cp312-manylinux_2_34_x86_64.whl   # 需已安装 vLLM
```

* 平台/ABI：**CPython 3.12 · x86_64 · glibc ≥ 2.34**（`manylinux_2_34`）；
* 内含 11 个 CPU 引擎变体（`scalar` / `avx2` / `avx512_base` / `avx512_vnni` /
  `avx512_bf16` / `avx512_bf16_vbmi` 及其 `_pf*` 微变体），运行时按 CPU 能力**自动选择**；
* checksum：`sha256:84207cb93adc302810e184c66c5d547bf5127aae4a0979dfe5851da2909e3f43`；
* ⚠️ **包名变更**：项目自 2026-10-27 起正式名为 **Xiaotu-LLM**，因此 wheel 文件名由
  `vllm_xtu_moe-*` 变为 **`xiaotu_llm-*`**（Python 包名仍是 `xiaotu_moe` / `vllm_xiaotu_moe`，
  插件入口不变，`pip uninstall vllm-xtu-moe` 后安装本包即可）。

---

## 四、升级说明

* 本版**不需要**改任何配置即可获得稳定性修复（清理线程默认不再启动）。
* 若确需诊断用途的周期性回收，显式设置 `XIAOTU_EMPTY_CACHE_EVERY_S=<秒>`（会解映射 VMM 页，慎用）。
* 建议部署后把 `/var/log/kern.log` 的 `Xid` 计数纳入监控——这是 GPU 侧最直接的故障证据来源。

---

## 五、已否证的假设（留档，避免重复走弯路）

崩溃原因曾先后被怀疑为：① 预热不足；② CUDA 惰性模块加载；③ 侧流预取跨流释放；
④ 新 Triton 特化首次加载（30 个构造扫描）；⑤ 新特化 × 高占用（27.8% KV + 40 万上下文 decode）；
⑥ 小尾块 + 长上下文（`8192k+q` 精确构造）。
以上六项均以**独立实验否证**，最终由内核日志的 `Xid 31 / FAULT_PDE / VIRT_READ` 定案。

---

## 六、勘误 · 2026-10-30：稳定性时长口径由 4 小时更新为 12 小时

> 本节为**追加勘误**，不改写上文任何记录（上文 r101–r110 的存活时间与结论是当时事实）。

后续在**关闭运行时自主 `empty_cache`** 的版本上继续压测，实例 **r115** 达成更长连续运行：

| 项 | 值 |
|---|---|
| 实例 / 版本 | r115（含本版"运行时不再自主 `empty_cache`"改动） |
| 口径 | 单流（单请求串行）、TP=2、`--max-ctx-tokens 200000` |
| 连续存活 | **12 小时 43 分**（2026-10-30 03:21:46 → 16:04 CST） |
| 致命信号 | **0**（无 `Xid 31`、无 `illegal memory access`） |
| 显存 | 两卡全程对称，峰值 **38.9 GB / 40 GB**，无加速爬升 |
| 退化 | 有效应答 278 样本，真退化 **0**（`finish_reason=length` 的截断不计） |
| 期间 | 后段由真实业务流量接入（`running=2`），未出现异常 |

因此 **README 首屏的稳定性口径已同步更新为"可稳定通过 12 小时满载压测"**；
上文 §1.5 标题中的「4 小时」保留为**当时的历史记录**。

⚠️ 说明：12 小时 ≠ 24 小时。§1.5 的 r110 与本节 r115 均为**单实例连续运行**结果，
未做 24 小时长稳；L2 判定（72 小时）仍未达成。
