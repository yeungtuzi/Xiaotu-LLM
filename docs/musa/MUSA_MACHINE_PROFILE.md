# MUSA 开发机档案（mtts4000）

> 本文件只记录 **MUSA/S4000 开发机**。CUDA 工作继续在远端 EPYC 机器上做，
> 不要把 CUDA 的构建/测试/权重路径混到本机。
> BMC 凭据不写入仓库；只记录地址。

## 1. 角色与机器边界

| 机器 | 角色 | 入口 | 计算路径 |
|---|---|---|---|
| **mtts4000** | MUSA / S4000 正确性开发机 | `192.168.100.248`（Mellanox） | `GPU_DEVICE=MUSA` |
| 远端 EPYC + A100 | CUDA 参考 / 既有性能工作 | 既有远端环境 | `GPU_DEVICE=CUDA`（默认） |

规则：

- MUSA 线只在 **mtts4000** 上跑；
- CUDA 线只在远端 EPYC 上跑；
- 不在这台机器上安装/运行 CUDA vLLM 栈；
- 不在 EPYC 上跑 MUSA/`torch_musa`；
- 默认 `GPU_DEVICE=CUDA`，MUSA 必须显式 `GPU_DEVICE=MUSA`。

## 2. 硬件

```text
主机名:      mtts4000
主板:        Suma X7840H0
BIOS:        CDSH051022 / 2023-09-14
CPU:         2x Hygon C86 7385 32-core（64 核 / 128 线程）
内存:        512 GiB DDR4-2933（实际可用约 503 GiB）
GPU:         2x MTT S4000 48GB
  GPU0:      0000:3d:00.0
  GPU1:      0000:bd:00.0
网卡:        Mellanox ConnectX-6 Lx（MT2894，2 口）
  在用:      enp33s0f0np0 / 192.168.100.248
  be2net:    已拆除（曾出现 FW not responding）
BMC:         192.168.100.238（HTTPS/Redfish；凭据不入库）
```

## 3. OS / 内核 / BIOS

```text
OS:       Ubuntu 22.04.4 LTS
Kernel:   5.15.0-105-generic（GRUB 已固定）
Secure Boot: Off
```

BIOS（通过 BMC Redfish 读取）：

```text
Above4GDecoding   = Enabled
CbsCmnIOMMU       = Enabled
CbsCmnSVMControl  = Enabled
NumaMode          = Enabled
```

GPU BAR：

```text
Region 2: 64G prefetchable
Physical Resizable BAR 存在
```

结论：Above 4G / IOMMU / ReBAR 均已开启；不要再把它们当故障根因。

## 4. CPU / NUMA 拓扑

```text
Socket:        2
NUMA nodes:    2
node0:         0-31,64-95    约 256 GiB
node1:         32-63,96-127  约 256 GiB
node distance: 10 / 22
```

- 每 socket 32 核 = 8 个 CCX × 4 核；
- 每 CCX 共享 8MB L3；
- 每 socket 4 个 die，每 die 2 个 DDR4 通道 ⇒ 8 通道/socket，16 通道/整机；
- 当前是 socket 级 NPS1 形态；BIOS 没有暴露 NPS1/2/4，只有 `NumaMode`。

## 5. MUSA / Python 软件栈

```text
MUSA driver:       5.2.0-server
musa-toolkit:      5.2.0
libmudnn3:         3.4.0.0
MCCL:              mccl-s4000 2.4.0
mtshmem:           installed
Python:            3.12.15
venv:              /root/venvs/vllm
torch:             2.11.0.post2+musa5.2.0
torch_musa:        2.11.0.post2+musa5.2.0
modelscope:        1.40.2
```

已知状态（2026-10-21）：

```text
torch_musa GPU 验证仍未通过
当前 import torch 报:
  Failed to load the backend extension: torch_musa
需要继续排查，不能把 GPU 路径标记为可用。
```

## 6. CPU 引擎

已构建：

```text
/root/lvllm/xiaotu_moe/build/_xiaotu_moe_C_avx2.cpython-312-x86_64-linux-gnu.so
```

CPU 只有 AVX2（无 AVX512），与 roadmap 的 AVX2→AVX512→AMX 分发一致。

qfn-shape CPU golden：

```text
E=16 H=2560 I=640 T=1 top_k=10
max_abs = 0.176422
rms     = 279.707
rel     = 0.0631%
PASS
```

引擎 NUMA 识别：

```text
[xiaotu/shard] unit=node rank_split=1 (world=1 nodes=2)
```

## 7. 内存带宽（bwprobe，8GB 单 buffer）

方法：`scripts/bwprobe.c`，每 64B 取一个 u64，顺序流式读；与 EPYC 750GB/s 同一口径。

| 线程 | 绑定 | GB/s |
|---:|---|---:|
| 1 | node | 15.5 |
| 24 | node | 102.9 |
| 60 | node | 100.1 |
| 128 | node | 168.1 |
| 128 | flat | **285.5** |

理论：

```text
2 socket × 8 channel × 2933 MT/s × 8 B = 375.4 GB/s
285.5 / 375.4 = 76%
```

EPYC 参考：

```text
2 × 12 channel × 4800 × 8 = 921.6 GB/s
750 / 921.6 = 81%
```

结论：本机达到理论带宽的 ~76%，与 EPYC 的 ~81% 同档；绝对低是 DDR4-2933 + 老平台导致，不是 NUMA/测试错误。

注意：`BIND=node` 只有一块 8GB buffer、由主线程 first-touch 在 node0，
所以 node 绑定会让一半线程跨 node；不代表每 socket 本地分片上限。

## 8. 模型下载

目标：

```text
Qwen/Qwen3.8-Flash-Next-FP8
/root/models/Qwen3.8-Flash-Next-FP8
```

下载由 systemd 托管：

```text
xtu-model-download.service
--max-workers 2
```

截至记录时约 121GB，仍在进行；不要删除 `.incomplete`。

## 9. 已知问题（按优先级）

### P0：mlx5 ICO 软锁死

高并发下载时曾出现：

```text
mlx5_core ... TX timeout
watchdog: BUG: soft lockup - CPU#21
RIP: mlx5e_poll_ico_cq
```

随后整机用户态卡死。当前下载降到 2 workers，暂无复现。

已下载但**未安装**：

```text
mlnx-ofed-kernel-dkms
mlnx-ofed-kernel-utils
mlnx-tools
mlnx-ethtool
```

安装会短暂断网，必须等模型下载完成；需要重启时会提前告知。

### P0：torch_musa backend load 失败

```text
Failed to load the backend extension: torch_musa
```

在 GPU 路径未修好前，不做 H2D ping/pong、不做 8K。

### P1：MUSA 补丁链

已有 6 个 qwen4 补丁，仍需实机验证：

- 平台分派；
- PDL/GDC；
- QSA top-k；
- QSA partial stages。

## 10. 下一步

1. 完成模型下载；
2. 排查 `torch_musa` 加载失败（先 `MUSA_VISIBLE_DEVICES=""` 分层）；
3. 安装 OFED kernel-only，必要时一次计划内重启；
4. GPU 稳定后跑 H2D ping/pong；
5. 再进入 qwen4_exp / 8K。
