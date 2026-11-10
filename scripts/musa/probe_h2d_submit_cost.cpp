// 独立 H2D 提交成本微基准（纯 C++/MUSA，**不** import torch、**不** 启 vLLM、**不** 载权重）
//
// 【为什么需要它】台账 §9.15：`[gpf-layer] dma_cpu` = 260–610 ms/层、`dma_calls` = 66–74/层。
//   `dma_cpu` 只是 `_dma_hostbuf_range()` 的墙钟包皮，内里仅一次
//   `musaMemcpyAsync(H2D)`（引擎走的是 `spitch==dpitch==width` 的**线性**分支）。
//   它可能是三件事之一，必须分开：
//     (a) 纯 CPU 提交开销；
//     (b) 异步队列满后的**阻塞节流**；
//     (c) 传输本身（PCIe 带宽受限）。
//   判据（写死，免得事后解释）：
//     * 单次 ≈ size/17.9GB/s            ⇒ (c) 带宽受限 ⇒ 只能减字节
//     * 单次 ≪ 该值 且 总时长≈传输时间  ⇒ (a) 提交便宜 ⇒ 问题在别处
//     * 后段单次明显 > 前段 或 总时长 ≫ 传输时间 ⇒ (b) 队列节流 ⇒ 减提交次数/加队列深度才有效
//
// 【纪律】局部验证纪律：不启动完整服务、不加载模型、默认用 GPU1（生产 qfn 在 GPU0）。
//   只读探测；不写仓库任何文件。
//
// 编译/运行（在容器内，因为 MUSA 头/库只在容器里）：
//   g++ -O2 -o /tmp/h2dsub probe_h2d_submit_cost.cpp -I/usr/local/musa/include \
//       -L/usr/local/musa/lib -lmusart -lpthread
//   MUSA_VISIBLE_DEVICES=1 SIZE_MB=36.5 N=74 /tmp/h2dsub
// License: Apache-2.0
#define _GNU_SOURCE
#include <musa_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <vector>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define CK(x)                                                              \
  do {                                                                     \
    musaError_t _e = (x);                                                   \
    if (_e != musaSuccess) {                                                \
      fprintf(stderr, "FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #x,     \
              musaGetErrorString(_e));                                      \
      return 1;                                                             \
    }                                                                       \
  } while (0)

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char **argv) {
  double size_mb = 36.5;
  int ncall = 74;
  int rounds = 5;
  double bw_ref = 17.9;  // GB/s，台账 gpf-dmabench 实测
  if (getenv("SIZE_MB")) size_mb = atof(getenv("SIZE_MB"));
  if (getenv("N")) ncall = atoi(getenv("N"));
  if (getenv("ROUNDS")) rounds = atoi(getenv("ROUNDS"));
  if (getenv("BW_REF")) bw_ref = atof(getenv("BW_REF"));

  int dev = 0;
  CK(musaSetDevice(dev));
  musaDeviceProp prop;
  CK(musaGetDeviceProperties(&prop, dev));
  size_t nbytes = (size_t)(size_mb * 1024.0 * 1024.0);
  printf("[probe] device=%s  size=%.2f MB  calls=%d  rounds=%d\n", prop.name,
         size_mb, ncall, rounds);

  // 宿主源：优先"锁页"，因为引擎 host_wbuf 是 pin 过的
  void *h = NULL;
  bool pinned = true;
  if (musaHostAlloc(&h, nbytes, musaHostAllocDefault) != musaSuccess) {
    pinned = false;
    h = malloc(nbytes);
    if (!h) { fprintf(stderr, "FATAL host alloc\n"); return 1; }
  }
  memset(h, 1, nbytes);
  printf("[probe] host buf %s; 同一批 %.0f MiB/轮；理论 @%.1fGB/s => %.2f ms/次\n",
         pinned ? "PINNED(musaHostAlloc)" : "PAGEABLE(malloc)", size_mb * ncall,
         bw_ref, size_mb / bw_ref * 1e3);

  // 4 个目的缓冲轮转（模拟引擎多槽，避免同地址强制串行）
  // ⭐ NDST 可调：引擎实测只有 nslots=2 个 tile 环槽 ⇒ 想验证"槽数是否就是 17.9 vs 52 GB/s 的成因"
  const int ND = getenv("NDST") ? atoi(getenv("NDST")) : 4;
  void *d[64] = {0};
  if (ND < 1 || ND > 64) { fprintf(stderr, "FATAL NDST 1..64\n"); return 1; }
  printf("[probe] NDST=%d（目的缓冲轮转数；引擎实际 nslots=2）\n", ND);
  for (int i = 0; i < ND; ++i) CK(musaMalloc(&d[i], nbytes));

  musaStream_t st;
  CK(musaStreamCreateWithFlags(&st, musaStreamNonBlocking));

  std::vector<double> totals, submits, first5, last5, syncs;
  std::vector<double> percall;

  // ⭐ LOAD=1：在**另一条流**上持续跑设备侧内核，模拟"引擎里 DMA 与 GEMM 并发"。
  //   目的（台账 §9.16-E）：probe 纯 DMA 能到 52 GB/s，引擎实测只有 21 GB/s，
  //   剩下的唯一候选就是"计算与拷贝争用"。若加上 LOAD 后掉到 ~21 GB/s ⇒ 假设成立。
  //   不依赖 BLAS：用大块 musaMemsetAsync（设备侧内存内核，持续占用 SM/显存带宽）
  //   在 load 流上反复发射，制造真实设备负载。
  const int LOAD = getenv("LOAD") ? atoi(getenv("LOAD")) : 0;
  musaStream_t st_load;
  CK(musaStreamCreateWithFlags(&st_load, musaStreamNonBlocking));
  const size_t load_bytes = (size_t)512 * 1024 * 1024;  // 512 MiB 目标
  void *dload = NULL;
  if (LOAD) {
    CK(musaMalloc(&dload, load_bytes));
    printf("[probe] LOAD=1：在另一条流上以 512MiB memset 制造设备侧负载（模拟 GEMM 争用）\n");
    // 预热负载流
    for (int i = 0; i < 4; ++i)
      CK(musaMemsetAsync(dload, i, load_bytes, st_load));
    CK(musaStreamSynchronize(st_load));
  }

  for (int r = -1; r < rounds; ++r) {  // r=-1 预热
    std::vector<double> pc(ncall, 0.0);
    double t0 = now_ms();
    for (int i = 0; i < ncall; ++i) {
      // LOAD=1：每次 H2D 之前，先在 load 流上再压一发设备内核（保持负载持续存在）
      if (LOAD) CK(musaMemsetAsync(dload, i & 0xff, load_bytes, st_load));
      double a = now_ms();
      CK(musaMemcpyAsync(d[i % ND], h, nbytes, musaMemcpyHostToDevice, st));
      pc[i] = now_ms() - a;
    }
    double t_submit_done = now_ms();
    CK(musaStreamSynchronize(st));
    if (LOAD) CK(musaStreamSynchronize(st_load));
    double t1 = now_ms();
    if (r < 0) continue;

    double sum = 0.0;
    for (int i = 0; i < ncall; ++i) sum += pc[i];
    totals.push_back(t1 - t0);
    submits.push_back(sum);
    syncs.push_back(t1 - t_submit_done);
    std::vector<double> f(pc.begin(), pc.begin() + 5);
    std::vector<double> l(pc.end() - 5, pc.end());
    first5.push_back(median(f));
    last5.push_back(median(l));
    percall.insert(percall.end(), pc.begin(), pc.end());
  }

  double m_total = median(totals), m_sub = median(submits), m_sync = median(syncs);
  double m_first = median(first5), m_last = median(last5);
  double m_percall = median(percall);
  double mib = size_mb * ncall;
  double bw = (m_total > 0) ? (mib / 1024.0) / (m_total / 1e3) : 0.0;
  double theoretical = size_mb / bw_ref * 1e3;

  printf("\n%-9s %-9s %-9s %-9s %-9s %-11s\n", "total_ms", "submit_ms",
         "sync_ms", "first5_ms", "last5_ms", "percall_ms");
  printf("%-9.2f %-9.2f %-9.2f %-9.3f %-9.3f %-11.3f\n", m_total, m_sub, m_sync,
         m_first, m_last, m_percall);
  printf("\n[probe] 每轮 %.0f MiB 墙钟 %.1f ms => 有效 %.2f GB/s\n", mib, m_total, bw);
  printf("[probe] 单次提交中位 %.3f ms ；同尺寸 @%.1fGB/s 理论 %.3f ms\n",
         m_percall, bw_ref, theoretical);

  const char *verdict;
  if (m_last > 2.0 * m_first && (m_last - m_first) > 0.5)
    verdict = "(b) 队列节流：后段单次提交显著变慢";
  else if (m_sub < 0.5 * m_total && m_sync > 0.3 * m_total)
    verdict = "(a) 提交便宜、时间花在等待完成 => 带宽/依赖受限 (c)";
  else if (m_sub > 0.7 * m_total)
    verdict = "(a)/(b) 提交本身就吃掉大部分墙钟 => 减提交次数是杠杆";
  else
    verdict = "(c) 传输受限：只能减字节或提带宽";
  printf("[probe] 判定：%s\n", verdict);
  printf("[probe] 折算：%d 次/层 × 36 层 = %.2f s/请求（若与引擎 260–610ms/层 对得上则假设成立）\n",
         ncall, (m_total / 1e3) * 36.0);

  if (pinned) musaFreeHost(h); else free(h);
  for (int i = 0; i < ND; ++i) CK(musaFree(d[i]));
  CK(musaStreamDestroy(st));
  return 0;
}
