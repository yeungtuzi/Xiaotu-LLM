// moe4_smallm.cu — 小 M 4-bit MXFP4 分组专家 GEMM（GPU 热专家路径）
//
// 来源与验证：docs/EXPERIMENTS.md B1099–B1103；原型 dev-docs/report/decode_pipeline_20261030/kernel/moe_smallm_proto.py
//   * 真实 checkpoint 权重 + 池内 K-major 布局：相对误差 1.17e-6（fp64 参照）
//   * 生产形状（T=6, topk=6, 6 个不同专家）：352 µs/层（CPU 引擎 871 µs、旧 GPU 路径 1272 µs）
//
// 布局（与 vllm_xiaotu_moe/expert_pool.py:221-229 的 planes 逐字一致，per-rank / TP 切分后）：
//   W13 [S, K/2, N]  u8   字节 = 2 个连续 k，低 nibble = 偶 k，高 nibble = 奇 k
//   S13 [S, K/32, N] u8   e8m0：scale = 2^(b-127)，每 32 个 k 一组
//   W2  [S, K/2, N]  u8   同上（down：K=I_local, N=H）
//   S2  [S, K/32, N] u8
//
// 构建：由 scripts/build_engine_variants.sh 用 nvcc 单独编译成 .o 并链入变体 .so
//       （BUILD_MOE4=1 时才编；NO_CUDA=1 跳过）。接线与审计见
//       dev-docs/report/decode_pipeline_20261030/KERNEL_INTEGRATION_PLAN.md

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

// 256 项字节表：每项 = 两个 bf16（低 nibble 值 | 高 nibble 值），一次查表得 2 个权重
__constant__ uint32_t c_lut[256];

constexpr int kMaxA = 8;   // 每个专家最多处理的 token 数（分段上界）

__device__ __forceinline__ float2 pair_from(uint32_t e) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&e);
  return __bfloat1622float2(v);
}

// 分组 MoE：gate_up。按 (token,expert) 槽位输出（SwiGLU 是非线性的，绝不能跨专家先求和）
template <int NB, int NT>
__global__ void moe4_gu_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ SC,
                               const __nv_bfloat16* __restrict__ X, float* __restrict__ Y,
                               const int* __restrict__ seg2exp, const int* __restrict__ seg_start,
                               const int* __restrict__ tok,
                               const int* __restrict__ slot_map, int slot_base,
                               int K, int N, int JK) {
  const int e = __ldg(seg2exp + blockIdx.z);
  if (e < 0) return;
  const int s = __ldg(seg_start + e), en = __ldg(seg_start + e + 1);
  int na = en - s; if (na > kMaxA) na = kMaxA;
  if (na <= 0) return;
  const int n = blockIdx.x * NT * NB + threadIdx.x * NB;
  if (n >= N) return;
  const int j0 = blockIdx.y * JK, j1 = min(j0 + JK, K >> 1);
  // expert id → 池内 plane 行号（PIN 池按**槽位**排布，不是按 expert 排布）。
  // slot_map 为空 = 恒等映射（离线/测试用）；<0 表示该专家不在池里，直接跳过。
  const int row = slot_map ? (__ldg(slot_map + e) - slot_base) : e;
  if (row < 0) return;
  const uint8_t* We = W + (int64_t)row * (K >> 1) * N;
  const uint8_t* Se = SC + (int64_t)row * (K >> 5) * N;

  float acc[kMaxA][NB], part[kMaxA][NB];
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[c][b] = 0.f; part[c][b] = 0.f; }

  int tv[kMaxA];
#pragma unroll
  for (int c = 0; c < kMaxA; ++c) tv[c] = (c < na) ? __ldg(tok + s + c) : -1;

  for (int j = j0; j < j1; ++j) {
    float xv[kMaxA][2];
#pragma unroll
    for (int c = 0; c < kMaxA; ++c) {
      if (tv[c] >= 0) {
        xv[c][0] = __bfloat162float(__ldg(X + (int64_t)tv[c] * K + 2 * j));
        xv[c][1] = __bfloat162float(__ldg(X + (int64_t)tv[c] * K + 2 * j + 1));
      } else { xv[c][0] = 0.f; xv[c][1] = 0.f; }
    }
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t*>(We + (int64_t)j * N + n));
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float2 wf = pair_from(__ldg(&c_lut[(wv >> (8 * b)) & 0xFF]));
#pragma unroll
      for (int c = 0; c < kMaxA; ++c) {
        part[c][b] = fmaf(wf.x, xv[c][0], part[c][b]);
        part[c][b] = fmaf(wf.y, xv[c][1], part[c][b]);
      }
    }
    if (((j - j0) & 15) == 15) {          // 32 个 k 一组（16 字节）
      const uint32_t sv = __ldg(reinterpret_cast<const uint32_t*>(Se + (int64_t)(j >> 4) * N + n));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const float sc = exp2f((float)((int)((sv >> (8 * b)) & 0xFF) - 127));
#pragma unroll
        for (int c = 0; c < kMaxA; ++c) { acc[c][b] = fmaf(sc, part[c][b], acc[c][b]); part[c][b] = 0.f; }
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
    if (c < na) {
#pragma unroll
      for (int b = 0; b < NB; ++b) atomicAdd(&Y[(int64_t)(s + c) * N + n + b], acc[c][b]);  // split-K 必须原子
    }
}

// 分组 MoE：down。按 top-k 权重累加到 token
template <int NB, int NT>
__global__ void moe4_dn_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ SC,
                               const float* __restrict__ X, float* __restrict__ Y,
                               const int* __restrict__ seg2exp, const int* __restrict__ seg_start,
                               const int* __restrict__ tok, const float* __restrict__ wts,
                               const int* __restrict__ slot_map, int slot_base,
                               int K, int N, int JK) {
  const int e = __ldg(seg2exp + blockIdx.z);
  if (e < 0) return;
  const int s = __ldg(seg_start + e), en = __ldg(seg_start + e + 1);
  int na = en - s; if (na > kMaxA) na = kMaxA;
  if (na <= 0) return;
  const int n = blockIdx.x * NT * NB + threadIdx.x * NB;
  if (n >= N) return;
  const int j0 = blockIdx.y * JK, j1 = min(j0 + JK, K >> 1);
  // expert id → 池内 plane 行号（PIN 池按**槽位**排布，不是按 expert 排布）。
  // slot_map 为空 = 恒等映射（离线/测试用）；<0 表示该专家不在池里，直接跳过。
  const int row = slot_map ? (__ldg(slot_map + e) - slot_base) : e;
  if (row < 0) return;
  const uint8_t* We = W + (int64_t)row * (K >> 1) * N;
  const uint8_t* Se = SC + (int64_t)row * (K >> 5) * N;

  float acc[kMaxA][NB], part[kMaxA][NB];
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[c][b] = 0.f; part[c][b] = 0.f; }

  int tv[kMaxA]; float wc[kMaxA];
#pragma unroll
  for (int c = 0; c < kMaxA; ++c) {
    tv[c] = (c < na) ? __ldg(tok + s + c) : -1;
    wc[c] = (c < na) ? __ldg(wts + s + c) : 0.f;
  }

  for (int j = j0; j < j1; ++j) {
    float xv[kMaxA][2];
#pragma unroll
    for (int c = 0; c < kMaxA; ++c) {
      if (c < na) {
        xv[c][0] = __ldg(X + (int64_t)(s + c) * K + 2 * j);      // act 按槽位索引
        xv[c][1] = __ldg(X + (int64_t)(s + c) * K + 2 * j + 1);
      } else { xv[c][0] = 0.f; xv[c][1] = 0.f; }
    }
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t*>(We + (int64_t)j * N + n));
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float2 wf = pair_from(__ldg(&c_lut[(wv >> (8 * b)) & 0xFF]));
#pragma unroll
      for (int c = 0; c < kMaxA; ++c) {
        part[c][b] = fmaf(wf.x, xv[c][0], part[c][b]);
        part[c][b] = fmaf(wf.y, xv[c][1], part[c][b]);
      }
    }
    if (((j - j0) & 15) == 15) {
      const uint32_t sv = __ldg(reinterpret_cast<const uint32_t*>(Se + (int64_t)(j >> 4) * N + n));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const float sc = exp2f((float)((int)((sv >> (8 * b)) & 0xFF) - 127));
#pragma unroll
        for (int c = 0; c < kMaxA; ++c) { acc[c][b] = fmaf(sc, part[c][b], acc[c][b]); part[c][b] = 0.f; }
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
    if (tv[c] >= 0) {
#pragma unroll
      for (int b = 0; b < NB; ++b) atomicAdd(&Y[(int64_t)tv[c] * N + n + b], wc[c] * acc[c][b]);
    }
}


// =====================================================================================
// 内核内 dispatch 版本（2026-10-31）：**不需要主机侧排序/分段**
//   直接吃 `topk_ids [T,K]`（PIN 池里是平面行号）与 `topk_weights`，
//   每个 block 负责一个"专家"（= 平面行号）e=blockIdx.z，块内先扫描一遍
//   ids 收集该 e 命中的 (token, 权重) 槽位，再按槽位做 GEMM。
//   ⇒ 省掉主机侧 argsort/gather/scatter/cumsum（实测 ~157 µs/层）。
//   inter 的行号 = e*kMaxA + c（槽位），dn 再按 token 累加。
// =====================================================================================
template <int NB, int NT>
__global__ void moe4_gu2_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ SC,
                                const __nv_bfloat16* __restrict__ X, float* __restrict__ Y,
                                const int* __restrict__ ids, int T, int TOPSEL,
                                int K, int N, int JK) {
  const int e = blockIdx.z;
  __shared__ int sh_tv[kMaxA];
  __shared__ int sh_na;
  if (threadIdx.x == 0) {
    int na = 0;
    for (int t = 0; t < T; ++t) {
      const int* row = ids + (int64_t)t * TOPSEL;
      for (int j = 0; j < TOPSEL; ++j) {
        if (__ldg(row + j) == e) {
          if (na < kMaxA) sh_tv[na] = t;
          ++na;
        }
      }
    }
    sh_na = na;
  }
  __syncthreads();
  int na = sh_na;
  if (na > kMaxA) na = kMaxA;          // 超界截断（与段表版同语义；Python 侧有 T<=8 守卫）
  if (na <= 0) return;
  const int n = blockIdx.x * NT * NB + threadIdx.x * NB;
  if (n >= N) return;
  const int j0 = blockIdx.y * JK, j1 = min(j0 + JK, K >> 1);
  const uint8_t* We = W + (int64_t)e * (K >> 1) * N;
  const uint8_t* Se = SC + (int64_t)e * (K >> 5) * N;

  float acc[kMaxA][NB], part[kMaxA][NB];
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[c][b] = 0.f; part[c][b] = 0.f; }

  for (int j = j0; j < j1; ++j) {
    float xv[kMaxA][2];
#pragma unroll
    for (int c = 0; c < kMaxA; ++c) {
      if (c < na) {
        xv[c][0] = __bfloat162float(__ldg(X + (int64_t)sh_tv[c] * K + 2 * j));
        xv[c][1] = __bfloat162float(__ldg(X + (int64_t)sh_tv[c] * K + 2 * j + 1));
      } else { xv[c][0] = 0.f; xv[c][1] = 0.f; }
    }
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t*>(We + (int64_t)j * N + n));
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float2 wf = pair_from(__ldg(&c_lut[(wv >> (8 * b)) & 0xFF]));
#pragma unroll
      for (int c = 0; c < kMaxA; ++c) {
        part[c][b] = fmaf(wf.x, xv[c][0], part[c][b]);
        part[c][b] = fmaf(wf.y, xv[c][1], part[c][b]);
      }
    }
    if (((j - j0) & 15) == 15) {
      const uint32_t sv = __ldg(reinterpret_cast<const uint32_t*>(Se + (int64_t)(j >> 4) * N + n));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const float sc = exp2f((float)((int)((sv >> (8 * b)) & 0xFF) - 127));
#pragma unroll
        for (int c = 0; c < kMaxA; ++c) { acc[c][b] = fmaf(sc, part[c][b], acc[c][b]); part[c][b] = 0.f; }
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
    if (c < na)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        atomicAdd(&Y[(int64_t)(e * kMaxA + c) * N + n + b], acc[c][b]);
}

template <int NB, int NT>
__global__ void moe4_dn2_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ SC,
                                const float* __restrict__ X, float* __restrict__ Y,
                                const int* __restrict__ ids, const float* __restrict__ wts,
                                int T, int TOPSEL, int K, int N, int JK) {
  const int e = blockIdx.z;
  __shared__ int sh_tv[kMaxA];
  __shared__ float sh_w[kMaxA];
  __shared__ int sh_na;
  if (threadIdx.x == 0) {
    int na = 0;
    for (int t = 0; t < T; ++t) {
      const int* row = ids + (int64_t)t * TOPSEL;
      const float* wrow = wts + (int64_t)t * TOPSEL;
      for (int j = 0; j < TOPSEL; ++j) {
        if (__ldg(row + j) == e) {
          if (na < kMaxA) { sh_tv[na] = t; sh_w[na] = __ldg(wrow + j); }
          ++na;
        }
      }
    }
    sh_na = na;
  }
  __syncthreads();
  int na = sh_na;
  if (na > kMaxA) na = kMaxA;
  if (na <= 0) return;
  const int n = blockIdx.x * NT * NB + threadIdx.x * NB;
  if (n >= N) return;
  const int j0 = blockIdx.y * JK, j1 = min(j0 + JK, K >> 1);
  const uint8_t* We = W + (int64_t)e * (K >> 1) * N;
  const uint8_t* Se = SC + (int64_t)e * (K >> 5) * N;

  float acc[kMaxA][NB], part[kMaxA][NB];
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
#pragma unroll
    for (int b = 0; b < NB; ++b) { acc[c][b] = 0.f; part[c][b] = 0.f; }

  for (int j = j0; j < j1; ++j) {
    float xv[kMaxA][2];
#pragma unroll
    for (int c = 0; c < kMaxA; ++c) {
      if (c < na) {
        xv[c][0] = __ldg(X + (int64_t)(e * kMaxA + c) * K + 2 * j);
        xv[c][1] = __ldg(X + (int64_t)(e * kMaxA + c) * K + 2 * j + 1);
      } else { xv[c][0] = 0.f; xv[c][1] = 0.f; }
    }
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t*>(We + (int64_t)j * N + n));
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float2 wf = pair_from(__ldg(&c_lut[(wv >> (8 * b)) & 0xFF]));
#pragma unroll
      for (int c = 0; c < kMaxA; ++c) {
        part[c][b] = fmaf(wf.x, xv[c][0], part[c][b]);
        part[c][b] = fmaf(wf.y, xv[c][1], part[c][b]);
      }
    }
    if (((j - j0) & 15) == 15) {
      const uint32_t sv = __ldg(reinterpret_cast<const uint32_t*>(Se + (int64_t)(j >> 4) * N + n));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const float sc = exp2f((float)((int)((sv >> (8 * b)) & 0xFF) - 127));
#pragma unroll
        for (int c = 0; c < kMaxA; ++c) { acc[c][b] = fmaf(sc, part[c][b], acc[c][b]); part[c][b] = 0.f; }
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kMaxA; ++c)
    if (c < na)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        atomicAdd(&Y[(int64_t)sh_tv[c] * N + n + b], sh_w[c] * acc[c][b]);
}

constexpr int kNB = 4;    // 必须 >= 4（uint32 载入需 4 字节对齐）
constexpr int kNT = 64;
constexpr int kKS = 32;   // K 维分片数

inline int split_bytes(int K) {
  const int H2 = K >> 1, per = (H2 + kKS - 1) / kKS;
  return ((per + 15) / 16) * 16;                  // 每片字节数，16 字节对齐
}

}  // namespace

extern "C" cudaError_t moe4_smallm_init() {
  static int g_inited = 0;
  if (g_inited) return cudaSuccess;
  float h[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                 -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  uint32_t e[256];
  for (int b = 0; b < 256; ++b) {
    __nv_bfloat16 lo = __float2bfloat16(h[b & 0xF]);
    __nv_bfloat16 hi = __float2bfloat16(h[(b >> 4) & 0xF]);
    const uint16_t l = *reinterpret_cast<uint16_t*>(&lo);
    const uint16_t hh = *reinterpret_cast<uint16_t*>(&hi);
    e[b] = (uint32_t)l | ((uint32_t)hh << 16);
  }
  const cudaError_t st = cudaMemcpyToSymbol(c_lut, e, sizeof(e));
  if (st == cudaSuccess) g_inited = 1;
  return st;
}

// nseg 必须 <= 段表长度；空段以 seg2exp=-1 短路（图捕获期形状固定用 nseg_max=T*topk）
extern "C" cudaError_t moe4_smallm_gu(const void* W, const void* SC, const void* X, void* Y,
                                      const int* seg2exp, const int* seg_start, const int* tok,
                                      const int* slot_map, int slot_base,
                                      int nseg, int K, int N, cudaStream_t stream) {
  // 不变量（只在 pybind 层查是不够的：本符号是导出的）
  if (K < 32 || K % 32 || N < 4 || N % kNB || nseg < 0) return cudaErrorInvalidValue;
  if (X == nullptr || Y == nullptr || seg2exp == nullptr || seg_start == nullptr || tok == nullptr)
    return cudaErrorInvalidValue;
  (void)cudaGetLastError();          // 清掉前序残留错误，避免嫁祸
  const int JK = split_bytes(K), ky = ((K >> 1) + JK - 1) / JK;
  const dim3 grid((N + kNB * kNT - 1) / (kNB * kNT), ky, nseg);
  moe4_gu_kernel<kNB, kNT><<<grid, kNT, 0, stream>>>(
      (const uint8_t*)W, (const uint8_t*)SC, (const __nv_bfloat16*)X, (float*)Y,
      seg2exp, seg_start, tok, slot_map, slot_base, K, N, JK);
  return cudaPeekAtLastError();      // 只代表 launch 合法；执行期错误在后续同步暴露
}

extern "C" cudaError_t moe4_smallm_dn(const void* W, const void* SC, const void* X, void* Y,
                                      const int* seg2exp, const int* seg_start, const int* tok,
                                      const float* wts,
                                      const int* slot_map, int slot_base,
                                      int nseg, int K, int N, cudaStream_t stream) {
  if (K < 32 || K % 32 || N < 4 || N % kNB || nseg < 0) return cudaErrorInvalidValue;
  if (X == nullptr || Y == nullptr || seg2exp == nullptr || seg_start == nullptr
      || tok == nullptr || wts == nullptr) return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  const int JK = split_bytes(K), ky = ((K >> 1) + JK - 1) / JK;
  const dim3 grid((N + kNB * kNT - 1) / (kNB * kNT), ky, nseg);
  moe4_dn_kernel<kNB, kNT><<<grid, kNT, 0, stream>>>(
      (const uint8_t*)W, (const uint8_t*)SC, (const float*)X, (float*)Y,
      seg2exp, seg_start, tok, wts, slot_map, slot_base, K, N, JK);
  return cudaPeekAtLastError();
}

// ---- 内核内 dispatch 入口（无主机侧分段）：grid.z = nexp（专家/平面行数）----
extern "C" cudaError_t moe4_smallm_gu2(const void* W, const void* SC, const void* X, void* Y,
                                       const int* ids, int T, int TOPSEL,
                                       int nexp, int K, int N, cudaStream_t stream) {
  if (K < 32 || K % 32 || N < 4 || N % kNB || T < 0 || TOPSEL < 0 || nexp < 1)
    return cudaErrorInvalidValue;
  if (W == nullptr || SC == nullptr || X == nullptr || Y == nullptr || ids == nullptr)
    return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  const int JK = split_bytes(K), ky = ((K >> 1) + JK - 1) / JK;
  const dim3 grid((N + kNB * kNT - 1) / (kNB * kNT), ky, nexp);
  moe4_gu2_kernel<kNB, kNT><<<grid, kNT, 0, stream>>>(
      (const uint8_t*)W, (const uint8_t*)SC, (const __nv_bfloat16*)X, (float*)Y,
      ids, T, TOPSEL, K, N, JK);
  return cudaPeekAtLastError();
}

extern "C" cudaError_t moe4_smallm_dn2(const void* W, const void* SC, const void* X, void* Y,
                                       const int* ids, const float* wts, int T, int TOPSEL,
                                       int nexp, int K, int N, cudaStream_t stream) {
  if (K < 32 || K % 32 || N < 4 || N % kNB || T < 0 || TOPSEL < 0 || nexp < 1)
    return cudaErrorInvalidValue;
  if (W == nullptr || SC == nullptr || X == nullptr || Y == nullptr || ids == nullptr
      || wts == nullptr) return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  const int JK = split_bytes(K), ky = ((K >> 1) + JK - 1) / JK;
  const dim3 grid((N + kNB * kNT - 1) / (kNB * kNT), ky, nexp);
  moe4_dn2_kernel<kNB, kNT><<<grid, kNT, 0, stream>>>(
      (const uint8_t*)W, (const uint8_t*)SC, (const float*)X, (float*)Y,
      ids, wts, T, TOPSEL, K, N, JK);
  return cudaPeekAtLastError();
}
