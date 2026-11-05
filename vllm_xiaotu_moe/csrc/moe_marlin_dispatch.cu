// 槽位适配层：生产路由 [T,H]+ids  <->  槽位式融合内核 [E,SLOT,H]
// 全程 GPU、无主机同步（生产内核也是这个前提）。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#define SLOT 16

__global__ void zero_i(int* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0;
}

// 1) 分配槽位：每个 (t,k) 取该专家内序号；>=SLOT 则丢弃（slot 号仍写回，供判界）
__global__ void place_kernel(const int* __restrict__ ids, const float* __restrict__ wts,
                             int TK, int* __restrict__ cnt,
                             int* __restrict__ slot_tok, float* __restrict__ slot_w,
                             int* __restrict__ slot_of, int TOPSEL, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= TK) return;
  int e = ids[i], s = atomicAdd(&cnt[e], 1);
  slot_of[i] = s;
  if (s < SLOT) { slot_tok[e * SLOT + s] = i / TOPSEL; slot_w[e * SLOT + s] = wts[i]; }
}

// 2) 抽 A：[T,H] -> [E*SLOT,H]，未命中槽位填 0
__global__ void gather_kernel(const __nv_bfloat16* __restrict__ x, int H,
                              const int* __restrict__ slot_tok,
                              __nv_bfloat16* __restrict__ As) {
  int es = blockIdx.y, h8 = blockIdx.x * blockDim.x + threadIdx.x;
  if (h8 >= (H >> 3)) return;
  int t = slot_tok[es];
  uint4 v = make_uint4(0, 0, 0, 0);
  if (t >= 0) v = __ldg(reinterpret_cast<const uint4*>(x + (int64_t)t * H + h8 * 8));
  *reinterpret_cast<uint4*>(&As[(int64_t)es * H + h8 * 8]) = v;
}

// 3) 散到 partial[t*TOPSEL+k, h]（无原子）
__global__ void scatter_kernel(const float* __restrict__ res, int H,
                               const int* __restrict__ ids, const float* __restrict__ wts,
                               const int* __restrict__ slot_of, int TOPSEL,
                               float* __restrict__ partial) {
  int tk = blockIdx.y, h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  int s = slot_of[tk];
  if (s >= SLOT) return;
  int e = ids[tk];
  partial[(int64_t)tk * H + h] = res[(int64_t)(e * SLOT + s) * H + h] * wts[tk];
}

// 4) 沿 topk 归约：[T*TOPSEL,H] -> [T,H]
__global__ void reduce_kernel(const float* __restrict__ partial, int H, int TOPSEL,
                              float* __restrict__ out) {
  int t = blockIdx.y, h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float a = 0.f;
  for (int k = 0; k < TOPSEL; ++k) a += partial[(int64_t)(t * TOPSEL + k) * H + h];
  out[(int64_t)t * H + h] = a;
}

extern "C" void e2e_dispatch(const void* x, const void* ids, const void* wts,
                             void* As, void* cnt, void* slot_tok, void* slot_w,
                             void* slot_of, int T, int TOPSEL, int E, int H,
                             cudaStream_t s) {
  int TK = T * TOPSEL;
  zero_i<<<(E + 127) / 128, 128, 0, s>>>((int*)cnt, E);
  place_kernel<<<(TK + 127) / 128, 128, 0, s>>>((const int*)ids, (const float*)wts, TK,
                  (int*)cnt, (int*)slot_tok, (float*)slot_w, (int*)slot_of, TOPSEL, E);
  dim3 g((H / 8 + 127) / 128, E * SLOT);
  gather_kernel<<<g, 128, 0, s>>>((const __nv_bfloat16*)x, H, (const int*)slot_tok,
                                  (__nv_bfloat16*)As);
}

extern "C" void e2e_combine(const void* res, const void* ids, const void* wts,
                            const void* slot_of, void* partial, void* out,
                            int T, int TOPSEL, int E, int H, cudaStream_t s) {
  int TK = T * TOPSEL;
  dim3 g1((H + 127) / 128, TK);
  scatter_kernel<<<g1, 128, 0, s>>>((const float*)res, H, (const int*)ids,
                                    (const float*)wts, (const int*)slot_of, TOPSEL,
                                    (float*)partial);
  dim3 g2((H + 127) / 128, T);
  reduce_kernel<<<g2, 128, 0, s>>>((const float*)partial, H, TOPSEL, (float*)out);
}
