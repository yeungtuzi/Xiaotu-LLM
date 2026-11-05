
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#define KC 64
#define WARPS 4
#define TPW 2
#define NTILE (16 * WARPS * TPW)      // 128 真实列
#define NT2 (NTILE / 2)               // 64 个 uint16 元素/行（每元素=2 列）
#define PADW 8

__device__ __forceinline__ unsigned short e2m1_bits(int v) {
  int e = (v >> 1) & 3, m = v & 1, s = (v >> 3) & 1;
  unsigned short b = (e == 0) ? (unsigned short)(m ? 0x3F00 : 0x0000)
                              : (unsigned short)(((126 + e) << 7) | (m ? 0x40 : 0));
  return (unsigned short)(b | (s ? 0x8000 : 0));
}
__launch_bounds__(128, 12) __global__ void gemm(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ W,
                     const uint8_t* __restrict__ SC, float* __restrict__ C,
                     int na, int K, int N, int NE) {
  __shared__ uint16_t sB[KC/2][NT2 + PADW];   // 每元素 = 相邻两列的原始字节（不扩展）
  __shared__ __nv_bfloat16 sA[16][KC + 8];
  __shared__ uint32_t lut[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    lut[i] = (uint32_t)e2m1_bits(i & 0xF) | ((uint32_t)e2m1_bits((i >> 4) & 0xF) << 16);
  __syncthreads();
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * NTILE, e = blockIdx.z;
  const int KSPLIT = gridDim.y;
  const int kPer = (K / KC + KSPLIT - 1) / KSPLIT * KC;
  const int kb0 = blockIdx.y * kPer, kb1 = min(K, kb0 + kPer);
  const uint8_t* We  = W  + (int64_t)e * (K/2) * N;
  const uint8_t* Sce = SC + (int64_t)e * (K/32) * N;
  // A 的布局契约是 [E, 16, K]（每个专家一组 token 槽位）。
  // 之前漏了这个偏移 ⇒ 所有专家都读同一批 A 行；单专家数值测试看不出来，
  // 一挂到多专家的端到端就全错（第 70 轮由 e2e 探针暴露）。
  const __nv_bfloat16* Ae = A + (int64_t)e * 16 * K;
  float acc[4][4];
  #pragma unroll
  for (int t = 0; t < 4; ++t)
    #pragma unroll
    for (int q = 0; q < 4; ++q) acc[t][q] = 0.f;

  for (int k0 = kb0; k0 < kb1; k0 += KC) {
    __syncthreads();
    { const int NV = 16 * KC / 8;
      for (int i = threadIdx.x; i < NV; i += blockDim.x) {
        int rr = (i * 8) / KC, kk2 = (i * 8) % KC;
        uint4 v;
        if (rr < na) v = __ldg(reinterpret_cast<const uint4*>(Ae + (int64_t)rr * K + k0 + kk2));
        else v = make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(&sA[rr][kk2]) = v;
      }
    }
    { const int Q = KC / 2, CH = NT2 / 8;      // CH = 8 组，每组 16 字节
      for (int t = threadIdx.x; t < Q * CH; t += blockDim.x) {
        int q = t / CH, ch = t % CH;
        uint4 v = __ldg(reinterpret_cast<const uint4*>(We + (int64_t)((k0 >> 1) + q) * N + n0 + ch*16));
        int r = q & 7, grp = q & ~7;
        int dst = grp + ((r < 4) ? (2*r) : (2*(r-4) + 1));
        *reinterpret_cast<uint4*>(&sB[dst][ch*8]) = v;   // 原样 16B，无扩展
      }
    }
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < KC; kk += 16) {
      uint32_t a0,a1,a2,a3;
      {   const __nv_bfloat16* aaddr = &sA[lane & 15][kk + ((lane >> 4) << 3)];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3)
                     : "l"(__cvta_generic_to_shared(aaddr)));
      }
      const int grp = (kk >> 1) & ~7;
      #pragma unroll
      for (int g = 0; g < 2; ++g) {            // 每 warp 2 个 16 列组
        const int ncol0 = n0 + (wid * 2 + g) * 16;   // 真实列起点
        const int nt2   = (wid * 2 + g) * 8;         // 元素列起点
        uint32_t reg;
        const uint16_t* addr = &sB[grp + (lane & 7)][nt2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                     : "=r"(reg) : "l"(__cvta_generic_to_shared(addr)));
        unsigned short elem0 = (unsigned short)(reg & 0xFFFF);   // 行 q,   列 2c/2c+1
        unsigned short elem1 = (unsigned short)(reg >> 16);      // 行 q+4, 列 2c/2c+1
        const int c = nt2 + (lane >> 2);
        const int colp = n0 + 2*c;                    // 偶列起点，两列连续
        const int q_ = lane & 3;
        // 相邻两列的 scale 在内存里连续 ⇒ 一次 uint16 读拿到 par=0/1 两个值
        const uint16_t sa2 = *reinterpret_cast<const uint16_t*>(
            Sce + (int64_t)((k0 + kk + 2*q_) >> 5) * N + colp);
        // k=2q 与 k=2q+8 恒在同一个 32-k 组（k-tile 宽 16 且 16 对齐）⇒ 两个 scale 相同，
        // 第二次加载是冗余的（第 60 轮发现）。
        const uint16_t sb2 = sa2;
        #pragma unroll
        for (int par = 0; par < 2; ++par) {    // par=0 偶列, par=1 奇列
          uint8_t by0 = (uint8_t)((elem0 >> (8*par)) & 0xFF);
          uint8_t by1 = (uint8_t)((elem1 >> (8*par)) & 0xFF);
          uint32_t b01 = lut[by0], b23 = lut[by1];
          unsigned short s_a = (unsigned short)((((sa2 >> (8*par)) & 0xFF)) << 7);
          unsigned short s_b = (unsigned short)((((sb2 >> (8*par)) & 0xFF)) << 7);
          unsigned short s2a[2] = {s_a, s_a}, s2b[2] = {s_b, s_b};
          *reinterpret_cast<__nv_bfloat162*>(&b01) = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&b01),
                                                            *reinterpret_cast<__nv_bfloat162*>(s2a));
          *reinterpret_cast<__nv_bfloat162*>(&b23) = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&b23),
                                                            *reinterpret_cast<__nv_bfloat162*>(s2b));
          const int tidx = g*2 + par;
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[tidx][0]), "+f"(acc[tidx][1]), "+f"(acc[tidx][2]), "+f"(acc[tidx][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b01), "r"(b23));
        }
      }
    }
  }
  const int rr0 = lane >> 2, cc0 = (lane & 3) * 2;
  // 偶列瓦片(par=0)与奇列瓦片(par=1)的第 k 个列正好相邻(2c0 / 2c0+1) ⇒ 合并成 float2 写
  #pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int base0 = n0 + (wid * 2 + g) * 16 + 2*cc0;       // 偶列起点
    // 4 个相邻列 (2c0, 2c0+1, 2c0+2, 2c0+3) 连续 ⇒ 一次 float4 写完（base0 是 4 的倍数，16B 对齐）
    float4 w0 = make_float4(acc[g*2+0][0], acc[g*2+1][0], acc[g*2+0][1], acc[g*2+1][1]);
    float4 w1 = make_float4(acc[g*2+0][2], acc[g*2+1][2], acc[g*2+0][3], acc[g*2+1][3]);
    float* rowA = C + (int64_t)e*16*N + (int64_t)rr0*N;
    float* rowB = C + (int64_t)e*16*N + (int64_t)(rr0+8)*N;
    // ⚠️ KSPLIT>1 时多个线程块写同一 C 位置 ⇒ 必须原子累加。
    //    第 50~64 轮这里用的是普通写，导致 ksplit>1 的结果全错（数值校验只跑过 ksplit=1，没覆盖到）。
    if (KSPLIT == 1) {
      if (rr0 < na)     { *(float4*)(rowA + base0) = w0; }
      if (rr0 + 8 < na) { *(float4*)(rowB + base0) = w1; }
    } else {
      const float* pw = reinterpret_cast<const float*>(&w0);
      const float* pw2 = reinterpret_cast<const float*>(&w1);
      if (rr0 < na) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) atomicAdd(rowA + base0 + i, pw[i]);
      }
      if (rr0 + 8 < na) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) atomicAdd(rowB + base0 + i, pw2[i]);
      }
    }
  }
}

__global__ void swiglu_kernel(const float* __restrict__ G, __nv_bfloat16* __restrict__ Out,
                             int na, int I, int NE, float limit) {
  int e = blockIdx.z, r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= I || r >= na) return;
  float g = G[((int64_t)e*16 + r) * 2 * I + i];
  float u = G[((int64_t)e*16 + r) * 2 * I + I + i];
  // swiglu_limit（ds41f 用 10）：与段表版 Triton 内核逐字一致 ——
  //   gate = min(gate, LIMIT); up = clamp(up, -LIMIT, LIMIT)
  // 缺省会熔掉（limit <= 0 时无分支差异），故不传 limit 的旧调用行为不变。
  if (limit > 0.f) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
  }
  Out[((int64_t)e*16 + r) * I + i] = __float2bfloat16((g / (1.f + __expf(-g))) * u);
}
extern "C" int swiglu(const void* G, void* Out, int na, int I, int NE, float limit,
                      cudaStream_t s) {
  dim3 g((I + 127) / 128, na, NE);
  swiglu_kernel<<<g, 128, 0, s>>>((const float*)G, (__nv_bfloat16*)Out, na, I, NE, limit);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) printf("[cu sw] %s\n", cudaGetErrorString(e));
  return (int)e;
}

extern "C" int run(const void* A, const void* W, const void* SC, void* C,
                   int na, int K, int N, int NE, int nblk, int ksplit, cudaStream_t s) {
  dim3 g(nblk, ksplit, NE);
  gemm<<<g, WARPS*32, 0, s>>>((const __nv_bfloat16*)A,(const uint8_t*)W,(const uint8_t*)SC,(float*)C,na,K,N,NE);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) printf("[cu] %s\n", cudaGetErrorString(e));
  return (int)e;
}
