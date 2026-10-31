// moe4_binding.inc — 小 M 4-bit 分组 MoE 内核的 pybind 绑定
//
// ⚠️ 本文件在 PYBIND11_MODULE 函数体内部展开：
//   * **绝不能 #include 任何头**（头里的 namespace 声明在函数体内非法）；
//   * **不能用 torch 类型**：binding.cpp 是 pybind11 + numpy 的纯 C++ 扩展，**不含 torch**
//     （实测 `torch::Tensor has not been declared`）⇒ 一律收 **裸地址 + 尺寸**，
//     Python 侧传 `tensor.data_ptr()` 与本进程的 CUDA stream 整数（`torch.cuda.current_stream().cuda_stream`）。
//   * 不能用 TORCH_CHECK（宏展开失败）⇒ 用下面的 MOE4_REQUIRE。
//
// 语义：seg2exp/seg_start/tok/wts 与 gpu_prefill.py:855 _build_segmentation 同语义；
//       nseg <= 段表长度（图捕获期用固定上界 T*topk，空段以 seg2exp=-1 短路）。

#define MOE4_REQUIRE(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while (0)

  m.def("moe4_smallm_init", []() {
    const cudaError_t e = moe4_smallm_init();
    MOE4_REQUIRE(e == cudaSuccess, std::string("moe4_smallm_init: ") + cudaGetErrorString(e));
  }, "初始化 256 项字节查表（幂等；失败抛异常，调用方据此禁用本内核）");

  m.def("moe4_smallm_gu",
        [](unsigned long long W, unsigned long long SC, unsigned long long X,
           unsigned long long Y, unsigned long long seg2exp, unsigned long long seg_start,
           unsigned long long tok, unsigned long long slot_map, int slot_base,
           int nseg, int K, int N, unsigned long long stream) {
          MOE4_REQUIRE(W && SC && X && Y, "moe4_smallm_gu: 指针为空");
          MOE4_REQUIRE(seg2exp && seg_start && tok, "moe4_smallm_gu: 段表指针为空");
          MOE4_REQUIRE(nseg >= 0, "moe4_smallm_gu: nseg 为负");
          MOE4_REQUIRE(N % 4 == 0, "moe4_smallm_gu: N 必须是 4 的倍数（4 字节对齐）");
          MOE4_REQUIRE(K % 32 == 0, "moe4_smallm_gu: K 必须是 32 的倍数（分组尺度）");
          const cudaError_t e = moe4_smallm_gu(
              (const void*)W, (const void*)SC, (const void*)X, (void*)Y,
              (const int*)seg2exp, (const int*)seg_start, (const int*)tok,
              (const int*)slot_map, slot_base, nseg, K, N, (cudaStream_t)stream);
          MOE4_REQUIRE(e == cudaSuccess, std::string("moe4_smallm_gu: ") + cudaGetErrorString(e));
        },
        "4-bit MXFP4 分组 gate_up：按 (token,expert) 槽位输出到 Y[A, N]（裸地址接口）");

  m.def("moe4_smallm_dn",
        [](unsigned long long W, unsigned long long SC, unsigned long long X,
           unsigned long long Y, unsigned long long seg2exp, unsigned long long seg_start,
           unsigned long long tok, unsigned long long wts, unsigned long long slot_map,
           int slot_base, int nseg, int K, int N, unsigned long long stream) {
          MOE4_REQUIRE(W && SC && X && Y && wts, "moe4_smallm_dn: 指针为空");
          MOE4_REQUIRE(seg2exp && seg_start && tok, "moe4_smallm_dn: 段表指针为空");
          MOE4_REQUIRE(nseg >= 0, "moe4_smallm_dn: nseg 为负");
          MOE4_REQUIRE(N % 4 == 0, "moe4_smallm_dn: N 必须是 4 的倍数");
          MOE4_REQUIRE(K % 32 == 0, "moe4_smallm_dn: K 必须是 32 的倍数");
          const cudaError_t e = moe4_smallm_dn(
              (const void*)W, (const void*)SC, (const void*)X, (void*)Y,
              (const int*)seg2exp, (const int*)seg_start, (const int*)tok,
              (const float*)wts, (const int*)slot_map, slot_base,
              nseg, K, N, (cudaStream_t)stream);
          MOE4_REQUIRE(e == cudaSuccess, std::string("moe4_smallm_dn: ") + cudaGetErrorString(e));
        },
        "4-bit MXFP4 分组 down：按 top-k 权重累加到 token（裸地址接口）");

  // ---- 内核内 dispatch（无主机侧分段；grid.z = nexp = 专家/平面行数）----
  m.def("moe4_smallm_gu2",
        [](unsigned long long W, unsigned long long SC, unsigned long long X,
           unsigned long long Y, unsigned long long ids, int T, int TOPSEL,
           int nexp, int K, int N, unsigned long long stream) {
          MOE4_REQUIRE(W && SC && X && Y && ids, "moe4_smallm_gu2: 指针为空");
          MOE4_REQUIRE(T >= 0 && TOPSEL >= 0 && nexp >= 1, "moe4_smallm_gu2: 形状非法");
          MOE4_REQUIRE(N % 4 == 0 && K % 32 == 0, "moe4_smallm_gu2: N/K 对齐非法");
          const cudaError_t e = moe4_smallm_gu2(
              (const void*)W, (const void*)SC, (const void*)X, (void*)Y,
              (const int*)ids, T, TOPSEL, nexp, K, N, (cudaStream_t)stream);
          MOE4_REQUIRE(e == cudaSuccess, std::string("moe4_smallm_gu2: ") + cudaGetErrorString(e));
        },
        "内核内 dispatch 的 gate_up（无需主机侧分段）；Y 行号 = e*kMaxA + 槽位");

  m.def("moe4_smallm_dn2",
        [](unsigned long long W, unsigned long long SC, unsigned long long X,
           unsigned long long Y, unsigned long long ids, unsigned long long wts,
           int T, int TOPSEL, int nexp, int K, int N, unsigned long long stream) {
          MOE4_REQUIRE(W && SC && X && Y && ids && wts, "moe4_smallm_dn2: 指针为空");
          MOE4_REQUIRE(T >= 0 && TOPSEL >= 0 && nexp >= 1, "moe4_smallm_dn2: 形状非法");
          MOE4_REQUIRE(N % 4 == 0 && K % 32 == 0, "moe4_smallm_dn2: N/K 对齐非法");
          const cudaError_t e = moe4_smallm_dn2(
              (const void*)W, (const void*)SC, (const void*)X, (void*)Y,
              (const int*)ids, (const float*)wts, T, TOPSEL, nexp, K, N, (cudaStream_t)stream);
          MOE4_REQUIRE(e == cudaSuccess, std::string("moe4_smallm_dn2: ") + cudaGetErrorString(e));
        },
        "内核内 dispatch 的 down（按 top-k 权重累加到 token）");

#undef MOE4_REQUIRE
