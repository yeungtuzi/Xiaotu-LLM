// 挎斗 host-function 协议（纯 host，无 CUDA 依赖）
//
// 设计见 dev-docs/SPEC_SIDECAR_HOSTFN.md：
//   - 生产默认 XIAOTU_MOE_ASYNC=0 ⇒ 每层 CPU MoE 跑在 cudaLaunchHostFunc 回调里；
//   - 该回调内**不得调用 CUDA API**，只碰原子/内存；
//   - 挎斗 worker 在独立线程里用 GPU0 stream 计算，与本回调并行；
//   - 回调先 host_begin（命中检查/拷请求/置 0/发请求），再 run_moe_and_ep，
//     最后 host_end（等 worker/合并）。命中项在 CPU 侧权重为 0 ⇒ 引擎跳过。
//
// 本头文件只做协议与合并，便于单测（g++ -pthread），不依赖 torch/CUDA。
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

namespace xiaotu_sidecar {

struct SidecarCtl {
    // ---- 请求/完成序号（不同 cache line）----
    alignas(64) std::atomic<uint32_t> req_seq{0};
    alignas(64) std::atomic<uint32_t> done_seq{0};
    alignas(64) std::atomic<uint32_t> enabled{0};
    alignas(64) std::atomic<uint32_t> stop{0};

    // ---- 每层请求区（host 写、worker 读；回调返回前一直有效）----
    uint32_t layer = 0;
    uint32_t qlen = 0;
    uint32_t k = 0;
    const uint16_t* hid = nullptr;   // bf16 [qlen, hidden]
    uint32_t* ids = nullptr;         // 请求副本 [k*qlen]
    float* wts = nullptr;            // 请求副本 [k*qlen]
    float* side_out = nullptr;       // worker 写 [qlen, hidden]，fp32
    uint32_t pending_hits = 0;       // host_begin 写入，host_end 统计用

    // ---- 静态表 ----
    const uint64_t* hit_masks = nullptr;  // [n_layers][n_words]
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;               // 384
    uint32_t n_words = 0;                 // ceil(n_experts/64) = 6
    uint32_t hidden = 0;                  // 5120
    uint32_t max_qlen = 0;
    uint32_t max_k = 0;
    uint32_t timeout_us = 50000;          // fail-closed：默认 50 ms

    // ---- 统计 ----
    alignas(64) std::atomic<uint64_t> stat_steps{0};
    alignas(64) std::atomic<uint64_t> stat_hits{0};
    alignas(64) std::atomic<uint64_t> stat_timeouts{0};
};

inline bool hit_test(const SidecarCtl* c, uint32_t layer, uint32_t expert) {
    if (layer >= c->n_layers || expert >= c->n_experts) return false;
    const uint64_t* m = c->hit_masks + (size_t)layer * c->n_words;
    return (m[expert >> 6] >> (expert & 63)) & 1ull;
}

// 阶段 1：命中检查、拷请求、命中项权重置 0、置 req_seq。返回命中数（0=无命中）。
// 必须在 run_moe_and_ep **之前**调用，这样 worker 与 CPU MoE 并行。
inline uint32_t host_begin(SidecarCtl* c, uint32_t layer, uint32_t qlen, uint32_t k,
                           const uint32_t* ids, float* wts, const uint16_t* hid) {
    if (!c || !c->enabled.load(std::memory_order_acquire)) return 0;
    if (layer >= c->n_layers || qlen == 0 || qlen > c->max_qlen || k > c->max_k) return 0;

    const uint32_t n = qlen * k;
    // 必须先拷原始权重到请求区，再在调用方数组上置 0（否则 worker 拿到的是 0 权重）
    std::memcpy(c->ids, ids, (size_t)n * sizeof(uint32_t));
    std::memcpy(c->wts, wts, (size_t)n * sizeof(float));

    uint32_t hits = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (hit_test(c, layer, ids[i])) {
            wts[i] = 0.0f;       // CPU 跳过（原值已在 c->wts）
            ++hits;
        }
    }
    if (hits == 0) return 0;

    c->layer = layer;
    c->qlen = qlen;
    c->k = k;
    c->hid = hid;
    c->pending_hits = hits;
    const uint32_t seq = c->req_seq.load(std::memory_order_relaxed) + 1;
    c->req_seq.store(seq, std::memory_order_release);
    return hits;
}

// 阶段 2：等 done_seq（带超时，fail-closed），把 side_out 累加进 out。
// 返回 false 表示超时/未完成，调用方应让该请求失败。
inline bool host_end(SidecarCtl* c, float* out) {
    if (!c) return true;
    const uint32_t seq = c->req_seq.load(std::memory_order_acquire);
    if (seq == 0) return true;

    const auto t0 = std::chrono::steady_clock::now();
    while (c->done_seq.load(std::memory_order_acquire) != seq) {
        const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        if ((uint64_t)dt > c->timeout_us) {
            c->stat_timeouts.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::this_thread::yield();
    }
    const size_t nout = (size_t)c->qlen * c->hidden;
    for (size_t i = 0; i < nout; ++i) out[i] += c->side_out[i];
    c->stat_steps.fetch_add(1, std::memory_order_relaxed);
    c->stat_hits.fetch_add(c->pending_hits, std::memory_order_relaxed);
    return true;
}

// 便捷组合（单测/无 CPU 并行时用）。
inline uint32_t host_step(SidecarCtl* c, uint32_t layer, uint32_t qlen, uint32_t k,
                          const uint32_t* ids, float* wts, float* out,
                          const uint16_t* hid) {
    const uint32_t hits = host_begin(c, layer, qlen, k, ids, wts, hid);
    if (!hits) return 0;
    return host_end(c, out) ? hits : 0;
}

// mock worker（单测 / 协议验证）：把 side_out 填成常数。
inline void mock_worker(SidecarCtl* c, float val) {
    uint32_t last = 0;
    while (!c->stop.load(std::memory_order_acquire)) {
        uint32_t s = c->req_seq.load(std::memory_order_acquire);
        if (s == last) { std::this_thread::yield(); continue; }
        last = s;
        const size_t n = (size_t)c->qlen * c->hidden;
        for (size_t i = 0; i < n; ++i) c->side_out[i] = val;
        c->done_seq.store(s, std::memory_order_release);
    }
}

}  // namespace xiaotu_sidecar
