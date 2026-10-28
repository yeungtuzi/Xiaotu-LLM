// 序号协议（seq）离线模拟器 —— 纯 CPU，无 CUDA，不占 GPU。
//
// 目的：在改生产代码之前，先验证 `PLAN_DEFERRAL_SEQ_PROTOCOL_2026-10-28.md` §3 的协议
//       在"宿主抢跑 + worker 随机停顿"下不会死锁、不丢任务、顺序正确。
//
// 三个线程，忠实模拟真实依赖结构：
//   host   = 模型的宿主线程：逐层 enqueue（不阻塞），最后 enqueue 一次 drain 等待
//   gpu    = 调用流的执行体：按序执行 WAIT / COPY / WRITE 三类"流内操作"
//   worker = CPU MoE 工作线程：看到 din_seq > last_seen 就计算，完成后写 dout_seq
//
// 用法：
//   ./test_seq_protocol            # 正常协议（应全部通过）
//   ./test_seq_protocol --old      # 旧协议的缺陷复现（worker 持锁自旋 ⇒ 应被判死锁）
//
// 退出码：0 = 通过；1 = 失败（死锁/丢任务/顺序错）。

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ------------------------------ 槽位 ------------------------------
struct Slot {
    std::atomic<uint32_t> din_seq{0};    // 宿主写（真实实现里是 cudaStreamWriteValue32）
    std::atomic<uint32_t> dout_seq{0};   // worker 写
    uint32_t host_seq = 0;               // 宿主本地递增
    uint32_t last_seen = 0;              // worker 本地
};

// ------------------------------ 流内操作 ------------------------------
enum class OpKind { Wait, Copy, Write };
struct Op {
    OpKind kind;
    int slot;
    uint32_t seq;
};

struct World {
    std::vector<Slot> slots{2};
    std::deque<Op> stream;               // 宿主 enqueue，gpu 按序执行
    std::timed_mutex m;
    std::condition_variable_any cv;          // 宿主 notify worker
    bool host_done = false;
    bool stop = false;

    // 统计/校验
    std::atomic<uint64_t> executed{0};
    std::atomic<uint64_t> computed{0};
    std::atomic<bool> violation{false};
    std::atomic<uint32_t> last_done{0};

    std::mt19937_64 rng{12345};
    std::mutex rng_m;
    int max_copy_us = 120, max_work_us = 900;

    void sleep_us(int u) {
        if (u > 0) std::this_thread::sleep_for(std::chrono::microseconds(u));
    }
    int rand_us(int hi) {
        std::lock_guard<std::mutex> lk(rng_m);
        return hi > 0 ? (int)(rng() % (uint64_t)hi) : 0;
    }
};

// ------------------------------ 宿主 ------------------------------
// 逐层 enqueue：defer 结构 = 用 slot[layer&1]，先兑现上一层再排入本层。
static void host_run(World& w, int layers, int steps) {
    int pending_slot = -1;
    uint32_t pending_seq = 0;
    for (int st = 0; st < steps; ++st) {
        pending_slot = -1;
        for (int L = 0; L < layers; ++L) {
            Slot& cur = w.slots[L & 1];
            // 宿主必须能拿到锁；旧协议下 worker 持锁自旋 ⇒ 这里超时 ⇒ 判死锁
            std::unique_lock<std::timed_mutex> lk(w.m, std::defer_lock);
            if (!lk.try_lock_for(std::chrono::milliseconds(500))) {
                fprintf(stderr, "[FAIL] 宿主等锁超时(>500ms)：worker 持锁自旋堵死了宿主（D-B）\n");
                w.violation = true;
                w.stop = true;
                w.cv.notify_all();
                return;
            }
            if (pending_slot >= 0) {
                w.stream.push_back({OpKind::Wait, pending_slot, pending_seq});
                w.stream.push_back({OpKind::Copy, pending_slot, pending_seq});
            } else {
                w.stream.push_back({OpKind::Copy, L & 1, 0});
            }
            uint32_t v = ++cur.host_seq;
            w.stream.push_back({OpKind::Write, L & 1, v});
            pending_slot = L & 1;
            pending_seq = v;
            w.cv.notify_one();           // 模拟宿主在写序号后唤醒 worker
        }
        // 末层兑现（cpu_decode_drain）
        {
            std::unique_lock<std::timed_mutex> lk(w.m, std::defer_lock);
            if (!lk.try_lock_for(std::chrono::milliseconds(500))) {
                fprintf(stderr, "[FAIL] 宿主等锁超时(>500ms)（drain 阶段）\n");
                w.violation = true;
                w.stop = true;
                w.cv.notify_all();
                return;
            }
            w.stream.push_back({OpKind::Wait, pending_slot, pending_seq});
            w.cv.notify_one();
        }
    }
    {   // 收尾也必须可超时：旧协议下 worker 持锁自旋，这里会永久阻塞
        std::unique_lock<std::timed_mutex> lk(w.m, std::defer_lock);
        if (!lk.try_lock_for(std::chrono::milliseconds(500))) {
            fprintf(stderr, "[FAIL] 宿主收尾等锁超时(>500ms)（D-B：worker 持锁自旋）\n");
            w.violation = true;
            w.stop = true;
            return;
        }
        w.host_done = true;
    }
    w.cv.notify_all();
}

// ------------------------------ GPU（流执行体）------------------------------
static void gpu_run(World& w) {
    for (;;) {
        Op op;
        {
            std::unique_lock<std::timed_mutex> lk(w.m);
            w.cv.wait_for(lk, std::chrono::microseconds(200), [&] {
                return !w.stream.empty() || w.host_done || w.stop;
            });
            if (w.stream.empty()) {
                if (w.host_done || w.stop) return;
                continue;
            }
            op = w.stream.front();
            w.stream.pop_front();
        }
        switch (op.kind) {
        case OpKind::Wait: {
            // 等 worker 完成第 seq 个任务：GEQ 语义，错过边沿也不会永久阻塞
            auto t0 = Clock::now();
            while (w.slots[op.slot].dout_seq.load(std::memory_order_acquire) < op.seq) {
                if (ms_since(t0) > 5000.0) {   // 5 s 看门狗
                    fprintf(stderr, "[FAIL] WAIT 超时 slot=%d seq=%u（dout=%u）\n", op.slot,
                            op.seq, w.slots[op.slot].dout_seq.load());
                    w.violation = true;
                    w.stop = true;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            break;
        }
        case OpKind::Copy:
            w.sleep_us(w.rand_us(w.max_copy_us));
            w.executed.fetch_add(1);
            break;
        case OpKind::Write: {
            uint32_t prev = w.slots[op.slot].din_seq.load(std::memory_order_relaxed);
            if (op.seq <= prev) {          // 序号必须严格递增
                fprintf(stderr, "[FAIL] din_seq 非单调 slot=%d prev=%u new=%u\n", op.slot, prev,
                        op.seq);
                w.violation = true;
                w.stop = true;
                return;
            }
            w.slots[op.slot].din_seq.store(op.seq, std::memory_order_release);
            std::lock_guard<std::timed_mutex> lk(w.m);
            w.cv.notify_one();
            break;
        }
        }
    }
}

// ------------------------------ worker ------------------------------
static void worker_run(World& w, bool old_protocol) {
    for (;;) {
        Slot* picked = nullptr;
        uint32_t v = 0;
        {
            std::unique_lock<std::timed_mutex> lk(w.m);
            w.cv.wait_for(lk, std::chrono::microseconds(200), [&] {
                for (auto& s : w.slots)
                    if (s.din_seq.load(std::memory_order_acquire) > s.last_seen) return true;
                return w.stop;
            });
            for (auto& s : w.slots) {
                if (s.din_seq.load(std::memory_order_acquire) > s.last_seen) {
                    picked = &s;
                    v = s.din_seq.load(std::memory_order_acquire);
                    break;
                }
            }
            if (!picked) {
                if (w.stop) return;
                continue;
            }
            if (old_protocol) {
                // 旧实现：持锁计算 + 持锁自旋等"归还"，宿主任何需要该锁的路径都会死在这里。
                w.sleep_us(w.rand_us(w.max_work_us));
                picked->last_seen = v;
                picked->dout_seq.store(v, std::memory_order_release);
                while (!w.stop && picked->din_seq.load() == v) std::this_thread::sleep_for(
                    std::chrono::microseconds(50));   // 等宿主写 din=0（新协议不存在此步）
                continue;
            }
        }
        // 新协议：计算在锁外，完成后写 done_seq；无需归还
        w.sleep_us(w.rand_us(w.max_work_us));
        picked->last_seen = v;
        picked->dout_seq.store(v, std::memory_order_release);
        w.computed.fetch_add(1);
        std::lock_guard<std::timed_mutex> lk(w.m);
        w.cv.notify_all();
    }
}

int main(int argc, char** argv) {
    const bool old_protocol = (argc > 1 && std::strcmp(argv[1], "--old") == 0);
    const int layers = 40, steps = 5;        // 1000 次层调用
    const int trials = old_protocol ? 1 : 50;  // 新协议跑 200 组随机种子

    int failures = 0;
    for (int t = 0; t < trials; ++t) {
        World w;
        w.rng.seed(1000 + t);
        // 随机化停顿幅度：模拟真实抖动
        w.max_copy_us = 20 + (t * 37) % 300;
        w.max_work_us = 100 + (t * 91) % 1800;

        std::thread gpu(gpu_run, std::ref(w));
        std::thread wk(worker_run, std::ref(w), old_protocol);
        std::thread host(host_run, std::ref(w), layers, steps);

        Clock::time_point t0 = Clock::now();
        host.join();
        fprintf(stderr, "[stage] host joined\n");
        bool deadlock = false;
        // 等 gpu 把流跑空（drain 完成后应自行退出）
        while (ms_since(t0) < 8000.0) {
            bool empty = false, got = false;
            {
                std::unique_lock<std::timed_mutex> lk(w.m, std::defer_lock);
                if (lk.try_lock_for(std::chrono::milliseconds(200))) {
                    got = true;
                    empty = w.stream.empty();
                }
            }
            if (got && empty) break;
            if (!got) {   // 拿不到锁 = worker 持锁自旋 = 死锁
                fprintf(stderr, "[FAIL] 主线程等锁超时：worker 持锁自旋（D-B）\n");
                deadlock = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (!deadlock) {
            std::unique_lock<std::timed_mutex> lk(w.m, std::defer_lock);
            if (!lk.try_lock_for(std::chrono::milliseconds(200))) deadlock = true;
            else if (!w.stream.empty()) deadlock = true;
        }
        w.stop = true;
        w.cv.notify_all();
        gpu.join();
        fprintf(stderr, "[stage] gpu joined\n");
        wk.join();
        fprintf(stderr, "[stage] worker joined\n");

        fprintf(stderr, "[stage] checking\n");
        // 校验：每个槽的 dout_seq 必须追上 host_seq（全部任务都被处理）
        bool lost = false;
        for (auto& s : w.slots)
            if (s.dout_seq.load() != s.host_seq) lost = true;

        const double dt = ms_since(t0);
        if (deadlock || lost || w.violation) {
            ++failures;
            fprintf(stderr, "[FAIL] trial=%d deadlock=%d lost=%d violation=%d 耗时=%.1fms\n", t,
                    (int)deadlock, (int)lost, (int)w.violation.load(), dt);
            if (failures > 3) break;
        } else if (t == 0 || t == trials - 1) {
            printf("[ok] trial=%d 层调用=%d 流操作=%llu worker完成=%llu 耗时=%.1fms\n", t,
                   layers * steps, (unsigned long long)w.executed.load(),
                   (unsigned long long)w.computed.load(), dt);
        }
    }

    if (old_protocol) {
        printf("\n旧协议模式：预期被判为死锁（复现 §2.2 的 D-B/D-C）\n");
        // 旧协议必然失败：把"失败"当成功信号
        return failures > 0 ? 0 : 1;
    }
    printf("\n%s：%d 组中失败 %d 组\n", failures == 0 ? "通过" : "不通过", trials, failures);
    return failures == 0 ? 0 : 1;
}
