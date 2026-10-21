/* Self-contained memory-bandwidth benchmark (no third-party libraries).
 *
 * Build: gcc -O3 -march=native -pthread scripts/musa/membw.c -o /tmp/membw
 * Run:   /tmp/membw -c 0,4,8,12 -b 256 -i 8 -m read
 *
 * Each worker is pinned before allocating/touching its buffer, so Linux
 * first-touch places pages on that CPU's NUMA node.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

static size_t g_bytes = 256ULL << 20;
static int g_iters = 8;
static int g_mode = 0; /* 0=read, 1=copy */
static pthread_barrier_t g_start, g_done;
static volatile uint64_t g_sink;

typedef struct {
    int cpu;
    unsigned char *a;
    unsigned char *b;
    uint64_t sum;
} Worker;

static void *worker(void *arg) {
    Worker *w = (Worker *)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        perror("pthread_setaffinity_np");
        exit(2);
    }
    w->a = mmap(NULL, g_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w->a == MAP_FAILED) { perror("mmap a"); exit(2); }
    if (g_mode == 1) {
        w->b = mmap(NULL, g_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (w->b == MAP_FAILED) { perror("mmap b"); exit(2); }
    }
    /* First touch on this CPU's NUMA node. */
    for (size_t i = 0; i < g_bytes; i += 4096) w->a[i] = (unsigned char)i;
    if (g_mode == 1)
        for (size_t i = 0; i < g_bytes; i += 4096) w->b[i] = 0;

    pthread_barrier_wait(&g_start);
    uint64_t sum = 0;
    if (g_mode == 0) {
        const uint64_t *p = (const uint64_t *)w->a;
        size_t n = g_bytes / sizeof(uint64_t);
        /* Eight independent accumulators: avoid a serial add dependency and
         * let the compiler emit wide loads (AVX2 on this host). */
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint64_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        for (int it = 0; it < g_iters; ++it)
            for (size_t i = 0; i + 7 < n; i += 8) {
                s0 += p[i + 0]; s1 += p[i + 1];
                s2 += p[i + 2]; s3 += p[i + 3];
                s4 += p[i + 4]; s5 += p[i + 5];
                s6 += p[i + 6]; s7 += p[i + 7];
            }
        sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    } else {
        uint64_t *dst = (uint64_t *)w->b;
        const uint64_t *src = (const uint64_t *)w->a;
        size_t n = g_bytes / sizeof(uint64_t);
        for (int it = 0; it < g_iters; ++it)
            for (size_t i = 0; i < n; ++i) dst[i] = src[i];
        sum = dst[0] + n;
    }
    w->sum = sum;
    pthread_barrier_wait(&g_done);
    return NULL;
}

static int parse_cpus(const char *s, int *cpus, int max) {
    int n = 0;
    char *copy = strdup(s), *p = copy;
    if (!copy) return -1;
    while (p && *p && n < max) {
        char *q = strchr(p, ',');
        if (q) *q = '\0';
        cpus[n++] = atoi(p);
        if (!q) break;
        p = q + 1;
    }
    free(copy);
    return n;
}

int main(int argc, char **argv) {
    const char *cpus_s = "0";
    char mode = 'r';
    int opt;
    while ((opt = getopt(argc, argv, "c:b:i:m:")) != -1) {
        switch (opt) {
        case 'c': cpus_s = optarg; break;
        case 'b': g_bytes = strtoull(optarg, NULL, 0) << 20; break;
        case 'i': g_iters = atoi(optarg); break;
        case 'm': mode = optarg[0]; break;
        default: return 2;
        }
    }
    g_mode = (mode == 'c') ? 1 : 0;
    if (g_bytes < (1ULL << 20) || g_iters < 1) return 2;

    int cpus[256];
    int n = parse_cpus(cpus_s, cpus, 256);
    if (n <= 0) return 2;
    Worker *ws = calloc((size_t)n, sizeof(*ws));
    pthread_t *th = calloc((size_t)n, sizeof(*th));
    if (!ws || !th) return 2;
    pthread_barrier_init(&g_start, NULL, (unsigned)n + 1);
    pthread_barrier_init(&g_done, NULL, (unsigned)n + 1);
    for (int i = 0; i < n; ++i) {
        ws[i].cpu = cpus[i];
        if (pthread_create(&th[i], NULL, worker, &ws[i]) != 0) {
            perror("pthread_create"); return 2;
        }
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pthread_barrier_wait(&g_start);
    pthread_barrier_wait(&g_done);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t sink = 0;
    for (int i = 0; i < n; ++i) {
        pthread_join(th[i], NULL);
        sink += ws[i].sum;
        munmap(ws[i].a, g_bytes);
        if (ws[i].b) munmap(ws[i].b, g_bytes);
    }
    g_sink = sink;
    double sec = (double)(t1.tv_sec - t0.tv_sec) +
                 (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    double bytes = (double)n * (double)g_bytes * (double)g_iters *
                   (g_mode ? 2.0 : 1.0);
    printf("threads=%d cpus=%s buf=%zuMiB iters=%d mode=%s "
           "seconds=%.6f GB/s=%.1f GiB/s=%.1f\n",
           n, cpus_s, g_bytes >> 20, g_iters,
           g_mode ? "copy" : "read", sec, bytes / sec / 1e9,
           bytes / sec / (1ULL << 30));
    free(ws); free(th);
    pthread_barrier_destroy(&g_start);
    pthread_barrier_destroy(&g_done);
    return 0;
}
