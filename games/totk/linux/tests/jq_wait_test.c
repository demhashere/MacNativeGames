/* The job-queue wait cave (runtime/jq_wait.h) run natively: its branches back to the game are two stubs that say which
   way it left. make -f linux/Makefile GEN=build/linux-dummy OUT=build/linux-test jqtest */
#include "jq_wait.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static uint8_t *g_code, *g_stats;
static struct { uint8_t pad[0x94]; _Atomic int32_t cursor; _Atomic uint8_t flag; } g_queue;
static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

static uint64_t ticks_hz(void) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); return f; }
/* enters the cave as the patched branch does; returns how it left (1 = the lock again, 2 = as shipped) with x0 and x8 */
static int run(uint32_t w14, uint64_t *x0, uint64_t *x8) {
    register uint64_t r0 __asm__("x0") = *x0, r8 __asm__("x8") = *x8, r14 __asm__("x14") = w14, r21 __asm__("x21") = (uintptr_t)&g_queue;
    register uint64_t r9 __asm__("x9") = 0;
    __asm__ volatile("blr %[cave]" : "+r"(r0), "+r"(r8), "+r"(r14), "+r"(r9) : [cave] "r"(g_code), "r"(r21)
                     : "x1", "x4", "x5", "x6", "x7", "x16", "x17", "x30", "cc", "memory");
    *x0 = r0; *x8 = r8; return (int)r9;
}
static uint64_t stat(int off) { uint64_t v; memcpy(&v, g_stats + off, 8); return v; }
static void *later(void *arg) {
    usleep((useconds_t)(uintptr_t)arg);
    if ((uintptr_t)arg == 5000) atomic_store(&g_queue.cursor, 3); else atomic_store(&g_queue.flag, 0);
    return NULL;
}
int main(void) {
    long pg = sysconf(_SC_PAGESIZE);
    uint8_t *m = mmap(NULL, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    g_code = m; g_stats = m + pg;
    uint64_t retry = (uintptr_t)g_code + 0x400, ok = (uintptr_t)g_code + 0x440;
    uint32_t w[JQW_WORDS];
    CHECK(jqw_range_ok((uintptr_t)g_code, (uintptr_t)g_stats, retry, ok), "ranges");
    jqw_build(w, (uintptr_t)g_code, (uintptr_t)g_stats, retry, ok);
    if (getenv("TOTK_JQ_NEG")) w[8] ^= 0xc ^ 0xd;   /* negative control: b.gt -> b.le, the cursor test inverted */
    memcpy(g_code, w, sizeof w);
    const uint32_t stub1[2] = {0xd2800029u /* mov x9, #1 */, 0xd65f03c0u /* ret */}, stub2[2] = {0xd2800049u /* mov x9, #2 */, 0xd65f03c0u};
    memcpy(g_code + 0x400, stub1, sizeof stub1); memcpy(g_code + 0x440, stub2, sizeof stub2);
    uint64_t hz = ticks_hz(), poll = hz / 500;   /* 2 ms */
    struct timespec ts = {0, 50000};
    memcpy(g_stats + JQW_POLL, &poll, 8); memcpy(g_stats + JQW_SLEEP, &ts, sizeof ts);
    if (mprotect(m, pg, PROT_READ | PROT_EXEC)) { perror("mprotect"); return 1; }
    __builtin___clear_cache((char *)g_code, (char *)g_code + pg);

    uint64_t x0 = 0x1234, x8 = 0x5678;
    CHECK(run(1, &x0, &x8) == 2, "w14 = 1 leaves as shipped");
    CHECK(stat(JQW_WAITS) == 0, "no wait counted for a free item");

    atomic_store(&g_queue.flag, 0); atomic_store(&g_queue.cursor, 0);
    CHECK(run(0, &x0, &x8) == 1, "flag clear: back to the lock at once");
    CHECK(stat(JQW_WAITS) == 1, "one wait counted");

    atomic_store(&g_queue.flag, 1); atomic_store(&g_queue.cursor, 0);
    pthread_t th; pthread_create(&th, NULL, later, (void *)(uintptr_t)5000);
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    int how = run(0, &x0, &x8);
    clock_gettime(CLOCK_MONOTONIC, &b); pthread_join(th, NULL);
    double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6, longest = stat(JQW_LONGEST) * 1e3 / hz;
    CHECK(how == 1, "cursor set: back to the lock");
    CHECK(x0 == 0x1234 && x8 == 0x5678, "x0 / x8 kept across the sleeps (%#llx %#llx)", (unsigned long long)x0, (unsigned long long)x8);
    CHECK(ms >= 4.5 && ms < 50, "waited %.2f ms for a cursor set after 5 ms", ms);
    CHECK(longest >= 4.5 && longest < 50, "longest wait %.2f ms", longest);
    CHECK(stat(JQW_WAITS) == 2, "two waits counted");

    atomic_store(&g_queue.flag, 1); atomic_store(&g_queue.cursor, -1);   /* a negative cursor still waits (signed compare) */
    pthread_create(&th, NULL, later, (void *)(uintptr_t)1000);
    how = run(0, &x0, &x8); pthread_join(th, NULL);
    CHECK(how == 1, "flag cleared while the cursor is negative: back to the lock");
    CHECK(stat(JQW_WAITS) == 3 && stat(JQW_TOTAL) * 1e3 / hz >= 5.5, "three waits, %.2f ms in all", stat(JQW_TOTAL) * 1e3 / hz);
    printf("%s: job-queue wait cave (%d words), waits %llu, longest %.2f ms, counter %llu Hz\n", g_fail ? "FAILED" : "passed", JQW_WORDS,
           (unsigned long long)stat(JQW_WAITS), stat(JQW_LONGEST) * 1e3 / hz, (unsigned long long)hz);
    return g_fail;
}
