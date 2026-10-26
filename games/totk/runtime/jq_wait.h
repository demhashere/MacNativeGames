/* The job-queue wait of Breath of the Wild 1.9.0 (game_patches_botw.c patch_jq_wait; after nxnative's spin gate,
   dm/tools/patch_nso.py "Spin waits"). A worker whose next item belongs to the main thread's core releases the queue's
   lock and loops back to take it again (main+0xef63e8 cbz w14, main+0xef6354) until that item is taken; two workers doing
   this at full speed kept the lock between their cores, and the main thread lost the race for it for 30-210 ms at a time.
   The cave below takes the loop's back-edge instead: it watches the queue's flag (+0x98) and cursor (+0x94) with plain
   loads, the two fields the check under the lock reads, and goes back to the lock only when one of them changed. After
   JQW_POLL ticks of watching it sleeps (Linux nanosleep, JQW_SLEEP) between looks. Each wait adds its length to the
   statistics on the data page. Clobbers x1, x4-x7, x14, x16, x17 and the flags, which the function does not keep live
   there; x0 (the core number) and x8 (the last index) are kept across the system call. jq_wait_test.c runs it natively. */
#pragma once
#include <stdint.h>

enum { JQW_WORDS = 34,
       JQW_TOTAL = 0, JQW_LONGEST = 8, JQW_WAITS = 16,   /* stats, in counter ticks (cntvct_el0) and waits */
       JQW_POLL = 24,                                    /* ticks of watching before the first sleep (the host writes it) */
       JQW_SLEEP = 32,                                   /* struct timespec of one sleep (the host writes it) */
       JQW_BYTES = 48 };

static inline uint32_t jqw_b(uint64_t pc, uint64_t t) { return 0x14000000u | ((uint32_t)((int64_t)(t - pc) >> 2) & 0x3ffffffu); }
static inline uint32_t jqw_bcond(uint64_t pc, uint64_t t, unsigned c) { return 0x54000000u | (((uint32_t)((int64_t)(t - pc) >> 2) & 0x7ffffu) << 5) | c; }
static inline uint32_t jqw_cbz(uint64_t pc, uint64_t t, int nz, unsigned rt) { return (nz ? 0x35000000u : 0x34000000u) | (((uint32_t)((int64_t)(t - pc) >> 2) & 0x7ffffu) << 5) | rt; }
static inline uint32_t jqw_adr(uint64_t pc, uint64_t t, unsigned rd) {
    uint32_t imm = (uint32_t)(int64_t)(t - pc) & 0x1fffffu;
    return 0x10000000u | (imm & 3u) << 29 | (imm >> 2) << 5 | rd;
}
/* 0: cave and branch ranges fine (B: 128 MB, ADR / CBZ / B.cond: 1 MB) */
static inline int jqw_range_ok(uint64_t cave, uint64_t stats, uint64_t retry, uint64_t ok) {
    int64_t a = (int64_t)(stats - cave), r = (int64_t)(retry - cave), o = (int64_t)(ok - cave);
    return a > -(1 << 20) + 256 && a < (1 << 20) - 256 && r > -(1 << 27) + 256 && r < (1 << 27) - 256 && o > -(1 << 27) + 256 && o < (1 << 27) - 256;
}
/* cave, stats: run-time addresses of the cave and of its statistics; retry: the lock again (main+0xef6354); ok: the
   instruction after the patched branch (main+0xef63ec) */
static inline void jqw_build(uint32_t w[JQW_WORDS], uint64_t cave, uint64_t stats, uint64_t retry, uint64_t ok) {
    enum { POLL = 4, SLEEP = 15, DONE = 24, OK = 33 };
#define AT(i) (cave + 4u * (i))
    const uint32_t c[JQW_WORDS] = {
        jqw_cbz(AT(0), AT(OK), 1, 14),   /* cbnz w14, ok          the item is free: on as shipped */
        0xd53be050u,                     /* mrs x16, cntvct_el0    the wait starts */
        jqw_adr(AT(2), stats, 17),       /* adr x17, <stats> */
        0xf9400e27u,                     /* ldr x7, [x17, #24]     JQW_POLL */
        0x394262aeu,                     /* poll: ldrb w14, [x21, #0x98]   the queue's flag */
        jqw_cbz(AT(5), AT(DONE), 0, 14), /* cbz w14, done */
        0xb94096aeu,                     /* ldr w14, [x21, #0x94]  its cursor */
        0x710001dfu,                     /* cmp w14, #0 */
        jqw_bcond(AT(8), AT(DONE), 0xc), /* b.gt done */
        0xd53be046u,                     /* mrs x6, cntvct_el0 */
        0xcb1000c6u,                     /* sub x6, x6, x16 */
        0xeb0700dfu,                     /* cmp x6, x7 */
        jqw_bcond(AT(12), AT(SLEEP), 0x2), /* b.hs sleep */
        0xd503203fu,                     /* yield */
        jqw_b(AT(14), AT(POLL)),         /* b poll */
        0xaa0003e5u,                     /* sleep: mov x5, x0 */
        0xaa0803e4u,                     /* mov x4, x8 */
        jqw_adr(AT(17), stats + JQW_SLEEP, 0),   /* adr x0, <timespec> */
        0xaa1f03e1u,                     /* mov x1, xzr */
        0xd2800ca8u,                     /* mov x8, #101           __NR_nanosleep */
        0xd4000001u,                     /* svc #0 */
        0xaa0403e8u,                     /* mov x8, x4 */
        0xaa0503e0u,                     /* mov x0, x5 */
        jqw_b(AT(23), AT(POLL)),         /* b poll */
        0xd53be046u,                     /* done: mrs x6, cntvct_el0 */
        0xcb1000c6u,                     /* sub x6, x6, x16 */
        0xf826023fu,                     /* stadd x6, [x17]        JQW_TOTAL */
        0x91002231u,                     /* add x17, x17, #8 */
        0xf826623fu,                     /* stumax x6, [x17]       JQW_LONGEST */
        0xd2800026u,                     /* mov x6, #1 */
        0x91002231u,                     /* add x17, x17, #8 */
        0xf826023fu,                     /* stadd x6, [x17]        JQW_WAITS */
        jqw_b(AT(32), retry),            /* b <the lock again> */
        jqw_b(AT(OK), ok),               /* ok: b <as shipped> */
    };
#undef AT
    for (int i = 0; i < JQW_WORDS; i++) w[i] = c[i];
}
