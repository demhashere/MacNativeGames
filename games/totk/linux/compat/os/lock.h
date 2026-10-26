/* os_unfair_lock: a futex mutex (0 free, 1 held, 2 held with waiters). */
#pragma once
#include <stdint.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
typedef struct { _Atomic uint32_t v; } os_unfair_lock, *os_unfair_lock_t;
#define OS_UNFAIR_LOCK_INIT {0}
static inline void os_unfair_lock_lock(os_unfair_lock_t l) {
    uint32_t c = 0;
    if (__atomic_compare_exchange_n((uint32_t *)&l->v, &c, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
    if (c != 2) c = __atomic_exchange_n((uint32_t *)&l->v, 2, __ATOMIC_ACQUIRE);
    while (c) { syscall(SYS_futex, &l->v, FUTEX_WAIT_PRIVATE, 2, NULL, NULL, 0); c = __atomic_exchange_n((uint32_t *)&l->v, 2, __ATOMIC_ACQUIRE); }
}
static inline int os_unfair_lock_trylock(os_unfair_lock_t l) { uint32_t c = 0; return __atomic_compare_exchange_n((uint32_t *)&l->v, &c, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED); }
static inline void os_unfair_lock_unlock(os_unfair_lock_t l) {
    if (__atomic_exchange_n((uint32_t *)&l->v, 0, __ATOMIC_RELEASE) == 2) syscall(SYS_futex, &l->v, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
}
static inline void os_unfair_lock_assert_owner(os_unfair_lock_t l) { (void)l; }
