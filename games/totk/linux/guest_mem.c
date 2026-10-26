/* Guest memory on Linux: every range the game can address (stack + heap, the alias region, the code modules) is a
   shared mapping of its own memfd. The CPU uses the mapping as plain memory; the Vulkan backend hands the same pages
   to the GPU as udmabuf dma-bufs (Honeykrisp has no VK_EXT_external_memory_host), so a memory pool needs no copy and
   CPU and GPU see one copy of the bytes, as on the Switch and with Metal's no-copy buffers.
   Decommitting punches the pages out of the memfd (fresh zero pages on the next touch) instead of mapping anonymous
   memory over them, which would detach the address from its memfd. */
#include "guest_mem.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAX_REGIONS 64
typedef struct { uintptr_t lo, hi; int fd; char name[24]; } region_t;
static region_t g_reg[MAX_REGIONS];
static int g_nreg;
static pthread_rwlock_t g_lock = PTHREAD_RWLOCK_INITIALIZER;

static int new_memfd(const char *name, uint64_t size) {
    char n[64]; snprintf(n, sizeof n, "guest-%s", name);
    int fd = memfd_create(n, MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) return -1;
    /* udmabuf takes pages only from memfds that can never shrink */
    if (ftruncate(fd, (off_t)size) || fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK)) { close(fd); return -1; }
    return fd;
}
static void add_region(uintptr_t lo, uint64_t size, int fd, const char *name) {
    pthread_rwlock_wrlock(&g_lock);
    if (g_nreg == MAX_REGIONS) { pthread_rwlock_unlock(&g_lock); fprintf(stderr, "[guest-mem] region table full\n"); abort(); }
    region_t *r = &g_reg[g_nreg++]; r->lo = lo; r->hi = lo + size; r->fd = fd; snprintf(r->name, sizeof r->name, "%s", name);
    pthread_rwlock_unlock(&g_lock);
}

uint8_t *guest_mem_reserve_fixed(uintptr_t at, uint64_t size, const char *name) {
    int fd = new_memfd(name, size);
    if (fd < 0) return NULL;
    void *p = mmap((void *)at, (size_t)size, PROT_NONE, MAP_SHARED | MAP_FIXED_NOREPLACE | MAP_NORESERVE, fd, 0);
    if (p == MAP_FAILED || (uintptr_t)p != at) { if (p != MAP_FAILED) munmap(p, (size_t)size); close(fd); return NULL; }
    add_region(at, size, fd, name);
    return p;
}
void *guest_mem_map_anywhere(size_t size, const char *name) {
    int fd = new_memfd(name, size);
    if (fd < 0) return MAP_FAILED;
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (p == MAP_FAILED) { close(fd); return MAP_FAILED; }
    add_region((uintptr_t)p, size, fd, name);
    return p;
}
void guest_mem_release(void *at, uint64_t size) {
    pthread_rwlock_wrlock(&g_lock);
    for (int i = 0; i < g_nreg; i++) if (g_reg[i].lo == (uintptr_t)at) {
        munmap(at, (size_t)size); close(g_reg[i].fd); g_reg[i] = g_reg[--g_nreg]; break;
    }
    pthread_rwlock_unlock(&g_lock);
}
int guest_mem_decommit(void *at, uint64_t size) {
    int fd; uint64_t off, run;
    if (!guest_mem_lookup((uintptr_t)at, &fd, &off, &run) || run < size) return munmap(at, size) ? -1 : 0;
    if (mprotect(at, (size_t)size, PROT_NONE)) return -1;
    return fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t)off, (off_t)size);
}
/* Aliases: ranges of a region whose pages are another region's (NVN virtual pools mapping physical pools). The GPU
   side (linux/vk/vk_pool.c) must see the pages that are actually mapped, so lookups take the aliases first. */
#define MAX_ALIASES 4096
typedef struct { uintptr_t lo, hi; int fd; uint64_t off; } alias_t;
static alias_t g_alias[MAX_ALIASES];
static int g_nalias;
static void alias_cut_locked(uintptr_t lo, uintptr_t hi) {   /* forget alias coverage of [lo, hi) */
    for (int i = 0; i < g_nalias; i++) {
        alias_t *a = &g_alias[i];
        if (a->hi <= lo || a->lo >= hi) continue;
        if (a->lo < lo && a->hi > hi && g_nalias < MAX_ALIASES) {   /* split */
            alias_t tail = {hi, a->hi, a->fd, a->off + (hi - a->lo)};
            a->hi = lo; g_alias[g_nalias++] = tail; continue;
        }
        if (a->lo < lo) { a->hi = lo; continue; }
        if (a->hi > hi) { a->off += hi - a->lo; a->lo = hi; continue; }
        g_alias[i--] = g_alias[--g_nalias];
    }
}
int guest_mem_lookup(uintptr_t addr, int *fd, uint64_t *off, uint64_t *run) {
    int ok = 0;
    pthread_rwlock_rdlock(&g_lock);
    for (int i = 0; i < g_nalias; i++) if (addr >= g_alias[i].lo && addr < g_alias[i].hi) {
        if (fd) *fd = g_alias[i].fd;
        if (off) *off = g_alias[i].off + (addr - g_alias[i].lo);
        if (run) *run = g_alias[i].hi - addr;
        pthread_rwlock_unlock(&g_lock);
        return 1;
    }
    for (int i = 0; i < g_nreg; i++) if (addr >= g_reg[i].lo && addr < g_reg[i].hi) {
        if (fd) *fd = g_reg[i].fd;
        if (off) *off = addr - g_reg[i].lo;
        uint64_t r = g_reg[i].hi - addr;
        for (int k = 0; k < g_nalias; k++) if (g_alias[k].lo > addr && g_alias[k].lo - addr < r) r = g_alias[k].lo - addr;   /* stop at the next alias */
        if (run) *run = r;
        ok = 1; break;
    }
    pthread_rwlock_unlock(&g_lock);
    return ok;
}
/* Map the pages behind src (guest memory) a second time at dst: shared, so CPU writes through either address meet. */
int guest_mem_alias(uintptr_t dst, uintptr_t src, uint64_t size, int prot) {
    int fd; uint64_t off, run;
    if (!guest_mem_lookup(src, &fd, &off, &run) || run < size) { errno = EINVAL; return -1; }
    void *p = mmap((void *)dst, (size_t)size, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)off);
    if (p == MAP_FAILED) return -1;
    pthread_rwlock_wrlock(&g_lock);
    alias_cut_locked(dst, dst + size);
    if (g_nalias < MAX_ALIASES) g_alias[g_nalias++] = (alias_t){dst, dst + size, fd, off};
    else fprintf(stderr, "[guest-mem] alias table full: the GPU may see stale pages\n");
    pthread_rwlock_unlock(&g_lock);
    return 0;
}
/* Give [dst, dst+size) of its own region its own (zeroed) pages back. */
int guest_mem_unalias(uintptr_t dst, uint64_t size, int prot) {
    pthread_rwlock_wrlock(&g_lock);
    alias_cut_locked(dst, dst + size);
    int fd = -1; uint64_t off = 0;
    for (int i = 0; i < g_nreg; i++) if (dst >= g_reg[i].lo && dst + size <= g_reg[i].hi) { fd = g_reg[i].fd; off = dst - g_reg[i].lo; break; }
    pthread_rwlock_unlock(&g_lock);
    if (fd < 0) { errno = EINVAL; return -1; }
    if (fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t)off, (off_t)size)) return -1;
    void *p = mmap((void *)dst, (size_t)size, prot, MAP_SHARED | MAP_FIXED, fd, (off_t)off);
    return p == MAP_FAILED ? -1 : 0;
}
