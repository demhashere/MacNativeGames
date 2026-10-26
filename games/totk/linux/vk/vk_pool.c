/* NVN memory pools on Vulkan. A pool is a range of guest memory; guest memory is shared memfd pages (linux/guest_mem.c),
   so the GPU gets the very same pages: the pool's range, widened to whole host pages, is cut into udmabuf dma-bufs of at
   most 64 MB (the udmabuf size limit), each imported as VkDeviceMemory and bound into one sparse VkBuffer that spans the
   pool. Guest address a then is offset (a - base) in that buffer: no copies, CPU and GPU writes meet in memory, as with
   the Metal backend's no-copy buffers. Honeykrisp has no VK_EXT_external_memory_host, so this is the zero-copy path.
   The GPU buffer is made at the first GPU use of the pool (like Metal's lazy residency): many pools are never used by the
   GPU. A destroyed pool's buffer and memory are released once the GPU is done with the submissions that may use it. */

#include "vk_priv.h"
#include "../guest_mem.h"
#include "nvn_mtl.h"
#include <errno.h>
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define UDMABUF_MAX (64ull << 20)
typedef struct pool_ent {
    uintptr_t lo, hi;          /* [mem, mem + size) as the game gave it */
    uintptr_t blo, bhi;        /* host-page aligned range the buffer covers */
    VkBuffer buf;
    VkDeviceMemory *mem; int nmem;
    int state;                 /* 0 not made yet, 1 made, -1 failed */
} pool_ent;
static pool_ent **g_pools; static int g_npools, g_cappools;   /* sorted by lo */
static pthread_rwlock_t g_plock = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t g_make_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_udmabuf = -1;
/* The replay resolves dozens of guest addresses per draw (uniform, storage, vertex and index buffers; thousands of draws a
   frame), each under g_plock (nvn_mtl.m measured its lookups at 0.5 ms of the submit thread a frame, half of it the lock).
   Each thread keeps the last pools it resolved (pool_tls), valid while g_pool_gen (bumped under the write lock on every
   pool add / remove) is unchanged. Only a pool no neighbour overlaps is kept, so a hit is the pool the locked search
   returns. TOTK_POOL_TLS=0: always the locked search. */
static _Atomic uint64_t g_pool_gen = 1;
enum { POOL_TLS_N = 8 };
typedef struct { uintptr_t lo, hi, blo, bhi; VkBuffer buf; uint64_t gen; } pool_tls;
static _Thread_local pool_tls g_ptls[POOL_TLS_N]; static _Thread_local unsigned g_ptls_next, g_ptls_last;
static int pool_tls_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("TOTK_POOL_TLS"); v = !(e && !strcmp(e, "0")); } return v; }
static uint64_t g_page;

void vk_pool_init(void) {
    if (g_page) return;
    g_page = (uint64_t)sysconf(_SC_PAGESIZE);
    g_udmabuf = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (g_udmabuf < 0) VDIE("/dev/udmabuf: %s (the GPU needs it to see guest memory; check the device's ACL / the udmabuf module)", strerror(errno));
}
static int find_first_hi_above(uintptr_t a) {   /* first pool with hi > a */
    int lo = 0, hi = g_npools;
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pools[m]->lo <= a) lo = m + 1; else hi = m; }
    return lo - 1;   /* last pool with lo <= a, or -1 */
}
void mtl_pool_add(const void *mem, uint64_t size) {
    vk_pool_init();
    if (!mem || !size) return;
    pool_ent *p = calloc(1, sizeof *p); if (!p) VDIE("pool");
    p->lo = (uintptr_t)mem; p->hi = p->lo + size;
    p->blo = p->lo & ~(uintptr_t)(g_page - 1); p->bhi = (p->hi + g_page - 1) & ~(uintptr_t)(g_page - 1);
    pthread_rwlock_wrlock(&g_plock);
    if (g_npools == g_cappools) { g_cappools = g_cappools ? g_cappools * 2 : 256; g_pools = realloc(g_pools, sizeof *g_pools * (size_t)g_cappools); if (!g_pools) VDIE("pools"); }
    int i = find_first_hi_above(p->lo) + 1;
    memmove(g_pools + i + 1, g_pools + i, sizeof *g_pools * (size_t)(g_npools - i));
    g_pools[i] = p; g_npools++;
    atomic_fetch_add(&g_pool_gen, 1);
    pthread_rwlock_unlock(&g_plock);
}
typedef struct { VkBuffer buf; VkDeviceMemory *mem; int nmem; } pool_trash;
static void pool_free(void *arg) {
    pool_trash *t = arg;
    if (t->buf) vkDestroyBuffer(V.dev, t->buf, NULL);
    for (int i = 0; i < t->nmem; i++) vkFreeMemory(V.dev, t->mem[i], NULL);
    free(t->mem); free(t);
}
void mtl_pool_del(const void *mem) {
    pthread_rwlock_wrlock(&g_plock);
    pool_ent *p = NULL;
    for (int i = 0; i < g_npools; i++) if (g_pools[i]->lo == (uintptr_t)mem) {
        p = g_pools[i]; memmove(g_pools + i, g_pools + i + 1, sizeof *g_pools * (size_t)(g_npools - i - 1)); g_npools--; break;
    }
    atomic_fetch_add(&g_pool_gen, 1);
    pthread_rwlock_unlock(&g_plock);
    if (!p) return;
    pthread_mutex_lock(&g_make_mu);   /* a concurrent first use may still be making its buffer */
    pool_trash *t = calloc(1, sizeof *t); if (!t) VDIE("pool trash");
    t->buf = p->buf; t->mem = p->mem; t->nmem = p->nmem;
    pthread_mutex_unlock(&g_make_mu);
    if (t->buf || t->nmem) vk_on_done(UINT64_MAX, pool_free, t); else free(t);
    free(p);
}

/* Make the sparse buffer of a pool (g_make_mu held). */
static int pool_make(pool_ent *p) {
    uint64_t total = p->bhi - p->blo;
    VkExternalMemoryBufferCreateInfo ebi = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ebi, .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
        .size = total, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
    VkBuffer buf;
    if (vkCreateBuffer(V.dev, &bci, NULL, &buf) != VK_SUCCESS) { VLOG("pool %#lx+%#llx: sparse buffer not made", (unsigned long)p->lo, (unsigned long long)(p->hi - p->lo)); return -1; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(V.dev, buf, &mr);
    if (mr.alignment > g_page) { VLOG("pool: sparse alignment %llu above the host page", (unsigned long long)mr.alignment); vkDestroyBuffer(V.dev, buf, NULL); return -1; }
    int cap = (int)(total / UDMABUF_MAX) + 4, n = 0;
    VkDeviceMemory *mems = calloc((size_t)cap, sizeof *mems); VkSparseMemoryBind *binds = calloc((size_t)cap, sizeof *binds);
    if (!mems || !binds) VDIE("pool binds");
    for (uint64_t off = 0; off < total;) {
        int fd; uint64_t foff, run;
        if (!guest_mem_lookup(p->blo + off, &fd, &foff, &run)) { VLOG("pool %#lx: page %#lx is not guest memory (no memfd)", (unsigned long)p->lo, (unsigned long)(p->blo + off)); goto fail; }
        uint64_t sz = total - off; if (sz > run) sz = run; if (sz > UDMABUF_MAX) sz = UDMABUF_MAX;
        sz &= ~(g_page - 1); if (!sz) { VLOG("pool %#lx: region end not page aligned", (unsigned long)p->lo); goto fail; }
        if (n == cap) { cap *= 2; mems = realloc(mems, sizeof *mems * (size_t)cap); binds = realloc(binds, sizeof *binds * (size_t)cap); if (!mems || !binds) VDIE("pool binds"); }
        struct udmabuf_create c = {.memfd = (uint32_t)fd, .flags = UDMABUF_FLAGS_CLOEXEC, .offset = foff, .size = sz};
        int dfd = ioctl(g_udmabuf, UDMABUF_CREATE, &c);
        if (dfd < 0) { VLOG("pool %#lx: UDMABUF_CREATE (%#llx bytes at memfd offset %#llx): %s", (unsigned long)p->lo, (unsigned long long)sz, (unsigned long long)foff, strerror(errno)); goto fail; }
        VkMemoryFdPropertiesKHR fp = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        if (V.GetMemoryFdPropertiesKHR(V.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dfd, &fp) != VK_SUCCESS || !(fp.memoryTypeBits & mr.memoryTypeBits)) {
            VLOG("pool: dma-buf memory types %#x / buffer %#x", fp.memoryTypeBits, mr.memoryTypeBits); close(dfd); goto fail; }
        uint32_t types = fp.memoryTypeBits & mr.memoryTypeBits;
        VkImportMemoryFdInfoKHR imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = dfd};
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = sz, .memoryTypeIndex = vk_mem_type(types, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
        if (vkAllocateMemory(V.dev, &mai, NULL, &mems[n]) != VK_SUCCESS) { close(dfd); VLOG("pool: dma-buf import failed"); goto fail; }   /* on success Vulkan owns dfd */
        binds[n] = (VkSparseMemoryBind){.resourceOffset = off, .size = sz, .memory = mems[n]};
        n++; off += sz;
    }
    {
        VkSparseBufferMemoryBindInfo sbi = {.buffer = buf, .bindCount = (uint32_t)n, .pBinds = binds};
        VkBindSparseInfo bsi = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, .bufferBindCount = 1, .pBufferBinds = &sbi};
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK(vkCreateFence(V.dev, &fci, NULL, &fence));
        pthread_mutex_lock(&V.qmu);
        VkResult r = vkQueueBindSparse(V.q, 1, &bsi, fence);
        pthread_mutex_unlock(&V.qmu);
        if (r == VK_SUCCESS) r = vkWaitForFences(V.dev, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(V.dev, fence, NULL);
        if (r != VK_SUCCESS) { VLOG("pool: vkQueueBindSparse failed %d", r); goto fail; }
    }
    free(binds);
    p->buf = buf; p->mem = mems; p->nmem = n;
    static uint64_t made, bytes; made++; bytes += total;
    if (made <= 4 || made % 100 == 0) VLOG("memory pool %#lx+%#llx on the GPU (%d dma-bufs); %llu pools, %llu MB so far", (unsigned long)p->lo,
                                         (unsigned long long)(p->hi - p->lo), n, (unsigned long long)made, (unsigned long long)(bytes >> 20));
    return 1;
fail:
    for (int i = 0; i < n; i++) vkFreeMemory(V.dev, mems[i], NULL);
    free(mems); free(binds); vkDestroyBuffer(V.dev, buf, NULL);
    return -1;
}

int vk_pool_lookup(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *avail) {
    if (!addr) return 0;
    if (!len) len = 1;
    uint64_t gen = atomic_load_explicit(&g_pool_gen, memory_order_acquire);
    if (pool_tls_on()) for (unsigned j = 0; j < POOL_TLS_N; j++) {
        unsigned k = (g_ptls_last + j) % POOL_TLS_N; const pool_tls *e = &g_ptls[k];
        if (e->gen == gen && e->lo <= addr && addr + len <= e->hi) {
            g_ptls_last = k;
            if (buf) *buf = e->buf;
            if (off) *off = addr - e->blo;
            if (avail) *avail = e->bhi - addr;
            return 1;
        }
    }
    pthread_rwlock_rdlock(&g_plock);
    pool_ent *p = NULL;
    int i = find_first_hi_above((uintptr_t)addr), at = -1;
    /* pools may be nested in another pool's pages only at their edges; take the one that holds the whole range */
    for (int k = i; k >= 0 && k >= i - 4; k--) if (g_pools[k]->lo <= addr && addr + len <= g_pools[k]->hi) { p = g_pools[k]; at = k; break; }
    if (p && p->state == 0) {
        pthread_mutex_lock(&g_make_mu);
        if (p->state == 0) p->state = pool_make(p);
        pthread_mutex_unlock(&g_make_mu);
    }
    int ok = p && p->state == 1;
    if (ok) {
        if (buf) *buf = p->buf;
        if (off) *off = addr - p->blo;
        if (avail) *avail = p->bhi - addr;
        int alone = pool_tls_on();   /* kept for this thread when no neighbour overlaps it */
        for (int k = at - 4; alone && k <= at + 4; k++) if (k >= 0 && k < g_npools && k != at && g_pools[k]->lo < p->hi && p->lo < g_pools[k]->hi) alone = 0;
        if (alone) { g_ptls[g_ptls_next] = (pool_tls){p->lo, p->hi, p->blo, p->bhi, p->buf, gen}; g_ptls_last = g_ptls_next; g_ptls_next = (g_ptls_next + 1) % POOL_TLS_N; }
    }
    pthread_rwlock_unlock(&g_plock);
    return ok;
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    VkBuffer b; VkDeviceSize o;
    if (!vk_pool_lookup(addr, len, &b, &o, NULL)) return 0;
    if (buf) *buf = (void *)b;
    if (off) *off = o;
    return 1;
}
/* Is [addr, addr+len) inside some pool (without making its GPU buffer)? */
int vk_pool_contains(uint64_t addr, uint64_t len) {
    uint64_t gen = atomic_load_explicit(&g_pool_gen, memory_order_acquire);
    if (pool_tls_on()) for (unsigned j = 0; j < POOL_TLS_N; j++) { const pool_tls *e = &g_ptls[j]; if (e->gen == gen && e->lo <= addr && addr + len <= e->hi) return 1; }
    pthread_rwlock_rdlock(&g_plock);
    int i = find_first_hi_above((uintptr_t)addr), ok = 0;
    for (int k = i; k >= 0 && k >= i - 4; k--) if (g_pools[k]->lo <= addr && addr + len <= g_pools[k]->hi) { ok = 1; break; }
    pthread_rwlock_unlock(&g_plock);
    return ok;
}
