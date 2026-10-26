/* NVN physical / virtual memory pools on Linux (in place of third_party/d2r_gfx/gfx/nvn/totk_vpool.c, which aliases with
   Darwin VM calls). A virtual pool is a guest-memory range whose 64 KiB pages map pages of physical pools; here the
   physical pool's memfd pages are mapped at the virtual address (guest_mem_alias), which keeps CPU and GPU writes
   through either view on the same pages, and the alias table tells the GPU side which pages are where. */

#include "totk_vpool.h"
#include "nvn_mtl.h"
#include "nvn_dirty.h"
#include "../guest_mem.h"
#include "../../third_party/d2r_gfx/host/hle.h"
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define F_CPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS
#define F_PHYSICAL NVN_MEMORY_POOL_FLAGS_PHYSICAL
#define F_VIRTUAL NVN_MEMORY_POOL_FLAGS_VIRTUAL
#define POOL_MAGIC MAGIC('P', 'O', 'O', 'L')
static pthread_mutex_t g_map_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t host_page(void) { return (uint64_t)getpagesize(); }
static uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

int totk_vpool_init(pool_t *o) {
    int phys = (o->flags & F_PHYSICAL) != 0, virt = (o->flags & F_VIRTUAL) != 0;
    o->game_mem = o->mem; o->host = NULL; o->span = 0;
    if (phys == virt) { fprintf(stderr, "[vpool] pool flags %#x must be exactly one of PHYSICAL / VIRTUAL\n", o->flags); return 0; }
    if (!o->size || o->size % TOTK_VPOOL_PAGE) { fprintf(stderr, "[vpool] pool size %#llx not a multiple of 64 KiB\n", (unsigned long long)o->size); return 0; }
    if (virt && o->mem) { fprintf(stderr, "[vpool] virtual pool given storage\n"); return 0; }
    if (phys && !o->mem) { fprintf(stderr, "[vpool] physical pool without storage\n"); return 0; }
    if (phys && !(o->flags & F_CPU_NO_ACCESS)) {
        if ((uintptr_t)o->mem % host_page()) { fprintf(stderr, "[vpool] CPU-visible physical pool %p not %#llx aligned: cannot alias\n", (void *)o->mem, (unsigned long long)host_page()); return 0; }
        return 1;   /* aliases the game storage directly */
    }
    o->span = round_up(o->size, host_page());
    void *p = guest_mem_map_anywhere((size_t)o->span, virt ? "vpool" : "ppool");
    if (p == MAP_FAILED) { fprintf(stderr, "[vpool] storage of %#llx bytes failed\n", (unsigned long long)o->span); return 0; }
    o->host = p; o->mem = o->host;
    return 1;
}
int totk_vpool_fini(pool_t *o) {
    if (!o->host) return 0;
    guest_mem_release(o->host, o->span);
    o->host = NULL; o->mem = o->game_mem; o->span = 0;
    return 1;
}
static int check_request(const pool_t *v, const totk_mapping_request *r, int i) {
    uint64_t end = (uint64_t)r->virtual_offset + r->size;
    if (r->virtual_offset < 0 || r->physical_offset < 0 || !r->size) return 0;
    if (((uint64_t)r->virtual_offset | (uint64_t)r->physical_offset | r->size) % TOTK_VPOOL_PAGE) { fprintf(stderr, "[vpool] request %d not 64 KiB aligned\n", i); return 0; }
    if (end < r->size || end > v->size) return 0;
    if (!r->physical_pool) return 1;
    const pool_t *p = (const pool_t *)(const void *)r->physical_pool;
    if (p->magic != POOL_MAGIC || !(p->flags & F_PHYSICAL)) return 0;
    uint64_t pend = (uint64_t)r->physical_offset + r->size;
    return pend >= r->size && pend <= p->size;
}
typedef struct { uintptr_t va, cpu; uint64_t size; } vmap_ent;
static vmap_ent g_vmap[512]; static int g_nvmap;
static void vmap_note(uintptr_t va, uint64_t size, uintptr_t cpu) {
    for (int i = 0; i < g_nvmap; i++) if (g_vmap[i].va == va) { g_vmap[i] = (vmap_ent){va, cpu, size}; return; }
    if (g_nvmap < 512) g_vmap[g_nvmap++] = (vmap_ent){va, cpu, size};
}
int totk_vpool_backing(uintptr_t va, uint64_t len, uintptr_t *cpu) {
    pthread_mutex_lock(&g_map_mu); int r = 0;
    for (int i = 0; i < g_nvmap; i++) { const vmap_ent *e = &g_vmap[i]; if (e->cpu && va >= e->va && va + len <= e->va + e->size) { *cpu = e->cpu + (va - e->va); r = 1; break; } }
    pthread_mutex_unlock(&g_map_mu); return r;
}
int totk_vpool_map(pool_t *v, int count, const totk_mapping_request *r) {
    if (!v || v->magic != POOL_MAGIC || !(v->flags & F_VIRTUAL) || !v->mem) return 0;
    if (count < 0 || (count && !r)) return 0;
    for (int i = 0; i < count; i++) if (!check_request(v, &r[i], i)) return 0;
    pthread_mutex_lock(&g_map_mu);
    nvn_dirty_remove_range(v->mem);
    mtl_pool_del(v->mem);
    int ok = 1;
    for (int i = 0; ok && i < count; i++) {
        uintptr_t dst = (uintptr_t)(v->mem + r[i].virtual_offset);
        if (!r[i].physical_pool) { ok = guest_mem_unalias(dst, r[i].size, PROT_READ | PROT_WRITE) == 0; vmap_note(dst, r[i].size, 0); }
        else {
            const pool_t *p = (const pool_t *)(const void *)r[i].physical_pool;
            uintptr_t src = (uintptr_t)(p->mem + r[i].physical_offset);
            ok = guest_mem_alias(dst, src, r[i].size, PROT_READ | PROT_WRITE) == 0;
            if (ok) vmap_note(dst, r[i].size, src);
        }
    }
    mtl_pool_add(v->mem, v->size);
    nvn_dirty_add_range(v->mem, v->size);
    pthread_mutex_unlock(&g_map_mu);
    if (!ok) die("vpool: mapping failed after validation; virtual pool %p is inconsistent", (void *)v);
    return 1;
}
