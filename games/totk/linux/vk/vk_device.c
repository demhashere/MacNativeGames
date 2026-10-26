/* The Vulkan device of the NVN replay: one queue (graphics + compute + transfer + sparse binding), one timeline semaphore
   all submissions signal in queue order, a completion thread that runs what has to happen once the GPU is done (the
   CPU-visible effects of a submission, the release of objects it used) and the host-visible arenas the replay copies
   uniforms and driver constants into. The ordering rule of the Metal backend holds: every "GPU done" effect the game can
   see happens only after the submission has completed on the GPU. */

#include "vk_priv.h"
#include "frontend_sdl.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

vk_dev V;

void vk_log(const char *fmt, ...) {
    char b[2048]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    fprintf(stderr, "[vk] %s\n", b);
}
_Noreturn void vk_die(const char *fmt, ...) {
    char b[2048]; va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    fprintf(stderr, "[vk] fatal: %s\n", b);
    fflush(stderr);
    abort();
}
uint64_t vk_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < V.memp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (V.memp.memoryTypes[i].propertyFlags & want) == want) return i;
    for (uint32_t i = 0; i < V.memp.memoryTypeCount; i++) if (bits & (1u << i)) return i;
    VDIE("no memory type for bits %#x", bits);
}
void vk_barrier_all(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

/* ---------------- completion thread: callbacks and deferred destruction ---------------- */
typedef enum { D_FN, D_IMAGE, D_VIEW, D_BUFFER, D_BUFVIEW, D_SHADER, D_FREE } dkind;
typedef struct { uint64_t v; dkind k; void (*fn)(void *); void *arg; uint64_t h; VmaAllocation a; } dent;
static dent *g_d; static size_t g_nd, g_capd;
static dent *g_dpend; static size_t g_ndpend, g_capdpend;   /* waiting for the next replay submission's value */
static pthread_mutex_t g_dmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_cv = PTHREAD_COND_INITIALIZER;

static void dpush(dent **arr, size_t *n, size_t *cap, dent e) {
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 1024; *arr = realloc(*arr, *cap * sizeof **arr); if (!*arr) VDIE("deferred list"); }
    (*arr)[(*n)++] = e;
}
static void dadd(dent e) {
    pthread_mutex_lock(&g_dmu);
    if (e.v == UINT64_MAX) dpush(&g_dpend, &g_ndpend, &g_capdpend, e);
    else dpush(&g_d, &g_nd, &g_capd, e);
    pthread_cond_broadcast(&g_work_cv);
    pthread_mutex_unlock(&g_dmu);
}
void vk_on_done(uint64_t v, void (*fn)(void *), void *arg) { dadd((dent){v, D_FN, fn, arg, 0, NULL}); }
void vk_defer_image(uint64_t v, VkImage img, VmaAllocation a) { if (img) dadd((dent){v, D_IMAGE, NULL, NULL, (uint64_t)(uintptr_t)img, a}); }
void vk_defer_view(uint64_t v, VkImageView view) { if (view) dadd((dent){v, D_VIEW, NULL, NULL, (uint64_t)(uintptr_t)view, NULL}); }
void vk_defer_buffer(uint64_t v, VkBuffer b, VmaAllocation a) { if (b) dadd((dent){v, D_BUFFER, NULL, NULL, (uint64_t)(uintptr_t)b, a}); }
void vk_defer_bufview(uint64_t v, VkBufferView bv) { if (bv) dadd((dent){v, D_BUFVIEW, NULL, NULL, (uint64_t)(uintptr_t)bv, NULL}); }
void vk_defer_shader(uint64_t v, VkShaderEXT s) { if (s) dadd((dent){v, D_SHADER, NULL, NULL, (uint64_t)(uintptr_t)s, NULL}); }
void vk_defer_free(uint64_t v, void *p) { if (p) dadd((dent){v, D_FREE, NULL, p, 0, NULL}); }
uint64_t vk_pending_value(void) { return UINT64_MAX; }

static void drun(const dent *e) {
    switch (e->k) {
        case D_FN: e->fn(e->arg); break;
        case D_IMAGE: vmaDestroyImage(V.vma, (VkImage)(uintptr_t)e->h, e->a); break;
        case D_VIEW: vkDestroyImageView(V.dev, (VkImageView)(uintptr_t)e->h, NULL); break;
        case D_BUFFER: if (e->a) vmaDestroyBuffer(V.vma, (VkBuffer)(uintptr_t)e->h, e->a); else vkDestroyBuffer(V.dev, (VkBuffer)(uintptr_t)e->h, NULL); break;
        case D_BUFVIEW: vkDestroyBufferView(V.dev, (VkBufferView)(uintptr_t)e->h, NULL); break;
        case D_SHADER: V.DestroyShaderEXT(V.dev, (VkShaderEXT)(uintptr_t)e->h, NULL); break;
        case D_FREE: free(e->arg); break;
    }
}
static void *completion_main(void *arg) {
    (void)arg;
    pthread_setname_np("vk-complete");
    uint64_t seen = 0;
    for (;;) {
        uint64_t last = vk_last_value();
        if (last <= seen) {
            pthread_mutex_lock(&g_dmu);
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += 2000000; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&g_work_cv, &g_dmu, &ts);
            pthread_mutex_unlock(&g_dmu);
            continue;
        }
        uint64_t want = seen + 1;
        VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &V.tl, .pValues = &want};
        VkResult r = vkWaitSemaphores(V.dev, &wi, 1000000000ull);
        if (r == VK_TIMEOUT) continue;
        if (r != VK_SUCCESS) VDIE("timeline wait failed: %d (device lost?)", r);
        uint64_t done = 0; VK(vkGetSemaphoreCounterValue(V.dev, V.tl, &done));
        if (done > last) done = last;
        /* run everything due, in registration order, outside the lock (callbacks may defer more work) */
        pthread_mutex_lock(&g_dmu);
        size_t n = g_nd, k = 0; dent *due = NULL; size_t ndue = 0;
        if (n) { due = malloc(n * sizeof *due); if (!due) VDIE("completion list"); }
        for (size_t i = 0; i < n; i++) { if (g_d[i].v <= done) due[ndue++] = g_d[i]; else g_d[k++] = g_d[i]; }
        g_nd = k;
        pthread_mutex_unlock(&g_dmu);
        for (size_t i = 0; i < ndue; i++) if (due[i].k == D_FN) drun(&due[i]);   /* CPU-visible effects first */
        for (size_t i = 0; i < ndue; i++) if (due[i].k != D_FN) drun(&due[i]);
        free(due);
        seen = done;
        pthread_mutex_lock(&V.done_mu); atomic_store(&V.tl_done, done); pthread_cond_broadcast(&V.done_cv); pthread_mutex_unlock(&V.done_mu);
    }
    return NULL;
}
uint64_t vk_done_value(void) { return atomic_load(&V.tl_done); }
int vk_wait_value(uint64_t v, uint64_t timeout_ns) {
    if (atomic_load(&V.tl_done) >= v) return 1;
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600000000000ull ? 3600000000000ull : timeout_ns);
    dl.tv_sec += (time_t)(ns / 1000000000ull); dl.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&V.done_mu);
    while (atomic_load(&V.tl_done) < v) if (pthread_cond_timedwait(&V.done_cv, &V.done_mu, &dl)) break;
    int ok = atomic_load(&V.tl_done) >= v;
    pthread_mutex_unlock(&V.done_mu);
    return ok;
}

/* Submissions go to the queue from one submit thread, in the order their timeline values were given out: the replay
   hands its command buffers over and goes on recording (a vkQueueSubmit2 costs about 0.5 ms on Asahi, six or more a
   frame). One with a binary semaphore (the present's) first waits until everything queued before it is submitted, and
   is then submitted at once: a binary semaphore must be signalled by a submitted batch before anything waits on it.
   Timeline waits may come before their submission (vk_wait_value, the completion thread). TOTK_VK_ASYNC_SUBMIT=0: every
   submission at once, on the caller's thread. */
enum { SQ = 64 };
typedef struct { VkCommandBuffer cb[8]; int n; uint64_t v; } subjob;
static subjob g_sq[SQ]; static uint64_t g_sq_in, g_sq_out;   /* jobs given / submitted (g_sqmu) */
static pthread_mutex_t g_sqmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_sqcv = PTHREAD_COND_INITIALIZER;
static void submit_now(VkCommandBuffer const *cbs, int n, uint64_t v, VkSemaphore wait, VkPipelineStageFlags2 wait_stage, VkSemaphore signal_bin) {
    VkCommandBufferSubmitInfo ci[8];
    for (int i = 0; i < n; i++) ci[i] = (VkCommandBufferSubmitInfo){VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cbs[i]};
    VkSemaphoreSubmitInfo sig[2] = {{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = V.tl, .value = v, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                    {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = signal_bin, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo wt = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = wait, .stageMask = wait_stage};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &wt,
                        .commandBufferInfoCount = (uint32_t)n, .pCommandBufferInfos = ci, .signalSemaphoreInfoCount = signal_bin ? 2 : 1, .pSignalSemaphoreInfos = sig};
    pthread_mutex_lock(&V.qmu);
    VkResult r = vkQueueSubmit2(V.q, 1, &si, VK_NULL_HANDLE);
    pthread_mutex_unlock(&V.qmu);
    if (r != VK_SUCCESS) VDIE("vkQueueSubmit2 failed: %d", r);
}
static void *submit_main(void *arg) {
    (void)arg;
    pthread_setname_np("vk-submit");
    for (;;) {
        pthread_mutex_lock(&g_sqmu);
        while (g_sq_out == g_sq_in) pthread_cond_wait(&g_sqcv, &g_sqmu);
        subjob j = g_sq[g_sq_out % SQ];
        pthread_mutex_unlock(&g_sqmu);
        submit_now(j.cb, j.n, j.v, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
        pthread_mutex_lock(&g_sqmu); g_sq_out++; pthread_cond_broadcast(&g_sqcv); pthread_mutex_unlock(&g_sqmu);
    }
    return NULL;
}
static int async_submit(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("TOTK_VK_ASYNC_SUBMIT"); on = !(e && !strcmp(e, "0"));
        if (on) { pthread_t t; if (pthread_create(&t, NULL, submit_main, NULL)) on = 0; else pthread_detach(t); }
    }
    return on;
}
uint64_t vk_last_value(void) { pthread_mutex_lock(&g_sqmu); uint64_t v = V.tl_last; pthread_mutex_unlock(&g_sqmu); return v; }
uint64_t vk_submit_ex(VkCommandBuffer const *cbs, int n, VkSemaphore wait, VkPipelineStageFlags2 wait_stage, VkSemaphore signal_bin, int replay) {
    if (n > 8) VDIE("too many command buffers in one submit");
    int async = async_submit() && !wait && !signal_bin;
    pthread_mutex_lock(&g_sqmu);
    if (async) while (g_sq_in - g_sq_out >= SQ) pthread_cond_wait(&g_sqcv, &g_sqmu);
    else while (g_sq_out != g_sq_in) pthread_cond_wait(&g_sqcv, &g_sqmu);   /* everything before it on the queue first */
    uint64_t v = ++V.tl_last;
    if (async) {
        subjob *j = &g_sq[g_sq_in % SQ]; memcpy(j->cb, cbs, sizeof *cbs * (size_t)n); j->n = n; j->v = v;
        g_sq_in++; pthread_cond_broadcast(&g_sqcv);
    } else submit_now(cbs, n, v, wait, wait_stage, signal_bin);   /* g_sqmu held: nothing is queued in between */
    pthread_mutex_unlock(&g_sqmu);
    /* objects released while the replay's submission was being recorded go with it (not with a present or a one-off
       copy that happens to be submitted first: those do not cover what the replay's command buffer still uses) */
    if (replay) {
        pthread_mutex_lock(&g_dmu);
        for (size_t i = 0; i < g_ndpend; i++) { g_dpend[i].v = v; dpush(&g_d, &g_nd, &g_capd, g_dpend[i]); }
        g_ndpend = 0;
        pthread_mutex_unlock(&g_dmu);
    }
    pthread_mutex_lock(&g_dmu); pthread_cond_broadcast(&g_work_cv); pthread_mutex_unlock(&g_dmu);
    return v;
}
uint64_t vk_submit(VkCommandBuffer const *cbs, int n, VkSemaphore wait, VkPipelineStageFlags2 wait_stage, VkSemaphore signal_bin) {
    return vk_submit_ex(cbs, n, wait, wait_stage, signal_bin, 0);
}

/* ---------------- arenas ---------------- */
#define ARENA_CHUNK (16u << 20)
typedef struct arena_chunk { VkBuffer buf; VmaAllocation a; uint8_t *cpu; size_t size, used; uint64_t v; struct arena_chunk *next; } arena_chunk;
static arena_chunk *g_cur;          /* chunks of the open submission, newest first */
static arena_chunk *g_retired;      /* waiting for their value */
static arena_chunk *g_free;
static pthread_mutex_t g_amu = PTHREAD_MUTEX_INITIALIZER;
static arena_chunk *chunk_new(size_t size) {
    arena_chunk *c = calloc(1, sizeof *c); if (!c) VDIE("arena");
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    VK(vmaCreateBuffer(V.vma, &bi, &ai, &c->buf, &c->a, &info));
    c->cpu = info.pMappedData; c->size = size;
    return c;
}
static void arena_recycle(void *arg) {   /* completion thread: a retired chunk is free again */
    arena_chunk *c = arg;
    pthread_mutex_lock(&g_amu);
    if (c->size == ARENA_CHUNK) { c->used = 0; c->next = g_free; g_free = c; }
    else { vmaDestroyBuffer(V.vma, c->buf, c->a); free(c); }
    pthread_mutex_unlock(&g_amu);
}
vk_slice vk_arena_alloc(size_t n, size_t align) {
    if (!align) align = 16;
    if (!n) n = 16;
    arena_chunk *c = g_cur;
    size_t at = c ? (c->used + align - 1) & ~(align - 1) : 0;
    if (!c || at + n > c->size) {
        if (n > ARENA_CHUNK / 2) c = chunk_new((n + 0xffff) & ~(size_t)0xffff);
        else {
            pthread_mutex_lock(&g_amu); c = g_free; if (c) g_free = c->next; pthread_mutex_unlock(&g_amu);
            if (!c) c = chunk_new(ARENA_CHUNK);
        }
        c->used = 0; c->next = g_cur; g_cur = c; at = 0;
    }
    c->used = at + n;
    return (vk_slice){c->buf, at, c->cpu + at, 0};
}
void vk_arena_retire(uint64_t v) {
    for (arena_chunk *c = g_cur, *nx; c; c = nx) { nx = c->next; c->v = v; vk_on_done(v, arena_recycle, c); }
    g_cur = NULL;
}

/* ---------------- one-off command buffers (any thread) ---------------- */
/* Each thread records from a command pool of its own (a pool must not be used by two threads at once, and recording uses
   it); a buffer whose submission completed is handed back to its pool's list by the completion thread and freed by the
   owning thread at its next one-off. */
typedef struct { VkCommandPool pool; pthread_mutex_t mu; VkCommandBuffer done[256]; int ndone; } oneoff_pool;
static _Thread_local oneoff_pool *t_oneoff;
typedef struct { oneoff_pool *op; VkCommandBuffer cb; } oneoff_ent;
static oneoff_ent g_oneoff_open[64]; static pthread_mutex_t g_oneoff_mu = PTHREAD_MUTEX_INITIALIZER;   /* cb -> its pool, while recorded */
VkCommandBuffer vk_oneoff_begin(void) {
    if (!t_oneoff) {
        t_oneoff = calloc(1, sizeof *t_oneoff); if (!t_oneoff) VDIE("one-off pool");
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = V.qfam};
        VK(vkCreateCommandPool(V.dev, &pci, NULL, &t_oneoff->pool)); pthread_mutex_init(&t_oneoff->mu, NULL);
    }
    oneoff_pool *op = t_oneoff;
    VkCommandBuffer done[256]; int n;
    pthread_mutex_lock(&op->mu); n = op->ndone; memcpy(done, op->done, sizeof *done * (size_t)n); op->ndone = 0; pthread_mutex_unlock(&op->mu);
    if (n) vkFreeCommandBuffers(V.dev, op->pool, (uint32_t)n, done);
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = op->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(V.dev, &ai, &cb));
    pthread_mutex_lock(&g_oneoff_mu);
    int k = 0; while (k < 64 && g_oneoff_open[k].cb) k++;
    if (k == 64) VDIE("more than 64 one-off command buffers recorded at once");
    g_oneoff_open[k] = (oneoff_ent){op, cb};
    pthread_mutex_unlock(&g_oneoff_mu);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK(vkBeginCommandBuffer(cb, &bi));
    vk_barrier_all(cb);
    return cb;
}
static void oneoff_free(void *arg) {   /* completion thread: back to the owner's list */
    oneoff_ent *e = arg;
    pthread_mutex_lock(&e->op->mu);
    if (e->op->ndone < 256) e->op->done[e->op->ndone++] = e->cb; else VK_ONCE(oneleak, "one-off command buffers: more than 256 waiting to be freed");
    pthread_mutex_unlock(&e->op->mu);
    free(e);
}
uint64_t vk_oneoff_end_submit(VkCommandBuffer cb) {
    vk_barrier_all(cb);
    VK(vkEndCommandBuffer(cb));
    oneoff_ent *e = malloc(sizeof *e); if (!e) VDIE("one-off");
    pthread_mutex_lock(&g_oneoff_mu);
    int k = 0; while (k < 64 && g_oneoff_open[k].cb != cb) k++;
    if (k == 64) VDIE("one-off command buffer not from vk_oneoff_begin");
    *e = g_oneoff_open[k]; g_oneoff_open[k].cb = VK_NULL_HANDLE;
    pthread_mutex_unlock(&g_oneoff_mu);
    uint64_t v = vk_submit(&cb, 1, VK_NULL_HANDLE, 0, VK_NULL_HANDLE);
    vk_on_done(v, oneoff_free, e);
    return v;
}
void vk_oneoff_wait(uint64_t v) { vk_wait_value(v, UINT64_MAX); }

/* ---------------- device creation ---------------- */
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
#define DPROC(name) V.name = (PFN_vk##name)vkGetDeviceProcAddr(V.dev, "vk" #name)

void vk_device_init(void) {
    static int inited; if (inited) return; inited = 1;
    pthread_mutex_init(&V.qmu, NULL); pthread_mutex_init(&V.done_mu, NULL); pthread_cond_init(&V.done_cv, NULL);
    uint32_t nwin = 0; const char *const *wexts = front_vk_instance_extensions(&nwin);
    const char *iexts[16]; uint32_t ni = 0;
    for (uint32_t i = 0; i < nwin && ni < 14; i++) iexts[ni++] = wexts[i];
    if (nwin) {   /* the surface's presentation timing support is queried through it (vk_present.c) */
        uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
        VkExtensionProperties *ie = calloc(nie ? nie : 1, sizeof *ie); vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
        for (uint32_t i = 0; i < nie; i++) if (!strcmp(ie[i].extensionName, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) { iexts[ni++] = VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME; V.surface_caps2 = 1; }
        free(ie);
    }
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "botw-native", .apiVersion = VK_API_VERSION_1_4};
    const char *layers[1]; uint32_t nl = 0;
    if (getenv("TOTK_VK_VALIDATION") && atoi(getenv("TOTK_VK_VALIDATION"))) { layers[nl++] = "VK_LAYER_KHRONOS_validation"; VLOG("validation layer requested"); }
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app, .enabledExtensionCount = ni, .ppEnabledExtensionNames = iexts,
                                .enabledLayerCount = nl, .ppEnabledLayerNames = layers};
    VK(vkCreateInstance(&ici, NULL, &V.inst));
    uint32_t npd = 0; VK(vkEnumeratePhysicalDevices(V.inst, &npd, NULL));
    VkPhysicalDevice pds[8]; if (npd > 8) npd = 8; VK(vkEnumeratePhysicalDevices(V.inst, &npd, pds));
    const char *want = getenv("TOTK_VK_DEVICE");
    for (uint32_t i = 0; i < npd && !V.pd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        if (want ? strstr(p.deviceName, want) != NULL : p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) V.pd = pds[i];
    }
    if (!V.pd) VDIE("no Vulkan GPU (Honeykrisp) found");
    vkGetPhysicalDeviceProperties(V.pd, &V.props);
    vkGetPhysicalDeviceMemoryProperties(V.pd, &V.memp);
    if (V.props.apiVersion < VK_API_VERSION_1_3) VDIE("%s supports Vulkan %u.%u only (1.3 or newer needed)", V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion));
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, qf);
    V.qfam = UINT32_MAX;
    VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    for (uint32_t i = 0; i < nq; i++) if ((qf[i].queueFlags & need) == need) { V.qfam = i; break; }
    if (V.qfam == UINT32_MAX) VDIE("no queue family with graphics, compute and sparse binding");

    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL);
    VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ex);
    const char *dexts[32]; uint32_t nd = 0;
    const char *required[] = {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
    for (unsigned i = 0; i < sizeof required / sizeof *required; i++) { if (!has_ext(ex, ne, required[i])) VDIE("%s lacks %s", V.props.deviceName, required[i]); dexts[nd++] = required[i]; }
    /* Border colours other than transparent / opaque black and opaque white (the game's grass interaction map clamps to
       0.5 grey, nvn_mtl.m samp_border_custom) need VK_EXT_custom_border_color, which Honeykrisp emulates in the shader:
       every sample of every shader gets a test of its sampler and a second sample behind it (hk_shader.c
       agx_nir_lower_custom_border). Shader objects get that whatever the device enables (Mesa 26.2 vk_shader.c compiles
       them with no feature list, which hk takes as custom borders on): 17% more instructions in the game's fragment
       shaders. Only HK_PERFTEST=noborder takes it out (hk_device.c), so by default the extension is off, the driver is
       told so, and such borders clamp to the edge texel instead (vk_tex.c sampler_get). TOTK_VK_CUSTOM_BORDER=1: exact
       borders, with the emulation in every shader. */
    {   const char *e = getenv("TOTK_VK_CUSTOM_BORDER");
        V.custom_border = e && !strcmp(e, "1") && has_ext(ex, ne, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
        if (V.custom_border) dexts[nd++] = VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME; }
    V.has_window = nwin > 0 && has_ext(ex, ne, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (V.has_window) dexts[nd++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    /* Presentation times (vk_present.c: the frame hold's vblank clock, the on-screen frame statistics), with the
       extensions VK_EXT_present_timing depends on. TOTK_VK_PRESENT_TIMING=0: none. */
    int want_pt = V.has_window && V.surface_caps2 && !(getenv("TOTK_VK_PRESENT_TIMING") && !strcmp(getenv("TOTK_VK_PRESENT_TIMING"), "0")) &&
                  has_ext(ex, ne, VK_EXT_PRESENT_TIMING_EXTENSION_NAME) && has_ext(ex, ne, VK_KHR_PRESENT_ID_2_EXTENSION_NAME) && has_ext(ex, ne, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    /* not enabled: depth clip enable / control and provoking vertex (each would add dynamic state every draw must set;
       depth clamp alone already turns clipping off, the translated shaders convert NVN's clip z, and Metal's provoking
       vertex is the first one, Vulkan's default) */
    int depth_clip = 0, provoking = 0, depth_ctl = 0;
    free(ex);

    /* Every feature the device has, except robust buffer access (the Metal path has none either; costs bounds checks) */
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan14Features f14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
    VkPhysicalDeviceShaderObjectFeaturesEXT fso = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT fvi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT feds3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT feds2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fcb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceDepthClipEnableFeaturesEXT fdc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT fpv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT};
    VkPhysicalDeviceDepthClipControlFeaturesEXT fdcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDevicePresentTimingFeaturesEXT fpt = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};
    VkPhysicalDevicePresentId2FeaturesKHR fpi2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR};
    {   /* chain in order: core 1.1 .. 1.4, then the extensions this device enables */
        void **next = &f2.pNext;
        #define LINK(s) do { *next = &(s); next = &(s).pNext; } while (0)
        LINK(f11); LINK(f12); LINK(f13); if (V.props.apiVersion >= VK_API_VERSION_1_4) LINK(f14);
        LINK(fso); LINK(fvi); LINK(feds3); LINK(feds2); if (V.custom_border) LINK(fcb);
        if (depth_clip) LINK(fdc);
        if (provoking) LINK(fpv);
        if (depth_ctl) LINK(fdcc);
        if (want_pt) { LINK(fpt); LINK(fpi2); }
        #undef LINK
    }
    vkGetPhysicalDeviceFeatures2(V.pd, &f2);
    if (!fso.shaderObject || !fvi.vertexInputDynamicState || !f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore || !f12.descriptorBindingPartiallyBound)
        VDIE("%s lacks shader objects / dynamic vertex input / dynamic rendering / sync2 / timeline semaphores / partially bound descriptors", V.props.deviceName);
    if (!f2.features.sparseBinding || !f2.features.sparseResidencyBuffer) VDIE("%s lacks sparse buffers (needed to map guest memory)", V.props.deviceName);
    f2.features.robustBufferAccess = VK_FALSE;
    f12.bufferDeviceAddressCaptureReplay = VK_FALSE; f12.bufferDeviceAddressMultiDevice = VK_FALSE;
    V.present_timing = want_pt && fpt.presentTiming && fpi2.presentId2;
    if (V.present_timing) { dexts[nd++] = VK_EXT_PRESENT_TIMING_EXTENSION_NAME; dexts[nd++] = VK_KHR_PRESENT_ID_2_EXTENSION_NAME; dexts[nd++] = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
                            fpt.presentAtAbsoluteTime = fpt.presentAtRelativeTime = VK_FALSE; }   /* only the times of past presents */
    else if (want_pt) { fpt.presentTiming = fpt.presentAtAbsoluteTime = fpt.presentAtRelativeTime = VK_FALSE; fpi2.presentId2 = VK_FALSE; }
    V.host_copy = V.props.apiVersion >= VK_API_VERSION_1_4 && f14.hostImageCopy && !(getenv("TOTK_VK_HOST_COPY") && !atoi(getenv("TOTK_VK_HOST_COPY")));
    /* Driver options (HK_PERFTEST, read when the device is made; options already set are kept):
       - batch: Honeykrisp hands every render pass and compute run of a submission to the kernel in an ioctl of its own
         (Mesa 26.2 hk_queue.c max_commands_per_submit is 1 unless "batch", then 64), 150 and more a frame here.
         TOTK_HK_BATCH=0: one per ioctl, as the driver defaults.
       - noborder: no custom border emulation in the shaders (see V.custom_border above). */
    {   const char *hb = getenv("TOTK_HK_BATCH"), *pt = getenv("HK_PERFTEST");
        char v[256]; snprintf(v, sizeof v, "%s", pt ? pt : "");
        if (!(hb && !strcmp(hb, "0")) && !strstr(v, "batch")) snprintf(v + strlen(v), sizeof v - strlen(v), "%sbatch", *v ? "," : "");
        if (!V.custom_border && !strstr(v, "noborder")) snprintf(v + strlen(v), sizeof v - strlen(v), "%snoborder", *v ? "," : "");
        if (*v) setenv("HK_PERFTEST", v, 1);
        VLOG("Honeykrisp options: HK_PERFTEST=%s", *v ? v : "(none)"); }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = V.qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = nd, .ppEnabledExtensionNames = dexts};
    VK(vkCreateDevice(V.pd, &dci, NULL, &V.dev));
    vkGetDeviceQueue(V.dev, V.qfam, 0, &V.q);
    DPROC(CreateShadersEXT); DPROC(DestroyShaderEXT); DPROC(CmdBindShadersEXT); DPROC(CmdSetVertexInputEXT);
    DPROC(CmdSetPolygonModeEXT); DPROC(CmdSetRasterizationSamplesEXT); DPROC(CmdSetSampleMaskEXT); DPROC(CmdSetAlphaToCoverageEnableEXT);
    DPROC(CmdSetAlphaToOneEnableEXT); DPROC(CmdSetLogicOpEnableEXT); DPROC(CmdSetLogicOpEXT); DPROC(CmdSetColorBlendEnableEXT);
    DPROC(CmdSetColorBlendEquationEXT); DPROC(CmdSetColorWriteMaskEXT); DPROC(CmdSetDepthClampEnableEXT); DPROC(CmdSetDepthClipEnableEXT);
    DPROC(CmdSetProvokingVertexModeEXT); DPROC(CmdSetLineRasterizationModeEXT); DPROC(CmdSetTessellationDomainOriginEXT); DPROC(CmdSetPatchControlPointsEXT);
    DPROC(CmdSetDepthClipNegativeOneToOneEXT); DPROC(GetMemoryFdPropertiesKHR); DPROC(GetShaderBinaryDataEXT);
    if (!V.CreateShadersEXT || !V.CmdSetVertexInputEXT || !V.CmdSetColorBlendEquationEXT) VDIE("shader object / dynamic state entry points missing");
    if (!depth_clip) V.CmdSetDepthClipEnableEXT = NULL;
    if (!feds3.extendedDynamicState3AlphaToOneEnable || !f2.features.alphaToOne) V.CmdSetAlphaToOneEnableEXT = NULL;
    if (!feds3.extendedDynamicState3LogicOpEnable || !f2.features.logicOp) V.CmdSetLogicOpEnableEXT = NULL;
    if (!feds2.extendedDynamicState2LogicOp) V.CmdSetLogicOpEXT = NULL;
    if (!feds3.extendedDynamicState3DepthClipEnable) V.CmdSetDepthClipEnableEXT = NULL;
    if (!feds3.extendedDynamicState3ProvokingVertexMode || !provoking) V.CmdSetProvokingVertexModeEXT = NULL;
    if (!feds3.extendedDynamicState3LineRasterizationMode) V.CmdSetLineRasterizationModeEXT = NULL;
    if (!feds3.extendedDynamicState3TessellationDomainOrigin) V.CmdSetTessellationDomainOriginEXT = NULL;
    if (!feds2.extendedDynamicState2PatchControlPoints) V.CmdSetPatchControlPointsEXT = NULL;
    if (!depth_ctl || !feds3.extendedDynamicState3DepthClipNegativeOneToOne) V.CmdSetDepthClipNegativeOneToOneEXT = NULL;

    VmaVulkanFunctions vf = {.vkGetInstanceProcAddr = vkGetInstanceProcAddr, .vkGetDeviceProcAddr = vkGetDeviceProcAddr};
    VmaAllocatorCreateInfo vci = {.physicalDevice = V.pd, .device = V.dev, .instance = V.inst, .vulkanApiVersion = VK_API_VERSION_1_3, .pVulkanFunctions = &vf};
    VK(vmaCreateAllocator(&vci, &V.vma));
    VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st};
    VK(vkCreateSemaphore(V.dev, &sci, NULL, &V.tl));
    pthread_t th; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setstacksize(&a, 1u << 20);
    if (pthread_create(&th, &a, completion_main, NULL)) VDIE("completion thread");
    pthread_attr_destroy(&a); pthread_detach(th);
    VLOG("device %s (Vulkan %u.%u.%u, driver %#x), host image copy %s, custom border colours %s, presentation times %s, %u memory types", V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion),
         VK_API_VERSION_MINOR(V.props.apiVersion), VK_API_VERSION_PATCH(V.props.apiVersion), V.props.driverVersion, V.host_copy ? "on" : "off", V.custom_border ? "on" : "off",
         V.present_timing ? "asked for" : "off", V.memp.memoryTypeCount);
}
