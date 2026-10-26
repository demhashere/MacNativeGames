/* The NVN command replay on Vulkan: the records of hle_nvn.c (nvn_obj.h OP_*) replayed into one command buffer per
   submission with dynamic rendering and shader objects (all state dynamic). The rules are nvn_mtl.m's:
   - SetRenderTargets opens a render pass lazily at its first draw / clear; clears before the first draw become load
     clears; a SetRenderTargets with the same targets continues the open pass.
   - Uniform buffers are read as of the submit call: a bound range is copied into the submission's arena (with the
     submission's earlier inline updates applied), unless the GPU writes it (then the pool memory is bound).
   - Inline uniform updates (OP_UPDATE_UB) are renamed: later bindings get private copies, and the updates reach memory
     at the end of the submission (or before anything else on the GPU touches those bytes).
   - CPU-visible effects (events, counters, occlusion results) happen once the GPU has completed the submission.
   Every pass, copy and dispatch is separated by a full memory barrier: correctness first. */

#include "vk_priv.h"
#include "nvn_hmap.h"
#include "vk_json.h"
#include "nvn_dirty.h"
#include "nvn_depth_clear.h"
#include "nvn_mtl.h"
#include "vk_builtin_spv.h"
#include "totk_ui_squeeze.h"
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

vk_state S; vk_rt_state R;
uint64_t vk_sub_serial;
static __thread int t_replay;
int vk_replaying(void) { return t_replay; }
uint64_t g_draw_skip[16];
static _Atomic uint64_t g_cur_frame;
static uint64_t g_stat_pass, g_stat_draw;
int vk_pool_contains(uint64_t addr, uint64_t len);
VkImageView vk_tex_storage_by_id(int tid, VkImageViewType want, vk_image **img_out);
const tex_t *vk_tex_slot(int tid);
vk_image *vk_tex_attach(const tex_t *t, const view_t *v, VkImageView *view, VkFormat *fmt, uint32_t *w, uint32_t *h, uint32_t *layers);
void mtl_present_flush(void);

/* ---------------- per-submission slots ---------------- */
#define NSLOT 6
#define DP_SETS 1024
#define NQUERY 4096
typedef struct {
    VkCommandPool pool; VkCommandBuffer cb, up;
    VkDescriptorPool dp[64]; int ndp, curdp;
    VkQueryPool qpool; uint32_t nq;
    VkQueryPool tsq;   /* GPU start and end of each segment of the submission (timestamps 2k, 2k + 1) */
    VkCommandBuffer seg_cb[16], seg_up[16]; int nseg;   /* mid-submission segments (seg_split), made on demand, reset with the pool */
    uint64_t value;
} slot_t;
static slot_t g_slots[NSLOT]; static int g_cs = -1;   /* current slot */
static VkCommandBuffer g_cb;                           /* the replay's command buffer of the open submission */
static int g_up_used;
static uint64_t g_last_fence;
static VkDescriptorSetLayout g_builtin_dsl_unused;
static VkPipelineLayout g_builtin_layout;

static VkDescriptorPool dpool_new(void) {
    VkDescriptorPoolSize ps[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VKB_NUBO * DP_SETS}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VKB_NSSBO * DP_SETS},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VKB_NCOMB * DP_SETS}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKB_NSAMPLED * DP_SETS},
        {VK_DESCRIPTOR_TYPE_SAMPLER, VKB_NSAMPLER * DP_SETS}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VKB_NSIMG * DP_SETS},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, VKB_NUTEX * DP_SETS}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VKB_NSTEX * DP_SETS}};
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = DP_SETS, .poolSizeCount = 8, .pPoolSizes = ps};
    VkDescriptorPool p; VK(vkCreateDescriptorPool(V.dev, &ci, NULL, &p));
    return p;
}
static void layouts_init(void) {
    VkDescriptorSetLayoutBinding b[VKB_COUNT]; VkDescriptorBindingFlags fl[VKB_COUNT];
    for (int i = 0; i < VKB_COUNT; i++) {
        VkDescriptorType t = i < VKB_SSBO0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : i < VKB_COMB0 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : i < VKB_SAMPLED0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                             i < VKB_SAMPLER0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : i < VKB_SIMG0 ? VK_DESCRIPTOR_TYPE_SAMPLER : i < VKB_UTEX0 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
                             i < VKB_STEX0 ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, t, 1, VK_SHADER_STAGE_ALL, NULL};
        fl[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, .bindingCount = VKB_COUNT, .pBindingFlags = fl};
    VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bf, .bindingCount = VKB_COUNT, .pBindings = b};
    VK(vkCreateDescriptorSetLayout(V.dev, &ci, NULL, &V.dsl));
    VkDescriptorSetLayout two[2] = {V.dsl, V.dsl};
    VkPipelineLayoutCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2, .pSetLayouts = two, .pushConstantRangeCount = 1, .pPushConstantRanges = &vk_gfx_pc};
    VK(vkCreatePipelineLayout(V.dev, &pci, NULL, &V.gfx_layout));
    pci.setLayoutCount = 1; pci.pushConstantRangeCount = 0; pci.pPushConstantRanges = NULL;
    VK(vkCreatePipelineLayout(V.dev, &pci, NULL, &V.cs_layout));
    VkPipelineLayoutCreateInfo bci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &vk_gfx_pc};
    VK(vkCreatePipelineLayout(V.dev, &bci, NULL, &g_builtin_layout));
    (void)g_builtin_dsl_unused;
}
static void slots_init(void) {
    for (int i = 0; i < NSLOT; i++) {
        slot_t *s = &g_slots[i];
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = V.qfam};
        VK(vkCreateCommandPool(V.dev, &pci, NULL, &s->pool));
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = s->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VK(vkAllocateCommandBuffers(V.dev, &ai, &s->cb)); VK(vkAllocateCommandBuffers(V.dev, &ai, &s->up));
        s->dp[0] = dpool_new(); s->ndp = 1;
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = NQUERY};
        VK(vkCreateQueryPool(V.dev, &qi, NULL, &s->qpool));
        vkResetQueryPool(V.dev, s->qpool, 0, NQUERY);
        VkQueryPoolCreateInfo ti = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 32};
        VK(vkCreateQueryPool(V.dev, &ti, NULL, &s->tsq));
        vkResetQueryPool(V.dev, s->tsq, 0, 32);
    }
}
/* GPU time per game frame (nvn_mtl.m gpu_tag_begin / gpu_tag_done): each submission's two timestamps, summed for the frame
   it was submitted in, for the timing lines (mtl_gpu_ms) */
#define GRING 64
static struct { _Atomic uint64_t frame, ns, t0, t1; _Atomic int pending; } g_gpu[GRING];
typedef struct { uint64_t frame; VkQueryPool q; int nseg; } gpu_ts;
static void gpu_tag_begin(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) { atomic_store(&g_gpu[k].frame, f); atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].t0, UINT64_MAX); atomic_store(&g_gpu[k].t1, 0); }
    atomic_fetch_add(&g_gpu[k].pending, 1);
}
static void gpu_tag_done(void *arg) {   /* completion thread: the slot is not reused before this ran */
    gpu_ts *g = arg; uint64_t ts[32] = {0}; unsigned k = g->frame % GRING; uint32_t nq = (uint32_t)g->nseg * 2;
    if (vkGetQueryPoolResults(V.dev, g->q, 0, nq, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[nq - 1] > ts[0] && atomic_load(&g_gpu[k].frame) == g->frame) {
        uint64_t busy = 0; for (uint32_t i = 0; i + 1 < nq; i += 2) if (ts[i + 1] > ts[i]) busy += ts[i + 1] - ts[i];   /* the segments' own spans */
        ts[1] = ts[nq - 1];
        uint64_t d = (uint64_t)((double)busy * V.props.limits.timestampPeriod), v;
        atomic_fetch_add(&g_gpu[k].ns, d);
        for (v = atomic_load(&g_gpu[k].t0); ts[0] < v && !atomic_compare_exchange_weak(&g_gpu[k].t0, &v, ts[0]);) {}
        for (v = atomic_load(&g_gpu[k].t1); ts[1] > v && !atomic_compare_exchange_weak(&g_gpu[k].t1, &v, ts[1]);) {}
    }
    if (atomic_load(&g_gpu[k].frame) == g->frame) atomic_fetch_sub(&g_gpu[k].pending, 1);
    free(g);
}
double mtl_gpu_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return -1;
    return atomic_load(&g_gpu[k].ns) / 1e6;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return 0;
    *t0 = atomic_load(&g_gpu[k].t0); *t1 = atomic_load(&g_gpu[k].t1);
    return *t1 >= *t0 && *t1;
}

/* ---------------- static buffers: zeros, support buffer, quad indices ---------------- */
typedef struct { VkBuffer buf; VmaAllocation a; uint8_t *cpu; VkDeviceSize size; } sbuf;
static sbuf g_zero, g_support, g_quad, g_zero_ssbo;
static sbuf sbuf_new(VkDeviceSize size, VkBufferUsageFlags usage) {
    sbuf b = {0}; b.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
    VmaAllocationInfo info; VK(vmaCreateBuffer(V.vma, &bi, &ai, &b.buf, &b.a, &info)); b.cpu = info.pMappedData; memset(b.cpu, 0, size);
    return b;
}
static void static_buffers_init(void) {
    VkBufferUsageFlags all = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    g_zero = sbuf_new(65536, all);
    g_zero_ssbo = sbuf_new(4u << 20, all);   /* unresolved storage buffers read 0 (and absorb writes) */
    g_support = sbuf_new(65536, all);
    for (int i = 0; i < 73; i++) *(float *)(g_support.cpu + 192 + 16 * i) = 1.0f;   /* render_scale[i].x (Ryujinx SupportBuffer) */
    g_quad = sbuf_new(65536 * 24, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    uint32_t *ix = (uint32_t *)g_quad.cpu;
    for (uint32_t q = 0; q < 65536; q++) { uint32_t b = q * 4; uint32_t v[6] = {b, b + 1, b + 2, b, b + 2, b + 3}; memcpy(ix + q * 6, v, 24); }
}

/* ---------------- built-in shaders (masked clears) ---------------- */
const VkPushConstantRange vk_gfx_pc = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
static VkShaderEXT g_clear_vs, g_clear_fs[3][8];
static void builtins_init(void) {
    VkShaderCreateInfoEXT ci = {VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = VK_SHADER_STAGE_VERTEX_BIT, .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = sizeof spv_clear_vs, .pCode = spv_clear_vs, .pName = "main", .pushConstantRangeCount = 1, .pPushConstantRanges = &vk_gfx_pc};
    VK(V.CreateShadersEXT(V.dev, 1, &ci, NULL, &g_clear_vs));
    for (int t = 0; t < 3; t++) for (int l = 0; l < 8; l++) {
        ci.stage = VK_SHADER_STAGE_FRAGMENT_BIT; ci.nextStage = 0; ci.codeSize = spv_clear_fs_size[t][l]; ci.pCode = spv_clear_fs[t][l];
        VK(V.CreateShadersEXT(V.dev, 1, &ci, NULL, &g_clear_fs[t][l]));
    }
}

/* ---------------- init ---------------- */
static int g_inited;
void mtl_init(void) {
    if (g_inited) return;
    g_inited = 1;
    vk_device_init();
    vk_pool_init();
    layouts_init();
    slots_init();
    static_buffers_init();
    builtins_init();
    vk_tex_init();
    vk_shader_init();
    vk_present_init();
}

/* ---------------- state tracking of the command buffer ---------------- */
typedef struct {
    int valid;
    VkShaderEXT vs, fs, cs;
    VkViewport vp; VkRect2D sc;
    VkCullModeFlags cull; VkFrontFace front; VkPolygonMode poly;
    int dbias_en; float dbias[3]; int dclamp;
    int dt, dw; VkCompareOp dop; int st; VkStencilOp sop[2][3]; VkCompareOp scmp[2]; uint32_t smask[2], swmask[2], sref[2];
    VkPrimitiveTopology topo; int restart;
    VkSampleCountFlagBits samples; int a2c;
    int logic_en; VkLogicOp logic;
    uint32_t nblend; VkBool32 ben[8]; VkColorBlendEquationEXT beq[8]; VkColorComponentFlags wmask[8];
    float bconst[4];
    uint32_t nvb, nva; VkVertexInputBindingDescription2EXT vb[17]; VkVertexInputAttributeDescription2EXT va[32];
    VkDescriptorSet set[2]; uint64_t set_hash[2];
    VkDescriptorSet cset; uint64_t cset_hash;
    VkBuffer vbuf[17]; VkDeviceSize voff[17];
    VkBuffer ibuf; VkDeviceSize ioff; VkIndexType itype;
    uint32_t pcl; int pcl_ok;   /* the layer push constant of a layered pass, as last pushed */
    float psq[2]; int psq_ok;   /* the UI squeeze push constants (offset 8: x, y factors), as last pushed */
} cbstate;
static cbstate C;

/* ---------------- render passes ---------------- */
typedef struct {
    int open, n;
    VkImageView cview[8]; VkFormat cfmt[8]; vk_image *cimg[8]; const tex_t *ctex[8]; int cint[8];   /* cint: 0 float, 1 uint, 2 sint */
    VkImageView dview; VkFormat dfmt; vk_image *dimg; int hd, hs;
    uint32_t w, h, layers; VkSampleCountFlagBits samples;
    int noatt, draws;
    int query_open; uint32_t query;
    uint64_t serial;
    vk_blk blk[8]; int nblk;   /* colour attachments rendered through block images (vk_tex_attach_blk) */
    uint32_t clevel[8], clayer[8], cleared;   /* pass dump: each attachment's level / layer; bits 0-7 colours, 8 depth, 9 stencil cleared */
    char pdk[8][13]; int npdk;                  /* pass dump: the fragment shaders the pass ran (key prefixes) */
    uint32_t wrote, plain;   /* draw (feedback rule): attachments written in the pass (bits 0-7 colours, 8 depth, 9 stencil), and those of
                                them a draw that samples no attachment came after */
} pass_t;
static int pd_on(void);
static void pd_pass(void);
static pass_t P;
/* Layered passes (lay_decide; nvn_mtl.m g_lay): the pass renders layers base .. base + count - 1 of one level, each draw
   into layer cur through the vertex shader's layer variant. merged: a SetRenderTargets of another layer was taken in. */
static struct { int active, refuse, merged, base, count, cur; uint64_t passes, merges, refused; } g_lay;
static const uint8_t *g_exec_cur;   /* the record being replayed (the lookaheads start after it) */
static int g_counting; static uint32_t *g_qlist; static int g_nq, g_capq, g_qskipped;   /* occlusion queries since the last reset */
static void query_end(void) { if (P.query_open) { vkCmdEndQuery(g_cb, g_slots[g_cs].qpool, P.query); P.query_open = 0; } }
/* Why passes end (the frame timing line every 600 game frames): a pass ended by a copy or an update costs its
   attachments' store and the next pass's load, and the barrier after it. */
static int g_seg; static uint64_t g_seg_mark, g_seg_commits;   /* mid-submission segments (seg_split) */
/* The GPU time of a frame (mtl_gpu_ms) comes from a timestamp at the start and end of each segment of its submissions.
   Outside a render pass Honeykrisp makes each timestamp a compute command of its own (hk_query_pool.c), which the next
   render pass then waits for: only every TOTK_GPU_TIME_EVERY-th frame (default 4; 1 every frame) is timed, the others
   report none (hle_nvn.c skips them). */
static int g_ts_on;
static int gpu_time_due(uint64_t f) {
    static int every = -1; if (every < 0) { const char *e = getenv("TOTK_GPU_TIME_EVERY"); every = e && atoi(e) > 0 ? atoi(e) : 4; }
    return f % (uint64_t)every == 0;
}
static int chunk_due(void); static void seg_split(void);
static const char *g_end_why = "other";
#define RP_END(why) (g_end_why = (why), rp_end())
#define RP_FLUSH(why) (g_end_why = (why), rp_flush())
enum { NWHY = 16 };
static const char *g_why_name[NWHY]; static uint64_t g_why_n[NWHY], g_why_small, g_why_passes;
static uint64_t g_carry_merged, g_carry_landed;   /* lazy clears: taken by a later pass, carried groups landed alone (carry_resolve) */
/* GPU time of every render pass and dispatch of one game frame (a timestamp before and after each); when it is done,
   the 25 longest are logged with the frame's total and the gaps ("[passtime]"). It measures the first frame after
   TOTK_VK_PASSTIME_AT seconds (default 120; -1: never by itself) and the next frame whenever F8 is pressed
   (frontend_sdl.c, mtl_passtime_next). */
enum { PT_MAX = 1024 };
static VkQueryPool g_ptq; static int g_pt_state, g_pt_n; static uint64_t g_pt_frame;   /* state: 0 idle, 1 measuring, 2 being reported */
static struct { char what[160]; } g_pt_lab[PT_MAX / 2];
static _Atomic int g_pt_req, g_pt_reported = 1;   /* F10: measure the next frame; the last report has been written */
void mtl_passtime_next(void) { atomic_store(&g_pt_req, 1); }
static int pt_on(void) {
    static double at = -2, t0; static int auto_done;
    struct timespec ts;
    if (at == -2) { const char *e = getenv("TOTK_VK_PASSTIME_AT"); at = e ? atof(e) : 120; clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec / 1e9; }
    if (g_pt_state == 2) { if (!atomic_load(&g_pt_reported)) return 0; g_pt_state = 0; }
    uint64_t f = atomic_load(&g_cur_frame);
    if (g_pt_state == 1) return f == g_pt_frame;   /* mtl_sub_commit ends the measurement once the frame has moved on */
    int req = atomic_load(&g_pt_req), timed = 0;
    if (!auto_done && at >= 0) { clock_gettime(CLOCK_MONOTONIC, &ts); timed = ts.tv_sec + ts.tv_nsec / 1e9 - t0 >= at; }
    if (!req && !timed) return 0;
    if (timed) auto_done = 1;
    atomic_store(&g_pt_req, 0); atomic_store(&g_pt_reported, 0);
    if (!g_ptq) { VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = PT_MAX}; VK(vkCreateQueryPool(V.dev, &qi, NULL, &g_ptq)); }
    vkResetQueryPool(V.dev, g_ptq, 0, PT_MAX);
    g_pt_state = 1; g_pt_frame = f; g_pt_n = 0;
    return 1;
}
static void pt_mark(int end, const char *fmt, ...) {   /* end 0: before the work (opens a record), 1: after it */
    if (g_pt_n >= PT_MAX) return;
    if (!end) { va_list ap; va_start(ap, fmt); vsnprintf(g_pt_lab[g_pt_n / 2].what, sizeof g_pt_lab[0].what, fmt, ap); va_end(ap); }
    vkCmdWriteTimestamp2(g_cb, end ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, g_ptq, (uint32_t)g_pt_n++);
}
static void pt_report_(void);
static void pt_report(void *arg) { (void)arg; pt_report_(); atomic_store(&g_pt_reported, 1); }   /* completion thread */
static void pt_report_(void) {
    int n = g_pt_n / 2; if (!n) return;
    uint64_t *ts = calloc((size_t)g_pt_n, sizeof *ts); if (!ts) return;
    VkResult qr = vkGetQueryPoolResults(V.dev, g_ptq, 0, (uint32_t)n * 2, sizeof *ts * (size_t)n * 2, ts, 8, VK_QUERY_RESULT_64_BIT);
    if (qr != VK_SUCCESS) { VLOG("[passtime] frame %llu: the %d timestamps were not all written (%d)", (unsigned long long)g_pt_frame, n * 2, qr); free(ts); return; }
    double per = V.props.limits.timestampPeriod / 1e6, busy = 0, span = (double)(ts[n * 2 - 1] - ts[0]) * per;
    int *ord = malloc(sizeof *ord * (size_t)n); for (int i = 0; i < n; i++) { ord[i] = i; busy += (double)(ts[2 * i + 1] - ts[2 * i]) * per; }
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (ts[2 * ord[j] + 1] - ts[2 * ord[j]] > ts[2 * ord[i] + 1] - ts[2 * ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    VLOG("[passtime] frame %llu: %d passes / dispatches, %.2f ms busy in them, %.2f ms from first to last", (unsigned long long)g_pt_frame, n, busy, span);
    for (int k = 0; k < n && k < 25; k++) { int i = ord[k]; VLOG("[passtime] %6.3f ms  #%d %s", (double)(ts[2 * i + 1] - ts[2 * i]) * per, i, g_pt_lab[i].what); }
    for (int i = 0; i < n; i++)   /* all of them in order: start from the first, time */
        VLOG("[passtime] seq #%d %+8.3f %6.3f %s", i, (double)(ts[2 * i] - ts[0]) * per, (double)(ts[2 * i + 1] - ts[2 * i]) * per, g_pt_lab[i].what);
    free(ord); free(ts);
}
/* The commonest pass shapes since the last report (the "pass shapes" line, for finding passes that could be merged):
   size, colour attachments, depth, the first colour target's NVN target and how it is viewed (kind: 0 plain, 1 one layer
   of a cube / array level, 2 a layered pass, 3 the six faces of one cube level), draws (0, 1, more). */
typedef struct { uint32_t w, h, n; uint8_t nc, depth, target, kind, draws; } shape_t;
enum { NSHAPE = 128 };
static shape_t g_shape[NSHAPE]; static int g_nshape; static uint64_t g_shape_other;
static void shape_count(void) {
    shape_t s = {P.w, P.h, 0, 0, P.dimg != NULL, 255, 0, (uint8_t)(P.draws > 1 ? 2 : P.draws)};
    for (int i = 0; i < P.n && i < 8; i++) if (P.cimg[i]) s.nc++;
    const tex_t *t0 = R.rt.n > 0 ? R.rt.color[0] : NULL;
    if (t0) s.target = (uint8_t)t0->d.target;
    if (g_lay.active) s.kind = 2;
    else if (t0 && R.rt.n == 6 && R.rt.has_cview[0]) {
        int faces = 1; for (int i = 1; i < 6; i++) faces &= R.rt.color[i] && R.rt.color[i]->serial == t0->serial && R.rt.has_cview[i] && R.rt.cview[i].set_layers && R.rt.cview[i].min_layer == i;
        s.kind = faces && R.rt.cview[0].set_layers && R.rt.cview[0].min_layer == 0 ? 3 : 0;
    } else if (t0 && R.rt.has_cview[0] && R.rt.cview[0].set_layers && R.rt.cview[0].nlayers == 1 && (t0->d.target == 4 || t0->d.target == 8 || t0->d.target == 9)) s.kind = 1;
    for (int i = 0; i < g_nshape; i++) {
        shape_t *e = &g_shape[i];
        if (e->w == s.w && e->h == s.h && e->nc == s.nc && e->depth == s.depth && e->target == s.target && e->kind == s.kind && e->draws == s.draws) { e->n++; return; }
    }
    if (g_nshape < NSHAPE) { s.n = 1; g_shape[g_nshape++] = s; } else g_shape_other++;
}
static void shape_report(uint64_t frames) {
    static const char *kinds[] = {"", " one-layer", " layered", " cube-faces"};
    char line[1400]; size_t o = (size_t)snprintf(line, sizeof line, "pass shapes per frame:");
    for (int k = 0; k < 14 && o < sizeof line; k++) {
        int best = -1; for (int i = 0; i < g_nshape; i++) if (g_shape[i].n && (best < 0 || g_shape[i].n > g_shape[best].n)) best = i;
        if (best < 0) break;
        shape_t *s = &g_shape[best];
        o += (size_t)snprintf(line + o, sizeof line - o, " %.1f x %ux%u c%u%s t%d%s %s;", (double)s->n / frames, s->w, s->h, s->nc, s->depth ? "+d" : "", s->target == 255 ? -1 : s->target,
                              kinds[s->kind & 3], s->draws == 0 ? "clear" : s->draws == 1 ? "1 draw" : "draws");
        s->n = 0;
    }
    VLOG("%s other %.1f; layered passes %llu (%llu targets merged, %llu refused)", line, (double)g_shape_other / frames,
         (unsigned long long)g_lay.passes, (unsigned long long)g_lay.merges, (unsigned long long)g_lay.refused);
    g_nshape = 0; g_shape_other = 0;
}
/* The passes and dispatches of the last PL_FRAMES game frames, for F9 (vk_present.c ring_save writes them next to the
   saved pictures): what differed in a frame that flickered. Written by the replay thread, read by the present thread
   for frames already done. */
enum { PL_FRAMES = 16, PL_MAX = 320 };
typedef struct { uint16_t w, h, nc, layers; uint8_t depth, disp; uint32_t draws; uint64_t c0, ds; char fs[13]; const char *why; } pl_ent;
static struct { uint64_t frame; int n; pl_ent e[PL_MAX]; } g_pl[PL_FRAMES];
static pl_ent *pl_next(void) {
    uint64_t f = atomic_load(&g_cur_frame); unsigned k = (unsigned)(f % PL_FRAMES);
    if (g_pl[k].frame != f) { g_pl[k].frame = f; g_pl[k].n = 0; }
    return g_pl[k].n < PL_MAX ? &g_pl[k].e[g_pl[k].n++] : NULL;
}
static void pl_pass(const char *why) {
    pl_ent *e = pl_next(); if (!e) return;
    memset(e, 0, sizeof *e);
    e->w = (uint16_t)P.w; e->h = (uint16_t)P.h; e->layers = (uint16_t)P.layers; e->depth = P.dimg != NULL; e->draws = (uint32_t)P.draws; e->why = why;
    for (int i = 0; i < P.n && i < 8; i++) if (P.cimg[i]) e->nc++;
    e->c0 = R.rt.n > 0 && R.rt.color[0] ? R.rt.color[0]->serial : 0; e->ds = R.rt.depth ? R.rt.depth->serial : 0;
    if (P.npdk) memcpy(e->fs, P.pdk[0], 12);
}
void vk_pass_log_dump(FILE *f, uint64_t from, uint64_t to) {   /* frames [from, to] still in the ring */
    for (uint64_t fr = from; fr <= to; fr++) {
        unsigned k = (unsigned)(fr % PL_FRAMES); if (g_pl[k].frame != fr) continue;
        fprintf(f, "frame %llu: %d passes and dispatches\n", (unsigned long long)fr, g_pl[k].n);
        for (int i = 0; i < g_pl[k].n; i++) {
            const pl_ent *e = &g_pl[k].e[i];
            if (e->disp) fprintf(f, "  %3d dispatch %.12s %ux%ux%u\n", i, e->fs, e->w, e->h, e->draws);
            else fprintf(f, "  %3d pass %ux%u%s c%u%s rt %llu/%llu draws %u fs %.12s end %s\n", i, e->w, e->h, e->layers > 1 ? " layered" : "", e->nc, e->depth ? "+d" : "",
                         (unsigned long long)e->c0, (unsigned long long)e->ds, e->draws, e->fs[0] ? e->fs : "-", e->why ? e->why : "?");
        }
    }
}
static void why_count(void) {
    shape_count();
    int k = 0; while (k < NWHY && g_why_name[k] && g_why_name[k] != g_end_why) k++;
    if (k == NWHY) return;
    g_why_name[k] = g_end_why; g_why_n[k]++; g_why_passes++; if (P.draws <= 1) g_why_small++;
}
static void why_report(uint64_t frames) {
    char line[900]; size_t o = (size_t)snprintf(line, sizeof line, "[vk] passes per frame %.1f (%.1f with at most one draw); ended by:", (double)g_why_passes / frames, (double)g_why_small / frames);
    for (int k = 0; k < NWHY && g_why_name[k] && o < sizeof line; k++) o += (size_t)snprintf(line + o, sizeof line - o, " %s %.1f", g_why_name[k], (double)g_why_n[k] / frames);
    void vk_tex_wait_stats(uint64_t *ns, uint64_t *n); uint64_t wns, wn; vk_tex_wait_stats(&wns, &wn);
    VLOG("%s; extra segments %.1f; read-back waits %.1f a frame, %.2f ms; lazy clears %.1f taken by a later pass, %.1f landed alone", line + 5, (double)g_seg_commits / frames, (double)wn / frames, wns / 1e6 / frames,
         (double)g_carry_merged / frames, (double)g_carry_landed / frames);
    g_carry_merged = g_carry_landed = 0;
    shape_report(frames);
    memset(g_why_n, 0, sizeof g_why_n); g_why_small = g_why_passes = 0; g_seg_commits = 0;
}
static void rp_end(void) {
    if (!P.open) return;
    const char *why = g_end_why;
    pl_pass(why);
    why_count(); g_end_why = "other";
    query_end();
    vkCmdEndRendering(g_cb);
    if (g_pt_state == 1 && g_pt_n % 2) { int k = g_pt_n / 2; size_t o = strlen(g_pt_lab[k].what);
        snprintf(g_pt_lab[k].what + o, sizeof g_pt_lab[0].what - o, " draws %d fs %.12s end %s", P.draws, P.npdk ? P.pdk[0] : "-", why); pt_mark(1, NULL); }
    vk_barrier_all(g_cb);
    if (P.nblk) {   /* the rendered blocks into their compressed textures */
        for (int i = 0; i < P.nblk; i++) vk_tex_blk_copy(g_cb, &P.blk[i], 1);
        vk_barrier_all(g_cb);
        P.nblk = 0;
    }
    if (pd_on()) pd_pass();
    P.open = 0; g_lay.active = g_lay.merged = 0;
}
static void carry_land_all(void);
VkCommandBuffer vk_cmd_outside_pass(void) { RP_END("texture"); carry_land_all(); return g_cb; }   /* an upload may touch a carried clear's texture */
static int rt_same(const p_set_rt *a, const p_set_rt *b) {
    if (a->n != b->n || !a->depth != !b->depth || (a->depth && b->depth && a->depth->serial != b->depth->serial)) return 0;
    if (a->has_dview != b->has_dview || (a->has_dview && memcmp(&a->dview, &b->dview, sizeof a->dview))) return 0;
    for (int i = 0; i < a->n && i < 8; i++) {
        if (!a->color[i] != !b->color[i] || (a->color[i] && a->color[i]->serial != b->color[i]->serial)) return 0;
        if (a->has_cview[i] != b->has_cview[i] || (a->has_cview[i] && memcmp(&a->cview[i], &b->cview[i], sizeof a->cview[i]))) return 0;
    }
    return 1;
}
static int rt_none(void) {
    if (!R.has_rt || R.rt.depth) return 0;
    for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) return 0;
    return 1;
}
static void noatt_dims(uint32_t *w, uint32_t *h) {
    double x1 = 0, y1 = 0;
    if (S.has_vp) { x1 = ceil(fmax(S.vp[0][0], S.vp[0][0] + S.vp[0][2])); y1 = ceil(fmax(S.vp[0][1], S.vp[0][1] + S.vp[0][3])); }
    if ((x1 < 1 || y1 < 1) && R.has_sc) { x1 = (double)R.sc.x + R.sc.w; y1 = (double)R.sc.y + R.sc.h; }
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1; *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}
static int clear_class(int nvn_fmt) {
    vk_fmt f; if (!vk_fmt_of(nvn_fmt, &f)) return 0;
    return f.is_uint ? 1 : f.is_int ? 2 : 0;
}
static VkClearColorValue clear_value(int cls, const float rgba[4]) {
    VkClearColorValue v;
    for (int i = 0; i < 4; i++) { if (cls == 1) v.uint32[i] = rgba[i] <= 0 ? 0 : (uint32_t)rgba[i]; else if (cls == 2) v.int32[i] = (int32_t)rgba[i]; else v.float32[i] = rgba[i]; }
    return v;
}
/* The draw uses the depth / stencil attachment: a depth test that can fail, depth writes (NVN writes only with the test
   on), or a stencil test. A pass opened by a draw that does not is opened without it (nvn_mtl.m's depth lookahead):
   a full-screen pass with an idle D32S8 attachment loads and stores 16 MB at 1080p, and the game binds its depth
   buffer to most of its post passes. A later draw that needs it ends the pass (draw). TOTK_VK_DEPTH_LAZY=0: always. */
static int draw_uses_ds(void) { return (S.ds.dt && (S.ds.df != NVN_DEPTH_FUNC_ALWAYS || S.ds.dw)) || S.ds.st; }
static int depth_lazy(void) { static int v = -1; if (v < 0) { const char *e = getenv("TOTK_VK_DEPTH_LAZY"); v = !(e && !strcmp(e, "0")); } return v; }
/* Open the render pass of R.rt; clear_only: only the attachments with pending clears (rp_flush). */
/* Key 4 / TOTK_ASPECT (runtime/game_patches_botw.c): the camera sees a view of the window's shape (or 21:9) squeezed into
   the game's 16:9 canvas and the present stretches the canvas over the window (vk_present.c), so the UI drawn into the
   canvas is squeezed back about its centre to keep its shape in the middle 16:9, as nvn_mtl.m uw_ui_squeeze. The canvas
   is learned as there (canvas_learn): the last 16:9-shaped RGB10A2 or float target, 640 or more wide, without depth,
   before a pass into a presented texture. Every Breath of the Wild draw into it is a 6-index UI quad except the scene's
   upscale (a full-screen triangle) and the menus' blurred picture of the scene (uw_scene_picture), which keep the full
   width (nvn_obj.h UW_NOT_UI_DRAW). TOTK_UI_SQUEEZE=0 squeezes nothing. */
extern float runtime_aspect_now(void) __attribute__((weak));
static double uw_ratio(void) { float a = runtime_aspect_now ? runtime_aspect_now() : 0; return a > 0 ? a : 0; }
static struct { uint64_t swap[4]; int nswap; uint32_t sw, sh; uint64_t canvas, last_c0, cand; int cand_n, odd, scene, edged; } g_uw;
void vk_uw_note_present(const tex_t *t) {
    if (!t || !t->serial) return;
    int known = 0; for (int i = 0; i < g_uw.nswap; i++) known |= g_uw.swap[i] == t->serial;
    if (!known) g_uw.swap[g_uw.nswap < 4 ? g_uw.nswap++ : (int)(t->serial % 4)] = t->serial;
    g_uw.sw = t->d.width; g_uw.sh = t->d.height;
}
static void uw_canvas_learn(void) {
    const tex_t *t0 = R.rt.n > 0 ? R.rt.color[0] : NULL;
    if (!t0 || !g_uw.sw || !g_uw.sh) return;
    int to_swap = 0; for (int i = 0; i < g_uw.nswap; i++) to_swap |= g_uw.swap[i] == t0->serial;
    if (to_swap) {
        g_uw.scene = g_uw.edged = 0;   /* the next frame's canvas */
        /* a new canvas wins after UW_SETTLE passes into presented textures in a row: the game now and then draws another
           16:9 target last (a picture for a menu or the album), and taking it for one frame left the HUD unsqueezed and
           squeezed that target's quads instead, a flash every 15 to 30 s */
        enum { UW_SETTLE = 12 };
        uint64_t c = g_uw.last_c0;
        if (!c || c == g_uw.canvas) { g_uw.cand_n = 0; return; }
        if (c != g_uw.cand) { g_uw.cand = c; g_uw.cand_n = 0; }
        if (g_uw.canvas && ++g_uw.cand_n < UW_SETTLE) { if (g_uw.cand_n == 1) g_uw.odd++; return; }
        g_uw.canvas = c; g_uw.cand_n = 0;
        VLOG("key 4: canvas %llu, copied to presented texture %llu (%d one-off targets passed over)", (unsigned long long)c, (unsigned long long)t0->serial, g_uw.odd);
        return;
    }
    int level = R.rt.has_cview[0] && R.rt.cview[0].set_levels ? R.rt.cview[0].base_level : 0;
    vk_fmt f; if (!vk_fmt_of(t0->d.format, &f)) return;
    int fmt_ok = f.vk == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f.vk == VK_FORMAT_A2R10G10B10_UNORM_PACK32 || f.vk == VK_FORMAT_R16G16B16A16_SFLOAT ||
                 f.vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f.vk == VK_FORMAT_R32G32B32A32_SFLOAT;
    if (t0->d.width >= 640 && fabs((double)t0->d.width * g_uw.sh / ((double)t0->d.height * g_uw.sw) - 1) < 0.01 && !level && fmt_ok && !R.rt.depth)
        g_uw.last_c0 = t0->serial;
}
static int uw_on_canvas(void) { return g_uw.canvas && R.rt.n > 0 && R.rt.color[0] && R.rt.color[0]->serial == g_uw.canvas; }
/* the quad draws a picture of the scene: its fragment texture 0 was rendered by the GPU at about 16:9, 320 or more wide */
static int uw_scene_picture(void) {
    const tex_t *t = vk_tex_slot((int)(S.tex[1][0] & 0xfffff));
    return t && t->d.width >= 320 && t->d.height && fabs((double)t->d.width / t->d.height * 9 / 16 - 1) < 0.02 && vk_tex_owned(t->serial);
}
static double g_sq_test[2]; static int g_sq_test_edges = -1;   /* the smoke test's squeeze of every 6-index draw (vk_ui_squeeze_test) */
void vk_ui_squeeze_test(double ux, double uy, int edges) { g_sq_test[0] = ux; g_sq_test[1] = uy; g_sq_test_edges = edges; }
static void uw_ui_squeeze(const p_draw *dw, const p_draw_idx *di, double *ux, double *uy) {
    *ux = *uy = 1;
    if (g_sq_test[0] > 0) { if ((dw ? dw->count : di ? di->count : 0) == 6) { *ux = g_sq_test[0]; *uy = g_sq_test[1]; } return; }
    static int off = -1; if (off < 0) { const char *e = getenv("TOTK_UI_SQUEEZE"); off = e && !strcmp(e, "0"); }
    double a = uw_ratio();
    if (off || !a || !uw_on_canvas() || (dw ? dw->count : di ? di->count : 0) != 6 || uw_scene_picture()) return;
    totk_ui_squeeze(a, ux, uy);
}

/* ---------------- lookahead over the submission's records (nvn_mtl.m fz_cursor) ---------------- */
static const uint64_t *g_rest; static int g_nrest;          /* the command handles of the submission after the one being replayed */
static const uint8_t *g_ret_stack[16]; static int g_ret_n;  /* mtl_exec's OP_CALL return points, innermost last */
void mtl_exec_rest(const uint64_t *h, int n) { g_rest = h; g_nrest = n; }
typedef struct { const uint8_t *p, *ret[16]; int ret_n, rest_i, deep; } op_cursor;
/* at the record after the one being replayed; it returns from called lists as mtl_exec will */
static void cursor_after_cur(op_cursor *c) {
    memset(c, 0, sizeof *c);
    if (!g_exec_cur) return;
    op_hdr h; memcpy(&h, g_exec_cur, sizeof h);
    c->p = g_exec_cur + h.size;
    if (g_ret_n > 16) { c->deep = 1; return; }   /* lists called deeper than the stack keeps: the scan ends with this one */
    memcpy(c->ret, g_ret_stack, sizeof *c->ret * (size_t)g_ret_n); c->ret_n = g_ret_n;
}
static int op_next(op_cursor *c, op_hdr *oh, const void **oa) {
    for (int guard = 0; guard < 1000000 && c->p; guard++) {
        op_hdr h; memcpy(&h, c->p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_) return 0;
        const void *a = c->p + 8;
        if (h.op == OP_END) {
            if (c->ret_n) c->p = c->ret[--c->ret_n];
            else if (!c->deep && c->rest_i < g_nrest) c->p = (const uint8_t *)(uintptr_t)g_rest[c->rest_i++];
            else return 0;
            continue;
        }
        if (h.op == OP_JUMP) { memcpy(&c->p, a, 8); continue; }
        if (h.op == OP_CALL) { uint64_t t; memcpy(&t, a, 8); if (c->ret_n >= 16) return 0; c->ret[c->ret_n++] = c->p + h.size; c->p = (const uint8_t *)(uintptr_t)t; continue; }
        c->p += h.size; *oh = h; *oa = a;
        return 1;
    }
    return 0;
}

/* ---------------- layered passes (nvn_mtl.m lay_*, TOTK_LAYER_MERGE) ---------------- */
/* The game renders each layer of a cube / array level in a pass of its own (TOTK's cube-map filter chain: six one-quad
   passes per level, 132 in heavy frames), and every pass is a render command of its own for the GPU, with a fixed cost
   (~50 us each on Apple GPUs, nvn_mtl.m). When a pass opens on one layer of such a level (one colour target, no depth) and
   the records ahead set the targets to further layers of the same level and view with only draws, clears and state in
   between (lay_scan), the pass covers that layer range and continues across those SetRenderTargets (lay_continue); each
   draw writes its layer through its vertex shader's layer variant (vk_shader_layered: the layer in a push constant).
   Layers of the range the game does not draw are loaded and stored unchanged. A clear after a taken SetRenderTargets, a
   draw that samples the attachment there or cannot write its layer end the pass as before (and the next one opens plain).
   TOTK_LAYER_MERGE=0 turns it off. */
enum { LAY_MAX = 256, LAY_MAX_EXTENT = 2048 };
static int lay_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("TOTK_LAYER_MERGE"); v = !(e && !strcmp(e, "0")); if (!v) VLOG("TOTK_LAYER_MERGE=0: layered passes off"); } return v; }
/* one colour target, no depth, a view of one layer: *layer = that layer */
static int lay_single(const p_set_rt *r, int *layer) {
    if (r->n != 1 || !r->color[0] || r->depth || !r->has_cview[0] || !r->cview[0].set_layers || r->cview[0].nlayers != 1) return 0;
    *layer = r->cview[0].min_layer; return 1;
}
/* the same texture, and the same view apart from the layer */
static int lay_same_group(const p_set_rt *a, const p_set_rt *b) {
    view_t v = b->cview[0]; v.min_layer = a->cview[0].min_layer;
    return a->color[0]->serial == b->color[0]->serial && !memcmp(&v, &a->cview[0], sizeof v);
}
/* the highest layer in [first, nlayers) a later SetRenderTargets of the group selects before a record at which the replay
   ends the pass anyway; it only sizes the attachment (lay_continue and the replay's own checks decide each step) */
static int lay_scan(const p_set_rt *rt, int first, int nlayers) {
    op_cursor c; cursor_after_cur(&c); int hi = first;
    for (int guard = 0; guard < 4096; guard++) {
        op_hdr h; const void *a;
        if (!op_next(&c, &h, &a)) break;
        switch (h.op) {
            case OP_SET_RT: { int l;
                if (!lay_single(a, &l) || !lay_same_group(rt, a) || l < first || l >= nlayers || l - first >= LAY_MAX) return hi;
                if (l > hi) hi = l;
                continue; }
            case OP_BEGIN: case OP_UPDATE_UB: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_COND: case OP_RENDER_ENABLE: case OP_CLEAR_COLOR:
            case OP_DRAW: case OP_DRAW_IDX: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_BIND_PROGRAM: case OP_BIND_STATE:
            case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP:
            case OP_BIND_IMG: case OP_STENCIL: case OP_POLY_OFFSET: case OP_BLEND_COLOR: case OP_COPY_STRIDE: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD:
            case OP_BARRIER: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: continue;
            default: return hi;   /* copies, dispatches, texture and depth / stencil clears, events */
        }
    }
    return hi;
}
static int lay_draw_ok(vk_shader *vs) { return vs && vk_shader_layered(vs) != VK_NULL_HANDLE; }
static vk_shader *g_fd_vs;   /* the vertex shader of the draw opening a pass (lay_decide), else NULL */
/* levels whose layered pass ended at a draw that samples the texture (a filter reading the level above, say): their next
   passes stay plain instead of loading and storing a layer range for one layer each time */
static struct { uint64_t serial; int level; } g_lay_bad[16]; static unsigned g_lay_nbad;
static int lay_level(const p_set_rt *r) { return r->cview[0].set_levels ? r->cview[0].base_level : 0; }
static void lay_refuse_level(void) {
    if (!R.rt.color[0]) return;
    g_lay_bad[g_lay_nbad++ % 16] = (typeof(g_lay_bad[0])){R.rt.color[0]->serial, lay_level(&R.rt)};
}
static int lay_refused_level(const p_set_rt *r) {
    for (unsigned i = 0; i < 16 && i < g_lay_nbad; i++) if (g_lay_bad[i].serial == r->color[0]->serial && g_lay_bad[i].level == lay_level(r)) return 1;
    return 0;
}
/* rp_begin: the number of layers of a layered pass from *first, or 0 for a plain one. Only for a pass opened by a draw or
   a colour clear being replayed, with no clear pending (a load clear would clear every layer of the range). */
static int lay_decide(int clear_only, int *first) {
    if (g_lay.refuse) { g_lay.refuse = 0; return 0; }
    if (clear_only || !lay_on() || !g_exec_cur || R.clear_c[0] || R.clear_d || R.clear_s || !lay_single(&R.rt, first) || lay_refused_level(&R.rt)) return 0;
    op_hdr h; memcpy(&h, g_exec_cur, sizeof h);
    if (h.op != OP_DRAW && h.op != OP_DRAW_IDX && h.op != OP_CLEAR_COLOR) return 0;
    const tex_t *t = R.rt.color[0];
    if ((t->d.target != 8 && t->d.target != 4) || t->d.samples > 1) return 0;   /* cube, 2D array */
    int lv = R.rt.cview[0].set_levels ? R.rt.cview[0].base_level : 0, levels = t->d.levels ? (int)t->d.levels : 1;
    int layers = t->d.target == 8 ? 6 : (int)(t->d.depth ? t->d.depth : 1);
    uint32_t w = t->d.width >> lv ? t->d.width >> lv : 1, hh = t->d.height >> lv ? t->d.height >> lv : 1;
    if (lv >= levels || w > LAY_MAX_EXTENT || hh > LAY_MAX_EXTENT || *first >= layers) return 0;
    vk_fmt f, tf;   /* no block-compressed texture (rendered through a block image of one layer, vk_tex_attach_blk) */
    if (!vk_fmt_of(R.rt.cview[0].set_format ? R.rt.cview[0].format : t->d.format, &f) || f.compressed || !vk_fmt_of(t->d.format, &tf) || tf.compressed) return 0;
    if (h.op != OP_CLEAR_COLOR && !lay_draw_ok(g_fd_vs)) return 0;
    int hi = lay_scan(&R.rt, *first, layers);
    return hi > *first ? hi - *first + 1 : 0;
}
/* SetRenderTargets in a layered pass: another layer of its range continues the pass */
static int lay_continue(const p_set_rt *r) {
    int l, c;
    if (!lay_single(r, &l) || !lay_single(&R.rt, &c) || !lay_same_group(&R.rt, r) || l < g_lay.base || l >= g_lay.base + g_lay.count) return 0;
    if (!g_lay.merges++) VLOG("layered pass: first SetRenderTargets taken in (layer %d of %d-%d)", l, g_lay.base, g_lay.base + g_lay.count - 1);
    g_lay.cur = l; R.rt = *r; g_lay.merged = 1;
    return 1;
}
static void carry_resolve(void);
static int rp_begin(int clear_only) {
    if (P.open) return 1;
    if (!R.has_rt) { VK_ONCE(nort, "draw without render targets set"); return 0; }
    if (!clear_only) carry_resolve();   /* lazy clears: carried clears of its attachments become its load clears */
    if (uw_ratio()) uw_canvas_learn();   /* key 4: the canvas the UI is squeezed in */
    memset(&P, 0, sizeof P);
    VkRenderingAttachmentInfo ca[8]; memset(ca, 0, sizeof ca);
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    /* a clear of a duplicate slot goes to the slot holding the same image (Breath of the Wild's G-buffer, rt_dup_of) */
    for (int i = 1; i < R.rt.n && i < 8; i++) { int j = rt_dup_of(&R.rt, i); if (j >= 0 && R.clear_c[i]) { if (!R.clear_c[j]) { R.clear_c[j] = 1; memcpy(R.cc[j], R.cc[i], 16); } R.clear_c[i] = 0; } }
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX; VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT; int any = 0;
    int n = R.rt.n > 8 ? 8 : R.rt.n;
    char why[192]; size_t wl = 0; why[0] = 0;   /* the bound targets that could not be used, for the log */
    int lay_first = 0, lay_n = lay_decide(clear_only, &lay_first);   /* a layered pass: colour 0 over lay_n layers from lay_first */
    view_t lay_view; if (lay_n) { lay_view = R.rt.cview[0]; lay_view.min_layer = (uint16_t)lay_first; lay_view.nlayers = (uint16_t)lay_n; }
    for (int i = 0; i < n; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        if (!R.rt.color[i] || (clear_only && !R.clear_c[i]) || rt_dup_of(&R.rt, i) >= 0) continue;
        VkImageView v; VkFormat f; uint32_t aw, ah, al;
        vk_image *im = vk_tex_attach(R.rt.color[i], lay_n && !i ? &lay_view : R.rt.has_cview[i] ? &R.rt.cview[i] : NULL, &v, &f, &aw, &ah, &al);
        if (lay_n && !i && im) lay_n = al > 1 ? (int)al : 0;   /* the image may hold fewer layers than the NVN texture says */
        if (!im && R.rt.has_cview[i] && (im = vk_tex_attach_blk(R.rt.color[i], &R.rt.cview[i], &v, &f, &aw, &ah, &al, &P.blk[P.nblk]))) {
            if (!R.clear_c[i]) { vk_barrier_all(g_cb); vk_tex_blk_copy(g_cb, &P.blk[P.nblk], 0); vk_barrier_all(g_cb); }   /* the pass loads: the current blocks */
            P.nblk++;
        }
        if (!im || im->depth || im->stencil) {
            if (wl < sizeof why) wl += (size_t)snprintf(why + wl, sizeof why - wl, " color %d format %#x target %d: %s", i, R.rt.color[i]->d.format, R.rt.color[i]->d.target, im ? "a depth image" : "no image or view");
            continue;
        }
        P.cview[i] = v; P.cfmt[i] = f; P.cimg[i] = im; P.ctex[i] = R.rt.color[i];
        P.clevel[i] = R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0;
        P.clayer[i] = R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0;
        if (R.clear_c[i]) P.cleared |= 1u << i;
        P.cint[i] = clear_class(R.rt.has_cview[i] && R.rt.cview[i].set_format ? R.rt.cview[i].format : R.rt.color[i]->d.format);
        ca[i].imageView = v; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = R.clear_c[i] ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca[i].clearValue.color = clear_value(P.cint[i], R.cc[i]);
        if (aw < w) w = aw; if (ah < h) h = ah; if (al < layers) layers = al;
        samples = (VkSampleCountFlagBits)im->samples;
        vk_tex_mark_owned(R.rt.color[i]); vk_tex_note_gpu_write(R.rt.color[i]);
        any = 1;
    }
    P.n = n;
    int want_ds = clear_only || R.clear_d || R.clear_s || !any || !depth_lazy() || draw_uses_ds();
    if (R.rt.depth && !(clear_only && !R.clear_d && !R.clear_s) && want_ds) {
        VkImageView v; VkFormat f; uint32_t aw, ah, al;
        vk_image *im = vk_tex_attach(R.rt.depth, R.rt.has_dview ? &R.rt.dview : NULL, &v, &f, &aw, &ah, &al);
        if (im && (im->depth || im->stencil)) {
            P.dview = v; P.dfmt = im->format; P.dimg = im; P.hd = im->depth; P.hs = im->stencil;
            P.cleared |= (R.clear_d && im->depth ? 1u << 8 : 0) | (R.clear_s && im->stencil ? 1u << 9 : 0);
            if (P.hd) { da.imageView = v; da.imageLayout = VK_IMAGE_LAYOUT_GENERAL; da.loadOp = R.clear_d ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        da.clearValue.depthStencil.depth = (float)(im->format == VK_FORMAT_D16_UNORM ? nvn_depth16_clear_value(R.cd) : R.cd); }
            if (P.hs) { sa.imageView = v; sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL; sa.loadOp = R.clear_s ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        sa.clearValue.depthStencil.stencil = R.cs; }
            if (aw < w) w = aw; if (ah < h) h = ah; if (al < layers) layers = al;
            samples = (VkSampleCountFlagBits)im->samples;
            vk_tex_mark_owned(R.rt.depth);
            any = 1;
        } else if (wl < sizeof why) wl += (size_t)snprintf(why + wl, sizeof why - wl, " depth format %#x: %s", R.rt.depth->d.format, im ? "not a depth image" : "no image or view");
    }
    if (!any && !clear_only && rt_none()) { noatt_dims(&w, &h); layers = 1; P.noatt = 1; any = 1; }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    if (!any) {
        static unsigned skipped; if (skipped++ < 6) VLOG("render pass without usable attachments (%s): its draws are skipped", why[0] ? why + 1 : "none bound");
        return 0;
    }
    P.w = w; P.h = h; P.layers = layers ? layers : 1; P.samples = samples;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = P.layers,
        .colorAttachmentCount = (uint32_t)n, .pColorAttachments = ca, .pDepthAttachment = P.hd ? &da : NULL, .pStencilAttachment = P.hs ? &sa : NULL};
    if (pt_on()) pt_mark(0, "pass %ux%u c%d%s%s rt %llu/%llu", w, h, n, P.hd ? "+d" : "", lay_n ? " layered" : "", R.rt.n > 0 && R.rt.color[0] ? (unsigned long long)R.rt.color[0]->serial : 0ull,
                         R.rt.depth ? (unsigned long long)R.rt.depth->serial : 0ull);
    vkCmdBeginRendering(g_cb, &ri);
    P.open = 1; P.serial = ++g_stat_pass;
    P.wrote = P.cleared & 0x3ffu;
    C.nblend = UINT32_MAX;   /* blend arrays are per attachment count */
    C.samples = 0;
    if (lay_n > 1 && P.cimg[0] && !P.nblk && P.layers == (uint32_t)lay_n) {
        g_lay.active = 1; g_lay.merged = 0; g_lay.base = g_lay.cur = lay_first; g_lay.count = lay_n;
        if (!g_lay.passes++) VLOG("layered pass: the first covers layers %d-%d of a %ux%u level (NVN target %d)", lay_first, lay_first + lay_n - 1, w, h, R.rt.color[0]->d.target);
    }
    return 1;
}
/* Lazy clears (nxnative DM_NVN_LAZY_CLEAR; TOTK_VK_LAZY_CLEAR=0: off). Clears with no draw after them used to open a pass
   of their own when the game set other targets: Breath of the Wild clears the scene targets that way, four clear-only
   passes before the G-buffer pass, each a full pass of fixed GPU cost on this GPU. The pending clears of the old targets
   are now carried (g_carry, with the targets they were made on): the next pass that attaches the same texture level and
   layer through the same view clears it as its load action (carry_resolve, at rp_begin), and the rest land in passes of
   their own before anything could see them: before that pass opens, a draw sampling one, a copy, texture clear, dispatch
   or upload touching one, an event, and the end of the submission (carry_land_all). A clear made later on the same
   texture, level and layer replaces a carried one. */
#define CARRY_N 8
typedef struct { p_set_rt rt; uint8_t clear_c[8]; float cc[8][4]; uint8_t clear_d, clear_s; float cd; uint32_t cs; } carry_t;
static carry_t g_carry[CARRY_N]; static int g_ncarry;
static int lazy_clear_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("TOTK_VK_LAZY_CLEAR"); v = !(e && !strcmp(e, "0")); } return v; }
static int carry_same_color(const p_set_rt *a, int i, const p_set_rt *b, int j) {
    return a->color[i] && b->color[j] && a->color[i]->serial == b->color[j]->serial && a->has_cview[i] == b->has_cview[j] &&
           (!a->has_cview[i] || !memcmp(&a->cview[i], &b->cview[j], sizeof a->cview[i]));
}
static int carry_same_depth(const p_set_rt *a, const p_set_rt *b) {
    return a->depth && b->depth && a->depth->serial == b->depth->serial && a->has_dview == b->has_dview && (!a->has_dview || !memcmp(&a->dview, &b->dview, sizeof a->dview));
}
static int carry_left(const carry_t *g) { int n = g->clear_d || g->clear_s; for (int i = 0; i < 8; i++) n |= g->clear_c[i]; return n; }
/* the group's clears in a pass of their own, as the flush at SetRenderTargets did */
static void carry_land(const carry_t *g) {
    if (P.open) RP_END("lazy_clear");   /* not reached: carried clears exist only between passes */
    vk_rt_state save = R;
    R.rt = g->rt; R.has_rt = 1;
    memcpy(R.clear_c, g->clear_c, sizeof R.clear_c); memcpy(R.cc, g->cc, sizeof R.cc);
    R.clear_d = g->clear_d; R.clear_s = g->clear_s; R.cd = g->cd; R.cs = g->cs;
    if (g_cb && rp_begin(1)) { g_end_why = "lazy_clear"; rp_end(); }
    R = save; g_carry_landed++;
}
static void carry_land_all(void) {
    int n = g_ncarry; g_ncarry = 0;   /* first: rp_begin(1) must not resolve them again */
    for (int k = 0; k < n; k++) if (carry_left(&g_carry[k])) carry_land(&g_carry[k]);
}
/* SetRenderTargets with clears pending and no pass open: carry them */
static void carry_push(void) {
    if (g_ncarry == CARRY_N) { int n = g_ncarry; g_ncarry = 0; if (carry_left(&g_carry[0])) carry_land(&g_carry[0]); memmove(g_carry, g_carry + 1, sizeof *g_carry * (size_t)(n - 1)); g_ncarry = n - 1; }
    carry_t *g = &g_carry[g_ncarry];
    g->rt = R.rt; memcpy(g->clear_c, R.clear_c, sizeof g->clear_c); memcpy(g->cc, R.cc, sizeof g->cc);
    g->clear_d = R.clear_d; g->clear_s = R.clear_s; g->cd = R.cd; g->cs = R.cs;
    for (int k = 0; k < g_ncarry; k++) {   /* older clears of the same texture, level and layer are replaced */
        carry_t *o = &g_carry[k];
        for (int i = 0; i < 8; i++) for (int j = 0; j < 8 && o->clear_c[i]; j++) if (g->clear_c[j] && carry_same_color(&o->rt, i, &g->rt, j)) o->clear_c[i] = 0;
        if (carry_same_depth(&o->rt, &g->rt)) { if (g->clear_d) o->clear_d = 0; if (g->clear_s) o->clear_s = 0; }
    }
    g_ncarry++;
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
}
/* rp_begin of a pass: the carried clears on its attachments become its load clears (unless it has newer clears of its
   own there), the others land first */
static void carry_resolve(void) {
    if (!g_ncarry) return;
    for (int k = g_ncarry - 1; k >= 0; k--) {   /* newest first: an older clear of the same attachment is replaced anyway */
        carry_t *g = &g_carry[k];
        for (int i = 0; i < 8; i++) {
            if (!g->clear_c[i]) continue;
            for (int j = 0; j < R.rt.n && j < 8; j++) {
                if (rt_dup_of(&R.rt, j) >= 0 || !carry_same_color(&g->rt, i, &R.rt, j)) continue;
                if (!R.clear_c[j]) { R.clear_c[j] = 1; memcpy(R.cc[j], g->cc[i], 16); g_carry_merged++; }
                g->clear_c[i] = 0; break;
            }
        }
        if ((g->clear_d || g->clear_s) && carry_same_depth(&g->rt, &R.rt)) {
            if (g->clear_d) { if (!R.clear_d) { R.clear_d = 1; R.cd = g->cd; g_carry_merged++; } g->clear_d = 0; }
            if (g->clear_s) { if (!R.clear_s) { R.clear_s = 1; R.cs = g->cs; g_carry_merged++; } g->clear_s = 0; }
        }
    }
    carry_land_all();
}
/* whether a carried clear is on an image */
static int carry_on(const vk_image *im) {
    for (int k = 0; im && k < g_ncarry; k++) {
        const carry_t *g = &g_carry[k];
        for (int i = 0; i < g->rt.n && i < 8; i++) if (g->clear_c[i] && g->rt.color[i] && vk_tex_image(g->rt.color[i]) == im) return 1;
        if ((g->clear_d || g->clear_s) && g->rt.depth && vk_tex_image(g->rt.depth) == im) return 1;
    }
    return 0;
}
/* whether an image has a clear pending for the next pass (R) or carried */
static int clear_pending_on(const vk_image *im) {
    if (!im) return 0;
    if (R.has_rt) {
        for (int i = 0; i < R.rt.n && i < 8; i++) if (R.clear_c[i] && R.rt.color[i] && vk_tex_image(R.rt.color[i]) == im) return 1;
        if ((R.clear_d || R.clear_s) && R.rt.depth && vk_tex_image(R.rt.depth) == im) return 1;
    }
    return carry_on(im);
}
static int any_clear_pending(void) {
    if (g_ncarry) return 1;
    int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
    return pend;
}
static void rp_flush(void) {
    carry_land_all();   /* older than the pending clears */
    int pend = R.clear_d || R.clear_s;
    for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
    if (pend && !P.open && g_cb) rp_begin(1);
    rp_end();
}
static int scissor_full(const tex_t *t, const view_t *v, int has_v) {
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    uint32_t w = t->d.width >> level ? t->d.width >> level : 1, h = (t->d.target == 0 || t->d.target == 3) ? 1 : t->d.height >> level ? t->d.height >> level : 1;
    return !R.has_sc || (R.sc.x <= 0 && R.sc.y <= 0 && (int64_t)R.sc.x + R.sc.w >= (int64_t)w && (int64_t)R.sc.y + R.sc.h >= (int64_t)h);
}
static int clamp_scissor(VkRect2D *r) {
    int64_t x0 = 0, y0 = 0, x1 = P.w, y1 = P.h;
    if (R.has_sc) { if (R.sc.x > x0) x0 = R.sc.x; if (R.sc.y > y0) y0 = R.sc.y; if ((int64_t)R.sc.x + R.sc.w < x1) x1 = (int64_t)R.sc.x + R.sc.w; if ((int64_t)R.sc.y + R.sc.h < y1) y1 = (int64_t)R.sc.y + R.sc.h; }
    if (x1 <= x0 || y1 <= y0) return 0;
    *r = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    return 1;
}

/* ---------------- pass dump (diagnostics) ---------------- */
/* TOTK_VK_PASSDUMP=<dir> and the input-file command "passdump <n>" (or TOTK_VK_PASSDUMP_FRAME=<frame>): every render pass
   of the n frames from two frames on is logged ("[passdump]": size, draws, clears, attachments, the fragment shaders it
   ran) and its colour attachments are written to <dir>/F<frame>-p<pass>-c<i>.ppm, at most 640 wide (float targets
   clipped to 0..1, sRGB ones kept encoded); dispatches are logged too. For finding the first pass of a wrong picture. */
static const char *g_pd_dir; static _Atomic uint64_t g_pd_frame; static _Atomic long g_pd_n = 1;
static int pd_on(void) {
    static int init;
    if (!init) { init = 1; g_pd_dir = getenv("TOTK_VK_PASSDUMP"); const char *f = getenv("TOTK_VK_PASSDUMP_FRAME"); if (g_pd_dir && f) atomic_store(&g_pd_frame, strtoull(f, NULL, 10)); }
    uint64_t f = atomic_load(&g_pd_frame), cur = atomic_load(&g_cur_frame);
    return g_pd_dir && f && cur >= f && cur < f + (uint64_t)atomic_load(&g_pd_n);
}
void dm_gfx_passdump_next(long n) {
    atomic_store(&g_pd_n, n > 0 && n <= 4 ? n : 1); atomic_store(&g_pd_frame, atomic_load(&g_cur_frame) + 2);
    VLOG("[passdump] frames %llu+%ld%s", (unsigned long long)atomic_load(&g_pd_frame), (long)atomic_load(&g_pd_n), getenv("TOTK_VK_PASSDUMP") ? "" : " (no TOTK_VK_PASSDUMP folder: nothing is written)");
}
typedef struct { VkBuffer b; VmaAllocation a; void *p; VkImage img; VmaAllocation ia; uint32_t w, h; char path[640]; } pd_shot;
static void pd_write(void *arg) {   /* completion thread */
    pd_shot *s = arg;
    vmaInvalidateAllocation(V.vma, s->a, 0, VK_WHOLE_SIZE);
    FILE *f = fopen(s->path, "wb");
    if (f) { fprintf(f, "P6\n%u %u\n255\n", s->w, s->h); const uint8_t *px = s->p; for (size_t i = 0; i < (size_t)s->w * s->h; i++) fwrite(px + i * 4, 1, 3, f); fclose(f); }
    vmaDestroyBuffer(V.vma, s->b, s->a); vmaDestroyImage(V.vma, s->img, s->ia); free(s);
}
static void pd_shoot(const vk_image *im, uint32_t level, uint32_t layer, const char *path) {   /* g_cb, no pass open */
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, im->format, &fp);
    if (im->depth || im->stencil || im->samples > 1 || !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || level >= im->levels) return;
    int srgb = im->format == VK_FORMAT_R8G8B8A8_SRGB || im->format == VK_FORMAT_B8G8R8A8_SRGB, is3d = im->type == VK_IMAGE_TYPE_3D;
    uint32_t sw = im->w >> level ? im->w >> level : 1, sh = im->h >> level ? im->h >> level : 1;
    uint32_t w = sw > 640 ? 640 : sw, h = (uint32_t)((uint64_t)sh * w / sw); if (!h) h = 1;
    pd_shot *s = calloc(1, sizeof *s); if (!s) return;
    s->w = w; s->h = h; snprintf(s->path, sizeof s->path, "%s", path);
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo iai = {.usage = VMA_MEMORY_USAGE_AUTO};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VmaAllocationInfo info;
    if (vmaCreateImage(V.vma, &ci, &iai, &s->img, &s->ia, NULL) != VK_SUCCESS) { free(s); return; }
    if (vmaCreateBuffer(V.vma, &bi, &bai, &s->b, &s->a, &info) != VK_SUCCESS) { vmaDestroyImage(V.vma, s->img, s->ia); free(s); return; }
    s->p = info.pMappedData;
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = s->img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(g_cb, &di);
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, is3d ? 0 : layer, 1}, .srcOffsets = {{0, 0, is3d ? (int32_t)layer : 0}, {(int32_t)sw, (int32_t)sh, is3d ? (int32_t)layer + 1 : 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, s->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl,
                   fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    vk_barrier_all(g_cb);
    VkBufferImageCopy rg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, s->b, 1, &rg);
    VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo hd = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(g_cb, &hd);
    vk_on_done(UINT64_MAX, pd_write, s);   /* after this submission */
}
typedef struct { VkBuffer b; VmaAllocation a; void *p; uint32_t n; VkFormat f; char what[96]; } pd_depth;
static void pd_depth_done(void *arg) {   /* completion thread: min / max / mean of the depth plane */
    pd_depth *d = arg; double sum = 0, lo = 1e30, hi = -1e30; uint32_t nan = 0;
    vmaInvalidateAllocation(V.vma, d->a, 0, VK_WHOLE_SIZE);
    for (uint32_t i = 0; i < d->n; i++) {
        double v; if (d->f == VK_FORMAT_D16_UNORM) v = ((const uint16_t *)d->p)[i] / 65535.0; else { float x; memcpy(&x, (const uint8_t *)d->p + 4 * (size_t)i, 4); v = x; }
        if (v != v) { nan++; continue; } sum += v; if (v < lo) lo = v; if (v > hi) hi = v;
    }
    VLOG("[passdump] %s depth: min %g max %g mean %g (%u NaN of %u)", d->what, lo, hi, d->n > nan ? sum / (d->n - nan) : 0, nan, d->n);
    vmaDestroyBuffer(V.vma, d->b, d->a); free(d);
}
static void pd_depth_stats(const vk_image *im, const char *what) {   /* g_cb, no pass open: level 0, layer 0 */
    if (!im->depth || im->samples > 1 || im->type == VK_IMAGE_TYPE_3D) return;
    uint32_t bpt = im->format == VK_FORMAT_D16_UNORM ? 2 : 4, n = im->w * im->h;
    if (im->format == VK_FORMAT_X8_D24_UNORM_PACK32 || im->format == VK_FORMAT_D24_UNORM_S8_UINT) return;   /* packed 24-bit: not decoded */
    pd_depth *d = calloc(1, sizeof *d); if (!d) return;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)n * bpt, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &ai, &d->b, &d->a, &info) != VK_SUCCESS) { free(d); return; }
    d->p = info.pMappedData; d->n = n; d->f = im->format; snprintf(d->what, sizeof d->what, "%s", what);
    VkBufferImageCopy rg = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, .imageExtent = {im->w, im->h, 1}};
    vkCmdCopyImageToBuffer(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &rg);
    VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo hd = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(g_cb, &hd);
    vk_on_done(UINT64_MAX, pd_depth_done, d);
}
/* the raw texels of a colour level / layer: per-channel min / max / mean (the 8-bit pictures hide small float values) */
typedef struct { VkBuffer b; VmaAllocation a; void *p; uint32_t n; VkFormat f; char what[160]; } pd_raw;
static float half_f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
static float uf11(uint32_t v, int mbits) {   /* unsigned small float: 5-bit exponent, mbits mantissa */
    uint32_t e = v >> mbits & 31, m = v & ((1u << mbits) - 1);
    if (e == 31) return m ? NAN : INFINITY;
    return e ? ldexpf(1.0f + (float)m / (float)(1u << mbits), (int)e - 15) : ldexpf((float)m / (float)(1u << mbits), -14);
}
static int raw_decode(VkFormat f, const uint8_t *p, float c[4]) {   /* channels decoded, 0 = format not handled */
    switch (f) {
        case VK_FORMAT_R32_SFLOAT: memcpy(c, p, 4); return 1;
        case VK_FORMAT_R32G32_SFLOAT: memcpy(c, p, 8); return 2;
        case VK_FORMAT_R32G32B32A32_SFLOAT: memcpy(c, p, 16); return 4;
        case VK_FORMAT_R16_SFLOAT: { uint16_t h; memcpy(&h, p, 2); c[0] = half_f(h); return 1; }
        case VK_FORMAT_R16G16_SFLOAT: for (int i = 0; i < 2; i++) { uint16_t h; memcpy(&h, p + 2 * i, 2); c[i] = half_f(h); } return 2;
        case VK_FORMAT_R16G16B16A16_SFLOAT: for (int i = 0; i < 4; i++) { uint16_t h; memcpy(&h, p + 2 * i, 2); c[i] = half_f(h); } return 4;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: { uint32_t v; memcpy(&v, p, 4); c[0] = uf11(v & 0x7ff, 6); c[1] = uf11(v >> 11 & 0x7ff, 6); c[2] = uf11(v >> 22, 5); return 3; }
        case VK_FORMAT_R8_UNORM: c[0] = p[0] / 255.0f; return 1;
        case VK_FORMAT_R8G8_UNORM: c[0] = p[0] / 255.0f; c[1] = p[1] / 255.0f; return 2;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: for (int i = 0; i < 4; i++) c[i] = p[i] / 255.0f; return 4;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: { uint32_t v; memcpy(&v, p, 4); c[0] = (v & 1023) / 1023.0f; c[1] = (v >> 10 & 1023) / 1023.0f; c[2] = (v >> 20 & 1023) / 1023.0f; c[3] = (v >> 30) / 3.0f; return 4; }
        default: return 0;
    }
}
static uint32_t raw_bpt(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: return 1;
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R8G8_UNORM: return 2;
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return 4;
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        default: return 0;
    }
}
static void pd_raw_done(void *arg) {   /* completion thread */
    pd_raw *r = arg; double lo[4], hi[4], sum[4]; uint32_t nan = 0, bpt = raw_bpt(r->f); int nc = 0;
    for (int i = 0; i < 4; i++) { lo[i] = 1e30; hi[i] = -1e30; sum[i] = 0; }
    vmaInvalidateAllocation(V.vma, r->a, 0, VK_WHOLE_SIZE);
    for (uint32_t i = 0; i < r->n; i++) {
        float c[4] = {0, 0, 0, 0}; nc = raw_decode(r->f, (const uint8_t *)r->p + (size_t)i * bpt, c);
        int bad = 0; for (int k = 0; k < nc; k++) bad |= c[k] != c[k] || isinf(c[k]);
        if (bad) { nan++; continue; }
        for (int k = 0; k < nc; k++) { sum[k] += c[k]; if (c[k] < lo[k]) lo[k] = c[k]; if (c[k] > hi[k]) hi[k] = c[k]; }
    }
    char line[400]; size_t o = (size_t)snprintf(line, sizeof line, "[trace] %s raw:", r->what); double m = r->n > nan ? (double)(r->n - nan) : 1;
    for (int k = 0; k < nc && o < sizeof line; k++) o += (size_t)snprintf(line + o, sizeof line - o, " ch%d %g..%g mean %g;", k, lo[k], hi[k], sum[k] / m);
    VLOG("%s %u non-finite of %u", line, nan, r->n);
    vmaDestroyBuffer(V.vma, r->b, r->a); free(r);
}
static void pd_raw_stats(const vk_image *im, uint32_t level, uint32_t layer, const char *what) {   /* g_cb, no pass open */
    uint32_t bpt = raw_bpt(im->format);
    if (!bpt || im->depth || im->stencil || im->samples > 1 || level >= im->levels) { VLOG("[trace] %s raw: format %d not decoded", what, im->format); return; }
    int is3d = im->type == VK_IMAGE_TYPE_3D;
    uint32_t w = im->w >> level ? im->w >> level : 1, h = im->h >> level ? im->h >> level : 1;
    pd_raw *r = calloc(1, sizeof *r); if (!r) return;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * bpt, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &ai, &r->b, &r->a, &info) != VK_SUCCESS) { free(r); return; }
    r->p = info.pMappedData; r->n = w * h; r->f = im->format; snprintf(r->what, sizeof r->what, "%s", what);
    VkBufferImageCopy rg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, is3d ? 0 : layer, 1}, .imageOffset = {0, 0, is3d ? (int32_t)layer : 0}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, r->b, 1, &rg);
    VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo hd = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(g_cb, &hd);
    vk_on_done(UINT64_MAX, pd_raw_done, r);
}
static void pd_pass(void) {   /* rp_end, the pass just ended */
    uint64_t f = atomic_load(&g_cur_frame);
    char line[1400]; size_t o = 0;
    o += (size_t)snprintf(line + o, sizeof line - o, "[passdump] F%llu p%llu %ux%u x%u draws %d cleared %#x", (unsigned long long)f, (unsigned long long)P.serial, P.w, P.h, P.layers, P.draws, P.cleared);
    for (int i = 0; i < P.n && o < sizeof line; i++) if (P.ctex[i])
        o += (size_t)snprintf(line + o, sizeof line - o, " | c%d %llu fmt %#x %ux%u lv%u ly%u", i, (unsigned long long)P.ctex[i]->serial, P.ctex[i]->d.format, P.ctex[i]->d.width, P.ctex[i]->d.height, P.clevel[i], P.clayer[i]);
    if (P.dimg && o < sizeof line) o += (size_t)snprintf(line + o, sizeof line - o, " | depth %llu fmt %#x", R.rt.depth ? (unsigned long long)R.rt.depth->serial : 0ull, R.rt.depth ? R.rt.depth->d.format : 0);
    for (int k = 0; k < P.npdk && o < sizeof line; k++) o += (size_t)snprintf(line + o, sizeof line - o, "%s%s", k ? " " : " | fs ", P.pdk[k]);
    VLOG("%s", line);
    for (int i = 0; i < P.n; i++) {
        if (!P.cimg[i] || P.cint[i]) continue;
        char path[640]; snprintf(path, sizeof path, "%s/F%llu-p%05llu-c%d.ppm", g_pd_dir, (unsigned long long)f, (unsigned long long)P.serial, i);
        pd_shoot(P.cimg[i], P.clevel[i], P.clayer[i], path);
    }
    if (P.dimg && P.hd && P.w >= 64) { char what[96]; snprintf(what, sizeof what, "F%llu p%llu serial %llu", (unsigned long long)f, (unsigned long long)P.serial, R.rt.depth ? (unsigned long long)R.rt.depth->serial : 0ull); pd_depth_stats(P.dimg, what); }
}

/* ---------------- GPU writes to memory and inline uniform updates ---------------- */
typedef struct { uint64_t lo, hi, value; } gw_t;   /* value UINT64_MAX: this submission */
static gw_t *g_gw; static int g_ngw, g_capgw;
/* gw_hits runs for every uniform binding: the 64 KiB granules the listed ranges touch are kept in a set (ranges over
   4 MiB in g_gw_big instead), so a miss costs a few probes and only a hit scans the list. The set is rebuilt by gw_commit;
   when it fills up, gw_hits scans. */
enum { GW_SET = 8192, GW_GRAN = 16, GW_BIG = 64 };
static uint64_t g_gw_key[GW_SET]; static uint32_t g_gw_gen[GW_SET], g_gw_cur = 1; static int g_gw_used, g_gw_full, g_gw_big;
static void gw_set_add(uint64_t lo, uint64_t hi) {
    uint64_t g0 = lo >> GW_GRAN, g1 = (hi - 1) >> GW_GRAN;
    if (g1 - g0 >= GW_BIG) { g_gw_big++; return; }
    for (uint64_t g = g0; g <= g1; g++) {
        if (g_gw_used >= GW_SET / 2) { g_gw_full = 1; return; }
        uint32_t h = (uint32_t)((g * 0x9e3779b97f4a7c15ull) >> 51);
        while (g_gw_gen[h] == g_gw_cur && g_gw_key[h] != g) h = (h + 1) & (GW_SET - 1);
        if (g_gw_gen[h] != g_gw_cur) { g_gw_gen[h] = g_gw_cur; g_gw_key[h] = g; g_gw_used++; }
    }
}
static void gw_note(uint64_t a, uint64_t n) {
    if (!a || !n) return;
    for (int i = g_ngw - 1; i >= 0 && i >= g_ngw - 8; i--) if (g_gw[i].value == UINT64_MAX && g_gw[i].lo <= a && a + n <= g_gw[i].hi) return;
    if (g_ngw == g_capgw) { g_capgw = g_capgw ? g_capgw * 2 : 1024; g_gw = realloc(g_gw, sizeof *g_gw * (size_t)g_capgw); if (!g_gw) VDIE("GPU write list"); }
    g_gw[g_ngw++] = (gw_t){a, a + n, UINT64_MAX};
    gw_set_add(a, a + n);
}
static int gw_hits(uint64_t a, uint64_t n) {
    if (!g_ngw) return 0;
    if (!g_gw_full && !g_gw_big) {
        int any = 0;
        for (uint64_t g = a >> GW_GRAN, g1 = (a + n - 1) >> GW_GRAN; g <= g1 && !any; g++) {
            uint32_t h = (uint32_t)((g * 0x9e3779b97f4a7c15ull) >> 51);
            while (g_gw_gen[h] == g_gw_cur) { if (g_gw_key[h] == g) { any = 1; break; } h = (h + 1) & (GW_SET - 1); }
        }
        if (!any) return 0;
    }
    for (int i = 0; i < g_ngw; i++) if (g_gw[i].lo < a + n && a < g_gw[i].hi) return 1;
    return 0;
}
static void gw_commit(uint64_t value) {   /* this submission's writes get its value; completed ones go */
    uint64_t done = vk_done_value(); int k = 0;
    for (int i = 0; i < g_ngw; i++) { if (g_gw[i].value == UINT64_MAX) g_gw[i].value = value; if (g_gw[i].value > done) g_gw[k++] = g_gw[i]; }
    g_ngw = k;
    if (++g_gw_cur == 0) { memset(g_gw_gen, 0, sizeof g_gw_gen); g_gw_cur = 1; }
    g_gw_used = g_gw_full = g_gw_big = 0;
    for (int i = 0; i < g_ngw; i++) gw_set_add(g_gw[i].lo, g_gw[i].hi);
}
/* Inline uniform updates (nvnCommandBufferUpdateUniformBuffer). A uniform buffer is bound as a copy of its bytes as the
   GPU will see them at that draw (ubo_resolve, pend_read): memory with the logged updates the GPU has not run yet laid
   over it, so an update never ends a render pass for a uniform read. The overlay is a shadow of the newest bytes per
   4 KiB page (written as the updates are recorded): the game rewrites the same per-object block before nearly every
   draw (3,400 a frame), and replaying the log for every binding cost more than the rest of the replay. A page's shadow
   goes when the GPU has run the last update in it (memory then holds those bytes, and a later CPU write wins).
   Reads on the GPU timeline (storage, vertex and index buffers, copies, GPU-written uniforms) need this submission's
   updates in memory first: upd_guard -> upd_flush, whose copies end the pass. */
typedef struct { uint64_t lo, hi; vk_slice data; uint8_t flushed; } upd_t;
static upd_t *g_upd; static int g_nupd, g_capupd, g_nunfl; static uint64_t g_upd_seq;   /* this submission's updates (for the copies) */
typedef struct { uint64_t page, seq, value; uint8_t sub; uint64_t mask[64], unfl[64]; uint32_t gsub[256]; uint8_t data[4096]; } ushadow;   /* value 0: this submission; unfl: bytes of updates not flushed yet */
/* each 16-byte granule of a shadow belongs to the submission that wrote it last: once that one has run on the GPU, its
   bytes are memory's again (a page the game updates every frame keeps its shadow, and a CPU write must still show) */
typedef struct { uint32_t serial; uint64_t value; } subval;
static subval g_subval[1024];
static int gran_live(uint32_t serial, uint64_t done) {
    if (serial == (uint32_t)vk_sub_serial) return 1;   /* this submission */
    const subval *v = &g_subval[serial & 1023];
    return v->serial == serial && v->value > done;
}
static nvn_hmap g_ushadow = {.klen = 8};
static ushadow **g_ush_sub; static int g_nush_sub, g_capush_sub;   /* pages written in this submission */
static ushadow *ush_live(uint64_t pg, uint64_t done) {   /* the page's shadow if the GPU has not run its last update yet */
    ushadow *h = nvn_hm_get(&g_ushadow, &pg);
    if (h && h->value && h->value <= done) { nvn_hm_del(&g_ushadow, &pg); free(h); return NULL; }
    return h;
}
static uint64_t upd_last_overlap(uint64_t lo, uint64_t hi) {   /* the newest update over the pages of [lo, hi), 0 = none (a cache key) */
    if (!g_ushadow.n || hi <= lo) return 0;
    uint64_t best = 0, done = vk_done_value();
    for (uint64_t pg = lo >> 12; pg <= (hi - 1) >> 12; pg++) { ushadow *h = ush_live(pg, done); if (h && h->seq > best) best = h->seq; }
    return best;
}
static void upd_overlay(uint64_t a, uint8_t *out, size_t n) {   /* the bytes [a, a+n) as the GPU will see them at this point */
    if (!g_ushadow.n || !n) return;
    uint64_t done = vk_done_value();
    for (uint64_t pg = a >> 12; pg <= (a + n - 1) >> 12; pg++) {
        ushadow *h = ush_live(pg, done); if (!h) continue;
        uint64_t p0 = pg << 12, s0 = p0 > a ? p0 : a, e0 = p0 + 4096 < a + n ? p0 + 4096 : a + n;
        for (uint32_t o = (uint32_t)(s0 - p0), oe = (uint32_t)(e0 - p0); o < oe;) {   /* by 16-byte granule: live ones, their shadowed bytes */
            uint32_t g = o >> 4, ge = (g + 1) << 4 < oe ? (g + 1) << 4 : oe;
            if (!gran_live(h->gsub[g], done)) { o = ge; continue; }
            uint64_t m = h->mask[o >> 6] >> (o & 63);   /* the granule's bits (a granule never crosses a mask word) */
            uint32_t len = ge - o;
            if ((m & ((len < 64 ? 1ull << len : 0) - 1)) == ((len < 64 ? 1ull << len : 0) - 1)) memcpy(out + (p0 + o - a), h->data + o, len);
            else for (uint32_t x = o; x < ge; x++) if (h->mask[x >> 6] >> (x & 63) & 1) out[p0 + x - a] = h->data[x];
            o = ge;
        }
    }
}
static const char *g_flush_why = "upd";
static void upd_flush(void) {   /* this submission's pending updates reach memory now (GPU order) */
    if (!g_nunfl) return;
    RP_END(g_flush_why); g_flush_why = "upd";
    for (int i = 0; i < g_nupd; i++) {
        if (g_upd[i].flushed) continue;
        g_upd[i].flushed = 1;
        VkBuffer b; VkDeviceSize off;
        if (!vk_pool_lookup(g_upd[i].lo, g_upd[i].hi - g_upd[i].lo, &b, &off, NULL)) continue;
        VkBufferCopy c = {g_upd[i].data.off, off, g_upd[i].hi - g_upd[i].lo};
        vkCmdCopyBuffer(g_cb, g_upd[i].data.buf, b, 1, &c);
        vk_barrier_all(g_cb);   /* overlapping updates in order */
    }
    g_nunfl = 0;
    for (int i = 0; i < g_nush_sub; i++) memset(g_ush_sub[i]->unfl, 0, sizeof g_ush_sub[i]->unfl);
}
static int ush_unflushed(const ushadow *h, uint64_t lo, uint64_t hi) {   /* bytes of [lo, hi) in the page not flushed yet */
    uint64_t p0 = h->page << 12; uint32_t o = lo > p0 ? (uint32_t)(lo - p0) : 0, e = hi < p0 + 4096 ? (uint32_t)(hi - p0) : 4096;
    for (uint32_t x = o; x < e;) {   /* by mask word */
        uint32_t wd = x >> 6, b0 = x & 63, b1 = (e - (wd << 6)) < 64 ? e - (wd << 6) : 64;
        uint64_t m = h->unfl[wd] >> b0; if (b1 - b0 < 64) m &= (1ull << (b1 - b0)) - 1;
        if (m) return 1;
        x = (wd + 1) << 6;
    }
    return 0;
}
static void upd_guard_w(uint64_t lo, uint64_t hi, const char *why) {   /* a GPU-timeline read of [lo, hi): pending updates there first */
    if (!g_nunfl || hi <= lo) return;
    uint64_t pg0 = lo >> 12, pg1 = (hi - 1) >> 12;
    if (pg1 - pg0 + 1 > (uint64_t)g_nush_sub) {   /* a range of more pages than this submission updated (vertex buffers): those pages */
        for (int i = 0; i < g_nush_sub; i++) {
            const ushadow *h = g_ush_sub[i];
            if (h->page >= pg0 && h->page <= pg1 && ush_unflushed(h, lo, hi)) { g_flush_why = why; upd_flush(); return; }
        }
        return;
    }
    for (uint64_t pg = pg0; pg <= pg1; pg++) {
        ushadow *h = nvn_hm_get(&g_ushadow, &pg); if (!h || !h->sub) continue;
        if (ush_unflushed(h, lo, hi)) { g_flush_why = why; upd_flush(); return; }
    }
}
/* mtl_sub_commit, after the submission: its pages' shadows stay until the GPU has run it */
static void upd_commit(uint64_t v) {
    g_subval[(uint32_t)vk_sub_serial & 1023] = (subval){(uint32_t)vk_sub_serial, v};
    for (int i = 0; i < g_nush_sub; i++) { g_ush_sub[i]->value = v; g_ush_sub[i]->sub = 0; }
    g_nush_sub = 0; g_nupd = 0; g_nunfl = 0;
}
static void mtl_sub_update(const void *data, uint64_t dst, uint64_t n) {
    if (!n) return;
    if (!vk_pool_contains(dst, n)) VDIE("uniform update target %#llx+%llu not in a memory pool", (unsigned long long)dst, (unsigned long long)n);
    vk_slice s = vk_arena_alloc((size_t)n, 16); memcpy(s.cpu, data, (size_t)n);
    if (g_nupd == g_capupd) { g_capupd = g_capupd ? g_capupd * 2 : 256; g_upd = realloc(g_upd, sizeof *g_upd * (size_t)g_capupd); if (!g_upd) VDIE("update log"); }
    g_upd[g_nupd++] = (upd_t){dst, dst + n, s, 0}; g_nunfl++;
    uint64_t seq = ++g_upd_seq;
    for (uint64_t pg = dst >> 12; pg <= (dst + n - 1) >> 12; pg++) {
        ushadow *h = nvn_hm_get(&g_ushadow, &pg);
        if (!h) { h = calloc(1, sizeof *h); if (!h) VDIE("update shadow"); h->page = pg; nvn_hm_put(&g_ushadow, &pg, h); }
        if (!h->sub) {
            h->sub = 1;
            if (g_nush_sub == g_capush_sub) { g_capush_sub = g_capush_sub ? g_capush_sub * 2 : 64; g_ush_sub = realloc(g_ush_sub, sizeof *g_ush_sub * (size_t)g_capush_sub); if (!g_ush_sub) VDIE("update shadow"); }
            g_ush_sub[g_nush_sub++] = h;
        }
        h->seq = seq; h->value = 0;
        uint64_t p0 = pg << 12, s0 = p0 > dst ? p0 : dst, e0 = p0 + 4096 < dst + n ? p0 + 4096 : dst + n;
        memcpy(h->data + (s0 - p0), (const uint8_t *)data + (s0 - dst), (size_t)(e0 - s0));
        for (uint64_t x = s0 - p0; x < e0 - p0; x++) { h->mask[x >> 6] |= 1ull << (x & 63); h->unfl[x >> 6] |= 1ull << (x & 63); }
        for (uint64_t g = (s0 - p0) >> 4; g <= (e0 - p0 - 1) >> 4; g++) h->gsub[g] = (uint32_t)vk_sub_serial;
    }
}
/* CPU read of guest bytes as they will be at this point of the submission */
static void pend_read(uint64_t a, void *out, size_t n) { memcpy(out, (const void *)(uintptr_t)a, n); upd_overlay(a, out, n); }

/* uniform copies of this submission: (addr, size, latest overlapping update) -> arena slice */
typedef struct { uint64_t addr, size, seq; vk_slice s; } ucopy;
#define UCACHE 4096
static ucopy g_ucache[UCACHE]; static uint64_t g_ucache_sub[UCACHE];
static int ubo_resolve(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *range) {
    if (!addr) return 0;
    uint64_t n = size ? size : 65536; if (n > 65536) n = 65536;
    VkBuffer pb; VkDeviceSize po, avail;
    if (!vk_pool_lookup(addr, 1, &pb, &po, &avail)) return 0;
    if (n > avail) n = avail;
    if (gw_hits(addr, n)) {   /* the GPU writes these bytes: read them on the GPU timeline */
        upd_guard_w(addr, addr + n, "upd:ubo_gpu");
        if (po % V.props.limits.minUniformBufferOffsetAlignment == 0) { *buf = pb; *off = po; *range = n; return 1; }
        VK_ONCE(ubomis, "GPU-written uniform range %#llx is not %llu-byte aligned: read as of the submit call", (unsigned long long)addr, (unsigned long long)V.props.limits.minUniformBufferOffsetAlignment);
    }
    uint64_t seq = g_ushadow.n ? upd_last_overlap(addr, addr + n) : 0;
    unsigned h = (unsigned)(((addr >> 4) ^ (addr >> 16) ^ n) & (UCACHE - 1));
    ucopy *c = &g_ucache[h];
    if (g_ucache_sub[h] == vk_sub_serial && c->addr == addr && c->size == n && c->seq == seq) { *buf = c->s.buf; *off = c->s.off; *range = n; return 1; }
    vk_slice s = vk_arena_alloc((size_t)n, V.props.limits.minUniformBufferOffsetAlignment);
    memcpy(s.cpu, (const void *)(uintptr_t)addr, (size_t)n);
    if (seq) upd_overlay(addr, s.cpu, (size_t)n);   /* the updates the GPU will have run by this draw */
    *c = (ucopy){addr, n, seq, s}; g_ucache_sub[h] = vk_sub_serial;
    *buf = s.buf; *off = s.off; *range = n;
    return 1;
}
static int ssbo_resolve(uint64_t addr, uint64_t size, int rw, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *range) {
    if (!addr) return 0;
    uint64_t a = addr & ~(uint64_t)15, mis = addr - a;
    VkBuffer pb; VkDeviceSize po, avail;
    if (!vk_pool_lookup(a, 1, &pb, &po, &avail)) return 0;
    uint64_t n = size ? size + mis : avail;
    if (n > avail) n = avail;
    if (n > V.props.limits.maxStorageBufferRange) n = V.props.limits.maxStorageBufferRange;
    upd_guard_w(a, a + n, "upd:ssbo");
    if (rw) gw_note(a, n);
    *buf = pb; *off = po; *range = n;
    return 1;
}

/* ---------------- driver constant banks ---------------- */
static uint32_t g_dispatch_grid[3];
static void driver_bank_fill(uint8_t *d, int bank, int stage) {
    memset(d, 0, 0x1000);
    if (bank == 0) {
        int base = stage == 0 ? 0x110 : stage == 5 ? 0x310 : stage == 1 ? 0x510 : -1;
        if (base >= 0) for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[stage][r].addr; uint32_t n = (uint32_t)S.ssbo[stage][r].size; memcpy(d + base + 16 * r, &a, 8); memcpy(d + base + 16 * r + 8, &n, 4); }
        if (stage == 5) {
            memcpy(d + 0x200, g_dispatch_grid, sizeof g_dispatch_grid);
            for (int r = 0; r < 16; r++) { uint64_t a = S.ubo[5][r].addr; uint32_t n = (uint32_t)S.ubo[5][r].size; memcpy(d + 0x210 + 16 * r, &a, 8); memcpy(d + 0x210 + 16 * r + 8, &n, 4); }
        } else if (stage == 0) { const float ps_max = 2047.9375f; memcpy(d + 0x610, &ps_max, sizeof ps_max); }
    } else {
        for (int r = 0; r < 128; r++) { uint32_t th = (uint32_t)S.tex[stage][r], sh_ = (uint32_t)S.samp[stage][r]; memcpy(d + 0x168 + 8 * r, &th, 4); memcpy(d + 0x568 + 8 * r, &sh_, 4); }
        for (int r = 0; r < 32; r++) { uint32_t th = (uint32_t)S.tex[stage][r]; memcpy(d + 0x20 + 8 * r, &th, 4); }
    }
}

/* ---------------- descriptor sets ---------------- */
static VkDescriptorSet set_alloc(void) {
    slot_t *s = &g_slots[g_cs];
    for (;;) {
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = s->dp[s->curdp], .descriptorSetCount = 1, .pSetLayouts = &V.dsl};
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(V.dev, &ai, &set);
        if (r == VK_SUCCESS) return set;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) VDIE("vkAllocateDescriptorSets: %d", r);
        if (++s->curdp == s->ndp) { if (s->ndp == 64) VDIE("descriptor pools exhausted in one submission"); s->dp[s->ndp++] = dpool_new(); }
    }
}
static inline uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); return h * 0xff51afd7ed558ccdull; }
typedef struct {
    int nw; VkWriteDescriptorSet w[NVN_REFL_MAXB];
    VkDescriptorBufferInfo bi[NVN_REFL_MAXB]; VkDescriptorImageInfo ii[NVN_REFL_MAXB]; VkBufferView tv[NVN_REFL_MAXB];
    uint64_t hash;
    vk_image *img[16]; int nimg;   /* the images the stage samples or stores to (att_feedback) */
    int simg;                      /* it stores to an image */
    int img_over;                  /* it names more images than img holds */
} dwrites;
/* which attachments of the open pass an image is: bits 0-7 colour slots, 8 the depth / stencil image */
static uint32_t sampled_is_attachment(vk_image *im) {
    if (!im || !P.open) return 0;
    uint32_t m = 0;
    for (int i = 0; i < P.n; i++) if (P.cimg[i] == im) m |= 1u << i;
    return m | (P.dimg == im ? 1u << 8 : 0);
}
/* the attachments of the open pass among a stage's sampled / stored images */
static uint32_t att_feedback(const dwrites *d) {
    uint32_t m = 0;
    if (P.open) for (int k = 0; k < d->nimg; k++) m |= sampled_is_attachment(d->img[k]);
    return m;
}
static int img_listed(const dwrites *d, const vk_image *im) { for (int k = 0; d && im && k < d->nimg; k++) if (d->img[k] == im) return 1; return 0; }
/* No pass open and a clear pending for the next one (its load clear) on a target the draw samples: on the Switch the
   clear lands before the draw reads, so it runs first, in a pass of its own (nvn_mtl.m att_unstored_read). */
static int pending_clear_sampled(const dwrites *a, const dwrites *b) {
    if (P.open || !R.has_rt || (!a->nimg && !(b && b->nimg))) return 0;
    for (int i = 0; i < R.rt.n && i < 8; i++) if (R.clear_c[i] && R.rt.color[i]) { vk_image *im = vk_tex_image(R.rt.color[i]); if (img_listed(a, im) || img_listed(b, im)) return 1; }
    if ((R.clear_d || R.clear_s) && R.rt.depth) { vk_image *im = vk_tex_image(R.rt.depth); if (img_listed(a, im) || img_listed(b, im)) return 1; }
    return 0;
}
/* a carried clear (lazy clears) on an image the draw samples, or a draw naming more images than the lists hold */
static int carry_sampled(const dwrites *a, const dwrites *b) {
    if (a->img_over || (b && b->img_over)) return 1;
    for (int k = 0; k < a->nimg; k++) if (carry_on(a->img[k])) return 1;
    for (int k = 0; b && k < b->nimg; k++) if (carry_on(b->img[k])) return 1;
    return 0;
}
/* TOTK_ATT_STORE=0: a draw that samples an attachment of its pass ends the pass only when that attachment was cleared in
   it (the rule before; draws then read what the pass loaded instead of what earlier draws of the pass drew) */
static int att_store_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("TOTK_ATT_STORE"); on = !(e && !strcmp(e, "0")); } return on; }
typedef struct { uint64_t h; uint32_t gen; uint64_t sub; int vt; uint8_t shadow, cls, depth; VkImageView view; vk_image *im; uint32_t epoch; } tbind;
static tbind g_tbind[2][128];   /* vertex / fragment texture slots, resolved this submission */
/* samplers by id (vk_samp_by_id takes the texture lock); a registration bumps the registry generation */
typedef struct { int sid; uint32_t gen; VkSampler s[2]; } sbind;
static sbind g_sbind[4096];
static VkSampler samp_cached(int sid, int nearest) {
    sbind *c = &g_sbind[(unsigned)sid & 4095]; uint32_t gen = (uint32_t)vk_tex_registry_gen();
    if (c->sid != sid || c->gen != gen) *c = (sbind){sid, gen, {VK_NULL_HANDLE, VK_NULL_HANDLE}};
    if (!c->s[nearest]) c->s[nearest] = vk_samp_by_id(sid, nearest);
    return c->s[nearest];
}
/* Resolve every binding of a shader into descriptor writes (and the hash of what they name). */
static void resolve(vk_shader *sh, int st, dwrites *d) {
    d->nw = 0; d->hash = 0x1234567 + (uint64_t)st; d->nimg = 0; d->simg = 0; d->img_over = 0;
    for (int i = 0; i < sh->nb; i++) {
        const vk_bind *b = &sh->b[i];
        VkWriteDescriptorSet *w = &d->w[d->nw];
        *w = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b->binding, .descriptorCount = 1};
        VkBuffer buf = g_zero.buf; VkDeviceSize off = 0, range = 65536;
        switch (b->type) {
            case VB_UBO: {
                w->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                if (b->kind == NVN_RB_SUPPORT) { buf = g_support.buf; }
                else if (b->kind == NVN_RB_C1) { if (sh->c1_buf) { buf = sh->c1_buf; off = sh->c1_off; range = sh->c1_range; } }
                else if (b->kind == NVN_RB_DRV0 || b->kind == NVN_RB_DRV2 || b->bank == 0 || b->bank == 2) {
                    int bank = b->kind == NVN_RB_DRV2 || b->bank == 2 ? 2 : 0;
                    if (sh->rf.drv_read[bank]) { vk_slice s = vk_arena_alloc(0x1000, V.props.limits.minUniformBufferOffsetAlignment); driver_bank_fill(s.cpu, bank, st); buf = s.buf; off = s.off; range = 0x1000; }
                } else if (b->kind == NVN_RB_CBUF && b->reg < 16) {
                    const p_bind_buf *u = &S.ubo[st][b->reg];
                    if (!ubo_resolve(u->addr, u->size, &buf, &off, &range)) { buf = g_zero.buf; off = 0; range = 65536; }
                }
                d->bi[d->nw] = (VkDescriptorBufferInfo){buf, off, range}; w->pBufferInfo = &d->bi[d->nw];
                d->hash = mix(mix(mix(d->hash, (uint64_t)(uintptr_t)buf), off), range ^ ((uint64_t)b->binding << 40));
                break;
            }
            case VB_SSBO: {
                w->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                buf = g_zero_ssbo.buf; off = 0; range = g_zero_ssbo.size;
                int ok = 0;
                if (b->kind == NVN_RB_SBUFP && b->reg < 16) ok = ssbo_resolve(S.ssbo[st][b->reg].addr, S.ssbo[st][b->reg].size, b->rw, &buf, &off, &range);
                else if (b->kind == NVN_RB_UBOP && b->reg < 16) {
                    /* a compute stage's uniform buffer, read through a pointer: copied as of the submit call like the
                       graphics stages' (ubo_resolve; GPU-written bytes stay bound in memory). Bound from memory, the
                       dispatch read what the game wrote for the next frame when it ran while the game recorded on
                       (TOTK_SYNC_LAG): the shadow cascades jumped for a frame now and then. The copy keeps the 16-byte
                       alignment the shader's address arithmetic expects. TOTK_VK_UBOP_COPY=0: from memory. */
                    static int copy = -1; if (copy < 0) { const char *e = getenv("TOTK_VK_UBOP_COPY"); copy = !(e && !strcmp(e, "0")) && V.props.limits.minUniformBufferOffsetAlignment % (V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 1) == 0; }
                    const p_bind_buf *u = &S.ubo[st][b->reg]; uint64_t a16 = u->addr & ~(uint64_t)15;
                    uint64_t un = (u->size ? u->size : 65536) + (u->addr - a16);
                    ok = copy && u->addr && !gw_hits(a16, un) && ubo_resolve(a16, un, &buf, &off, &range);
                    if (!ok) ok = ssbo_resolve(u->addr, u->size, 0, &buf, &off, &range);
                }
                else if (b->kind == NVN_RB_SBUFU && b->reg < 16) {
                    const p_bind_buf *u = &S.ubo[st][b->reg]; uint64_t va = 0;
                    if (u->addr && u->size >= (uint64_t)b->off + 8) pend_read(u->addr + b->off, &va, 8);
                    ok = va && ssbo_resolve(va, 0, b->rw, &buf, &off, &range);
                    if (!ok && va) VK_ONCE(sbufu, "storage buffer pointer %#llx (uniform slot %d + %u) is not in a memory pool", (unsigned long long)va, b->reg, b->off);
                }
                if (!ok) { buf = g_zero_ssbo.buf; off = 0; range = g_zero_ssbo.size; }
                d->bi[d->nw] = (VkDescriptorBufferInfo){buf, off, range}; w->pBufferInfo = &d->bi[d->nw];
                d->hash = mix(mix(mix(d->hash, (uint64_t)(uintptr_t)buf), off), range ^ ((uint64_t)b->binding << 40));
                break;
            }
            case VB_COMBINED: case VB_SAMPLED: {
                w->descriptorType = b->type == VB_COMBINED ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                uint64_t h = b->reg < 128 ? S.tex[st][b->reg] : 0;
                int tid = (int)(h & 0xfffff), fmt = 0; vk_image *im = NULL; VkImageView view; int depth_tex;
                /* the slot's view as resolved earlier in this submission, when nothing it depends on changed (vk_tex_by_id
                   takes the texture lock; a draw binds about ten textures a stage, 3,400 draws a frame) */
                tbind *tc = st < 2 && b->reg < 128 ? &g_tbind[st][b->reg] : NULL;
                uint32_t gen = (uint32_t)vk_tex_registry_gen();
                if (tc && tc->h == h && tc->gen == gen && tc->sub == vk_sub_serial && tc->vt == b->view_type && tc->shadow == b->shadow && tc->cls == b->cls &&
                    (!tc->im || tc->im->epoch == tc->epoch)) { view = tc->view; im = tc->im; depth_tex = tc->depth; }
                else {
                    view = tid ? vk_tex_by_id(tid, (VkImageViewType)b->view_type, &im, &fmt) : VK_NULL_HANDLE;
                    depth_tex = im && (im->depth || im->stencil);
                    if (view && b->shadow && !depth_tex) view = VK_NULL_HANDLE;   /* compare sampling needs a depth texture */
                    if (view && !b->shadow && depth_tex && im->stencil && !im->depth) view = VK_NULL_HANDLE;
                    if (!view) { view = vk_dummy_view((VkImageViewType)b->view_type, b->shadow ? 3 : b->cls); depth_tex = b->shadow; im = NULL; }
                    if (tc) *tc = (tbind){h, gen, vk_sub_serial, (int)b->view_type, b->shadow, b->cls, (uint8_t)depth_tex, view, im, im ? im->epoch : 0};
                }
                if (im) { if (d->nimg < 16) d->img[d->nimg++] = im; else d->img_over = 1; }
                VkSampler smp = VK_NULL_HANDLE;
                if (b->type == VB_COMBINED) {
                    int sid = b->samp_kind == NVN_RB_SAMP ? (int)((b->samp_reg < 128 ? S.samp[st][b->samp_reg] : 0) >> 20)
                            : b->samp_kind == NVN_RB_SAMPT ? (int)((b->samp_reg < 128 ? S.tex[st][b->samp_reg] : 0) >> 20) : (int)(h >> 20);
                    smp = sid ? samp_cached(sid, depth_tex && !b->shadow) : vk_samp_default();
                    if (!smp) smp = vk_samp_default();
                }
                d->ii[d->nw] = (VkDescriptorImageInfo){smp, view, VK_IMAGE_LAYOUT_GENERAL}; w->pImageInfo = &d->ii[d->nw];
                d->hash = mix(mix(mix(d->hash, (uint64_t)(uintptr_t)view), (uint64_t)(uintptr_t)smp), b->binding);
                break;
            }
            case VB_SAMPLER: {
                w->descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                int sid = (int)((b->reg < 128 ? (b->kind == NVN_RB_SAMPT ? S.tex[st][b->reg] : S.samp[st][b->reg]) : 0) >> 20);
                VkSampler smp = sid ? samp_cached(sid, 0) : vk_samp_default(); if (!smp) smp = vk_samp_default();
                d->ii[d->nw] = (VkDescriptorImageInfo){smp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}; w->pImageInfo = &d->ii[d->nw];
                d->hash = mix(mix(d->hash, (uint64_t)(uintptr_t)smp), b->binding);
                break;
            }
            case VB_STORAGE_IMG: {
                w->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                int tid = b->reg < 32 ? (int)(S.img[st][b->reg] & 0xfffff) : 0; vk_image *im = NULL;
                VkImageView view = tid ? vk_tex_storage_by_id(tid, (VkImageViewType)b->view_type, &im) : VK_NULL_HANDLE;
                if (view && im) { d->simg = 1; if (d->nimg < 16) d->img[d->nimg++] = im; else d->img_over = 1; }
                if (!view) view = vk_dummy_storage((VkImageViewType)b->view_type, b->cls);
                else { const tex_t *tt = vk_tex_slot(tid); if (tt) vk_tex_note_gpu_write(tt); }
                d->ii[d->nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL}; w->pImageInfo = &d->ii[d->nw];
                d->hash = mix(mix(d->hash, (uint64_t)(uintptr_t)view), b->binding);
                break;
            }
            case VB_UNIFORM_TEXEL: case VB_STORAGE_TEXEL: {
                int storage = b->type == VB_STORAGE_TEXEL;
                w->descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                uint64_t h = storage ? (b->reg < 32 ? S.img[st][b->reg] : 0) : (b->reg < 128 ? S.tex[st][b->reg] : 0);
                int tid = (int)(h & 0xfffff);
                VkBufferView bv = tid ? vk_tex_buffer_view(tid, storage) : VK_NULL_HANDLE;
                if (bv && storage) { const tex_t *tt = vk_tex_slot(tid); if (tt && tt->mem) gw_note((uint64_t)(uintptr_t)tt->mem, tt->L.size ? tt->L.size : (uint64_t)tt->d.width * 16); }
                if (!bv) bv = vk_dummy_texel(storage, b->cls);
                d->tv[d->nw] = bv; w->pTexelBufferView = &d->tv[d->nw];
                d->hash = mix(mix(d->hash, (uint64_t)(uintptr_t)bv), b->binding);
                break;
            }
        }
        d->nw++;
    }
}
/* A set written in this submission with the same contents (dwrites' hash) is bound again instead of a new one: the
   sets live until the submission's slot is reused. */
enum { SCACHE = 2048 };
static struct { uint64_t hash, sub; VkDescriptorSet set; } g_scache[SCACHE];
static VkDescriptorSet set_write(dwrites *d) {
    unsigned h = (unsigned)((d->hash * 0x9e3779b97f4a7c15ull) >> 53);
    if (g_scache[h].sub == vk_sub_serial && g_scache[h].hash == d->hash && g_scache[h].set) return g_scache[h].set;
    VkDescriptorSet set = set_alloc();
    for (int i = 0; i < d->nw; i++) d->w[i].dstSet = set;
    if (d->nw) vkUpdateDescriptorSets(V.dev, (uint32_t)d->nw, d->w, 0, NULL);
    g_scache[h].hash = d->hash; g_scache[h].sub = vk_sub_serial; g_scache[h].set = set;
    return set;
}

/* ---------------- dynamic state ---------------- */
uint32_t vk_tex_bind_gen[6];
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    for (int i = 0; i < 6; i++) vk_tex_bind_gen[i]++;
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1;
}
static void bind_null_stages(void) {
    VkShaderStageFlagBits st[3] = {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT};
    VkShaderEXT nul[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    V.CmdBindShadersEXT(g_cb, 3, st, nul);
}
static void bind_gfx_shaders(VkShaderEXT vs, VkShaderEXT fs) {
    if (C.vs == vs && C.fs == fs && C.valid) return;
    VkShaderStageFlagBits st[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkShaderEXT sh[2] = {vs, fs};
    V.CmdBindShadersEXT(g_cb, 2, st, sh);
    C.vs = vs; C.fs = fs;
}
/* everything a draw needs, set when it changed */
static void set_raster_state(int prim_lines, VkPrimitiveTopology topo, int restart) {
    VkCullModeFlags cull = S.poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : S.poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    VkFrontFace front = S.poly.front == NVN_FRONT_FACE_CCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    VkPolygonMode poly = S.poly.mode == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : S.poly.mode == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL;
    if (!C.valid || C.cull != cull) { vkCmdSetCullMode(g_cb, cull); C.cull = cull; }
    if (!C.valid || C.front != front) { vkCmdSetFrontFace(g_cb, front); C.front = front; }
    if (!C.valid || C.poly != poly) { V.CmdSetPolygonModeEXT(g_cb, poly); C.poly = poly; }
    int ben = (S.poly.offset_en & 4) != 0;   /* NVN polygon offset enables: point 1, line 2, fill 4 (Metal uses fill) */
    if (!C.valid || C.dbias_en != ben) { vkCmdSetDepthBiasEnable(g_cb, ben); C.dbias_en = ben; }
    if (ben && (!C.valid || C.dbias[0] != S.po.units || C.dbias[1] != S.po.clamp || C.dbias[2] != S.po.factor)) {
        vkCmdSetDepthBias(g_cb, S.po.units, S.po.clamp, S.po.factor); C.dbias[0] = S.po.units; C.dbias[1] = S.po.clamp; C.dbias[2] = S.po.factor; }
    if (!C.valid || C.dclamp != S.depth_clamp) { V.CmdSetDepthClampEnableEXT(g_cb, S.depth_clamp != 0); C.dclamp = S.depth_clamp; }
    if (!C.valid || C.topo != topo) { vkCmdSetPrimitiveTopology(g_cb, topo); C.topo = topo; }
    if (!C.valid || C.restart != restart) { vkCmdSetPrimitiveRestartEnable(g_cb, restart); C.restart = restart; }
    if (prim_lines || poly == VK_POLYGON_MODE_LINE) vkCmdSetLineWidth(g_cb, 1.0f);
}
static void set_ds_state(void) {
    int dt = P.hd && S.ds.dt, dw = P.hd && S.ds.dw, st = P.hs && S.ds.st;
    VkCompareOp dop = vk_compare(S.ds.df);
    if (!C.valid || C.dt != dt) { vkCmdSetDepthTestEnable(g_cb, dt); C.dt = dt; }
    if (!C.valid || C.dw != dw) { vkCmdSetDepthWriteEnable(g_cb, dw); C.dw = dw; }
    if (!C.valid || C.dop != dop) { vkCmdSetDepthCompareOp(g_cb, dop); C.dop = dop; }
    if (!C.valid || C.st != st) { vkCmdSetStencilTestEnable(g_cb, st); C.st = st; }
    VkStencilOp so[2][3] = {{vk_stencil_op(S.ds.fail_f), vk_stencil_op(S.ds.zpass_f), vk_stencil_op(S.ds.zfail_f)}, {vk_stencil_op(S.ds.fail_b), vk_stencil_op(S.ds.zpass_b), vk_stencil_op(S.ds.zfail_b)}};
    VkCompareOp sc[2] = {vk_compare(S.ds.sf_f), vk_compare(S.ds.sf_b)};
    for (int f = 0; f < 2; f++) {
        VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
        if (!C.valid || memcmp(C.sop[f], so[f], sizeof so[f]) || C.scmp[f] != sc[f]) { vkCmdSetStencilOp(g_cb, face, so[f][0], so[f][1], so[f][2], sc[f]); memcpy(C.sop[f], so[f], sizeof so[f]); C.scmp[f] = sc[f]; }
        if (!C.valid || C.smask[f] != S.st_vmask[f]) { vkCmdSetStencilCompareMask(g_cb, face, S.st_vmask[f]); C.smask[f] = S.st_vmask[f]; }
        if (!C.valid || C.swmask[f] != S.st_wmask[f]) { vkCmdSetStencilWriteMask(g_cb, face, S.st_wmask[f]); C.swmask[f] = S.st_wmask[f]; }
        if (!C.valid || C.sref[f] != S.st_ref[f]) { vkCmdSetStencilReference(g_cb, face, S.st_ref[f]); C.sref[f] = S.st_ref[f]; }
    }
}
static void set_blend_state(void) {
    uint32_t n = (uint32_t)P.n;
    VkBool32 en[8]; VkColorBlendEquationEXT eq[8]; VkColorComponentFlags wm[8];
    for (uint32_t i = 0; i < 8; i++) {
        en[i] = P.cview[i] && (S.color.blend_en >> i & 1) && !P.cint[i];
        const blend_st *b = &S.blend[i];
        eq[i] = (VkColorBlendEquationEXT){vk_blend_factor(b->src_rgb), vk_blend_factor(b->dst_rgb), vk_blend_op(b->eq_rgb), vk_blend_factor(b->src_a), vk_blend_factor(b->dst_a), vk_blend_op(b->eq_a)};
        uint32_t m = P.cview[i] ? (S.chmask.mask >> (4 * i)) & 0xf : 0;
        wm[i] = (m & 1 ? VK_COLOR_COMPONENT_R_BIT : 0) | (m & 2 ? VK_COLOR_COMPONENT_G_BIT : 0) | (m & 4 ? VK_COLOR_COMPONENT_B_BIT : 0) | (m & 8 ? VK_COLOR_COMPONENT_A_BIT : 0);
    }
    if (!n) return;
    if (C.nblend != n || memcmp(C.ben, en, sizeof en[0] * n)) { V.CmdSetColorBlendEnableEXT(g_cb, 0, n, en); memcpy(C.ben, en, sizeof en); }
    if (C.nblend != n || memcmp(C.beq, eq, sizeof eq[0] * n)) { V.CmdSetColorBlendEquationEXT(g_cb, 0, n, eq); memcpy(C.beq, eq, sizeof eq); }
    if (C.nblend != n || memcmp(C.wmask, wm, sizeof wm[0] * n)) { V.CmdSetColorWriteMaskEXT(g_cb, 0, n, wm); memcpy(C.wmask, wm, sizeof wm); }
    C.nblend = n;
    if (!C.valid || memcmp(C.bconst, S.blend_color, 16)) { vkCmdSetBlendConstants(g_cb, S.blend_color); memcpy(C.bconst, S.blend_color, 16); }
    int logic = S.color.logic_op != NVN_LOGIC_OP_COPY;
    if (V.CmdSetLogicOpEnableEXT && (!C.valid || C.logic_en != logic)) { V.CmdSetLogicOpEnableEXT(g_cb, logic); C.logic_en = logic; }
    if (logic && V.CmdSetLogicOpEXT && (!C.valid || C.logic != vk_logic_op(S.color.logic_op))) { V.CmdSetLogicOpEXT(g_cb, vk_logic_op(S.color.logic_op)); C.logic = vk_logic_op(S.color.logic_op); }
    if (logic && !V.CmdSetLogicOpEnableEXT) VK_ONCE(logic, "logic op %u: not available on this device (drawn without)", S.color.logic_op);
}
static void set_ms_state(void) {
    VkSampleCountFlagBits s = P.samples ? P.samples : VK_SAMPLE_COUNT_1_BIT;
    if (C.samples != s) { V.CmdSetRasterizationSamplesEXT(g_cb, s); VkSampleMask m[2] = {~0u, ~0u}; V.CmdSetSampleMaskEXT(g_cb, s, m); C.samples = s; }
    int a2c = S.ms.a2c != 0;
    if (!C.valid || C.a2c != a2c) { V.CmdSetAlphaToCoverageEnableEXT(g_cb, a2c); if (V.CmdSetAlphaToOneEnableEXT) V.CmdSetAlphaToOneEnableEXT(g_cb, VK_FALSE); C.a2c = a2c; }
}
static void cb_state_defaults(void) {   /* state every draw needs that the game never changes */
    vkCmdSetRasterizerDiscardEnable(g_cb, VK_FALSE);
    vkCmdSetDepthBoundsTestEnable(g_cb, VK_FALSE);
    bind_null_stages();
}
static VkFormat unbound_attr_format(int ct) { return ct == NVN_CT_UINT ? VK_FORMAT_R32G32B32A32_UINT : ct == NVN_CT_INT ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT; }
static void set_vertex_input(const vk_shader *vs) {
    VkVertexInputBindingDescription2EXT vb[17]; VkVertexInputAttributeDescription2EXT va[32]; uint32_t nvb = 0, nva = 0;
    uint32_t used = 0;
    for (int v = 0; v < vs->rf.nv && nva < 32; v++) {
        int loc = vs->rf.v[v].loc, a = vs->rf.v[v].nvn;
        VkFormat f = a < 16 && a < S.nattr ? vk_vertex_format(S.attr[a].fmt) : VK_FORMAT_UNDEFINED;
        if (f == VK_FORMAT_UNDEFINED || S.attr[a].stream >= 16) {
            va[nva++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, (uint32_t)loc, 16, unbound_attr_format(vs->rf.v[v].ct), 0};
            used |= 1u << 16;
            continue;
        }
        int st = S.attr[a].stream;
        va[nva++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, (uint32_t)loc, (uint32_t)st, f, S.attr[a].offset};
        used |= 1u << st;
    }
    for (uint32_t st = 0; st < 17; st++) if (used & (1u << st)) {
        if (st == 16) vb[nvb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, 16, 0, VK_VERTEX_INPUT_RATE_VERTEX, 1};
        else {
            uint32_t stride = S.stream[st].stride, div = S.stream[st].divisor;
            vb[nvb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, st, stride,
                                                              div && stride ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, div && stride ? div : 1};
        }
    }
    if (!C.valid || C.nvb != nvb || C.nva != nva || memcmp(C.vb, vb, sizeof vb[0] * nvb) || memcmp(C.va, va, sizeof va[0] * nva)) {
        V.CmdSetVertexInputEXT(g_cb, nvb, vb, nva, va);
        C.nvb = nvb; C.nva = nva; memcpy(C.vb, vb, sizeof vb[0] * nvb); memcpy(C.va, va, sizeof va[0] * nva);
    }
    /* buffers */
    for (uint32_t st = 0; st < 17; st++) if (used & (1u << st)) {
        VkBuffer b = g_zero.buf; VkDeviceSize off = 0;
        if (st < 16) {
            VkBuffer pb; VkDeviceSize po;
            if (S.vbuf[st].address && vk_pool_lookup(S.vbuf[st].address, 1, &pb, &po, NULL)) { b = pb; off = po; }   /* its pending updates: draw, before the pass */
        }
        if (C.vbuf[st] != b || C.voff[st] != off || !C.valid) { vkCmdBindVertexBuffers(g_cb, st, 1, &b, &off); C.vbuf[st] = b; C.voff[st] = off; }
    }
}

/* ---------------- draws ---------------- */
static vk_shader *g_last_fs_warned;
static void query_need(void) {   /* the draw counts samples: an active occlusion query in this pass */
    if (!g_counting || P.query_open) return;
    slot_t *s = &g_slots[g_cs];
    if (s->nq >= NQUERY) { VK_ONCE(qfull, "occlusion queries exhausted in one submission"); return; }
    P.query = s->nq++; P.query_open = 1;
    vkCmdBeginQuery(g_cb, s->qpool, P.query, VK_QUERY_CONTROL_PRECISE_BIT);
    if (g_nq == g_capq) { g_capq = g_capq ? g_capq * 2 : 64; g_qlist = realloc(g_qlist, sizeof *g_qlist * (size_t)g_capq); if (!g_qlist) VDIE("query list"); }
    g_qlist[g_nq++] = P.query;
}
/* TOTK_VK_TRACE_FS=<key prefix>[,<prefix>...] (diagnostics): the first 12 draws of the listed fragment shaders (with
   TOTK_VK_PASSDUMP: those of the dumped frame) log what their two stages are given: every uniform buffer (slot, address,
   size, its first 8 floats with the pending inline updates) and every sampled texture (slot, serial, format, whether a
   real view resolved); with TOTK_VK_PASSDUMP each sampled texture is also written there, with its raw value ranges. */
static void trace_stage(vk_shader *sh, int st, const char *tag) {
    if (!sh) return;
    for (int i = 0; i < sh->nb; i++) {
        const vk_bind *b = &sh->b[i];
        if (b->type == VB_UBO && b->kind == NVN_RB_CBUF && b->reg < 16) {
            const p_bind_buf *u = &S.ubo[st][b->reg]; float f[8] = {0};
            if (u->addr && u->size >= 32) pend_read(u->addr, f, sizeof f);
            VLOG("[trace]   %s ubo slot %d (bank %d) %#llx+%llu: %g %g %g %g | %g %g %g %g", tag, b->reg, b->bank, (unsigned long long)u->addr, (unsigned long long)u->size, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
        } else if (b->type == VB_UBO) VLOG("[trace]   %s ubo kind %d bank %d (driver / support / c1)", tag, b->kind, b->bank);
        else if (b->type == VB_COMBINED || b->type == VB_SAMPLED) {
            uint64_t h = b->reg < 128 ? S.tex[st][b->reg] : 0; int tid = (int)(h & 0xfffff), fmt = 0; vk_image *im = NULL;
            const tex_t *t = tid ? vk_tex_slot(tid) : NULL;
            VkImageView v = tid ? vk_tex_by_id(tid, (VkImageViewType)b->view_type, &im, &fmt) : VK_NULL_HANDLE;
            VLOG("[trace]   %s texture slot %d tid %d serial %llu %ux%u format %#x target %d: %s%s (sampler %llu%s)", tag, b->reg, tid, t ? (unsigned long long)t->serial : 0ull,
                 t ? t->d.width : 0, t ? t->d.height : 0, t ? t->d.format : 0, t ? t->d.target : 0, v ? "view" : "NO VIEW (dummy)", im && (im->depth || im->stencil) ? ", depth image" : "",
                 (unsigned long long)(h >> 20), b->shadow ? ", compare" : "");
            if (v && im && g_pd_dir && !(im->depth || im->stencil)) {   /* the input's picture and its raw values (no pass is open here) */
                char path[640], what[160]; snprintf(what, sizeof what, "draw %llu %s slot %d serial %llu", (unsigned long long)g_stat_draw + 1, tag, b->reg, t ? (unsigned long long)t->serial : 0ull);
                snprintf(path, sizeof path, "%s/draw%llu-%s-t%d-s%llu.ppm", g_pd_dir, (unsigned long long)g_stat_draw + 1, tag, b->reg, t ? (unsigned long long)t->serial : 0ull);
                pd_shoot(im, 0, 0, path); pd_raw_stats(im, 0, 0, what);
            }
        }
    }
}
static int trace_match(const vk_shader *fs) {
    static const char *list; static int init; if (!init) { init = 1; list = getenv("TOTK_VK_TRACE_FS"); }
    if (!list || !fs) return 0;
    if (getenv("TOTK_VK_PASSDUMP") && !pd_on()) return 0;   /* with a pass dump: the draws of the dumped frame */
    for (const char *p = list; *p;) { size_t n = strcspn(p, ","); if (n && !strncmp(fs->key, p, n)) return 1; p += n + (p[n] == ','); }
    return 0;
}
static void draw(const p_draw *dw, const p_draw_idx *di) {
    if (!S.prog) { VK_ONCE(noprog, "draw without a program"); g_draw_skip[0]++; return; }
    if (S.render_disabled || S.raster_discard) { g_draw_skip[S.render_disabled ? 3 : 4]++; return; }
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) { g_draw_skip[5]++; return; }
    vk_prog *pr = vk_prog_get(S.prog);
    if (!pr || !pr->sh[0]) { g_draw_skip[0]++; return; }
    vk_prog_fit(pr);
    vk_shader *vs = pr->sh[0], *fs = pr->sh[1];
    if (trace_match(fs)) {
        static int traced; if (traced++ < 12) {
            rp_end();   /* the input dumps copy outside a pass; the draw opens it again */
            VLOG("[trace] draw %llu (render targets: %d colour, depth %llu): vs %.20s fs %.20s, depth test %d write %d", (unsigned long long)g_stat_draw + 1, R.rt.n,
                 R.rt.depth ? (unsigned long long)R.rt.depth->serial : 0ull, vs->key, fs->key, S.ds.dt, S.ds.dw);
            trace_stage(vs, 0, "vs"); trace_stage(fs, 1, "fs");
        }
    }
    if (fs && (fs->in_locs & ~vs->out_locs) && fs != g_last_fs_warned) {
        g_last_fs_warned = fs; VK_ONCE(vary, "fragment shader %s reads varyings %#x its vertex shader %s does not write (undefined values, as on the Switch)", fs->key, fs->in_locs & ~vs->out_locs, vs->key);
    }
    int prim = dw ? dw->prim : di->prim;
    VkPrimitiveTopology topo; int restart = 0, lines = 0;
    switch (prim) {
        case 0: topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case 1: topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; lines = 1; break;
        case 3: topo = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; restart = 1; lines = 1; break;
        case 4: case 7: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        case 5: case 8: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; restart = 1; break;
        case 6: case 9: topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; restart = 1; break;
        default: VK_ONCE(prim, "primitive type %d not implemented", prim); g_draw_skip[11]++; return;
    }
    if (prim == 7 && di) { VK_ONCE(qidx, "indexed quads not implemented"); return; }
    /* textures / uniforms first: they may upload, copy or end the pass */
    dwrites dv, df;
    resolve(vs, 0, &dv);
    if (fs) resolve(fs, 1, &df);
    /* the vertex and index bytes too: inline updates pending there go to memory now (upd_flush ends the pass), not after
       the pass is opened for the draw */
    if (g_nunfl) {
        for (int v = 0; v < vs->rf.nv; v++) {
            int a = vs->rf.v[v].nvn, st = a < 16 && a < S.nattr ? S.attr[a].stream : 16;
            if (st < 16 && S.vbuf[st].address) upd_guard_w(S.vbuf[st].address, S.vbuf[st].address + (S.vbuf[st].size ? S.vbuf[st].size : 1), "upd:vertex");
        }
        if (di) upd_guard_w(di->ibuf, di->ibuf + (uint64_t)di->count * (di->itype == 0 ? 1u : di->itype == 1 ? 2u : 4u), "upd:index");
    }
    /* The draw samples an attachment of its own pass (nvn_mtl.m att_unstored_read). The Switch renders straight to memory,
       so the sample sees what the pass drew before it; here the pass's pixels stay in tile memory until it ends, and the
       sample sees what memory held when the pass began. That is what the draw needs only while nothing of the pass has
       reached the attachment yet: when the pass cleared it, or a draw that samples no attachment drew it (P.plain), the
       pass is stored first. A run of sampling draws shares one pass (Breath of the Wild's deferred lights sample the
       G-buffer they are bound with, write-masked: the game orders them against each other no more than the Switch does).
       A clear still pending for the next pass on a texture the draw samples lands first. TOTK_VK_SPLIT_ALL=1: split at
       every such draw (400 one-draw passes a frame); TOTK_ATT_STORE=0: cleared attachments only. */
    static int split_all = -1; if (split_all < 0) split_all = getenv("TOTK_VK_SPLIT_ALL") != NULL;
    if (!P.open && att_store_on() && pending_clear_sampled(&dv, fs ? &df : NULL)) RP_FLUSH("clear_read");
    if (!P.open && g_ncarry && carry_sampled(&dv, fs ? &df : NULL)) carry_land_all();   /* lazy clears: before the draw reads */
    uint32_t fb = att_feedback(&dv) | (fs ? att_feedback(&df) : 0);
    /* a sampled depth / stencil image reads its depth (bit 8), or its stencil when it has no depth: stencil-only writes
       (light volumes marking the stencil) do not end the pass for draws that read depth */
    uint32_t touched = P.cleared | (att_store_on() ? P.plain : 0);
    uint32_t unstored = (touched & 0x1ffu) | (!P.hd && (touched & 0x200u) ? 1u << 8 : 0);
    if (P.open && (split_all ? fb != 0 : (fb & unstored) != 0)) RP_END("feedback");   /* store it first */
    if (P.open && g_lay.active) {   /* a layered pass: the draw must write its layer, and sample none of the layers drawn */
        int ok = lay_draw_ok(vs);
        if (!ok || (g_lay.merged && fb)) { if (ok) lay_refuse_level(); RP_END(ok ? "layer_read" : "layer_draw"); g_lay.refuse = 1; g_lay.refused++; }
    }
    if (P.open && P.noatt) { uint32_t w, h; noatt_dims(&w, &h); if (w > P.w || h > P.h) RP_END("noatt_size"); }
    if (P.open && !P.dimg && R.rt.depth && draw_uses_ds() && depth_lazy()) RP_END("depth_late");   /* opened without its depth buffer */
    if (!P.open && chunk_due()) seg_split();
    if (!P.open) { g_fd_vs = vs; int ok = rp_begin(0); g_fd_vs = NULL; if (!ok) { g_draw_skip[8]++; return; } }
    uint32_t samples_att = att_feedback(&dv) | (fs ? att_feedback(&df) : 0);   /* of the pass it draws in */
    VkRect2D sc; if (!clamp_scissor(&sc)) { g_draw_skip[10]++; return; }
    if (!C.valid) cb_state_defaults();
    /* viewport as Metal sees it: y flipped by a negative height (the translated shaders write Metal's clip space) */
    double vx = S.has_vp ? S.vp[0][0] : 0, vw = S.has_vp ? S.vp[0][2] : P.w, vy = S.has_vp ? S.vp[0][1] : 0, vh = S.has_vp ? S.vp[0][3] : P.h;
    double ux, uy; uw_ui_squeeze(dw, di, &ux, &uy);   /* key 4: UI squeezed about the canvas centre */
    /* A UI draw over the whole canvas is squeezed per vertex, so what lies on the screen's edge (a full-screen fade or
       backdrop) still reaches the window's (vk_shader.c spv_squeeze_patch); other viewports, and vertex shaders the patch
       cannot take, through the viewport as before. TOTK_UI_EDGES=0: through the viewport always. */
    static int sq_edges = -1; if (sq_edges < 0) { const char *e = getenv("TOTK_UI_EDGES"); sq_edges = !(e && !strcmp(e, "0")); }
    VkShaderEXT sq = (g_sq_test_edges >= 0 ? g_sq_test_edges : sq_edges) && (ux != 1 || uy != 1) && !g_lay.active && fabs(vx) < 0.5 && fabs(vy) < 0.5 && fabs(vw - P.w) < 0.5 && fabs(vh - P.h) < 0.5 ? vk_shader_ui_squeezed(vs) : VK_NULL_HANDLE;
    bind_gfx_shaders(g_lay.active ? vk_shader_layered(vs) : sq ? sq : vs->obj, fs ? fs->obj : VK_NULL_HANDLE);
    if (sq) {
        float f[2] = {(float)(1 / ux), (float)(1 / uy)};
        if (!C.psq_ok || memcmp(C.psq, f, sizeof f)) { vkCmdPushConstants(g_cb, V.gfx_layout, vk_gfx_pc.stageFlags, 8, sizeof f, f); memcpy(C.psq, f, sizeof f); C.psq_ok = 1; }
    }
    { double cx = P.w / 2.0, cy = P.h / 2.0;
      if (ux != 1 && !sq) { vx = cx + (vx - cx) / ux; vw /= ux; }
      if (uy != 1 && !sq) { vy = cy + (vy - cy) / uy; vh /= uy; }
      if (ux == 1 && uy == 1) { if (uw_on_canvas()) g_uw.scene = 1; }
      else if (!g_uw.scene && !g_uw.edged && P.cview[0]) {
          /* UI with no scene under it this frame (the loading screens, a fade): its full-screen backdrop is squeezed
             too, so the canvas outside the middle is cleared to black instead of keeping an old frame (a flicker) */
          VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {.float32 = {0, 0, 0, 1}}}};
          VkClearRect cr[2]; uint32_t n = 0;
          if (ux != 1) {
              uint32_t e = (uint32_t)(cx - cx / ux);
              if (e) { cr[n++] = (VkClearRect){{{0, 0}, {e, P.h}}, 0, P.layers}; cr[n++] = (VkClearRect){{{(int32_t)(P.w - e), 0}, {e, P.h}}, 0, P.layers}; }
          } else {
              uint32_t e = (uint32_t)(cy - cy / uy);
              if (e) { cr[n++] = (VkClearRect){{{0, 0}, {P.w, e}}, 0, P.layers}; cr[n++] = (VkClearRect){{{0, (int32_t)(P.h - e)}, {P.w, e}}, 0, P.layers}; }
          }
          if (n) vkCmdClearAttachments(g_cb, 1, &ca, n, cr);
          g_uw.edged = 1;
      } }
    float n0 = S.has_vp ? S.dr[0] : 0, f0 = S.has_vp ? S.dr[1] : 1;
    n0 = n0 < 0 ? 0 : n0 > 1 ? 1 : n0; f0 = f0 < 0 ? 0 : f0 > 1 ? 1 : f0;
    VkViewport vp = {(float)vx, (float)(vy + vh), (float)vw, (float)-vh, n0, f0};
    if (vw == 0 || vh == 0) { g_draw_skip[10]++; return; }
    if (!C.valid || memcmp(&C.vp, &vp, sizeof vp)) { vkCmdSetViewportWithCount(g_cb, 1, &vp); C.vp = vp; }
    if (!C.valid || memcmp(&C.sc, &sc, sizeof sc)) { vkCmdSetScissorWithCount(g_cb, 1, &sc); C.sc = sc; }
    set_raster_state(lines, topo, restart);
    set_ds_state();
    set_blend_state();
    set_ms_state();
    C.valid = 1;
    set_vertex_input(vs);
    if (C.set_hash[0] != dv.hash || !C.set[0]) { C.set[0] = set_write(&dv); C.set_hash[0] = dv.hash; vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.gfx_layout, 0, 1, &C.set[0], 0, NULL); }
    if (fs && (C.set_hash[1] != df.hash || !C.set[1])) { C.set[1] = set_write(&df); C.set_hash[1] = df.hash; vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.gfx_layout, 1, 1, &C.set[1], 0, NULL); }
    if (!fs && !C.set[1]) { C.set[1] = set_write(&(dwrites){0}); C.set_hash[1] = 0; vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, V.gfx_layout, 1, 1, &C.set[1], 0, NULL); }
    query_need();
    P.draws++; g_stat_draw++;
    if (g_lay.active) {   /* the layer this draw writes, within the layered pass */
        uint32_t l = (uint32_t)(g_lay.cur - g_lay.base);
        if (!C.pcl_ok || C.pcl != l) { vkCmdPushConstants(g_cb, V.gfx_layout, vk_gfx_pc.stageFlags, 0, 4, &l); C.pcl = l; C.pcl_ok = 1; }
    }
    {   /* what the draw writes, for the draws after it that sample attachments (the feedback rule above) */
        uint32_t wr = 0;
        for (int i = 0; i < P.n && i < 8; i++) if (P.cview[i] && (S.chmask.mask >> (4 * i) & 0xf)) wr |= 1u << i;
        if (P.dimg && P.hd && S.ds.dw) wr |= 1u << 8;
        if (P.dimg && P.hs && S.ds.st && (S.st_wmask[0] | S.st_wmask[1])) wr |= 1u << 9;
        if (dv.simg || (fs && df.simg)) wr |= 0x3ffu;   /* image stores may land on any of them */
        P.wrote |= wr;
        if (!samples_att) P.plain |= P.wrote;
    }

    if (fs && (!P.npdk || g_pt_state == 1 || pd_on())) { int k = 0; while (k < P.npdk && strncmp(P.pdk[k], fs->key, 12)) k++; if (k == P.npdk && k < 8) { memcpy(P.pdk[k], fs->key, 12); P.pdk[k][12] = 0; P.npdk++; } }
    if (dw && prim == 7) {
        if (dw->count >= 4) {
            if (C.ibuf != g_quad.buf || C.ioff || C.itype != VK_INDEX_TYPE_UINT32) { vkCmdBindIndexBuffer(g_cb, g_quad.buf, 0, VK_INDEX_TYPE_UINT32); C.ibuf = g_quad.buf; C.ioff = 0; C.itype = VK_INDEX_TYPE_UINT32; }
            vkCmdDrawIndexed(g_cb, (uint32_t)(dw->count / 4) * 6, dw->instances > 1 ? (uint32_t)dw->instances : 1, 0, dw->first, (uint32_t)dw->base_instance);
        }
    } else if (dw) {
        if (dw->count && dw->instances >= 0) vkCmdDraw(g_cb, (uint32_t)dw->count, dw->instances > 1 ? (uint32_t)dw->instances : 1, (uint32_t)dw->first, (uint32_t)dw->base_instance);
    } else {
        VkBuffer b; VkDeviceSize off;
        if (!vk_pool_lookup(di->ibuf, 1, &b, &off, NULL)) { VK_ONCE(ib, "index buffer not in a memory pool"); return; }
        VkIndexType it = di->itype == 0 ? VK_INDEX_TYPE_UINT8 : di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        uint32_t isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
        if (off % isz) { VK_ONCE(ibal, "index buffer offset not aligned to its index size"); return; }
        if (C.ibuf != b || C.ioff != off || C.itype != it) { vkCmdBindIndexBuffer(g_cb, b, off, it); C.ibuf = b; C.ioff = off; C.itype = it; }
        if (di->count) vkCmdDrawIndexed(g_cb, (uint32_t)di->count, di->instances ? (uint32_t)di->instances : 1, 0, di->base_vertex, (uint32_t)di->base_instance);
    }
}

/* ---------------- clears ---------------- */
static void builtin_draw_begin(void) {   /* the game's shaders and sets are bound again by the next draw */
    C.vs = C.fs = VK_NULL_HANDLE; C.set_hash[0] = C.set_hash[1] = 1; C.set[0] = C.set[1] = VK_NULL_HANDLE;
    C.valid = 0; C.pcl_ok = 0; C.psq_ok = 0;   /* its push constants replace the layer and the UI squeeze */
    if (!P.open) return;
    cb_state_defaults();
    VkVertexInputBindingDescription2EXT vb = {VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT};
    V.CmdSetVertexInputEXT(g_cb, 0, &vb, 0, NULL);
    VkViewport vp = {0, 0, (float)P.w, (float)P.h, 0, 1};
    vkCmdSetViewportWithCount(g_cb, 1, &vp);
    vkCmdSetCullMode(g_cb, VK_CULL_MODE_NONE); vkCmdSetFrontFace(g_cb, VK_FRONT_FACE_COUNTER_CLOCKWISE); V.CmdSetPolygonModeEXT(g_cb, VK_POLYGON_MODE_FILL);
    vkCmdSetDepthBiasEnable(g_cb, VK_FALSE); V.CmdSetDepthClampEnableEXT(g_cb, VK_FALSE);
    vkCmdSetPrimitiveTopology(g_cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST); vkCmdSetPrimitiveRestartEnable(g_cb, VK_FALSE);
    VkSampleCountFlagBits s = P.samples ? P.samples : VK_SAMPLE_COUNT_1_BIT; V.CmdSetRasterizationSamplesEXT(g_cb, s); VkSampleMask m[2] = {~0u, ~0u}; V.CmdSetSampleMaskEXT(g_cb, s, m);
    V.CmdSetAlphaToCoverageEnableEXT(g_cb, VK_FALSE); if (V.CmdSetAlphaToOneEnableEXT) V.CmdSetAlphaToOneEnableEXT(g_cb, VK_FALSE);
    if (V.CmdSetLogicOpEnableEXT) V.CmdSetLogicOpEnableEXT(g_cb, VK_FALSE);
    C.nblend = UINT32_MAX; C.samples = 0;
}
static void clear_color_op(const p_clear_color *c0) {
    p_clear_color cc = *c0; const p_clear_color *c = &cc;
    if (S.render_disabled) { g_draw_skip[12]++; return; }
    if (c->index < 0 || c->index >= R.rt.n || c->index >= 8 || !R.rt.color[c->index]) { VK_ONCE(clr, "clear of colour target %d which is not set", c->index); return; }
    { int f = rt_dup_of(&R.rt, c->index); if (f >= 0) cc.index = f; }
    if (P.open && g_lay.active && g_lay.merged) RP_END("layer_clear");   /* the clear goes to a pass that opens on the layer it clears */
    if (!P.open && (c->mask & 0xf) == 0xf && scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index])) {
        R.clear_c[c->index] = 1; memcpy(R.cc[c->index], c->rgba, 16); return;
    }
    if (!(c->mask & 0xf)) return;
    if (!P.open && !rp_begin(0)) return;
    if (!P.cview[c->index]) { VK_ONCE(clr2, "clear of colour target %d not in the pass", c->index); return; }
    VkRect2D sc; if (!clamp_scissor(&sc)) return;
    query_end();
    if ((c->mask & 0xf) == 0xf) {
        VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {.color = clear_value(P.cint[c->index], c->rgba)}};
        VkClearRect cr = {sc, 0, g_lay.active ? 1 : P.layers};   /* a layered pass: the layer drawn (its first, see clear_color_op) */
        vkCmdClearAttachments(g_cb, 1, &ca, 1, &cr);
        return;
    }
    /* channel-masked: a triangle with the write mask */
    builtin_draw_begin();
    VkShaderStageFlagBits st[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkShaderEXT sh[2] = {g_clear_vs, g_clear_fs[P.cint[c->index]][c->index]};
    V.CmdBindShadersEXT(g_cb, 2, st, sh);
    vkCmdSetScissorWithCount(g_cb, 1, &sc);
    vkCmdSetDepthTestEnable(g_cb, VK_FALSE); vkCmdSetDepthWriteEnable(g_cb, VK_FALSE); vkCmdSetStencilTestEnable(g_cb, VK_FALSE);
    vkCmdSetDepthCompareOp(g_cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilOp(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilCompareMask(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff); vkCmdSetStencilWriteMask(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0); vkCmdSetStencilReference(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    uint32_t n = (uint32_t)P.n; VkBool32 en[8] = {0}; VkColorBlendEquationEXT eq[8]; VkColorComponentFlags wm[8] = {0};
    for (uint32_t i = 0; i < 8; i++) eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
    wm[c->index] = (c->mask & 1 ? VK_COLOR_COMPONENT_R_BIT : 0) | (c->mask & 2 ? VK_COLOR_COMPONENT_G_BIT : 0) | (c->mask & 4 ? VK_COLOR_COMPONENT_B_BIT : 0) | (c->mask & 8 ? VK_COLOR_COMPONENT_A_BIT : 0);
    V.CmdSetColorBlendEnableEXT(g_cb, 0, n, en); V.CmdSetColorBlendEquationEXT(g_cb, 0, n, eq); V.CmdSetColorWriteMaskEXT(g_cb, 0, n, wm);
    vkCmdSetBlendConstants(g_cb, (float[4]){0, 0, 0, 0});
    float pc[8] = {c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3], 0, 0, 0, 0};
    vkCmdPushConstants(g_cb, g_builtin_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(g_cb, 3, 1, 0, 0);
}
static void clear_ds_op(const p_clear_ds *c) {
    if (S.render_disabled) { g_draw_skip[12]++; return; }
    if (!R.rt.depth || (!c->depth_mask && !(c->stencil_mask & 255))) return;
    int smask = c->stencil_mask & 255;
    if (!P.open && (!smask || smask == 255) && scissor_full(R.rt.depth, &R.rt.dview, R.rt.has_dview)) {
        if (c->depth_mask) { R.clear_d = 1; R.cd = c->depth; }
        if (smask) { R.clear_s = 1; R.cs = (uint32_t)c->stencil; }
        return;
    }
    if (!P.open && !rp_begin(0)) return;
    if (!P.dview) { VK_ONCE(clrds, "depth / stencil clear without a depth attachment in the pass"); return; }
    VkRect2D sc; if (!clamp_scissor(&sc)) return;
    query_end();
    int dw = c->depth_mask && P.hd, sw = smask && P.hs;
    if (!dw && !sw) return;
    if (!sw || smask == 255) {
        VkClearAttachment ca = {(dw ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (sw ? VK_IMAGE_ASPECT_STENCIL_BIT : 0), 0,
                                {.depthStencil = {(float)(P.dfmt == VK_FORMAT_D16_UNORM ? nvn_depth16_clear_value(c->depth) : c->depth), (uint32_t)c->stencil}}};
        VkClearRect cr = {sc, 0, g_lay.active ? 1 : P.layers};   /* a layered pass: the layer drawn (its first, see clear_color_op) */
        vkCmdClearAttachments(g_cb, 1, &ca, 1, &cr);
        return;
    }
    /* stencil bits masked: a triangle that replaces them */
    builtin_draw_begin();
    VkShaderStageFlagBits st[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    VkShaderEXT sh[2] = {g_clear_vs, VK_NULL_HANDLE};
    V.CmdBindShadersEXT(g_cb, 2, st, sh);
    vkCmdSetScissorWithCount(g_cb, 1, &sc);
    vkCmdSetDepthTestEnable(g_cb, dw); vkCmdSetDepthWriteEnable(g_cb, dw); vkCmdSetDepthCompareOp(g_cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilTestEnable(g_cb, VK_TRUE);
    vkCmdSetStencilOp(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilCompareMask(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff); vkCmdSetStencilWriteMask(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, (uint32_t)smask);
    vkCmdSetStencilReference(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, (uint32_t)c->stencil);
    uint32_t n = (uint32_t)P.n;
    if (n) { VkBool32 en[8] = {0}; VkColorBlendEquationEXT eq[8]; VkColorComponentFlags wm[8] = {0};
             for (uint32_t i = 0; i < 8; i++) eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
             V.CmdSetColorBlendEnableEXT(g_cb, 0, n, en); V.CmdSetColorBlendEquationEXT(g_cb, 0, n, eq); V.CmdSetColorWriteMaskEXT(g_cb, 0, n, wm); }
    vkCmdSetBlendConstants(g_cb, (float[4]){0, 0, 0, 0});
    float depth = c->depth < 0 ? 0 : c->depth > 1 ? 1 : c->depth;
    float pc[8] = {0, 0, 0, 0, depth, 0, 0, 0};
    vkCmdPushConstants(g_cb, g_builtin_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    vkCmdDraw(g_cb, 3, 1, 0, 0);
}

/* ---------------- copies and texture clears ---------------- */
static VkImageAspectFlags img_aspect(const vk_image *im) { return im->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT; }
static void copy_b2t(const p_copy_b2t *c) {
    vk_image *im = vk_tex_image(c->tex); if (!im) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb, rows = (c->r.height + bh - 1) / bh, img = row * rows;
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    upd_guard_w(c->src, c->src + img * nz, "upd:copy_b2t");
    VkBuffer b; VkDeviceSize off;
    if (!vk_pool_lookup(c->src, img * nz, &b, &off, NULL)) { VK_ONCE(b2t, "buffer-to-texture source not in a memory pool"); return; }
    if (row % bpb) { VK_ONCE(b2trow, "buffer-to-texture row stride %zu not a multiple of the block size", row); return; }
    if (clear_pending_on(im)) RP_FLUSH("clear_hazard");   /* a clear of the texture waiting for a pass lands first */
    RP_END("copy_b2t");
    VkBufferImageCopy rg = {.bufferOffset = off, .bufferRowLength = (uint32_t)(row / bpb * bw), .bufferImageHeight = (uint32_t)(rows * bh),
        .imageSubresource = {img_aspect(im), level, is3d ? 0 : layer0 + c->r.zoffset, is3d ? 1 : (uint32_t)nz},
        .imageOffset = {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, .imageExtent = {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    vkCmdCopyBufferToImage(g_cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    vk_barrier_all(g_cb);
    vk_tex_mark_owned(c->tex); vk_tex_note_gpu_write(c->tex);
}
static void copy_t2b(const p_copy_b2t *c) {
    vk_image *im = vk_tex_image(c->tex); if (!im) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + bw - 1) / bw) * bpb, rows = (c->r.height + bh - 1) / bh, img = row * rows;
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    upd_guard_w(c->src, c->src + img * nz, "upd:copy_t2b");
    VkBuffer b; VkDeviceSize off;
    if (!vk_pool_lookup(c->src, img * nz, &b, &off, NULL)) { VK_ONCE(t2b, "texture-to-buffer destination not in a memory pool"); return; }
    if (row % bpb) { VK_ONCE(t2brow, "texture-to-buffer row stride %zu not a multiple of the block size", row); return; }
    if (clear_pending_on(im)) RP_FLUSH("clear_hazard");
    RP_END("copy_t2b");
    VkBufferImageCopy rg = {.bufferOffset = off, .bufferRowLength = (uint32_t)(row / bpb * bw), .bufferImageHeight = (uint32_t)(rows * bh),
        .imageSubresource = {img_aspect(im), level, is3d ? 0 : layer0 + c->r.zoffset, is3d ? 1 : (uint32_t)nz},
        .imageOffset = {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, .imageExtent = {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    vkCmdCopyImageToBuffer(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &rg);
    vk_barrier_all(g_cb);
    gw_note(c->src, img * nz);
    vk_tex_wb_supersede((const void *)(uintptr_t)c->src, img * nz);
}
static void copy_t2t(const p_copy_t2t *c) {
    RP_FLUSH("copy_t2t");
    vk_image *s = vk_tex_image(c->src), *d = vk_tex_image(c->dst);
    if (!s || !d) return;
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    int s3 = c->src->d.target == 2, d3 = c->dst->d.target == 2, nz = c->sr.depth ? c->sr.depth : 1;
    int same = c->sr.width == c->dr.width && c->sr.height == c->dr.height && (c->sr.depth ? c->sr.depth : 1) == (c->dr.depth ? c->dr.depth : 1);
    if (same && s->format == d->format && s3 == d3) {
        VkImageCopy rg = {.srcSubresource = {img_aspect(s), sl, s3 ? 0 : sz0 + c->sr.zoffset, s3 ? 1 : (uint32_t)nz}, .srcOffset = {c->sr.xoffset, c->sr.yoffset, s3 ? c->sr.zoffset : 0},
                          .dstSubresource = {img_aspect(d), dl, d3 ? 0 : dz0 + c->dr.zoffset, d3 ? 1 : (uint32_t)nz}, .dstOffset = {c->dr.xoffset, c->dr.yoffset, d3 ? c->dr.zoffset : 0},
                          .extent = {(uint32_t)c->sr.width, (uint32_t)c->sr.height, s3 ? (uint32_t)nz : 1}};
        vkCmdCopyImage(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    } else if (!s3 && !d3 && !s->depth && !d->depth && !s->stencil && !d->stencil) {   /* scaled or converted: a blit (the 2D engine) */
        VkImageBlit rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, sl, sz0 + c->sr.zoffset, (uint32_t)nz},
                          .srcOffsets = {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + c->sr.height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, dl, dz0 + c->dr.zoffset, (uint32_t)nz},
                          .dstOffsets = {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
        vkCmdBlitImage(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg, (c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    } else { VK_ONCE(t2t, "texture copy form not implemented: %dx%dx%d -> %dx%dx%d, formats %d -> %d", c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth, s->format, d->format); return; }
    vk_barrier_all(g_cb);
    vk_tex_mark_owned(c->dst); vk_tex_note_gpu_write(c->dst);
}
static void clear_tex(const p_clear_tex *c) {
    vk_image *im = vk_tex_image(c->tex); if (!im) return;
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    int nz = c->r.depth ? c->r.depth : 1;
    if (clear_pending_on(im)) RP_FLUSH("clear_hazard");
    RP_END("clear_tex");
    if (c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width != w || (uint32_t)c->r.height != h || (c->mask & 0xf) != 0xf || im->type == VK_IMAGE_TYPE_3D) {
        VK_ONCE(clrtex, "texture clear form not implemented: region %d,%d %dx%d (level %ux%u) mask %#x", c->r.xoffset, c->r.yoffset, c->r.width, c->r.height, w, h, c->mask);
        return;
    }
    VkImageSubresourceRange r = {img_aspect(im) | (im->depth && im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0), lv, 1, l0 + c->r.zoffset, (uint32_t)nz};
    if (im->depth || im->stencil) {
        VkClearDepthStencilValue v = {im->format == VK_FORMAT_D16_UNORM ? (float)nvn_depth16_clear_value(c->c.f[0]) : c->c.f[0], im->depth ? c->c.u[1] : c->c.u[0]};
        vkCmdClearDepthStencilImage(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    } else {
        VkClearColorValue v; memcpy(&v, c->c.u, 16);
        int cls = clear_class(c->has_view && c->view.set_format ? c->view.format : c->tex->d.format);
        if (!c->is_uint && cls) for (int i = 0; i < 4; i++) { if (cls == 1) v.uint32[i] = (uint32_t)c->c.f[i]; else v.int32[i] = (int32_t)c->c.f[i]; }
        if (c->is_uint && !cls) for (int i = 0; i < 4; i++) v.float32[i] = (float)c->c.u[i];
        vkCmdClearColorImage(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
    }
    vk_barrier_all(g_cb);
    vk_tex_mark_owned(c->tex);
}
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    if (!n) return;
    upd_guard_w(src, src + n, "upd:copy_b2b"); upd_guard_w(dst, dst + n, "upd:copy_b2b");
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!vk_pool_lookup(src, n, &sb, &so, NULL) || !vk_pool_lookup(dst, n, &db, &dof, NULL))
        VDIE("buffer copy outside every memory pool src=%#llx dst=%#llx n=%#llx", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n);
    RP_END("mtl_sub_copy");
    VkBufferCopy c = {so, dof, n};
    vkCmdCopyBuffer(g_cb, sb, db, 1, &c);
    vk_barrier_all(g_cb);
    gw_note(dst, n);
}

/* ---------------- dispatch ---------------- */
static void dispatch(const p_dispatch *dp) {
    if (!S.prog) { VK_ONCE(nocs, "dispatch without a program"); return; }
    vk_prog *pr = vk_prog_get(S.prog);
    vk_shader *cs = pr ? pr->sh[5] : NULL;
    if (!cs) { VK_ONCE(nocs2, "dispatch of a program without a compute shader"); return; }
    if (!dp->x || !dp->y || !dp->z) return;
    RP_END("dispatch");
    if (pd_on()) VLOG("[passdump] F%llu dispatch %.12s %ux%ux%u", (unsigned long long)atomic_load(&g_cur_frame), cs->key, dp->x, dp->y, dp->z);
    g_dispatch_grid[0] = (uint32_t)dp->x; g_dispatch_grid[1] = (uint32_t)dp->y; g_dispatch_grid[2] = (uint32_t)dp->z;
    dwrites d; resolve(cs, 5, &d);
    {   /* a clear waiting for a pass on an image the dispatch reads or writes lands first */
        int hz = d.img_over && any_clear_pending();
        for (int k = 0; k < d.nimg && !hz; k++) hz = clear_pending_on(d.img[k]);
        if (hz) RP_FLUSH("clear_hazard");
    }
    VkShaderStageFlagBits st = VK_SHADER_STAGE_COMPUTE_BIT;
    if (C.cs != cs->obj) { V.CmdBindShadersEXT(g_cb, 1, &st, &cs->obj); C.cs = cs->obj; }
    if (C.cset_hash != d.hash || !C.cset) { C.cset = set_write(&d); C.cset_hash = d.hash; vkCmdBindDescriptorSets(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, V.cs_layout, 0, 1, &C.cset, 0, NULL); }
    { pl_ent *e = pl_next(); if (e) { memset(e, 0, sizeof *e); e->disp = 1; memcpy(e->fs, cs->key, 12); e->w = (uint16_t)dp->x; e->h = (uint16_t)dp->y; e->draws = (uint32_t)dp->z; } }
    int pt = pt_on() && g_pt_n % 2 == 0; if (pt) pt_mark(0, "dispatch %.12s %ux%ux%u", cs->key, dp->x, dp->y, dp->z);
    vkCmdDispatch(g_cb, (uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z);
    if (pt) pt_mark(1, NULL);
    vk_barrier_all(g_cb);
}

/* ---------------- CPU-visible effects after completion ---------------- */
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t *queries; int nq, skipped; } post_op;
static post_op *g_post; static int g_npost, g_cappost;
static void post(post_op o) {
    if (g_npost == g_cappost) { g_cappost = g_cappost ? g_cappost * 2 : 64; g_post = realloc(g_post, sizeof *g_post * (size_t)g_cappost); if (!g_post) VDIE("post ops"); }
    g_post[g_npost++] = o;
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, NULL, 0, 0}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { (void)type; post((post_op){2, NULL, 0, 0, addr, NULL, 0, 0}); }
static uint64_t gpu_ticks(void) { extern uint64_t mng_mach_absolute_time(void); return (uint64_t)((__uint128_t)mng_mach_absolute_time() * 12 / 625); }
typedef struct { post_op *ops; int n; VkQueryPool qp; uint64_t value; } post_batch;
/* counter reports pending in GPU order (cond_eval reads them before they reach memory) */
typedef struct { uint64_t addr, value, val; int known; } cpend_ent;
#define CPEND_N 4096
static cpend_ent g_cpend[CPEND_N]; static unsigned g_cpend_head; static pthread_mutex_t g_cpend_mu = PTHREAD_MUTEX_INITIALIZER;
static void cpend_push(uint64_t addr, uint64_t value, uint64_t val, int known) {
    pthread_mutex_lock(&g_cpend_mu); g_cpend[g_cpend_head % CPEND_N] = (cpend_ent){addr, value, val, known}; g_cpend_head++; pthread_mutex_unlock(&g_cpend_mu);
}
static void cpend_commit(uint64_t value) {   /* the reports of the submission just committed complete with it */
    pthread_mutex_lock(&g_cpend_mu);
    for (unsigned k = 0; k < CPEND_N; k++) if (g_cpend[k].addr && g_cpend[k].value == UINT64_MAX) g_cpend[k].value = value;
    pthread_mutex_unlock(&g_cpend_mu);
}
static void cpend_done(uint64_t value) {
    pthread_mutex_lock(&g_cpend_mu);
    for (unsigned k = 0; k < CPEND_N; k++) if (g_cpend[k].addr && g_cpend[k].value <= value) g_cpend[k].addr = 0;
    pthread_mutex_unlock(&g_cpend_mu);
}
static int cpend_lookup(uint64_t addr, uint64_t *val) {
    int r = -1; pthread_mutex_lock(&g_cpend_mu);
    for (unsigned k = 0; k < CPEND_N; k++) { const cpend_ent *e = &g_cpend[(g_cpend_head - 1 - k) % CPEND_N]; if (e->addr == addr) { r = e->known; *val = e->val; break; } }
    pthread_mutex_unlock(&g_cpend_mu); return r;
}
static void post_run(void *arg) {   /* completion thread */
    post_batch *b = arg;
    for (int i = 0; i < b->n; i++) {
        post_op *o = &b->ops[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else if (o->kind == 3) {
            uint64_t sum = 0;
            for (int q = 0; q < o->nq; q++) { uint64_t r = 0; if (vkGetQueryPoolResults(V.dev, b->qp, o->queries[q], 1, 8, &r, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) sum += r; }
            if (!sum && o->skipped) sum = 1;
            uint64_t rec[2] = {sum, gpu_ticks()}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);
            free(o->queries);
        } else { uint64_t rec[2] = {0, gpu_ticks()}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }
    }
    cpend_done(b->value);
    free(b->ops); free(b);
}
typedef struct { const void *op; uint64_t frame; int result; } cond_cache_ent;
static cond_cache_ent g_cond_cache[1024];
static int cond_eval(const p_render_cond *c, const void *op) {
    uint64_t frame = atomic_load(&g_cur_frame);
    cond_cache_ent *ce = &g_cond_cache[((uintptr_t)op >> 3) & 1023];
    if (ce->op == op && ce->frame == frame) return ce->result;
    uint64_t x = 0, y = 0; int kx = cpend_lookup(c->addr, &x), ky = cpend_lookup(c->addr + 16, &y), enabled = 1;
    if (kx == 0 || ky == 0) { enabled = 1; g_draw_skip[14]++; }
    else if (!vk_pool_contains(c->addr, 32)) { VK_ONCE(cond, "conditional render counters at %#llx are not in a memory pool", (unsigned long long)c->addr); }
    else {
        if (kx < 0) memcpy(&x, (const void *)(uintptr_t)c->addr, 8);
        if (ky < 0) memcpy(&y, (const void *)(uintptr_t)(c->addr + 16), 8);
        enabled = c->mode == 0 ? x == y : x != y;
        if (!enabled) g_draw_skip[13]++;
    }
    *ce = (cond_cache_ent){op, frame, enabled};
    return enabled;
}

/* ---------------- state ops ---------------- */
static void state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: R.sc = *(const p_scissor *)a; R.has_sc = 1; break;
        case OP_BIND_PROGRAM: S.prog = ((const p_bind_program *)a)->prog; break;
        case OP_BIND_STATE: {
            const p_bind_state *b = a; void *dst = NULL; size_t n = 0; blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); dst = &S.blend[bs.target & 7]; n = sizeof bs; break;
                case 1: dst = &S.color; n = sizeof S.color; break;
                case 2: dst = &S.chmask; n = sizeof S.chmask; break;
                case 3: dst = &S.ms; n = sizeof S.ms; break;
                case 4: dst = &S.poly; n = sizeof S.poly; break;
                case 5: dst = &S.ds; n = sizeof S.ds; break;
            }
            if (dst) memcpy(dst, b->bytes, n);
            break;
        }
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; memcpy(S.attr, b->st, 4 * (size_t)b->n); S.nattr = b->n; break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; memcpy(S.stream, b->st, 8 * (size_t)b->n); S.nstream = b->n; break; }
        case OP_BIND_VBUF: { const p_bind_vbuf *b = a; if (b->first >= 0 && b->first + b->n <= 16) memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * (size_t)b->n); break; }
        case OP_BIND_UBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16) S.ubo[b->stage][b->index] = *b; break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16) S.ssbo[b->stage][b->index] = *b; break; }
        case OP_BIND_TEX: { const p_bind_handles *b = a; if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 128) { memcpy(&S.tex[b->stage][b->first], b->h, 8 * (size_t)b->n); vk_tex_bind_gen[b->stage]++; } break; }
        case OP_BIND_SAMP: { const p_bind_handles *b = a; if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 128) memcpy(&S.samp[b->stage][b->first], b->h, 8 * (size_t)b->n); break; }
        case OP_BIND_IMG: { const p_bind_handles *b = a; if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 32) memcpy(&S.img[b->stage][b->first], b->h, 8 * (size_t)b->n); break; }
        case OP_STENCIL: { const p_stencil *b = a;
            for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = (uint32_t)b->value_mask; else if (b->which == 1) S.st_wmask[f] = (uint32_t)b->mask; else S.st_ref[f] = (uint32_t)b->ref; }
            break; }
        case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; break;
        case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n && b->first + i < 16; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_COPY_STRIDE: S.copy_stride = ((const p_copy_stride *)a)->stride; break;
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_flag *)a)->value; break;
        case OP_RENDER_ENABLE: S.render_disabled = !((const p_flag *)a)->value; break;
        case OP_RENDER_COND: S.render_disabled = !cond_eval(a, (const uint8_t *)a - 8); break;
        case OP_RASTER_DISCARD: S.raster_discard = ((const p_flag *)a)->value; break;
    }
}

/* ---------------- submissions ---------------- */
static int g_sub_open;
static VkCommandBuffer g_up;   /* the upload command buffer of the current segment (runs before its main one) */
VkCommandBuffer vk_upload_cb(void) {
    if (!g_sub_open) VDIE("upload outside a submission");
    if (!g_up_used) {
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK(vkBeginCommandBuffer(g_up, &bi));
        vk_barrier_all(g_up);
        g_up_used = 1;
    }
    return g_up;
}
/* Mid-submission commits (nvn_mtl.m mtl_sub_split): the GPU starts on the frame's first passes while the replay records
   the rest, where the whole frame used to wait for its last draw (GPU 22 ms after a replay of 18 ms). Only the command
   buffers end here; the submission's own end (events, write-back stages, the update log, the slot) stays at
   mtl_sub_commit. Between passes only. TOTK_VK_CHUNK_DRAWS (default 400; 0 off) and TOTK_VK_CHUNK_FIRST (150) draws. */
static int chunk_due(void) {
    static int every = -1, first;
    if (every < 0) { const char *e = getenv("TOTK_VK_CHUNK_DRAWS"), *f = getenv("TOTK_VK_CHUNK_FIRST"); every = e ? atoi(e) : 400; first = f ? atoi(f) : 150; if (first <= 0) first = every; }
    return every > 0 && g_seg + 1 < 16 && g_stat_draw - g_seg_mark >= (uint64_t)(g_seg ? every : first);
}
static void seg_split(void) {
    slot_t *s = &g_slots[g_cs];
    if (g_ts_on) vkCmdWriteTimestamp2(g_cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, s->tsq, 2 * (uint32_t)g_seg + 1);
    VK(vkEndCommandBuffer(g_cb));
    VkCommandBuffer cbs[2]; int n = 0;
    if (g_up_used) { vk_barrier_all(g_up); VK(vkEndCommandBuffer(g_up)); cbs[n++] = g_up; }
    cbs[n++] = g_cb;
    vk_submit_ex(cbs, n, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0);   /* the deferred releases wait for the submission's end */
    if (g_seg >= s->nseg) {
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = s->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VK(vkAllocateCommandBuffers(V.dev, &ai, &s->seg_cb[g_seg])); VK(vkAllocateCommandBuffers(V.dev, &ai, &s->seg_up[g_seg]));
        s->nseg = g_seg + 1;
    }
    g_cb = s->seg_cb[g_seg]; g_up = s->seg_up[g_seg]; g_up_used = 0; g_seg++;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK(vkBeginCommandBuffer(g_cb, &bi));
    if (g_ts_on) vkCmdWriteTimestamp2(g_cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, s->tsq, 2 * (uint32_t)g_seg);
    vk_barrier_all(g_cb);
    memset(&C, 0, sizeof C); C.nblend = UINT32_MAX; C.set_hash[0] = C.set_hash[1] = C.cset_hash = 1;   /* state does not carry over */
    g_seg_mark = g_stat_draw; g_seg_commits++;
}
void mtl_sub_begin(void) {
    mtl_init();
    if (g_sub_open) VDIE("nested submission");
    vk_sub_serial++;
    /* the oldest slot whose GPU work is done */
    int pick = -1; uint64_t done = vk_done_value(), oldest = UINT64_MAX;
    for (int i = 0; i < NSLOT; i++) if (g_slots[i].value <= done) { pick = i; break; }
    if (pick < 0) { for (int i = 0; i < NSLOT; i++) if (g_slots[i].value < oldest) { oldest = g_slots[i].value; pick = i; } vk_wait_value(oldest, UINT64_MAX); }
    slot_t *s = &g_slots[pick]; g_cs = pick;
    VK(vkResetCommandPool(V.dev, s->pool, 0));
    for (int i = 0; i < s->ndp; i++) VK(vkResetDescriptorPool(V.dev, s->dp[i], 0));
    s->curdp = 0;
    if (s->nq) { vkResetQueryPool(V.dev, s->qpool, 0, s->nq); s->nq = 0; }
    vkResetQueryPool(V.dev, s->tsq, 0, 32);
    g_cb = s->cb; g_up = s->up; g_up_used = 0; g_seg = 0; g_seg_mark = g_stat_draw;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK(vkBeginCommandBuffer(g_cb, &bi));
    g_ts_on = gpu_time_due(atomic_load(&g_cur_frame));
    if (g_ts_on) vkCmdWriteTimestamp2(g_cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, s->tsq, 0);
    vk_barrier_all(g_cb);
    memset(&C, 0, sizeof C); C.nblend = UINT32_MAX; C.set_hash[0] = C.set_hash[1] = C.cset_hash = 1;
    memset(&P, 0, sizeof P);
    g_npost = 0; g_nq = 0; g_counting = 0;
    g_exec_cur = NULL; g_ret_n = 0; g_lay.refuse = 0;
    g_sub_open = 1; t_replay = 1;
    static int inited; if (!inited) { inited = 1; state_reset(); }
}
void *mtl_sub_capture_ubos(const uint64_t *h, int n) { (void)h; (void)n; return NULL; }   /* the replay runs in the submit call: bindings read memory then */
void mtl_sub_snapshot_ubos_cap(const uint64_t *h, int n, void *cap) { (void)h; (void)n; (void)cap; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
uint64_t mtl_sub_commit(void) {
    if (!g_sub_open) VDIE("commit without a submission");
    RP_FLUSH("submit");
    { static uint64_t last; uint64_t f = atomic_load(&g_cur_frame); if (!last) last = f; if (f >= last + 600) { why_report(f - last); last = f; } }
    if (g_pt_state == 1 && atomic_load(&g_cur_frame) != g_pt_frame) { g_pt_state = 2; vk_on_done(UINT64_MAX, pt_report, NULL); }
    upd_flush();
    vk_tex_wb_stage(g_cb);
    if (g_ts_on) vkCmdWriteTimestamp2(g_cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, g_slots[g_cs].tsq, 2 * (uint32_t)g_seg + 1);
    VK(vkEndCommandBuffer(g_cb));
    VkCommandBuffer cbs[2]; int n = 0;
    if (g_up_used) { vk_barrier_all(g_up); VK(vkEndCommandBuffer(g_up)); cbs[n++] = g_up; }
    cbs[n++] = g_cb;
    uint64_t v = vk_submit_ex(cbs, n, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 1);
    g_slots[g_cs].value = v;
    g_last_fence = v;
    if (g_ts_on) { gpu_ts *g = malloc(sizeof *g); if (!g) VDIE("GPU timing"); g->frame = atomic_load(&g_cur_frame); g->q = g_slots[g_cs].tsq; g->nseg = g_seg + 1; gpu_tag_begin(g->frame); vk_on_done(v, gpu_tag_done, g); }
    vk_tex_wb_submitted(v);
    vk_arena_retire(v);
    gw_commit(v);
    upd_commit(v);
    cpend_commit(v);
    if (g_npost) {
        post_batch *b = calloc(1, sizeof *b); if (!b) VDIE("post batch");
        b->ops = malloc(sizeof *b->ops * (size_t)g_npost); if (!b->ops) VDIE("post ops");
        memcpy(b->ops, g_post, sizeof *b->ops * (size_t)g_npost); b->n = g_npost; b->qp = g_slots[g_cs].qpool; b->value = v;
        vk_on_done(v, post_run, b);
    }
    g_npost = 0;
    vk_tex_frame_sweep(v);
    g_sub_open = 0; t_replay = 0; g_cb = VK_NULL_HANDLE; g_exec_cur = NULL;
    return v;
}
uint64_t mtl_last_fence(void) { return g_last_fence; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { if (!fence) return 1; return vk_wait_value(fence, timeout_ns); }
void mtl_finish(void) {
    mtl_present_flush();
    uint64_t last = vk_last_value();
    if (last) vk_wait_value(last, UINT64_MAX);
}
void mtl_drain(void) { mtl_finish(); }

/* ---------------- the replay loop ---------------- */
void mtl_exec(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_) VDIE("corrupt command record at %p: op %u size %u (call depth %d)", (const void *)p, (unsigned)h.op, (unsigned)h.size, g_ret_n);
        const void *a = p + 8;
        g_exec_cur = p;
        switch (h.op) {
            case OP_END: return;
            case OP_BEGIN: break;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (g_ret_n < 16) g_ret_stack[g_ret_n] = p + h.size; g_ret_n++; mtl_exec((const uint8_t *)(uintptr_t)t); g_ret_n--; break; }
            case OP_SIGNAL_EVENT: { carry_land_all(); volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, (const uint8_t *)a + 8, 4); memcpy(&v, (const uint8_t *)a + 12, 4); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: {
                uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8);
                cpend_push(ad, UINT64_MAX, 0, t != NVN_COUNTER_TYPE_SAMPLES_PASSED);
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
                    query_end();
                    post_op q = {3, NULL, 0, 0, ad, NULL, g_nq, g_qskipped};
                    if (g_nq) { q.queries = malloc(sizeof *q.queries * (size_t)g_nq); if (!q.queries) VDIE("query list"); memcpy(q.queries, g_qlist, sizeof *q.queries * (size_t)g_nq); }
                    post(q);
                } else mtl_sub_counter(ad, (uint32_t)t);
                break;
            }
            case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) { query_end(); g_counting = 1; g_nq = 0; g_qskipped = 0; } break; }
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8); mtl_sub_copy(s_, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8); mtl_sub_update((const uint8_t *)a + 16, d, n); break; }
            case OP_SET_RT: {
                const p_set_rt *r = a;
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (P.open && g_lay.active && !pend && lay_continue(r)) break;   /* another layer of the layered pass */
                if (P.open && !pend && R.has_rt && rt_same(&R.rt, r)) { R.rt = *r; break; }   /* same targets: the pass continues */
                if (pend && !P.open && lazy_clear_on()) {   /* lazy clears: the pending clears wait for a pass that attaches their targets */
                    if (!(R.has_rt && rt_same(&R.rt, r))) carry_push();
                    R.rt = *r; R.has_rt = 1; break;
                }
                RP_FLUSH("targets"); R.rt = *r; R.has_rt = 1;
                break;
            }
            case OP_CLEAR_COLOR: clear_color_op(a); break;
            case OP_CLEAR_DS: clear_ds_op(a); break;
            case OP_DRAW: draw(a, NULL); break;
            case OP_DRAW_IDX: draw(NULL, a); break;
            case OP_SCISSOR: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
            case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_STENCIL:
            case OP_POLY_OFFSET: case OP_VIEWPORTS: case OP_DEPTH_RANGE: case OP_BLEND_COLOR: case OP_COPY_STRIDE: case OP_DEPTH_CLAMP:
            case OP_RENDER_ENABLE: case OP_RASTER_DISCARD: case OP_RENDER_COND:
                state_op(h.op, a); break;
            case OP_COPY_B2T: copy_b2t(a); break;
            case OP_COPY_T2B: copy_t2b(a); break;
            case OP_COPY_T2T: copy_t2t(a); break;
            case OP_CLEAR_TEX: clear_tex(a); break;
            case OP_DISPATCH: dispatch(a); break;
            case OP_BARRIER: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: break;
        }
        p += h.size;
    }
}
void mtl_set_frame(uint64_t f) { atomic_store(&g_cur_frame, f); }
uint64_t vk_cur_frame(void) { return atomic_load(&g_cur_frame); }
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = g_stat_pass; *draws = g_stat_draw; }
/* passes, layered passes, SetRenderTargets taken into them, layered passes ended at a draw (hle_nvn.c "渲染通道" line) */
void mtl_layer_stats(uint64_t v[4]) { v[0] = g_stat_pass; v[1] = g_lay.passes; v[2] = g_lay.merges; v[3] = g_lay.refused; }
void mtl_pin_refs(void (*mark)(const void *)) {
    if (!R.has_rt) return;
    for (int i = 0; i < R.rt.n && i < 8; i++) mark(R.rt.color[i]);
    mark(R.rt.depth);
}
void mtl_probe_pin_rt(const tex_t *t) { memset(&R.rt, 0, sizeof R.rt); R.has_rt = t != NULL; R.rt.n = t ? 1 : 0; R.rt.color[0] = t; }
