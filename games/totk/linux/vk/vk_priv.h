/* The NVN replay on Vulkan (Linux, Honeykrisp): the interface nvn_mtl.m gives hle_nvn.c (nvn_mtl.h, the mtl_* calls),
   implemented on one Vulkan device. Shared declarations of the backend's files:
     vk_device.c   instance / device / queue, the timeline and its completion thread, deferred destruction, arenas
     vk_pool.c     NVN memory pools as sparse buffers over the guest memory's own pages (udmabuf), address lookup
     vk_format.c   NVN formats, vertex formats, blend / compare / stencil enums
     vk_tex.c      textures (mirrored from pool memory or GPU-owned), views, the texture / sampler pools, texels
     vk_shader.c   the translation cache (variants, the translator service), SPIR-V binding remap, shader objects
     vk_exec.c     the command replay: render passes, draws, dispatches, copies, clears, uniforms, queries
     vk_present.c  the swapchain and the present thread
   Conventions follow the Metal backend so that the game sees the same behaviour (see the comments there for the why). */
#pragma once
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "nvn_obj.h"
#include "nvn_refl.h"
#include "nvn_tex.h"

void vk_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
_Noreturn void vk_die(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#define VLOG(...) vk_log(__VA_ARGS__)
#define VDIE(...) vk_die(__VA_ARGS__)
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) vk_die("%s failed: %d (%s:%d)", #x, r_, __FILE__, __LINE__); } while (0)
#define VK_ONCE(k, ...) do { static int once_##k; if (!once_##k++) vk_log(__VA_ARGS__); } while (0)

/* ---------------- device (vk_device.c) ---------------- */
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qfam;
    VmaAllocator vma;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties memp;
    VkSemaphore tl;                  /* timeline: value v reached = everything submitted with values <= v is done */
    pthread_mutex_t qmu;             /* queue submissions, sparse binds and presents */
    uint64_t tl_last;                /* last value given out (vk_device.c g_sqmu; vk_last_value) */
    _Atomic uint64_t tl_done;        /* highest value the completion thread has finished with */
    pthread_mutex_t done_mu; pthread_cond_t done_cv;
    int host_copy;                   /* VK_EXT_host_image_copy (Vulkan 1.4 core) usable for uploads */
    PFN_vkCreateShadersEXT CreateShadersEXT; PFN_vkDestroyShaderEXT DestroyShaderEXT; PFN_vkCmdBindShadersEXT CmdBindShadersEXT;
    PFN_vkCmdSetVertexInputEXT CmdSetVertexInputEXT;
    PFN_vkCmdSetPolygonModeEXT CmdSetPolygonModeEXT; PFN_vkCmdSetRasterizationSamplesEXT CmdSetRasterizationSamplesEXT;
    PFN_vkCmdSetSampleMaskEXT CmdSetSampleMaskEXT; PFN_vkCmdSetAlphaToCoverageEnableEXT CmdSetAlphaToCoverageEnableEXT;
    PFN_vkCmdSetAlphaToOneEnableEXT CmdSetAlphaToOneEnableEXT; PFN_vkCmdSetLogicOpEnableEXT CmdSetLogicOpEnableEXT;
    PFN_vkCmdSetLogicOpEXT CmdSetLogicOpEXT; PFN_vkCmdSetColorBlendEnableEXT CmdSetColorBlendEnableEXT;
    PFN_vkCmdSetColorBlendEquationEXT CmdSetColorBlendEquationEXT; PFN_vkCmdSetColorWriteMaskEXT CmdSetColorWriteMaskEXT;
    PFN_vkCmdSetDepthClampEnableEXT CmdSetDepthClampEnableEXT; PFN_vkCmdSetDepthClipEnableEXT CmdSetDepthClipEnableEXT;
    PFN_vkCmdSetProvokingVertexModeEXT CmdSetProvokingVertexModeEXT; PFN_vkCmdSetLineRasterizationModeEXT CmdSetLineRasterizationModeEXT;
    PFN_vkCmdSetTessellationDomainOriginEXT CmdSetTessellationDomainOriginEXT; PFN_vkCmdSetPatchControlPointsEXT CmdSetPatchControlPointsEXT;
    PFN_vkCmdSetDepthClipNegativeOneToOneEXT CmdSetDepthClipNegativeOneToOneEXT;
    PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR;
    PFN_vkGetShaderBinaryDataEXT GetShaderBinaryDataEXT;
    VkDescriptorSetLayout dsl;       /* the one descriptor set layout every stage uses (vk_exec.c layout) */
    VkPipelineLayout gfx_layout;     /* sets 0 (vertex) and 1 (fragment) */
    VkPipelineLayout cs_layout;      /* set 0 (compute) */
    int has_window;
    int custom_border;               /* VK_EXT_custom_border_color enabled (vk_device.c border_on) */
    int present_timing;              /* VK_EXT_present_timing and VK_KHR_present_id2 enabled: presentation times (vk_present.c) */
    int surface_caps2;               /* VK_KHR_get_surface_capabilities2 (instance) */
} vk_dev;
extern vk_dev V;

void vk_device_init(void);
uint64_t vk_submit(VkCommandBuffer const *cbs, int n, VkSemaphore wait, VkPipelineStageFlags2 wait_stage, VkSemaphore signal_bin);   /* returns the timeline value */
uint64_t vk_last_value(void);   /* the last timeline value given out (its submission may still be queued) */
uint64_t vk_submit_ex(VkCommandBuffer const *cbs, int n, VkSemaphore wait, VkPipelineStageFlags2 wait_stage, VkSemaphore signal_bin, int replay);
int vk_wait_value(uint64_t v, uint64_t timeout_ns);   /* 1 when the completion thread has finished value v */
uint64_t vk_done_value(void);
/* Completion callbacks: run on the completion thread once the timeline reaches value v (in order of v). */
void vk_on_done(uint64_t v, void (*fn)(void *), void *arg);
/* Deferred destruction once the timeline reaches v (the value of the last submission that may use the object). */
void vk_defer_image(uint64_t v, VkImage img, VmaAllocation a);
void vk_defer_view(uint64_t v, VkImageView view);
void vk_defer_buffer(uint64_t v, VkBuffer b, VmaAllocation a);
void vk_defer_bufview(uint64_t v, VkBufferView bv);
void vk_defer_shader(uint64_t v, VkShaderEXT s);
void vk_defer_free(uint64_t v, void *p);
uint64_t vk_pending_value(void);   /* the value the next submission of the replay will get or later: for deferral */

/* Host-visible arenas (per submission): uniform copies, driver banks, inline update data, staging. */
typedef struct { VkBuffer buf; VkDeviceSize off; uint8_t *cpu; VkDeviceAddress unused; } vk_slice;
vk_slice vk_arena_alloc(size_t n, size_t align);
void vk_arena_retire(uint64_t v);   /* the current arena chunks are released once v completes */

/* One-off command buffers outside the replay (texel reads / writes, write-back, the present). */
VkCommandBuffer vk_oneoff_begin(void);
uint64_t vk_oneoff_end_submit(VkCommandBuffer cb);
void vk_oneoff_wait(uint64_t v);

/* Generic helpers */
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want);
void vk_barrier_all(VkCommandBuffer cb);
uint64_t vk_now_ns(void);

/* ---------------- pools (vk_pool.c) ---------------- */
/* The GPU view of a guest address range: a sparse VkBuffer per memory pool, bound to udmabuf chunks of the guest
   memory's memfd pages (CPU and GPU share the bytes). */
int vk_pool_lookup(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *avail);
void vk_pool_init(void);

/* ---------------- formats (vk_format.c) ---------------- */
typedef struct {
    VkFormat vk; uint8_t swz[4];         /* VkComponentSwizzle per output r,g,b,a (format remap, before the NVN swizzle) */
    uint8_t depth, stencil, compressed, srgb, is_int, is_uint, bpb, bw, bh;
} vk_fmt;
int vk_fmt_of(int nvn_format, vk_fmt *out);   /* 0 when unmapped */
VkFormat vk_srgb_pair(VkFormat f);            /* the sRGB / UNORM counterpart, or VK_FORMAT_UNDEFINED */
VkFormat vk_vertex_format(int nvn_vfmt);
VkBlendFactor vk_blend_factor(int nvn);
VkBlendOp vk_blend_op(int nvn);
VkCompareOp vk_compare(int nvn_func);        /* NVN NEVER..ALWAYS = 1..8 */
VkStencilOp vk_stencil_op(int nvn);
VkLogicOp vk_logic_op(int nvn);
int vk_fmt_class(int nvn_format);             /* 0 float, 1 uint, 2 sint (fmt_class of nvn_mtl.m) */

/* ---------------- textures (vk_tex.c) ---------------- */
typedef struct vk_image vk_image;
struct vk_image {
    VkImage img; VmaAllocation alloc;
    VkFormat format; VkImageType type; VkImageCreateFlags flags; VkImageUsageFlags usage;
    uint32_t w, h, d, levels, layers, samples;
    int nvn_format, nvn_target;
    uint8_t depth, stencil, host_copy, listed;   /* host_copy: created with HOST_TRANSFER usage; listed: with a view format list */
    VkImageView full;                    /* view of the whole image in its own format (attachments use per-subresource views) */
    int refs;                            /* serials holding it (storage sharing) */
    uint64_t last_use;                   /* timeline value of the last submission that used it */
    int owned;                           /* rendered / written by the GPU: its pool bytes are stale */
    struct vk_view_cache *views;         /* attachment / storage / typed views made on demand */
    uint32_t epoch;                      /* bumped when the image is made again (image_unlist): views of an older epoch are stale */
};
void vk_tex_init(void);
/* The texture for a NVN texture object (created / uploaded / re-uploaded as needed). Replay thread only. */
vk_image *vk_tex_image(const tex_t *t);
/* A view of a texture for sampling / storage / attachment use: view = NVN view or NULL (whole texture). For sampling the
   NVN swizzle is applied. kind 0 sample, 1 storage, 2 attachment (one level / layer range). */
VkImageView vk_tex_view(const tex_t *t, const view_t *v, int kind, VkImageViewType want_type, VkFormat *fmt_out);
/* The VkImageView bound for texture-pool id tid (registered with mtl_reg_tex), or VK_NULL_HANDLE. */
VkImageView vk_tex_by_id(int tid, VkImageViewType want_type, vk_image **img_out, int *nvn_format_out);
VkBufferView vk_tex_buffer_view(int tid, int storage);
VkSampler vk_samp_by_id(int sid, int nearest_depth);
VkSampler vk_samp_default(void);
VkImageView vk_dummy_view(VkImageViewType type, int cls);   /* cls 0 float, 1 uint, 2 sint, 3 depth (compare) */
VkImageView vk_dummy_storage(VkImageViewType type, int cls);
VkBufferView vk_dummy_texel(int storage, int cls);
void vk_tex_mark_owned(const tex_t *t);       /* the GPU renders into it */
int vk_tex_owned(uint64_t serial);            /* the GPU rendered into that serial */
void vk_uw_note_present(const tex_t *t);      /* vk_exec.c, key 4: a presented texture (replay thread) */
void vk_tex_note_gpu_write(const tex_t *t);   /* for write-back at nvnMemoryPoolInvalidateMappedRange */
int vk_tex_registry_gen(void);
const tex_t *vk_tex_registered(int tid);      /* the texture registered at tid (caller holds no lock; copy out) */
int vk_tex_is_depth_format(int nvn_format);
void vk_tex_frame_sweep(uint64_t value);
/* FSR 1 at the present (vk_fsr.c): whether it takes src (x0, y0, x1, y1) of im to w x h; and the upscale itself, drawn into
   dst (a colour attachment view of dst_fmt, the target tw x th) over (dx, dy, w, h), clear: the rest black. *view: im's view
   it made, destroy it when cb has run. 0 when not done (nothing recorded). */
int vk_fsr_will(const vk_image *im, const int32_t src[4], uint32_t w, uint32_t h);
int vk_fsr_upscale(VkCommandBuffer cb, const vk_image *im, const int32_t src[4], VkImageView dst, VkFormat dst_fmt, uint32_t tw, uint32_t th,
                   int32_t dx, int32_t dy, uint32_t w, uint32_t h, int clear, VkImageView *view);
/* Uploads recorded during a replay go into the submission's upload command buffer (executed before the replay's). */
VkCommandBuffer vk_upload_cb(void);
void vk_tex_wb_flush_list(void);
/* Rendering into a block-compressed texture through an uncompressed view whose texel is one block (nvn_mtl.m blk_aux):
   the pass renders into an auxiliary image of the view format sized in blocks; vk_tex_blk_copy moves the blocks into
   the compressed image when the pass ends (to_dst) and back before a pass that loads the old contents. */
typedef struct { vk_image *aux, *dst; uint32_t level, slice, nslices, is3d, bx, by, lw, lh; } vk_blk;
vk_image *vk_tex_attach_blk(const tex_t *t, const view_t *v, VkImageView *view, VkFormat *fmt, uint32_t *w, uint32_t *h, uint32_t *layers, vk_blk *b);
void vk_tex_blk_copy(VkCommandBuffer cb, const vk_blk *b, int to_dst);
void vk_tex_wb_stage(VkCommandBuffer cb);
void vk_tex_wb_supersede(const void *mem, size_t n);   /* a GPU copy into pool memory: older renderings there are not written back */     /* mtl_sub_commit: small renderings of the submission into their write-back stages */
void vk_tex_wb_submitted(uint64_t value);    /* mtl_sub_commit: the submission's timeline value for them */

/* ---------------- shaders (vk_shader.c) ---------------- */
enum { VB_UBO, VB_SSBO, VB_COMBINED, VB_SAMPLED, VB_SAMPLER, VB_STORAGE_IMG, VB_UNIFORM_TEXEL, VB_STORAGE_TEXEL, VB_NTYPES };
/* Universal set layout bindings */
#define VKB_UBO0 0        /* 0 support buffer, 1 + bank (c0..c17) */
#define VKB_NUBO 19
#define VKB_SSBO0 19      /* shader-local storage buffer index */
#define VKB_NSSBO 16
#define VKB_COMB0 35      /* combined image samplers, shader order */
#define VKB_NCOMB 48
#define VKB_SAMPLED0 83
#define VKB_NSAMPLED 8
#define VKB_SAMPLER0 91
#define VKB_NSAMPLER 8
#define VKB_SIMG0 99
#define VKB_NSIMG 8
#define VKB_UTEX0 107
#define VKB_NUTEX 8
#define VKB_STEX0 115
#define VKB_NSTEX 8
#define VKB_COUNT 123

typedef struct {
    uint8_t type;            /* VB_* */
    uint8_t binding;         /* universal binding number */
    uint8_t kind;            /* NVN_RB_* of the resource (CBUF / C1 / SUPPORT / DRV0 / DRV2 / SBUFP / UBOP / SBUFU / TEX / TBUF / IMG / SAMP / SAMPT) */
    uint8_t rw;
    uint16_t reg, off;       /* NVN slot; SBUFU byte offset */
    uint8_t samp_kind;       /* combined: NVN_RB_SAMP / NVN_RB_SAMPT (0 = the texture slot's combined handle) */
    uint16_t samp_reg;
    uint8_t view_type;       /* VkImageViewType of the SPIR-V image (dummies, 2D <-> 2D array fix-ups) */
    uint8_t shadow, ms, cls; /* depth compare image; multisample; sampled class 0 float 1 uint 2 sint */
    uint8_t bank;            /* uniform buffers: the Maxwell constant bank (c0..c17) */
    uint8_t ji;              /* index of the reflection entry */
} vk_bind;

typedef struct vk_shader {
    char key[96];            /* "<variant>_b" */
    int stage;               /* NVN stage: 0 vertex, 1 fragment, 5 compute */
    VkShaderEXT obj;
    nvn_refl rf;
    int nb; vk_bind b[NVN_REFL_MAXB];
    VkBuffer c1_buf; VkDeviceSize c1_off, c1_range;
    uint64_t compare_slots[2];
    int writes_depth, keeps;
    uint32_t out_locs, in_locs;   /* user varyings written (vertex) / read (fragment) */
    VkShaderEXT lay_obj; int lay_state;   /* vertex: the variant that writes the layer of a layered pass (vk_shader_layered); state 1 made, -1 not possible */
    VkShaderEXT sq_obj; int sq_state;     /* vertex: the variant that squeezes the UI per vertex (vk_shader_ui_squeezed) */
} vk_shader;

typedef struct vk_prog vk_prog;
struct vk_prog {
    uint64_t serial;
    vk_shader *sh[6];
    const uint8_t *ctl[6], *code[6];
    char key[6][65];
    uint64_t sig[6];
    struct { uint64_t sig; vk_shader *sh; } alt[2][16]; int nalt[2];
    int keeps;
};
void vk_shader_init(void);
/* The push constant range of every graphics shader object and of the graphics pipeline layout (identical everywhere, so
   any vertex and fragment shader objects draw together): the builtin clears' colour, a layered pass's layer at 0. */
extern const VkPushConstantRange vk_gfx_pc;
/* The vertex shader's variant for a layered pass: gl_Layer = the uint push constant at offset 0. VK_NULL_HANDLE when the
   shader cannot write it (it writes a layer itself). Made on first use, replay thread. */
VkShaderEXT vk_shader_layered(vk_shader *vs);
VkShaderEXT vk_shader_ui_squeezed(vk_shader *sh);   /* key 4: per-vertex UI squeeze keeping the screen's edges (vk_shader.c), VK_NULL_HANDLE when not possible */
vk_prog *vk_prog_get(const prog_t *p);          /* replay thread */
int vk_prog_fit(vk_prog *pr);                    /* pick the variants for the bound state; 1 = changed, -1 = unavailable */
void vk_prog_forget(uint64_t serial);
/* What the translator answers depend on, from the replay state (vk_exec.c) */
struct jv;
struct jv *vk_answers(int stage);
uint64_t vk_stage_sig(int stage);
int vk_depth_near_minus_w(void);

/* ---------------- replay state shared with the shader code (vk_exec.c) ---------------- */
typedef struct {
    const prog_t *prog;
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream;
    NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][128], samp[6][128], img[6][32];
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp; int depth_clamp; int render_disabled; int raster_discard;
    float blend_color[4]; int64_t copy_stride;
} vk_state;
extern vk_state S;
extern uint32_t vk_tex_bind_gen[6];   /* bumped when a stage's texture slots change (OP_BIND_TEX, state_reset): vk_stage_sig's cache */
typedef struct {
    p_set_rt rt; int has_rt;
    uint8_t clear_c[8]; float cc[8][4];
    uint8_t clear_d, clear_s; float cd; uint32_t cs;
    p_scissor sc; int has_sc;
} vk_rt_state;
extern vk_rt_state R;

/* texture-pool registry lookups used by the answers (vk_tex.c) */
int vk_treg_target_format(uint64_t handle, int *target, int *format);   /* 0 when the slot is empty / unregistered */

/* ---------------- present (vk_present.c) ---------------- */
void vk_present_init(void);
