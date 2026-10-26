/* Textures on Vulkan, following nvn_mtl.m's rules (see the comments there for the measurements behind them):
   - A texture in a CPU-visible pool that the CPU fills (tex_mirrored) mirrors its pool bytes: it is uploaded at its first
     use (deswizzled on the CPU, nvn_deswizzle) and uploaded again into a new image when the dirty-page tracker
     (nvn_dirty.c) says the game wrote its pages, at most once per submission. Draws already recorded keep the old image.
     Once the GPU renders into it, it is GPU-owned: never uploaded again (its pool bytes are stale).
   - Other textures are GPU-owned from the start. A texture object created again at the same storage with the same layout
     shares the earlier one's image (the game re-creates transient render targets every frame and expects their content).
   - Texture-pool slots (mtl_reg_tex) are snapshots of texture objects; a slot whose texture was finalised follows its
     storage to the live texture there, as the NVN descriptor (which points at memory) does on the console.
   - GPU renderings into CPU-visible pools are written back (swizzled) when the game invalidates the range
     (nvnMemoryPoolInvalidateMappedRange): Breath of the Wild reads its 1x1 / 3x3 picking targets and save thumbnails so.
   Uploads use host image copies (Vulkan 1.4) where the driver offers them for the format, else a staging copy in the
   submission's upload command buffer, which runs before the replay's. */

#include "vk_priv.h"
#include <stdatomic.h>
#include "nvn_dirty.h"
#include "nvn_hmap.h"
#include "../../third_party/d2r_gfx/gfx/tex/nvn_deswizzle.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern uint64_t vk_sub_serial;            /* vk_exec.c: number of the submission being replayed */
extern int vk_replaying(void);             /* vk_exec.c: a submission is being recorded on this thread */
VkCommandBuffer vk_cmd_outside_pass(void); /* vk_exec.c: the replay's command buffer with no render pass open */
int nvn_tex_live_at(const void *mem, tex_t *out);
void dm_guest_write_prepare(void *p, size_t n);

static pthread_mutex_t g_tmu = PTHREAD_MUTEX_INITIALIZER;   /* everything below */
#define LOCK() pthread_mutex_lock(&g_tmu)
#define UNLOCK() pthread_mutex_unlock(&g_tmu)

/* ---------------- images and their views ---------------- */
typedef struct vk_view_cache {
    struct vk_view_cache *next;
    VkFormat fmt; VkImageViewType type; VkImageAspectFlags aspect; uint32_t l0, nl, s0, ns; uint8_t swz[4]; int usage_kind;
    VkImageView view;
} vk_view_cache;

static VkImageUsageFlags fmt_usage_ok(VkFormat f, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags, uint32_t *max_samples) {
    /* drop usage bits the format cannot have (e.g. storage on sRGB, attachment on compressed) */
    VkImageUsageFlags bits[] = {VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_USAGE_HOST_TRANSFER_BIT};
    for (int pass = 0; pass < 5; pass++) {
        VkPhysicalDeviceImageFormatInfo2 fi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .format = f, .type = type, .tiling = tiling, .usage = usage, .flags = flags};
        VkImageFormatProperties2 fp = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
        if (vkGetPhysicalDeviceImageFormatProperties2(V.pd, &fi, &fp) == VK_SUCCESS) { if (max_samples) *max_samples = fp.imageFormatProperties.sampleCounts; return usage; }
        if (pass < 4) usage &= ~bits[pass];
    }
    return 0;
}
static VkImageViewType view_type_of(int target, uint32_t layers) {
    switch (target) {
        case 0: return VK_IMAGE_VIEW_TYPE_1D; case 2: return VK_IMAGE_VIEW_TYPE_3D; case 3: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        case 4: case 6: return VK_IMAGE_VIEW_TYPE_2D_ARRAY; case 8: return VK_IMAGE_VIEW_TYPE_CUBE; case 9: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        default: return layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}
static uint32_t layers_of(const nvn_texdesc *d) {
    switch (d->target) {
        case 3: return d->height ? d->height : 1;
        case 4: case 6: case 9: return d->depth ? d->depth : 1;
        case 8: return 6;
        default: return 1;
    }
}
static uint64_t g_images_made, g_images_live;
/* A new image for a NVN texture description (no contents). host: with host-transfer usage (mirrored uploads). */
static vk_image *image_new(const tex_t *t, int host, VkFormat extra_view_fmt) {
    vk_fmt f; if (!vk_fmt_of(t->d.format, &f)) { VK_ONCE(fmt0, "texture format %#x has no Vulkan counterpart (first: serial %llu): bound as empty", t->d.format, (unsigned long long)t->serial); return NULL; }
    vk_image *im = calloc(1, sizeof *im); if (!im) VDIE("image");
    const nvn_texdesc *d = &t->d;
    im->w = d->width ? d->width : 1; im->h = d->height ? d->height : 1; im->d = 1; im->layers = layers_of(d);
    im->levels = d->levels ? d->levels : 1; im->samples = (d->target == 5 || d->target == 6) && d->samples > 1 ? (uint32_t)d->samples : 1;
    im->type = VK_IMAGE_TYPE_2D;
    switch (d->target) {
        case 0: case 3: im->type = VK_IMAGE_TYPE_1D; im->h = 1; break;
        case 2: im->type = VK_IMAGE_TYPE_3D; im->d = d->depth ? d->depth : 1; break;
    }
    if (im->w > 16384) im->w = 16384; if (im->h > 16384) im->h = 16384; if (im->d > 2048) im->d = 2048; if (im->layers > 2048) im->layers = 2048;
    { uint32_t big = im->w > im->h ? im->w : im->h; if (im->type == VK_IMAGE_TYPE_3D && im->d > big) big = im->d; uint32_t allowed = 1; while (big > 1) { big >>= 1; allowed++; }
      if (im->levels > allowed) im->levels = allowed; }
    if (im->samples > 1) im->levels = 1;
    im->format = f.vk; im->nvn_format = d->format; im->nvn_target = d->target; im->depth = f.depth; im->stencil = f.stencil;
    VkImageCreateFlags flags = 0;
    if (d->target == 8 || d->target == 9 || (d->target == 4 && im->layers >= 6 && im->w == im->h)) flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    VkFormat list[3]; uint32_t nlist = 0;
    if (!f.depth && !f.stencil) {
        flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; list[nlist++] = f.vk;
        VkFormat p = vk_srgb_pair(f.vk); if (p) list[nlist++] = p;
        if (extra_view_fmt && extra_view_fmt != f.vk && extra_view_fmt != p) nlist = 0;   /* an unrelated view format: no list (any compatible view) */
        if (f.compressed) flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    }
    if (im->type == VK_IMAGE_TYPE_3D) flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (f.depth || f.stencil) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    else if (!f.compressed) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (d->flags & NVN_TEXTURE_FLAGS_IMAGE) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (host && V.host_copy) usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT;
    uint32_t sample_counts = 0;
    VkImageUsageFlags ok = fmt_usage_ok(f.vk, im->type, VK_IMAGE_TILING_OPTIMAL, usage, flags & ~(VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT), &sample_counts);
    if (!ok) { VK_ONCE(fmtimg, "image of Vulkan format %d (NVN %#x) target %d not supported by the device: bound as empty", f.vk, d->format, d->target); free(im); return NULL; }
    usage = ok;
    if (im->samples > 1 && !(sample_counts & im->samples)) im->samples = 1;
    VkImageFormatListCreateInfo fl = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = nlist, .pViewFormats = list};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = nlist ? &fl : NULL, .flags = flags, .imageType = im->type, .format = f.vk,
        .extent = {im->w, im->h, im->d}, .mipLevels = im->levels, .arrayLayers = im->layers, .samples = (VkSampleCountFlagBits)im->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkResult r = vmaCreateImage(V.vma, &ci, &ai, &im->img, &im->alloc, NULL);
    if (r != VK_SUCCESS) { VLOG("image %ux%ux%u x%u lv%u format %d failed: %d", im->w, im->h, im->d, im->layers, im->levels, f.vk, r); free(im); return NULL; }
    im->flags = flags; im->usage = usage; im->host_copy = (usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT) != 0; im->listed = nlist != 0;
    VkImageAspectFlags aspect = f.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | (f.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0) : f.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageSubresourceRange all = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    /* GENERAL layout throughout (Honeykrisp keys compression off the image, not the layout); fresh contents are zero */
    if (im->host_copy) {
        VkHostImageLayoutTransitionInfo hi = {VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO, .image = im->img, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                              .newLayout = VK_IMAGE_LAYOUT_GENERAL, .subresourceRange = all};
        VK(vkTransitionImageLayout(V.dev, 1, &hi));
    } else {
        VkCommandBuffer cb = vk_replaying() ? vk_upload_cb() : vk_oneoff_begin();
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im->img, .subresourceRange = all};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(cb, &di);
        if (!(f.depth || f.stencil) && !f.compressed) {   /* Metal textures start as zero; NVN storage is the pool's bytes. Zero, as Metal shared storage */
            VkClearColorValue z = {{0, 0, 0, 0}}; VkImageSubresourceRange cr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &cr);
        } else if (f.depth || f.stencil) {
            VkClearDepthStencilValue z = {0, 0};
            vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &all);
        }
        vk_barrier_all(cb);
        if (!vk_replaying()) vk_oneoff_wait(vk_oneoff_end_submit(cb));
    }
    g_images_made++; g_images_live++;
    return im;
}
static void image_free(void *arg) {   /* completion thread */
    vk_image *im = arg;
    for (vk_view_cache *v = im->views, *n; v; v = n) { n = v->next; vkDestroyImageView(V.dev, v->view, NULL); free(v); }
    if (im->full) vkDestroyImageView(V.dev, im->full, NULL);
    vmaDestroyImage(V.vma, im->img, im->alloc);
    free(im);
}
/* after the next replay submission completes: the replay may still hold the struct (it records with it outside the lock) */
static void image_destroy(vk_image *im) {
    if (!im) return;
    g_images_live--;
    vk_on_done(UINT64_MAX, image_free, im);
}
static VkImageAspectFlags view_aspect(const vk_image *im, VkFormat vf, int kind) {
    if (!im->depth && !im->stencil) return VK_IMAGE_ASPECT_COLOR_BIT;
    if (kind == 2) return (im->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    if (vf == VK_FORMAT_S8_UINT || !im->depth) return VK_IMAGE_ASPECT_STENCIL_BIT;
    return VK_IMAGE_ASPECT_DEPTH_BIT;
}
/* A cached view (g_tmu held). kind: 0 sampled, 1 storage, 2 attachment. */
static VkImageView image_view(vk_image *im, VkFormat fmt, VkImageViewType type, uint32_t l0, uint32_t nl, uint32_t s0, uint32_t ns, const uint8_t swz[4], int kind) {
    VkImageAspectFlags aspect = view_aspect(im, fmt, kind);
    if (im->depth || im->stencil) fmt = im->format;   /* depth / stencil views keep the image format (the aspect picks the plane) */
    uint8_t sw[4] = {0, 0, 0, 0}; if (swz && kind == 0) memcpy(sw, swz, 4);
    for (vk_view_cache *v = im->views; v; v = v->next)
        if (v->fmt == fmt && v->type == type && v->aspect == aspect && v->l0 == l0 && v->nl == nl && v->s0 == s0 && v->ns == ns && !memcmp(v->swz, sw, 4) && v->usage_kind == kind) return v->view;
    VkImageViewUsageCreateInfo uci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .usage = kind == 1 ? VK_IMAGE_USAGE_STORAGE_BIT : kind == 2 ? (im->depth || im->stencil ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) : VK_IMAGE_USAGE_SAMPLED_BIT};
    uci.usage &= im->usage;
    if (!uci.usage) return VK_NULL_HANDLE;
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &uci, .image = im->img, .viewType = type, .format = fmt,
        .components = {(VkComponentSwizzle)sw[0], (VkComponentSwizzle)sw[1], (VkComponentSwizzle)sw[2], (VkComponentSwizzle)sw[3]},
        .subresourceRange = {aspect, l0, nl, s0, ns}};
    VkImageView view;
    if (vkCreateImageView(V.dev, &ci, NULL, &view) != VK_SUCCESS) { VK_ONCE(view0, "image view (format %d type %d levels %u+%u layers %u+%u) not made", fmt, type, l0, nl, s0, ns); return VK_NULL_HANDLE; }
    vk_view_cache *c = calloc(1, sizeof *c); if (!c) VDIE("view cache");
    c->fmt = fmt; c->type = type; c->aspect = aspect; c->l0 = l0; c->nl = nl; c->s0 = s0; c->ns = ns; memcpy(c->swz, sw, 4); c->usage_kind = kind; c->view = view;
    c->next = im->views; im->views = c;
    return view;
}
/* In-place upgrade (nvn_mtl.m tex_upgrade_pfv): an image made with a format list, which keeps Honeykrisp's lossless
   compression, that is then needed in a format outside the list (a texture at the same storage or a view in another
   format: Breath of the Wild reads an R16F target as RG8) is made again without a list, so any format with the same
   bytes per texel can view it. Its contents are copied in command order (inside a replay: in the replay's command buffer,
   with the render pass ended) and its old views stay valid until the submission completes; the epoch makes the
   texture registry make new ones. 0: not possible (depth / stencil, compressed, another texel size). */
typedef struct { VkImage img; VmaAllocation alloc; vk_view_cache *views; } old_image;
static void old_image_free(void *arg) {   /* completion thread */
    old_image *o = arg;
    for (vk_view_cache *v = o->views, *n; v; v = n) { n = v->next; vkDestroyImageView(V.dev, v->view, NULL); free(v); }
    vmaDestroyImage(V.vma, o->img, o->alloc);
    free(o);
}
static int image_unlist(vk_image *im, const vk_fmt *want) {
    if ((im->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) && !im->listed) return 1;
    vk_fmt cur;
    if (im->depth || im->stencil || want->depth || want->stencil || want->compressed || !vk_fmt_of(im->nvn_format, &cur) || cur.compressed || cur.bpb != want->bpb) return 0;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = im->flags | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, .imageType = im->type, .format = im->format,
        .extent = {im->w, im->h, im->d}, .mipLevels = im->levels, .arrayLayers = im->layers, .samples = (VkSampleCountFlagBits)im->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = im->usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkImage img; VmaAllocation alloc;
    if (vmaCreateImage(V.vma, &ci, &ai, &img, &alloc, NULL) != VK_SUCCESS) { VK_ONCE(unlist0, "image of format %d made again without a format list: failed", im->format); return 0; }
    int inline_cb = vk_replaying();
    VkCommandBuffer cb = inline_cb ? vk_cmd_outside_pass() : vk_oneoff_begin();
    vk_barrier_all(cb);   /* the image's earlier writes are complete */
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    VkImageCopy rg[16]; uint32_t n = 0, layers = im->type == VK_IMAGE_TYPE_3D ? 1 : im->layers;
    for (uint32_t l = 0; l < im->levels && n < 16; l++, n++)
        rg[n] = (VkImageCopy){.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers},
                              .extent = {im->w >> l ? im->w >> l : 1, im->h >> l ? im->h >> l : 1, im->type == VK_IMAGE_TYPE_3D && im->d >> l ? im->d >> l : 1}};
    vkCmdCopyImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    vk_barrier_all(cb);
    if (!inline_cb) vk_oneoff_wait(vk_oneoff_end_submit(cb));
    old_image *o = calloc(1, sizeof *o); if (!o) VDIE("image upgrade");
    o->img = im->img; o->alloc = im->alloc; o->views = im->views;
    vk_on_done(UINT64_MAX, old_image_free, o);   /* after the replay submission that may still read them */
    im->img = img; im->alloc = alloc; im->views = NULL;
    im->flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; im->listed = 0; im->epoch++;
    static uint64_t made; ++made;
    if (made <= 8 || made % 100 == 0) VLOG("image %ux%u x%u of format %d (NVN %#x) made again for a view in format %d (%llu so far)",
                                           im->w, im->h, im->layers, im->format, im->nvn_format, want->vk, (unsigned long long)made);
    return 1;
}

/* ---------------- texture objects ---------------- */
typedef struct texobj {
    uint64_t serial;
    tex_t t;
    vk_image *img;
    uint8_t mirrored, hashed, tracked, owned, stor;   /* stor: img is the storage-shared image (counted in its refs) */
    uint64_t stamp, seen, csub, chash;
    uint8_t *edge; uint32_t eh, et; uint8_t edge_want;
    VkBufferView bview[2]; VkFormat bview_fmt;
    uint64_t gen;                                     /* bumped when img changes (view caches look again) */
} texobj;
static nvn_hmap g_obj = {.klen = 8};                  /* serial -> texobj */
static nvn_hmap g_dead = {.klen = 8};                 /* finalised serials -> (void *)1 */
static nvn_hmap g_owned_set = {.klen = 8};            /* serials the GPU rendered into */
typedef struct { uint64_t mem, size; uint32_t w, h, d, levels, target, samples, bpb, depth_fmt; } stor_key;
static nvn_hmap g_stor = {.klen = sizeof(stor_key)};   /* storage -> vk_image (GPU-owned, shared) */
static uint64_t g_tex_gen = 1;                         /* bumped whenever any texobj's image changes */
static uint64_t g_uploads, g_reuploads, g_upload_bytes;

static int tex_mirrored(const tex_t *t) {
    if (!t || !t->mem) return 0;
    if (t->has_layout) return 1;
    static int on = -1; if (on < 0) { const char *e = getenv("TOTK_TEX_MIRROR"); on = !e || atoi(e); }
    if (!on) return 0;
    const pool_t *p = t->pool;
    if (!p || p->host || !(p->flags & (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_CPU_CACHED))) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL | NVN_TEXTURE_FLAGS_DEPTH16_PREFER_FAST_CLEAR | NVN_TEXTURE_FLAGS_ZCULL_SUPPORT_STENCIL)) return 0;
    if (t->d.samples > 1 || t->d.target == 10) return 0;
    if (t->d.format >= 0x31 && t->d.format <= 0x36) return 0;
    return 1;
}
int vk_tex_is_depth_format(int f) { return f >= 0x31 && f <= 0x36; }
#define PKG_HASH_MAX (4u << 10)
static uint64_t pkg_hash(const void *p, uint64_t n) {
    const uint8_t *b = p; uint64_t a[4] = {0x9E3779B97F4A7C15ull ^ n, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull, 0x27D4EB2F165667C5ull}, i = 0;
    for (; i + 32 <= n; i += 32) for (int k = 0; k < 4; k++) { uint64_t w; memcpy(&w, b + i + 8 * k, 8); a[k] = (a[k] ^ w) * 0xff51afd7ed558ccdull; a[k] ^= a[k] >> 29; }
    uint64_t h = a[0] ^ (a[1] * 3) ^ (a[2] * 5) ^ (a[3] * 7);
    for (; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; return h ^ (h >> 33);
}
static int stor_key_of(const tex_t *t, stor_key *k) {
    if (!t->mem || !t->L.size) return 0;
    int bw = 1, bh = 1, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb || bw != 1 || bh != 1) return 0;
    memset(k, 0, sizeof *k);
    k->mem = (uint64_t)(uintptr_t)t->mem; k->size = t->L.size; k->w = t->d.width ? t->d.width : 1; k->h = t->d.height ? t->d.height : 1; k->d = t->d.depth ? t->d.depth : 1;
    k->levels = t->d.levels ? t->d.levels : 1; k->target = (uint32_t)t->d.target; k->samples = t->d.samples > 1 ? (uint32_t)t->d.samples : 1; k->bpb = (uint32_t)bpb;
    k->depth_fmt = (uint32_t)vk_tex_is_depth_format(t->d.format);
    return 1;
}
static void obj_release_image(texobj *o) {
    if (!o->img) return;
    if (o->stor) {
        if (--o->img->refs <= 0) {
            stor_key k; if (stor_key_of(&o->t, &k) && nvn_hm_get(&g_stor, &k) == o->img) nvn_hm_del(&g_stor, &k);
            image_destroy(o->img);
        }
    } else image_destroy(o->img);
    o->img = NULL; o->stor = 0; o->gen = ++g_tex_gen;
}
static void obj_free(texobj *o) {
    obj_release_image(o);
    for (int i = 0; i < 2; i++) vk_defer_bufview(UINT64_MAX, o->bview[i]);
    free(o->edge); free(o);
}

/* ---------- uploads ---------- */
static VkCommandBuffer g_up_cb;   /* set by vk_exec.c (vk_upload_cb) */
static int copy_level_layer(const tex_t *t, unsigned l, unsigned y, uint8_t *dst, size_t pitch, size_t rows) {
    int r;
    if (nvn_tex_is_linear(&t->d)) {
        size_t stride = nvn_tex_linear_stride(&t->d);
        uint64_t off = nvn_tex_level_offset(&t->d, l);
        for (size_t i = 0; i < rows; i++) memcpy(dst + i * pitch, t->mem + off + i * stride, pitch);
        r = 0;
    } else r = nvn_deswizzle(&t->d, l, y, t->mem, dst, pitch);
    if (!r && t->d.format == 0x98)   /* A1BGR5 (alpha in bit 0) -> A1R5G5B5 (alpha in bit 15): rotate right by one */
        for (size_t i = 0; i + 1 < pitch * rows; i += 2) { uint16_t v; memcpy(&v, dst + i, 2); v = (uint16_t)(v >> 1 | (v & 1) << 15); memcpy(dst + i, &v, 2); }
    return r;
}
/* Fill an image from the texture's pool bytes. */
static int upload(const tex_t *t, vk_image *im) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) return 0;
    unsigned layers = im->layers;
    int is3d = t->d.target == 2;
    size_t total = 0;
    for (unsigned l = 0; l < im->levels; l++) {
        uint32_t w = t->d.width >> l ? t->d.width >> l : 1, h = t->d.height >> l ? t->d.height >> l : 1, z = is3d ? (t->d.depth >> l ? t->d.depth >> l : 1) : 1;
        if (t->d.target == 0 || t->d.target == 3) h = 1;
        total += (size_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh) * z * layers;
    }
    g_upload_bytes += total;
    VkImageAspectFlags aspect = im->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    if (im->host_copy) {
        uint8_t *lin = NULL; size_t cap = 0; int ok = 1;
        for (unsigned l = 0; l < im->levels && ok; l++) {
            uint32_t w = t->d.width >> l ? t->d.width >> l : 1, h = t->d.height >> l ? t->d.height >> l : 1, z = is3d ? (t->d.depth >> l ? t->d.depth >> l : 1) : 1;
            if (t->d.target == 0 || t->d.target == 3) h = 1;
            size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh, need = pitch * rows * z;
            if (need > cap) { free(lin); lin = malloc(need); cap = need; if (!lin) VDIE("upload buffer"); }
            for (unsigned y = 0; y < (is3d ? 1 : layers); y++) {
                if (copy_level_layer(t, l, y, lin, pitch, rows * z)) { VK_ONCE(desw, "deswizzle failed: format %#x target %d", t->d.format, t->d.target); ok = 0; break; }
                VkMemoryToImageCopy rg = {VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY, .pHostPointer = lin, .memoryRowLength = 0, .memoryImageHeight = 0,
                    .imageSubresource = {aspect, l, y, 1}, .imageOffset = {0, 0, 0}, .imageExtent = {w, h, z}};
                VkCopyMemoryToImageInfo ci = {VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO, .dstImage = im->img, .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL, .regionCount = 1, .pRegions = &rg};
                VkResult r = vkCopyMemoryToImage(V.dev, &ci);
                if (r != VK_SUCCESS) { VK_ONCE(hostcopy, "host image copy failed (%d), format %d", r, im->format); ok = 0; break; }
            }
        }
        free(lin);
        return ok;
    }
    /* staging copy: in the replay's upload command buffer, or a one-off outside a submission */
    int replay = vk_replaying();
    VkCommandBuffer cb = replay ? vk_upload_cb() : vk_oneoff_begin();
    vk_slice s = {0}; VkBuffer sbuf = VK_NULL_HANDLE; VmaAllocation salloc = NULL; uint8_t *base;
    if (replay) { s = vk_arena_alloc(total, 16); sbuf = s.buf; base = s.cpu; }
    else {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, .usage = VMA_MEMORY_USAGE_AUTO};
        VmaAllocationInfo info; VK(vmaCreateBuffer(V.vma, &bi, &ai, &sbuf, &salloc, &info)); base = info.pMappedData;
    }
    size_t off = 0; int ok = 1;
    VkBufferImageCopy regs[64]; int nr = 0;
    for (unsigned l = 0; l < im->levels && ok; l++) {
        uint32_t w = t->d.width >> l ? t->d.width >> l : 1, h = t->d.height >> l ? t->d.height >> l : 1, z = is3d ? (t->d.depth >> l ? t->d.depth >> l : 1) : 1;
        if (t->d.target == 0 || t->d.target == 3) h = 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
        for (unsigned y = 0; y < (is3d ? 1 : layers); y++) {
            if (copy_level_layer(t, l, y, base + off, pitch, rows * z)) { VK_ONCE(desw2, "deswizzle failed: format %#x target %d", t->d.format, t->d.target); ok = 0; break; }
            regs[nr++] = (VkBufferImageCopy){.bufferOffset = s.off + off, .imageSubresource = {aspect, l, y, 1}, .imageExtent = {w, h, z}};
            off += pitch * rows * z;
            if (nr == 64) { vkCmdCopyBufferToImage(cb, sbuf, im->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)nr, regs); nr = 0; }
        }
    }
    if (nr) vkCmdCopyBufferToImage(cb, sbuf, im->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)nr, regs);
    vk_barrier_all(cb);
    if (!replay) { vk_oneoff_wait(vk_oneoff_end_submit(cb)); vmaDestroyBuffer(V.vma, sbuf, salloc); }
    return ok;
}
/* First upload or re-upload of a mirrored texture (g_tmu held): a new image with the pool bytes. */
static vk_image *mirror_upload(texobj *o) {
    const tex_t *t = &o->t;
    uint64_t stamp = nvn_dirty_now();   /* before protecting: a write during the copy is newer than the copy */
    if (!o->hashed) nvn_dirty_protect(t->mem, t->L.size);
    vk_image *im = image_new(t, 1, VK_FORMAT_UNDEFINED);
    if (!im) return NULL;
    if (!upload(t, im)) { /* keep the (zeroed / partial) image: retrying every use would not help */ }
    o->stamp = stamp; o->seen = stamp; o->tracked = nvn_dirty_tracked(t->mem, t->L.size);
    if (o->hashed) { o->chash = pkg_hash(t->mem, t->L.size); o->csub = vk_sub_serial; }
    return im;
}
static int is_owned(uint64_t serial) { return nvn_hm_get(&g_owned_set, &serial) != NULL; }
static uint64_t *g_deadmade; static int g_ndeadmade, g_capdeadmade;   /* texture objects made for finalised serials this submission */
#ifdef GAME_BOTW
/* botw.md, the black save thumbnail (nvn_mtl.m botw_tex_follow / botw_dead_keep). The game replays a recorded caption draw
   and copy at every save while it finalises and re-initialises their textures, and it finalises the pause menu's screen
   copy before the recording samples it. On hardware a command addresses memory, so: a command's texture whose serial
   was finalised follows its storage to the texture living there with the same layout (a recording's copy is re-keyed
   in place, so every recording holding it sees the live texture), and the image of a finalised serial a recording
   still names stays until the pin sweep frees that copy (mtl_dead_forget). g_tmu held; the pin lock nests inside it. */
int dm_nvn_pin_rekey(const tex_t *copy, const tex_t *live);
int dm_nvn_pin_has_serial(uint64_t serial);
static nvn_hmap g_dead_kept = {.klen = 8};
static void tex_follow(tex_t *t) {
    if (!t->mem || !nvn_hm_get(&g_dead, &t->serial)) return;
    tex_t live;
    if (!nvn_tex_live_at(t->mem, &live) || live.serial == t->serial || nvn_hm_get(&g_dead, &live.serial)) return;
    if (live.L.size != t->L.size || live.d.format != t->d.format || live.d.width != t->d.width || live.d.height != t->d.height ||
        live.d.depth != t->d.depth || live.d.target != t->d.target || live.d.levels != t->d.levels) return;
    uint64_t from = t->serial; int pinned = dm_nvn_pin_rekey(t, &live);
    if (!pinned) *t = live;
    static uint64_t n; if (n++ < 8 || n % 1000 == 0) VLOG("[tex-follow] %s serial %llu -> %llu (%ux%u format %#x; %llu so far)", pinned ? "recording copy" : "texture",
                                                         (unsigned long long)from, (unsigned long long)live.serial, live.d.width, live.d.height, live.d.format, (unsigned long long)n);
}
/* Only a rendering is kept: its picture exists in the image alone, where a texture loaded by the CPU is made again from
   its pool bytes. Keeping every finalised serial a recording names (as nvn_mtl.m does) kept 20,000 images in the first
   200 s of play here and ran this 8 GB machine out of memory; DEAD_KEEP_MAX bounds what is kept at once. */
enum { DEAD_KEEP_MAX = 1024 };
static int dead_keep(uint64_t serial) {
    if (nvn_hm_get(&g_dead_kept, &serial)) return 1;
    texobj *o = nvn_hm_get(&g_obj, &serial);
    if (!(o ? o->owned : is_owned(serial)) || g_dead_kept.n >= DEAD_KEEP_MAX || !dm_nvn_pin_has_serial(serial)) return 0;
    nvn_hm_put(&g_dead_kept, &serial, (void *)1);
    static uint64_t n; if (n++ < 4 || n % 1000 == 0) VLOG("[dead-keep] finalised rendering %llu stays while a recording names it (%llu so far, %u now)", (unsigned long long)serial, (unsigned long long)n, g_dead_kept.n);
    return 1;
}
#endif
static void dead_note(uint64_t serial) {
    if (g_ndeadmade == g_capdeadmade) { g_capdeadmade = g_capdeadmade ? g_capdeadmade * 2 : 64; g_deadmade = realloc(g_deadmade, sizeof *g_deadmade * (size_t)g_capdeadmade); if (!g_deadmade) VDIE("dead list"); }
    g_deadmade[g_ndeadmade++] = serial;
}
/* The image of a texture object (g_tmu held). */
static vk_image *tex_image_locked(const tex_t *t, texobj **out) {
    if (!t || !t->serial) return NULL;
#ifdef GAME_BOTW
    if (vk_replaying()) tex_follow((tex_t *)t);   /* the command's own copy (nvn_mtl.m tex_mtl) */
#endif
    texobj *o = nvn_hm_get(&g_obj, &t->serial);
    if (!o) {
        o = calloc(1, sizeof *o); if (!o) VDIE("texture object");
        o->serial = t->serial; o->t = *t; o->mirrored = (uint8_t)tex_mirrored(t); o->owned = (uint8_t)is_owned(t->serial);
        o->hashed = o->mirrored && t->L.size && t->L.size <= PKG_HASH_MAX;
        nvn_hm_put(&g_obj, &o->serial, o);
        if (nvn_hm_get(&g_dead, &o->serial)) dead_note(o->serial);   /* made for a finalised serial: released after this submission */
    }
    if (out) *out = o;
    if (o->t.d.target == 10) return NULL;   /* buffer textures have texel-buffer views only */
    if (o->mirrored && !o->owned) {
        if (o->img) {
            int clean;
            if (o->hashed) {
                if (o->csub == vk_sub_serial) return o->img;
                o->csub = vk_sub_serial; uint64_t h = pkg_hash(o->t.mem, o->t.L.size); clean = h == o->chash;
            } else {
                if (o->csub == vk_sub_serial) return o->img;
                o->csub = vk_sub_serial;
                uint64_t now = nvn_dirty_now();
                int tracked = now == o->seen && o->tracked ? 1 : nvn_dirty_tracked(o->t.mem, o->t.L.size);
                clean = tracked && (now == o->seen || !nvn_dirty_since(o->t.mem, o->t.L.size, o->stamp));
                o->seen = now; o->tracked = (uint8_t)tracked;
            }
            if (clean) return o->img;
            vk_image *n = mirror_upload(o);
            if (n) { image_destroy(o->img); o->img = n; o->gen = ++g_tex_gen; g_reuploads++; }
            return o->img;
        }
        o->img = mirror_upload(o);
        if (o->img) { g_uploads++; o->gen = ++g_tex_gen; }
        return o->img;
    }
    if (o->img) return o->img;
    /* GPU-owned: share the image of an earlier texture object at the same storage with the same layout */
    stor_key k; int keyed = !o->mirrored && stor_key_of(&o->t, &k);
    vk_image *im = keyed ? nvn_hm_get(&g_stor, &k) : NULL;
    vk_fmt f; vk_fmt_of(o->t.d.format, &f);
    if (im && im->format != f.vk && vk_srgb_pair(im->format) != f.vk) {
        /* another format at the same storage: shared, its views in this format upgrade a listed image (image_unlist);
           depth / stencil and compressed images cannot be viewed so, and the texture gets an image of its own */
        if (!(im->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) || im->depth || im->stencil) {
            VK_ONCE(storfmt, "texture of format %#x at the storage of an image of format %#x (depth, stencil or compressed): an image of its own", o->t.d.format, im->nvn_format);
            im = NULL;
        }
    }
    if (im) { im->refs++; o->img = im; o->stor = 1; o->gen = ++g_tex_gen; return im; }
    if (o->mirrored) {   /* owned mirrored texture without an image yet: start from its pool bytes */
        o->img = mirror_upload(o); o->gen = ++g_tex_gen; return o->img;
    }
    im = image_new(&o->t, 0, VK_FORMAT_UNDEFINED);
    if (!im) return NULL;
    if (keyed) { im->refs = 1; nvn_hm_put(&g_stor, &k, im); o->stor = 1; }
    o->img = im; o->gen = ++g_tex_gen;
    return im;
}
vk_image *vk_tex_image(const tex_t *t) { LOCK(); vk_image *im = tex_image_locked(t, NULL); UNLOCK(); return im; }
int vk_tex_owned(uint64_t serial) { LOCK(); int r = is_owned(serial); UNLOCK(); return r; }
void vk_tex_mark_owned(const tex_t *t) {
    if (!t || !t->serial) return;
    LOCK();
    if (!is_owned(t->serial)) nvn_hm_put(&g_owned_set, &t->serial, (void *)1);
    texobj *o = nvn_hm_get(&g_obj, &t->serial); if (o) o->owned = 1;
    UNLOCK();
}

/* A view in the format / levels / layers / type of a NVN view (or the whole texture). */
static VkImageView view_for(vk_image *im, const tex_t *t, const view_t *v, int kind, VkImageViewType want, VkFormat *fmt_out) {
    VkFormat fmt = im->format; uint8_t swz[4] = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    int nvn_fmt = v && v->set_format ? v->format : t->d.format;
    vk_fmt f;
    if (vk_fmt_of(nvn_fmt, &f)) { fmt = f.vk; memcpy(swz, f.swz, 4); }
    else if (v && v->set_format) { VK_ONCE(vfmt, "view format %#x has no Vulkan counterpart: the texture's own format is used", v->format); }
    /* bit-compatibility: a view format the image's list does not hold needs the unrestricted image (recreated with a copy) */
    if (fmt != im->format && !(im->depth || im->stencil)) {
        VkFormat p = vk_srgb_pair(im->format);
        if (fmt != p && (im->listed || !(im->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) && !image_unlist(im, &f)) {
            VK_ONCE(vlist, "view format %d of an image made for format %d (NVN %#x), another texel size: read in the image's format", fmt, im->format, im->nvn_format);
            fmt = im->format; memcpy(swz, (uint8_t[4]){VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A}, 4);
        }
    }
    /* the NVN swizzle composed with the format's remap; a view with its own swizzle replaces the texture's */
    unsigned sel[4] = {t->d.swizzle_r, t->d.swizzle_g, t->d.swizzle_b, t->d.swizzle_a};
    if (v && v->set_swizzle) for (int i = 0; i < 4; i++) sel[i] = v->swz[i];
    uint8_t out[4];
    for (int i = 0; i < 4; i++) out[i] = sel[i] == 0 ? VK_COMPONENT_SWIZZLE_ZERO : sel[i] == 1 ? VK_COMPONENT_SWIZZLE_ONE : sel[i] <= 5 ? swz[sel[i] - 2] : swz[i];
    if (im->depth || im->stencil)   /* a depth / stencil read is (d, 0, 0, 1) before the NVN selectors */
        for (int i = 0; i < 4; i++) out[i] = sel[i] == 0 ? VK_COMPONENT_SWIZZLE_ZERO : sel[i] == 1 ? VK_COMPONENT_SWIZZLE_ONE : sel[i] == 2 ? VK_COMPONENT_SWIZZLE_R :
                                           sel[i] == 5 ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_ZERO;
    uint32_t l0 = v && v->set_levels ? v->base_level : 0, nl = v && v->set_levels && v->nlevels ? v->nlevels : im->levels - (l0 < im->levels ? l0 : 0);
    if (l0 >= im->levels) l0 = im->levels - 1;
    if (l0 + nl > im->levels) nl = im->levels - l0;
    uint32_t total_layers = im->type == VK_IMAGE_TYPE_3D ? 1 : im->layers;
    uint32_t s0 = v && v->set_layers ? v->min_layer : 0, ns = v && v->set_layers && v->nlayers ? v->nlayers : total_layers - (s0 < total_layers ? s0 : 0);
    if (s0 >= total_layers) s0 = total_layers - 1;
    if (s0 + ns > total_layers) ns = total_layers - s0;
    VkImageViewType type = view_type_of(v && v->set_target ? v->target : t->d.target, ns);
    if (im->type == VK_IMAGE_TYPE_1D && type != VK_IMAGE_VIEW_TYPE_1D && type != VK_IMAGE_VIEW_TYPE_1D_ARRAY) type = ns > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    if (im->type == VK_IMAGE_TYPE_3D) type = VK_IMAGE_VIEW_TYPE_3D;
    if (im->type == VK_IMAGE_TYPE_2D && (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY || type == VK_IMAGE_VIEW_TYPE_3D)) type = ns > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if ((type == VK_IMAGE_VIEW_TYPE_CUBE && (ns != 6 || !(im->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))) ||
        (type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY && (ns % 6 || !(im->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)))) type = ns > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if (type == VK_IMAGE_VIEW_TYPE_2D && ns > 1) ns = 1;
    if (type == VK_IMAGE_VIEW_TYPE_1D && ns > 1) ns = 1;
    /* the shader's declared type wins where the two are interchangeable (a 2D texture read as a 2D array and back) */
    if (want != VK_IMAGE_VIEW_TYPE_MAX_ENUM && want != type) {
        if (want == VK_IMAGE_VIEW_TYPE_2D_ARRAY && type == VK_IMAGE_VIEW_TYPE_2D) type = want;
        else if (want == VK_IMAGE_VIEW_TYPE_2D && type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) { type = want; ns = 1; }
        else if (want == VK_IMAGE_VIEW_TYPE_2D_ARRAY && type == VK_IMAGE_VIEW_TYPE_CUBE) type = want;
        else if (want == VK_IMAGE_VIEW_TYPE_CUBE && type == VK_IMAGE_VIEW_TYPE_2D_ARRAY && ns >= 6 && (im->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)) { type = want; ns = 6; }
        else if (want == VK_IMAGE_VIEW_TYPE_1D_ARRAY && type == VK_IMAGE_VIEW_TYPE_1D) type = want;
        else if (want == VK_IMAGE_VIEW_TYPE_1D && type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) { type = want; ns = 1; }
    }
    if (kind == 1) { nl = 1; }   /* storage views see one level */
    if (fmt_out) *fmt_out = fmt;
    return image_view(im, fmt, type, l0, nl, s0, ns, kind == 0 ? out : NULL, kind);
}
VkImageView vk_tex_view(const tex_t *t, const view_t *v, int kind, VkImageViewType want, VkFormat *fmt_out) {
    LOCK();
    vk_image *im = tex_image_locked(t, NULL);
    VkImageView view = im ? view_for(im, t, v, kind, want, fmt_out) : VK_NULL_HANDLE;
    UNLOCK();
    return view;
}
vk_image *vk_tex_attach(const tex_t *t, const view_t *v, VkImageView *view, VkFormat *fmt, uint32_t *w, uint32_t *h, uint32_t *layers) {
    LOCK();
    vk_image *im = tex_image_locked(t, NULL);
    if (!im) { UNLOCK(); return NULL; }
    uint32_t l0 = v && v->set_levels ? v->base_level : 0; if (l0 >= im->levels) l0 = im->levels - 1;
    uint32_t total = im->type == VK_IMAGE_TYPE_3D ? (im->d >> l0 ? im->d >> l0 : 1) : im->layers;
    uint32_t s0 = v && v->set_layers ? v->min_layer : 0, ns = v && v->set_layers && v->nlayers ? v->nlayers : 1;
    if (s0 >= total) s0 = total - 1;
    if (s0 + ns > total) ns = total - s0;
    VkFormat vf = im->format;   /* the view's format, else the texture's own (it may share the storage of another format) */
    vk_fmt f;
    if (!(im->depth || im->stencil) && vk_fmt_of(v && v->set_format ? v->format : t->d.format, &f) && f.vk != im->format &&
        (f.vk == vk_srgb_pair(im->format) || image_unlist(im, &f))) vf = f.vk;
    VkImageViewType type = ns > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if (im->type == VK_IMAGE_TYPE_1D) type = ns > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    VkImageView iv;
    if (im->type == VK_IMAGE_TYPE_3D) iv = image_view(im, vf, VK_IMAGE_VIEW_TYPE_2D_ARRAY, l0, 1, s0, ns, NULL, 2);   /* 2D_ARRAY_COMPATIBLE: depth slices as layers */
    else iv = image_view(im, vf, type, l0, 1, s0, ns, NULL, 2);
    if (view) *view = iv; if (fmt) *fmt = vf;
    if (w) *w = im->w >> l0 ? im->w >> l0 : 1;
    if (h) *h = im->h >> l0 ? im->h >> l0 : 1;
    if (layers) *layers = ns;
    UNLOCK();
    return iv ? im : NULL;
}

/* Rendering into a block-compressed texture (TOTK: BC4 through RGBA16UI; Breath of the Wild: a BC4 3D texture through a
   block-sized view), as nvn_mtl.m blk_aux: Vulkan cannot render to compressed formats, so the pass renders into an
   auxiliary image of the view format sized in blocks, one per texture serial (it keeps the blocks rendered last). */
static nvn_hmap g_blk = {.klen = 8};
vk_image *vk_tex_attach_blk(const tex_t *t, const view_t *v, VkImageView *view, VkFormat *fmt, uint32_t *w, uint32_t *h, uint32_t *layers, vk_blk *b) {
    vk_fmt bf, vf;
    if (!v || !v->set_format || !vk_fmt_of(t->d.format, &bf) || !bf.compressed || !vk_fmt_of(v->format, &vf) || vf.compressed || vf.bw != 1 || vf.bh != 1 || vf.bpb != bf.bpb) return NULL;
    LOCK();
    vk_image *base = tex_image_locked(t, NULL);
    if (!base || base->samples > 1 || base->type == VK_IMAGE_TYPE_1D) { UNLOCK(); return NULL; }
    uint32_t bxs = (base->w + bf.bw - 1) / bf.bw, bys = (base->h + bf.bh - 1) / bf.bh;
    uint64_t k = t->serial;
    vk_image *a = nvn_hm_get(&g_blk, &k);
    if (a && (a->format != vf.vk || a->w != bxs || a->h != bys || a->d != base->d || a->layers != base->layers || a->levels != base->levels)) { nvn_hm_del(&g_blk, &k); image_destroy(a); a = NULL; }
    if (!a) {
        tex_t at = *t; at.d.format = v->format; at.d.width = bxs; at.d.height = bys; at.d.flags &= ~NVN_TEXTURE_FLAGS_IMAGE;
        a = image_new(&at, 0, VK_FORMAT_UNDEFINED);
        if (!a) { UNLOCK(); return NULL; }
        nvn_hm_put(&g_blk, &k, a);
        static unsigned made; if (made++ < 4) VLOG("rendering into compressed texture format %#x (target %d) through view format %#x: via a %ux%ux%u block image", t->d.format, t->d.target, v->format, bxs, bys, a->d);
    }
    uint32_t l0 = v->set_levels ? v->base_level : 0; if (l0 >= a->levels) l0 = a->levels - 1;
    int is3d = a->type == VK_IMAGE_TYPE_3D;
    uint32_t total = is3d ? (base->d >> l0 ? base->d >> l0 : 1) : base->layers;
    uint32_t s0 = v->set_layers ? v->min_layer : 0, ns = v->set_layers && v->nlayers ? v->nlayers : 1;
    if (s0 >= total) s0 = total - 1;
    if (s0 + ns > total) ns = total - s0;
    VkImageView iv = image_view(a, vf.vk, ns > 1 || is3d ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, l0, 1, s0, ns, NULL, 2);
    if (!iv) { UNLOCK(); return NULL; }
    uint32_t lw = base->w >> l0 ? base->w >> l0 : 1, lh = base->h >> l0 ? base->h >> l0 : 1;
    *b = (vk_blk){a, base, l0, s0, ns, (uint32_t)is3d, (lw + bf.bw - 1) / bf.bw, (lh + bf.bh - 1) / bf.bh, lw, lh};
    *view = iv; *fmt = vf.vk; *w = b->bx; *h = b->by; *layers = ns;
    UNLOCK();
    return base;
}
/* the blocks between the two images (a copy between size-compatible formats: the extent counts the source's texels) */
void vk_tex_blk_copy(VkCommandBuffer cb, const vk_blk *b, int to_dst) {
    VkImageSubresourceLayers sub = {VK_IMAGE_ASPECT_COLOR_BIT, b->level, b->is3d ? 0 : b->slice, b->is3d ? 1 : b->nslices};
    VkOffset3D o = {0, 0, b->is3d ? (int32_t)b->slice : 0};
    uint32_t nz = b->is3d ? b->nslices : 1;
    VkImageCopy r = {.srcSubresource = sub, .srcOffset = o, .dstSubresource = sub, .dstOffset = o};
    if (to_dst) { r.extent = (VkExtent3D){b->bx, b->by, nz}; vkCmdCopyImage(cb, b->aux->img, VK_IMAGE_LAYOUT_GENERAL, b->dst->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
    else { r.extent = (VkExtent3D){b->lw, b->lh, nz}; vkCmdCopyImage(cb, b->dst->img, VK_IMAGE_LAYOUT_GENERAL, b->aux->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
}

/* ---------------- texture pool registry ---------------- */
typedef struct { tex_t t; int ok; view_t v; int has_v; VkImageView view; VkImageViewType vtype; uint64_t gen, obj_gen; vk_image *img; uint32_t img_epoch; int fmt_used; } treg_t;
static treg_t *g_treg; static int g_ntreg, g_tpool_n;
static int g_reg_gen = 1;
int vk_tex_registry_gen(void) { return g_reg_gen; }
void mtl_set_tex_pool_n(int n) { g_tpool_n = n; }
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
    if (id < 0) return;
    LOCK();
    if (id >= g_ntreg) { int n = id + 1024; g_treg = realloc(g_treg, sizeof *g_treg * (size_t)n); if (!g_treg) VDIE("texture registry"); memset(g_treg + g_ntreg, 0, sizeof *g_treg * (size_t)(n - g_ntreg)); g_ntreg = n; }
    treg_t *r = &g_treg[id];
    if (t && r->ok && r->has_v == (v != NULL) && !memcmp(&r->t, t, sizeof *t) && (!v || !memcmp(&r->v, v, sizeof *v))) { UNLOCK(); return; }
    g_reg_gen++;
    r->ok = t != NULL; if (t) r->t = *t;
    r->has_v = v != NULL; if (v) r->v = *v; else memset(&r->v, 0, sizeof r->v);
    r->view = VK_NULL_HANDLE; r->img = NULL; r->gen = 0;
    UNLOCK();
}
const tex_t *vk_tex_registered(int tid) { return tid > 0 && tid < g_ntreg && g_treg[tid].ok ? &g_treg[tid].t : NULL; }
int vk_treg_target_format(uint64_t h, int *target, int *format) {
    int tid = (int)(h & 0xfffff); int ok = 0;
    LOCK();
    if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) {
        const treg_t *r = &g_treg[tid];
        if (target) *target = r->has_v && r->v.set_target ? r->v.target : r->t.d.target;
        if (format) *format = r->has_v && r->v.set_format ? r->v.format : r->t.d.format;
        ok = 1;
    }
    UNLOCK();
    return ok;
}
/* The live texture at a finalised registration's storage (the NVN descriptor points at memory). g_tmu held. */
static void treg_follow(treg_t *r) {
    if (!r->t.mem || !nvn_hm_get(&g_dead, &r->t.serial)) return;
    tex_t live;
    if (nvn_tex_live_at(r->t.mem, &live) && live.serial != r->t.serial && live.L.size == r->t.L.size && live.d.format == r->t.d.format && live.d.width == r->t.d.width &&
        live.d.height == r->t.d.height && live.d.depth == r->t.d.depth && live.d.target == r->t.d.target && live.d.levels == r->t.d.levels) {
        static uint64_t n; if (n++ < 4) VLOG("texture pool slot followed its storage from finalised serial %llu to live serial %llu", (unsigned long long)r->t.serial, (unsigned long long)live.serial);
        r->t = live; r->view = VK_NULL_HANDLE;
    }
}
VkImageView vk_tex_by_id(int tid, VkImageViewType want, vk_image **img_out, int *fmt_out) {
    if (img_out) *img_out = NULL;
    if (tid <= 0) return VK_NULL_HANDLE;
    if (g_tpool_n && tid >= g_tpool_n) return VK_NULL_HANDLE;   /* out of the pool: an empty slot by design (reads 0) */
    LOCK();
    if (tid >= g_ntreg || !g_treg[tid].ok) { UNLOCK(); VK_ONCE(treg0, "texture id %d not registered (first): bound as empty", tid); return VK_NULL_HANDLE; }
    treg_t *r = &g_treg[tid];
    treg_follow(r);
    texobj *o = NULL;
    vk_image *im = tex_image_locked(&r->t, &o);
    if (!im) { UNLOCK(); return VK_NULL_HANDLE; }
    if (!r->view || r->img != im || r->img_epoch != im->epoch || r->vtype != want || (o && r->obj_gen != o->gen)) {
        r->view = view_for(im, &r->t, r->has_v ? &r->v : NULL, 0, want, NULL);
        r->img = im; r->img_epoch = im->epoch; r->vtype = want; r->obj_gen = o ? o->gen : 0;
    }
    VkImageView view = r->view;
    if (img_out) *img_out = im;
    if (fmt_out) *fmt_out = r->has_v && r->v.set_format ? r->v.format : r->t.d.format;
    UNLOCK();
    return view;
}
VkImageView vk_tex_storage_by_id(int tid, VkImageViewType want, vk_image **img_out) {
    if (img_out) *img_out = NULL;
    if (tid <= 0) return VK_NULL_HANDLE;
    LOCK();
    if (tid >= g_ntreg || !g_treg[tid].ok) { UNLOCK(); return VK_NULL_HANDLE; }
    treg_t *r = &g_treg[tid];
    treg_follow(r);
    vk_image *im = tex_image_locked(&r->t, NULL);
    VkImageView v = im ? view_for(im, &r->t, r->has_v ? &r->v : NULL, 1, want, NULL) : VK_NULL_HANDLE;
    tex_t t = r->t;
    if (img_out) *img_out = im;
    UNLOCK();
    if (v) vk_tex_mark_owned(&t);   /* a shader writes it */
    return v;
}
const tex_t *vk_tex_slot(int tid) {
    if (tid <= 0 || tid >= g_ntreg || !g_treg[tid].ok) return NULL;
    return &g_treg[tid].t;
}
/* Texel buffer view of a buffer texture in slot tid (TARGET_BUFFER): on the pool buffer, zero copy. */
VkBufferView vk_tex_buffer_view(int tid, int storage) {
    if (tid <= 0) return VK_NULL_HANDLE;
    LOCK();
    if (tid >= g_ntreg || !g_treg[tid].ok) { UNLOCK(); return VK_NULL_HANDLE; }
    treg_t *r = &g_treg[tid];
    texobj *o = NULL; (void)tex_image_locked(&r->t, &o);
    if (!o || o->t.d.target != 10) { UNLOCK(); return VK_NULL_HANDLE; }
    if (o->bview[storage]) { VkBufferView bv = o->bview[storage]; UNLOCK(); return bv; }
    vk_fmt f; int bpb;
    if (!vk_fmt_of(o->t.d.format, &f) || !(bpb = nvn_fmt_bpb(o->t.d.format, NULL, NULL))) { UNLOCK(); VK_ONCE(tbuf0, "buffer texture format %#x unmapped", o->t.d.format); return VK_NULL_HANDLE; }
    VkBuffer buf; VkDeviceSize off, avail;
    uint64_t bytes = (uint64_t)o->t.d.width * (uint64_t)bpb;
    if (!o->t.mem || !vk_pool_lookup((uint64_t)(uintptr_t)o->t.mem, bytes ? bytes : 1, &buf, &off, &avail)) { UNLOCK(); VK_ONCE(tbuf1, "buffer texture storage is not in a memory pool"); return VK_NULL_HANDLE; }
    if (off % V.props.limits.minTexelBufferOffsetAlignment) { UNLOCK(); VK_ONCE(tbuf2, "buffer texture offset %#llx not aligned", (unsigned long long)off); return VK_NULL_HANDLE; }
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = buf, .format = f.vk, .offset = off, .range = bytes};
    if (vkCreateBufferView(V.dev, &ci, NULL, &o->bview[storage]) != VK_SUCCESS) { UNLOCK(); VK_ONCE(tbuf3, "texel buffer view (format %d) not made", f.vk); return VK_NULL_HANDLE; }
    VkBufferView bv = o->bview[storage];
    UNLOCK();
    return bv;
}
void mtl_tex_release(tex_t *t) {
    if (!t->serial) return;
    LOCK();
    uint64_t k = t->serial;
    nvn_hm_put(&g_dead, &k, (void *)1);
#ifdef GAME_BOTW
    if (dead_keep(k)) { UNLOCK(); return; }
#endif
    texobj *o = nvn_hm_del(&g_obj, &k);
    if (o) obj_free(o);
    UNLOCK();
}
/* the pin sweep freed the recording copies naming these serials (the dead set keeps every finalised serial, as nvn_mtl.m) */
void mtl_dead_forget(const uint64_t *serials, size_t n) {
#ifdef GAME_BOTW
    LOCK();
    for (size_t i = 0; i < n; i++) {
        uint64_t k = serials[i];
        if (!nvn_hm_del(&g_dead_kept, &k)) continue;
        texobj *o = nvn_hm_del(&g_obj, &k); if (o) obj_free(o);
    }
    UNLOCK();
#else
    (void)serials; (void)n;
#endif
}
/* After a submission: the texture objects made for finalised serials go (their images after the GPU is done). */
void vk_tex_frame_sweep(uint64_t value) {
    (void)value;
    LOCK();
    for (int i = 0; i < g_ndeadmade; i++) {
        uint64_t k = g_deadmade[i];
#ifdef GAME_BOTW
        if (dead_keep(k)) continue;
#endif
        texobj *o = nvn_hm_del(&g_obj, &k); if (o) obj_free(o);
    }
    g_ndeadmade = 0;
    UNLOCK();
}

/* ---------------- samplers ---------------- */
typedef struct { samp_par p; int ok; VkSampler s[2]; } sreg_t;
static sreg_t *g_sreg; static int g_nsreg;
typedef struct { VkSamplerCreateInfo ci; VkSamplerCustomBorderColorCreateInfoEXT cb; VkSampler s; } samp_cache_ent;
static samp_cache_ent *g_scache; static int g_nscache, g_capscache;
void mtl_reg_samp(int id, const samp_par *p) {
    if (id < 0) return;
    LOCK();
    if (id >= g_nsreg) { int n = id + 256; g_sreg = realloc(g_sreg, sizeof *g_sreg * (size_t)n); if (!g_sreg) VDIE("sampler registry"); memset(g_sreg + g_nsreg, 0, sizeof *g_sreg * (size_t)(n - g_nsreg)); g_nsreg = n; }
    g_sreg[id].p = *p; g_sreg[id].ok = 1; g_sreg[id].s[0] = g_sreg[id].s[1] = VK_NULL_HANDLE;   /* VkSamplers live in the cache */
    g_reg_gen++;
    UNLOCK();
}
static VkSamplerAddressMode wrapm(int w) {
    switch (w) { case 0: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE; case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}
static VkSampler sampler_get(VkSamplerCreateInfo *ci, const float border[4]) {   /* g_tmu held; dedupe (the device allows 4000) */
    VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    int custom = 0;
    if (ci->addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci->addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci->addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) {
        int grey = border[0] == border[1] && border[1] == border[2];
        if (grey && border[0] == 0 && border[3] == 0) ci->borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (grey && border[0] == 0 && border[3] == 1) ci->borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (grey && border[0] == 1 && border[3] == 1) ci->borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (!V.custom_border) {
            /* no custom borders (vk_device.c): the edge texel instead, as the Metal backend's sampler does inside the
               texture. Its grass interaction map (0.5 grey border, two targets scrolled with the player) reads the
               "untouched" value at its far edges from them; transparent black there laid the grass flat (nvn_mtl.m) */
            if (ci->addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) ci->addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (ci->addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) ci->addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            if (ci->addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) ci->addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            ci->borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
            static unsigned said; if (said++ < 4) VLOG("sampler border colour %g %g %g %g: clamped to the edge instead (TOTK_VK_CUSTOM_BORDER=1: exact)", border[0], border[1], border[2], border[3]);
        } else { custom = 1; ci->borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT; memcpy(cb.customBorderColor.float32, border, 16); cb.format = VK_FORMAT_UNDEFINED; }
    } else ci->borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    for (int i = 0; i < g_nscache; i++) {
        samp_cache_ent *e = &g_scache[i];
        if (!memcmp(&e->ci.flags, &ci->flags, sizeof *ci - offsetof(VkSamplerCreateInfo, flags)) && (!custom || !memcmp(e->cb.customBorderColor.float32, cb.customBorderColor.float32, 16))) return e->s;
    }
    if (custom) ci->pNext = &cb;
    VkSampler s;
    if (vkCreateSampler(V.dev, ci, NULL, &s) != VK_SUCCESS) { VK_ONCE(samp0, "sampler not made (%d in use)", g_nscache); ci->pNext = NULL; return g_nscache ? g_scache[0].s : VK_NULL_HANDLE; }
    ci->pNext = NULL;
    if (g_nscache == g_capscache) { g_capscache = g_capscache ? g_capscache * 2 : 256; g_scache = realloc(g_scache, sizeof *g_scache * (size_t)g_capscache); if (!g_scache) VDIE("sampler cache"); }
    g_scache[g_nscache++] = (samp_cache_ent){*ci, cb, s};
    if (g_nscache == 3000) VLOG("3000 distinct samplers (the device allows %u)", V.props.limits.maxSamplerAllocationCount);
    return s;
}
VkSampler vk_samp_by_id(int sid, int nearest) {
    LOCK();
    if (sid < 0 || sid >= g_nsreg || !g_sreg[sid].ok) { UNLOCK(); VK_ONCE(sreg0, "sampler id %d not registered (first): default sampler", sid); return vk_samp_default(); }
    sreg_t *r = &g_sreg[sid];
    if (r->s[nearest]) { VkSampler s = r->s[nearest]; UNLOCK(); return s; }
    const samp_par *p = &r->p;
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = (p->minf == 1 || p->minf == 3 || p->minf == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = (p->minf == 2 || p->minf == 3) ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = wrapm(p->ws); ci.addressModeV = wrapm(p->wt); ci.addressModeW = wrapm(p->wr);
    ci.minLod = p->lod_min < 0 ? 0 : p->lod_min; ci.maxLod = p->lod_max < ci.minLod ? ci.minLod : p->lod_max;
    if (p->minf <= 1) ci.maxLod = ci.minLod + 0.25f;   /* "not mipmapped": the base level of the view only */
    ci.mipLodBias = p->lod_bias > 15 ? 15 : p->lod_bias < -15 ? -15 : p->lod_bias;
    ci.anisotropyEnable = p->aniso > 1; ci.maxAnisotropy = p->aniso > 16 ? 16 : p->aniso < 1 ? 1 : p->aniso;
    ci.compareEnable = p->cmp_mode != 0; ci.compareOp = vk_compare(p->cmp_func);
    if (p->reduction) VK_ONCE(red, "sampler min / max reduction not supported by this device: averaged");
    if (nearest) {   /* non-compare reads of a depth texture land on one texel, as on the Switch (Service.cs MatchSwitchDepthSampling) */
        ci.magFilter = ci.minFilter = VK_FILTER_NEAREST; ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; ci.anisotropyEnable = VK_FALSE; ci.maxAnisotropy = 1;
        ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; ci.minLod = 0; ci.maxLod = 0.25f; ci.mipLodBias = 0;
    }
    VkSampler s = sampler_get(&ci, p->border);
    r->s[nearest] = s;
    UNLOCK();
    return s;
}
VkSampler vk_samp_default(void) {
    static VkSampler s;
    if (!s) { VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST, .maxLod = VK_LOD_CLAMP_NONE};
              VK(vkCreateSampler(V.dev, &ci, NULL, &s)); }
    return s;
}

/* ---------------- dummies: unbound slots read 0 (D3D / empty-slot semantics) ---------------- */
static VkImageView g_dummy[8][4], g_dummy_st[8][4];
static vk_image *dummy_image(VkImageViewType type, int cls, int storage) {
    tex_t t; memset(&t, 0, sizeof t);
    static const int targets[] = {0, 1, 2, 8, 3, 4, 9};   /* by VkImageViewType 1D, 2D, 3D, CUBE, 1D_ARRAY, 2D_ARRAY, CUBE_ARRAY */
    t.d.target = targets[type < 7 ? type : 1]; t.d.width = t.d.height = 1; t.d.depth = type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY ? 6 : 1; t.d.levels = 1;
    t.d.format = cls == 3 ? 0x34 : cls == 1 ? (storage ? 0x2F : 0x2F) : cls == 2 ? 0x30 : (storage ? 0x29 : 0x25);
    if (storage) t.d.flags = NVN_TEXTURE_FLAGS_IMAGE;
    t.serial = 0;
    return image_new(&t, 0, VK_FORMAT_UNDEFINED);
}
VkImageView vk_dummy_view(VkImageViewType type, int cls) {
    if (type > VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) type = VK_IMAGE_VIEW_TYPE_2D;
    LOCK();
    VkImageView v = g_dummy[type][cls & 3];
    if (!v) {
        vk_image *im = dummy_image(type, cls & 3, 0);
        uint8_t swz[4] = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
        if (im) v = g_dummy[type][cls & 3] = image_view(im, im->format, type, 0, 1, 0, im->layers, swz, 0);
    }
    UNLOCK();
    return v;
}
VkImageView vk_dummy_storage(VkImageViewType type, int cls) {
    if (type > VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) type = VK_IMAGE_VIEW_TYPE_2D;
    if (type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    LOCK();
    VkImageView v = g_dummy_st[type][cls & 3];
    if (!v) {
        vk_image *im = dummy_image(type, cls == 3 ? 0 : cls & 3, 1);
        if (im) v = g_dummy_st[type][cls & 3] = image_view(im, im->format, type, 0, 1, 0, im->layers, NULL, 1);
    }
    UNLOCK();
    return v;
}
VkBufferView vk_dummy_texel(int storage, int cls) {
    static VkBufferView bv[2][3]; static VkBuffer b; static VmaAllocation a;
    LOCK();
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 1u << 20, .usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
        VmaAllocationInfo info; VK(vmaCreateBuffer(V.vma, &bi, &ai, &b, &a, &info)); memset(info.pMappedData, 0, 1u << 20);
    }
    int c = cls == 1 ? 1 : cls == 2 ? 2 : 0;
    if (!bv[storage][c]) {
        VkFormat f = c == 1 ? VK_FORMAT_R32G32B32A32_UINT : c == 2 ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT;
        VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = b, .format = f, .offset = 0, .range = VK_WHOLE_SIZE};
        VK(vkCreateBufferView(V.dev, &ci, NULL, &bv[storage][c]));
    }
    VkBufferView r = bv[storage][c];
    UNLOCK();
    return r;
}

/* ---------------- CPU texel access ---------------- */
static int gpu_content(const tex_t *t) { LOCK(); texobj *o = nvn_hm_get(&g_obj, &t->serial); int r = !tex_mirrored(t) || (o && o->owned) || is_owned(t->serial); UNLOCK(); return r; }
/* GPU image -> linear rows (whole region), after all submitted GPU work (the caller waited per NVN rules) */
static int image_read(vk_image *im, unsigned level, unsigned layer, int x, int y, int z, uint32_t w, uint32_t h, uint32_t d, uint8_t *dst, size_t row_bytes, size_t img_bytes, int bpb, int bw, int bh) {
    extern void mtl_finish(void);
    mtl_finish();
    size_t wb = (size_t)((w + bw - 1) / bw) * bpb, hb = (h + bh - 1) / bh, total = wb * hb * d;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VkBuffer buf; VmaAllocation a; VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &ai, &buf, &a, &info) != VK_SUCCESS) return 0;
    VkCommandBuffer cb = vk_oneoff_begin();
    VkImageAspectFlags aspect = im->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkBufferImageCopy rg = {.bufferOffset = 0, .imageSubresource = {aspect, level, im->type == VK_IMAGE_TYPE_3D ? 0 : layer, im->type == VK_IMAGE_TYPE_3D ? 1 : d},
                            .imageOffset = {x, y, im->type == VK_IMAGE_TYPE_3D ? z : 0}, .imageExtent = {w, h, im->type == VK_IMAGE_TYPE_3D ? d : 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &rg);
    vk_oneoff_wait(vk_oneoff_end_submit(cb));
    vmaInvalidateAllocation(V.vma, a, 0, VK_WHOLE_SIZE);
    for (uint32_t k = 0; k < d; k++) for (size_t r = 0; r < hb; r++) memcpy(dst + k * img_bytes + r * row_bytes, (uint8_t *)info.pMappedData + k * wb * hb + r * wb, wb);
    vmaDestroyBuffer(V.vma, buf, a);
    return 1;
}
static int image_write(vk_image *im, unsigned level, unsigned layer, int x, int y, int z, uint32_t w, uint32_t h, uint32_t d, const uint8_t *src, size_t row_bytes, size_t img_bytes, int bpb, int bw, int bh) {
    extern void mtl_finish(void);
    mtl_finish();
    size_t wb = (size_t)((w + bw - 1) / bw) * bpb, hb = (h + bh - 1) / bh, total = wb * hb * d;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total ? total : 16, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
    VkBuffer buf; VmaAllocation a; VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &ai, &buf, &a, &info) != VK_SUCCESS) return 0;
    for (uint32_t k = 0; k < d; k++) for (size_t r = 0; r < hb; r++) memcpy((uint8_t *)info.pMappedData + k * wb * hb + r * wb, src + k * img_bytes + r * row_bytes, wb);
    VkCommandBuffer cb = vk_oneoff_begin();
    VkImageAspectFlags aspect = im->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkBufferImageCopy rg = {.bufferOffset = 0, .imageSubresource = {aspect, level, im->type == VK_IMAGE_TYPE_3D ? 0 : layer, im->type == VK_IMAGE_TYPE_3D ? 1 : d},
                            .imageOffset = {x, y, im->type == VK_IMAGE_TYPE_3D ? z : 0}, .imageExtent = {w, h, im->type == VK_IMAGE_TYPE_3D ? d : 1}};
    vkCmdCopyBufferToImage(cb, buf, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    vk_oneoff_wait(vk_oneoff_end_submit(cb));
    vmaDestroyBuffer(V.vma, buf, a);
    return 1;
}
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) return;
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    if (tex_mirrored(t) && !gpu_content(t)) {   /* the pool bytes are the truth */
        uint32_t lw = t->d.width >> level ? t->d.width >> level : 1, lh = t->d.height >> level ? t->d.height >> level : 1, lz = is3d ? (t->d.depth >> level ? t->d.depth >> level : 1) : 1;
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        uint8_t *lin = malloc(lp * lr * lz); if (!lin) return;
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) { free(lin); return; }
            const uint8_t *src = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, src + (y0 + y) * lp + x0, wb);
        }
        free(lin);
        return;
    }
    vk_image *im = vk_tex_image(t);
    if (!im) { memset(dst, 0, img * nz); return; }
    image_read(im, level, is3d ? 0 : layer0 + r->zoffset, r->xoffset, r->yoffset, r->zoffset, (uint32_t)r->width, (uint32_t)r->height, (uint32_t)nz, dst, row, img, bpb, bw, bh);
}
void mtl_write_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) VDIE("WriteTexels: unknown format %d", t->d.format);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    if (!r->width || !r->height) return;
    if (tex_mirrored(t)) {   /* block-linear bytes in the pool (the dirty tracker re-uploads), and the image too if the GPU owns it */
        uint32_t lw = t->d.width >> level ? t->d.width >> level : 1, lh = t->d.height >> level ? t->d.height >> level : 1, lz = is3d ? (t->d.depth >> level ? t->d.depth >> level : 1) : 1;
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        dm_guest_write_prepare(t->mem, t->L.size);
        uint8_t *lin = malloc(lp * lr * lz); if (!lin) VDIE("WriteTexels: out of memory");
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) VDIE("WriteTexels: unsupported layout");
            uint8_t *d = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(d + (y0 + y) * lp + x0, src + z * img + y * row, wb);
            if (!is3d || z == nz - 1) if (nvn_swizzle(&t->d, level, layer, lin, lp, t->mem)) VDIE("WriteTexels: unsupported layout");
        }
        free(lin);
        if (!gpu_content(t)) return;
    }
    vk_image *im = vk_tex_image(t);
    if (!im) VDIE("WriteTexels: texture %llu has no Vulkan image", (unsigned long long)t->serial);
    image_write(im, level, is3d ? 0 : layer0 + r->zoffset, r->xoffset, r->yoffset, r->zoffset, (uint32_t)r->width, (uint32_t)r->height, (uint32_t)nz, src, row, img, bpb, bw, bh);
}

/* ---------------- write-back of GPU renderings into CPU-visible pools ---------------- */
/* value: the timeline value of the submission that rendered the texture last (UINT64_MAX while it is open).
   Write-back staging (nvn_mtl.m BOTW_WB_*, docs/botw-v1.9.0-port.md): the game invalidates 1x1 and 3x3 renderings every
   frame. A submission that renders a small one (WB_STAGE_MAX) copies it into the entry's host-visible stage at its end
   (vk_tex_wb_stage), so the invalidate waits for that submission only and swizzles from the stage, where it waited for
   all GPU work and read each image back. stage_value: the submission whose end filled the stage. */
typedef struct { tex_t t; uint64_t stamp, value, stage_value, written; uint8_t *snap; uint32_t snap_n; int stage_open; VkBuffer stage; VmaAllocation stage_alloc; void *stage_ptr; size_t stage_n; } wb_ent;
static nvn_hmap g_wb = {.klen = 8};
enum { WB_SNAP_MAX = 4096, WB_STAGE_MAX = 256 << 10 };
static uint64_t g_wb_sub[64]; static int g_wb_nsub;   /* replay thread: serials rendered in the open submission */
static uint64_t g_wb_staged, g_wb_from_stage;
static void wb_ent_free(wb_ent *e) {   /* the stage may be the target of a copy in the open submission */
    if (e->stage) vk_defer_buffer(UINT64_MAX, e->stage, e->stage_alloc);
    free(e->snap); free(e);
}
/* level l of a written-back texture as the NVN-linear rows nvn_swizzle reads: its size, pitch and extent */
static size_t wb_level_size(const tex_t *t, unsigned l, size_t *pitch, uint32_t *w, uint32_t *h) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    uint32_t lw = t->d.width >> l ? t->d.width >> l : 1, lh = t->d.height >> l ? t->d.height >> l : 1;
    size_t lp = (size_t)((lw + bw - 1) / bw) * bpb;
    if (pitch) *pitch = lp; if (w) *w = lw; if (h) *h = lh;
    return lp * ((lh + bh - 1) / bh);
}
static unsigned wb_levels(const tex_t *t, const vk_image *im) { unsigned n = t->d.levels ? t->d.levels : 1; return im && im->levels < n ? im->levels : n; }
void vk_tex_note_gpu_write(const tex_t *t) {
    if (!t || !t->mem || !t->L.size || !tex_mirrored(t)) return;
    uint64_t stamp = nvn_dirty_now(), k = t->serial;
    nvn_dirty_protect(t->mem, t->L.size);
    LOCK();
    wb_ent *e = nvn_hm_get(&g_wb, &k);
    if (!e) { e = calloc(1, sizeof *e); if (!e) VDIE("write-back record"); nvn_hm_put(&g_wb, &k, e); }
    e->t = *t; e->stamp = stamp; e->value = UINT64_MAX;
    if (t->L.size <= WB_SNAP_MAX) { if (e->snap_n != t->L.size) { free(e->snap); e->snap = malloc(t->L.size); e->snap_n = e->snap ? (uint32_t)t->L.size : 0; } if (e->snap) memcpy(e->snap, t->mem, t->L.size); }
    UNLOCK();
    int i = 0; while (i < g_wb_nsub && g_wb_sub[i] != k) i++;
    if (i == g_wb_nsub && g_wb_nsub < 64) g_wb_sub[g_wb_nsub++] = k;
}
/* mtl_sub_commit, outside any render pass, before the command buffer ends: the small renderings of this submission into
   their stages */
void vk_tex_wb_stage(VkCommandBuffer cb) {
    int copied = 0;
    LOCK();
    for (int i = 0; i < g_wb_nsub; i++) {
        uint64_t k = g_wb_sub[i];
        wb_ent *e = nvn_hm_get(&g_wb, &k);
        if (!e || e->t.d.target == 2 || e->t.d.samples > 1) continue;
        texobj *o = nvn_hm_get(&g_obj, &k);
        vk_image *im = o ? o->img : NULL;
        vk_fmt f;
        if (!im || im->depth || im->stencil || im->samples > 1 || im->type != VK_IMAGE_TYPE_2D || !vk_fmt_of(e->t.d.format, &f) || f.compressed || f.vk != im->format) continue;
        unsigned layers = e->t.L.layers ? e->t.L.layers : 1, levels = wb_levels(&e->t, im);
        if (layers > im->layers) continue;
        size_t total = 0;
        for (unsigned l = 0; l < levels; l++) total += wb_level_size(&e->t, l, NULL, NULL, NULL) * layers;
        if (!total || total > WB_STAGE_MAX) continue;
        if (!e->stage || e->stage_n < total) {
            if (e->stage) { vk_defer_buffer(UINT64_MAX, e->stage, e->stage_alloc); e->stage = VK_NULL_HANDLE; }
            VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
            VmaAllocationInfo info;
            if (vmaCreateBuffer(V.vma, &bi, &ai, &e->stage, &e->stage_alloc, &info) != VK_SUCCESS) { e->stage = VK_NULL_HANDLE; continue; }
            e->stage_ptr = info.pMappedData; e->stage_n = total;
        }
        if (!copied++) vk_barrier_all(cb);   /* the submission's renderings are complete */
        VkBufferImageCopy rg[32]; uint32_t nr = 0; size_t off = 0;
        for (unsigned l = 0; l < levels; l++) for (unsigned y = 0; y < layers; y++) {
            uint32_t lw, lh; size_t n = wb_level_size(&e->t, l, NULL, &lw, &lh);
            if (nr == 32) { vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, e->stage, nr, rg); nr = 0; }
            rg[nr++] = (VkBufferImageCopy){.bufferOffset = off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, y, 1}, .imageExtent = {lw, lh, 1}};
            off += n;
        }
        if (nr) vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, e->stage, nr, rg);
        e->stage_open = 1; g_wb_staged++;
    }
    UNLOCK();
    if (copied) {
        VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &hb};
        vkCmdPipelineBarrier2(cb, &di);
    }
}
/* A staged rendering is written into memory when its submission completes, as the GPU writes memory on hardware
   (completion thread; the callbacks run before the timeline value counts as reached, so a fence wait for the submission
   sees the bytes). The game's invalidate then waits for nothing: it reads what the GPU finished last (Breath of the Wild
   invalidates its luminance targets every frame while the frame that renders them is still on the GPU, and waiting for
   it there serialised the CPU and the GPU: about 10 ms a frame). A CPU write since the rendering keeps the CPU's bytes. */
static int wb_from_stage(wb_ent *e);
static int wb_log(void);
static void wb_log_texel(const tex_t *t);
/* TOTK_VK_WB_WAIT=1: the invalidate waits for the renderings again and writes them back itself */
static int wb_async(void) { static int on = -1; if (on < 0) { const char *e = getenv("TOTK_VK_WB_WAIT"); on = !(e && !strcmp(e, "1")); } return on; }
typedef struct { uint64_t serial, value; } wb_due;
static int wb_fresh(const wb_ent *e) {
    return !nvn_dirty_since(e->t.mem, e->t.L.size, e->stamp) || (e->snap && e->snap_n == e->t.L.size && !memcmp(e->snap, e->t.mem, e->snap_n));
}
static void wb_done(void *arg) {   /* the submission that filled the stage has completed */
    wb_due d = *(wb_due *)arg; free(arg);
    uint64_t k = d.serial, value = d.value; tex_t t;
    LOCK();
    wb_ent *e = nvn_hm_get(&g_wb, &k);
    int due = e && e->stage && e->stage_value == value && e->value != UINT64_MAX && e->value <= value && e->written < value;
    if (due) t = e->t;
    UNLOCK();
    if (!due || t.d.target == 2 || t.d.samples > 1) return;
    void *tok = nvn_dirty_tracked(t.mem, t.L.size) ? nvn_dirty_gpu_write_begin(t.mem, t.L.size) : NULL;
    if (!tok) dm_guest_write_prepare(t.mem, t.L.size);
    LOCK();
    e = nvn_hm_get(&g_wb, &k);   /* still the same rendering, and the CPU has not written the bytes since */
    if (e && e->stage_value == value && e->value != UINT64_MAX && e->value <= value && wb_fresh(e) && wb_from_stage(e) == 1) {
        e->written = value;
        if (e->t.d.width <= 4 && e->t.d.height <= 4 && wb_log()) wb_log_texel(&e->t);
    }
    UNLOCK();
    nvn_dirty_gpu_write_end(tok);
}
/* mtl_sub_commit, after the submission: the renderings noted in it and the stages it filled get its timeline value */
void vk_tex_wb_submitted(uint64_t value) {
    uint64_t due[64]; int nd = 0;
    LOCK();
    for (int i = 0; i < g_wb_nsub; i++) {
        wb_ent *e = nvn_hm_get(&g_wb, &g_wb_sub[i]);
        if (!e) continue;
        if (e->value == UINT64_MAX) e->value = value;
        if (e->stage_open) { e->stage_value = value; e->stage_open = 0; if (wb_async() && nd < 64) due[nd++] = g_wb_sub[i]; }
    }
    g_wb_nsub = 0;
    UNLOCK();
    for (int i = 0; i < nd; i++) { wb_due *d = malloc(sizeof *d); if (!d) VDIE("write-back"); *d = (wb_due){due[i], value}; vk_on_done(value, wb_done, d); }
}
/* A GPU copy into pool memory is newer than an earlier GPU rendering of a texture over the same bytes: that rendering's
   write-back record is dropped, else the invalidate wrote it back over the copy (the black save thumbnail; nvn_mtl.m
   botw_gpu_copy_supersedes). Replay thread, in GPU order. */
void vk_tex_wb_supersede(const void *mem, size_t n) {
    const uint8_t *a = mem, *z = a + n;
    wb_ent *list[64]; int nl = 0;
    LOCK();
    for (uint32_t i = 0; g_wb.n && g_wb.vals && i < g_wb.cap && nl < 64; i++) {
        wb_ent *e = g_wb.vals[i];
        if (!e || e->t.mem >= z || e->t.mem + e->t.L.size <= a) continue;
        uint64_t k = e->t.serial; nvn_hm_del(&g_wb, &k); list[nl++] = e; i = (uint32_t)-1;
    }
    UNLOCK();
    for (int i = 0; i < nl; i++) wb_ent_free(list[i]);
}
static int wb_from_stage(wb_ent *e) {   /* 1 written back, 0 failed, -1 not staged after its last rendering */
    const tex_t *t = &e->t;
    if (!e->stage || !e->stage_value || e->value == UINT64_MAX || e->stage_value < e->value) return -1;
    vmaInvalidateAllocation(V.vma, e->stage_alloc, 0, VK_WHOLE_SIZE);
    unsigned layers = t->L.layers ? t->L.layers : 1, levels = wb_levels(t, NULL); size_t off = 0;
    texobj *o = nvn_hm_get(&g_obj, &t->serial); if (o && o->img) levels = wb_levels(t, o->img);
    for (unsigned l = 0; l < levels; l++) for (unsigned y = 0; y < layers; y++) {
        size_t lp, n = wb_level_size(t, l, &lp, NULL, NULL);
        if (off + n > e->stage_n || nvn_swizzle(&t->d, l, y, (const uint8_t *)e->stage_ptr + off, lp, t->mem)) return 0;
        off += n;
    }
    if (++g_wb_from_stage == 1 || g_wb_from_stage % 5000 == 0)
        VLOG("[writeback] %llu written back from their submission's own copy (%llu copies staged)", (unsigned long long)g_wb_from_stage, (unsigned long long)g_wb_staged);
    return 1;
}
/* TOTK_VK_WB_LOG=1 (diagnostics): the first texel of the small renderings the game reads back (its luminance targets),
   as written into memory: the first 8, then every 200th */
static int wb_log(void) { static int v = -1; if (v < 0) v = getenv("TOTK_VK_WB_LOG") != NULL; return v; }
static void wb_log_texel(const tex_t *t) {
    static uint64_t n; if (++n > 8 && n % 200) return;
    float c[4] = {0, 0, 0, 0}; const uint8_t *m = t->mem;   /* texel (0, 0) is at offset 0 in both layouts */
    if (t->d.format == 0x2E) memcpy(c, m, 16);
    else if (t->d.format == 0x29) for (int i = 0; i < 4; i++) { _Float16 h; memcpy(&h, m + 2 * i, 2); c[i] = (float)h; }
    else if (t->d.format == 0x0A) memcpy(c, m, 4);   /* R32F */
    else if (t->d.format == 0x05) { _Float16 h; memcpy(&h, m, 2); c[0] = (float)h; }
    else { VLOG("[writeback-log] #%llu serial %llu %ux%u format %#x: not decoded", (unsigned long long)n, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format); return; }
    VLOG("[writeback-log] #%llu serial %llu %ux%u format %#x texel 0: %g %g %g %g", (unsigned long long)n, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, c[0], c[1], c[2], c[3]);
}
/* the game thread's waits for GPU renderings it reads back (the frame timing line) */
static _Atomic uint64_t g_inv_wait_ns, g_inv_n;
void vk_tex_wait_stats(uint64_t *ns, uint64_t *n) { *ns = atomic_exchange(&g_inv_wait_ns, 0); *n = atomic_exchange(&g_inv_n, 0); }
void mtl_pool_invalidate(const pool_t *p, uint64_t off, uint64_t n) {
    if (!p || !p->mem) return;
    const uint8_t *a = p->mem + off, *z = a + n;
    int hit = 0;
    LOCK();
    for (uint32_t i = 0; i < g_wb.cap && g_wb.n && !hit; i++) { wb_ent *e = g_wb.vals ? g_wb.vals[i] : NULL; hit = e && e->t.mem < z && e->t.mem + e->t.L.size > a; }
    UNLOCK();
    if (!hit) return;
    /* the renderings queued so far are submitted, so their timeline values are known (vk_tex_wb_submitted) */
    { extern void nvn_x_drain(void) __attribute__((weak)); if (nvn_x_drain) nvn_x_drain(); }
    wb_ent *list[512]; int nl = 0;
    LOCK();
    for (uint32_t i = 0; i < g_wb.cap && nl < 512; i++) {
        wb_ent *e = g_wb.vals ? g_wb.vals[i] : NULL;
        if (!e || e->t.mem >= z || e->t.mem + e->t.L.size <= a) continue;
        if (wb_async() && e->stage) continue;   /* written back when its submission completes (wb_done): nothing to wait for */
        uint64_t k = e->t.serial; nvn_hm_del(&g_wb, &k); list[nl++] = e; i = (uint32_t)-1;
    }
    UNLOCK();
    if (!nl) return;
    uint64_t need = 0; int open = 0;   /* wait for the listed renderings and stages only */
    for (int i = 0; i < nl; i++) {
        if (list[i]->value == UINT64_MAX) open = 1; else if (list[i]->value > need) need = list[i]->value;
        if (list[i]->stage && list[i]->stage_value > need) need = list[i]->stage_value;
    }
    { struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
      if (open) { extern void mtl_finish(void); mtl_finish(); } else if (need) vk_wait_value(need, UINT64_MAX);
      clock_gettime(CLOCK_MONOTONIC, &t1); atomic_fetch_add(&g_inv_wait_ns, (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec))); atomic_fetch_add(&g_inv_n, 1); }
    int fresh[512];
    for (int i = 0; i < nl; i++) {
        fresh[i] = !nvn_dirty_since(list[i]->t.mem, list[i]->t.L.size, list[i]->stamp);
        if (!fresh[i] && list[i]->snap && list[i]->snap_n == list[i]->t.L.size && !memcmp(list[i]->snap, list[i]->t.mem, list[i]->snap_n)) fresh[i] = 1;
    }
    for (int i = 0; i < nl; i++) for (int j = 0; j < nl; j++)
        if (i != j && fresh[i] && fresh[j] && list[j]->t.mem == list[i]->t.mem && list[j]->stamp > list[i]->stamp) { fresh[i] = 2; break; }
    static uint64_t done, skipped;
    for (int i = 0; i < nl; i++) {
        wb_ent *e = list[i]; const tex_t *t = &e->t;
        if (fresh[i] == 1 && t->d.target != 2 && t->d.samples <= 1) {
            vk_image *im = vk_tex_image(t);
            int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
            if (im && bpb) {
                void *tok = nvn_dirty_tracked(t->mem, t->L.size) ? nvn_dirty_gpu_write_begin(t->mem, t->L.size) : NULL;
                if (!tok) dm_guest_write_prepare(t->mem, t->L.size);
                LOCK(); int ok = wb_from_stage(e); UNLOCK();
                if (ok < 0) {   /* not staged: read the image back */
                    unsigned layers = t->L.layers ? t->L.layers : 1; ok = 1;
                    for (unsigned l = 0; l < im->levels && ok; l++) for (unsigned y = 0; y < layers && ok; y++) {
                        uint32_t lw = t->d.width >> l ? t->d.width >> l : 1, lh = t->d.height >> l ? t->d.height >> l : 1;
                        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
                        uint8_t *lin = malloc(lp * lr); if (!lin) { ok = 0; break; }
                        ok = image_read(im, l, y, 0, 0, 0, lw, lh, 1, lin, lp, lp * lr, bpb, bw, bh) && !nvn_swizzle(&t->d, l, y, lin, lp, t->mem);
                        free(lin);
                    }
                }
                nvn_dirty_gpu_write_end(tok);
                if (ok && t->d.width <= 4 && t->d.height <= 4 && wb_log()) wb_log_texel(t);
                if (ok && (++done <= 8 || done % 500 == 0)) VLOG("[writeback] serial %llu %ux%u format %#x: GPU rendering written back to memory (%llu so far, %llu skipped)",
                                                               (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, (unsigned long long)done, (unsigned long long)skipped);
            }
        } else if (!fresh[i]) skipped++;
        wb_ent_free(e);
    }
}
void vk_tex_wb_flush_list(void) {}

/* ---------------- statistics for the shared code ---------------- */
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = g_uploads; *reuploads = g_reuploads; }
uint64_t mtl_pkg_miss(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return g_upload_bytes; }
void vk_tex_init(void) {}
void vk_tex_stats(uint64_t *made, uint64_t *live) { *made = g_images_made; *live = g_images_live; }
