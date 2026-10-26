/* Presenting on Linux: the game's swapchain texture is copied (scaled, letterboxed to its aspect) into the window's
   Vulkan swapchain by a present thread, as the Metal backend draws it into its CAMetalLayer. mtl_present only queues
   the frame and returns a ticket; mtl_present_wait waits until that frame's copy has been submitted to the queue, after
   which the game may draw into the texture again (later GPU work is ordered after the copy on the one queue).
   The game paces itself (hle_nvn.c: display rate, present interval); the swapchain presents in mailbox mode where the
   compositor offers it, else FIFO. */

#include "vk_priv.h"
#include "frontend_sdl.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define PQ 4
typedef struct { tex_t t; double period; int crop[4]; int has; uint64_t frame; } pq_item;
static void pt_swapchain(VkSwapchainKHR sc, int pt, int pid2);
static void pt_poll(void);
/* presentation times and the vblank clock (pt_poll, below) */
static struct { int on, ids; uint64_t counter, answers, dropped, domain; PFN_vkSetSwapchainPresentTimingQueueSizeEXT qsize; PFN_vkGetPastPresentationTimingEXT past;
                PFN_vkGetSwapchainTimingPropertiesEXT props; PFN_vkGetSwapchainTimeDomainPropertiesEXT domains; } g_pt;
static struct { pthread_mutex_t mu; uint64_t t, n, period; int bad; } g_vb = {PTHREAD_MUTEX_INITIALIZER};
static pq_item g_pq[PQ];
static uint64_t g_pq_in, g_pq_read, g_pq_done;
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER;
#define PRING 256
static double g_pres_t[PRING]; static uint64_t g_pres_n[PRING];
static uint64_t g_present_n;

static VkSurfaceKHR g_surf;
static VkSwapchainKHR g_sc;
static VkImage g_scimg[8]; static VkImageView g_scview[8], g_scview_old[8]; static uint32_t g_nsc, g_nsc_old;   /* views: FSR's RCAS draws into them */
static VkFormat g_scfmt; static VkExtent2D g_ext; static int g_sc_srgb = -1;
static VkSemaphore g_acq[8], g_rel[8]; static unsigned g_acq_i;
static VkCommandPool g_ppool;
static int g_have_window;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
/* The replaced swapchain is destroyed a few presents later (retire_old), not at once: on Wayland, waiting for the device to
   go idle (or destroying the old swapchain) while the compositor still holds its last buffer hung the present thread
   when niri resized the window (Super+F): the compositor lets that buffer go only after a new frame comes. */
static VkSwapchainKHR g_sc_old; static unsigned g_sc_old_wait;
static void retire_old(int now) {
    if (!g_sc_old || (!now && --g_sc_old_wait)) return;
    for (uint32_t i = 0; i < g_nsc_old; i++) if (g_scview_old[i]) vkDestroyImageView(V.dev, g_scview_old[i], NULL);
    g_nsc_old = 0;
    vkDestroySwapchainKHR(V.dev, g_sc_old, NULL); g_sc_old = VK_NULL_HANDLE;
}
static void swapchain_make(int srgb) {
    if (!g_surf) return;
    if (!vk_wait_value(vk_last_value(), 2000000000ull)) VK_ONCE(scwait, "swapchain: GPU work still running after 2 s; recreating anyway");   /* our blits into the old images */
    retire_old(1);
    VkSurfaceCapabilitiesKHR caps; VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(V.pd, g_surf, &caps));
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(V.pd, g_surf, &nf, NULL);
    VkSurfaceFormatKHR fm[64]; if (nf > 64) nf = 64; vkGetPhysicalDeviceSurfaceFormatsKHR(V.pd, g_surf, &nf, fm);
    VkSurfaceFormatKHR pick = fm[0];
    VkFormat want[2] = {srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM, srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM};
    int found = 0;
    for (int w = 0; w < 2 && !found; w++) for (uint32_t i = 0; i < nf; i++) if (fm[i].format == want[w]) { pick = fm[i]; found = 1; break; }
    uint32_t np = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(V.pd, g_surf, &np, NULL);
    VkPresentModeKHR pm[16]; if (np > 16) np = 16; vkGetPhysicalDeviceSurfacePresentModesKHR(V.pd, g_surf, &np, pm);
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    const char *e = getenv("TOTK_PRESENT_MODE");
    for (uint32_t i = 0; i < np; i++) if (pm[i] == VK_PRESENT_MODE_MAILBOX_KHR && !(e && !strcmp(e, "fifo"))) mode = pm[i];
    uint32_t w, h; front_drawable_size(&w, &h);
    VkExtent2D ext = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : (VkExtent2D){w, h};
    if (ext.width < caps.minImageExtent.width) ext.width = caps.minImageExtent.width;
    if (ext.height < caps.minImageExtent.height) ext.height = caps.minImageExtent.height;
    uint32_t count = caps.minImageCount + 1; if (count < 3) count = 3; if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount; if (count > 8) count = 8;
    int pt = 0, pid2 = 0;   /* the surface reports when frames reach the screen (VK_EXT_present_timing, vk_device.c) */
    if (V.present_timing) {
        static PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR caps2;
        if (!caps2) caps2 = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(V.inst, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
        VkPhysicalDeviceSurfaceInfo2KHR si = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, .surface = g_surf};
        VkPresentTimingSurfaceCapabilitiesEXT ptc = {VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
        VkSurfaceCapabilitiesPresentId2KHR pic = {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR, &ptc};
        VkSurfaceCapabilities2KHR c2 = {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR, &pic};
        if (caps2 && caps2(V.pd, &si, &c2) == VK_SUCCESS) { pt = ptc.presentTimingSupported && (ptc.presentStageQueries & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT); pid2 = pic.presentId2Supported; }
    }
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_surf, .minImageCount = count, .imageFormat = pick.format, .imageColorSpace = pick.colorSpace,
        .flags = (pt ? VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT : 0) | (pt && pid2 ? VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR : 0),
        .imageExtent = ext, .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = caps.currentTransform, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = mode, .clipped = VK_TRUE, .oldSwapchain = g_sc};
    VkSwapchainKHR sc;
    VK(vkCreateSwapchainKHR(V.dev, &ci, NULL, &sc));
    if (g_sc) { g_sc_old = g_sc; g_sc_old_wait = 8; }
    g_sc = sc; g_scfmt = pick.format; g_ext = ext; g_sc_srgb = srgb;
    pt_swapchain(sc, pt, pid2);
    if (g_sc_old) { memcpy(g_scview_old, g_scview, sizeof g_scview); g_nsc_old = g_nsc; }   /* the old images' views go with the old swapchain (retire_old) */
    else for (uint32_t i = 0; i < g_nsc; i++) if (g_scview[i]) vkDestroyImageView(V.dev, g_scview[i], NULL);
    memset(g_scview, 0, sizeof g_scview);
    g_nsc = 8; VK(vkGetSwapchainImagesKHR(V.dev, g_sc, &g_nsc, g_scimg));
    for (uint32_t i = 0; i < g_nsc; i++) {
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_scimg[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = pick.format, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(V.dev, &vi, NULL, &g_scview[i]) != VK_SUCCESS) g_scview[i] = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < 8; i++) if (!g_acq[i]) {
        VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK(vkCreateSemaphore(V.dev, &si, NULL, &g_acq[i])); VK(vkCreateSemaphore(V.dev, &si, NULL, &g_rel[i]));
    }
    static int logged; if (logged++ < 40) VLOG("swapchain %ux%u, %u images, format %d, %s", ext.width, ext.height, g_nsc, g_scfmt, mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : "fifo");
}
/* A present's command buffer, once its submission completed (completion thread), is handed back to the present thread,
   which frees it: a command pool must not be used by two threads at once, and the present thread records from it. */
static VkCommandBuffer g_cb_done[64]; static int g_ncb_done; static pthread_mutex_t g_cb_mu = PTHREAD_MUTEX_INITIALIZER;
static void cb_free(void *arg) {
    pthread_mutex_lock(&g_cb_mu);
    if (g_ncb_done < 64) { g_cb_done[g_ncb_done++] = arg; arg = NULL; }
    pthread_mutex_unlock(&g_cb_mu);
    if (arg) VK_ONCE(cbleak, "present: more than 64 finished command buffers waiting to be freed");
}
static void cb_reap(void) {   /* present thread */
    VkCommandBuffer cbs[64]; int n;
    pthread_mutex_lock(&g_cb_mu); n = g_ncb_done; memcpy(cbs, g_cb_done, sizeof *cbs * (size_t)n); g_ncb_done = 0; pthread_mutex_unlock(&g_cb_mu);
    if (n) vkFreeCommandBuffers(V.dev, g_ppool, (uint32_t)n, cbs);
}
/* the source region x0, y0, x1, y1: the game's crop rectangle (x, y, width, height; nvnWindowSetCrop), else the whole texture */
static void src_rect(const pq_item *it, const vk_image *im, int32_t r[4]) {
    r[0] = 0; r[1] = 0; r[2] = (int32_t)im->w; r[3] = (int32_t)im->h;
    if ((it->crop[2] || it->crop[3]) && it->crop[0] >= 0 && it->crop[1] >= 0 && it->crop[2] > 0 && it->crop[3] > 0 &&
        it->crop[0] + it->crop[2] <= (int)im->w && it->crop[1] + it->crop[3] <= (int)im->h) {
        r[0] = it->crop[0]; r[1] = it->crop[1]; r[2] = it->crop[0] + it->crop[2]; r[3] = it->crop[1] + it->crop[3];
    }
}
/* TOTK_VK_CAPTURE=<dir>: every TOTK_VK_CAPTURE_EVERY seconds (default 5) the presented frame is also written to
   <dir>/frame-<seconds since the first present>.ppm, with a window or without one (TOTK_HEADLESS): the picture, checked offscreen.
   TOTK_VK_CAPTURE_BURST=<s>: also the 24 frames presented one after another from second s, as burst-<n>.ppm (flicker). */
static void capture(const vk_image *im, const int32_t r[4]) {
    static const char *dir; static double every = -1, t0, next, burst_at = -1; static int burst_n;
    if (every < 0) {
        dir = getenv("TOTK_VK_CAPTURE"); const char *e = getenv("TOTK_VK_CAPTURE_EVERY"), *b = getenv("TOTK_VK_CAPTURE_BURST");
        every = e && atof(e) > 0 ? atof(e) : 5; t0 = now_s(); next = t0; if (b) burst_at = atof(b);
    }
    if (!dir) return;
    int burst = burst_at >= 0 && now_s() - t0 >= burst_at && burst_n < 24;
    if (!burst && now_s() < next) return;
    if (!burst) next = now_s() + every;
    uint32_t w = (uint32_t)(r[2] - r[0]), h = (uint32_t)(r[3] - r[1]);
    int srgb = im->format == VK_FORMAT_R8G8B8A8_SRGB || im->format == VK_FORMAT_B8G8R8A8_SRGB;   /* the blit keeps the encoding */
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo iai = {.usage = VMA_MEMORY_USAGE_AUTO};
    VkImage img; VmaAllocation ia;
    if (vmaCreateImage(V.vma, &ci, &iai, &img, &ia, NULL) != VK_SUCCESS) return;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VkBuffer buf; VmaAllocation ba; VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &bai, &buf, &ba, &info) != VK_SUCCESS) { vmaDestroyImage(V.vma, img, ia); return; }
    VkCommandBuffer cb = vk_oneoff_begin();   /* begins with a full barrier: the game's rendering of the frame is complete */
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{r[0], r[1], 0}, {r[2], r[3], 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_NEAREST);
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier2(cb, &di);
    VkBufferImageCopy rg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &rg);
    VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo hd = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(cb, &hd);
    vk_oneoff_wait(vk_oneoff_end_submit(cb));
    vmaInvalidateAllocation(V.vma, ba, 0, VK_WHOLE_SIZE);
    char path[1024];
    if (burst) snprintf(path, sizeof path, "%s/burst-%02d.ppm", dir, burst_n++);
    else snprintf(path, sizeof path, "%s/frame-%04d.ppm", dir, (int)(now_s() - t0));
    FILE *f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%u %u\n255\n", w, h);
        const uint8_t *px = info.pMappedData;
        for (size_t i = 0; i < (size_t)w * h; i++) fwrite(px + i * 4, 1, 3, f);
        fclose(f);
    } else VK_ONCE(capture, "TOTK_VK_CAPTURE: cannot write %s", path);
    vmaDestroyBuffer(V.vma, buf, ba);
    vmaDestroyImage(V.vma, img, ia);
}
/* F9 (frontend_sdl.c): the last RING_N presented frames, scaled to RING_W x RING_H, are written to the log folder (next to
   botw.log) as frames-<date>/<n>-<ms>.ppm: a flicker that comes and goes in a frame or two, caught after the player saw
   it. Every present blits its picture into a ring slot on the GPU (about 0.4 MB, 37 MB in all). TOTK_FRAME_RING=0: off. */
enum { RING_N = 96, RING_W = 480, RING_H = 270 };
static struct { VkImage img[2]; VmaAllocation ia[2]; VkBuffer buf; VmaAllocation ba; uint8_t *cpu; uint64_t value[RING_N], frame[RING_N]; double t[RING_N]; uint64_t n; int on; } g_ring;
static _Atomic int g_ring_save;
void mtl_frame_ring_save(void) { atomic_store(&g_ring_save, 1); }
static int ring_init(void) {
    if (g_ring.on) return g_ring.on > 0;
    const char *e = getenv("TOTK_FRAME_RING"); g_ring.on = -1;
    if (e && !strcmp(e, "0")) return 0;
    VkCommandBuffer cb = vk_oneoff_begin();
    for (int k = 0; k < 2; k++) {   /* UNORM and sRGB: the blit keeps the picture's encoding */
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = k ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {RING_W, RING_H, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo iai = {.usage = VMA_MEMORY_USAGE_AUTO};
        if (vmaCreateImage(V.vma, &ci, &iai, &g_ring.img[k], &g_ring.ia[k], NULL) != VK_SUCCESS) { vk_oneoff_end_submit(cb); return 0; }
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_ring.img[k], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(cb, &di);
    }
    vk_oneoff_wait(vk_oneoff_end_submit(cb));
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)RING_N * RING_W * RING_H * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &bai, &g_ring.buf, &g_ring.ba, &info) != VK_SUCCESS) return 0;
    g_ring.cpu = info.pMappedData; g_ring.on = 1;
    VLOG("frame ring: the last %d frames at %dx%d are kept for F9 (TOTK_FRAME_RING=0: off)", RING_N, RING_W, RING_H);
    return 1;
}
/* the present's command buffer, after its blit: the picture into the next ring slot */
static void ring_record(VkCommandBuffer cb, const vk_image *im, const int32_t src[4], uint64_t frame) {
    if (!ring_init()) return;
    int srgb = im->format == VK_FORMAT_R8G8B8A8_SRGB || im->format == VK_FORMAT_B8G8R8A8_SRGB;
    unsigned slot = (unsigned)(g_ring.n % RING_N);
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{src[0], src[1], 0}, {src[2], src[3], 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {RING_W, RING_H, 1}}};
    vk_barrier_all(cb);
    vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, g_ring.img[srgb], VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    vk_barrier_all(cb);
    VkBufferImageCopy rg = {.bufferOffset = (VkDeviceSize)slot * RING_W * RING_H * 4, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {RING_W, RING_H, 1}};
    vkCmdCopyImageToBuffer(cb, g_ring.img[srgb], VK_IMAGE_LAYOUT_GENERAL, g_ring.buf, 1, &rg);
    VkMemoryBarrier2 hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo hd = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(cb, &hd);
    g_ring.t[slot] = now_s(); g_ring.frame[slot] = frame;
}
static void ring_committed(uint64_t v) { if (g_ring.on > 0) { g_ring.value[g_ring.n % RING_N] = v; g_ring.n++; } }
static void ring_save(void) {
    if (!atomic_exchange(&g_ring_save, 0) || g_ring.on <= 0 || !g_ring.n) return;
    uint64_t n = g_ring.n < RING_N ? g_ring.n : RING_N, first = g_ring.n - n;
    vk_wait_value(g_ring.value[(g_ring.n - 1) % RING_N], 2000000000ull);
    char dir[1100]; const char *d = getenv("TOTK_FRAME_RING_DIR"), *prof = getenv("TOTK_PROFILE_DIR");
    time_t now = time(NULL); struct tm tmv; localtime_r(&now, &tmv); char stamp[32]; strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tmv);
    if (d && *d) snprintf(dir, sizeof dir, "%s/frames-%s", d, stamp);
    else if (prof && *prof) snprintf(dir, sizeof dir, "%s/../logs/frames-%s", prof, stamp);
    else snprintf(dir, sizeof dir, "/tmp/botw-frames-%s", stamp);
    if (mkdir(dir, 0755) && errno != EEXIST) { VLOG("frame ring: cannot make %s: %s", dir, strerror(errno)); return; }
    vmaInvalidateAllocation(V.vma, g_ring.ba, 0, VK_WHOLE_SIZE);
    double t0 = g_ring.t[first % RING_N]; uint8_t *row = malloc(RING_W * 3); int wrote = 0;
    for (uint64_t i = first; i < g_ring.n && row; i++) {
        unsigned slot = (unsigned)(i % RING_N); char path[1300];
        snprintf(path, sizeof path, "%s/%03u-%05.0fms-f%llu.ppm", dir, (unsigned)(i - first), (g_ring.t[slot] - t0) * 1000, (unsigned long long)g_ring.frame[slot]);
        FILE *f = fopen(path, "wb"); if (!f) continue;
        fprintf(f, "P6\n%d %d\n255\n", RING_W, RING_H);
        const uint8_t *px = g_ring.cpu + (size_t)slot * RING_W * RING_H * 4;
        for (int y = 0; y < RING_H; y++) { for (int x = 0; x < RING_W; x++) memcpy(row + 3 * x, px + ((size_t)y * RING_W + x) * 4, 3); fwrite(row, 1, RING_W * 3, f); }
        fclose(f); wrote++;
    }
    free(row);
    {   /* the passes of the last frames (vk_exec.c pass log), named by the game frame each picture shows (-f<n>) */
        char path[1300]; snprintf(path, sizeof path, "%s/passes.txt", dir); FILE *f = fopen(path, "w");
        uint64_t last = g_ring.frame[(g_ring.n - 1) % RING_N];
        void vk_pass_log_dump(FILE *, uint64_t, uint64_t);
        if (f) { vk_pass_log_dump(f, last > 15 ? last - 15 : 0, last); fclose(f); }
    }
    VLOG("frame ring: %d frames (%.1f s) saved to %s", wrote, g_ring.t[(g_ring.n - 1) % RING_N] - t0, dir);
}
/* TOTK_STRETCH=1, or key 4's ratio (runtime_aspect_now): the picture fills the window whatever its shape (the camera then
   sees a view of that shape squeezed into the 16:9 canvas, vk_exec.c uw_ui_squeeze). Otherwise it keeps its shape, centred. */
extern float runtime_aspect_now(void) __attribute__((weak));
static int present_stretch(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("TOTK_STRETCH"); on = e && !strcmp(e, "1"); }
    return on || (runtime_aspect_now && runtime_aspect_now() > 0);
}
static uint64_t g_pt_ticket;   /* the ticket of the frame being presented (present_main) */
static void view_free(void *arg) { vkDestroyImageView(V.dev, (VkImageView)arg, NULL); }   /* the FSR input view, when its present has run */
static void present_frame(const pq_item *it) {
    vk_image *im = it->has ? vk_tex_image(&it->t) : NULL;
    if (!im) return;
    int32_t src[4]; src_rect(it, im, src);
    capture(im, src);
    if (!g_surf) {   /* no window: the frame ring only (offscreen runs, the smoke test) */
        if (ring_init()) { VkCommandBuffer cb = vk_oneoff_begin(); ring_record(cb, im, src, it->frame); ring_committed(vk_oneoff_end_submit(cb)); }
        ring_save();
        return;
    }
    int srgb = im->format == VK_FORMAT_R8G8B8A8_SRGB || im->format == VK_FORMAT_B8G8R8A8_SRGB;
    if (!g_sc || srgb != g_sc_srgb) swapchain_make(srgb);
    uint32_t w, h; front_drawable_size(&w, &h);
    if (w != g_ext.width || h != g_ext.height) swapchain_make(srgb);
    VkSemaphore acq = g_acq[g_acq_i++ % 8];
    uint32_t idx;
    VkResult r = vkAcquireNextImageKHR(V.dev, g_sc, 1000000000ull, acq, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { swapchain_make(srgb); return; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { VK_ONCE(acq, "vkAcquireNextImageKHR: %d", r); return; }
    cb_reap();
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_ppool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb; VK(vkAllocateCommandBuffers(V.dev, &ai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK(vkBeginCommandBuffer(cb, &bi));
    VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    int32_t sx0 = src[0], sy0 = src[1], sx1 = src[2], sy1 = src[3];
    double sa = (double)(sx1 - sx0) / (double)(sy1 - sy0), da = (double)g_ext.width / (double)g_ext.height;
    int32_t dw = (int32_t)g_ext.width, dh = (int32_t)g_ext.height, dx = 0, dy = 0;
    if (present_stretch()) {}   /* the whole window */
    else if (da > sa) { dw = (int32_t)lround(g_ext.height * sa); dx = ((int32_t)g_ext.width - dw) / 2; } else { dh = (int32_t)lround(g_ext.width / sa); dy = ((int32_t)g_ext.height - dh) / 2; }
    int bars = dx > 0 || dy > 0 || dw < (int32_t)g_ext.width || dh < (int32_t)g_ext.height;   /* black (a picture filling the window needs no clear) */
    /* a window bigger than the picture: FSR 1 draws it into the swapchain image (vk_fsr.c); else the blit scales it */
    int fsr = g_scview[idx] && vk_fsr_will(im, src, (uint32_t)dw, (uint32_t)dh);
    VkImageLayout lay = fsr ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkImageMemoryBarrier2 b0 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = lay, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = g_scimg[idx], .subresourceRange = rr};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b0};
    vkCmdPipelineBarrier2(cb, &di);
    vk_barrier_all(cb);   /* the game's rendering of the texture is complete before the copy */
    VkImageView fsr_view = VK_NULL_HANDLE;
    if (fsr && !vk_fsr_upscale(cb, im, src, g_scview[idx], g_scfmt, g_ext.width, g_ext.height, dx, dy, (uint32_t)dw, (uint32_t)dh, bars, &fsr_view)) {
        VkImageMemoryBarrier2 bt = b0; bt.oldLayout = lay; bt.newLayout = lay = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;   /* FSR could not: the blit after all */
        di.pImageMemoryBarriers = &bt; vkCmdPipelineBarrier2(cb, &di); di.pImageMemoryBarriers = &b0;
        fsr = 0;
    }
    if (!fsr) {
        if (bars) { VkClearColorValue black = {{0, 0, 0, 1}}; vkCmdClearColorImage(cb, g_scimg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr); vk_barrier_all(cb); }
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{sx0, sy0, 0}, {sx1, sy1, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{dx, dy, 0}, {dx + dw, dy + dh, 1}}};
        vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, g_scimg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    }
    ring_record(cb, im, src, it->frame);
    VkImageMemoryBarrier2 b1 = b0; b1.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT; b1.dstAccessMask = 0; b1.oldLayout = lay; b1.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    di.pImageMemoryBarriers = &b1;
    vkCmdPipelineBarrier2(cb, &di);
    VK(vkEndCommandBuffer(cb));
    uint64_t v = vk_submit(&cb, 1, acq, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_rel[idx]);
    vk_on_done(v, cb_free, cb);
    if (fsr_view) vk_on_done(v, view_free, (void *)fsr_view);
    ring_committed(v);
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &g_rel[idx], .swapchainCount = 1, .pSwapchains = &g_sc, .pImageIndices = &idx};
    uint64_t pid = g_pt_ticket;   /* the frame's ticket is its present id (increasing) */
    VkPresentId2KHR pi2 = {VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, NULL, 1, &pid};
    VkPresentTimingInfoEXT ti = {VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT, .timeDomainId = g_pt.domain, .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT};
    VkPresentTimingsInfoEXT tsi = {VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, g_pt.ids ? &pi2 : NULL, 1, &ti};
    if (g_pt.on) pi.pNext = &tsi;
    pthread_mutex_lock(&V.qmu);
    r = vkQueuePresentKHR(V.q, &pi);
    pthread_mutex_unlock(&V.qmu);
    if (r == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT) { VK_ONCE(ptfull, "presentation times: the queue of answers is full; times off"); g_pt.on = 0; }
    else if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) swapchain_make(srgb);
    else if (r != VK_SUCCESS) VK_ONCE(pres, "vkQueuePresentKHR: %d", r);
    pt_poll();
    retire_old(0);
    ring_save();
    front_frame_presented();
}
static void *present_main(void *arg) {
    (void)arg;
    pthread_setname_np("nvn-present");
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_read == g_pq_in) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        pq_item it = g_pq[g_pq_read % PQ]; uint64_t ticket = ++g_pq_read;
        pthread_mutex_unlock(&g_pq_mu);
        g_pt_ticket = ticket;
        present_frame(&it);
        pthread_mutex_lock(&g_pq_mu);
        g_present_n++;
        if (!g_pt.ids || g_pres_n[ticket % PRING] != ticket) { g_pres_n[ticket % PRING] = ticket; g_pres_t[ticket % PRING] = g_pt.ids ? 0 : now_s(); }   /* with present ids: the screen's time, when it comes (pt_note) */
        if (ticket > g_pq_done) g_pq_done = ticket;
        pthread_cond_broadcast(&g_pq_cv);
        pthread_mutex_unlock(&g_pq_mu);
    }
    return NULL;
}
void vk_present_init(void) {
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = V.qfam};
    VK(vkCreateCommandPool(V.dev, &pci, NULL, &g_ppool));
    if (V.has_window && front_vk_create_surface(V.inst, &g_surf)) {
        VkBool32 ok = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(V.pd, V.qfam, g_surf, &ok);
        if (!ok) { VLOG("the queue cannot present to the window"); g_surf = VK_NULL_HANDLE; }
    }
    g_have_window = g_surf != VK_NULL_HANDLE;
    if (!g_have_window) VLOG("no window: presents are counted only");
    pthread_t th; if (pthread_create(&th, NULL, present_main, NULL)) VDIE("present thread"); pthread_detach(th);
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_in - g_pq_done >= PQ) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
    pq_item *it = &g_pq[g_pq_in % PQ]; memset(it, 0, sizeof *it);
    vk_uw_note_present(t);
    if (t) { it->t = *t; it->has = 1; }
    it->period = period_s; if (crop) memcpy(it->crop, crop, sizeof it->crop);
    uint64_t vk_cur_frame(void); it->frame = vk_cur_frame();
    uint64_t tk = ++g_pq_in;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    return tk;
}
void mtl_present_wait(uint64_t tk) {
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_done < tk) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
    pthread_mutex_unlock(&g_pq_mu);
}
void mtl_present_flush(void) { pthread_mutex_lock(&g_pq_mu); uint64_t h = g_pq_in; pthread_mutex_unlock(&g_pq_mu); mtl_present_wait(h); }
double mtl_presented_time(uint64_t n) {
    pthread_mutex_lock(&g_pq_mu);
    double t = g_pres_n[n % PRING] == n ? g_pres_t[n % PRING] : 0;
    pthread_mutex_unlock(&g_pq_mu);
    return t;
}
int mtl_has_window(void) { return g_have_window; }
int mtl_display_hz(void) { return g_have_window ? front_display_hz() : 0; }
/* Presentation times (VK_EXT_present_timing, vk_device.c; niri reports them through wp_presentation): each present asks
   for the time its first pixel went out, and the present thread collects the answers after each present (pt_poll).
   With present ids (the tickets) they become mtl_presented_time, the on-screen intervals of the frame statistics (-1 for
   a frame the compositor dropped). Every answer with a time also sets the display's vblank clock (g_vb): a vblank, its
   number, and the refresh period the swapchain reports. The frame hold paces the game by it (hle_nvn.c: mtl_link_active,
   mtl_link_wait_tick). */
static void pt_swapchain(VkSwapchainKHR sc, int pt, int pid2) {
    if (!g_pt.qsize && pt) {
        g_pt.qsize = (PFN_vkSetSwapchainPresentTimingQueueSizeEXT)vkGetDeviceProcAddr(V.dev, "vkSetSwapchainPresentTimingQueueSizeEXT");
        g_pt.past = (PFN_vkGetPastPresentationTimingEXT)vkGetDeviceProcAddr(V.dev, "vkGetPastPresentationTimingEXT");
        g_pt.props = (PFN_vkGetSwapchainTimingPropertiesEXT)vkGetDeviceProcAddr(V.dev, "vkGetSwapchainTimingPropertiesEXT");
        g_pt.domains = (PFN_vkGetSwapchainTimeDomainPropertiesEXT)vkGetDeviceProcAddr(V.dev, "vkGetSwapchainTimeDomainPropertiesEXT");
    }
    g_pt.on = pt && g_pt.qsize && g_pt.past && g_pt.props && g_pt.domains && g_pt.qsize(V.dev, sc, 64) == VK_SUCCESS;
    if (g_pt.on) {   /* a present names one of the swapchain's time domains (only for a target time, which is not asked for) */
        VkTimeDomainKHR dom[4]; uint64_t ids[4] = {0};
        VkSwapchainTimeDomainPropertiesEXT dp = {VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT, .timeDomainCount = 4, .pTimeDomains = dom, .pTimeDomainIds = ids};
        VkResult r = g_pt.domains(V.dev, sc, &dp, NULL);
        g_pt.domain = (r == VK_SUCCESS || r == VK_INCOMPLETE) && dp.timeDomainCount ? ids[0] : 0;
    }
    g_pt.ids = g_pt.on && pid2; g_pt.counter = 0;
    static int said = -1; if (said != g_pt.on) { said = g_pt.on; VLOG("presentation times %s", g_pt.on ? (g_pt.ids ? "on (with present ids)" : "on") : V.present_timing ? "not reported for this window" : "off"); }
}
static void vb_note(uint64_t t) {   /* a frame reached the screen at t (CLOCK_MONOTONIC ns) */
    uint64_t now = vk_now_ns();
    pthread_mutex_lock(&g_vb.mu);
    if (t > now + 100000000ull || now - t > 1000000000ull) {   /* not this clock: no vblank clock */
        if (!g_vb.bad++) VLOG("presentation times are not on the monotonic clock (%.3f s from now): no vblank clock", ((double)t - (double)now) / 1e9);
        g_vb.t = 0; pthread_mutex_unlock(&g_vb.mu); return;
    }
    uint64_t p = g_vb.period ? g_vb.period : 1000000000ull / (uint64_t)(front_display_hz() > 0 ? front_display_hz() : 60);
    if (!g_vb.t || t + 4 * p < g_vb.t) { g_vb.t = t; g_vb.n = 1; }
    else if (t > g_vb.t) {   /* the vblanks are a fixed grid: a time near it moves it an eighth of the way, one far off starts it again */
        uint64_t m = (t - g_vb.t + p / 2) / p; int64_t err = (int64_t)(t - (g_vb.t + m * p));
        g_vb.n += m; g_vb.t = (err < 0 ? -err : err) < (int64_t)(p / 4) ? (uint64_t)((int64_t)(g_vb.t + m * p) + err / 8) : t;
    }
    pthread_mutex_unlock(&g_vb.mu);
}
static void pt_note(uint64_t id, uint64_t t) {
    g_pt.answers++; if (!t) g_pt.dropped++;
    if (t) vb_note(t);
    if (!g_pt.ids || !id) return;
    pthread_mutex_lock(&g_pq_mu);
    g_pres_n[id % PRING] = id; g_pres_t[id % PRING] = t ? t / 1e9 : -1;
    pthread_mutex_unlock(&g_pq_mu);
}
static void pt_poll(void) {   /* present thread */
    if (!g_pt.on) return;
    for (int round = 0; round < 8; round++) {
        VkPresentStageTimeEXT st[16][2]; VkPastPresentationTimingEXT pt[16];
        for (int i = 0; i < 16; i++) pt[i] = (VkPastPresentationTimingEXT){VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT, .presentStageCount = 2, .pPresentStages = st[i]};
        VkPastPresentationTimingPropertiesEXT pp = {VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT, .presentationTimingCount = 16, .pPresentationTimings = pt};
        VkPastPresentationTimingInfoEXT pi = {VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT, .swapchain = g_sc};
        VkResult r = g_pt.past(V.dev, &pi, &pp);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) { VK_ONCE(ptpast, "vkGetPastPresentationTimingEXT: %d; presentation times off", r); g_pt.on = 0; return; }
        for (uint32_t i = 0; i < pp.presentationTimingCount && i < 16; i++) {
            uint64_t t = 0;
            for (uint32_t k = 0; k < pt[i].presentStageCount && k < 2; k++) if (pt[i].pPresentStages[k].stage == VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) t = pt[i].pPresentStages[k].time;
            pt_note(pt[i].presentId, t);
        }
        if (r != VK_INCOMPLETE) break;
    }
    VkSwapchainTimingPropertiesEXT tp = {VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT}; uint64_t c = 0;
    if (g_pt.props(V.dev, g_sc, &tp, &c) == VK_SUCCESS && c && c != g_pt.counter && tp.refreshDuration) {
        g_pt.counter = c;
        pthread_mutex_lock(&g_vb.mu); int was = g_vb.period != 0; g_vb.period = tp.refreshDuration; pthread_mutex_unlock(&g_vb.mu);
        if (!was || c < 4) VLOG("display refresh %.3f ms (%.2f Hz), from the compositor", tp.refreshDuration / 1e6, 1e9 / tp.refreshDuration);
    }
}
/* The frame hold (hle_nvn.c, weak here for the smoke test) paces the game by the vblank clock while frames reach the screen */
__attribute__((weak)) int nvn_frame_hold_now(void) { return 0; }
uint64_t vk_vblank_period(void) { pthread_mutex_lock(&g_vb.mu); uint64_t p = g_vb.t ? g_vb.period : 0; pthread_mutex_unlock(&g_vb.mu); return p; }
int mtl_link_active(void) {
    if (!nvn_frame_hold_now() || !g_pt.on) return 0;
    pthread_mutex_lock(&g_vb.mu); uint64_t t = g_vb.t, p = g_vb.period; pthread_mutex_unlock(&g_vb.mu);
    return t && p && vk_now_ns() - t < 250000000ull;   /* a frame reached the screen in the last quarter second */
}
/* Wait for the vblank frame_ticks after the one the previous frame started at; a frame already past it starts at once
   (the Metal backend's display-link semantics). 1: started on the clock, 0: no clock or the wait would pass timeout_ns. */
int mtl_link_wait_tick(uint64_t timeout_ns, unsigned frame_ticks) {
    static uint64_t started;   /* the vblank number the previous frame started at (the game's render thread only) */
    pthread_mutex_lock(&g_vb.mu); uint64_t t0 = g_vb.t, n0 = g_vb.n, p = g_vb.period; pthread_mutex_unlock(&g_vb.mu);
    if (!t0 || !p) return 0;
    uint64_t now = vk_now_ns(), cur = n0 + (now > t0 ? (now - t0) / p : 0), target = started + (frame_ticks ? frame_ticks : 1);
    if (!started || cur >= target || target > cur + 8) { started = cur; return 1; }
    uint64_t at = t0 + (target - n0) * p;
    if (at > now + timeout_ns) { started = cur; return 0; }
    struct timespec ts = {(time_t)(at / 1000000000ull), (long)(at % 1000000000ull)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    started = target;
    return 1;
}
/* answers, of them dropped by the compositor (the frame statistics) */
void vk_present_timing_stats(uint64_t *answers, uint64_t *dropped) { *answers = g_pt.answers; *dropped = g_pt.dropped; }
uint32_t mtl_present_qos(void) { return 0; }
uint64_t mtl_present_count(void) { return g_present_n; }
void mtl_present_discard(int on) { (void)on; }
void mtl_present_test_stall(uint64_t ns) { (void)ns; }
