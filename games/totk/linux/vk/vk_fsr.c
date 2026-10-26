/* FSR 1 at the present (vk_present.c; TOTK_UPSCALE=bilinear: the blit as before). AMD FidelityFX Super Resolution 1.0
   (MIT, builtin/ffx/license.txt): EASU scales the game's picture to the window's size with edge-adaptive filtering, a
   full-screen draw into an image of that size (RGBA8; RGBA16F for an sRGB picture, whose samples come decoded), and RCAS
   sharpens that straight into the swapchain image, a draw over the picture's rectangle. Used when the window is larger
   than the picture both ways. TOTK_FSR_SHARPNESS=<stops> (0.25, AMD's default; 0 the sharpest, 2 soft). Measured at
   1920x1080 -> 2560x1608 on an M2: AMD's compute passes with RCAS into a second RGBA16F image and a copy into the window
   took 3.5 ms of GPU time a frame; with RCAS drawn into the window 2.6; as draws (vk_fsr_test's timing in the notes). */
#include "vk_priv.h"
#include "vk_builtin_spv.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define A_CPU 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"   /* AMD's headers in C */
#pragma clang diagnostic ignored "-Wunused-parameter"
#include "builtin/ffx/ffx_a.h"
#include "builtin/ffx/ffx_fsr1.h"
#pragma clang diagnostic pop

enum { UP_BILINEAR, UP_SGSR, UP_FSR };
static struct {
    int state;   /* 0 not set up, 1 ready, -1 not available */
    int mode;    /* UP_SGSR or UP_FSR (TOTK_UPSCALE) */
    VkPipeline sgsr; VkFormat sgsr_fmt;
    VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkPipeline easu[2], rcas; VkFormat rcas_fmt; VkSampler samp;   /* easu: into RGBA8, RGBA16F */
    PFN_vkCmdPushDescriptorSet push;
    VkImage mid; VmaAllocation mem; VkImageView mid_view; VkFormat mid_fmt; uint32_t w, h;   /* EASU's output */
    float sharp; int half;   /* half: the FP16 shaders */
} g_fsr;

static VkShaderModule fsr_module(const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m; return vkCreateShaderModule(V.dev, &mi, NULL, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
}
static int fsr_setup(void) {
    if (g_fsr.state) return g_fsr.state > 0;
    g_fsr.state = -1;
    const char *e = getenv("TOTK_UPSCALE");
    g_fsr.mode = !e || !*e || !strcmp(e, "sgsr") ? UP_SGSR : !strcmp(e, "fsr") ? UP_FSR : UP_BILINEAR;
    if (g_fsr.mode == UP_BILINEAR) { VLOG("upscale: bilinear (TOTK_UPSCALE=%s)", e); return 0; }
    g_fsr.push = (PFN_vkCmdPushDescriptorSet)vkGetDeviceProcAddr(V.dev, "vkCmdPushDescriptorSet");
    VkFormatProperties f8, f16; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_R8G8B8A8_UNORM, &f8); vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_R16G16B16A16_SFLOAT, &f16);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (!g_fsr.push || (f8.optimalTilingFeatures & need) != need || (f16.optimalTilingFeatures & need) != need) {
        VLOG("upscale: FSR not available on this device (push descriptors, render targets); bilinear"); return 0;
    }
    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT, .bindingCount = 1, .pBindings = &b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64};
    if (vkCreateDescriptorSetLayout(V.dev, &dci, NULL, &g_fsr.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_fsr.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(V.dev, &lci, NULL, &g_fsr.layout) != VK_SUCCESS) return 0;
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    if (vkCreateSampler(V.dev, &sci, NULL, &g_fsr.samp) != VK_SUCCESS) return 0;
    {   /* the FP16 path where the device has 16-bit floats and ints (enabled with every other feature, vk_device.c); TOTK_FSR_HALF=0: FP32 */
        VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
        vkGetPhysicalDeviceFeatures2(V.pd, &f2);
        const char *h = getenv("TOTK_FSR_HALF");
        g_fsr.half = f12.shaderFloat16 && f2.features.shaderInt16 && !(h && !strcmp(h, "0"));
    }
    const char *s = getenv("TOTK_FSR_SHARPNESS"); g_fsr.sharp = s && *s ? strtof(s, NULL) : 0.25f;
    if (!(g_fsr.sharp >= 0 && g_fsr.sharp <= 4)) g_fsr.sharp = 0.25f;
    g_fsr.state = 1;
    if (g_fsr.mode == UP_FSR) VLOG("upscale: FSR 1 (EASU + RCAS, %s, sharpness %.2f stops) when the window is bigger than the picture (TOTK_UPSCALE=sgsr / bilinear: the others)", g_fsr.half ? "FP16" : "FP32", g_fsr.sharp);
    else VLOG("upscale: Snapdragon GSR 1 (edge direction on) when the window is bigger than the picture (TOTK_UPSCALE=fsr / bilinear: the others)");
    return 1;
}
/* a full-screen draw of the fragment shader fs into one colour attachment of format fmt */
static VkPipeline fsr_pipe(const uint32_t *fs_code, size_t fs_size, VkFormat fmt) {
    VkShaderModule vs = fsr_module(spv_fsr_vs, sizeof spv_fsr_vs), fs = fsr_module(fs_code, fs_size);
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &ba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
    VkGraphicsPipelineCreateInfo pi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds, .layout = g_fsr.layout};
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vs && fs ? vkCreateGraphicsPipelines(V.dev, VK_NULL_HANDLE, 1, &pi, NULL, &p) : VK_ERROR_INITIALIZATION_FAILED;
    if (vs) vkDestroyShaderModule(V.dev, vs, NULL);
    if (fs) vkDestroyShaderModule(V.dev, fs, NULL);
    if (r != VK_SUCCESS) { VK_ONCE(fsrpipe, "upscale: an FSR pipeline was refused (%d); bilinear", r); return VK_NULL_HANDLE; }
    return p;
}
/* RCAS's pipeline for the target's format (made again when it changes) and EASU's for the intermediate's */
static int fsr_rcas_for(VkFormat fmt) {
    if (g_fsr.rcas && g_fsr.rcas_fmt == fmt) return 1;
    if (g_fsr.rcas) { vk_wait_value(vk_last_value(), 2000000000ull); vkDestroyPipeline(V.dev, g_fsr.rcas, NULL); g_fsr.rcas = VK_NULL_HANDLE; }
    g_fsr.rcas = g_fsr.half ? fsr_pipe(spv_fsr_rcas_h, sizeof spv_fsr_rcas_h, fmt) : fsr_pipe(spv_fsr_rcas, sizeof spv_fsr_rcas, fmt);
    g_fsr.rcas_fmt = fmt;
    return g_fsr.rcas != VK_NULL_HANDLE;
}
static int sgsr_for(VkFormat fmt) {
    if (g_fsr.sgsr && g_fsr.sgsr_fmt == fmt) return 1;
    if (g_fsr.sgsr) { vk_wait_value(vk_last_value(), 2000000000ull); vkDestroyPipeline(V.dev, g_fsr.sgsr, NULL); g_fsr.sgsr = VK_NULL_HANDLE; }
    g_fsr.sgsr = fsr_pipe(spv_sgsr, sizeof spv_sgsr, fmt); g_fsr.sgsr_fmt = fmt;
    return g_fsr.sgsr != VK_NULL_HANDLE;
}
static VkPipeline fsr_easu_for(VkFormat fmt) {
    int k = fmt == VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!g_fsr.easu[k]) g_fsr.easu[k] = g_fsr.half ? fsr_pipe(spv_fsr_easu_h, sizeof spv_fsr_easu_h, fmt) : fsr_pipe(spv_fsr_easu, sizeof spv_fsr_easu, fmt);
    return g_fsr.easu[k];
}
static void fsr_mid_free(void) {
    if (g_fsr.mid_view) vkDestroyImageView(V.dev, g_fsr.mid_view, NULL);
    if (g_fsr.mid) vmaDestroyImage(V.vma, g_fsr.mid, g_fsr.mem);
    g_fsr.mid_view = VK_NULL_HANDLE; g_fsr.mid = VK_NULL_HANDLE; g_fsr.w = g_fsr.h = 0;
}
static int fsr_mid(uint32_t w, uint32_t h, VkFormat fmt) {
    if (g_fsr.w == w && g_fsr.h == h && g_fsr.mid_fmt == fmt && g_fsr.mid) return 1;
    if (g_fsr.mid) vk_wait_value(vk_last_value(), 2000000000ull);   /* a new window size: the old image is no longer read */
    fsr_mid_free();
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO};
    if (vmaCreateImage(V.vma, &ci, &ai, &g_fsr.mid, &g_fsr.mem, NULL) != VK_SUCCESS) { g_fsr.mid = VK_NULL_HANDLE; return 0; }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_fsr.mid, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(V.dev, &vi, NULL, &g_fsr.mid_view) != VK_SUCCESS) { g_fsr.mid_view = VK_NULL_HANDLE; fsr_mid_free(); return 0; }
    g_fsr.w = w; g_fsr.h = h; g_fsr.mid_fmt = fmt;
    VLOG("upscale: FSR to %ux%u (%s between its passes)", w, h, fmt == VK_FORMAT_R8G8B8A8_UNORM ? "RGBA8" : "RGBA16F");
    return 1;
}
static int srgb_format(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32; }
/* whether FSR takes this upscale: a window bigger than the picture both ways, a picture it can sample */
int vk_fsr_will(const vk_image *im, const int32_t src[4], uint32_t w, uint32_t h) {
    int32_t sw = src[2] - src[0], sh = src[3] - src[1];
    if (sw <= 0 || sh <= 0 || w <= (uint32_t)sw || h <= (uint32_t)sh || im->type != VK_IMAGE_TYPE_2D || im->samples > 1 || im->depth || im->stencil ||
        !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return 0;
    return fsr_setup();
}
/* EASU of the region src (x0, y0, x1, y1) of im into the intermediate at w x h, then RCAS drawn into dst (a colour
   attachment view of format dst_fmt in COLOR_ATTACHMENT_OPTIMAL, the target tw x th) over (dx, dy, w, h); clear: the
   rest of the target black. Recorded into cb after the barrier that orders the game's rendering of im. 0 when it
   could not (nothing recorded); *view: the view of im it made, to destroy when cb has run. */
int vk_fsr_upscale(VkCommandBuffer cb, const vk_image *im, const int32_t src[4], VkImageView dst, VkFormat dst_fmt, uint32_t tw, uint32_t th,
                   int32_t dx, int32_t dy, uint32_t w, uint32_t h, int clear, VkImageView *view) {
    *view = VK_NULL_HANDLE;
    if (!vk_fsr_will(im, src, w, h)) return 0;
    if (g_fsr.mode == UP_SGSR) {   /* one draw straight into the window */
        if (!sgsr_for(dst_fmt)) return 0;
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = im->format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(V.dev, &vi, NULL, view) != VK_SUCCESS) { *view = VK_NULL_HANDLE; return 0; }
        VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = dst, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.color = {{0, 0, 0, 1}}}};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {tw, th}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_fsr.sgsr);
        VkDescriptorImageInfo ii = {g_fsr.samp, *view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
        g_fsr.push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_fsr.layout, 0, 1, &wr);
        /* ViewportInfo: 1/w, 1/h, w, h of the texture; Map: the window position (dx, dy .. + w, h) to the texture coordinate of src */
        float tw_ = (float)im->w, th_ = (float)im->h, sxs = (float)(src[2] - src[0]) / (float)w, sys = (float)(src[3] - src[1]) / (float)h;
        float pcv[8] = {1 / tw_, 1 / th_, tw_, th_, sxs / tw_, sys / th_, ((float)src[0] - (float)dx * sxs) / tw_, ((float)src[1] - (float)dy * sys) / th_};
        vkCmdPushConstants(cb, g_fsr.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pcv, pcv);
        VkViewport vp = {(float)dx, (float)dy, (float)w, (float)h, 0, 1}; VkRect2D sc = {{dx, dy}, {w, h}};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
        return 1;
    }
    VkFormat mf = srgb_format(im->format) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    VkPipeline easu = fsr_easu_for(mf);
    if (!easu || !fsr_rcas_for(dst_fmt) || !fsr_mid(w, h, mf)) return 0;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = im->format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(V.dev, &vi, NULL, view) != VK_SUCCESS) { *view = VK_NULL_HANDLE; return 0; }
    int32_t sw = src[2] - src[0], sh = src[3] - src[1];
    AU1 con[16] = {0};
    FsrEasuConOffset(con, con + 4, con + 8, con + 12, (AF1)sw, (AF1)sh, (AF1)im->w, (AF1)im->h, (AF1)w, (AF1)h, (AF1)src[0], (AF1)src[1]);
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_fsr.mid, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    VkRenderingAttachmentInfo ma = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_fsr.mid_view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo mri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ma};
    vkCmdBeginRendering(cb, &mri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, easu);
    VkDescriptorImageInfo ii = {g_fsr.samp, *view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    g_fsr.push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_fsr.layout, 0, 1, &wr);
    vkCmdPushConstants(cb, g_fsr.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64, con);
    VkViewport mvp = {0, 0, (float)w, (float)h, 0, 1}; VkRect2D msc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &mvp); vkCmdSetScissor(cb, 0, 1, &msc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    b.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(cb, &di);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = dst, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.color = {{0, 0, 0, 1}}}};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {tw, th}}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_fsr.rcas);
    VkDescriptorImageInfo mi = {g_fsr.samp, g_fsr.mid_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet mw = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &mi};
    g_fsr.push(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_fsr.layout, 0, 1, &mw);
    AU1 rc[8] = {0}; FsrRcasCon(rc, g_fsr.sharp); rc[4] = (AU1)dx; rc[5] = (AU1)dy;
    vkCmdPushConstants(cb, g_fsr.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof rc, rc);
    VkViewport vp = {(float)dx, (float)dy, (float)w, (float)h, 0, 1}; VkRect2D sc = {{dx, dy}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    return 1;
}
/* tests: t's whole picture to w x h through mode (0 the bilinear blit, 1 SGSR, 2 FSR), read back as RGBA8 (out: w * h * 4
   bytes); 0 when it did not run */
int vk_fsr_test(const tex_t *t, uint32_t w, uint32_t h, int mode, void *out) {
    int bilinear = mode == UP_BILINEAR;
    if (!bilinear && !fsr_setup()) return 0;
    int was = g_fsr.mode; if (!bilinear) g_fsr.mode = mode;
    vk_image *im = vk_tex_image(t); if (!im) return 0;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO};
    VkImage dimg; VmaAllocation dm; if (vmaCreateImage(V.vma, &ci, &ai, &dimg, &dm, NULL) != VK_SUCCESS) return 0;
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = dimg, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView dview; vkCreateImageView(V.dev, &vci, NULL, &dview);
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)w * h * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT};
    VkBuffer buf; VmaAllocation ba; VmaAllocationInfo info;
    if (vmaCreateBuffer(V.vma, &bi, &bai, &buf, &ba, &info) != VK_SUCCESS) { vkDestroyImageView(V.dev, dview, NULL); vmaDestroyImage(V.vma, dimg, dm); return 0; }
    VkCommandBuffer cb = vk_oneoff_begin();
    vk_barrier_all(cb);
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = bilinear ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = dimg, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
    int32_t src[4] = {0, 0, (int32_t)im->w, (int32_t)im->h};
    VkImageView view = VK_NULL_HANDLE; int ran = 1;
    if (bilinear) {
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {src[2], src[3], 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
        vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, dimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    } else ran = vk_fsr_upscale(cb, im, src, dview, VK_FORMAT_R8G8B8A8_UNORM, w, h, 0, 0, w, h, 0, &view);
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT; b.oldLayout = b.newLayout; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier2(cb, &di);
    VkBufferImageCopy rg = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, dimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &rg);
    vk_oneoff_wait(vk_oneoff_end_submit(cb));
    if (view) vkDestroyImageView(V.dev, view, NULL);
    if (ran) memcpy(out, info.pMappedData, (size_t)w * h * 4);
    vmaDestroyBuffer(V.vma, buf, ba); vkDestroyImageView(V.dev, dview, NULL); vmaDestroyImage(V.vma, dimg, dm);
    g_fsr.mode = was;
    return ran;
}
