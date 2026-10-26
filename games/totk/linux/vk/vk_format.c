/* NVN enums to Vulkan. The texture format table follows third_party/d2r_gfx/gfx/tex/nvn_fmt_mtl.c entry by entry (the
   same substitutes and the same "no counterpart" entries, so the game sees what it sees on the Mac): depth 24 becomes
   32-bit float, "X" formats read alpha as 1, the 16-bit packed formats take the Vulkan format whose bit layout matches
   the Metal format chosen there. The vertex format table follows vfmt() in nvn_mtl.m; Vulkan's SCALED formats stand in
   for Metal's numeric conversion of integer formats read as float (*_UI2F / *_I2F). */
#include "vk_priv.h"
#include <string.h>

#define I VK_COMPONENT_SWIZZLE_IDENTITY
typedef struct { uint32_t vk; uint8_t noalpha, rb; } ent;
#define E(f) {VK_FORMAT_##f, 0, 0}
#define X(f) {VK_FORMAT_##f, 1, 0}
#define R(f) {VK_FORMAT_##f, 0, 1}
#define XR(f) {VK_FORMAT_##f, 1, 1}
#define N {0, 0, 0}
static const ent T[0x9D] = {
    [0x01] = E(R8_UNORM), E(R8_SNORM), E(R8_UINT), E(R8_SINT),
    [0x05] = E(R16_SFLOAT), E(R16_UNORM), E(R16_SNORM), E(R16_UINT), E(R16_SINT),
    [0x0A] = E(R32_SFLOAT), E(R32_UINT), E(R32_SINT),
    [0x0D] = E(R8G8_UNORM), E(R8G8_SNORM), E(R8G8_UINT), E(R8G8_SINT),
    [0x11] = E(R16G16_SFLOAT), E(R16G16_UNORM), E(R16G16_SNORM), E(R16G16_UINT), E(R16G16_SINT),
    [0x16] = E(R32G32_SFLOAT), E(R32G32_UINT), E(R32G32_SINT),
    [0x19] = N, N, N, N, N, N, N, N, N, N, N, N,   /* RGB8 / RGB16 / RGB32: none on Metal either */
    [0x25] = E(R8G8B8A8_UNORM), E(R8G8B8A8_SNORM), E(R8G8B8A8_UINT), E(R8G8B8A8_SINT),
    [0x29] = E(R16G16B16A16_SFLOAT), E(R16G16B16A16_UNORM), E(R16G16B16A16_SNORM), E(R16G16B16A16_UINT), E(R16G16B16A16_SINT),
    [0x2E] = E(R32G32B32A32_SFLOAT), E(R32G32B32A32_UINT), E(R32G32B32A32_SINT),
    [0x31] = E(S8_UINT), E(D16_UNORM),
    [0x33] = E(D32_SFLOAT), E(D32_SFLOAT),
    [0x35] = E(D32_SFLOAT_S8_UINT), E(D32_SFLOAT_S8_UINT),
    [0x37] = X(R8G8B8A8_SRGB), E(R8G8B8A8_SRGB),
    /* 16-bit packed: Metal ABGR4Unorm / BGR5A1Unorm (r,b swapped) / B5G6R5Unorm (r,b swapped) = these bit layouts */
    [0x39] = E(R4G4B4A4_UNORM_PACK16), X(A1B5G5R5_UNORM_PACK16_KHR), E(A1B5G5R5_UNORM_PACK16_KHR), E(B5G6R5_UNORM_PACK16),
    [0x3D] = E(A2B10G10R10_UNORM_PACK32), E(A2B10G10R10_UINT_PACK32), E(B10G11R11_UFLOAT_PACK32), E(E5B9G9R9_UFLOAT_PACK32),
    [0x41] = X(BC1_RGBA_UNORM_BLOCK), E(BC1_RGBA_UNORM_BLOCK), E(BC2_UNORM_BLOCK), E(BC3_UNORM_BLOCK),
    [0x45] = X(BC1_RGBA_SRGB_BLOCK), E(BC1_RGBA_SRGB_BLOCK), E(BC2_SRGB_BLOCK), E(BC3_SRGB_BLOCK),
    [0x49] = E(BC4_UNORM_BLOCK), E(BC4_SNORM_BLOCK), E(BC5_UNORM_BLOCK), E(BC5_SNORM_BLOCK), E(BC7_UNORM_BLOCK), E(BC7_SRGB_BLOCK), E(BC6H_SFLOAT_BLOCK), E(BC6H_UFLOAT_BLOCK),
    [0x51] = E(R8_UINT), E(R8_SINT), E(R16_UINT), E(R16_SINT), E(R32_UINT), E(R32_SINT),
    [0x57] = E(R8G8_UINT), E(R8G8_SINT), E(R16G16_UINT), E(R16G16_SINT), E(R32G32_UINT), E(R32G32_SINT),
    [0x5D] = N, N, N, N, N, N,
    [0x63] = E(R8G8B8A8_UINT), E(R8G8B8A8_SINT), E(R16G16B16A16_UINT), E(R16G16B16A16_SINT), E(R32G32B32A32_UINT), E(R32G32B32A32_SINT),
    [0x69] = N, N, E(A2B10G10R10_UINT_PACK32), N,
    [0x6D] = X(R8G8B8A8_UNORM), X(R8G8B8A8_SNORM), X(R8G8B8A8_UINT), X(R8G8B8A8_SINT),
    [0x71] = X(R16G16B16A16_SFLOAT), X(R16G16B16A16_UNORM), X(R16G16B16A16_SNORM), X(R16G16B16A16_UINT), X(R16G16B16A16_SINT),
    [0x76] = X(R32G32B32A32_SFLOAT), X(R32G32B32A32_UINT), X(R32G32B32A32_SINT),
    /* 0x98 (alpha in bit 0) has no Honeykrisp format: uploads rotate each texel into A1R5G5B5 (vk_tex.c) */
    [0x95] = E(R5G6B5_UNORM_PACK16), X(A1R5G5B5_UNORM_PACK16), E(A1R5G5B5_UNORM_PACK16), E(A1R5G5B5_UNORM_PACK16),
    [0x99] = X(B8G8R8A8_UNORM), E(B8G8R8A8_UNORM), X(B8G8R8A8_SRGB), E(B8G8R8A8_SRGB),
};

int vk_fmt_of(int f, vk_fmt *o) {
    memset(o, 0, sizeof *o);
    ent e = {0, 0, 0};
    if (f >= 0x79 && f <= 0x94) { int k = (f - 0x79) % 14; e.vk = (uint32_t)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * k + (f >= 0x87)); }
    else if (f > 0 && f < (int)(sizeof T / sizeof *T)) e = T[f];
    if (!e.vk) return 0;
    o->vk = (VkFormat)e.vk;
    o->swz[0] = e.rb ? VK_COMPONENT_SWIZZLE_B : VK_COMPONENT_SWIZZLE_R; o->swz[1] = VK_COMPONENT_SWIZZLE_G;
    o->swz[2] = e.rb ? VK_COMPONENT_SWIZZLE_R : VK_COMPONENT_SWIZZLE_B; o->swz[3] = e.noalpha ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_A;
    VkFormat v = o->vk;
    o->depth = v == VK_FORMAT_D16_UNORM || v == VK_FORMAT_D32_SFLOAT || v == VK_FORMAT_D32_SFLOAT_S8_UINT;
    o->stencil = v == VK_FORMAT_S8_UINT || v == VK_FORMAT_D32_SFLOAT_S8_UINT;
    o->compressed = (f >= 0x41 && f <= 0x50) || (f >= 0x79 && f <= 0x94);
    o->srgb = f == 0x37 || f == 0x38 || (f >= 0x45 && f <= 0x48) || f == 0x4E || (f >= 0x87 && f <= 0x94) || f == 0x9B || f == 0x9C;
    int c = vk_fmt_class(f); o->is_uint = c == 1; o->is_int = c == 2;
    if (!o->is_uint && !o->is_int) {   /* the substitutes of *_UI2F / *_I2F and the 10:10:10:2 integer are integer storage */
        if ((f >= 0x51 && f <= 0x68) || f == 0x3E || f == 0x6B) { if (f == 0x3E || f == 0x6B || (f - 0x51) % 2 == 0) o->is_uint = 1; else o->is_int = 1; }
        if (f >= 0x6D && f <= 0x78) { if (f == 0x6F || f == 0x74 || f == 0x77) o->is_uint = 1; if (f == 0x70 || f == 0x75 || f == 0x78) o->is_int = 1; }
    }
    int bw, bh; o->bpb = (uint8_t)nvn_fmt_bpb(f, &bw, &bh); o->bw = (uint8_t)bw; o->bh = (uint8_t)bh;
    return 1;
}
VkFormat vk_srgb_pair(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK; case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case VK_FORMAT_BC2_UNORM_BLOCK: return VK_FORMAT_BC2_SRGB_BLOCK; case VK_FORMAT_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
        case VK_FORMAT_BC3_UNORM_BLOCK: return VK_FORMAT_BC3_SRGB_BLOCK; case VK_FORMAT_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
        case VK_FORMAT_BC7_UNORM_BLOCK: return VK_FORMAT_BC7_SRGB_BLOCK; case VK_FORMAT_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
        default:
            if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) return (VkFormat)(((f - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) & 1) ? f - 1 : f + 1);
            return VK_FORMAT_UNDEFINED;
    }
}
int vk_fmt_class(int f) {
    switch (f) {
        case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x1B: case 0x20: case 0x23: case 0x27: case 0x2C: case 0x2F: return 1;
        case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x1C: case 0x21: case 0x24: case 0x28: case 0x2D: case 0x30: return 2;
        default: return 0;
    }
}
VkFormat vk_vertex_format(int f) {
    switch (f) {
        case 0x01: return VK_FORMAT_R8_UNORM; case 0x02: return VK_FORMAT_R8_SNORM; case 0x03: return VK_FORMAT_R8_UINT; case 0x04: return VK_FORMAT_R8_SINT;
        case 0x05: return VK_FORMAT_R16_SFLOAT; case 0x06: return VK_FORMAT_R16_UNORM; case 0x07: return VK_FORMAT_R16_SNORM; case 0x08: return VK_FORMAT_R16_UINT;
        case 0x09: return VK_FORMAT_R16_SINT; case 0x0A: return VK_FORMAT_R32_SFLOAT; case 0x0B: return VK_FORMAT_R32_UINT; case 0x0C: return VK_FORMAT_R32_SINT;
        case 0x0D: return VK_FORMAT_R8G8_UNORM; case 0x0E: return VK_FORMAT_R8G8_SNORM; case 0x0F: return VK_FORMAT_R8G8_UINT; case 0x10: return VK_FORMAT_R8G8_SINT;
        case 0x11: return VK_FORMAT_R16G16_SFLOAT; case 0x12: return VK_FORMAT_R16G16_UNORM; case 0x13: return VK_FORMAT_R16G16_SNORM; case 0x14: return VK_FORMAT_R16G16_UINT;
        case 0x15: return VK_FORMAT_R16G16_SINT; case 0x16: return VK_FORMAT_R32G32_SFLOAT; case 0x17: return VK_FORMAT_R32G32_UINT; case 0x18: return VK_FORMAT_R32G32_SINT;
        case 0x19: return VK_FORMAT_R8G8B8_UNORM; case 0x1A: return VK_FORMAT_R8G8B8_SNORM; case 0x1B: return VK_FORMAT_R8G8B8_UINT; case 0x1C: return VK_FORMAT_R8G8B8_SINT;
        case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT; case 0x1E: return VK_FORMAT_R16G16B16_UNORM; case 0x1F: return VK_FORMAT_R16G16B16_SNORM; case 0x20: return VK_FORMAT_R16G16B16_UINT;
        case 0x21: return VK_FORMAT_R16G16B16_SINT; case 0x22: return VK_FORMAT_R32G32B32_SFLOAT; case 0x23: return VK_FORMAT_R32G32B32_UINT; case 0x24: return VK_FORMAT_R32G32B32_SINT;
        case 0x25: return VK_FORMAT_R8G8B8A8_UNORM; case 0x26: return VK_FORMAT_R8G8B8A8_SNORM; case 0x27: return VK_FORMAT_R8G8B8A8_UINT; case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
        case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT; case 0x2A: return VK_FORMAT_R16G16B16A16_UNORM; case 0x2B: return VK_FORMAT_R16G16B16A16_SNORM; case 0x2C: return VK_FORMAT_R16G16B16A16_UINT;
        case 0x2D: return VK_FORMAT_R16G16B16A16_SINT; case 0x2E: return VK_FORMAT_R32G32B32A32_SFLOAT; case 0x2F: return VK_FORMAT_R32G32B32A32_UINT; case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
        case 0x3D: return VK_FORMAT_A2B10G10R10_UNORM_PACK32; case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32; case 0x3F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case 0x9A: return VK_FORMAT_B8G8R8A8_UNORM;
        /* *_UI2F / *_I2F: integers read as float without normalisation = SCALED. 32-bit has no SCALED format (see vk_exec.c). */
        case 0x51: return VK_FORMAT_R8_USCALED; case 0x52: return VK_FORMAT_R8_SSCALED; case 0x53: return VK_FORMAT_R16_USCALED; case 0x54: return VK_FORMAT_R16_SSCALED;
        case 0x55: return VK_FORMAT_R32_UINT; case 0x56: return VK_FORMAT_R32_SINT;
        case 0x57: return VK_FORMAT_R8G8_USCALED; case 0x58: return VK_FORMAT_R8G8_SSCALED; case 0x59: return VK_FORMAT_R16G16_USCALED; case 0x5A: return VK_FORMAT_R16G16_SSCALED;
        case 0x5B: return VK_FORMAT_R32G32_UINT; case 0x5C: return VK_FORMAT_R32G32_SINT;
        case 0x5D: return VK_FORMAT_R8G8B8_USCALED; case 0x5E: return VK_FORMAT_R8G8B8_SSCALED; case 0x5F: return VK_FORMAT_R16G16B16_USCALED; case 0x60: return VK_FORMAT_R16G16B16_SSCALED;
        case 0x61: return VK_FORMAT_R32G32B32_UINT; case 0x62: return VK_FORMAT_R32G32B32_SINT;
        case 0x63: return VK_FORMAT_R8G8B8A8_USCALED; case 0x64: return VK_FORMAT_R8G8B8A8_SSCALED; case 0x65: return VK_FORMAT_R16G16B16A16_USCALED; case 0x66: return VK_FORMAT_R16G16B16A16_SSCALED;
        case 0x67: return VK_FORMAT_R32G32B32A32_UINT; case 0x68: return VK_FORMAT_R32G32B32A32_SINT;
    }
    return VK_FORMAT_UNDEFINED;
}
VkBlendFactor vk_blend_factor(int f) {
    switch (f) {
        case 1: return VK_BLEND_FACTOR_ZERO; case 2: return VK_BLEND_FACTOR_ONE; case 3: return VK_BLEND_FACTOR_SRC_COLOR; case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA; case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; case 7: return VK_BLEND_FACTOR_DST_ALPHA; case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR; case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR; case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR; case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA; case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR; case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}
VkBlendOp vk_blend_op(int e) { switch (e) { case 2: return VK_BLEND_OP_SUBTRACT; case 3: return VK_BLEND_OP_REVERSE_SUBTRACT; case 4: return VK_BLEND_OP_MIN; case 5: return VK_BLEND_OP_MAX; } return VK_BLEND_OP_ADD; }
VkCompareOp vk_compare(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
VkStencilOp vk_stencil_op(int o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
VkLogicOp vk_logic_op(int o) { return o >= 0 && o <= 15 ? (VkLogicOp)o : VK_LOGIC_OP_COPY; }
