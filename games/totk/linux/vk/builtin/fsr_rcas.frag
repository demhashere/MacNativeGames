#version 450
#extension GL_GOOGLE_include_directive : require
// FSR 1 RCAS at the present (vk_fsr.c): AMD FidelityFX Super Resolution 1.0 (MIT, builtin/ffx/license.txt) sharpens EASU's
// image straight into the window's swapchain image, drawn over the picture's rectangle (Offset: its top-left corner).
layout(push_constant) uniform P { uvec4 Const0; ivec4 Offset; } pc;
layout(set = 0, binding = 0) uniform sampler2D InputTexture;
layout(location = 0) out vec4 o;
#define A_GPU 1
#define A_GLSL 1
#if HALF
#define A_HALF 1
#endif
#include "ffx/ffx_a.h"
#if HALF   // the FP16 path (-DHALF=1): shaderFloat16, shaderInt16
#define FSR_RCAS_H 1
AH4 FsrRcasLoadH(ASW2 p) { return AH4(texelFetch(InputTexture, ASU2(p), 0)); }
void FsrRcasInputH(inout AH1 r, inout AH1 g, inout AH1 b) {}
#else
#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(InputTexture, p, 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#endif
#include "ffx/ffx_fsr1.h"
void main() {
    AU2 p = AU2(ivec2(gl_FragCoord.xy) - pc.Offset.xy);
#if HALF
    AH3 c;
    FsrRcasH(c.r, c.g, c.b, p, pc.Const0);
#else
    AF3 c;
    FsrRcasF(c.r, c.g, c.b, p, pc.Const0);
#endif
    o = vec4(c, 1);
}
