#version 450
#extension GL_GOOGLE_include_directive : require
// FSR 1 EASU at the present (vk_fsr.c), as a full-screen draw into an image of the window's size: AMD FidelityFX Super
// Resolution 1.0 (MIT, builtin/ffx/license.txt); fsr_rcas.frag then sharpens it into the window. A compute version (AMD's
// sample, 64 threads a group, four pixels each) took 1.9 ms at 2560x1608 on an M2.
layout(push_constant) uniform P { uvec4 Const0, Const1, Const2, Const3; } pc;
layout(set = 0, binding = 0) uniform sampler2D InputTexture;
layout(location = 0) out vec4 o;
#define A_GPU 1
#define A_GLSL 1
#if HALF
#define A_HALF 1
#endif
#include "ffx/ffx_a.h"
#if HALF   // the FP16 path (-DHALF=1): shaderFloat16
#define FSR_EASU_H 1
AH4 FsrEasuRH(AF2 p) { return AH4(textureGather(InputTexture, p, 0)); }
AH4 FsrEasuGH(AF2 p) { return AH4(textureGather(InputTexture, p, 1)); }
AH4 FsrEasuBH(AF2 p) { return AH4(textureGather(InputTexture, p, 2)); }
#else
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return textureGather(InputTexture, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(InputTexture, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(InputTexture, p, 2); }
#endif
#include "ffx/ffx_fsr1.h"
void main() {
    AU2 pos = AU2(gl_FragCoord.xy);
#if HALF
    AH3 c;
    FsrEasuH(c, pos, pc.Const0, pc.Const1, pc.Const2, pc.Const3);
#else
    AF3 c;
    FsrEasuF(c, pos, pc.Const0, pc.Const1, pc.Const2, pc.Const3);
#endif
    o = vec4(c, 1);
}
