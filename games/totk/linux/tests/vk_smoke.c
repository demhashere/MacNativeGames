/* Offscreen smoke test of the Vulkan backend (linux/vk) without game data: NVN command records written by hand are
   replayed through mtl_exec and the results read back from guest memory.
   1. a render target cleared by its load action, a channel-masked scissored clear (built-in shader), then copied to a
      CPU-visible buffer (texture -> buffer copy) and an event signalled after completion;
   2. a texture in a CPU-visible pool with block-linear contents (nvn_swizzle) uploaded at its first use and copied to
      the render target;
   3. the same texture after the CPU rewrote its pages (dirty-page tracking: uploaded again);
   4. an inline uniform update followed by a buffer copy of it (the update reaches memory first);
   5. ReadTexels of the GPU-owned render target.
   Negative control: TOTK_SMOKE_NEG=1 expects a wrong colour, so the check must fail. */
#include "../vk/vk_priv.h"
#include "nvn_dirty.h"
#include "nvn_mtl.h"
#include "../guest_mem.h"
#include "../../third_party/d2r_gfx/gfx/tex/nvn_deswizzle.h"
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <ucontext.h>

/* ---- what the runtime and the frontend would provide ---- */
static int (*g_fault)(void *, int, void *);
static void (*g_prep)(void *, size_t);
void dm_fault_hook_register(int (*fn)(void *, int, void *)) { g_fault = fn; }
void dm_guest_write_prep_register(void (*fn)(void *, size_t)) { g_prep = fn; }
void dm_guest_write_prepare(void *addr, size_t n) { if (g_prep) g_prep(addr, n); }
int totk_depth_near_minus_w(void) { return 1; }
int nvn_tex_live_at(const void *mem, tex_t *out) { (void)mem; (void)out; return 0; }
int dm_nvn_pin_rekey(const tex_t *copy, const tex_t *live) { (void)copy; (void)live; return 0; }
int dm_nvn_pin_has_serial(uint64_t serial) { (void)serial; return 0; }
const char *const *front_vk_instance_extensions(uint32_t *count) { *count = 0; return NULL; }
int front_vk_create_surface(VkInstance i, VkSurfaceKHR *s) { (void)i; (void)s; return 0; }
void front_drawable_size(uint32_t *w, uint32_t *h) { *w = 1280; *h = 720; }
int front_display_hz(void) { return 60; }
void front_frame_presented(void) {}
uint64_t mng_mach_absolute_time(void) { return vk_now_ns(); }
static void on_segv(int sig, siginfo_t *si, void *uc) {
    if (g_fault && g_fault(si->si_addr, 1, uc)) return;
    signal(sig, SIG_DFL); raise(sig);
}

/* ---- records ---- */
static uint8_t g_rec[1 << 16]; static size_t g_rn;
static void rec(uint32_t op, const void *payload, size_t n) {
    op_hdr h = {op, (uint32_t)((sizeof h + n + 7) & ~7u)};
    memcpy(g_rec + g_rn, &h, sizeof h); memcpy(g_rec + g_rn + sizeof h, payload, n);
    memset(g_rec + g_rn + sizeof h + n, 0, h.size - sizeof h - n);
    g_rn += h.size;
}
static void run(void) {
    rec(OP_END, NULL, 0);
    mtl_sub_begin(); mtl_exec(g_rec); uint64_t f = mtl_sub_commit();
    if (!mtl_wait(f, 5000000000ull)) { fprintf(stderr, "smoke: GPU did not complete\n"); exit(1); }
    g_rn = 0;
}
static uint64_t g_serial = 1000;
static void tex_init(tex_t *t, pool_t *p, int64_t off, int fmt, uint32_t w, uint32_t h) {
    memset(t, 0, sizeof *t);
    t->d.target = 1; t->d.format = fmt; t->d.width = w; t->d.height = h; t->d.depth = 1; t->d.levels = 1;
    t->d.swizzle_r = 2; t->d.swizzle_g = 3; t->d.swizzle_b = 4; t->d.swizzle_a = 5;
    nvn_tex_layout(&t->d, &t->L);
    t->pool = p; t->off = off; t->mem = p->mem + off; t->serial = ++g_serial;
}
static int fails;
static void expect(const char *what, int ok) { printf("%-60s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) fails++; }

int main(void) {
    struct sigaction sa = {0}; sa.sa_sigaction = on_segv; sa.sa_flags = SA_SIGINFO; sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    int neg = getenv("TOTK_SMOKE_NEG") && atoi(getenv("TOTK_SMOKE_NEG"));
    const uint64_t SZ = 16u << 20;
    uint8_t *cpu = guest_mem_map_anywhere(SZ, "smoke-cpu"), *gpu = guest_mem_map_anywhere(SZ, "smoke-gpu");
    if (cpu == MAP_FAILED || gpu == MAP_FAILED) { fprintf(stderr, "smoke: guest memory\n"); return 1; }
    pool_t pc = {MAGIC('P', 'O', 'O', 'L'), NVN_MEMORY_POOL_FLAGS_CPU_CACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED, cpu, SZ};
    pool_t pg = {MAGIC('P', 'O', 'O', 'L'), NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED, gpu, SZ};
    mtl_init();
    mtl_pool_add(cpu, SZ); nvn_dirty_add_range(cpu, SZ);
    mtl_pool_add(gpu, SZ);

    /* 1: clears and readback */
    tex_t rt; tex_init(&rt, &pg, 0, 0x25, 64, 64);
    uint8_t *out = cpu + (8u << 20);
    volatile uint32_t ev = 0;
    p_set_rt srt = {0}; srt.n = 1; srt.color[0] = &rt;
    rec(OP_SET_RT, &srt, sizeof srt);
    p_clear_color cc = {0, 0xf, {0.25f, 0.5f, 0.75f, 1.0f}}; rec(OP_CLEAR_COLOR, &cc, sizeof cc);
    p_scissor sc = {16, 16, 16, 16}; rec(OP_SCISSOR, &sc, sizeof sc);
    p_clear_color cm = {0, 0x1, {1.0f, 0, 0, 0}}; rec(OP_CLEAR_COLOR, &cm, sizeof cm);   /* red channel only, inside the scissor */
    p_scissor full = {0, 0, 64, 64}; rec(OP_SCISSOR, &full, sizeof full);
    p_copy_b2t t2b = {0}; t2b.src = (uint64_t)(uintptr_t)out; t2b.tex = &rt; t2b.r.width = 64; t2b.r.height = 64; t2b.r.depth = 1;
    rec(OP_COPY_T2B, &t2b, sizeof t2b);
    struct { volatile uint32_t *val; uint32_t mode, v; } se = {&ev, 0, 7}; rec(OP_SIGNAL_EVENT, &se, sizeof se);
    run();
    const uint8_t *px = out + (5 * 64 + 5) * 4, *in = out + (20 * 64 + 20) * 4;
    uint8_t want[4] = {64, 128, 191, 255}; if (neg) want[0] = 10;
    expect("load clear colour", abs(px[0] - want[0]) <= 1 && abs(px[1] - want[1]) <= 1 && abs(px[2] - want[2]) <= 1 && px[3] == want[3]);
    expect("masked scissored clear (red only, built-in shader)", in[0] == 255 && abs(in[1] - 128) <= 1 && abs(in[2] - 191) <= 1);
    expect("event written after completion", ev == 7);

    /* 2: mirrored texture upload (block-linear in the pool) */
    tex_t src; tex_init(&src, &pc, 0, 0x25, 64, 64);
    uint8_t lin[64 * 64 * 4];
    for (int i = 0; i < 64 * 64; i++) { lin[4 * i] = (uint8_t)i; lin[4 * i + 1] = (uint8_t)(i >> 6); lin[4 * i + 2] = 0x5a; lin[4 * i + 3] = 0xff; }
    nvn_swizzle(&src.d, 0, 0, lin, 64 * 4, src.mem);
    p_copy_t2t tt = {0}; tt.src = &src; tt.dst = &rt; tt.sr.width = tt.dr.width = 64; tt.sr.height = tt.dr.height = 64; tt.sr.depth = tt.dr.depth = 1;
    rec(OP_COPY_T2T, &tt, sizeof tt); rec(OP_COPY_T2B, &t2b, sizeof t2b);
    run();
    expect("upload of a block-linear texture (deswizzle)", !memcmp(out, lin, sizeof lin));

    /* 3: the CPU rewrites the texture: uploaded again at its next use */
    for (int i = 0; i < 64 * 64; i++) lin[4 * i + 2] = 0x33;
    nvn_swizzle(&src.d, 0, 0, lin, 64 * 4, src.mem);   /* the pages were protected at the upload: this faults into the tracker */
    rec(OP_COPY_T2T, &tt, sizeof tt); rec(OP_COPY_T2B, &t2b, sizeof t2b);
    run();
    expect("re-upload after a CPU write (dirty-page tracking)", !memcmp(out, lin, sizeof lin));

    /* 4: inline uniform update, then a buffer copy of it */
    uint8_t *ua = cpu + (12u << 20), *ub = cpu + (12u << 20) + 4096;
    memset(ua, 0x11, 64); memset(ub, 0, 64);
    struct { uint64_t dst, n; uint8_t data[64]; } up = {(uint64_t)(uintptr_t)ua, 64, {0}};
    for (int i = 0; i < 64; i++) up.data[i] = (uint8_t)(0xa0 + i);
    rec(OP_UPDATE_UB, &up, sizeof up);
    struct { uint64_t s, d, n; } cp = {(uint64_t)(uintptr_t)ua, (uint64_t)(uintptr_t)ub, 64}; rec(OP_COPY_B2B, &cp, sizeof cp);
    run();
    expect("inline uniform update before a buffer copy", !memcmp(ub, up.data, 64) && !memcmp(ua, up.data, 64));

    /* 5: ReadTexels of the GPU render target */
    uint8_t rb[16 * 4]; NVNcopyRegion r = {0}; r.xoffset = 8; r.yoffset = 3; r.width = 16; r.height = 1; r.depth = 1;
    mtl_read_texels(&rt, 0, 0, &r, rb, 16 * 4, 16 * 4);
    expect("ReadTexels of a GPU texture", !memcmp(rb, lin + (3 * 64 + 8) * 4, sizeof rb));

    /* 6: a draw through the whole shader chain: hand-assembled Maxwell code -> translator service -> SPIR-V -> shader
       objects. Vertex: position = attribute 0 (ALD.128 / AST.128); fragment: colour = uniform buffer c3[0..3]. */
    if (!getenv("TOTK_SMOKE_NO_SHADERS")) {
        static uint8_t vsc[0x880], fsc[0x880];
        static uint64_t vcode[64], fcode[64];   /* 0x30 NVN header + 0x50 SPH + instructions */
        uint32_t *vsph = (uint32_t *)((uint8_t *)vcode + 0x30), *fsph = (uint32_t *)((uint8_t *)fcode + 0x30);
        vsph[0] = 1 | 3u << 5 | 1u << 10;                 /* SPH type 1, version 3, vertex */
        fsph[0] = 2 | 3u << 5 | 5u << 10 | 1u << 14;      /* SPH type 2, version 3, fragment, MRT */
        fsph[18] = 0xf;                                   /* omap: target 0 RGBA */
        uint64_t vi[] = {0x001f8000fc0007e0ull, 0xEFD9FF800807FF00ull /* ALD.128 R0, a[0x80] */, 0xEFF1FF800707FF00ull /* AST.128 a[0x70], R0 */, 0xE30000000007000Full /* EXIT */};
        uint64_t fi[] = {0x001f8000fc0007e0ull, 0x4C98078C00070000ull, 0x4C98078C00170001ull, 0x4C98078C00270002ull,   /* MOV R0..R2, c[3][0..2] */
                         0x001f8000fc0007e0ull, 0x4C98078C00370003ull, 0xE30000000007000Full, 0xE30000000007000Full};   /* MOV R3, c[3][3]; EXIT */
        memcpy((uint8_t *)vcode + 0x80, vi, sizeof vi); memcpy((uint8_t *)fcode + 0x80, fi, sizeof fi);
        uint32_t vlen = 0x80 + sizeof vi, flen = 0x80 + sizeof fi, st0 = 0, st1 = 1;
        memcpy(vsc + 0x700, &vlen, 4); memcpy(vsc + 0x704, &vlen, 4); memcpy(vsc + 0x714, &st0, 4);   /* c1 at the end: none */
        memcpy(fsc + 0x700, &flen, 4); memcpy(fsc + 0x704, &flen, 4); memcpy(fsc + 0x714, &st1, 4);
        vsc[0] = 0x5a; fsc[0] = 0xa5;   /* keys differ from other runs' caches only by code */
        prog_t prog; memset(&prog, 0, sizeof prog); prog.nstage = 2; prog.serial = 77;
        prog.sd[0].control = vsc; prog.sd[0].data = (uintptr_t)vcode; prog.sd[1].control = fsc; prog.sd[1].data = (uintptr_t)fcode;
        float *vtx = (float *)(cpu + (13u << 20)); float quad[16] = {-1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1, 1, 1, 0, 1}; memcpy(vtx, quad, sizeof quad);
        float *col = (float *)(cpu + (13u << 20) + 4096); col[0] = 0.0f; col[1] = 1.0f; col[2] = 0.5f; col[3] = 1.0f;
        rec(OP_SET_RT, &srt, sizeof srt);
        p_clear_color c0 = {0, 0xf, {0, 0, 0, 0}}; rec(OP_CLEAR_COLOR, &c0, sizeof c0);
        p_bind_program bp = {&prog, 3}; rec(OP_BIND_PROGRAM, &bp, sizeof bp);
        p_bind_vattr va = {0, 1, {0}}; va_st a0 = {0x2E, 0, 0}; memcpy(&va.st[0], &a0, 4); rec(OP_BIND_VATTR, &va, sizeof va);
        p_bind_vstream vs = {0, 1, {{16, 0}}}; rec(OP_BIND_VSTREAM, &vs, sizeof vs);
        p_bind_vbuf vb = {0, 1, {{(uint64_t)(uintptr_t)vtx, sizeof quad}}}; rec(OP_BIND_VBUF, &vb, sizeof vb);
        p_bind_buf ubo = {1, 0, (uint64_t)(uintptr_t)col, 256}; rec(OP_BIND_UBO, &ubo, sizeof ubo);
        p_viewports vp = {0, 1, {{0, 0, 64, 64, 0, 1}}}; rec(OP_VIEWPORTS, &vp, sizeof vp);
        p_draw dw = {5, 0, 4, 0, 1}; rec(OP_DRAW, &dw, sizeof dw);
        rec(OP_COPY_T2B, &t2b, sizeof t2b);
        run();
        const uint8_t *c = out + (32 * 64 + 32) * 4;
        expect("draw with translated Maxwell shaders (attribute + uniform colour)", c[0] == 0 && c[1] == 255 && abs(c[2] - 128) <= 1 && c[3] == 255);
        if (fails) printf("   pixel %u %u %u %u\n", c[0], c[1], c[2], c[3]);

        /* 7: the same with a texture fetch: TEXS R0..R3 from the combined handle at c2+0x20 (NVN texture slot 0) at (0.5, 0.5) */
        static uint8_t f2c[0x880]; static uint64_t f2code[64];
        uint32_t *f2sph = (uint32_t *)((uint8_t *)f2code + 0x30);
        f2sph[0] = 2 | 3u << 5 | 5u << 10 | 1u << 14; f2sph[18] = 0xf;
        uint64_t f2i[] = {0x001f8000fc0007e0ull, 0x0103f0000007f004ull, 0x0103f0000007f005ull, 0xD850008020570400ull,   /* MOV32I R4, R5 = 0.5; TEXS R0:R1, R2:R3 */
                          0x001f8000fc0007e0ull, 0xE30000000007000Full, 0xE30000000007000Full, 0xE30000000007000Full};
        memcpy((uint8_t *)f2code + 0x80, f2i, sizeof f2i);
        uint32_t f2len = 0x80 + sizeof f2i;
        memcpy(f2c + 0x700, &f2len, 4); memcpy(f2c + 0x704, &f2len, 4); memcpy(f2c + 0x714, &st1, 4); f2c[0] = 0x77;
        prog_t prog2 = prog; prog2.serial = 78; prog2.sd[1].control = f2c; prog2.sd[1].data = (uintptr_t)f2code;
        tex_t tx; tex_init(&tx, &pc, 6u << 20, 0x25, 2, 2);
        uint8_t tl[16]; for (int i = 0; i < 4; i++) { tl[4 * i] = 10; tl[4 * i + 1] = 200; tl[4 * i + 2] = 60; tl[4 * i + 3] = 255; }
        nvn_swizzle(&tx.d, 0, 0, tl, 8, tx.mem);
        mtl_reg_tex(5, &tx, NULL);
        samp_par sp; memset(&sp, 0, sizeof sp); sp.lod_max = 1000; sp.aniso = 1; mtl_reg_samp(3, &sp);
        rec(OP_SET_RT, &srt, sizeof srt); rec(OP_CLEAR_COLOR, &c0, sizeof c0);
        p_bind_program bp2 = {&prog2, 3}; rec(OP_BIND_PROGRAM, &bp2, sizeof bp2);
        p_bind_handles th; memset(&th, 0, sizeof th); th.stage = 1; th.first = 0; th.n = 1; th.h[0] = 5 | (uint64_t)3 << 20; rec(OP_BIND_TEX, &th, sizeof th);
        rec(OP_DRAW, &dw, sizeof dw);
        rec(OP_COPY_T2B, &t2b, sizeof t2b);
        run();
        expect("draw sampling a texture (TEXS, combined handle, mirrored upload)", c[0] == 10 && c[1] == 200 && c[2] == 60 && c[3] == 255);
        if (c[1] != 200) printf("   pixel %u %u %u %u\n", c[0], c[1], c[2], c[3]);

        /* 8: each layer of a 2D array level drawn in a pass of its own: one layered pass (vk_exec.c g_lay), every layer
           keeps its own colour (the vertex shader's layer variant writes the layer from the push constant) */
        tex_t arr; tex_init(&arr, &pg, 2u << 20, 0x25, 16, 16);
        arr.d.target = 4; arr.d.depth = 6; nvn_tex_layout(&arr.d, &arr.L);
        uint64_t lay0[4]; void mtl_layer_stats(uint64_t v[4]); mtl_layer_stats(lay0);
        float *lc = (float *)(cpu + (13u << 20) + 8192);
        rec(OP_BIND_PROGRAM, &bp, sizeof bp);
        p_viewports vp16 = {0, 1, {{0, 0, 16, 16, 0, 1}}}; rec(OP_VIEWPORTS, &vp16, sizeof vp16);
        for (int l = 0; l < 6; l++) {
            float *c4 = lc + 64 * l; c4[0] = l * 0.15f; c4[1] = 1.0f - l * 0.15f; c4[2] = 0.5f; c4[3] = 1.0f;
            p_set_rt lr = {0}; lr.n = 1; lr.color[0] = &arr; lr.has_cview[0] = 1; lr.cview[0].set_layers = 1; lr.cview[0].min_layer = (uint16_t)l; lr.cview[0].nlayers = 1;
            rec(OP_SET_RT, &lr, sizeof lr);
            p_bind_buf lu = {1, 0, (uint64_t)(uintptr_t)c4, 256}; rec(OP_BIND_UBO, &lu, sizeof lu);
            rec(OP_DRAW, &dw, sizeof dw);
        }
        uint8_t *lout = cpu + (9u << 20);
        for (int l = 0; l < 6; l++) {
            p_copy_b2t lb = {0}; lb.src = (uint64_t)(uintptr_t)(lout + l * 16 * 16 * 4); lb.tex = &arr; lb.has_view = 1; lb.view.set_layers = 1; lb.view.min_layer = (uint16_t)l; lb.view.nlayers = 1;
            lb.r.width = 16; lb.r.height = 16; lb.r.depth = 1; rec(OP_COPY_T2B, &lb, sizeof lb);
        }
        run();
        uint64_t lay1[4]; mtl_layer_stats(lay1);
        int lok = 1;
        for (int l = 0; l < 6; l++) {
            const uint8_t *q = lout + (l * 16 * 16 + 8 * 16 + 8) * 4; int r0 = (int)lroundf(l * 0.15f * 255), g0 = (int)lroundf((1.0f - l * 0.15f) * 255);
            if (abs(q[0] - r0) > 1 || abs(q[1] - g0) > 1 || abs(q[2] - 128) > 1 || q[3] != 255) { lok = 0; printf("   layer %d pixel %u %u %u %u (want %d %d 128 255)\n", l, q[0], q[1], q[2], q[3], r0, g0); }
        }
        expect("layers of an array level drawn in one layered pass", lok);
        expect("  ... as one pass with the other five targets taken in", lay1[1] - lay0[1] == 1 && lay1[2] - lay0[2] == 5);
        if (lay1[1] - lay0[1] != 1 || lay1[2] - lay0[2] != 5) printf("   layered passes %llu, targets taken in %llu\n", (unsigned long long)(lay1[1] - lay0[1]), (unsigned long long)(lay1[2] - lay0[2]));

        /* 9: in one pass a draw samples the attachment an earlier draw of the pass drew (nvn_mtl.m att_unstored_read):
           it sees that draw's colour, not what the attachment held when the pass began */
        tex_t fbt; tex_init(&fbt, &pg, 3u << 20, 0x25, 64, 64);
        mtl_reg_tex(6, &fbt, NULL);
        p_set_rt fr = {0}; fr.n = 1; fr.color[0] = &fbt;
        p_clear_color red = {0, 0xf, {1, 0, 0, 1}};
        rec(OP_SET_RT, &fr, sizeof fr); rec(OP_CLEAR_COLOR, &red, sizeof red);
        p_set_rt other = srt; rec(OP_SET_RT, &other, sizeof other); rec(OP_CLEAR_COLOR, &c0, sizeof c0);   /* the red reaches memory */
        run();
        float *green = lc + 64 * 7; green[0] = 0; green[1] = 1; green[2] = 0; green[3] = 1;
        rec(OP_SET_RT, &fr, sizeof fr);
        rec(OP_BIND_PROGRAM, &bp, sizeof bp); rec(OP_VIEWPORTS, &vp, sizeof vp);
        p_bind_buf gu = {1, 0, (uint64_t)(uintptr_t)green, 256}; rec(OP_BIND_UBO, &gu, sizeof gu);
        rec(OP_DRAW, &dw, sizeof dw);                                    /* green into the attachment, sampling nothing */
        rec(OP_BIND_PROGRAM, &bp2, sizeof bp2);
        p_bind_handles th2 = th; th2.h[0] = 6 | (uint64_t)3 << 20; rec(OP_BIND_TEX, &th2, sizeof th2);
        p_scissor tile = {32, 32, 16, 16}; rec(OP_SCISSOR, &tile, sizeof tile);
        rec(OP_DRAW, &dw, sizeof dw);   /* 32..47 square: the attachment's own centre texel (32, 32), in the same GPU tile (a tile is stored only when all of it is drawn) */
        rec(OP_SCISSOR, &full, sizeof full);
        p_copy_b2t fb2 = t2b; fb2.tex = &fbt; rec(OP_COPY_T2B, &fb2, sizeof fb2);
        run();
        const uint8_t *fl = out + (40 * 64 + 40) * 4;
        expect("a draw sampling what an earlier draw of its pass drew", fl[0] == 0 && fl[1] == 255 && fl[2] == 0);
        if (fl[1] != 255) printf("   pixel %u %u %u %u\n", fl[0], fl[1], fl[2], fl[3]);

        /* 11: an inline update into a vertex buffer of several pages (the guard walks the submission's updated pages),
           then a draw: its vertices are the updated ones (the old ones are degenerate) */
        float *vtx3 = (float *)(cpu + (14u << 20)); memset(vtx3, 0, 3 * 4096);
        struct { uint64_t dst, n; float data[16]; } vu = {(uint64_t)(uintptr_t)vtx3, sizeof quad, {0}}; memcpy(vu.data, quad, sizeof quad);
        rec(OP_UPDATE_UB, &vu, sizeof vu);
        rec(OP_SET_RT, &srt, sizeof srt); rec(OP_CLEAR_COLOR, &c0, sizeof c0);
        rec(OP_BIND_PROGRAM, &bp, sizeof bp); rec(OP_VIEWPORTS, &vp, sizeof vp);
        p_bind_vbuf vb3 = {0, 1, {{(uint64_t)(uintptr_t)vtx3, 3 * 4096}}}; rec(OP_BIND_VBUF, &vb3, sizeof vb3);
        rec(OP_BIND_UBO, &ubo, sizeof ubo);
        rec(OP_DRAW, &dw, sizeof dw);
        rec(OP_COPY_T2B, &t2b, sizeof t2b);
        run();
        const uint8_t *vq = out + (32 * 64 + 32) * 4;
        expect("draw after an inline update of its multi-page vertex buffer", vq[0] == 0 && vq[1] == 255 && abs(vq[2] - 128) <= 1);
        if (vq[1] != 255) printf("   pixel %u %u %u %u\n", vq[0], vq[1], vq[2], vq[3]);

        /* 12: lazy clears (vk_exec.c g_carry). A clear with no draw after it waits past SetRenderTargets for the next pass
           on its texture, which clears it as its load action; a clear of another texture lands in a pass of its own
           before that pass. A draw sampling a texture whose clear is carried, and a copy of one, see the clear. */
        tex_t la, lb, lz, ld, le;
        tex_init(&la, &pg, 4u << 20, 0x25, 64, 64); tex_init(&lb, &pg, (4u << 20) + (1u << 19), 0x25, 64, 64); tex_init(&lz, &pg, 5u << 20, 0x25, 64, 64);
        tex_init(&ld, &pg, (5u << 20) + (1u << 19), 0x25, 64, 64); tex_init(&le, &pg, 6u << 20, 0x25, 64, 64);
        p_set_rt ra = {0}, rb_ = {0}, rc = {0}, rd = {0}, re = {0};
        ra.n = rb_.n = rc.n = rd.n = re.n = 1; ra.color[0] = &la; rb_.color[0] = &lb; rc.color[0] = &lz; rd.color[0] = &ld; re.color[0] = &le;
        p_clear_color blue = {0, 0xf, {0, 0, 1, 1}}, yellow = {0, 0xf, {1, 1, 0, 1}}, grey = {0, 0xf, {0.5f, 0.5f, 0.5f, 1}};
        uint8_t *o2 = cpu + (10u << 20), *o3 = o2 + 65536, *o4 = o2 + 2 * 65536;
        uint64_t ps0, ds0, ps1, ds1; mtl_stats(&ps0, &ds0);
        rec(OP_SET_RT, &ra, sizeof ra); rec(OP_CLEAR_COLOR, &blue, sizeof blue);
        rec(OP_SET_RT, &rb_, sizeof rb_); rec(OP_CLEAR_COLOR, &yellow, sizeof yellow);
        rec(OP_SET_RT, &ra, sizeof ra);
        rec(OP_BIND_PROGRAM, &bp, sizeof bp); rec(OP_VIEWPORTS, &vp, sizeof vp); rec(OP_BIND_VBUF, &vb, sizeof vb); rec(OP_BIND_UBO, &ubo, sizeof ubo);
        p_scissor left = {0, 0, 32, 64}; rec(OP_SCISSOR, &left, sizeof left);
        rec(OP_DRAW, &dw, sizeof dw);   /* the left half: (0, 255, 128) */
        rec(OP_SCISSOR, &full, sizeof full);
        mtl_stats(&ps1, &ds1);
        p_copy_b2t ca_ = t2b; ca_.src = (uint64_t)(uintptr_t)o2; ca_.tex = &la; rec(OP_COPY_T2B, &ca_, sizeof ca_);
        p_copy_b2t cb_ = t2b; cb_.src = (uint64_t)(uintptr_t)o3; cb_.tex = &lb; rec(OP_COPY_T2B, &cb_, sizeof cb_);
        run();
        uint64_t ps2, ds2; mtl_stats(&ps2, &ds2);
        const uint8_t *al = o2 + (32 * 64 + 16) * 4, *ar = o2 + (32 * 64 + 48) * 4, *bq = o3 + (32 * 64 + 32) * 4;
        int lazy = !(getenv("TOTK_VK_LAZY_CLEAR") && !strcmp(getenv("TOTK_VK_LAZY_CLEAR"), "0"));
        expect("lazy clears: the clear taken by the next pass on its target", al[0] == 0 && al[1] == 255 && ar[0] == 0 && ar[1] == 0 && ar[2] == 255 && bq[0] == 255 && bq[1] == 255 && bq[2] == 0);
        if (ar[2] != 255 || bq[0] != 255) printf("   drawn %u %u %u, cleared %u %u %u, other target %u %u %u\n", al[0], al[1], al[2], ar[0], ar[1], ar[2], bq[0], bq[1], bq[2]);
        expect("  ... in two passes instead of three", ps2 - ps0 == (lazy ? 2u : 3u));
        if (ps2 - ps0 != (lazy ? 2u : 3u)) printf("   %llu passes (%llu before the copies)\n", (unsigned long long)(ps2 - ps0), (unsigned long long)(ps1 - ps0));
        mtl_reg_tex(7, &lz, NULL);
        rec(OP_SET_RT, &rc, sizeof rc); rec(OP_CLEAR_COLOR, &red, sizeof red);
        p_set_rt rdz = rd; rdz.n = 2; rdz.color[1] = &lz;   /* the pass attaches the texture it samples: its carried clear must not become the load clear */
        rec(OP_SET_RT, &rdz, sizeof rdz);
        rec(OP_BIND_PROGRAM, &bp2, sizeof bp2);
        p_bind_handles th3 = th; th3.h[0] = 7 | (uint64_t)3 << 20; rec(OP_BIND_TEX, &th3, sizeof th3);
        rec(OP_DRAW, &dw, sizeof dw);   /* samples lz, whose clear is carried */
        p_copy_b2t cd_ = t2b; cd_.src = (uint64_t)(uintptr_t)o4; cd_.tex = &ld; rec(OP_COPY_T2B, &cd_, sizeof cd_);
        run();
        const uint8_t *dq = o4 + (32 * 64 + 32) * 4;
        expect("lazy clears: a draw sampling a texture whose clear is carried", dq[0] == 255 && dq[1] == 0 && dq[2] == 0 && dq[3] == 255);
        if (dq[0] != 255) printf("   pixel %u %u %u %u\n", dq[0], dq[1], dq[2], dq[3]);
        rec(OP_SET_RT, &re, sizeof re); rec(OP_CLEAR_COLOR, &grey, sizeof grey);
        rec(OP_SET_RT, &srt, sizeof srt);
        p_copy_b2t ce_ = t2b; ce_.src = (uint64_t)(uintptr_t)o2; ce_.tex = &le; rec(OP_COPY_T2B, &ce_, sizeof ce_);
        run();
        const uint8_t *eq = o2 + (32 * 64 + 32) * 4;
        expect("lazy clears: a copy of a texture whose clear is carried", abs(eq[0] - 128) <= 1 && abs(eq[1] - 128) <= 1 && abs(eq[2] - 128) <= 1);
        if (abs(eq[0] - 128) > 1) printf("   pixel %u %u %u %u\n", eq[0], eq[1], eq[2], eq[3]);

        /* 13: the UI squeeze per vertex (vk_shader.c spv_squeeze_patch), every 6-index draw squeezed to half the height: a
           full-screen quad still fills the target (its vertices lie on the edges), a quad of half the screen is squeezed
           to a quarter. Through the viewport (the way before) the full-screen quad covered only the middle half. */
        void vk_ui_squeeze_test(double ux, double uy, int edges);
        float *sqv = (float *)(cpu + (15u << 20)); uint16_t *sqi = (uint16_t *)(cpu + (15u << 20) + 4096);
        float *sqc = (float *)(cpu + (15u << 20) + 8192);
        const float fullq[16] = {-1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1, 1, 1, 0, 1}, halfq[16] = {-0.5f, -0.5f, 0, 1, 0.5f, -0.5f, 0, 1, -0.5f, 0.5f, 0, 1, 0.5f, 0.5f, 0, 1};
        memcpy(sqv, fullq, sizeof fullq); memcpy(sqv + 16, halfq, sizeof halfq);
        const uint16_t qi[6] = {0, 1, 2, 2, 1, 3}; memcpy(sqi, qi, sizeof qi);
        const float redc[4] = {1, 0, 0, 1}, greenc[4] = {0, 1, 0, 1}; memcpy(sqc, redc, 16); memcpy(sqc + 64, greenc, 16);
        tex_t sqt; tex_init(&sqt, &pg, 7u << 20, 0x25, 64, 64);
        p_set_rt rsq = {0}; rsq.n = 1; rsq.color[0] = &sqt;
        p_clear_color black = {0, 0xf, {0, 0, 0, 1}};
        p_draw_idx dq6 = {4, 1, 6, 0, 0, 1, (uint64_t)(uintptr_t)sqi};
        for (int edges = 1; edges >= 0; edges--) {
            vk_ui_squeeze_test(1, 2, edges);
            rec(OP_SET_RT, &rsq, sizeof rsq); rec(OP_CLEAR_COLOR, &black, sizeof black);
            rec(OP_BIND_PROGRAM, &bp, sizeof bp); rec(OP_VIEWPORTS, &vp, sizeof vp); rec(OP_SCISSOR, &full, sizeof full);
            p_bind_vbuf vf = {0, 1, {{(uint64_t)(uintptr_t)sqv, sizeof fullq}}}; rec(OP_BIND_VBUF, &vf, sizeof vf);
            p_bind_buf ur = {1, 0, (uint64_t)(uintptr_t)sqc, 256}; rec(OP_BIND_UBO, &ur, sizeof ur);
            rec(OP_DRAW_IDX, &dq6, sizeof dq6);
            p_bind_vbuf vh = {0, 1, {{(uint64_t)(uintptr_t)(sqv + 16), sizeof halfq}}}; rec(OP_BIND_VBUF, &vh, sizeof vh);
            p_bind_buf ug = {1, 0, (uint64_t)(uintptr_t)(sqc + 64), 256}; rec(OP_BIND_UBO, &ug, sizeof ug);
            rec(OP_DRAW_IDX, &dq6, sizeof dq6);
            p_copy_b2t cs_ = t2b; cs_.src = (uint64_t)(uintptr_t)o2; cs_.tex = &sqt; rec(OP_COPY_T2B, &cs_, sizeof cs_);
            run();
            vk_ui_squeeze_test(0, 0, -1);
            const uint8_t *top = o2 + (2 * 64 + 32) * 4, *bot = o2 + (61 * 64 + 32) * 4, *mid = o2 + (32 * 64 + 32) * 4, *off = o2 + (20 * 64 + 32) * 4;
            if (edges) {
                expect("UI squeeze per vertex: a full-screen quad still reaches the edges", top[0] == 255 && top[1] == 0 && bot[0] == 255 && bot[1] == 0);
                expect("  ... and a smaller quad is squeezed", mid[1] == 255 && mid[0] == 0 && off[0] == 255 && off[1] == 0);
                if (top[0] != 255 || off[1]) printf("   top %u %u %u, bottom %u %u %u, middle %u %u %u, y 20 %u %u %u\n", top[0], top[1], top[2], bot[0], bot[1], bot[2], mid[0], mid[1], mid[2], off[0], off[1], off[2]);
            } else {
                expect("  (through the viewport the full-screen quad stops short of the edges)", top[0] == 0 && top[1] == 0 && bot[0] == 0 && mid[1] == 255);
            }
        }
    }
    /* 14: the upscale at the present (vk_fsr.c): a 64x64 picture to 128x128 by Snapdragon GSR 1 and by FSR 1. A flat colour
       stays that colour, black and white stay black and white next to a hard edge (no ringing past them), and the edge is
       at least as sharp as the bilinear blit's, which spreads it over two pixels at this scale. */
    {
        int vk_fsr_test(const tex_t *t, uint32_t w, uint32_t h, int mode, void *out);
        tex_t ft; tex_init(&ft, &pc, 7u << 20, 0x25, 64, 64);
        static uint8_t pat[64 * 64 * 4];
        for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) {
            uint8_t *q = pat + (y * 64 + x) * 4;
            if (y < 32) { q[0] = 200; q[1] = 100; q[2] = 50; } else { uint8_t v = x < 32 ? 0 : 255; q[0] = q[1] = q[2] = v; }
            q[3] = 255;
        }
        nvn_swizzle(&ft.d, 0, 0, pat, 64 * 4, ft.mem);
        static uint8_t bo[128 * 128 * 4], fo[128 * 128 * 4];
        int ranb = vk_fsr_test(&ft, 128, 128, 0, bo);
        int d_b = bo[(100 * 128 + 64) * 4] - bo[(100 * 128 + 63) * 4];
        for (int mode = 1; mode <= 2; mode++) {
            const char *name = mode == 1 ? "Snapdragon GSR" : "FSR 1";
            int ran = vk_fsr_test(&ft, 128, 128, mode, fo);
            const uint8_t *fl = fo + (40 * 128 + 64) * 4, *blk = fo + (100 * 128 + 40) * 4, *wht = fo + (100 * 128 + 88) * 4;
            int flat = ran && abs(fl[0] - 200) <= 2 && abs(fl[1] - 100) <= 2 && abs(fl[2] - 50) <= 2;
            int bw = ran && blk[0] <= 2 && wht[0] >= 253;
            int d_f = fo[(100 * 128 + 64) * 4] - fo[(100 * 128 + 63) * 4];
            int lo = 255, hi = 0; for (int x = 0; x < 128; x++) { int v = fo[(100 * 128 + x) * 4]; if (v < lo) lo = v; if (v > hi) hi = v; }
            char what[96];
            snprintf(what, sizeof what, "upscale by %s: a flat colour keeps its colour", name); expect(what, flat);
            expect("  ... black and white stay so beside an edge, nothing rings past them", bw && lo <= 2 && hi >= 253);
            expect("  ... and the edge is at least as sharp as the bilinear blit's", ran && ranb && d_f >= d_b && d_f > 100);
            printf("   flat %u %u %u, black %u, white %u, step at the edge %d (bilinear %d), range %d..%d\n", fl[0], fl[1], fl[2], blk[0], wht[0], d_f, d_b, lo, hi);
        }
    }
    /* 10: the frame ring (F9, vk_present.c): presented frames saved after the fact, in order, at the ring's size */
    {
        char dir[] = "/tmp/vk-smoke-ring-XXXXXX"; if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        setenv("TOTK_FRAME_RING_DIR", dir, 1);
        rec(OP_SET_RT, &srt, sizeof srt); rec(OP_CLEAR_COLOR, &cc, sizeof cc); run();   /* (64, 128, 191) */
        for (int i = 0; i < 3; i++) mtl_present_wait(mtl_present(&rt, 1.0 / 60, NULL));
        void mtl_frame_ring_save(void); mtl_frame_ring_save();
        mtl_present_wait(mtl_present(&rt, 1.0 / 60, NULL));
        char cmd[300]; snprintf(cmd, sizeof cmd, "ls %s/frames-*/*.ppm 2>/dev/null | wc -l", dir);
        FILE *pp = popen(cmd, "r"); int nf = 0; if (pp) { if (fscanf(pp, "%d", &nf) != 1) nf = 0; pclose(pp); }
        snprintf(cmd, sizeof cmd, "ls %s/frames-*/000-*.ppm 2>/dev/null | head -1", dir);
        pp = popen(cmd, "r"); char path[400] = {0}; if (pp) { if (!fgets(path, sizeof path, pp)) path[0] = 0; pclose(pp); }
        path[strcspn(path, "\n")] = 0;
        int ok = nf == 4; FILE *f = path[0] ? fopen(path, "rb") : NULL;
        if (f) { int w = 0, h = 0, mx = 0; ok &= fscanf(f, "P6 %d %d %d", &w, &h, &mx) == 3 && w == 480 && h == 270; fgetc(f);
                 uint8_t px3[3]; fseek(f, (long)((135 * 480 + 240) * 3), SEEK_CUR); ok &= fread(px3, 1, 3, f) == 3 && abs(px3[0] - 64) <= 1 && abs(px3[1] - 128) <= 1 && abs(px3[2] - 191) <= 1; fclose(f); }
        else ok = 0;
        expect("frame ring: presented frames saved after the fact (F9)", ok);
        if (!ok) printf("   %d frames, first %s\n", nf, path);
        snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {}
    }
    printf(fails ? "smoke: %d FAILED\n" : "smoke: all passed\n", fails);
    return fails ? 1 : 0;
}
