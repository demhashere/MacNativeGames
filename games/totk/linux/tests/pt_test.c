/* Presentation times and the vblank clock (vk_present.c) with a real compositor that shows nothing: a headless Weston
   (make -f linux/Makefile GEN=build/linux-dummy OUT=build/linux-test pttest starts one). A window's frames are presented
   with each frame paced by mtl_link_wait_tick, the frame hold's wait: one vblank a frame, then two; the compositor's
   answers must come back, give the refresh period, and show the frames on screen one and two refreshes apart. */
#include "../vk/vk_priv.h"
#include "nvn_dirty.h"
#include "nvn_mtl.h"
#include "../guest_mem.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static SDL_Window *g_win;
static int (*g_fault)(void *, int, void *);
void dm_fault_hook_register(int (*fn)(void *, int, void *)) { g_fault = fn; }
void dm_guest_write_prep_register(void (*fn)(void *, size_t)) { (void)fn; }
void dm_guest_write_prepare(void *addr, size_t n) { (void)addr; (void)n; }
int totk_depth_near_minus_w(void) { return 1; }
int nvn_tex_live_at(const void *mem, tex_t *out) { (void)mem; (void)out; return 0; }
int dm_nvn_pin_rekey(const tex_t *copy, const tex_t *live) { (void)copy; (void)live; return 0; }
int dm_nvn_pin_has_serial(uint64_t serial) { (void)serial; return 0; }
const char *const *front_vk_instance_extensions(uint32_t *count) { Uint32 n = 0; const char *const *e = SDL_Vulkan_GetInstanceExtensions(&n); *count = n; return e; }
int front_vk_create_surface(VkInstance i, VkSurfaceKHR *s) { return SDL_Vulkan_CreateSurface(g_win, i, NULL, s); }
void front_drawable_size(uint32_t *w, uint32_t *h) { int a = 0, b = 0; SDL_GetWindowSizeInPixels(g_win, &a, &b); *w = (uint32_t)a; *h = (uint32_t)b; }
int front_display_hz(void) { return 60; }
void front_frame_presented(void) {}
uint64_t mng_mach_absolute_time(void) { return vk_now_ns(); }
int nvn_frame_hold_now(void) { return 1; }   /* the frame hold is on: mtl_link_active follows the vblank clock */
uint64_t vk_vblank_period(void);
void vk_present_timing_stats(uint64_t *answers, uint64_t *dropped);
static void on_segv(int sig, siginfo_t *si, void *uc) { if (g_fault && g_fault(si->si_addr, 1, uc)) return; signal(sig, SIG_DFL); raise(sig); }
static uint8_t g_rec[4096]; static size_t g_rn;
static void rec(uint32_t op, const void *payload, size_t n) {
    op_hdr h = {op, (uint32_t)((sizeof h + n + 7) & ~7u)};
    memcpy(g_rec + g_rn, &h, sizeof h); memcpy(g_rec + g_rn + sizeof h, payload, n); memset(g_rec + g_rn + sizeof h + n, 0, h.size - sizeof h - n); g_rn += h.size;
}
static void run(void) { rec(OP_END, NULL, 0); mtl_sub_begin(); mtl_exec(g_rec); uint64_t f = mtl_sub_commit(); if (!mtl_wait(f, 5000000000ull)) { fprintf(stderr, "pt: GPU did not complete\n"); exit(1); } g_rn = 0; }
static int fails;
static void expect(const char *what, int ok) { printf("%-64s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) fails++; }
static int dcmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n) { qsort(v, (size_t)n, sizeof *v, dcmp); return n ? v[n / 2] : 0; }

int main(void) {
    struct sigaction sa = {0}; sa.sa_sigaction = on_segv; sa.sa_flags = SA_SIGINFO; sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "pt: SDL_Init: %s\n", SDL_GetError()); return 1; }
    int ww = 640, wh = 360; { const char *e = getenv("PT_WINDOW"); if (e) sscanf(e, "%dx%d", &ww, &wh); }   /* bigger than the picture: the FSR upscale at the present */
    g_win = SDL_CreateWindow("pt-test", ww, wh, SDL_WINDOW_VULKAN);
    if (!g_win) { fprintf(stderr, "pt: window: %s\n", SDL_GetError()); return 1; }
    const uint64_t SZ = 16u << 20;
    uint8_t *gpu = guest_mem_map_anywhere(SZ, "pt-gpu");
    pool_t pg = {MAGIC('P', 'O', 'O', 'L'), NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED, gpu, SZ};
    mtl_init(); mtl_pool_add(gpu, SZ);
    tex_t rt; memset(&rt, 0, sizeof rt);
    rt.d.target = 1; rt.d.format = 0x25; rt.d.width = 640; rt.d.height = 360; rt.d.depth = 1; rt.d.levels = 1;
    rt.d.swizzle_r = 2; rt.d.swizzle_g = 3; rt.d.swizzle_b = 4; rt.d.swizzle_a = 5;
    nvn_tex_layout(&rt.d, &rt.L); rt.pool = &pg; rt.off = 0; rt.mem = gpu; rt.serial = 1;
    p_set_rt srt = {0}; srt.n = 1; srt.color[0] = &rt;
    double iv[2][200]; int niv[2] = {0, 0}, linked = 0; uint64_t first_ticket[2] = {0, 0}, last_ticket[2] = {0, 0};
    for (int phase = 0; phase < 2; phase++) {   /* one vblank a frame, then two */
        uint64_t prev = 0;
        for (int i = 0; i < (phase ? 90 : 150); i++) {
            SDL_Event e; while (SDL_PollEvent(&e)) {}
            int on = mtl_link_active();
            if (on) { linked++; mtl_link_wait_tick(100000000ull, (unsigned)(phase + 1)); }
            else { struct timespec ts = {0, 16666667L * (phase + 1)}; nanosleep(&ts, NULL); }
            uint64_t now = vk_now_ns();
            if (on && prev && i > 20 && niv[phase] < 200) iv[phase][niv[phase]++] = (now - prev) / 1e6;
            prev = now;
            p_clear_color cc = {0, 0xf, {(float)(i % 2), 0.3f, (float)phase, 1}};
            rec(OP_SET_RT, &srt, sizeof srt); rec(OP_CLEAR_COLOR, &cc, sizeof cc); run();
            uint64_t tk = mtl_present(&rt, (phase + 1) / 60.0, NULL);
            if (i == 30) first_ticket[phase] = tk;
            last_ticket[phase] = tk;
        }
    }
    mtl_present_flush();
    struct timespec ts = {0, 200000000L}; nanosleep(&ts, NULL);   /* the last answers */
    { SDL_Event e; while (SDL_PollEvent(&e)) {} }
    uint64_t answers = 0, dropped = 0; vk_present_timing_stats(&answers, &dropped);
    uint64_t period = vk_vblank_period();
    printf("answers %llu (%llu dropped), refresh %.3f ms, frames paced by the vblank clock %d\n", (unsigned long long)answers, (unsigned long long)dropped, period / 1e6, linked);
    expect("presentation times come back for the presents", answers >= 200);
    expect("the compositor reports its refresh period", period > 15000000 && period < 18000000);
    expect("frames are paced by the vblank clock once it runs", linked >= 200);
    double m1 = median(iv[0], niv[0]), m2 = median(iv[1], niv[1]);
    printf("frame starts %.2f ms apart at one vblank a frame, %.2f ms at two\n", m1, m2);
    expect("one vblank a frame: frames start a refresh apart", fabs(m1 - period / 1e6) < 1.0);
    expect("two vblanks a frame: frames start two refreshes apart", fabs(m2 - 2 * period / 1e6) < 1.5);
    for (int phase = 0; phase < 2; phase++) {   /* on screen: the compositor's times of consecutive frames */
        double d[200]; int n = 0, missing = 0;
        for (uint64_t t = first_ticket[phase] + 1; t < last_ticket[phase] - 2 && n < 200; t++) {
            double a = mtl_presented_time(t), b = mtl_presented_time(t - 1);
            if (a > 0 && b > 0) d[n++] = (a - b) * 1e3; else missing++;
        }
        double md = median(d, n);
        printf("on screen at %s: %d intervals, median %.2f ms, %d without a time\n", phase ? "two vblanks" : "one vblank", n, md, missing);
        expect(phase ? "on screen two refreshes apart" : "on screen one refresh apart", n > 30 && fabs(md - (phase + 1) * period / 1e6) < 1.0);
    }
    printf(fails ? "pt: %d FAILED\n" : "pt: all passed\n", fails);
    fflush(stdout); SDL_DestroyWindow(g_win); SDL_Quit();
    _exit(fails ? 1 : 0);   /* the backend's threads are still running */
}
