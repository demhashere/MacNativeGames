/* The game window on Linux (in place of runtime/frontend.m): an SDL3 window with a Vulkan surface, the keyboard and the
   mouse, the VI service the game opens its display through, the vsync event, the boot status and the text entry of the
   software keyboard. SDL video and events stay on the main thread; the game runs on its own thread. */
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "runtime.h"
#include "game_version.h"
#include "totk_ui_lang.h"
#include "frontend_sdl.h"

void hle_bind_system_event(void *, int);
void hle_signal_system_event(void *);
int fps_overlay_toggle_hints(void);
int runtime_fs_save_busy(void);

static SDL_Window *window;
static Uint32 g_post_event;
static _Atomic uint64_t presentedFrames;
static uint64_t bootStartNS;
static _Atomic int focused = 1, warmPending;
static pthread_mutex_t bootMu = PTHREAD_MUTEX_INITIALIZER;
static char bootStage[160], bootDetail[200];
static int headless(void) { static int v = -1; if (v < 0) v = getenv("TOTK_HEADLESS") != NULL; return v; }

void front_sdl_post(void (*fn)(void)) {
    if (!g_post_event) return;
    SDL_Event e; SDL_zero(e); e.type = g_post_event; e.user.data1 = (void *)fn; SDL_PushEvent(&e);
}
SDL_Window *front_sdl_window(void) { return window; }
const char *const *front_vk_instance_extensions(uint32_t *count) {
    if (headless() || !window) { *count = 0; return NULL; }
    return SDL_Vulkan_GetInstanceExtensions(count);
}
/* Surface creation on the main thread (the Wayland / X11 objects belong to SDL's event thread). */
static struct { VkInstance inst; VkSurfaceKHR surf; int done, ok; } g_surf_req;
static pthread_mutex_t g_surf_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_surf_cv = PTHREAD_COND_INITIALIZER;
static void make_surface_main(void) {
    int ok = window && SDL_Vulkan_CreateSurface(window, g_surf_req.inst, NULL, &g_surf_req.surf);
    if (!ok) fprintf(stderr, "[totk] Vulkan surface: %s\n", SDL_GetError());
    pthread_mutex_lock(&g_surf_mu); g_surf_req.ok = ok; g_surf_req.done = 1; pthread_cond_broadcast(&g_surf_cv); pthread_mutex_unlock(&g_surf_mu);
}
int front_vk_create_surface(VkInstance instance, VkSurfaceKHR *surface) {
    if (!window) return 0;
    pthread_mutex_lock(&g_surf_mu); g_surf_req.inst = instance; g_surf_req.done = 0; pthread_mutex_unlock(&g_surf_mu);
    if (SDL_IsMainThread()) make_surface_main(); else front_sdl_post(make_surface_main);
    pthread_mutex_lock(&g_surf_mu); while (!g_surf_req.done) pthread_cond_wait(&g_surf_cv, &g_surf_mu); int ok = g_surf_req.ok; pthread_mutex_unlock(&g_surf_mu);
    if (ok) *surface = g_surf_req.surf;
    return ok;
}
void front_drawable_size(uint32_t *w, uint32_t *h) {
    int pw = 0, ph = 0; if (window) SDL_GetWindowSizeInPixels(window, &pw, &ph);
    *w = pw > 0 ? (uint32_t)pw : 1920; *h = ph > 0 ? (uint32_t)ph : 1080;
}
int front_display_hz(void) {
    if (!window) return 60;
    const SDL_DisplayMode *m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
    int hz = m && m->refresh_rate > 1 ? (int)lroundf(m->refresh_rate) : 60;
    return hz < 24 ? 60 : hz;
}

/* ---------- boot status: the window title and the log (the backend draws it into the window until the first frame) ---------- */
static void title_main(void) {
    if (!window) return;
    char t[512];
    pthread_mutex_lock(&bootMu);
    if (atomic_load(&presentedFrames) && !atomic_load(&warmPending)) snprintf(t, sizeof t, "%s  -  Ver. %s", GAME_NAME, TOTK_GAME_VERSION);
    else snprintf(t, sizeof t, "%s  -  %s%s%s", GAME_NAME, bootStage[0] ? bootStage : "Starting", bootDetail[0] ? ": " : "", bootDetail);
    pthread_mutex_unlock(&bootMu);
    SDL_SetWindowTitle(window, t);
}
static void boot_set(const char *stage, const char *detail) {
    pthread_mutex_lock(&bootMu);
    snprintf(bootStage, sizeof bootStage, "%s", stage ? stage : TOTK_UI("正在启动", "Starting"));
    snprintf(bootDetail, sizeof bootDetail, "%s", detail ? detail : "");
    pthread_mutex_unlock(&bootMu);
    front_sdl_post(title_main);
}
const char *front_boot_line(char *out, size_t n) {
    if (atomic_load(&presentedFrames) && !atomic_load(&warmPending)) return NULL;
    pthread_mutex_lock(&bootMu); snprintf(out, n, "%s%s%s", bootStage, bootDetail[0] ? "  " : "", bootDetail); pthread_mutex_unlock(&bootMu);
    return out;
}
void front_frame_presented(void) {
    if (atomic_fetch_add_explicit(&presentedFrames, 1, memory_order_relaxed) == 0) {
        fprintf(stderr, "[totk-boot] %.3f first presented frame\n", (runtime_now_ns() - bootStartNS) / 1e9);
        front_sdl_post(title_main);
    }
}
void runtime_boot_progress(const char *stage, uint64_t done, uint64_t total) {
    static const char *last;
    if (stage != last) { fprintf(stderr, "[totk-boot] %.3f %s\n", bootStartNS ? (runtime_now_ns() - bootStartNS) / 1e9 : 0, stage ? stage : "startup"); last = stage; }
    char d[96]; if (total) snprintf(d, sizeof d, "%.1f / %.1f GB", done / 1e9, total / 1e9); else snprintf(d, sizeof d, "%s", TOTK_UI("正在处理，请稍候", "Working, please wait"));
    boot_set(stage, d);
}
void runtime_boot_count(const char *stage, uint64_t done, uint64_t total) {
    static const char *last;
    if (stage != last) { fprintf(stderr, "[totk-boot] %.3f %s\n", bootStartNS ? (runtime_now_ns() - bootStartNS) / 1e9 : 0, stage ? stage : "startup"); last = stage; }
    char d[64]; snprintf(d, sizeof d, "%llu / %llu", (unsigned long long)done, (unsigned long long)total);
    boot_set(stage, d);
}
void runtime_boot_status(const char *stage, const char *detail) {
    if (bootStartNS) fprintf(stderr, "[totk-boot] %.3f %s — %s\n", (runtime_now_ns() - bootStartNS) / 1e9, stage ? stage : "startup", detail ? detail : "");
    boot_set(stage, detail ? detail : TOTK_UI("正在处理，请稍候", "Working, please wait"));
}
void runtime_warm_progress(uint64_t done, uint64_t total, uint64_t shaders, uint64_t shader_total, uint64_t pipelines, uint64_t pipeline_total) {
    atomic_store(&warmPending, 1);
    double percent = total ? fmin(100.0, (double)done / (double)total * 100.0) : 100.0;
    char s[96], d[160];
    snprintf(s, sizeof s, TOTK_UI("预热着色器与管线 %.0f%%", "Warming shaders and pipelines %.0f%%"), floor(percent));
    snprintf(d, sizeof d, TOTK_UI("着色器 %llu/%llu · 管线 %llu/%llu", "shaders %llu/%llu, pipelines %llu/%llu"), (unsigned long long)shaders, (unsigned long long)shader_total,
             (unsigned long long)pipelines, (unsigned long long)pipeline_total);
    boot_set(s, d);
}
void runtime_warm_done(void) { atomic_store(&warmPending, 0); front_sdl_post(title_main); }
uint64_t front_game_res(int docked) { (void)docked; return (720ULL << 32) | 1280; }
int runtime_front_focus(void) { return 1; }
int runtime_front_window_visible(void) {
    if (!window) return -1;
    SDL_WindowFlags f = SDL_GetWindowFlags(window);
    return !(f & (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED | SDL_WINDOW_OCCLUDED));
}
static void quit_main(void) { SDL_Event e; SDL_zero(e); e.type = SDL_EVENT_QUIT; SDL_PushEvent(&e); }
void runtime_front_request_close(void) { front_sdl_post(quit_main); }
void runtime_shortcut_help_toggle(void) {
    int mode = fps_overlay_toggle_hints();
    fprintf(stderr, "[totk] overlay (key 1): %s\n", mode == 1 ? "all" : mode == 2 ? "FPS only" : "off");
}

/* ---------- keyboard: SDL scancodes to the Mac virtual key codes runtime/input.c maps ---------- */
static int mac_key(SDL_Scancode s) {
    switch (s) {
    case SDL_SCANCODE_A: return 0; case SDL_SCANCODE_S: return 1; case SDL_SCANCODE_D: return 2; case SDL_SCANCODE_F: return 3;
    case SDL_SCANCODE_H: return 4; case SDL_SCANCODE_G: return 5; case SDL_SCANCODE_Z: return 6; case SDL_SCANCODE_X: return 7;
    case SDL_SCANCODE_C: return 8; case SDL_SCANCODE_V: return 9; case SDL_SCANCODE_B: return 11; case SDL_SCANCODE_Q: return 12;
    case SDL_SCANCODE_W: return 13; case SDL_SCANCODE_E: return 14; case SDL_SCANCODE_R: return 15; case SDL_SCANCODE_Y: return 16;
    case SDL_SCANCODE_T: return 17; case SDL_SCANCODE_1: return 18; case SDL_SCANCODE_2: return 19; case SDL_SCANCODE_3: return 20;
    case SDL_SCANCODE_4: return 21; case SDL_SCANCODE_6: return 22; case SDL_SCANCODE_5: return 23; case SDL_SCANCODE_EQUALS: return 24;
    case SDL_SCANCODE_9: return 25; case SDL_SCANCODE_7: return 26; case SDL_SCANCODE_MINUS: return 27; case SDL_SCANCODE_8: return 28;
    case SDL_SCANCODE_0: return 29; case SDL_SCANCODE_RIGHTBRACKET: return 30; case SDL_SCANCODE_O: return 31; case SDL_SCANCODE_U: return 32;
    case SDL_SCANCODE_LEFTBRACKET: return 33; case SDL_SCANCODE_I: return 34; case SDL_SCANCODE_P: return 35; case SDL_SCANCODE_RETURN: return 36;
    case SDL_SCANCODE_KP_ENTER: return 36;
    case SDL_SCANCODE_L: return 37; case SDL_SCANCODE_J: return 38; case SDL_SCANCODE_APOSTROPHE: return 39; case SDL_SCANCODE_K: return 40;
    case SDL_SCANCODE_SEMICOLON: return 41; case SDL_SCANCODE_BACKSLASH: return 42; case SDL_SCANCODE_COMMA: return 43; case SDL_SCANCODE_SLASH: return 44;
    case SDL_SCANCODE_N: return 45; case SDL_SCANCODE_M: return 46; case SDL_SCANCODE_PERIOD: return 47; case SDL_SCANCODE_TAB: return 48;
    case SDL_SCANCODE_SPACE: return 49; case SDL_SCANCODE_GRAVE: return 50; case SDL_SCANCODE_BACKSPACE: return 51; case SDL_SCANCODE_ESCAPE: return 53;
    case SDL_SCANCODE_LSHIFT: case SDL_SCANCODE_RSHIFT: return 56;   /* Shift: ZR */
    case SDL_SCANCODE_LALT: case SDL_SCANCODE_RALT: return 58;       /* Option: ZL */
    case SDL_SCANCODE_LEFT: return 123; case SDL_SCANCODE_RIGHT: return 124; case SDL_SCANCODE_DOWN: return 125; case SDL_SCANCODE_UP: return 126;
    default: return -1;
    }
}

/* ---------- mouse mode (key 7): the window holds the pointer while it has the keyboard ---------- */
static int textEntryOpen, mouseHeld;
static void front_mouse_update(void) {   /* main thread */
    int want = window && runtime_mouse_enabled() && atomic_load(&focused) && !textEntryOpen;
    if (want == mouseHeld) return;
    mouseHeld = want;
    SDL_SetWindowRelativeMouseMode(window, want);
    if (!want) runtime_mouse_release();
}

/* ---------- text entry (nn::swkbd, runtime/swkbd_host.c): typed into the window, prompt in the title ---------- */
static struct { const RuntimeTextRequest *req; char text[1024]; int active, done, ok; } g_tx;
static pthread_mutex_t g_tx_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tx_cv = PTHREAD_COND_INITIALIZER;
static size_t utf8_units(const char *s) { size_t n = 0; for (; *s; s++) if ((*s & 0xc0) != 0x80) n++; return n; }
static int tx_allowed(uint32_t c) {
    const RuntimeTextRequest *r = g_tx.req; int digit = c >= '0' && c <= '9'; uint32_t d = r->disabled;
    if (r->digits_only && !digit) return 0;
    if (c < 0x20 || c == 0x7f) return 0;
    if ((d & 1u << 1 && c == ' ') || (d & 1u << 2 && c == '@') || (d & 1u << 3 && c == '%') || (d & 1u << 4 && c == '/') || (d & 1u << 5 && c == '\\') || (d & 1u << 6 && digit)) return 0;
    return 1;
}
static void tx_title(void) {
    char t[1400];
    snprintf(t, sizeof t, "%s: %s_   (%s)", g_tx.req->prompt ? g_tx.req->prompt : "", g_tx.req->secure ? "***" : g_tx.text,
             TOTK_UI("回车确定，Esc 取消", "Return to confirm, Esc to cancel"));
    SDL_SetWindowTitle(window, t);
}
static void tx_finish(int ok) {
    SDL_StopTextInput(window);
    pthread_mutex_lock(&g_tx_mu); g_tx.ok = ok; g_tx.done = 1; g_tx.active = 0; pthread_cond_broadcast(&g_tx_cv); pthread_mutex_unlock(&g_tx_mu);
    textEntryOpen = 0; front_mouse_update(); title_main();
}
static void tx_open_main(void) {
    runtime_input_clear();
    textEntryOpen = 1; front_mouse_update();
    SDL_StartTextInput(window);
    tx_title();
    fprintf(stderr, "[totk] keyboard: text entry shown (%s)\n", g_tx.req->prompt ? g_tx.req->prompt : "");
}
static void tx_type(const char *utf8) {
    for (const unsigned char *p = (const unsigned char *)utf8; *p;) {
        int len = *p < 0x80 ? 1 : (*p >> 5) == 6 ? 2 : (*p >> 4) == 14 ? 3 : 4;
        uint32_t c = len == 1 ? *p : 0x100;   /* only ASCII is filtered by character */
        size_t at = strlen(g_tx.text);
        if ((len == 1 && !tx_allowed(c)) || (g_tx.req->max_units > 0 && utf8_units(g_tx.text) + 1 > (size_t)g_tx.req->max_units) || at + len >= sizeof g_tx.text) { p += len; continue; }
        memcpy(g_tx.text + at, p, len); g_tx.text[at + len] = 0; p += len;
    }
    tx_title();
}
static int tx_key(SDL_Scancode s) {   /* 1: the key belongs to the text entry */
    if (s == SDL_SCANCODE_RETURN || s == SDL_SCANCODE_KP_ENTER) {
        if ((int)utf8_units(g_tx.text) < g_tx.req->min_units || !g_tx.text[0]) return 1;
        tx_finish(1); return 1;
    }
    if (s == SDL_SCANCODE_ESCAPE) { tx_finish(0); return 1; }
    if (s == SDL_SCANCODE_BACKSPACE) { size_t n = strlen(g_tx.text); while (n && (g_tx.text[n - 1] & 0xc0) == 0x80) n--; if (n) n--; g_tx.text[n] = 0; tx_title(); return 1; }
    return 1;
}
int runtime_front_text_input(const RuntimeTextRequest *req, char *out, size_t out_size) {
    if (!window || !req || !out || !out_size) { fprintf(stderr, "[totk] keyboard: no game window for text entry; cancelled\n"); return 0; }
    pthread_mutex_lock(&g_tx_mu);
    g_tx.req = req; snprintf(g_tx.text, sizeof g_tx.text, "%s", req->initial ? req->initial : ""); g_tx.done = 0; g_tx.active = 1;
    pthread_mutex_unlock(&g_tx_mu);
    front_sdl_post(tx_open_main);
    pthread_mutex_lock(&g_tx_mu); while (!g_tx.done) pthread_cond_wait(&g_tx_cv, &g_tx_mu); int ok = g_tx.ok; pthread_mutex_unlock(&g_tx_mu);
    if (ok) snprintf(out, out_size, "%s", g_tx.text);
    return ok;
}
static char g_tt_op[16], g_tt_text[256];
static void text_test_main(void) {
    if (!g_tx.active) { fprintf(stderr, "[totk] keyboard test: no text entry is open\n"); return; }
    if (!strcmp(g_tt_op, "type")) tx_type(g_tt_text);
    else if (!strcmp(g_tt_op, "enter")) tx_key(SDL_SCANCODE_RETURN);
    else if (!strcmp(g_tt_op, "esc")) tx_key(SDL_SCANCODE_ESCAPE);
    fprintf(stderr, "[totk] keyboard test: %s -> field \"%s\"\n", g_tt_op, g_tx.text);
}
void runtime_front_text_test(const char *op, const char *text) {
    snprintf(g_tt_op, sizeof g_tt_op, "%s", op); snprintf(g_tt_text, sizeof g_tt_text, "%s", text ? text : "");
    front_sdl_post(text_test_main);
}

/* ---------- VI: one display, one layer, the window as its native window; vsync from a display-rate clock ---------- */
static pthread_mutex_t vsyncLock = PTHREAD_MUTEX_INITIALIZER;
static void *vsyncEvents[16];
static unsigned vsyncCount;
static int viReady;
static unsigned displayIdentity, layerIdentity;
static void *vsync_main(void *arg) {
    (void)arg;
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    for (;;) {
        long period = 1000000000L / front_display_hz();
        t.tv_nsec += period; while (t.tv_nsec >= 1000000000L) { t.tv_nsec -= 1000000000L; t.tv_sec++; }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);
        pthread_mutex_lock(&vsyncLock); for (unsigned i = 0; i < vsyncCount; i++) hle_signal_system_event(vsyncEvents[i]); pthread_mutex_unlock(&vsyncLock);
    }
    return NULL;
}
static void vi_initialize(void) {
    if (!window && !headless()) runtime_fail("VI requires the game window");
    viReady = 1; fprintf(stderr, "[totk] VI display initialized%s\n", window ? "" : " (headless)");
    static int started; if (!started) { started = 1; pthread_t t; pthread_create(&t, NULL, vsync_main, NULL); pthread_detach(t); }
}
static uint32_t vi_open(void **out) { if (!viReady || !out) return 0xf601; *out = &displayIdentity; return 0; }
static uint32_t vi_layer(void **out, void *display) { if (!out || display != &displayIdentity) return 0xf601; *out = &layerIdentity; return 0; }
static uint32_t vi_window(void **out, void *layer) { if (!out || layer != &layerIdentity) return 0xf601; *out = window ? (void *)window : (void *)&layerIdentity; return 0; }
static uint32_t vi_vsync(void *ev, void *display) {
    if (!ev || display != &displayIdentity) return 0xf601; hle_bind_system_event(ev, 1);
    pthread_mutex_lock(&vsyncLock); if (vsyncCount >= 16) runtime_fail("VI display event limit"); vsyncEvents[vsyncCount++] = ev; pthread_mutex_unlock(&vsyncLock); return 0;
}
static uint32_t vi_scaling_mode(void *layer, int mode) { (void)layer; (void)mode; return 0; }
void *runtime_front_override(const char *name) {
    static const struct { const char *name; void *fn; } entries[] = {
        {"_ZN2nn2vi10InitializeEv", vi_initialize}, {"_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE", vi_open},
        {"_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE", vi_layer}, {"_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE", vi_window},
        {"_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE", vi_vsync},
        {"_ZN2nn2vi19SetLayerScalingModeEPNS0_5LayerENS0_11ScalingModeE", vi_scaling_mode},
    };
    for (unsigned i = 0; i < sizeof entries / sizeof *entries; i++) if (!strcmp(name, entries[i].name)) return entries[i].fn;
    return NULL;
}

/* ---------- the window and the event loop ---------- */
static void *guest_main(void *callback) { ((void (*)(void))callback)(); return NULL; }
static void *close_after_save(void *arg) {
    (void)arg; uint64_t t0 = runtime_now_ns();
    while (runtime_fs_save_busy() && runtime_now_ns() - t0 < 15000000000ull) usleep(20000);
    fprintf(stderr, "[totk] save %s after %.2f s; quitting\n", runtime_fs_save_busy() ? "still in progress (gave up)" : "done", (runtime_now_ns() - t0) / 1e9);
    _exit(0);
}
static void request_quit(void) {
    if (!runtime_fs_save_busy()) { fprintf(stderr, "[totk] window closed\n"); _exit(0); }
    fprintf(stderr, "[totk] window closed while the game is saving: quitting when the save is done\n");
    SDL_HideWindow(window);
    pthread_t t; pthread_create(&t, NULL, close_after_save, NULL);
}
static void sync_drawable(void) { int w = 0, h = 0; SDL_GetWindowSizeInPixels(window, &w, &h); if (w > 0 && h > 0) { runtime_aspect_view(w, h); fprintf(stderr, "[totk] window drawable %dx%d\n", w, h); } }
void runtime_front_run(void (*callback)(void)) {
    bootStartNS = runtime_now_ns();
    SDL_SetAppMetadata(GAME_NAME, TOTK_GAME_VERSION, "org.macnativegames.botw");
    SDL_SetHint(SDL_HINT_APP_ID, "botw-native");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD | SDL_INIT_EVENTS | SDL_INIT_AUDIO)) { fprintf(stderr, "[totk] SDL: %s\n", SDL_GetError()); runtime_fail("SDL initialization"); }
    g_post_event = SDL_RegisterEvents(1);
    /* the docked output size, 1920x1080, scaled down (16:9 kept) to the display's usable area; TOTK_WINDOW_SIZE=<w>x<h> */
    float ww = 1920, wh = 1080;
    { const char *e = getenv("TOTK_WINDOW_SIZE"); int w = 0, h = 0; if (e && sscanf(e, "%dx%d", &w, &h) == 2 && w >= 320 && h >= 180 && w <= 8192 && h <= 8192) { ww = (float)w; wh = (float)h; } }
    SDL_Rect usable; if (SDL_GetDisplayUsableBounds(SDL_GetPrimaryDisplay(), &usable) && usable.w > 0 && usable.h > 0) {
        float k = fminf(1, fminf(usable.w / ww, (usable.h - 32) / wh)); ww = floorf(ww * k); wh = floorf(wh * k);
    }
    char title[256]; snprintf(title, sizeof title, "%s  -  Ver. %s", GAME_NAME, TOTK_GAME_VERSION);
    /* opened maximized: a tiling compositor (niri) otherwise gives it a half-width column; TOTK_WINDOW_MAXIMIZED=0 not */
    const char *mx = getenv("TOTK_WINDOW_MAXIMIZED");
    SDL_WindowFlags wf = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | (mx && !strcmp(mx, "0") ? 0 : SDL_WINDOW_MAXIMIZED);
    window = SDL_CreateWindow(title, (int)ww, (int)wh, wf);
    if (!window) { fprintf(stderr, "[totk] window: %s\n", SDL_GetError()); runtime_fail("game window"); }
    if (getenv("TOTK_FULLSCREEN") && atoi(getenv("TOTK_FULLSCREEN"))) SDL_SetWindowFullscreen(window, true);
    fprintf(stderr, "[totk] window %s on %s\n", title, SDL_GetCurrentVideoDriver());
    sync_drawable();
    runtime_mouse_set_hook(front_mouse_update);
    runtime_gamepad_start();
    pthread_t guest; pthread_attr_t attr; pthread_attr_init(&attr); pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
    if (pthread_create(&guest, &attr, guest_main, (void *)callback)) runtime_fail("game thread creation");
    pthread_attr_destroy(&attr);
    for (;;) {
        SDL_Event e;
        if (!SDL_WaitEventTimeout(&e, 100)) continue;
        if (e.type == g_post_event) { ((void (*)(void))e.user.data1)(); continue; }
        switch (e.type) {
        case SDL_EVENT_QUIT: case SDL_EVENT_WINDOW_CLOSE_REQUESTED: request_quit(); break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED: atomic_store(&focused, 1); runtime_oe_push_message(15); front_mouse_update(); fprintf(stderr, "[totk] window gained focus\n"); break;
        case SDL_EVENT_WINDOW_FOCUS_LOST: atomic_store(&focused, 0); front_mouse_update(); runtime_input_clear(); runtime_oe_push_message(15); fprintf(stderr, "[totk] window lost focus\n"); break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: sync_drawable(); break;
        case SDL_EVENT_KEY_DOWN: case SDL_EVENT_KEY_UP: {
            int down = e.type == SDL_EVENT_KEY_DOWN;
            if (g_tx.active) { if (down) tx_key(e.key.scancode); break; }
            if (down && e.key.scancode == SDL_SCANCODE_F11) { SDL_SetWindowFullscreen(window, !(SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN)); break; }
            /* F9: the last seconds of frames, for a flicker report (vk_present.c); F8: the GPU time of each pass of the next
               frame, in the log (vk_exec.c). A Mac keyboard sends its media keys there unless Fn is held (next track,
               play / pause): those count too. F10 is mute, which the compositor takes. */
            if (down && (e.key.scancode == SDL_SCANCODE_F9 || e.key.scancode == SDL_SCANCODE_MEDIA_NEXT_TRACK)) {
                fprintf(stderr, "[totk] F9: saving the last frames\n"); void mtl_frame_ring_save(void); mtl_frame_ring_save(); break; }
            if (down && (e.key.scancode == SDL_SCANCODE_F8 || e.key.scancode == SDL_SCANCODE_MEDIA_PLAY_PAUSE || e.key.scancode == SDL_SCANCODE_MEDIA_PLAY)) {
                fprintf(stderr, "[totk] F8: GPU time of each pass of the next frame\n"); void mtl_passtime_next(void); mtl_passtime_next(); break; }
            /* F7 (previous track without Fn): the frame hold, on -> strict -> off (hle_nvn.c); Fn+F6: the render scale, the
               game's own -> 1 -> 0.85 -> 0.75 -> 0.67 -> 0.5 (game_patches_botw.c). F6 alone is the keyboard backlight. */
            if (down && !e.key.repeat && (e.key.scancode == SDL_SCANCODE_F7 || e.key.scancode == SDL_SCANCODE_MEDIA_PREVIOUS_TRACK)) {
                void nvn_frame_hold_toggle(void); nvn_frame_hold_toggle(); break; }
            if (down && !e.key.repeat && e.key.scancode == SDL_SCANCODE_F6) { int runtime_render_scale_cycle(void); runtime_render_scale_cycle(); break; }
            if (down && e.key.repeat) break;
            int k = mac_key(e.key.scancode); if (k >= 0) runtime_input_key((unsigned)k, down);
            break; }
        case SDL_EVENT_TEXT_INPUT: if (g_tx.active) tx_type(e.text.text); break;
        case SDL_EVENT_MOUSE_MOTION: if (mouseHeld) runtime_mouse_move(e.motion.xrel, e.motion.yrel); break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP: {
            int b = e.button.button == SDL_BUTTON_LEFT ? 0 : e.button.button == SDL_BUTTON_RIGHT ? 1 : e.button.button == SDL_BUTTON_MIDDLE ? 2 :
                    e.button.button == SDL_BUTTON_X1 ? 3 : e.button.button == SDL_BUTTON_X2 ? 4 : -1;
            if (b >= 0 && (mouseHeld || e.type == SDL_EVENT_MOUSE_BUTTON_UP)) runtime_mouse_button(b, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            break; }
        default: gamepad_sdl_event(&e); break;
        }
    }
}
