/* Controllers on Linux through SDL3's gamepad API: the same physical-button bits, layout file (controller_map.h),
   button modes, mouse mode and settings files as runtime/gamepad.m, with rumble and motion from SDL instead of
   CoreHaptics / GameController. Events arrive on the main thread from the window's event loop (linux/frontend_sdl.c). */
#include <SDL3/SDL.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "runtime.h"
#include "gamepad_layout.h"
#include "controller_map.h"
#include "mouse_look.h"
#include "frontend_sdl.h"

void runtime_motion_feed_mouse(const double acc[3], const double gyro[3], uint64_t now);   /* runtime/motion_fusion.c */
void runtime_motion_deadzone_default(double deg);
void runtime_motion_live(int on);
void runtime_motion_mouse(int on);

static pthread_mutex_t gamepadLock = PTHREAD_MUTEX_INITIALIZER;
static RuntimeGamepadState gamepadState;
static uint64_t pressedEdges;
static int started;
static uint16_t controllerMap[CONTROLLER_BUTTONS];
static uint8_t controllerModes[CONTROLLER_BUTTONS];
static uint64_t physicalPrev;
static uint64_t tapUntil[CONTROLLER_BUTTONS];
static int toggleOn[CONTROLLER_BUTTONS], toggleMoved[CONTROLLER_BUTTONS];
#define MAX_PADS 8
static SDL_Gamepad *pads[MAX_PADS];
static SDL_Gamepad *motionPad;
static void refresh_gamepads(void);

static void load_controller_map(void) {
    controller_map_default(controllerMap, controllerModes);
    const char *path = getenv("TOTK_CONTROLS");
    FILE *f = path && *path ? fopen(path, "rb") : NULL;
    if (!f) { fprintf(stderr, "[totk-input] controller layout: default\n"); return; }
    char text[16384]; size_t n = fread(text, 1, sizeof text - 1, f); int whole = feof(f); fclose(f);
    if (!whole) runtime_fail("controller layout file too large");
    text[n] = 0;
    const char *error = NULL; int line = controller_map_parse(text, controllerMap, controllerModes, &error);
    if (line) { fprintf(stderr, "[totk-input] %s line %d: %s\n", path, line, error); runtime_fail("controller layout file"); }
    uint16_t defaults[CONTROLLER_BUTTONS]; uint8_t defaultModes[CONTROLLER_BUTTONS]; controller_map_default(defaults, defaultModes);
    int changed = 0; for (int i = 0; i < CONTROLLER_BUTTONS; i++) changed += controllerMap[i] != defaults[i];
    fprintf(stderr, "[totk-input] controller layout: %s (%d buttons changed)\n", path, changed);
}

static int32_t stick_axis(float value) {
    const float deadzone = 0.12f;
    float magnitude = fabsf(value);
    if (magnitude <= deadzone) return 0;
    if (magnitude > 1.0f) magnitude = 1.0f;
    int32_t result = (int32_t)((magnitude - deadzone) / (1.0f - deadzone) * 32767.0f);
    return value < 0 ? -result : result;
}
static int32_t stronger_axis(int32_t a, int32_t b) { return abs(b) > abs(a) ? b : a; }

/* settings files next to controls.txt (controller-type.txt, mouse.txt), as on the Mac */
static int setting_path(const char *name, char *out, size_t size) {
    const char *controls = getenv("TOTK_CONTROLS");
    if (!controls || !*controls) return 0;
    const char *slash = strrchr(controls, '/');
    int dir = slash ? (int)(slash - controls) : 0;
    int n = slash ? snprintf(out, size, "%.*s/%s", dir, controls, name) : snprintf(out, size, "%s", name);
    return n > 0 && (size_t)n < size;
}
static int setting_read(const char *name, char *word, size_t size, char *path, size_t path_size) {
    if (!setting_path(name, path, path_size)) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(word, 1, size - 1, f); fclose(f); word[n] = 0;
    return 1;
}
static void setting_write(const char *name, const char *word, const char *what) {
    char path[4096], temp[4200];
    if (!setting_path(name, path, sizeof path)) return;
    snprintf(temp, sizeof temp, "%s.tmp", path);
    FILE *f = fopen(temp, "w");
    int ok = f && fprintf(f, "%s\n", word) > 0;
    if (f && fclose(f)) ok = 0;
    if (ok && rename(temp, path)) ok = 0;
    if (!ok) { fprintf(stderr, "[totk-input] could not save the %s to %s; it holds until the game closes\n", what, path); unlink(temp); }
}

static _Atomic int controllerType = GAMEPAD_TYPE_AUTO;
static void load_controller_type(void) {
    char path[4096], word[64] = {0};
    if (!setting_read("controller-type.txt", word, sizeof word, path, sizeof path)) return;
    int type = gamepad_type_parse(word);
    if (type < 0) { fprintf(stderr, "[totk-input] %s: not a controller type (auto, xbox, playstation, switch); using auto\n", path); return; }
    atomic_store(&controllerType, type);
    fprintf(stderr, "[totk-input] controller type: %s (%s)\n", gamepad_type_name(type), path);
}

/* The names GameController would give (productCategory / vendorName), for gamepad_layout.h's rules. */
static const char *pad_category(SDL_Gamepad *g) {
    switch (SDL_GetGamepadType(g)) {
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO: case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT: case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR: return "Nintendo Switch";
    case SDL_GAMEPAD_TYPE_PS3: case SDL_GAMEPAD_TYPE_PS4: return "DualShock";
    case SDL_GAMEPAD_TYPE_PS5: return "DualSense";
    case SDL_GAMEPAD_TYPE_XBOX360: case SDL_GAMEPAD_TYPE_XBOXONE: return "Xbox";
    default: return SDL_GetGamepadName(g) ? SDL_GetGamepadName(g) : "Gamepad";
    }
}
static int face_by_position(SDL_Gamepad *g) {
    const char *name = SDL_GetGamepadName(g);
    return gamepad_type_face_by_position(atomic_load(&controllerType), pad_category(g), name ? name : "");
}
/* GameController's buttonA / B / X / Y: the buttons labelled A, B, X, Y (PlayStation: Cross, Circle, Square, Triangle),
   wherever they sit, which SDL reports per position. */
static int gc_button(SDL_Gamepad *g, int which) {   /* 0 A, 1 B, 2 X, 3 Y */
    static const SDL_GamepadButtonLabel want[4][2] = {
        {SDL_GAMEPAD_BUTTON_LABEL_A, SDL_GAMEPAD_BUTTON_LABEL_CROSS}, {SDL_GAMEPAD_BUTTON_LABEL_B, SDL_GAMEPAD_BUTTON_LABEL_CIRCLE},
        {SDL_GAMEPAD_BUTTON_LABEL_X, SDL_GAMEPAD_BUTTON_LABEL_SQUARE}, {SDL_GAMEPAD_BUTTON_LABEL_Y, SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE}};
    static const SDL_GamepadButton fallback[4] = {SDL_GAMEPAD_BUTTON_SOUTH, SDL_GAMEPAD_BUTTON_EAST, SDL_GAMEPAD_BUTTON_WEST, SDL_GAMEPAD_BUTTON_NORTH};
    static const SDL_GamepadButton faces[4] = {SDL_GAMEPAD_BUTTON_SOUTH, SDL_GAMEPAD_BUTTON_EAST, SDL_GAMEPAD_BUTTON_WEST, SDL_GAMEPAD_BUTTON_NORTH};
    for (int i = 0; i < 4; i++) {
        SDL_GamepadButtonLabel l = SDL_GetGamepadButtonLabel(g, faces[i]);
        if (l == want[which][0] || l == want[which][1]) return SDL_GetGamepadButton(g, faces[i]);
    }
    return SDL_GetGamepadButton(g, fallback[which]);
}
/* physical bits as runtime/gamepad.m gamepad_buttons() makes them (controller_physical_names) */
static uint64_t gamepad_buttons(SDL_Gamepad *g, int by_position) {
    uint64_t b = gamepad_face_positions(gc_button(g, 0), gc_button(g, 1), gc_button(g, 2), gc_button(g, 3), by_position);
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_LEFT_STICK)) b |= 1ULL << 4;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_RIGHT_STICK)) b |= 1ULL << 5;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER)) b |= 1ULL << 6;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER)) b |= 1ULL << 7;
    if (SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 16384) b |= 1ULL << 8;
    if (SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16384) b |= 1ULL << 9;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_START)) b |= 1ULL << 10;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_BACK)) b |= 1ULL << 11;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_DPAD_LEFT)) b |= 1ULL << 12;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_DPAD_UP)) b |= 1ULL << 13;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_DPAD_RIGHT)) b |= 1ULL << 14;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_DPAD_DOWN)) b |= 1ULL << 15;
    if (SDL_GetGamepadButton(g, SDL_GAMEPAD_BUTTON_TOUCHPAD)) b |= 1ULL << 16;
    return b;
}

/* ---- mouse mode (key 7): as runtime/gamepad.m ---- */
static _Atomic int mouseMode;
static unsigned mouseButtons;
static MouseLook mouseLook;
static MouseLookSettings mouseSettings;
static int32_t mouseRX, mouseRY;
static int mouseGyro;
static SDL_TimerID mouseTimer;
static void (*mouseHook)(void);
static Uint32 mouse_tick(void *ud, SDL_TimerID id, Uint32 interval) {
    (void)ud; (void)id;
    double gyro[3], acc[3]; int32_t rx, ry;
    pthread_mutex_lock(&gamepadLock);
    mouse_look_tick(&mouseLook, &mouseSettings, &rx, &ry, gyro, acc);
    mouseRX = rx; mouseRY = ry;
    int feed = mouseGyro;
    pthread_mutex_unlock(&gamepadLock);
    if (feed) runtime_motion_feed_mouse(acc, gyro, runtime_now_ns());
    return interval;
}
static void mouse_apply(int on) {
    if (on) {
        const char *g = getenv("TOTK_MOUSE_GYRO");
        MouseLookSettings settings = mouse_look_settings(getenv("TOTK_MOUSE_SENSITIVITY"), getenv("TOTK_MOUSE_INVERT_Y"),
                                                         getenv("TOTK_MOUSE_COUNTERWEIGHT"), getenv("TOTK_MOUSE_DECAY"), getenv("TOTK_MOUSE_MIN_DECAY"));
        int gyro = runtime_gyro_enabled() && !(g && !strcmp(g, "0"));
        pthread_mutex_lock(&gamepadLock);
        mouseSettings = settings; mouse_look_reset(&mouseLook); mouseRX = mouseRY = 0; mouseGyro = gyro;
        pthread_mutex_unlock(&gamepadLock);
        runtime_motion_mouse(gyro);
        fprintf(stderr, "[totk-input] mouse mode on: sensitivity %d%s, gyro %s\n", settings.sensitivity, settings.invert_y ? " (Y inverted)" : "",
                gyro ? "from the mouse" : "not from the mouse");
        if (!mouseTimer) mouseTimer = SDL_AddTimer(MOUSE_LOOK_TICK_NS / 1000000, mouse_tick, NULL);
    } else {
        if (mouseTimer) { SDL_RemoveTimer(mouseTimer); mouseTimer = 0; }
        pthread_mutex_lock(&gamepadLock);
        mouse_look_reset(&mouseLook); mouseRX = mouseRY = 0; mouseGyro = 0;
        pthread_mutex_unlock(&gamepadLock);
        runtime_motion_mouse(0);
        fprintf(stderr, "[totk-input] mouse mode off\n");
    }
    if (mouseHook) front_sdl_post(mouseHook);
}
static void load_mouse_mode(void) {
    char path[4096], word[16] = {0};
    if (!setting_read("mouse.txt", word, sizeof word, path, sizeof path)) return;
    int on = !strcmp(word, "on\n") || !strcmp(word, "on");
    if (on != atomic_load(&mouseMode)) { atomic_store(&mouseMode, on); mouse_apply(on); }
}
int runtime_mouse_enabled(void) { return atomic_load(&mouseMode); }
void runtime_mouse_set_hook(void (*hook)(void)) { mouseHook = hook; }
static void mouse_release_main(void) { runtime_mouse_release(); }
int runtime_mouse_toggle(void) {
    int on = !atomic_load(&mouseMode);
    atomic_store(&mouseMode, on);
    setting_write("mouse.txt", on ? "on" : "off", "mouse mode");
    if (!on) front_sdl_post(mouse_release_main);
    mouse_apply(on);
    return on;
}
void runtime_mouse_move(double dx, double dy) {
    if (!atomic_load(&mouseMode)) return;
    pthread_mutex_lock(&gamepadLock); mouse_look_move(&mouseLook, dx, dy, &mouseSettings); pthread_mutex_unlock(&gamepadLock);
}
void runtime_mouse_button(int button, int down) {
    if (button < 0 || button >= CONTROLLER_MOUSE_BUTTONS || (down && !atomic_load(&mouseMode))) return;
    unsigned bit = 1u << button, next = down ? mouseButtons | bit : mouseButtons & ~bit;
    if (next == mouseButtons) return;
    mouseButtons = next; refresh_gamepads();
}
void runtime_mouse_release(void) {
    pthread_mutex_lock(&gamepadLock); mouse_look_reset(&mouseLook); mouseRX = mouseRY = 0; pthread_mutex_unlock(&gamepadLock);
    if (mouseButtons) { mouseButtons = 0; refresh_gamepads(); }
}

static void refresh_main(void) { refresh_gamepads(); }
static Uint32 tap_timer(void *ud, SDL_TimerID id, Uint32 interval) { (void)ud; (void)id; (void)interval; front_sdl_post(refresh_main); return 0; }

static void refresh_gamepads(void) {   /* main thread */
    RuntimeGamepadState next = {0};
    uint64_t physical = (uint64_t)mouseButtons << CONTROLLER_MOUSE_FIRST;
    for (int i = 0; i < MAX_PADS; i++) {
        SDL_Gamepad *g = pads[i]; if (!g) continue;
        next.connected = 1;
        physical |= gamepad_buttons(g, face_by_position(g));
        next.lx = stronger_axis(next.lx, stick_axis(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX) / 32767.0f));
        next.ly = stronger_axis(next.ly, stick_axis(-SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY) / 32767.0f));   /* SDL: down positive */
        next.rx = stronger_axis(next.rx, stick_axis(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX) / 32767.0f));
        next.ry = stronger_axis(next.ry, stick_axis(-SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY) / 32767.0f));
    }
    uint64_t now = runtime_now_ns(), edges = physical & ~physicalPrev, synthetic = 0, suppressed = 0;
    int moving = next.lx || next.ly;
    physicalPrev = physical;
    for (int i = 0; i < CONTROLLER_BUTTONS; i++) {
        uint64_t bit = 1ULL << i;
        int kind = controller_mode_kind(controllerModes[i]);
        if (kind == CONTROLLER_TAP && (edges & bit)) { tapUntil[i] = now + 400000000ULL; SDL_AddTimer(401, tap_timer, NULL); }
        if (kind == CONTROLLER_TOGGLE) {
            if (edges & bit) { toggleOn[i] = !toggleOn[i]; toggleMoved[i] = moving; }
            if (toggleOn[i] && moving) toggleMoved[i] = 1;
            if (toggleOn[i] && ((toggleMoved[i] && !moving) || !next.connected)) toggleOn[i] = 0;
        }
        if (kind == CONTROLLER_AIM && (physical & bit)) {
            uint64_t fire = 1ULL << controller_mode_fire(controllerModes[i]);
            suppressed |= fire;
            if (!(physical & fire)) synthetic |= controllerMap[i];
        }
        int active = kind == CONTROLLER_TAP ? now < tapUntil[i] : kind == CONTROLLER_TOGGLE ? toggleOn[i] : 0;
        if (active) synthetic |= controllerMap[i];
    }
    uint64_t held = physical & ~suppressed;
    for (int i = 0; i < CONTROLLER_BUTTONS; i++) if (controllerModes[i] != CONTROLLER_HOLD) held &= ~(1ULL << i);
    next.buttons |= controller_map_apply(controllerMap, held) | synthetic;
    pthread_mutex_lock(&gamepadLock);
    pressedEdges |= next.buttons & ~gamepadState.buttons;
    gamepadState = next;
    pthread_mutex_unlock(&gamepadLock);
}

/* ---- rumble and motion (opt-in, TOTK_RUMBLE=1 / TOTK_GYRO=1, as on the Mac) ---- */
static float rumbleAmp[2][2];   /* [side][low, high] */
void runtime_rumble_set(unsigned side, float amp_low, float freq_low, float amp_high, float freq_high) {
    (void)freq_low; (void)freq_high;
    if (side > 1) return;
    rumbleAmp[side][0] = amp_low; rumbleAmp[side][1] = amp_high;
    float lo = fmaxf(rumbleAmp[0][0], rumbleAmp[1][0]), hi = fmaxf(rumbleAmp[0][1], rumbleAmp[1][1]);
    if (lo < 0) lo = 0; if (lo > 1) lo = 1; if (hi < 0) hi = 0; if (hi > 1) hi = 1;
    for (int i = 0; i < MAX_PADS; i++) if (pads[i]) SDL_RumbleGamepad(pads[i], (Uint16)(lo * 65535), (Uint16)(hi * 65535), lo > 0 || hi > 0 ? 200 : 0);
}
static void motion_attach(SDL_Gamepad *g) {
    if (!runtime_gyro_enabled() || motionPad) return;
    if (!SDL_GamepadHasSensor(g, SDL_SENSOR_GYRO) || !SDL_GamepadHasSensor(g, SDL_SENSOR_ACCEL)) return;
    if (!SDL_SetGamepadSensorEnabled(g, SDL_SENSOR_GYRO, true) || !SDL_SetGamepadSensorEnabled(g, SDL_SENSOR_ACCEL, true)) return;
    const char *axes = getenv("TOTK_GYRO_AXES"); int r = runtime_motion_axes(axes);
    if (r < 0) fprintf(stderr, "[totk-input] TOTK_GYRO_AXES=%s is not like \"-x,y,-z\": ignored\n", axes);
    runtime_motion_deadzone_default(1.2); runtime_motion_live(1);
    motionPad = g;
    fprintf(stderr, "[totk-input] motion from %s (gyro + accelerometer)\n", SDL_GetGamepadName(g));
}
static float lastAccel[3];
static void motion_event(const SDL_GamepadSensorEvent *e) {
    if (!motionPad || SDL_GetGamepadID(motionPad) != e->which) return;
    if (e->sensor == SDL_SENSOR_ACCEL) { for (int i = 0; i < 3; i++) lastAccel[i] = e->data[i]; return; }
    if (e->sensor != SDL_SENSOR_GYRO) return;
    /* SDL: x right, y up, z towards the player (the frame motion_fusion.c reads), accel in m/s^2, gyro in rad/s */
    double acc[3] = {lastAccel[0] / SDL_STANDARD_GRAVITY, lastAccel[1] / SDL_STANDARD_GRAVITY, lastAccel[2] / SDL_STANDARD_GRAVITY};
    double gyro[3] = {e->data[0], e->data[1], e->data[2]};
    runtime_motion_feed(acc, gyro, runtime_now_ns());
}

static void attach(SDL_JoystickID id) {
    SDL_Gamepad *g = SDL_OpenGamepad(id); if (!g) return;
    int slot = -1; for (int i = 0; i < MAX_PADS; i++) if (!pads[i]) { slot = i; break; }
    if (slot < 0) { SDL_CloseGamepad(g); return; }
    pads[slot] = g;
    fprintf(stderr, "[totk-input] gamepad connected: %s (%s), face buttons by %s\n", SDL_GetGamepadName(g) ? SDL_GetGamepadName(g) : "unknown",
            pad_category(g), face_by_position(g) ? "position" : "label");
    motion_attach(g);
    refresh_gamepads();
}
static void detach(SDL_JoystickID id) {
    for (int i = 0; i < MAX_PADS; i++) if (pads[i] && SDL_GetGamepadID(pads[i]) == id) {
        fprintf(stderr, "[totk-input] gamepad disconnected: %s\n", SDL_GetGamepadName(pads[i]) ? SDL_GetGamepadName(pads[i]) : "unknown");
        if (motionPad == pads[i]) { motionPad = NULL; runtime_motion_live(0); }
        SDL_CloseGamepad(pads[i]); pads[i] = NULL;
    }
    refresh_gamepads();
}

void gamepad_sdl_event(const SDL_Event *e) {   /* main thread (frontend_sdl.c event loop) */
    switch (e->type) {
    case SDL_EVENT_GAMEPAD_ADDED: attach(e->gdevice.which); break;
    case SDL_EVENT_GAMEPAD_REMOVED: detach(e->gdevice.which); break;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: case SDL_EVENT_GAMEPAD_BUTTON_DOWN: case SDL_EVENT_GAMEPAD_BUTTON_UP: refresh_gamepads(); break;
    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE: motion_event(&e->gsensor); break;
    default: break;
    }
}

int runtime_gamepad_type_cycle(void) {
    int type = gamepad_type_next(atomic_load(&controllerType));
    atomic_store(&controllerType, type);
    setting_write("controller-type.txt", gamepad_type_name(type), "controller type");
    fprintf(stderr, "[totk-input] controller type: %s (key 6)\n", gamepad_type_name(type));
    front_sdl_post(refresh_main);
    return type;
}
int runtime_gamepad_type(int *detected) {
    int found = -1;
    for (int i = 0; i < MAX_PADS && found < 0; i++) if (pads[i]) found = gamepad_type_detect(pad_category(pads[i]), SDL_GetGamepadName(pads[i]));
    if (detected) *detected = found;
    return atomic_load(&controllerType);
}
void runtime_gamepad_start(void) {   /* main thread, after SDL_Init(SDL_INIT_GAMEPAD) */
    if (started) return;
    started = 1;
    load_controller_map(); load_controller_type(); load_mouse_mode();
    int n = 0; SDL_JoystickID *ids = SDL_GetGamepads(&n);
    for (int i = 0; ids && i < n; i++) attach(ids[i]);
    SDL_free(ids);
}
void runtime_gamepad_clear(void) {
    physicalPrev = 0;
    memset(tapUntil, 0, sizeof tapUntil); memset(toggleOn, 0, sizeof toggleOn); memset(toggleMoved, 0, sizeof toggleMoved);
    pthread_mutex_lock(&gamepadLock);
    gamepadState = (RuntimeGamepadState){0}; pressedEdges = 0;
    mouse_look_reset(&mouseLook); mouseRX = mouseRY = 0;
    pthread_mutex_unlock(&gamepadLock);
}
void runtime_touch_pad_set(const RuntimeGamepadState *s) { (void)s; }
void runtime_gamepad_read(RuntimeGamepadState *out) {
    pthread_mutex_lock(&gamepadLock);
    *out = gamepadState;
    out->buttons |= pressedEdges;
    out->rx = stronger_axis(out->rx, mouseRX); out->ry = stronger_axis(out->ry, mouseRY);
    pressedEdges = 0;
    pthread_mutex_unlock(&gamepadLock);
}
