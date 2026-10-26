/* Shaders on Vulkan. The translation chain is the Mac's (tools/totk_xlate: the game's Maxwell code -> SPIR-V through the
   vendored Ryujinx translator, then SPIRV-Cross MSL and the reflection file built from it): the Vulkan backend uses the
   SPIR-V it produces directly and keeps the MSL only as the reflection's source. Variants, their names, their deps and
   the cache files are those of totk_shader_rt.m, so a cache or seed made by the Mac's tools serves here too.
   The SPIR-V puts every resource in set 0 with its own binding; at load the decorations are rewritten to one universal
   layout per stage (vk_priv.h VKB_*), set 0 for the vertex and compute stages and set 1 for the fragment stage, so any
   vertex shader object can be drawn with any fragment shader object. */

#include "vk_priv.h"
#include "vk_json.h"
#include "nvn_hmap.h"
#include "dmsh_chain.h"
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char **environ;
int totk_depth_near_minus_w(void);   /* hle_nvn.c */
int vk_depth_near_minus_w(void) { return totk_depth_near_minus_w(); }

/* ---------------- paths ---------------- */
static char g_exe_dir[4096], g_cache[4096];
static void paths_init(void) {
    if (g_exe_dir[0]) return;
    char p[4096]; ssize_t n = readlink("/proc/self/exe", p, sizeof p - 1);
    if (n <= 0) VDIE("/proc/self/exe");
    p[n] = 0; snprintf(g_exe_dir, sizeof g_exe_dir, "%s", dirname(p));
    const char *e = getenv("TOTK_SHADER_CACHE");
    if (e && *e) snprintf(g_cache, sizeof g_cache, "%s", e);
    else snprintf(g_cache, sizeof g_cache, "%s/shader-cache/rt-%s", g_exe_dir, DMSH_CHAIN);
    char mk[4200]; snprintf(mk, sizeof mk, "%s", g_cache);
    for (char *s = mk + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(mk, 0755); *s = '/'; }
    mkdir(mk, 0755);
    VLOG("shader cache %s", g_cache);
}
static void cache_path(char *out, size_t n, const char *name, const char *suffix) { snprintf(out, n, "%s/%s%s", g_cache, name, suffix); }
static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)n, f); fclose(f); b[got] = 0;
    if (len) *len = got;
    return b;
}
static int write_file_atomic(const char *path, const void *data, size_t n) {
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp%d", path, (int)getpid());
    FILE *f = fopen(tmp, "wb"); if (!f) return 0;
    int ok = fwrite(data, 1, n, f) == n; ok &= fclose(f) == 0;
    if (ok) ok = rename(tmp, path) == 0; else unlink(tmp);
    return ok;
}
static void hex(const uint8_t *p, size_t n, char *out) { for (size_t i = 0; i < n; i++) snprintf(out + 2 * i, 3, "%02x", p[i]); }

/* ---------------- the translation service ---------------- */
typedef struct { FILE *in, *out; pid_t pid; } svc_t;
static svc_t g_svc;
static void service_start(svc_t *s) {
    if (s->in) return;
    paths_init();
    char xl[4200], dll[4200], sc[4200], dotnet[4200];
    snprintf(xl, sizeof xl, "%s/totk_xlate", g_exe_dir); snprintf(dll, sizeof dll, "%s/totk_xlate.dll", g_exe_dir);
    snprintf(sc, sizeof sc, "%s/spirv-cross", g_exe_dir);
    const char *dn = getenv("TOTK_DOTNET"); snprintf(dotnet, sizeof dotnet, "%s", dn && *dn ? dn : "dotnet");
    int to[2], from[2];
    if (pipe(to) || pipe(from)) VDIE("shader service pipes");
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, to[0], 0); posix_spawn_file_actions_adddup2(&fa, from[1], 1);
    posix_spawn_file_actions_addclose(&fa, to[1]); posix_spawn_file_actions_addclose(&fa, from[0]);
    posix_spawnattr_t sa; posix_spawnattr_init(&sa); sigset_t none; sigemptyset(&none); posix_spawnattr_setsigmask(&sa, &none);
    posix_spawnattr_setflags(&sa, POSIX_SPAWN_SETSIGMASK);
    int native = access(xl, X_OK) == 0;
    if (!native && access(dll, R_OK)) VDIE("no shader translator next to the executable (%s or %s)", xl, dll);
    if (access(sc, X_OK)) VDIE("no spirv-cross next to the executable (%s)", sc);
    const char *argv_native[] = {xl, "--service", "--spirv-cross", sc, "--out", g_cache, "--chain", DMSH_CHAIN, NULL};
    const char *argv_dll[] = {dotnet, dll, "--service", "--spirv-cross", sc, "--out", g_cache, "--chain", DMSH_CHAIN, NULL};
    int r = native ? posix_spawn(&s->pid, xl, &fa, &sa, (char *const *)argv_native, environ) : posix_spawnp(&s->pid, dotnet, &fa, &sa, (char *const *)argv_dll, environ);
    posix_spawn_file_actions_destroy(&fa); posix_spawnattr_destroy(&sa);
    if (r) VDIE("start shader translator: %s", strerror(r));
    close(to[0]); close(from[1]);
    fcntl(to[1], F_SETFD, FD_CLOEXEC); fcntl(from[0], F_SETFD, FD_CLOEXEC);
    s->in = fdopen(to[1], "w"); s->out = fdopen(from[0], "r");
    if (!s->in || !s->out) VDIE("shader service streams");
    VLOG("shader translation service started (pid %d, %s)", (int)s->pid, native ? "native" : "dotnet");
}
static jv *translate(svc_t *s, jv *rq) {
    service_start(s);
    char *line = jv_dump(rq);
    fprintf(s->in, "%s\n", line); fflush(s->in);
    free(line);
    char buf[8192];
    if (!fgets(buf, sizeof buf, s->out)) VDIE("shader translation service exited while translating %s", jv_str(jv_get(rq, "variant")));
    jv *reply = jv_parse(buf, strlen(buf), NULL, 0);
    if (!reply || !jv_truthy(jv_get(reply, "ok")))
        VDIE("shader %s (%s) translation failed: %s", jv_str(jv_get(rq, "variant")), jv_str(jv_get(rq, "stage")), reply ? jv_dump(jv_get(reply, "error")) : buf);
    jv_free(reply);
    char p[4200]; cache_path(p, sizeof p, jv_str(jv_get(rq, "variant")), "_b.json");
    jv *refl = jv_parse_file(p);
    if (!refl) VDIE("shader %s: reflection missing after translation", jv_str(jv_get(rq, "variant")));
    return refl;
}

/* ---------------- variant indexes (totk_shader_rt.m) ---------------- */
typedef struct { char shader[65]; jv *rows; } vindex;
static nvn_hmap g_vidx = {.klen = 64};
static pthread_mutex_t g_rt_mu = PTHREAD_MUTEX_INITIALIZER;
static jv *variants_of(const char *shader) {
    vindex *v = nvn_hm_get(&g_vidx, shader);
    if (v) return v->rows;
    v = calloc(1, sizeof *v); if (!v) VDIE("variant index");
    memcpy(v->shader, shader, 64);
    char p[4200]; cache_path(p, sizeof p, shader, ".variants");
    v->rows = jv_parse_file(p);
    if (!v->rows || v->rows->k != JV_ARR) { jv_free(v->rows); v->rows = jv_new_arr(); }
    nvn_hm_put(&g_vidx, v->shader, v);
    return v->rows;
}
static int deps_match(const jv *deps, const jv *ans) {
    static const char *kinds[] = {"tex", "img", "attr", "fout"};
    for (int k = 0; k < 4; k++) {
        const jv *want = jv_get(deps, kinds[k]), *have = jv_get(ans, kinds[k]);
        if (!want || want->k != JV_OBJ) continue;
        for (int i = 0; i < want->len; i++) {
            const char *slot = want->keys[i]; const jv *w = want->items[i], *h = jv_get(have, slot);
            if (!h && k == 2) continue;   /* an attribute the answers leave out is unbound: any variant serves */
            jv *dflt = NULL;
            if (!h) { dflt = k == 1 ? jv_new_obj() : jv_new_str(k == 0 ? "2d" : "float"); h = dflt; }
            const char *ws = jv_str(w);
            int ok;
            if (ws && (!strcmp(ws, "*") || !strcmp(ws, "*_depth"))) {
                const char *hs = jv_str(h); size_t hl = hs ? strlen(hs) : 0;
                ok = hs && ((hl > 6 && !strcmp(hs + hl - 6, "_depth")) == !strcmp(ws, "*_depth"));
            } else ok = jv_equal(h, w);
            jv_free(dflt);
            if (!ok) return 0;
        }
    }
    return 1;
}
static const char *variant_lookup(const char *shader, const jv *answers) {
    jv *vs = variants_of(shader);
    for (int i = 0; i < vs->len; i++) {
        const jv *row = vs->items[i];
        if (!deps_match(jv_get(row, "deps"), answers)) continue;
        const char *name = jv_str(jv_get(row, "variant"));
        if (!name) continue;
        char p[4200]; cache_path(p, sizeof p, name, ".spv");
        if (access(p, R_OK)) continue;   /* translated on a machine that kept only the MSL: translate again */
        return name;
    }
    return NULL;
}
static const char *stage_name(int st) { return st == 0 ? "vs" : st == 1 ? "ps" : st == 5 ? "cs" : NULL; }
/* "<shader>-<12 hex>": the answers as NSJSONSerialization writes them with sorted keys, hashed */
static void variant_name(const char *shader, const jv *answers, char out[80]) {
    char *a = jv_dump_sorted(answers); uint8_t md[32]; CC_SHA256(a, (CC_LONG)strlen(a), md); free(a);
    char h[13]; hex(md, 6, h); snprintf(out, 80, "%s-%s", shader, h);
}
/* The variant name for these answers, translating when the cache has none. */
static const char *rt_variant(const char *shader, int stage, const uint8_t *control, const uint8_t *code, const jv *answers, const uint32_t *local, char out[80]) {
    pthread_mutex_lock(&g_rt_mu);
    const char *found = variant_lookup(shader, answers);
    if (found) { snprintf(out, 80, "%s", found); pthread_mutex_unlock(&g_rt_mu); return out; }
    uint32_t c1o, c1t; memcpy(&c1o, control + 0x700, 4); memcpy(&c1t, control + 0x704, 4);
    if (!c1t || c1t > 16u << 20 || c1o > c1t || (stage != 5 && c1o < 0x30)) VDIE("shader %s: code area %#x / c1 %#x", shader, c1t, c1o);
    if (!stage_name(stage)) VDIE("shader %s: NVN stage %d has no translation chain", shader, stage);
    variant_name(shader, answers, out);
    char codep[4200]; cache_path(codep, sizeof codep, out, ".code");
    size_t skip = stage == 5 ? 0 : 0x30;
    if (!write_file_atomic(codep, code + skip, c1t - skip)) VDIE("write shader code %s", codep);
    jv *rq = jv_new_obj();
    jv_set(rq, "variant", jv_new_str(out)); jv_set(rq, "stage", jv_new_str(stage_name(stage))); jv_set(rq, "code", jv_new_str(codep));
    char *c1 = malloc((size_t)(c1t - c1o) * 2 + 1); if (!c1) VDIE("c1 hex"); c1[0] = 0; hex(code + c1o, c1t - c1o, c1); c1[(size_t)(c1t - c1o) * 2] = 0;
    jv_set(rq, "c1", jv_new_str(c1)); free(c1);
    for (int i = 0; i < answers->len; i++) jv_set(rq, answers->keys[i], jv_clone(answers->items[i]));
    if (local) { jv *l = jv_new_arr(); for (int i = 0; i < 3; i++) jv_push(l, jv_new_num(local[i])); jv_set(rq, "local", l); }
    jv_set(rq, "depth_minus_w", jv_new_bool(vk_depth_near_minus_w()));
    uint64_t t0 = vk_now_ns();
    jv *refl = translate(&g_svc, rq);
    jv_free(rq);
    jv *vs = variants_of(shader);
    jv *row = jv_new_obj(); jv_set(row, "variant", jv_new_str(out)); jv_set(row, "deps", jv_get(refl, "deps") ? jv_clone(jv_get(refl, "deps")) : jv_new_obj());
    jv_push(vs, row);
    char *idx = jv_dump_sorted(vs); char ip[4200]; cache_path(ip, sizeof ip, shader, ".variants");
    write_file_atomic(ip, idx, strlen(idx)); free(idx);
    jv_free(refl);
    static uint64_t n; n++;
    if (n <= 8 || n % 100 == 0) VLOG("translated %llu: %s %s in %.1f ms", (unsigned long long)n, stage_name(stage), out, (vk_now_ns() - t0) / 1e6);
    pthread_mutex_unlock(&g_rt_mu);
    return out;
}

/* ---------------- reflection (nvn_refl.m, refl_mark_stores, drv_scan, c1_reads_bounded of nvn_mtl.m) ---------------- */
static int msl_stores_through(const char *s, const char *name) {
    if (!s || !name) return 0;
    char pat[200]; snprintf(pat, sizeof pat, "&(*spvDescriptorSet0.%s)", name);
    if (strstr(s, pat)) return 1;
    snprintf(pat, sizeof pat, "(*spvDescriptorSet0.%s)._m0[", name);
    size_t pl = strlen(pat);
    for (const char *p = strstr(s, pat); p; p = strstr(p, pat)) {
        p += pl; int depth = 1;
        while (*p && depth) { if (*p == '[') depth++; else if (*p == ']') depth--; p++; }
        if (*p == '.') { p++; while (*p == 'x' || *p == 'y' || *p == 'z' || *p == 'w') p++; }
        while (*p == ' ') p++;
        if (p[0] == '=' && p[1] != '=') return 1;
    }
    return 0;
}
static void drv_scan(nvn_refl *rf, const char *s) {
    if (!s) return;
    for (const char *p = strstr(s, "p_c"); p; p = strstr(p + 3, "p_c")) {
        if (p - s < 2 || p[-2] != '.' || (p[-1] != 'v' && p[-1] != 'f' && p[-1] != 'c')) continue;
        int bank = p[3] - '0';
        if ((bank != 0 && bank != 2) || strncmp(p + 4, ")._m0[", 6)) continue;
        rf->drv_read[bank] = 1;
    }
}
static int c1_reads_bounded(const char *src, size_t len) {
    if (!src) return 0;
    const size_t limit = (((len + 255u) & ~(size_t)255u) + 256u) / 16u;
    for (const char *p = src; (p = strstr(p, "_c1)._m0[")); ) {
        p += 9; char *end; unsigned long idx = strtoul(p, &end, 10);
        if (end == p || *p < '0' || *p > '9') return 0;
        if (*end == 'u') end++;
        if (*end != ']' || idx >= limit) return 0;
        p = end;
    }
    return 1;
}
static int refl_from_json(const jv *j, nvn_refl *o, char *err, size_t errlen) {
    memset(o, 0, sizeof *o);
    #define FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); return 1; } while (0)
    if (!jv_str(jv_get(j, "route")) || strcmp(jv_str(jv_get(j, "route")), "B")) FAIL("not a route B reflection");
    const jv *bs = jv_get(j, "bindings");
    if (!bs || bs->k != JV_ARR) FAIL("no bindings");
    if (bs->len > NVN_REFL_MAXB) FAIL("%d bindings, more than %d", bs->len, NVN_REFL_MAXB);
    o->route_b = 1;
    for (int i = 0; i < bs->len; i++) {
        const jv *b = bs->items[i]; const char *k = jv_str(jv_get(b, "kind")); nvn_rbind *r = &o->b[o->nb];
        const jv *rg = jv_get(b, "reg"); int reg = rg && rg->k == JV_NUM ? (int)rg->n : -1;
        r->aid = (uint16_t)jv_int(jv_get(b, "arg_id"), 0); r->reg = reg < 0 ? 0 : (uint16_t)reg;
        const char *ty = jv_str(jv_get(b, "msl_type"));
        int rw = jv_truthy(jv_get(b, "rw")) || (ty && (strstr(ty, "access::write") || strstr(ty, "access::read_write")));
        if (!k) FAIL("binding without kind");
        if (!strcmp(k, "cbuf")) r->kind = NVN_RB_CBUF;
        else if (!strcmp(k, "samp")) r->kind = NVN_RB_SAMP;
        else if (!strcmp(k, "samp_tex")) r->kind = NVN_RB_SAMPT;
        else if (!strcmp(k, "tex")) r->kind = NVN_RB_TEX;
        else if (!strcmp(k, "img")) { r->kind = NVN_RB_IMG; r->rw = (uint8_t)rw; }
        else if (!strcmp(k, "tbuf")) r->kind = NVN_RB_TBUF;
        else if (!strcmp(k, "sbuf")) { r->kind = NVN_RB_SBUFP; r->rw = (uint8_t)rw; }
        else if (!strcmp(k, "ubo_ptr")) r->kind = NVN_RB_UBOP;
        else if (!strcmp(k, "sbuf_ubo")) { r->kind = NVN_RB_SBUFU; r->rw = (uint8_t)rw; r->off = (uint16_t)jv_int(jv_get(b, "off"), 0); }
        else if (!strcmp(k, "support")) r->kind = NVN_RB_SUPPORT;
        else if (!strcmp(k, "c1")) r->kind = NVN_RB_C1;
        else if (!strcmp(k, "drvcb")) r->kind = jv_int(jv_get(b, "bank"), 0) == 2 ? NVN_RB_DRV2 : NVN_RB_DRV0;
        else FAIL("binding kind %s not implemented (%s)", k, jv_str(jv_get(b, "msl_name")));
        if ((r->kind == NVN_RB_CBUF || r->kind == NVN_RB_SAMP || r->kind == NVN_RB_SAMPT || r->kind == NVN_RB_TEX || r->kind == NVN_RB_IMG || r->kind == NVN_RB_TBUF ||
             r->kind == NVN_RB_SBUFP || r->kind == NVN_RB_UBOP || r->kind == NVN_RB_SBUFU) && reg < 0)
            FAIL("binding %s has no NVN slot", jv_str(jv_get(b, "msl_name")));
        o->nb++;
    }
    const jv *dr = jv_get(j, "drv_reads");
    if (dr && dr->k == JV_OBJ) { o->drv_read[0] = jv_get(dr, "0") != NULL; o->drv_read[2] = jv_get(dr, "2") != NULL; }
    const jv *uf = jv_get(j, "drv_unfillable");
    if (uf && uf->k == JV_ARR && uf->len) FAIL("reads driver constants the replay does not fill (%s)", jv_dump(uf));
    const char *c1 = jv_str(jv_get(j, "c1"));
    if (c1) {
        size_t n = strlen(c1);
        if (n % 2 || n / 2 > sizeof o->c1) FAIL("c1 of %zu bytes", n / 2);
        if (n) { o->has_c1 = 1; o->c1_len = (uint16_t)(n / 2); for (unsigned i = 0; i < o->c1_len; i++) { char h[3] = {c1[2 * i], c1[2 * i + 1], 0}; o->c1[i] = (uint8_t)strtoul(h, NULL, 16); } }
    }
    const jv *vi = jv_get(j, "vertex_inputs");
    for (int i = 0; vi && vi->k == JV_ARR && i < vi->len; i++) {
        int loc = jv_int(jv_get(vi->items[i], "location"), -1); const char *t = jv_str(jv_get(vi->items[i], "msl_type"));
        if (loc < 0 || loc >= NVN_REFL_MAXV || o->nv >= NVN_REFL_MAXV) FAIL("vertex input %d out of range", loc);
        o->v[o->nv].loc = o->v[o->nv].nvn = (uint8_t)loc;
        o->v[o->nv].ct = t && !strncmp(t, "uint", 4) ? NVN_CT_UINT : t && !strncmp(t, "int", 3) ? NVN_CT_INT : NVN_CT_FLOAT;
        o->nv++;
    }
    const jv *tg = jv_get(j, "threadgroup_size");
    if (tg && tg->k == JV_ARR && tg->len == 3) for (int i = 0; i < 3; i++) o->tg[i] = (uint16_t)jv_int(tg->items[i], 0);
    o->pos_attr = -1; o->may_discard = 1; o->writes_smask = 1;
    return 0;
    #undef FAIL
}

/* ---------------- SPIR-V: resource variables and the binding rewrite ---------------- */
enum { SpvOpName = 5, SpvOpEntryPoint = 15, SpvOpTypeInt = 21, SpvOpTypeFloat = 22, SpvOpTypeImage = 25, SpvOpTypeSampler = 26, SpvOpTypeSampledImage = 27,
       SpvOpTypeArray = 28, SpvOpTypeRuntimeArray = 29, SpvOpTypeStruct = 30, SpvOpTypePointer = 32, SpvOpVariable = 59, SpvOpDecorate = 71 };
enum { DecBlock = 2, DecBufferBlock = 3, DecBuiltIn = 11, DecLocation = 30, DecBinding = 33, DecDescriptorSet = 34 };
enum { ScUniformConstant = 0, ScInput = 1, ScUniform = 2, ScOutput = 3, ScStorageBuffer = 12 };
typedef struct { uint32_t id; const char *name; uint32_t set_word, bind_word; uint32_t ptr_type, sc; } spv_var;
typedef struct {
    uint32_t *w; size_t n;
    uint32_t bound;
    /* per id */
    const char **name; uint8_t *op; uint32_t *a0, *a1, *a2; uint8_t *block, *bufblock, *builtin_dec; int32_t *loc; uint32_t *set_word, *bind_word;
    uint8_t *img_dim, *img_depth, *img_arr, *img_ms, *img_sampled; uint32_t *img_stype;
    uint8_t *var_sc;
} spv_mod;
static int spv_parse(spv_mod *m, uint32_t *w, size_t n) {
    memset(m, 0, sizeof *m);
    if (n < 5 || w[0] != 0x07230203) return 0;
    m->w = w; m->n = n; m->bound = w[3];
    uint32_t b = m->bound;
    m->name = calloc(b, sizeof *m->name); m->op = calloc(b, 1); m->a0 = calloc(b, 4); m->a1 = calloc(b, 4); m->a2 = calloc(b, 4);
    m->block = calloc(b, 1); m->bufblock = calloc(b, 1); m->builtin_dec = calloc(b, 1); m->loc = malloc(b * 4); m->set_word = calloc(b, 4); m->bind_word = calloc(b, 4);
    m->img_dim = calloc(b, 1); m->img_depth = calloc(b, 1); m->img_arr = calloc(b, 1); m->img_ms = calloc(b, 1); m->img_sampled = calloc(b, 1); m->img_stype = calloc(b, 4);
    m->var_sc = calloc(b, 1);
    if (!m->loc) return 0;
    memset(m->loc, 0xff, b * 4);
    for (size_t i = 5; i < n;) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) return 0;
        uint32_t *a = w + i + 1;
        switch (op) {
            case SpvOpName: if (a[0] < b) m->name[a[0]] = (const char *)(a + 1); break;
            case SpvOpDecorate:
                if (a[0] >= b) break;
                if (a[1] == DecBlock) m->block[a[0]] = 1;
                else if (a[1] == DecBufferBlock) m->bufblock[a[0]] = 1;
                else if (a[1] == DecBuiltIn) m->builtin_dec[a[0]] = (uint8_t)(a[2] + 1 > 255 ? 255 : a[2] + 1);
                else if (a[1] == DecLocation) m->loc[a[0]] = (int32_t)a[2];
                else if (a[1] == DecDescriptorSet) m->set_word[a[0]] = (uint32_t)(i + 3);
                else if (a[1] == DecBinding) m->bind_word[a[0]] = (uint32_t)(i + 3);
                break;
            case SpvOpTypeInt: if (a[0] < b) { m->op[a[0]] = (uint8_t)op; m->a1[a[0]] = a[2]; } break;   /* a1: signedness */
            case SpvOpTypeFloat: if (a[0] < b) m->op[a[0]] = (uint8_t)op; break;
            case SpvOpTypeImage: if (a[0] < b) { m->op[a[0]] = (uint8_t)op; m->img_stype[a[0]] = a[1]; m->img_dim[a[0]] = (uint8_t)a[2]; m->img_depth[a[0]] = (uint8_t)a[3];
                                                   m->img_arr[a[0]] = (uint8_t)a[4]; m->img_ms[a[0]] = (uint8_t)a[5]; m->img_sampled[a[0]] = (uint8_t)a[6]; } break;
            case SpvOpTypeSampler: if (a[0] < b) m->op[a[0]] = (uint8_t)op; break;
            case SpvOpTypeSampledImage: if (a[0] < b) { m->op[a[0]] = (uint8_t)op; m->a0[a[0]] = a[1]; } break;
            case SpvOpTypeArray: case SpvOpTypeRuntimeArray: if (a[0] < b) { m->op[a[0]] = (uint8_t)op; m->a0[a[0]] = a[1]; } break;
            case SpvOpTypeStruct: if (a[0] < b) m->op[a[0]] = (uint8_t)op; break;
            case SpvOpTypePointer: if (a[0] < b) { m->op[a[0]] = (uint8_t)op; m->a0[a[0]] = a[1]; m->a1[a[0]] = a[2]; } break;   /* sc, pointee */
            case SpvOpVariable: if (a[1] < b) { m->op[a[1]] = (uint8_t)op; m->a0[a[1]] = a[0]; m->var_sc[a[1]] = (uint8_t)a[2]; } break;   /* type */
        }
        i += wc;
    }
    return 1;
}
static void spv_free(spv_mod *m) {
    free(m->name); free(m->op); free(m->a0); free(m->a1); free(m->a2); free(m->block); free(m->bufblock); free(m->builtin_dec); free(m->loc);
    free(m->set_word); free(m->bind_word); free(m->img_dim); free(m->img_depth); free(m->img_arr); free(m->img_ms); free(m->img_sampled); free(m->img_stype); free(m->var_sc);
}
/* Declarations the device rejects even when unused: Ryujinx declares Float64 (the Mac path keeps it, spirv-cross ignores
   it; Honeykrisp has no FP64, and shaders that use doubles are translated with the soft-float lowering anyway) and fragment
   shader interlock. They are dropped when nothing in the module uses them. Returns the new word count. */
static size_t spv_strip(uint32_t *w, size_t n, const char *key) {
    int f64_used = 0, interlock_used = 0;
    for (size_t i = 5; i < n;) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff; if (!wc) break;
        if (op == SpvOpTypeFloat && wc >= 3 && w[i + 2] == 64) f64_used = 1;
        if (op == 5364 || op == 5365) interlock_used = 1;   /* OpBegin / OpEndInvocationInterlockEXT */
        if (op == 16 && wc >= 3 && w[i + 2] >= 5366 && w[i + 2] <= 5371) interlock_used = 1;   /* OpExecutionMode *InterlockEXT */
        i += wc;
    }
    size_t o = 5;
    for (size_t i = 5; i < n;) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff; if (!wc) break;
        int drop = 0;
        if (op == 17 && wc == 2) {   /* OpCapability */
            uint32_t cap = w[i + 1];
            if (cap == 10 && !f64_used) drop = 1;
            if ((cap == 5378 || cap == 5363 || cap == 5372) && !interlock_used) drop = 1;
        }
        if (op == 10 && !interlock_used && !strcmp((const char *)(w + i + 1), "SPV_EXT_fragment_shader_interlock")) drop = 1;   /* OpExtension */
        if (!drop) { if (o != i) memmove(w + o, w + i, wc * 4); o += wc; }
        i += wc;
    }
    if (f64_used) VK_ONCE(f64, "shader %s uses 64-bit floats, which this GPU does not have", key);
    if (interlock_used) VK_ONCE(ilk, "shader %s uses fragment shader interlock, which this GPU does not have", key);
    return o;
}
static uint32_t spv_find_var(const spv_mod *m, const char *name) {
    for (uint32_t id = 1; id < m->bound; id++) if (m->op[id] == SpvOpVariable && m->name[id] && !strcmp(m->name[id], name)) return id;
    return 0;
}
/* descriptor class of a resource variable: VB_* or -1; *img: the image type id (images) */
static int spv_class(const spv_mod *m, uint32_t var, uint32_t *img, int *arrayed_res) {
    uint32_t pt = m->a0[var]; if (m->op[pt] != SpvOpTypePointer) return -1;
    uint32_t sc = m->a0[pt], t = m->a1[pt];
    *arrayed_res = 0;
    if (m->op[t] == SpvOpTypeArray || m->op[t] == SpvOpTypeRuntimeArray) { *arrayed_res = 1; t = m->a0[t]; }
    if (sc == ScUniform || sc == ScStorageBuffer) {
        if (m->op[t] != SpvOpTypeStruct) return -1;
        if (sc == ScStorageBuffer || m->bufblock[t]) return VB_SSBO;
        return VB_UBO;
    }
    if (sc != ScUniformConstant) return -1;
    if (m->op[t] == SpvOpTypeSampler) return VB_SAMPLER;
    if (m->op[t] == SpvOpTypeSampledImage) { *img = m->a0[t]; return m->img_dim[*img] == 5 ? VB_UNIFORM_TEXEL : VB_COMBINED; }
    if (m->op[t] == SpvOpTypeImage) { *img = t; int buf = m->img_dim[t] == 5; return m->img_sampled[t] == 2 ? (buf ? VB_STORAGE_TEXEL : VB_STORAGE_IMG) : (buf ? VB_UNIFORM_TEXEL : VB_SAMPLED); }
    return -1;
}
static uint8_t img_view_type(const spv_mod *m, uint32_t img) {
    int arr = m->img_arr[img];
    switch (m->img_dim[img]) {
        case 0: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
        case 2: return VK_IMAGE_VIEW_TYPE_3D;
        case 3: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}
static uint8_t img_class(const spv_mod *m, uint32_t img) {
    uint32_t st = m->img_stype[img];
    if (m->op[st] == SpvOpTypeInt) return m->a1[st] ? 2 : 1;
    return 0;
}

/* ---------------- shader objects ---------------- */
static nvn_hmap g_shcache = {.klen = 80};   /* "<variant>_b" (zero padded) -> vk_shader */
static VkBuffer g_c1_buf; static VmaAllocation g_c1_alloc; static uint8_t *g_c1_cpu; static size_t g_c1_used, g_c1_size;
#define C1_HEAP (64u << 20)
static void c1_place(vk_shader *sh, const char *msl) {
    size_t len = sh->rf.c1_len;
    int bounded = c1_reads_bounded(msl, len);
    size_t need = bounded ? ((len + 255u) & ~(size_t)255u) + 256u : 65536u;
    if (!g_c1_buf || g_c1_used + need + 65536 > g_c1_size) {   /* the old heap stays (shaders hold it) */
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = C1_HEAP, .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
        VmaAllocationInfo info; VK(vmaCreateBuffer(V.vma, &bi, &ai, &g_c1_buf, &g_c1_alloc, &info));
        g_c1_cpu = info.pMappedData; g_c1_size = C1_HEAP; g_c1_used = 0; memset(g_c1_cpu, 0, C1_HEAP);
    }
    memcpy(g_c1_cpu + g_c1_used, sh->rf.c1, len);
    sh->c1_buf = g_c1_buf; sh->c1_off = g_c1_used; sh->c1_range = 65536;   /* reads past the data read zeros (the gap / the exclusive 64 KiB) */
    g_c1_used += need;
}
static int shader_binds(vk_shader *sh, spv_mod *m, const jv *refl, char *err, size_t errlen) {
    const jv *bs = jv_get(refl, "bindings");
    int ncomb = 0, nsampled = 0, nsampler = 0, nsimg = 0, nutex = 0, nstex = 0;
    uint32_t set = sh->stage == 1 ? 1 : 0;
    for (int i = 0; bs && i < bs->len; i++) {
        const jv *b = bs->items[i]; const nvn_rbind *rb = &sh->rf.b[i];
        const char *name = jv_str(jv_get(b, "msl_name")); if (!name) continue;
        size_t nl = strlen(name);
        if ((rb->kind == NVN_RB_SAMP || rb->kind == NVN_RB_SAMPT) && nl > 5 && !strcmp(name + nl - 5, "Smplr")) {
            /* the sampler half of a combined image sampler: attach to its texture */
            char base[200]; snprintf(base, sizeof base, "%.*s", (int)(nl - 5), name);
            int found = 0;
            for (int k = 0; k < sh->nb; k++) {
                vk_bind *vb = &sh->b[k];
                const char *tn = jv_str(jv_get(bs->items[vb->ji], "msl_name"));
                if (vb->type == VB_COMBINED && tn && !strcmp(tn, base)) {
                    vb->samp_kind = rb->kind; vb->samp_reg = rb->reg; found = 1; break; }
            }
            if (!found) { /* texture listed after its sampler: remember it on a pending entry */
                for (int k = i + 1; bs && k < bs->len; k++) { const char *tn = jv_str(jv_get(bs->items[k], "msl_name")); if (tn && !strcmp(tn, base)) { found = 2; break; } }
                if (found != 2) { snprintf(err, errlen, "sampler %s has no texture", name); return 0; }
            }
            continue;
        }
        uint32_t var = spv_find_var(m, name);
        if (!var) { snprintf(err, errlen, "binding %s not in the SPIR-V", name); return 0; }
        uint32_t img = 0; int arrayed = 0, cls = spv_class(m, var, &img, &arrayed);
        if (cls < 0) { snprintf(err, errlen, "binding %s: unknown descriptor class", name); return 0; }
        if (arrayed) { snprintf(err, errlen, "binding %s: arrays of descriptors are not supported", name); return 0; }
        if (sh->nb >= NVN_REFL_MAXB) { snprintf(err, errlen, "too many bindings"); return 0; }
        vk_bind *vb = &sh->b[sh->nb];
        memset(vb, 0, sizeof *vb);
        vb->type = (uint8_t)cls; vb->kind = rb->kind; vb->rw = rb->rw; vb->reg = rb->reg; vb->off = rb->off; vb->ji = (uint8_t)i;   /* ji: index of the json entry (for the sampler match) */
        if (img) { vb->view_type = img_view_type(m, img); vb->shadow = m->img_depth[img] == 1; vb->ms = m->img_ms[img]; vb->cls = img_class(m, img); }
        int binding = -1;
        switch (cls) {
            case VB_UBO: {
                if (rb->kind == NVN_RB_SUPPORT) binding = VKB_UBO0;
                else { const char *c = strrchr(name, 'c'); int bank = c && c > name && c[-1] == '_' ? atoi(c + 1) : -1;
                       if (bank < 0 || bank > 17) { snprintf(err, errlen, "uniform buffer %s: no bank in the name", name); return 0; }
                       binding = VKB_UBO0 + 1 + bank; vb->bank = (uint8_t)bank; }
                break; }
            case VB_SSBO: {
                const char *s = strrchr(name, 's'); int idx = s && s > name && s[-1] == '_' ? atoi(s + 1) : -1;
                if (idx < 0 || idx >= VKB_NSSBO) { snprintf(err, errlen, "storage buffer %s: index out of range", name); return 0; }
                binding = VKB_SSBO0 + idx; break; }
            case VB_COMBINED: if (ncomb >= VKB_NCOMB) { snprintf(err, errlen, "more than %d textures", VKB_NCOMB); return 0; } binding = VKB_COMB0 + ncomb++; break;
            case VB_SAMPLED: if (nsampled >= VKB_NSAMPLED) { snprintf(err, errlen, "too many separate textures"); return 0; } binding = VKB_SAMPLED0 + nsampled++; break;
            case VB_SAMPLER: if (nsampler >= VKB_NSAMPLER) { snprintf(err, errlen, "too many separate samplers"); return 0; } binding = VKB_SAMPLER0 + nsampler++; break;
            case VB_STORAGE_IMG: if (nsimg >= VKB_NSIMG) { snprintf(err, errlen, "too many images"); return 0; } binding = VKB_SIMG0 + nsimg++; break;
            case VB_UNIFORM_TEXEL: if (nutex >= VKB_NUTEX) { snprintf(err, errlen, "too many texel buffers"); return 0; } binding = VKB_UTEX0 + nutex++; break;
            case VB_STORAGE_TEXEL: if (nstex >= VKB_NSTEX) { snprintf(err, errlen, "too many storage texel buffers"); return 0; } binding = VKB_STEX0 + nstex++; break;
        }
        vb->binding = (uint8_t)binding;
        if (!m->set_word[var] || !m->bind_word[var]) { snprintf(err, errlen, "binding %s lacks set / binding decorations", name); return 0; }
        m->w[m->set_word[var]] = set; m->w[m->bind_word[var]] = (uint32_t)binding;
        sh->nb++;
    }
    /* samplers listed before their textures */
    for (int i = 0; bs && i < bs->len; i++) {
        const nvn_rbind *rb = &sh->rf.b[i]; const char *name = jv_str(jv_get(bs->items[i], "msl_name")); if (!name) continue;
        size_t nl = strlen(name);
        if (!((rb->kind == NVN_RB_SAMP || rb->kind == NVN_RB_SAMPT) && nl > 5 && !strcmp(name + nl - 5, "Smplr"))) continue;
        for (int k = 0; k < sh->nb; k++) {
            vk_bind *vb = &sh->b[k]; const char *tn = jv_str(jv_get(bs->items[vb->ji], "msl_name"));
            if (vb->type == VB_COMBINED && !vb->samp_kind && tn && strlen(tn) == nl - 5 && !strncmp(tn, name, nl - 5)) { vb->samp_kind = rb->kind; vb->samp_reg = rb->reg; }
        }
    }
    /* varyings for the interface between independent stages */
    for (uint32_t id = 1; id < m->bound; id++) {
        if (m->op[id] != SpvOpVariable || m->builtin_dec[id] || m->loc[id] < 0 || m->loc[id] >= 32) continue;
        if (m->var_sc[id] == ScOutput && sh->stage == 0) sh->out_locs |= 1u << m->loc[id];
        if (m->var_sc[id] == ScInput && sh->stage == 1) sh->in_locs |= 1u << m->loc[id];
        if (m->var_sc[id] == ScOutput && sh->stage == 1) { /* colour outputs */ }
    }
    for (uint32_t id = 1; id < m->bound; id++) if (m->op[id] == SpvOpVariable && m->var_sc[id] == ScOutput && m->builtin_dec[id] == 22 + 1) sh->writes_depth = 1;   /* BuiltIn FragDepth */
    return 1;
}
static vk_shader *shader_load(const char *pc, int stage) {
    paths_init();
    char p[4200], err[300];
    cache_path(p, sizeof p, pc, ".json");
    jv *j = jv_parse_file(p);
    if (!j) { VLOG("shader %s: no reflection (%s)", pc, p); return NULL; }
    const char *chain = jv_str(jv_get(j, "chain"));
    if (!chain || strcmp(chain, DMSH_CHAIN)) { VLOG("shader %s: translation chain %s, this host wants %s", pc, chain ? chain : "?", DMSH_CHAIN); jv_free(j); return NULL; }
    vk_shader *sh = calloc(1, sizeof *sh); if (!sh) VDIE("shader");
    snprintf(sh->key, sizeof sh->key, "%s", pc); sh->stage = stage;
    if (refl_from_json(j, &sh->rf, err, sizeof err)) { VLOG("shader %s: reflection: %s", pc, err); jv_free(j); free(sh); return NULL; }
    char variant[96]; snprintf(variant, sizeof variant, "%.*s", (int)(strlen(pc) > 2 ? strlen(pc) - 2 : 0), pc);   /* strip "_b" */
    cache_path(p, sizeof p, pc, ".metal");
    char *msl = read_file(p, NULL);
    if (msl) {
        drv_scan(&sh->rf, msl);
        const jv *bs = jv_get(j, "bindings");
        for (int i = 0; bs && i < sh->rf.nb; i++) { nvn_rbind *b = &sh->rf.b[i]; if (!b->rw && (b->kind == NVN_RB_SBUFP || b->kind == NVN_RB_SBUFU) && msl_stores_through(msl, jv_str(jv_get(bs->items[i], "msl_name")))) b->rw = 1; }
    }
    const jv *cs = jv_get(j, "compare_slots");
    for (int i = 0; cs && cs->k == JV_ARR && i < cs->len; i++) { int s = jv_int(cs->items[i], -1); if (s >= 0 && s < 128) sh->compare_slots[s >> 6] |= 1ull << (s & 63); }
    size_t spv_len = 0;
    cache_path(p, sizeof p, variant, ".spv");
    uint32_t *code = (uint32_t *)read_file(p, &spv_len);
    if (!code || spv_len % 4) { VLOG("shader %s: SPIR-V missing (%s)", pc, p); free(code); free(msl); jv_free(j); free(sh); return NULL; }
    spv_mod m;
    if (!spv_parse(&m, code, spv_len / 4)) { VLOG("shader %s: SPIR-V unreadable", pc); free(code); free(msl); jv_free(j); free(sh); return NULL; }
    if (!shader_binds(sh, &m, j, err, sizeof err)) { VLOG("shader %s: %s", pc, err); spv_free(&m); free(code); free(msl); jv_free(j); free(sh); return NULL; }
    spv_free(&m);
    spv_len = spv_strip(code, spv_len / 4, pc) * 4;
    if (sh->rf.has_c1) c1_place(sh, msl);
    for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].rw) sh->keeps = 1;
    VkDescriptorSetLayout sets[2] = {V.dsl, V.dsl};
    VkShaderCreateInfoEXT ci = {VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .stage = stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT : stage == 1 ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_COMPUTE_BIT,
        .nextStage = stage == 0 ? VK_SHADER_STAGE_FRAGMENT_BIT : 0, .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = spv_len, .pCode = code, .pName = "main",
        .setLayoutCount = stage == 5 ? 1 : 2, .pSetLayouts = sets,   /* graphics stages drawn together need identical arrays */
        .pushConstantRangeCount = stage == 5 ? 0 : 1, .pPushConstantRanges = &vk_gfx_pc};
    uint64_t t0 = vk_now_ns();
    VkResult r = V.CreateShadersEXT(V.dev, 1, &ci, NULL, &sh->obj);
    free(code); free(msl); jv_free(j);
    if (r != VK_SUCCESS) { VLOG("shader %s: vkCreateShadersEXT failed %d", pc, r); free(sh); return NULL; }
    static uint64_t n, ns; n++; ns += vk_now_ns() - t0;
    if (n <= 4 || n % 200 == 0) VLOG("shader objects: %llu made, %.1f ms on average", (unsigned long long)n, ns / 1e6 / (double)n);
    return sh;
}
static vk_shader *shader_get(const char *variant, int stage) {
    char pc[80] = {0}; snprintf(pc, sizeof pc, "%s_b", variant);
    vk_shader *sh = nvn_hm_get(&g_shcache, pc);
    if (sh) return sh;
    sh = shader_load(pc, stage);
    if (sh) nvn_hm_put(&g_shcache, pc, sh);
    return sh;
}

/* ---------------- the layer output of layered passes (vk_exec.c g_lay) ---------------- */
/* The module of a cached shader as shader_load hands it to the device (bindings rewritten, declarations stripped), read
   again from the cache: the loaded shaders keep no SPIR-V (13,000 of them). NULL when a file is missing. */
static uint32_t *spirv_for(const char *pc, int stage, size_t *nwords) {
    char p[4200], err[300];
    cache_path(p, sizeof p, pc, ".json");
    jv *j = jv_parse_file(p); if (!j) return NULL;
    vk_shader *tmp = calloc(1, sizeof *tmp); if (!tmp) VDIE("shader");
    tmp->stage = stage;
    char variant[96]; snprintf(variant, sizeof variant, "%.*s", (int)(strlen(pc) > 2 ? strlen(pc) - 2 : 0), pc);
    size_t len = 0; uint32_t *code = NULL;
    if (!refl_from_json(j, &tmp->rf, err, sizeof err)) {
        cache_path(p, sizeof p, variant, ".spv");
        code = (uint32_t *)read_file(p, &len);
    }
    spv_mod m; memset(&m, 0, sizeof m);
    if (code && len % 4 == 0 && spv_parse(&m, code, len / 4) && shader_binds(tmp, &m, j, err, sizeof err)) *nwords = spv_strip(code, len / 4, pc);
    else { free(code); code = NULL; }
    spv_free(&m); free(tmp); jv_free(j);
    return code;
}
enum { SpvOpExtension = 10, SpvOpCapability = 17, SpvOpMemberDecorate = 72, SpvOpConstant = 43, SpvOpFunction = 54, SpvOpFunctionParameter = 55,
       SpvOpLabel = 248, SpvOpAccessChain = 65, SpvOpLoad = 61, SpvOpStore = 62, SpvOpLine = 8, SpvOpNoLine = 317 };
/* instructions of the module's sections before the types (capabilities .. annotations) */
static int spv_preamble_op(uint32_t op) {
    switch (op) {
        case 2: case 3: case 4: case 5: case 6: case 7: case 10: case 11: case 14: case 15: case 16: case 17: case 71: case 72: case 73: case 74: case 75:
        case 330: case 331: case 332: case 5632: case 5633: return 1;
        default: return 0;
    }
}
typedef struct { uint32_t *w; size_t n, cap; } spv_out;
static void spv_put(spv_out *o, const uint32_t *w, size_t n) {
    if (o->n + n > o->cap) { o->cap = (o->n + n) * 2 + 256; o->w = realloc(o->w, o->cap * 4); if (!o->w) VDIE("SPIR-V"); }
    memcpy(o->w + o->n, w, n * 4); o->n += n;
}
/* The vertex shader writes gl_Layer = the uint push constant at offset 0 (the draw's layer within the layered pass),
   stored at the top of main, as nvn_mtl.m vs_fn_for stores totk_layer_in. NULL when the module writes a layer itself or
   has another shape than one vertex entry point. */
static uint32_t *spv_layer_patch(const uint32_t *w, size_t n, size_t *out_n) {
    if (n < 5 || w[0] != 0x07230203) return NULL;
    uint32_t version = w[1], id = w[3], entry = 0, uint_t = 0, sint_t = 0;
    size_t ep = 0, last_cap = 0, first_type = 0, first_fn = 0, ep_fn = 0;
    int has_cap = 0, has_ext = 0;
    for (size_t i = 5; i < n;) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) return NULL;
        switch (op) {
            case SpvOpCapability: last_cap = i + wc; if (wc == 2 && (w[i + 1] == 5254 || w[i + 1] == 69)) has_cap = 1; break;
            case SpvOpExtension: if (!strcmp((const char *)(w + i + 1), "SPV_EXT_shader_viewport_index_layer")) has_ext = 1; break;
            case SpvOpEntryPoint: if (ep || wc < 4 || w[i + 1] != 0) return NULL; ep = i; entry = w[i + 2]; break;   /* one entry point, Vertex */
            case SpvOpDecorate: if (wc >= 4 && w[i + 2] == DecBuiltIn && w[i + 3] == 9) return NULL; break;           /* BuiltIn Layer already */
            case SpvOpMemberDecorate: if (wc >= 5 && w[i + 3] == DecBuiltIn && w[i + 4] == 9) return NULL; break;
            case SpvOpTypeInt: if (wc == 4 && w[i + 2] == 32) { if (w[i + 3]) { if (!sint_t) sint_t = w[i + 1]; } else if (!uint_t) uint_t = w[i + 1]; } break;
            case SpvOpFunction: if (!first_fn) first_fn = i; if (wc >= 3 && entry && w[i + 2] == entry) ep_fn = i; break;
        }
        if (!first_type && !spv_preamble_op(op)) first_type = i;
        i += wc;
    }
    if (!ep || !ep_fn || !last_cap || !first_type || !first_fn || first_type > first_fn || ep > first_type) return NULL;
    /* the first instruction of main's body: after its parameters, label, local variables (and line markers) */
    size_t at = ep_fn + (w[ep_fn] >> 16); int label = 0;
    while (at < n) {
        uint32_t wc = w[at] >> 16, op = w[at] & 0xffff;
        if (!wc) return NULL;
        if (op == SpvOpFunctionParameter && !label) { at += wc; continue; }
        if (op == SpvOpLabel && !label) { label = 1; at += wc; continue; }
        if (label && (op == SpvOpVariable || op == SpvOpLine || op == SpvOpNoLine)) { at += wc; continue; }
        break;
    }
    if (!label || at >= n) return NULL;
    int new_int = !uint_t && !sint_t;
    uint32_t t_u = uint_t ? uint_t : sint_t ? sint_t : id++;
    uint32_t t_pcs = id++, t_pcs_ptr = id++, v_pc = id++, t_u_pc = id++, c0 = id++, t_out = id++, v_layer = id++, x_ac = id++, x_ld = id++;
    int v15 = version >= 0x00010500, v14 = version >= 0x00010400;
    spv_out o = {0};
    uint32_t hdr[5] = {w[0], w[1], w[2], id, w[4]}; spv_put(&o, hdr, 5);
    spv_put(&o, w + 5, last_cap - 5);
    if (!has_cap) { uint32_t c[2] = {2u << 16 | SpvOpCapability, v15 ? 69u : 5254u}; spv_put(&o, c, 2); }   /* ShaderLayer / ShaderViewportIndexLayerEXT */
    if (!v15 && !has_ext) {
        static const char ext[] = "SPV_EXT_shader_viewport_index_layer";
        uint32_t e[1 + (sizeof ext + 3) / 4]; memset(e, 0, sizeof e); memcpy(e + 1, ext, sizeof ext);
        e[0] = (uint32_t)(sizeof e / 4) << 16 | SpvOpExtension; spv_put(&o, e, sizeof e / 4);
    }
    spv_put(&o, w + last_cap, ep - last_cap);
    {   /* the entry point lists the new interface variables (SPIR-V 1.4 on: every global it uses) */
        uint32_t wc = w[ep] >> 16, add[2] = {v_layer, v_pc}, na = v14 ? 2 : 1;
        uint32_t h = (wc + na) << 16 | SpvOpEntryPoint; spv_put(&o, &h, 1); spv_put(&o, w + ep + 1, wc - 1); spv_put(&o, add, na);
        spv_put(&o, w + ep + wc, first_type - (ep + wc));
    }
    uint32_t dec[] = {4u << 16 | SpvOpDecorate, v_layer, DecBuiltIn, 9, 3u << 16 | SpvOpDecorate, t_pcs, DecBlock, 5u << 16 | SpvOpMemberDecorate, t_pcs, 0, 35 /* Offset */, 0};
    spv_put(&o, dec, sizeof dec / 4);
    spv_put(&o, w + first_type, first_fn - first_type);
    if (new_int) { uint32_t t[4] = {4u << 16 | SpvOpTypeInt, t_u, 32, 0}; spv_put(&o, t, 4); }
    uint32_t glob[] = {3u << 16 | SpvOpTypeStruct, t_pcs, t_u, 4u << 16 | SpvOpTypePointer, t_pcs_ptr, 9 /* PushConstant */, t_pcs,
                       4u << 16 | SpvOpVariable, t_pcs_ptr, v_pc, 9, 4u << 16 | SpvOpTypePointer, t_u_pc, 9, t_u, 4u << 16 | SpvOpConstant, t_u, c0, 0,
                       4u << 16 | SpvOpTypePointer, t_out, ScOutput, t_u, 4u << 16 | SpvOpVariable, t_out, v_layer, ScOutput};
    spv_put(&o, glob, sizeof glob / 4);
    spv_put(&o, w + first_fn, at - first_fn);
    uint32_t body[] = {5u << 16 | SpvOpAccessChain, t_u_pc, x_ac, v_pc, c0, 4u << 16 | SpvOpLoad, t_u, x_ld, x_ac, 3u << 16 | SpvOpStore, v_layer, x_ld};
    spv_put(&o, body, sizeof body / 4);
    spv_put(&o, w + at, n - at);
    *out_n = o.n;
    return o.w;
}
VkShaderEXT vk_shader_layered(vk_shader *sh) {
    if (sh->lay_state) return sh->lay_state > 0 ? sh->lay_obj : VK_NULL_HANDLE;
    sh->lay_state = -1;
    if (sh->stage != 0) return VK_NULL_HANDLE;
    size_t n = 0, pn = 0;
    uint32_t *code = spirv_for(sh->key, 0, &n), *p = code ? spv_layer_patch(code, n, &pn) : NULL;
    free(code);
    if (!p) { static unsigned said; if (said++ < 4) VLOG("layered pass: vertex shader %s cannot write the layer (its passes stay one layer each)", sh->key); return VK_NULL_HANDLE; }
    VkDescriptorSetLayout sets[2] = {V.dsl, V.dsl};
    VkShaderCreateInfoEXT ci = {VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = VK_SHADER_STAGE_VERTEX_BIT, .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = pn * 4, .pCode = p, .pName = "main", .setLayoutCount = 2, .pSetLayouts = sets,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &vk_gfx_pc};
    VkResult r = V.CreateShadersEXT(V.dev, 1, &ci, NULL, &sh->lay_obj);
    free(p);
    if (r != VK_SUCCESS) { sh->lay_obj = VK_NULL_HANDLE; VLOG("layered pass: the layer variant of vertex shader %s was refused (%d)", sh->key, r); return VK_NULL_HANDLE; }
    sh->lay_state = 1;
    static unsigned made; if (made++ < 4) VLOG("layered pass: vertex shader %s writes its layer (variant %u)", sh->key, made);
    return sh->lay_obj;
}

/* ---------------- the UI squeeze of key 4 / TOTK_ASPECT (vk_exec.c uw_ui_squeeze) ---------------- */
/* With the picture filling a window of another shape than 16:9, the UI drawn into the canvas is squeezed about the
   centre to keep its shape. Squeezed through the viewport, nothing of it reached outside the middle 16:9: a full-screen
   fade or backdrop (the game-over screen's black) left the scene showing above and below it. This variant of a vertex
   shader squeezes each vertex instead, at the end of main: position x *= the float push constant at offset 8, y *= the
   one at 12, except a coordinate on or beyond the screen's edge (|x| or |y| >= 0.999 w), which stays, so a full-screen
   quad still fills the window and the rest keeps its shape. NULL when the module writes its position otherwise than
   through one gl_PerVertex block (or a BuiltIn Position variable), has push constants of its own or another shape than
   one vertex entry point. */
enum { SpvOpTypeBool = 20, SpvOpFMul = 133, SpvOpFOrdGreaterThanEqual = 190, SpvOpSelect = 169, SpvOpReturn = 253, SpvOpFunctionEnd = 56 };
static uint32_t *spv_squeeze_patch(const uint32_t *w, size_t n, size_t *out_n) {
    if (n < 5 || w[0] != 0x07230203) return NULL;
    uint32_t version = w[1], id = w[3], entry = 0, t_float = 0, t_bool = 0, t_int = 0, blk = 0, blk_member = 0, pos_var = 0, t_v4 = 0;
    size_t ep = 0, first_type = 0, first_fn = 0, ep_fn = 0, ep_end = 0;
    uint32_t ptr_blk[16]; int nptr_blk = 0, nret = 0;
    for (size_t i = 5; i < n;) {   /* the types and the position's block first */
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) return NULL;
        switch (op) {
            case SpvOpEntryPoint: if (ep || wc < 4 || w[i + 1] != 0) return NULL; ep = i; entry = w[i + 2]; break;
            case SpvOpMemberDecorate: if (wc >= 5 && w[i + 3] == DecBuiltIn && w[i + 4] == 0) { if (blk) return NULL; blk = w[i + 1]; blk_member = w[i + 2]; } break;
            case SpvOpDecorate: if (wc >= 4 && w[i + 2] == DecBuiltIn && w[i + 3] == 0) { if (pos_var) return NULL; pos_var = w[i + 1]; } break;
            case SpvOpTypeFloat: if (wc == 3 && w[i + 2] == 32 && !t_float) t_float = w[i + 1]; break;
            case SpvOpTypeBool: if (wc == 2 && !t_bool) t_bool = w[i + 1]; break;
            case SpvOpTypeInt: if (wc == 4 && w[i + 2] == 32 && !t_int) t_int = w[i + 1]; break;
            case 23 /* OpTypeVector */: if (wc == 4 && w[i + 2] == t_float && w[i + 3] == 4 && t_float && !t_v4) t_v4 = w[i + 1]; break;
            case SpvOpTypePointer: if (wc == 4 && w[i + 2] == ScOutput && blk && w[i + 3] == blk && nptr_blk < 16) ptr_blk[nptr_blk++] = w[i + 1]; break;
            case SpvOpVariable: if (wc >= 4 && w[i + 3] == 9 /* PushConstant */) return NULL; break;
            case SpvOpFunction: if (!first_fn) first_fn = i; if (wc >= 3 && entry && w[i + 2] == entry) ep_fn = i; break;
            case SpvOpFunctionEnd: if (ep_fn && !ep_end) ep_end = i; break;
            case SpvOpReturn: if (ep_fn && !ep_end) nret++; break;
        }
        if (!first_type && !spv_preamble_op(op)) first_type = i;
        i += wc;
    }
    if (!ep || !ep_fn || !ep_end || !nret || !t_float || !first_type || !first_fn || first_type > first_fn || ep > first_type || (blk && pos_var) || (!blk && !pos_var)) return NULL;
    uint32_t pv = 0;   /* the Output variable of the block */
    if (blk) {
        for (size_t i = 5; i < first_fn && !pv;) {
            uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
            if (op == SpvOpVariable && wc >= 4 && w[i + 3] == ScOutput) for (int k = 0; k < nptr_blk; k++) if (w[i + 1] == ptr_blk[k]) pv = w[i + 2];
            i += wc;
        }
        if (!pv) return NULL;
    } else if (!t_v4) return NULL;
    int new_bool = !t_bool, new_int = !t_int;
    if (new_bool) t_bool = id++;
    if (new_int) t_int = id++;
    uint32_t t_pcs = id++, t_pcs_ptr = id++, v_pc = id++, t_f_pc = id++, t_f_out = id++, c0 = id++, c1 = id++, c3 = id++, cm = id++, clim = id++;
    uint32_t base = id; id += (uint32_t)nret * 21;   /* the ids of each insertion */
    int v14 = version >= 0x00010400;
    spv_out o = {0};
    uint32_t hdr[5] = {w[0], w[1], w[2], id, w[4]}; spv_put(&o, hdr, 5);
    spv_put(&o, w + 5, ep - 5);
    {   uint32_t wc = w[ep] >> 16;   /* SPIR-V 1.4 on: the entry point lists the push constant variable too */
        if (v14) { uint32_t h = (wc + 1) << 16 | SpvOpEntryPoint; spv_put(&o, &h, 1); spv_put(&o, w + ep + 1, wc - 1); spv_put(&o, &v_pc, 1); }
        else spv_put(&o, w + ep, wc);
        spv_put(&o, w + ep + wc, first_type - (ep + wc));
    }
    uint32_t dec[] = {3u << 16 | SpvOpDecorate, t_pcs, DecBlock, 5u << 16 | SpvOpMemberDecorate, t_pcs, 0, 35 /* Offset */, 8, 5u << 16 | SpvOpMemberDecorate, t_pcs, 1, 35, 12};
    spv_put(&o, dec, sizeof dec / 4);
    spv_put(&o, w + first_type, first_fn - first_type);
    if (new_bool) { uint32_t t[2] = {2u << 16 | SpvOpTypeBool, t_bool}; spv_put(&o, t, 2); }
    if (new_int) { uint32_t t[4] = {4u << 16 | SpvOpTypeInt, t_int, 32, 1}; spv_put(&o, t, 4); }
    float lim = 0.998f; uint32_t limb; memcpy(&limb, &lim, 4);   /* (0.999 w)^2 */
    uint32_t glob[] = {4u << 16 | SpvOpTypeStruct, t_pcs, t_float, t_float, 4u << 16 | SpvOpTypePointer, t_pcs_ptr, 9, t_pcs,
                       4u << 16 | SpvOpVariable, t_pcs_ptr, v_pc, 9, 4u << 16 | SpvOpTypePointer, t_f_pc, 9, t_float,
                       4u << 16 | SpvOpTypePointer, t_f_out, ScOutput, t_float,
                       4u << 16 | SpvOpConstant, t_int, c0, 0, 4u << 16 | SpvOpConstant, t_int, c1, 1, 4u << 16 | SpvOpConstant, t_int, c3, 3,
                       4u << 16 | SpvOpConstant, t_int, cm, blk_member, 4u << 16 | SpvOpConstant, t_float, clim, limb};
    spv_put(&o, glob, sizeof glob / 4);
    size_t at = first_fn; int r = 0;
    for (size_t i = first_fn; i < n;) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (op == SpvOpReturn && i > ep_fn && i < ep_end) {
            spv_put(&o, w + at, i - at); at = i;
            uint32_t b = base + 21u * (uint32_t)r++, ax = b, ay = b + 1, aw = b + 2, x = b + 3, y = b + 4, wv = b + 5, ww = b + 6, lm = b + 7, xx = b + 8, yy = b + 9,
                     ex = b + 10, ey = b + 11, pfx = b + 12, pfy = b + 13, fx = b + 14, fy = b + 15, sx = b + 16, sy = b + 17, nx = b + 18, ny = b + 19;
            uint32_t root = blk ? pv : pos_var;
            uint32_t code[] = {
                (blk ? 6u : 5u) << 16 | SpvOpAccessChain, t_f_out, ax, root, blk ? cm : c0, blk ? c0 : 0,
                (blk ? 6u : 5u) << 16 | SpvOpAccessChain, t_f_out, ay, root, blk ? cm : c1, blk ? c1 : 0,
                (blk ? 6u : 5u) << 16 | SpvOpAccessChain, t_f_out, aw, root, blk ? cm : c3, blk ? c3 : 0,
            };
            /* the access chains: drop the unused last operand when the position is a plain vector variable */
            for (int k = 0; k < 3; k++) spv_put(&o, code + 6 * k, blk ? 6 : 5);
            uint32_t math[] = {
                4u << 16 | SpvOpLoad, t_float, x, ax, 4u << 16 | SpvOpLoad, t_float, y, ay, 4u << 16 | SpvOpLoad, t_float, wv, aw,
                5u << 16 | SpvOpFMul, t_float, ww, wv, wv, 5u << 16 | SpvOpFMul, t_float, lm, ww, clim,
                5u << 16 | SpvOpFMul, t_float, xx, x, x, 5u << 16 | SpvOpFMul, t_float, yy, y, y,
                5u << 16 | SpvOpFOrdGreaterThanEqual, t_bool, ex, xx, lm, 5u << 16 | SpvOpFOrdGreaterThanEqual, t_bool, ey, yy, lm,
                5u << 16 | SpvOpAccessChain, t_f_pc, pfx, v_pc, c0, 5u << 16 | SpvOpAccessChain, t_f_pc, pfy, v_pc, c1,
                4u << 16 | SpvOpLoad, t_float, fx, pfx, 4u << 16 | SpvOpLoad, t_float, fy, pfy,
                5u << 16 | SpvOpFMul, t_float, sx, x, fx, 5u << 16 | SpvOpFMul, t_float, sy, y, fy,
                6u << 16 | SpvOpSelect, t_float, nx, ex, x, sx, 6u << 16 | SpvOpSelect, t_float, ny, ey, y, sy,
                3u << 16 | SpvOpStore, ax, nx, 3u << 16 | SpvOpStore, ay, ny,
            };
            spv_put(&o, math, sizeof math / 4);
        }
        i += wc;
    }
    spv_put(&o, w + at, n - at);
    *out_n = o.n;
    return o.w;
}
VkShaderEXT vk_shader_ui_squeezed(vk_shader *sh) {
    if (sh->sq_state) return sh->sq_state > 0 ? sh->sq_obj : VK_NULL_HANDLE;
    sh->sq_state = -1;
    if (sh->stage != 0) return VK_NULL_HANDLE;
    size_t n = 0, pn = 0;
    uint32_t *code = spirv_for(sh->key, 0, &n), *p = code ? spv_squeeze_patch(code, n, &pn) : NULL;
    free(code);
    if (!p) { static unsigned said; if (said++ < 4) VLOG("UI squeeze: vertex shader %s cannot squeeze per vertex (its draws are squeezed through the viewport)", sh->key); return VK_NULL_HANDLE; }
    VkDescriptorSetLayout sets[2] = {V.dsl, V.dsl};
    VkShaderCreateInfoEXT ci = {VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .stage = VK_SHADER_STAGE_VERTEX_BIT, .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT, .codeSize = pn * 4, .pCode = p, .pName = "main", .setLayoutCount = 2, .pSetLayouts = sets,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &vk_gfx_pc};
    VkResult r = V.CreateShadersEXT(V.dev, 1, &ci, NULL, &sh->sq_obj);
    free(p);
    if (r != VK_SUCCESS) { sh->sq_obj = VK_NULL_HANDLE; VLOG("UI squeeze: the variant of vertex shader %s was refused (%d)", sh->key, r); return VK_NULL_HANDLE; }
    sh->sq_state = 1;
    static unsigned made; if (made++ < 4) VLOG("UI squeeze: vertex shader %s squeezes per vertex, its edges kept (variant %u)", sh->key, made);
    return sh->sq_obj;
}
/* tests: the squeeze variant's code of a cached shader as the device would get it (free it), or NULL */
uint32_t *vk_shader_squeeze_code(const char *variant, size_t *out_n) {
    char pc[96]; snprintf(pc, sizeof pc, "%s_b", variant);
    size_t n = 0; uint32_t *code = spirv_for(pc, 0, &n), *p = code ? spv_squeeze_patch(code, n, out_n) : NULL;
    free(code); return p;
}

/* ---------------- programs ---------------- */
static nvn_hmap g_progs = {.klen = 8};
static uint32_t ctl_key_len(const uint8_t *control) { uint32_t n; memcpy(&n, control + 0x20, 4); return n >= 0x7cc && n < 0x880 ? n : 0x880; }
static void shader_key_of(const uint8_t *control, const void *code, uint32_t length, char key[65]) {
    uint8_t md[32]; CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, control, ctl_key_len(control)); CC_SHA256_Update(&c, code, length); CC_SHA256_Final(md, &c);
    hex(md, 32, key);
}
static int sph_keeps_memory(const uint8_t *code, int st) {
    if (st != 0 && st != 1) return 1;
    uint32_t w0; memcpy(&w0, code + 0x30, 4);
    return (w0 >> 16 & 1) || (w0 >> 28 & 0xf);
}
static const uint32_t *rt_local(const uint8_t *control, int st, uint32_t out[3]) {
    if (st != 5) return NULL;
    memcpy(out, control + 0x748, 12);
    if (!out[0] || !out[1] || !out[2] || out[0] > 1024 || out[1] > 1024 || out[2] > 64 || (uint64_t)out[0] * out[1] * out[2] > 1024)
        VDIE("compute shader: implausible thread group size %u x %u x %u at control+0x748", out[0], out[1], out[2]);
    return out;
}
static vk_shader *stage_variant(vk_prog *pr, int st) {
    jv *ans = vk_answers(st);
    uint32_t local[3]; char vn[80];
    const char *v = rt_variant(pr->key[st], st, pr->ctl[st], pr->code[st], ans, rt_local(pr->ctl[st], st, local), vn);
    jv_free(ans);
    vk_shader *sh = v ? shader_get(v, st) : NULL;
    if (!sh) VDIE("shader %s (stage %d): translation or Vulkan shader object unavailable", vn, st);
    return sh;
}
static pthread_mutex_t g_prog_mu = PTHREAD_MUTEX_INITIALIZER;
vk_prog *vk_prog_get(const prog_t *p) {
    static const prog_t *lp; static uint64_t lser; static vk_prog *lpr;
    if (p == lp && p->serial == lser && lpr) return lpr;
    pthread_mutex_lock(&g_prog_mu);
    vk_prog *pr = p->serial ? nvn_hm_get(&g_progs, &p->serial) : NULL;
    pthread_mutex_unlock(&g_prog_mu);
    if (pr) { lp = p; lser = p->serial; lpr = pr; return pr; }
    pr = calloc(1, sizeof *pr); if (!pr) VDIE("program");
    pr->serial = p->serial;
    for (unsigned i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *control = p->sd[i].control; const uint8_t *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        if (!control || !code) VDIE("null shader binary");
        uint32_t length = 0, st = 0; memcpy(&length, control + 0x704, 4); memcpy(&st, control + 0x714, 4);
        if (!length || length > 16 * 1024 * 1024) VDIE("invalid shader code length %u", length);
        if (st != 0 && st != 1 && st != 5) VDIE("shader stage %u (tessellation / geometry) is not translated", st);
        pr->ctl[st] = control; pr->code[st] = code; shader_key_of(control, code, length, pr->key[st]);
        pr->keeps |= sph_keeps_memory(code, (int)st);
    }
    for (int st = 0; st < 6; st++) if (pr->ctl[st]) {
        pr->sh[st] = stage_variant(pr, st);
        pr->sig[st] = vk_stage_sig(st);
        if (st < 2) { pr->alt[st][0].sig = pr->sig[st]; pr->alt[st][0].sh = pr->sh[st]; pr->nalt[st] = 1; }
        if (pr->sh[st]->keeps) pr->keeps = 1;
    }
    if (p->serial) { pthread_mutex_lock(&g_prog_mu); vk_prog *old = nvn_hm_put(&g_progs, &pr->serial, pr); pthread_mutex_unlock(&g_prog_mu); if (old) vk_defer_free(UINT64_MAX, old); }
    lp = p; lser = p->serial; lpr = pr;
    return pr;
}
int vk_prog_fit(vk_prog *pr) {
    int changed = 0;
    for (int st = 0; st < 2; st++) {
        if (!pr->sh[st]) continue;
        uint64_t sg = vk_stage_sig(st);
        if (sg == pr->sig[st]) continue;
        vk_shader *v = NULL;
        for (int i = 0; i < pr->nalt[st]; i++) if (pr->alt[st][i].sig == sg) { v = pr->alt[st][i].sh; break; }
        if (!v) {
            v = stage_variant(pr, st);
            int i = pr->nalt[st] < 16 ? pr->nalt[st]++ : 15;
            pr->alt[st][i].sig = sg; pr->alt[st][i].sh = v;
        }
        if (v != pr->sh[st]) changed = 1;
        pr->sh[st] = v; pr->sig[st] = sg;
    }
    return changed;
}
void mtl_prog_release(prog_t *p) {
    if (!p->serial) return;
    pthread_mutex_lock(&g_prog_mu);
    vk_prog *pr = nvn_hm_del(&g_progs, &p->serial);
    pthread_mutex_unlock(&g_prog_mu);
    if (pr) vk_defer_free(UINT64_MAX, pr);   /* the replay may still use it; shader objects stay in the cache (shared by programs) */
}
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
void vk_shader_init(void) { paths_init(); }

/* ---------------- translator answers from the bound state (rt_answers / stage_sig of nvn_mtl.m) ---------------- */
static const char *fmt_class_name(int f) { int c = vk_fmt_class(f); return c == 1 ? "uint" : c == 2 ? "sint" : "float"; }
static const char *target_name(int t) {
    switch (t) { case 0: return "1d"; case 2: return "3d"; case 3: return "1darray"; case 4: return "2darray"; case 5: return "2dms";
                 case 6: return "2dmsarray"; case 8: return "cube"; case 9: return "cubearray"; case 10: return "buffer"; default: return "2d"; }
}
static const char *image_format(int f) {
    switch (f) {
        case 0x01: return "R8Unorm"; case 0x0D: return "R8G8Unorm"; case 0x25: return "R8G8B8A8Unorm"; case 0x27: return "R8G8B8A8Uint";
        case 0x05: return "R16Float"; case 0x11: return "R16G16Float"; case 0x29: return "R16G16B16A16Float"; case 0x08: return "R16Uint"; case 0x2C: return "R16G16B16A16Uint";
        case 0x0A: return "R32Float"; case 0x16: return "R32G32Float"; case 0x2E: return "R32G32B32A32Float"; case 0x0B: return "R32Uint"; case 0x0C: return "R32Sint";
        case 0x17: return "R32G32Uint"; case 0x2F: return "R32G32B32A32Uint"; case 0x3D: return "R10G10B10A2Unorm"; case 0x3F: return "R11G11B10Float";
        default: return NULL;
    }
}
struct jv *vk_answers(int st) {
    jv *tex = jv_new_obj(), *img = jv_new_obj(), *attr = jv_new_obj(), *fout = jv_new_obj();
    char k[8];
    for (int r = 0; r < 128; r++) if (S.tex[st][r]) {
        int target, f; if (!vk_treg_target_format(S.tex[st][r], &target, &f)) continue;
        char name[32]; snprintf(name, sizeof name, "%s%s", target_name(target), f >= 0x32 && f <= 0x36 ? "_depth" : "");
        snprintf(k, sizeof k, "%d", r); jv_set(tex, k, jv_new_str(name));
    }
    for (int r = 0; r < 32; r++) if (S.img[st][r]) {
        int target, f; if (!vk_treg_target_format(S.img[st][r], &target, &f)) continue;
        jv *o = jv_new_obj(); jv_set(o, "type", jv_new_str(target_name(target)));
        const char *fn = image_format(f); if (fn) jv_set(o, "fmt", jv_new_str(fn));
        snprintf(k, sizeof k, "%d", r); jv_set(img, k, o);
    }
    if (st == 0) for (int i = 0; i < S.nattr && i < 16; i++) { snprintf(k, sizeof k, "%d", i); jv_set(attr, k, jv_new_str(fmt_class_name(S.attr[i].fmt))); }
    if (st == 1 && R.has_rt) for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) {
        snprintf(k, sizeof k, "%d", i); jv_set(fout, k, jv_new_str(fmt_class_name(R.rt.has_cview[i] && R.rt.cview[i].set_format ? R.rt.cview[i].format : R.rt.color[i]->d.format)));
    }
    jv *a = jv_new_obj();
    jv_set(a, "tex", tex); jv_set(a, "img", img); jv_set(a, "attr", attr); jv_set(a, "fout", fout);
    return a;
}
static int cls_of(int f) { return vk_fmt_class(f); }
uint64_t vk_stage_sig(int st) {
    uint64_t sg = 0;
    if (st == 1 && R.has_rt) for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i])
        sg |= (uint64_t)(1 + cls_of(R.rt.has_cview[i] && R.rt.cview[i].set_format ? R.rt.cview[i].format : R.rt.color[i]->d.format)) << (3 * i);
    if (st == 0) for (int i = 0; i < S.nattr && i < 16; i++) sg |= (uint64_t)cls_of(S.attr[i].fmt) << (2 * i);
    /* the depth textures among the stage's slots: looked up (a lock per texture) only when the slots or the registry
       changed (two stages a draw, 3,400 draws a frame) */
    static uint32_t c_bind[6]; static uint64_t c_dm[6]; static int c_gen[6] = {-1, -1, -1, -1, -1, -1};
    extern int vk_tex_registry_gen(void); int gen = vk_tex_registry_gen();
    uint64_t dm = 0, th = st < 6 ? vk_tex_bind_gen[st] : 0;
    if (st < 6 && c_bind[st] == th && c_gen[st] == gen) dm = c_dm[st];
    else {
        for (int r = 0; r < 128; r++) if (S.tex[st][r]) { int target, f; if (vk_treg_target_format(S.tex[st][r], &target, &f) && f >= 0x32 && f <= 0x36) dm ^= 0x9E3779B97F4A7C15ull * (uint64_t)(r + 1); }
        if (st < 6) { c_bind[st] = (uint32_t)th; c_gen[st] = gen; c_dm[st] = dm; }
    }
    if (dm) sg ^= dm | 1ull << 63;
    return sg;
}
