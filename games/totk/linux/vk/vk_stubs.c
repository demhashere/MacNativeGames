/* The Metal backend's statistics and diagnostics that the shared code (hle_nvn.c, the runtime) reads: the Vulkan
   backend has none of these counters yet, so they read 0 / "off". */
#include "vk_priv.h"
#include "nvn_mtl.h"
#include <string.h>

void dm_gfx_anomdump(const char *path) { (void)path; }
void dm_gfx_ashot(const char *path) { (void)path; }
void dm_gfx_querylog_next(long n) { (void)n; }
void dm_gfx_set_flag(const char *name, int v) { VLOG("flag %s = %d: not available on Vulkan", name, v); }
void dm_gfx_shot_at(long frame, const char *path) { (void)frame; (void)path; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_window_shot(const char *path) { (void)path; }
void dm_nvn_probe_prediction(void) {}
void mtl_alloc_stats(uint64_t *waits, uint64_t *grown, int *n) { *waits = *grown = 0; *n = 0; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void mtl_blit_stats(uint64_t v[6]) { memset(v, 0, 6 * sizeof *v); }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = *store = *full = 0; }
uint64_t mtl_chunk_commits(void) { return 0; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = *pso_ns = 0; }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = *pso = 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
void mtl_keepalive_stats(uint64_t v[3]) { memset(v, 0, 3 * sizeof *v); }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) {
    static unsigned long long last; if (frame - last < 3000 && last) return; last = frame;
    uint64_t made = 0, live = 0; void vk_tex_stats(uint64_t *, uint64_t *); vk_tex_stats(&made, &live);
    VLOG("memory at frame %llu: resident %llu MB, images %llu live (%llu made)", frame, rss_mb, (unsigned long long)live, (unsigned long long)made);
}
void mtl_note_world(void) {}
void mtl_pool_res_stats(uint64_t v[4]) { memset(v, 0, 4 * sizeof *v); }
void mtl_ren_stats(uint64_t v[6]) { memset(v, 0, 6 * sizeof *v); }
void mtl_shot_at_present(const tex_t *t) { (void)t; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
void mtl_ub_present(double iv) { (void)iv; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
void mtl_upload_wait_stats(uint64_t *n, uint64_t *ns) { *n = *ns = 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
void mtl_variant_stats(uint64_t *n, uint64_t *ns, uint64_t *pre, uint64_t *bg, uint64_t *plain) { *n = *ns = *pre = *bg = *plain = 0; }
uint64_t mtl_view_stats(void) { return 0; }
int mtl_warm_wait(int timeout_ms) { (void)timeout_ms; return 0; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_res_counts(uint32_t tick, const char *why) { (void)tick; (void)why; }
void mtl_res_snapshot(mtl_res_count *out) { memset(out, 0, sizeof *out); }
void mtl_counter_stats(uint64_t rep[32], uint64_t rst[32]) { memset(rep, 0, 32 * sizeof *rep); memset(rst, 0, 32 * sizeof *rst); }
void bf_want_note(const char *key) { (void)key; }
void mtl_ui_ghost(int on, float px, float py, const float grid[4], float cx, float cy) { (void)on; (void)px; (void)py; (void)grid; (void)cx; (void)cy; }
int mtl_ui_ghost_rect(float r[4]) { (void)r; return 0; }
void mtl_cube_merge_stats(uint64_t v[4]) { memset(v, 0, 4 * sizeof *v); }
void xv_test_counts(int *dc, int *viol) { *dc = *viol = 0; }
int mtl_last_commit_error(void) { return 0; }
void mtl_wait_state_dump(void) {}
void mtl_app_active(int active) { (void)active; }
void mtl_wb_eager(void) {}
