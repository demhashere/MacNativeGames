/* Mac-only features with no Linux counterpart: MetalFX upscaling / frame interpolation (keys 2, 3), MetalFX temporal
   anti-aliasing (key 5), the standalone Metal GPU probes, the deterministic test clock and libSystem's per-thread
   instruction counters. Each answers "off" / "not available", the state the shared code already handles. */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

int mfx_enabled(void) { return 0; }
int mfx_latch(void) { return 0; }
void mfx_set(int mode) { if (mode) fprintf(stderr, "[totk] frame interpolation (key 2) needs MetalFX: not available on Linux\n"); }
int mfxup_enabled(void) { return 0; }
int mfxup_output_enabled(void) { return 0; }
int taa_enabled(void) { return 0; }
void taa_toggle(void) { fprintf(stderr, "[totk] temporal anti-aliasing (key 5) needs MetalFX: not available on Linux\n"); }

int dm_detclock_enabled(void) { return 0; }
void dm_guest_frame_tick(void) {}
int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; (void)buf; (void)nbytes; return -1; }

void runtime_gpu_path_probe(const char *which) { fprintf(stderr, "[totk] %s: the Metal GPU path probes do not exist on Linux\n", which); }
void runtime_vpool_probe(void) { fprintf(stderr, "[totk] probe-vpool: Metal only\n"); }
