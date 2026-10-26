/* Frame hold in whole vsyncs (Linux, hle_nvn.c fhold_tick; after nxnative's frame step, dm/gfx/nvn/nvn_fstep.h, MIT,
   unit test linux/tests/fhold_test.c). The host keeps each frame on screen k vsyncs of 1/60 s (60, 30 or 20 fps) and paces
   the game to the display's vblanks; the game's time step is then exactly k vsyncs (fh_next returns 1 for a frame on
   time: shown and cost within k + FH_LATE). Three frames in a row that cost more than k + up vsyncs step up to their cost
   (cost = max(CPU, GPU) of the frame, in vsyncs); FH_RUN frames in a row that would fit k - 1 + down, after a hold-off
   since the step up, try k - 1 for FH_RUN frames, and a trial that fails doubles the hold-off (1 s to 8 s), so a scene on
   the edge does not flap. up / down per mode (nvn_fhold_mode): the default, up 0.35 and down 0.2, steps up only below
   about 44 fps, so a scene between that and 60 still runs as fast as it can (each frame starts as soon as the last is
   done), and one far below is held at an even 30; strict, nxnative's (up 0.1, down 0.85 (k - 1) - (k - 1) = fit at 85%),
   holds only even steps: 60, or 30 as soon as 60 does not hold. */
#pragma once
#include <math.h>
#include <stdint.h>

#define FH_MAX 3            /* the longest hold: 20 fps */
#define FH_LATE 0.1         /* a frame is late when it or its cost took this many vsyncs (1.7 ms) more than its hold */
#define FH_RUN 30           /* frames in a row that fit k - 1 before a trial, and the frames a trial must hold */
#define FH_HOLD0 60.0       /* vsyncs after a step up before a trial (1 s), doubled after each failed trial */
#define FH_HOLD_MAX 480.0   /* 8 s */

typedef struct {
    int k;                    /* vsyncs each frame stays on screen (0 = not decided yet) */
    double up, down;          /* the mode's margins (nvn_fhold_mode): step up above k + up, fit k - 1 at k - 1 + down (down < 0: a share, fit at -down * (k - 1)) */
    int up_run, fit_run, trial;   /* costly frames in a row; frames in a row that fit k - 1; frames left of a trial at k - 1 */
    double hold, since;       /* vsyncs a trial waits after a step up; vsyncs since the last step up */
    uint64_t frames, ontime, ups, downs, failed;   /* statistics for the log */
} nvn_fhold;

/* 0: the default (60 when it holds, as fast as it can between 44 and 60 fps, an even 30 below), 1: strict (60 or 30) */
static inline void nvn_fhold_mode(nvn_fhold *s, int strict) { s->up = strict ? FH_LATE : 0.35; s->down = strict ? -0.85 : 0.2; }

/* Once per frame. shown: real time since the previous frame's start, in vsyncs; cost: the frame's work, max(CPU, GPU), in
   vsyncs; kmin: 1 at the 60 fps cap (2 while the game needs 30, a movie); kmax: the longest hold. Returns whether the frame
   was on time. s->k is how long the next frame stays on screen. */
static inline int nvn_fhold_next(nvn_fhold *s, double shown, double cost, int kmin, int kmax) {
    if (kmin < 1) kmin = 1;
    if (kmax > FH_MAX) kmax = FH_MAX;
    if (kmax < kmin) kmax = kmin;
    if (!s->hold) s->hold = FH_HOLD0;
    if (!s->up && !s->down) nvn_fhold_mode(s, 0);
    if (!s->k) { s->k = kmin; return 0; }   /* first frame: nothing measured yet */
    if (s->k < kmin) s->k = kmin;
    if (s->k > kmax) s->k = kmax;
    int held = s->k, ontime = shown <= held + FH_LATE && cost <= held + FH_LATE;
    s->frames++; s->ontime += ontime; s->since += shown;
    s->up_run = cost > held + s->up ? s->up_run + 1 : 0;
    if (s->up_run >= 3 && s->k < kmax) {   /* step up to the cost, at least one vsync */
        int want = (int)ceil(cost - 0.05);
        if (want < s->k + 1) want = s->k + 1;
        if (want > kmax) want = kmax;
        if (s->trial) { s->failed++; s->hold = s->hold * 2 < FH_HOLD_MAX ? s->hold * 2 : FH_HOLD_MAX; }
        s->k = want; s->ups++; s->trial = s->up_run = s->fit_run = 0; s->since = 0;
        return ontime;
    }
    if (s->trial && !--s->trial) s->hold = FH_HOLD0;   /* the trial held */
    double fit = s->down < 0 ? -s->down * (s->k - 1) : s->k - 1 + s->down;
    s->fit_run = s->k > kmin && cost <= fit ? s->fit_run + 1 : 0;
    if (s->fit_run >= FH_RUN && s->since >= s->hold && !s->trial) { s->k--; s->downs++; s->trial = FH_RUN; s->fit_run = 0; }
    return ontime;
}
