/* The frame hold's decisions (nvn_fhold.h) on made-up frame costs. make -f linux/Makefile OUT=build/linux-test fhtest */
#include "nvn_fhold.h"
#include <stdio.h>
#include <stdlib.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)
/* n frames of the given cost: shown is the hold when the frame fits it, else the cost (a late frame starts the next at once) */
static int run(nvn_fhold *s, int n, double cost) {
    int ontime = 0;
    for (int i = 0; i < n; i++) { double shown = s->k && cost <= s->k ? s->k : cost; ontime += nvn_fhold_next(s, shown, cost, 1, FH_MAX); }
    return ontime;
}
int main(void) {
    int neg = getenv("TOTK_FH_NEG") != NULL;
    nvn_fhold s = {0}; nvn_fhold_mode(&s, 0);
    run(&s, 1, 0.8);
    CHECK(s.k == 1, "starts at 60 fps (k %d)", s.k);
    CHECK(run(&s, 600, 0.85) == 600, "a scene that fits 60 is on time every frame");
    run(&s, 600, 1.2);   /* 50 fps: between 44 and 60, the default mode keeps k = 1 */
    CHECK(s.k == (neg ? 2 : 1) && s.ups == 0, "50 fps stays unheld in the default mode (k %d, ups %llu)", s.k, (unsigned long long)s.ups);
    run(&s, 3, 1.6);     /* three frames under 40 fps step up */
    CHECK(s.k == 2 && s.ups == 1, "three frames at 37 fps hold 30 (k %d)", s.k);
    run(&s, 30, 1.1);    /* 30 frames that fit 60 at 30 fps: 1 s, the hold-off */
    CHECK(s.k == 1 && s.trial > 0, "a trial at 60 after 30 fitting frames and the hold-off (k %d, trial %d)", s.k, s.trial);
    run(&s, 3, 1.5);     /* the trial fails */
    CHECK(s.k == 2 && s.failed == 1 && s.hold == 2 * FH_HOLD0, "a failed trial holds 30 again and doubles the hold-off (k %d, hold %.0f)", s.k, s.hold);
    run(&s, 30, 1.0);
    CHECK(s.k == 2, "no trial before the doubled hold-off (k %d, since %.0f)", s.k, s.since);
    run(&s, 30, 1.0);
    CHECK(s.k == 1 && s.trial > 0, "a trial after the doubled hold-off (k %d)", s.k);
    run(&s, 30, 1.0);
    CHECK(s.k == 1 && s.trial == 0 && s.hold == FH_HOLD0, "a trial that holds resets the hold-off (hold %.0f)", s.hold);
    run(&s, 3, 2.6);
    CHECK(s.k == 3, "a frame cost of 2.6 vsyncs steps straight to 20 fps (k %d)", s.k);

    nvn_fhold t = {0}; nvn_fhold_mode(&t, 1);   /* strict: 60 or 30 */
    run(&t, 1, 0.8); run(&t, 3, 1.2);
    CHECK(t.k == 2, "strict: three frames at 50 fps hold 30 (k %d)", t.k);
    run(&t, 120, 0.9);
    CHECK(t.k == 2, "strict: 0.9 vsyncs does not fit 85%% of 60 (k %d)", t.k);
    run(&t, 120, 0.8);
    CHECK(t.k == 1, "strict: 0.8 vsyncs fits and tries 60 (k %d)", t.k);
    printf("%s: frame hold decisions (%llu frames, %llu on time, %llu up, %llu down, %llu failed trials)\n", g_fail ? "FAILED" : "passed",
           (unsigned long long)s.frames, (unsigned long long)s.ontime, (unsigned long long)s.ups, (unsigned long long)s.downs, (unsigned long long)s.failed);
    return g_fail;
}
