/* Linux pieces behind the Darwin stand-ins (linux/compat/darwin_linux.h): thread QoS, sysctlbyname keys, the time
   function the assembly calls, and the per-thread guest slots the rewritten game code reads through TPIDR_EL0. */
#include "compat/darwin_linux.h"
#include <dirent.h>
#include <linux/sched.h>
#include <stdatomic.h>

/* ---------- guest thread slots ----------
   The rewritten modules read their per-thread slots (K18, KTLS, KTLR, KGATE: the guest's x18, its TLS pointer, its
   ThreadLocalRegion and the gate) at thread pointer + offset, where the offset is read from the module's key page. On
   Darwin the thread pointer is TPIDRRO_EL0 (the TSD base) and the offsets are pthread key slots; Linux user space cannot
   use TPIDRRO_EL0, so the loader turns those reads into TPIDR_EL0 reads (glibc's thread pointer) and the slots are
   initial-exec TLS of this executable, which glibc places at the same offset from the thread pointer in every thread.
   Slot 4 is the host-stack slot of runtime/hle_stack.S. */
__thread uint64_t mng_slots[8] __attribute__((tls_model("initial-exec"), aligned(64)));
uint64_t mng_slot_offset(unsigned i) { return (uint64_t)((uintptr_t)&mng_slots[i] - (uintptr_t)__builtin_thread_pointer()); }
uint64_t mng_mach_absolute_time(void) { return mach_absolute_time(); }

/* ---------- QoS ----------
   Darwin's interactive classes keep the game's threads on performance cores at full clock. Linux has no QoS; with
   utilization clamping (CONFIG_UCLAMP_TASK) a minimum utilization asks the energy-aware scheduler for a big core and a
   high frequency. Without it the class is only remembered. TOTK_LINUX_UCLAMP=0 turns this off. */
__thread qos_class_t mng_qos_current = QOS_CLASS_UNSPECIFIED;
struct mng_sched_attr { uint32_t size, sched_policy; uint64_t sched_flags; int32_t sched_nice; uint32_t sched_priority;
                        uint64_t sched_runtime, sched_deadline, sched_period; uint32_t sched_util_min, sched_util_max; };
#ifndef SCHED_FLAG_KEEP_POLICY
#define SCHED_FLAG_KEEP_POLICY 0x08
#define SCHED_FLAG_KEEP_PARAMS 0x10
#define SCHED_FLAG_UTIL_CLAMP_MIN 0x20
#define SCHED_FLAG_UTIL_CLAMP_MAX 0x40
#endif
static _Atomic int g_uclamp = -1;
int mng_qos_apply(qos_class_t q, int relpri) {
    (void)relpri;
    mng_qos_current = q;
    int on = atomic_load(&g_uclamp);
    if (on < 0) { const char *e = getenv("TOTK_LINUX_UCLAMP"); on = !(e && !strcmp(e, "0")); atomic_store(&g_uclamp, on); }
    if (!on) return 0;
    uint32_t umin = q == QOS_CLASS_USER_INTERACTIVE ? 768 : q == QOS_CLASS_USER_INITIATED ? 512 : 0;
    struct mng_sched_attr a = {.size = sizeof a, .sched_flags = SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MIN | SCHED_FLAG_UTIL_CLAMP_MAX,
                               .sched_util_min = umin, .sched_util_max = 1024};
    if (syscall(SYS_sched_setattr, 0, &a, 0) && errno != EINVAL && errno != EPERM) return 0;
    return 0;
}

/* ---------- sysctlbyname: the keys the shared code reads ---------- */
static int cpu_capacity(int cpu) {
    char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
    FILE *f = fopen(p, "r"); int v = -1; if (f) { if (fscanf(f, "%d", &v) != 1) v = -1; fclose(f); }
    return v;
}
/* performance cores (perflevel0) = the CPUs with the highest capacity, efficiency cores (perflevel1) the rest */
static void core_levels(int *perf, int *eff) {
    int n = (int)sysconf(_SC_NPROCESSORS_CONF), best = -1; *perf = *eff = 0;
    for (int i = 0; i < n; i++) { int c = cpu_capacity(i); if (c > best) best = c; }
    for (int i = 0; i < n; i++) { int c = cpu_capacity(i); if (best <= 0 || c == best) (*perf)++; else (*eff)++; }
}
static int put_int(void *out, size_t *len, uint64_t v) {
    if (!len) return -1;
    if (!out) { *len = *len == 4 ? 4 : 8; return 0; }
    if (*len == 4) { uint32_t w = (uint32_t)v; memcpy(out, &w, 4); return 0; }
    if (*len >= 8) { memcpy(out, &v, 8); *len = 8; return 0; }
    errno = ENOMEM; return -1;
}
int mng_sysctlbyname(const char *name, void *out, size_t *len, const void *newp, size_t newlen) {
    (void)newp; (void)newlen;
    if (!strcmp(name, "hw.memsize") || !strcmp(name, "hw.physmem")) { struct sysinfo si; if (sysinfo(&si)) return -1; return put_int(out, len, (uint64_t)si.totalram * si.mem_unit); }
    if (!strcmp(name, "hw.ncpu") || !strcmp(name, "hw.logicalcpu") || !strcmp(name, "hw.physicalcpu") || !strcmp(name, "hw.activecpu"))
        return put_int(out, len, (uint64_t)sysconf(_SC_NPROCESSORS_ONLN));
    if (!strcmp(name, "hw.perflevel0.logicalcpu") || !strcmp(name, "hw.perflevel1.logicalcpu") || !strcmp(name, "hw.nperflevels")) {
        int p, e; core_levels(&p, &e);
        return put_int(out, len, !strcmp(name, "hw.nperflevels") ? (uint64_t)(e ? 2 : 1) : !strcmp(name, "hw.perflevel0.logicalcpu") ? (uint64_t)p : (uint64_t)e);
    }
    if (!strcmp(name, "hw.cachelinesize")) return put_int(out, len, 128);
    if (!strcmp(name, "hw.pagesize")) return put_int(out, len, (uint64_t)sysconf(_SC_PAGESIZE));
    if (!strcmp(name, "kern.osproductversion") || !strcmp(name, "kern.osversion") || !strcmp(name, "hw.model") || !strcmp(name, "machdep.cpu.brand_string")) {
        const char *v = !strcmp(name, "hw.model") ? "Linux arm64" : !strcmp(name, "machdep.cpu.brand_string") ? "Apple silicon (Asahi Linux)" : "Linux";
        size_t n = strlen(v) + 1; if (!out) { *len = n; return 0; } if (*len < n) { errno = ENOMEM; return -1; } memcpy(out, v, n); *len = n; return 0;
    }
    errno = ENOENT; return -1;
}

/* No core dump when the game dies (abort_report has logged the call stack): systemd-coredump copied this process's
   address space (the game's memory regions) for minutes and peaked at 4.9 GB of RAM on an 8 GB machine. TOTK_CORE_DUMP=1:
   dumps as the system is set up. */
#include <sys/prctl.h>
#include <sys/resource.h>
__attribute__((constructor)) static void no_core_dump(void) {
    const char *e = getenv("TOTK_CORE_DUMP");
    if (e && !strcmp(e, "1")) return;
    struct rlimit r = {0, 0}; setrlimit(RLIMIT_CORE, &r);
    prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
}
