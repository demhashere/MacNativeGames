/* Linux (Asahi, arm64) stand-ins for the Darwin APIs the shared runtime and NVN layer call. Each one does the same job
   with Linux primitives; a call with no Linux equivalent fails the way the Darwin call does when it is unavailable
   (KERN_FAILURE), and every caller already handles that. Included through the compat headers named like the Apple ones
   (mach/mach.h, CommonCrypto/CommonDigest.h, ...), so the shared sources keep their includes. */
#pragma once
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <linux/fs.h>

/* ---------- time: one clock for mach_absolute_time, CLOCK_UPTIME_RAW and the runtime's own CLOCK_MONOTONIC ---------- */
#ifndef CLOCK_UPTIME_RAW
#define CLOCK_UPTIME_RAW CLOCK_MONOTONIC
#endif
#ifndef CLOCK_UPTIME_RAW_APPROX
#define CLOCK_UPTIME_RAW_APPROX CLOCK_MONOTONIC
#endif
static inline uint64_t clock_gettime_nsec_np(clockid_t clk) {
    struct timespec ts; if (clock_gettime(clk, &ts)) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
typedef struct { uint32_t numer, denom; } mach_timebase_info_data_t;
typedef mach_timebase_info_data_t *mach_timebase_info_t;
static inline int mach_timebase_info(mach_timebase_info_data_t *i) { i->numer = 1; i->denom = 1; return 0; }
static inline uint64_t mach_absolute_time(void) { return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
static inline uint64_t mach_approximate_time(void) { return clock_gettime_nsec_np(CLOCK_MONOTONIC_COARSE); }
static inline uint64_t mach_continuous_time(void) { return clock_gettime_nsec_np(CLOCK_BOOTTIME); }
static inline int mach_wait_until(uint64_t deadline) {
    struct timespec ts = {(time_t)(deadline / 1000000000ull), (long)(deadline % 1000000000ull)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    return 0;
}

/* ---------- Mach types and the calls that have a Linux equivalent ---------- */
typedef int kern_return_t;
#define KERN_SUCCESS 0
#define KERN_INVALID_ADDRESS 1
#define KERN_PROTECTION_FAILURE 2
#define KERN_NO_SPACE 3
#define KERN_INVALID_ARGUMENT 4
#define KERN_FAILURE 5
#define KERN_NOT_SUPPORTED 46
typedef unsigned int natural_t;
typedef int integer_t;
typedef natural_t mach_msg_type_number_t;
typedef natural_t mach_port_t;
typedef mach_port_t task_t, thread_t, thread_act_t, vm_map_t, host_t, mem_entry_name_port_t, host_name_port_t, processor_flavor_t_port;
typedef mach_port_t *thread_act_array_t;
#define MACH_PORT_NULL 0u
#define MEMORY_OBJECT_NULL 0u
typedef uint64_t mach_vm_address_t, mach_vm_size_t, mach_vm_offset_t, memory_object_offset_t;
typedef uintptr_t vm_address_t, vm_size_t, vm_offset_t;
typedef int vm_prot_t;
typedef unsigned int vm_inherit_t;
typedef int boolean_t;
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define VM_PROT_NONE 0
#define VM_PROT_READ 1
#define VM_PROT_WRITE 2
#define VM_PROT_EXECUTE 4
#define VM_PROT_DEFAULT 3
#define VM_PROT_ALL 7
#define VM_INHERIT_SHARE 0u
#define VM_INHERIT_COPY 1u
#define VM_INHERIT_NONE 2u
#define VM_INHERIT_DEFAULT VM_INHERIT_COPY
#define VM_FLAGS_FIXED 0
#define VM_FLAGS_ANYWHERE 1
#define VM_FLAGS_OVERWRITE 0x4000
#define vm_page_size ((vm_size_t)sysconf(_SC_PAGESIZE))
static inline mach_port_t mach_task_self(void) { return 1; }
static inline mach_port_t mach_host_self(void) { return 1; }
static inline mach_port_t mach_thread_self(void) { return (mach_port_t)syscall(SYS_gettid); }
static inline kern_return_t mach_port_deallocate(mach_port_t t, mach_port_t p) { (void)t; (void)p; return KERN_SUCCESS; }
static inline int mng_prot(vm_prot_t p) { return (p & VM_PROT_READ ? PROT_READ : 0) | (p & VM_PROT_WRITE ? PROT_WRITE : 0) | (p & VM_PROT_EXECUTE ? PROT_EXEC : 0); }
static inline kern_return_t mach_vm_allocate(vm_map_t t, mach_vm_address_t *a, mach_vm_size_t s, int f) {
    (void)t;
    int fl = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (!(f & VM_FLAGS_ANYWHERE)) fl |= (f & VM_FLAGS_OVERWRITE) ? MAP_FIXED : MAP_FIXED_NOREPLACE;
    void *p = mmap((f & VM_FLAGS_ANYWHERE) ? NULL : (void *)(uintptr_t)*a, (size_t)s, PROT_READ | PROT_WRITE, fl, -1, 0);
    if (p == MAP_FAILED) return KERN_NO_SPACE;
    *a = (uintptr_t)p; return KERN_SUCCESS;
}
static inline kern_return_t mach_vm_deallocate(vm_map_t t, mach_vm_address_t a, mach_vm_size_t s) {
    (void)t; return munmap((void *)(uintptr_t)a, (size_t)s) ? KERN_INVALID_ADDRESS : KERN_SUCCESS;
}
static inline kern_return_t vm_deallocate(vm_map_t t, vm_address_t a, vm_size_t s) { return mach_vm_deallocate(t, a, s); }
static inline kern_return_t mach_vm_protect(vm_map_t t, mach_vm_address_t a, mach_vm_size_t s, boolean_t set_max, vm_prot_t p) {
    (void)t; (void)set_max; return mprotect((void *)(uintptr_t)a, (size_t)s, mng_prot(p)) ? KERN_PROTECTION_FAILURE : KERN_SUCCESS;
}
/* Anonymous reservations only (the runtime's guest regions; memfd-backed regions are mapped by services.c itself). */
static inline kern_return_t mach_vm_map(vm_map_t t, mach_vm_address_t *a, mach_vm_size_t s, mach_vm_offset_t mask, int f,
                                        mem_entry_name_port_t obj, memory_object_offset_t off, boolean_t copy,
                                        vm_prot_t cur, vm_prot_t max, vm_inherit_t inh) {
    (void)t; (void)mask; (void)off; (void)copy; (void)max; (void)inh;
    if (obj != MEMORY_OBJECT_NULL) return KERN_NOT_SUPPORTED;
    int fl = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (!(f & VM_FLAGS_ANYWHERE)) fl |= (f & VM_FLAGS_OVERWRITE) ? MAP_FIXED : MAP_FIXED_NOREPLACE;
    void *p = mmap((f & VM_FLAGS_ANYWHERE) ? NULL : (void *)(uintptr_t)*a, (size_t)s, mng_prot(cur), fl, -1, 0);
    if (p == MAP_FAILED) return KERN_NO_SPACE;
    if (!(f & VM_FLAGS_ANYWHERE) && (uintptr_t)p != *a) { munmap(p, (size_t)s); return KERN_NO_SPACE; }
    *a = (uintptr_t)p; return KERN_SUCCESS;
}
/* Copy-on-write / shared remaps need shared mappings on Linux; the callers that rely on them are ported explicitly
   (totk_vpool.c) and the iOS embed path is never built here. */
static inline kern_return_t mach_vm_remap(vm_map_t t, mach_vm_address_t *a, mach_vm_size_t s, mach_vm_offset_t mask, int f,
                                          vm_map_t src, mach_vm_address_t src_a, boolean_t copy, vm_prot_t *cur,
                                          vm_prot_t *max, vm_inherit_t inh) {
    (void)t; (void)a; (void)s; (void)mask; (void)f; (void)src; (void)src_a; (void)copy; (void)cur; (void)max; (void)inh;
    return KERN_NOT_SUPPORTED;
}
/* A read that may hit unmapped memory: the kernel reports it instead of faulting (process_vm_readv on ourselves). */
static inline kern_return_t mach_vm_read_overwrite(vm_map_t t, mach_vm_address_t a, mach_vm_size_t s, mach_vm_address_t data,
                                                   mach_vm_size_t *out) {
    (void)t;
    struct iovec l = {(void *)(uintptr_t)data, (size_t)s}, r = {(void *)(uintptr_t)a, (size_t)s};
    ssize_t n = syscall(SYS_process_vm_readv, (long)getpid(), &l, 1ul, &r, 1ul, 0ul);
    if (out) *out = n > 0 ? (mach_vm_size_t)n : 0;
    return n == (ssize_t)s ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}
static inline kern_return_t vm_read_overwrite(vm_map_t t, vm_address_t a, vm_size_t s, vm_address_t data, vm_size_t *out) {
    mach_vm_size_t n = 0; kern_return_t r = mach_vm_read_overwrite(t, a, s, data, &n); if (out) *out = (vm_size_t)n; return r;
}

/* vm_region: one /proc/self/maps line at or after the address, read with raw syscalls (also used by crash reports). */
typedef int vm_region_flavor_t;
typedef int *vm_region_info_t;
typedef int *vm_region_recurse_info_t;
#define VM_REGION_BASIC_INFO_64 9
typedef struct {
    vm_prot_t protection, max_protection; vm_inherit_t inheritance; boolean_t shared, reserved;
    memory_object_offset_t offset; int behavior; unsigned short user_wired_count;
} vm_region_basic_info_data_64_t;
#define VM_REGION_BASIC_INFO_COUNT_64 ((mach_msg_type_number_t)(sizeof(vm_region_basic_info_data_64_t) / sizeof(int)))
typedef struct { unsigned pages_resident, pages_dirtied, pages_swapped_out; } vm_region_submap_info_data_64_t;
#define VM_REGION_SUBMAP_INFO_COUNT_64 ((mach_msg_type_number_t)(sizeof(vm_region_submap_info_data_64_t) / sizeof(int)))
static inline unsigned long long mng_hex(const char **p) {
    unsigned long long v = 0;
    for (;;) { char c = **p; int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; if (d < 0) break; v = v * 16 + (unsigned)d; (*p)++; }
    return v;
}
static inline kern_return_t mach_vm_region(vm_map_t t, mach_vm_address_t *a, mach_vm_size_t *s, vm_region_flavor_t flavor,
                                           vm_region_info_t info, mach_msg_type_number_t *cnt, mach_port_t *obj) {
    (void)t; (void)flavor; (void)cnt;
    if (obj) *obj = MACH_PORT_NULL;
    int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return KERN_FAILURE;
    char buf[8192]; size_t have = 0; int eof = 0; kern_return_t kr = KERN_INVALID_ADDRESS;
    for (;;) {
        if (!eof && have < sizeof buf - 1) { ssize_t n = read(fd, buf + have, sizeof buf - 1 - have); if (n <= 0) eof = 1; else have += (size_t)n; }
        char *nl = memchr(buf, '\n', have);
        if (!nl) { if (eof) break; if (have == sizeof buf - 1) have = 0; continue; }
        *nl = 0;
        const char *p = buf; unsigned long long lo = mng_hex(&p); if (*p == '-') p++; unsigned long long hi = mng_hex(&p);
        while (*p == ' ') p++;
        if (hi > *a) {
            vm_region_basic_info_data_64_t *bi = (vm_region_basic_info_data_64_t *)info;
            if (bi) {
                memset(bi, 0, sizeof *bi);
                bi->protection = (p[0] == 'r' ? VM_PROT_READ : 0) | (p[1] == 'w' ? VM_PROT_WRITE : 0) | (p[2] == 'x' ? VM_PROT_EXECUTE : 0);
                bi->max_protection = VM_PROT_ALL; bi->shared = p[3] == 's';
            }
            *a = lo; *s = hi - lo; kr = KERN_SUCCESS; break;
        }
        size_t used = (size_t)(nl + 1 - buf); memmove(buf, nl + 1, have - used); have -= used;
    }
    close(fd);
    return kr;
}
static inline kern_return_t vm_region_64(vm_map_t t, vm_address_t *a, vm_size_t *s, vm_region_flavor_t flavor, vm_region_info_t info,
                                         mach_msg_type_number_t *cnt, mach_port_t *obj) {
    mach_vm_address_t ma = *a; mach_vm_size_t ms = 0; kern_return_t r = mach_vm_region(t, &ma, &ms, flavor, info, cnt, obj);
    *a = (vm_address_t)ma; *s = (vm_size_t)ms; return r;
}
static inline kern_return_t vm_region_recurse_64(vm_map_t t, vm_address_t *a, vm_size_t *s, natural_t *depth, vm_region_recurse_info_t info,
                                                 mach_msg_type_number_t *cnt) {
    (void)t; (void)a; (void)s; (void)depth; (void)info; (void)cnt; return KERN_FAILURE;
}

/* task / thread / host introspection: not available (memory reports, CPU-time lines and sampling just stay empty) */
typedef int task_flavor_t, thread_flavor_t, host_flavor_t, processor_flavor_t, thread_policy_flavor_t, thread_state_flavor_t;
typedef integer_t *task_info_t, *thread_info_t, *thread_policy_t, *processor_info_array_t, *thread_state_t;
typedef struct { uint64_t virtual_size; integer_t region_count, page_size; uint64_t resident_size, resident_size_peak, device, device_peak,
                 internal, internal_peak, external, external_peak, reusable, reusable_peak, purgeable_volatile_pmap, purgeable_volatile_resident,
                 purgeable_volatile_virtual, compressed, compressed_peak, compressed_lifetime, phys_footprint, min_address, max_address; } task_vm_info_data_t;
#define TASK_VM_INFO 22
#define TASK_VM_INFO_COUNT ((mach_msg_type_number_t)(sizeof(task_vm_info_data_t) / sizeof(natural_t)))
static inline kern_return_t task_info(task_t t, task_flavor_t f, task_info_t i, mach_msg_type_number_t *n) { (void)t; (void)f; (void)i; (void)n; return KERN_FAILURE; }
static inline kern_return_t task_threads(task_t t, thread_act_array_t *l, mach_msg_type_number_t *n) { (void)t; (void)l; if (n) *n = 0; return KERN_FAILURE; }
static inline kern_return_t thread_suspend(thread_act_t t) { (void)t; return KERN_FAILURE; }
static inline kern_return_t thread_resume(thread_act_t t) { (void)t; return KERN_FAILURE; }
static inline kern_return_t thread_info(thread_act_t t, thread_flavor_t f, thread_info_t i, mach_msg_type_number_t *n) { (void)t; (void)f; (void)i; (void)n; return KERN_FAILURE; }
static inline kern_return_t thread_get_state(thread_act_t t, thread_state_flavor_t f, thread_state_t s, mach_msg_type_number_t *n) { (void)t; (void)f; (void)s; (void)n; return KERN_FAILURE; }
static inline kern_return_t thread_policy_set(thread_act_t t, thread_policy_flavor_t f, thread_policy_t p, mach_msg_type_number_t n) { (void)t; (void)f; (void)p; (void)n; return KERN_FAILURE; }
static inline kern_return_t thread_policy_get(thread_act_t t, thread_policy_flavor_t f, thread_policy_t p, mach_msg_type_number_t *n, boolean_t *d) { (void)t; (void)f; (void)p; (void)n; (void)d; return KERN_FAILURE; }
static inline kern_return_t host_processor_info(host_t h, processor_flavor_t f, natural_t *n, processor_info_array_t *i, mach_msg_type_number_t *c) {
    (void)h; (void)f; (void)n; (void)i; (void)c; return KERN_FAILURE;
}
#define PROCESSOR_CPU_LOAD_INFO 2
typedef struct { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; } arm_thread_state64_t;
#define ARM_THREAD_STATE64 6
#define ARM_THREAD_STATE64_COUNT ((mach_msg_type_number_t)(sizeof(arm_thread_state64_t) / sizeof(uint32_t)))
#define arm_thread_state64_get_pc(s) ((s).__pc)
#define arm_thread_state64_get_lr(s) ((s).__lr)
#define arm_thread_state64_get_fp(s) ((s).__fp)
#define arm_thread_state64_get_sp(s) ((s).__sp)
/* thread_policy.h */
#define THREAD_TIME_CONSTRAINT_POLICY 2
#define THREAD_PRECEDENCE_POLICY 3
#define THREAD_AFFINITY_POLICY 4
typedef struct { uint32_t period, computation, constraint; boolean_t preemptible; } thread_time_constraint_policy_data_t;
#define THREAD_TIME_CONSTRAINT_POLICY_COUNT ((mach_msg_type_number_t)(sizeof(thread_time_constraint_policy_data_t) / sizeof(integer_t)))
typedef struct { integer_t importance; } thread_precedence_policy_data_t;
#define THREAD_PRECEDENCE_POLICY_COUNT ((mach_msg_type_number_t)1)
typedef struct { integer_t affinity_tag; } thread_affinity_policy_data_t;
#define THREAD_AFFINITY_POLICY_COUNT ((mach_msg_type_number_t)1)
typedef struct { integer_t pth_curpri, pth_priority, pth_policy; } thread_extended_info_data_t;
#define THREAD_EXTENDED_INFO 5
#define THREAD_EXTENDED_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_extended_info_data_t) / sizeof(natural_t)))
typedef struct { integer_t seconds, microseconds; } time_value_t;
typedef struct { time_value_t user_time, system_time; integer_t cpu_usage, policy, run_state, flags, suspend_count, sleep_time; } thread_basic_info_data_t;
#define THREAD_BASIC_INFO 3
#define THREAD_BASIC_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_basic_info_data_t) / sizeof(natural_t)))

/* ---------- pthread extras ---------- */
static inline mach_port_t pthread_mach_thread_np(pthread_t t) { (void)t; return MACH_PORT_NULL; }
static inline int mng_threadid(uintptr_t t, uint64_t *id) {
    if (!id) return EINVAL;
    *id = (!t || pthread_equal((pthread_t)t, pthread_self())) ? (uint64_t)syscall(SYS_gettid) : (uint64_t)t;
    return 0;
}
#define pthread_threadid_np(t, id) mng_threadid((uintptr_t)(t), (id))
static inline int pthread_main_np(void) { return getpid() == (pid_t)syscall(SYS_gettid); }
static inline int pthread_cond_timedwait_relative_np(pthread_cond_t *cv, pthread_mutex_t *mu, const struct timespec *rel) {
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += rel->tv_sec; dl.tv_nsec += rel->tv_nsec;
    if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }
    return pthread_cond_timedwait(cv, mu, &dl);
}
/* Darwin names the calling thread with one argument; Linux keeps 15 characters. */
static inline int mng_setname_self(const char *name) {
    char n[16]; snprintf(n, sizeof n, "%s", name ? name : ""); return pthread_setname_np(pthread_self(), n);
}
#define pthread_setname_np(name) mng_setname_self(name)

/* ---------- QoS classes: the interactive ones ask the scheduler for a fast core (uclamp where the kernel has it) ---------- */
typedef enum {
    QOS_CLASS_USER_INTERACTIVE = 0x21, QOS_CLASS_USER_INITIATED = 0x19, QOS_CLASS_DEFAULT = 0x15,
    QOS_CLASS_UTILITY = 0x11, QOS_CLASS_BACKGROUND = 0x09, QOS_CLASS_UNSPECIFIED = 0x00,
} qos_class_t;
int mng_qos_apply(qos_class_t q, int relpri);   /* linux/platform.c */
extern __thread qos_class_t mng_qos_current;
static inline int pthread_set_qos_class_self_np(qos_class_t q, int relpri) { return mng_qos_apply(q, relpri); }
static inline int pthread_get_qos_class_np(pthread_t t, qos_class_t *q, int *relpri) {
    (void)t; if (q) *q = mng_qos_current; if (relpri) *relpri = 0; return 0;
}
static inline qos_class_t qos_class_self(void) { return mng_qos_current; }
static inline int pthread_attr_set_qos_class_np(pthread_attr_t *a, qos_class_t q, int relpri) { (void)a; (void)q; (void)relpri; return 0; }

/* ---------- caches ---------- */
static inline void sys_icache_invalidate(void *start, size_t len) { __builtin___clear_cache((char *)start, (char *)start + len); }
static inline void sys_dcache_flush(void *start, size_t len) {
    uintptr_t a = (uintptr_t)start & ~(uintptr_t)63, e = (uintptr_t)start + len;
    for (; a < e; a += 64) __asm__ volatile("dc civac, %0" :: "r"(a) : "memory");
    __asm__ volatile("dsb ish" ::: "memory");
}

/* ---------- malloc ---------- */
static inline size_t malloc_size(const void *p) { return p ? malloc_usable_size((void *)p) : 0; }

/* ---------- executable path ---------- */
static inline int _NSGetExecutablePath(char *buf, uint32_t *size) {
    ssize_t n = readlink("/proc/self/exe", buf, *size ? *size - 1 : 0);
    if (n < 0 || (uint32_t)n >= *size - 1) { *size = 4096; return -1; }
    buf[n] = 0; return 0;
}

/* ---------- APFS clones: btrfs / XFS reflinks, else a plain copy ---------- */
#define CLONE_NOFOLLOW 0x0001
#define CLONE_NOOWNERCOPY 0x0002
static inline int clonefile(const char *src, const char *dst, uint32_t flags) {
    (void)flags;
    int in = open(src, O_RDONLY | O_CLOEXEC); if (in < 0) return -1;
    struct stat st; if (fstat(in, &st)) { close(in); return -1; }
    int out = open(dst, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, st.st_mode & 0777); if (out < 0) { close(in); return -1; }
    int ok = ioctl(out, FICLONE, in) == 0;
    if (!ok) {
        off_t left = st.st_size; ok = 1;
        while (left > 0) { ssize_t n = copy_file_range(in, NULL, out, NULL, (size_t)left, 0); if (n <= 0) { ok = 0; break; } left -= n; }
    }
    close(in); if (close(out)) ok = 0;
    if (!ok) { unlink(dst); errno = EIO; return -1; }
    return 0;
}

/* ---------- sysctlbyname: the keys the runtime reads ---------- */
int mng_sysctlbyname(const char *name, void *out, size_t *len, const void *newp, size_t newlen);   /* linux/platform.c */
#define sysctlbyname mng_sysctlbyname

/* ---------- Mach-O sections: ELF sections found through the linker's __start_ / __stop_ symbols ---------- */
#define MNG_SECTIONS(X) X(swnvn) X(totknvnx) X(totkgfx)
#define MNG_SECT_DECL(n) extern const char __start_##n[] __attribute__((weak)); extern const char __stop_##n[] __attribute__((weak));
MNG_SECTIONS(MNG_SECT_DECL)
#undef MNG_SECT_DECL
static inline const uint8_t *mng_getsectiondata(const char *seg, const char *sect, unsigned long *size) {
    (void)seg;
#define MNG_SECT_FIND(n) if (!strcmp(sect, #n)) { *size = __start_##n ? (unsigned long)(__stop_##n - __start_##n) : 0; return (const uint8_t *)__start_##n; }
    MNG_SECTIONS(MNG_SECT_FIND)
#undef MNG_SECT_FIND
    *size = 0; return NULL;
}
#define getsectiondata(mh, seg, sect, size) mng_getsectiondata((seg), (sect), (size))

/* ---------- struct stat: Darwin's timespec member names (birth time: Linux keeps it only in statx; the change time
   stands in, which also changes whenever the file is replaced) ---------- */
#define st_mtimespec st_mtim
#define st_ctimespec st_ctim
#define st_atimespec st_atim
#define st_birthtimespec st_ctim

/* ---------- thread stacks: Darwin returns the top (highest address) and the size ---------- */
static inline void *pthread_get_stackaddr_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t sz = 0;
    if (pthread_getattr_np(t, &a)) return NULL;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return (char *)lo + sz;
}
static inline size_t pthread_get_stacksize_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t sz = 0;
    if (pthread_getattr_np(t, &a)) return 0;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return sz;
}
