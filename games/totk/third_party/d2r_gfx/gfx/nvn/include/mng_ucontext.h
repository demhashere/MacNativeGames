/* Signal context registers on both hosts: Darwin's mcontext pointer (__ss / __es) and Linux's in-place mcontext
   (regs[] / sp / pc, the fault's ESR in an esr_context record of __reserved). */
#pragma once
#include <stdint.h>
#include <signal.h>
#include <sys/ucontext.h>
#ifdef __APPLE__
#define UC_PC(u) ((u)->uc_mcontext->__ss.__pc)
#define UC_LR(u) ((u)->uc_mcontext->__ss.__lr)
#define UC_FP(u) ((u)->uc_mcontext->__ss.__fp)
#define UC_SP(u) ((u)->uc_mcontext->__ss.__sp)
#define UC_X(u) ((u)->uc_mcontext->__ss.__x)
#define UC_ESR(u) ((u)->uc_mcontext->__es.__esr)
#else
#define UC_PC(u) ((u)->uc_mcontext.pc)
#define UC_LR(u) ((u)->uc_mcontext.regs[30])
#define UC_FP(u) ((u)->uc_mcontext.regs[29])
#define UC_SP(u) ((u)->uc_mcontext.sp)
#define UC_X(u) ((uint64_t *)(u)->uc_mcontext.regs)
/* arm64 sigcontext records (asm/sigcontext.h): {magic, size} headers in __reserved, ESR_MAGIC carries the fault's ESR. */
static inline uint64_t mng_uc_esr(const ucontext_t *u) {
    const uint8_t *p = (const uint8_t *)u->uc_mcontext.__reserved, *end = p + sizeof u->uc_mcontext.__reserved;
    while (p + 8 <= end) {
        uint32_t magic, size; __builtin_memcpy(&magic, p, 4); __builtin_memcpy(&size, p + 4, 4);
        if (!magic || size < 8 || p + size > end) break;
        if (magic == 0x45535201u && size >= 16) { uint64_t esr; __builtin_memcpy(&esr, p + 8, 8); return esr; }
        p += size;
    }
    return 0;
}
#define UC_ESR(u) mng_uc_esr(u)
#endif
