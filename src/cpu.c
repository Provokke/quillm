/* Runtime CPU feature detection via cpuid + xgetbv. */
#include <string.h>

#include "qm_base.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>

static uint64_t read_xcr0(void) {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

typedef struct {
    int avx2, fma, f16c, avx512f, avx512bw, avx512vl, os_ymm, os_zmm;
} x86_feats;

static x86_feats detect(void) {
    x86_feats f;
    memset(&f, 0, sizeof f);
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return f;
    int osxsave = (c >> 27) & 1;
    f.fma = (c >> 12) & 1;
    f.f16c = (c >> 29) & 1;
    if (osxsave) {
        uint64_t xcr0 = read_xcr0();
        f.os_ymm = (xcr0 & 0x6) == 0x6;     /* SSE + AVX state */
        f.os_zmm = (xcr0 & 0xe6) == 0xe6;   /* + opmask, ZMM_Hi256, Hi16_ZMM */
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        f.avx2 = (b >> 5) & 1;
        f.avx512f = (b >> 16) & 1;
        f.avx512bw = (b >> 30) & 1;
        f.avx512vl = (b >> 31) & 1;
    }
    return f;
}

int qm_cpu_supports(qm_isa isa) {
    x86_feats f = detect();
    switch (isa) {
    case QM_ISA_SCALAR: return 1;
    case QM_ISA_AVX2: return f.os_ymm && f.avx2 && f.fma && f.f16c;
    case QM_ISA_AVX512:
        return f.os_ymm && f.os_zmm && f.avx2 && f.fma && f.f16c && f.avx512f && f.avx512bw && f.avx512vl;
    default: return 0;
    }
}
#elif defined(__aarch64__)
int qm_cpu_supports(qm_isa isa) { return isa == QM_ISA_SCALAR || isa == QM_ISA_NEON; }
#else
int qm_cpu_supports(qm_isa isa) { return isa == QM_ISA_SCALAR; }
#endif

qm_isa qm_cpu_best(void) {
    if (qm_cpu_supports(QM_ISA_AVX512)) return QM_ISA_AVX512;
    if (qm_cpu_supports(QM_ISA_AVX2)) return QM_ISA_AVX2;
    if (qm_cpu_supports(QM_ISA_NEON)) return QM_ISA_NEON;
    return QM_ISA_SCALAR;
}

static const char *names[QM_ISA_COUNT] = {"scalar", "avx2", "avx512", "neon"};

const char *qm_isa_name(qm_isa isa) { return (isa >= 0 && isa < QM_ISA_COUNT) ? names[isa] : "?"; }

int qm_isa_from_name(const char *s) {
    for (int i = 0; i < QM_ISA_COUNT; i++)
        if (strcmp(s, names[i]) == 0) return i;
    return -1;
}
