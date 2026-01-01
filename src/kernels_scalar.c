/* Portable scalar reference kernels + dispatch. */
#include <stdlib.h>

#include "kernels.h"

static float dot_f32_scalar(const float *a, const float *b, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float dot_q8_scalar(const qm_block_q8 *w, const qm_block_q8 *x, int64_t nb) {
    float s = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        int32_t acc = 0;
        for (int i = 0; i < 32; i++) acc += (int32_t)w[b].qs[i] * (int32_t)x[b].qs[i];
        s += (float)acc * (qm_fp16_to_fp32(w[b].d) * qm_fp16_to_fp32(x[b].d));
    }
    return s;
}

static float dot_q4_scalar(const qm_block_q4 *w, const qm_block_q8 *x, int64_t nb) {
    float s = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        int32_t acc = 0;
        for (int j = 0; j < 16; j++) {
            int lo = (w[b].qs[j] & 0xf) - 8;
            int hi = (w[b].qs[j] >> 4) - 8;
            acc += lo * x[b].qs[2 * j] + hi * x[b].qs[2 * j + 1];
        }
        s += (float)acc * (qm_fp16_to_fp32(w[b].d) * qm_fp16_to_fp32(x[b].d));
    }
    return s;
}

static void micro_scalar(const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc,
                         int64_t kc) {
    float acc[4][4] = {{0}};
    for (int64_t k = 0; k < kc; k++)
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) acc[i][j] += A[i * lda + k] * B[j * ldb + k];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) C[i * ldc + j] += acc[i][j];
}

#define QM_MICRO_4x4 micro_scalar
#define QM_DOTF dot_f32_scalar
#define QM_GEMM_FN gemm_nt_scalar
#include "gemm_driver.h"

const qm_kernels qm_kernels_scalar = {QM_ISA_SCALAR, dot_f32_scalar, dot_q8_scalar, dot_q4_scalar, gemm_nt_scalar};

/* ---------------- dispatch ---------------- */
const qm_kernels *qm_kernels_for(qm_isa isa) {
    if (!qm_cpu_supports(isa)) return NULL;
    switch (isa) {
    case QM_ISA_SCALAR: return &qm_kernels_scalar;
#if defined(__x86_64__)
    case QM_ISA_AVX2: return &qm_kernels_avx2;
    case QM_ISA_AVX512: return &qm_kernels_avx512;
#endif
#if defined(__aarch64__)
    case QM_ISA_NEON: return &qm_kernels_neon;
#endif
    default: return NULL;
    }
}

static const qm_kernels *g_active;

int qm_kernels_select(qm_isa isa) {
    const qm_kernels *k = qm_kernels_for(isa);
    if (!k) return -1;
    g_active = k;
    return 0;
}

const qm_kernels *qm_kernels_active(void) {
    if (!g_active) {
        const char *env = getenv("QUILLM_ISA");
        int isa = env ? qm_isa_from_name(env) : -1;
        if (isa < 0 || qm_kernels_select((qm_isa)isa) != 0) qm_kernels_select(qm_cpu_best());
    }
    return g_active;
}
