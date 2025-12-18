/* ARM NEON kernels. Compiled out on x86 (this repo is tested on x86-64 only; see README).
 * `make neon-check` type-checks this file with clang --target=aarch64. */
#include "kernels.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static float dot_f32_neon(const float *a, const float *b, int64_t n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline int32_t dot_i8x32(const int8_t *w, const int8_t *x) {
    int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
    int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
    p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));
    int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
    p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
    return vaddlvq_s16(p0) + vaddlvq_s16(p1);
}

static float dot_q8_neon(const qm_block_q8 *w, const qm_block_q8 *x, int64_t nb) {
    float s = 0;
    for (int64_t b = 0; b < nb; b++)
        s += (float)dot_i8x32(w[b].qs, x[b].qs) * (qm_fp16_to_fp32(w[b].d) * qm_fp16_to_fp32(x[b].d));
    return s;
}

static float dot_q4_neon(const qm_block_q4 *w, const qm_block_q8 *x, int64_t nb) {
    float s = 0;
    for (int64_t b = 0; b < nb; b++) {
        uint8x16_t bytes = vld1q_u8(w[b].qs);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0xf))), vdupq_n_s8(8));
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), vdupq_n_s8(8));
        int8x16x2_t z = vzipq_s8(lo, hi); /* interleave -> element order */
        int8_t tmp[32];
        vst1q_s8(tmp, z.val[0]);
        vst1q_s8(tmp + 16, z.val[1]);
        s += (float)dot_i8x32(tmp, x[b].qs) * (qm_fp16_to_fp32(w[b].d) * qm_fp16_to_fp32(x[b].d));
    }
    return s;
}

static void micro_neon(const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc, int64_t kc) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) C[i * ldc + j] += dot_f32_neon(A + i * lda, B + j * ldb, kc);
}

#define QM_MICRO_4x4 micro_neon
#define QM_DOTF dot_f32_neon
#define QM_GEMM_FN gemm_nt_neon
#include "gemm_driver.h"

const qm_kernels qm_kernels_neon = {QM_ISA_NEON, dot_f32_neon, dot_q8_neon, dot_q4_neon, gemm_nt_neon};
#else
typedef int qm_neon_unused;
#endif
