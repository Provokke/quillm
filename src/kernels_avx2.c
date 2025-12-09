/* AVX2 + FMA + F16C kernels. Compiled with -mavx2 -mfma -mf16c; only called after cpuid check. */
#include "kernels.h"

#if defined(__x86_64__)
#include <immintrin.h>

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

static float dot_f32_avx2(const float *a, const float *b, int64_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* int8 x int8 -> 8 x int32 partial sums using the sign trick (|w| * sign(x,w)) */
static inline __m256i mul_i8(__m256i w, __m256i x) {
    __m256i aw = _mm256_sign_epi8(w, w);
    __m256i sx = _mm256_sign_epi8(x, w);
    __m256i p16 = _mm256_maddubs_epi16(aw, sx); /* max |.| = 2*127*127 < 32767 */
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

static float dot_q8_avx2(const qm_block_q8 *w, const qm_block_q8 *x, int64_t nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)w[b].qs);
        __m256i xv = _mm256_loadu_si256((const __m256i *)x[b].qs);
        __m256 p = _mm256_cvtepi32_ps(mul_i8(wv, xv));
        float d = _cvtsh_ss(w[b].d) * _cvtsh_ss(x[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), p, acc);
    }
    return hsum256(acc);
}

static inline __m256i unpack_q4(const uint8_t *qs) {
    __m128i bytes = _mm_loadu_si128((const __m128i *)qs);
    __m128i m = _mm_set1_epi8(0x0f);
    __m128i lo = _mm_and_si128(bytes, m);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m);
    __m256i v = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
    return _mm256_sub_epi8(v, _mm256_set1_epi8(8));
}

static float dot_q4_avx2(const qm_block_q4 *w, const qm_block_q8 *x, int64_t nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        __m256i wv = unpack_q4(w[b].qs);
        __m256i xv = _mm256_loadu_si256((const __m256i *)x[b].qs);
        __m256 p = _mm256_cvtepi32_ps(mul_i8(wv, xv));
        float d = _cvtsh_ss(w[b].d) * _cvtsh_ss(x[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), p, acc);
    }
    return hsum256(acc);
}

static void micro_avx2(const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc, int64_t kc) {
    __m256 c[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) c[i][j] = _mm256_setzero_ps();
    int64_t k = 0;
    for (; k + 8 <= kc; k += 8) {
        __m256 b0 = _mm256_loadu_ps(B + k), b1 = _mm256_loadu_ps(B + ldb + k);
        __m256 b2 = _mm256_loadu_ps(B + 2 * ldb + k), b3 = _mm256_loadu_ps(B + 3 * ldb + k);
        for (int i = 0; i < 4; i++) {
            __m256 a = _mm256_loadu_ps(A + i * lda + k);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
            c[i][2] = _mm256_fmadd_ps(a, b2, c[i][2]);
            c[i][3] = _mm256_fmadd_ps(a, b3, c[i][3]);
        }
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = hsum256(c[i][j]);
            for (int64_t kk = k; kk < kc; kk++) s += A[i * lda + kk] * B[j * ldb + kk];
            C[i * ldc + j] += s;
        }
}

#define QM_MICRO_4x4 micro_avx2
#define QM_DOTF dot_f32_avx2
#define QM_GEMM_FN gemm_nt_avx2
#include "gemm_driver.h"

const qm_kernels qm_kernels_avx2 = {QM_ISA_AVX2, dot_f32_avx2, dot_q8_avx2, dot_q4_avx2, gemm_nt_avx2};
#else
typedef int qm_avx2_unused;
#endif
