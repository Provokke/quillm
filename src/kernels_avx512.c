/* AVX-512 (F/BW/VL) kernels. Masked loads handle tails without scalar loops. */
#include "kernels.h"

#if defined(__x86_64__)
#include <immintrin.h>

static float dot_f32_avx512(const float *a, const float *b, int64_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    for (; i < n; i += 16) {
        int64_t r = n - i;
        __mmask16 m = r >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << r) - 1u);
        s0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), s0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}

/* 32 int8 x 32 int8 -> 16 x int32 via sign-extension to int16 + vpmaddwd */
static inline __m512i mul_i8x32(__m256i w, __m256i x) {
    return _mm512_madd_epi16(_mm512_cvtepi8_epi16(w), _mm512_cvtepi8_epi16(x));
}


static float dot_q8_avx512(const qm_block_q8 *w, const qm_block_q8 *x, int64_t nb) {
    __m512 acc = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)w[b].qs);
        __m256i xv = _mm256_loadu_si256((const __m256i *)x[b].qs);
        __m512 p = _mm512_cvtepi32_ps(mul_i8x32(wv, xv));
        float d = _cvtsh_ss(w[b].d) * _cvtsh_ss(x[b].d);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), p, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float dot_q4_avx512(const qm_block_q4 *w, const qm_block_q8 *x, int64_t nb) {
    __m512 acc = _mm512_setzero_ps();
    const __m128i m = _mm_set1_epi8(0x0f);
    for (int64_t b = 0; b < nb; b++) {
        __m128i bytes = _mm_loadu_si128((const __m128i *)w[b].qs);
        __m128i lo = _mm_and_si128(bytes, m);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m);
        __m256i wv = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        wv = _mm256_sub_epi8(wv, _mm256_set1_epi8(8));
        __m256i xv = _mm256_loadu_si256((const __m256i *)x[b].qs);
        __m512 p = _mm512_cvtepi32_ps(mul_i8x32(wv, xv));
        float d = _cvtsh_ss(w[b].d) * _cvtsh_ss(x[b].d);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), p, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static void micro_avx512(const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc,
                         int64_t kc) {
    __m512 c[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) c[i][j] = _mm512_setzero_ps();
    for (int64_t k = 0; k < kc; k += 16) {
        int64_t r = kc - k;
        __mmask16 m = r >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << r) - 1u);
        __m512 b0 = _mm512_maskz_loadu_ps(m, B + k), b1 = _mm512_maskz_loadu_ps(m, B + ldb + k);
        __m512 b2 = _mm512_maskz_loadu_ps(m, B + 2 * ldb + k), b3 = _mm512_maskz_loadu_ps(m, B + 3 * ldb + k);
        for (int i = 0; i < 4; i++) {
            __m512 a = _mm512_maskz_loadu_ps(m, A + i * lda + k);
            c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
            c[i][2] = _mm512_fmadd_ps(a, b2, c[i][2]);
            c[i][3] = _mm512_fmadd_ps(a, b3, c[i][3]);
        }
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) C[i * ldc + j] += _mm512_reduce_add_ps(c[i][j]);
}

#define QM_MICRO_4x4 micro_avx512
#define QM_DOTF dot_f32_avx512
#define QM_GEMM_FN gemm_nt_avx512
#include "gemm_driver.h"

const qm_kernels qm_kernels_avx512 = {QM_ISA_AVX512, dot_f32_avx512, dot_q8_avx512, dot_q4_avx512, gemm_nt_avx512};
#else
typedef int qm_avx512_unused;
#endif
