/* Per-ISA compute kernels with runtime dispatch. */
#ifndef QM_KERNELS_H
#define QM_KERNELS_H

#include "qm_base.h"

typedef struct {
    qm_isa isa;
    /* sum_i a[i]*b[i] */
    float (*dot_f32)(const float *a, const float *b, int64_t n);
    /* dot of nb Q8_0 weight blocks with nb Q8_0 activation blocks */
    float (*dot_q8)(const qm_block_q8 *w, const qm_block_q8 *x, int64_t nb);
    /* dot of nb Q4_0 weight blocks with nb Q8_0 activation blocks */
    float (*dot_q4)(const qm_block_q4 *w, const qm_block_q8 *x, int64_t nb);
    /* C[M,N] = A[M,K] * B[N,K]^T, cache-blocked + register-tiled (single thread) */
    void (*gemm_nt)(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K, int64_t lda,
                    int64_t ldb, int64_t ldc);
} qm_kernels;

extern const qm_kernels qm_kernels_scalar;
#if defined(__x86_64__)
extern const qm_kernels qm_kernels_avx2;
extern const qm_kernels qm_kernels_avx512;
#endif
#if defined(__aarch64__)
extern const qm_kernels qm_kernels_neon;
#endif

/* kernels for an ISA, or NULL if not compiled in / not supported by this CPU */
const qm_kernels *qm_kernels_for(qm_isa isa);
/* currently selected kernels (defaults to best; env QUILLM_ISA overrides) */
const qm_kernels *qm_kernels_active(void);
/* force an ISA; returns 0 on success */
int qm_kernels_select(qm_isa isa);

/* GEMM blocking parameters shared by all ISAs */
#define QM_GEMM_NB 64  /* columns of C (rows of B) per cache block */
#define QM_GEMM_KC 256 /* depth per cache block (B panel = 64 KiB) */

#endif
