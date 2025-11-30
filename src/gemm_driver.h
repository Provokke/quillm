/* Cache-blocked GEMM driver template, included by each ISA kernel file.
 * Requires: QM_MICRO_4x4(A, lda, B, ldb, C, ldc, kc)  -- C[4x4] += A[4,kc] * B[4,kc]^T
 *           QM_DOTF(a, b, n)                         -- scalar-result dot product
 *           QM_GEMM_FN                               -- name of the generated function
 * The tile decomposition depends only on (M, N, K), never on threading, so results are
 * bitwise reproducible as long as callers split M at multiples of 4. */
static void QM_GEMM_FN(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K, int64_t lda,
                       int64_t ldb, int64_t ldc) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) C[i * ldc + j] = 0.0f;
    for (int64_t j0 = 0; j0 < N; j0 += QM_GEMM_NB) {
        int64_t j1 = j0 + QM_GEMM_NB < N ? j0 + QM_GEMM_NB : N;
        for (int64_t k0 = 0; k0 < K; k0 += QM_GEMM_KC) {
            int64_t kc = k0 + QM_GEMM_KC < K ? QM_GEMM_KC : K - k0;
            for (int64_t i = 0; i < M; i += 4) {
                for (int64_t j = j0; j < j1; j += 4) {
                    if (i + 4 <= M && j + 4 <= j1) {
                        QM_MICRO_4x4(A + i * lda + k0, lda, B + j * ldb + k0, ldb, C + i * ldc + j, ldc, kc);
                    } else {
                        for (int64_t ii = i; ii < M && ii < i + 4; ii++)
                            for (int64_t jj = j; jj < j1 && jj < j + 4; jj++)
                                C[ii * ldc + jj] += QM_DOTF(A + ii * lda + k0, B + jj * ldb + k0, kc);
                    }
                }
            }
        }
    }
}
