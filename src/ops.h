/* Tensor type and (multithreaded) high-level ops built on the dispatched kernels. */
#ifndef QM_OPS_H
#define QM_OPS_H

#include "kernels.h"
#include "threadpool.h"

/* Row-major tensor. ne[0] is the outermost dimension (numpy shape order). */
typedef struct {
    char name[64];
    qm_dtype type;
    int ndim;
    int64_t ne[4];
    const void *data;
} qm_tensor;

int64_t qm_tensor_nelem(const qm_tensor *t);
int64_t qm_tensor_rows(const qm_tensor *t); /* product of all but last dim */
int64_t qm_tensor_cols(const qm_tensor *t); /* last dim */
size_t qm_tensor_bytes(const qm_tensor *t);
const void *qm_tensor_row(const qm_tensor *t, int64_t r);

/* y[rows] = W[rows, cols] * x[cols]. For quantized W, x is quantized to Q8_0 into xq
 * (must hold qm_nblocks(cols) blocks). Bitwise identical for any thread count. */
void qm_matvec(qm_pool *p, const qm_tensor *W, const float *x, float *y, qm_block_q8 *xq);
/* C[M,N] = A[M,K] * B[N,K]^T (contiguous). Bitwise identical for any thread count. */
void qm_matmul_nt(qm_pool *p, const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K);

void qm_rmsnorm(float *out, const float *x, const float *w, int64_t n, float eps);
void qm_softmax(float *x, int64_t n);
/* rotate interleaved pairs (2i, 2i+1) of each head by pos * theta^(-2i/head_dim) */
void qm_rope(float *v, int64_t n_heads, int64_t head_dim, int64_t pos, float theta);
/* gate[i] = silu(gate[i]) * up[i] */
void qm_swiglu(float *gate, const float *up, int64_t n);

#endif
