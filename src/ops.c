#include "ops.h"

#include <math.h>

int64_t qm_tensor_nelem(const qm_tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->ne[i];
    return n;
}

int64_t qm_tensor_cols(const qm_tensor *t) { return t->ndim ? t->ne[t->ndim - 1] : 1; }

int64_t qm_tensor_rows(const qm_tensor *t) {
    int64_t r = 1;
    for (int i = 0; i + 1 < t->ndim; i++) r *= t->ne[i];
    return r;
}

size_t qm_tensor_bytes(const qm_tensor *t) { return (size_t)qm_tensor_rows(t) * qm_row_bytes(t->type, qm_tensor_cols(t)); }

const void *qm_tensor_row(const qm_tensor *t, int64_t r) {
    return (const uint8_t *)t->data + (size_t)r * qm_row_bytes(t->type, qm_tensor_cols(t));
}

/* ---------------- matvec ---------------- */
typedef struct {
    const qm_kernels *k;
    const qm_tensor *W;
    const float *x;
    const qm_block_q8 *xq;
    float *y;
    int64_t cols;
    size_t row_bytes;
} matvec_ctx;

static void matvec_task(void *vctx, int64_t r0, int64_t r1, int tid) {
    (void)tid;
    matvec_ctx *c = vctx;
    const uint8_t *base = c->W->data;
    int64_t nb = qm_nblocks(c->cols);
    switch (c->W->type) {
    case QM_F32:
        for (int64_t r = r0; r < r1; r++)
            c->y[r] = c->k->dot_f32((const float *)(base + (size_t)r * c->row_bytes), c->x, c->cols);
        break;
    case QM_Q8_0:
        for (int64_t r = r0; r < r1; r++)
            c->y[r] = c->k->dot_q8((const qm_block_q8 *)(base + (size_t)r * c->row_bytes), c->xq, nb);
        break;
    case QM_Q4_0:
        for (int64_t r = r0; r < r1; r++)
            c->y[r] = c->k->dot_q4((const qm_block_q4 *)(base + (size_t)r * c->row_bytes), c->xq, nb);
        break;
    }
}

void qm_matvec(qm_pool *p, const qm_tensor *W, const float *x, float *y, qm_block_q8 *xq) {
    matvec_ctx c = {qm_kernels_active(), W, x, xq, y, qm_tensor_cols(W), qm_row_bytes(W->type, qm_tensor_cols(W))};
    if (W->type != QM_F32) qm_quantize_row_q8(x, xq, c.cols);
    qm_pool_run(p, qm_tensor_rows(W), 16, matvec_task, &c);
}

/* ---------------- matmul ---------------- */
typedef struct {
    const qm_kernels *k;
    const float *A, *B;
    float *C;
    int64_t M, N, K;
    int split_n;
} matmul_ctx;

static void matmul_task(void *vctx, int64_t s, int64_t e, int tid) {
    (void)tid;
    matmul_ctx *c = vctx;
    if (c->split_n)
        c->k->gemm_nt(c->A, c->B + s * c->K, c->C + s, c->M, e - s, c->K, c->K, c->K, c->N);
    else
        c->k->gemm_nt(c->A + s * c->K, c->B, c->C + s * c->N, e - s, c->N, c->K, c->K, c->K, c->N);
}

void qm_matmul_nt(qm_pool *p, const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K) {
    /* split along M at multiples of 4, or along N at multiples of the N cache block, so the
     * tile decomposition (and therefore every rounding) is the same as single-threaded */
    int nt = qm_pool_size(p);
    matmul_ctx c = {qm_kernels_active(), A, B, C, M, N, K, M < 4 * nt};
    if (c.split_n)
        qm_pool_run(p, N, QM_GEMM_NB, matmul_task, &c);
    else
        qm_pool_run(p, M, 4, matmul_task, &c);
}

/* ---------------- elementwise ---------------- */
void qm_rmsnorm(float *out, const float *x, const float *w, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

void qm_softmax(float *x, int64_t n) {
    if (n <= 0) return;
    float mx = x[0];
    for (int64_t i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    float inv = (float)(1.0 / sum);
    for (int64_t i = 0; i < n; i++) x[i] *= inv;
}

void qm_rope(float *v, int64_t n_heads, int64_t head_dim, int64_t pos, float theta) {
    for (int64_t i = 0; i < head_dim / 2; i++) {
        double freq = pow((double)theta, -2.0 * (double)i / (double)head_dim);
        double ang = (double)pos * freq;
        float c = (float)cos(ang), s = (float)sin(ang);
        for (int64_t h = 0; h < n_heads; h++) {
            float *p = v + h * head_dim + 2 * i;
            float x0 = p[0], x1 = p[1];
            p[0] = x0 * c - x1 * s;
            p[1] = x0 * s + x1 * c;
        }
    }
}

void qm_swiglu(float *gate, const float *up, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}
