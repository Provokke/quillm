/* quillm-bench: GFLOP/s per kernel per ISA, and decode tokens/sec for model files.
 * usage: quillm-bench [model.qlm ...]   (env QM_BENCH_THREADS, default 4) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "sampler.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float *rvec(int64_t n) {
    float *v = aligned_alloc(64, ((size_t)n * 4 + 63) & ~(size_t)63);
    for (int64_t i = 0; i < n; i++) v[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    return v;
}

/* run fn repeatedly for ~min_time seconds, return best-of-3 seconds per call */
typedef void (*bench_fn)(void *);
static double time_it(bench_fn fn, void *ctx, double min_time) {
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        int iters = 0;
        double t0 = now_s(), t;
        do {
            fn(ctx);
            iters++;
            t = now_s() - t0;
        } while (t < min_time);
        if (t / iters < best) best = t / iters;
    }
    return best;
}

typedef struct {
    const qm_kernels *k;
    float *a, *b, *c;
    int64_t n, m, kk;
    qm_tensor W;
    qm_block_q8 *xq;
    qm_pool *pool;
    volatile float sink;
} ctx_t;

static void b_dot(void *v) {
    ctx_t *c = v;
    c->sink = c->k->dot_f32(c->a, c->b, c->n);
}
static void b_matvec(void *v) {
    ctx_t *c = v;
    qm_matvec(c->pool, &c->W, c->b, c->c, c->xq);
}
static void b_gemm(void *v) {
    ctx_t *c = v;
    qm_matmul_nt(c->pool, c->a, c->b, c->c, c->m, c->n, c->kk);
}

int main(int argc, char **argv) {
    int nt = getenv("QM_BENCH_THREADS") ? atoi(getenv("QM_BENCH_THREADS")) : 4;
    double mt = 0.2;
    printf("quillm-bench (threads for MT rows: %d)\n\n", nt);
    printf("%-34s %10s %10s %10s\n", "kernel", "scalar", "avx2", "avx512");
    printf("%-34s %10s %10s %10s\n", "", "GFLOP/s", "GFLOP/s", "GFLOP/s");

    const int64_t R = 4096, C = 4096, G = 512;
    float *Wf = rvec(R * C), *x = rvec(C), *y = rvec(R), *A = rvec(G * G), *B = rvec(G * G), *Cm = rvec(G * G);
    void *W8 = aligned_alloc(64, R * qm_row_bytes(QM_Q8_0, C)), *W4 = aligned_alloc(64, R * qm_row_bytes(QM_Q4_0, C));
    for (int64_t r = 0; r < R; r++) {
        qm_quantize_row_q8(Wf + r * C, (qm_block_q8 *)((uint8_t *)W8 + r * qm_row_bytes(QM_Q8_0, C)), C);
        qm_quantize_row_q4(Wf + r * C, (qm_block_q4 *)((uint8_t *)W4 + r * qm_row_bytes(QM_Q4_0, C)), C);
    }
    qm_block_q8 *xq = aligned_alloc(64, qm_nblocks(C) * sizeof(qm_block_q8) + 64);
    qm_pool *pool = qm_pool_create(nt);

    const char *rows[] = {"dot f32 (n=4096, L1)", "matvec f32 4096x4096 (1T)", "matvec f32 4096x4096 (MT)",
                          "matvec q8_0 4096x4096 (1T)", "matvec q8_0 4096x4096 (MT)", "matvec q4_0 4096x4096 (1T)",
                          "matvec q4_0 4096x4096 (MT)", "gemm f32 512^3 (1T)", "gemm f32 512^3 (MT)"};
    for (int row = 0; row < 9; row++) {
        printf("%-34s", rows[row]);
        for (int isa = 0; isa < 3; isa++) {
            if (qm_kernels_select((qm_isa)isa) != 0) {
                printf(" %10s", "n/a");
                continue;
            }
            ctx_t c;
            memset(&c, 0, sizeof c);
            c.k = qm_kernels_active();
            c.xq = xq;
            c.c = y;
            double flops, t;
            if (row == 0) {
                c.a = Wf;
                c.b = x;
                c.n = 4096;
                flops = 2.0 * 4096;
                t = time_it(b_dot, &c, mt / 4);
            } else if (row <= 6) {
                int dt = (row - 1) / 2;
                c.pool = (row % 2 == 0) ? pool : NULL;
                c.b = x;
                qm_tensor W = {"w", dt == 0 ? QM_F32 : dt == 1 ? QM_Q8_0 : QM_Q4_0, 2, {R, C}, dt == 0 ? (void *)Wf : dt == 1 ? W8 : W4};
                c.W = W;
                flops = 2.0 * R * C;
                t = time_it(b_matvec, &c, mt);
            } else {
                c.pool = row == 8 ? pool : NULL;
                c.a = A;
                c.b = B;
                c.c = Cm;
                c.m = c.n = c.kk = G;
                flops = 2.0 * G * G * G;
                t = time_it(b_gemm, &c, mt);
            }
            printf(" %10.2f", flops / t * 1e-9);
            fflush(stdout);
        }
        printf("\n");
    }

    /* decode tokens/sec */
    if (argc > 1) printf("\n%-28s %8s %12s %12s %12s\n", "model (decode, 128 tok)", "threads", "scalar tok/s", "avx2 tok/s", "avx512 tok/s");
    for (int i = 1; i < argc; i++) {
        const char *base = strrchr(argv[i], '/') ? strrchr(argv[i], '/') + 1 : argv[i];
        int tcounts[2] = {1, nt};
        for (int ti = 0; ti < 2; ti++) {
            printf("%-28s %8d", base, tcounts[ti]);
            for (int isa = 0; isa < 3; isa++) {
                if (qm_kernels_select((qm_isa)isa) != 0) {
                    printf(" %12s", "n/a");
                    continue;
                }
                qm_model m;
                char err[256];
                if (qm_model_load(&m, argv[i], tcounts[ti], 160, err, sizeof err) != 0) {
                    printf(" error: %s\n", err);
                    break;
                }
                int32_t tok = QM_TOK_BOS;
                for (int p = 0; p < 8; p++) tok = qm_argmax(qm_model_forward(&m, tok, p), m.cfg.vocab_size); /* warm-up */
                int steps = isa == 0 ? 32 : 128;
                double t0 = now_s();
                for (int p = 8; p < 8 + steps; p++) tok = qm_argmax(qm_model_forward(&m, tok, p), m.cfg.vocab_size);
                double t = now_s() - t0;
                printf(" %12.1f", steps / t);
                fflush(stdout);
                qm_model_free(&m);
            }
            printf("\n");
        }
    }
    qm_pool_destroy(pool);
    free(Wf), free(x), free(y), free(A), free(B), free(Cm), free(W8), free(W4), free(xq);
    return 0;
}
