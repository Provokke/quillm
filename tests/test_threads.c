/* Thread pool coverage and thread-count invariance (bitwise) of matvec / matmul. */
#include "ops.h"
#include "test.h"

typedef struct {
    int *hits;
    int64_t grain;
    int bad_start;
} cover_ctx;

static void cover_task(void *v, int64_t s, int64_t e, int tid) {
    cover_ctx *c = v;
    (void)tid;
    if (s % c->grain) c->bad_start = 1;
    for (int64_t i = s; i < e; i++) c->hits[i]++;
}

static float *rvec(int64_t n) {
    float *v = malloc(sizeof(float) * (size_t)n);
    for (int64_t i = 0; i < n; i++) v[i] = rnd_normal();
    return v;
}

int main(void) {
    int q = quick();
    /* coverage: every index visited exactly once, chunk starts aligned to grain */
    for (int nt = 1; nt <= 5; nt++) {
        qm_pool *p = qm_pool_create(nt);
        for (int it = 0; it < (q ? 20 : 300); it++) {
            int64_t n = rnd_int(0, 1000), grain = rnd_int(1, 70);
            int *hits = calloc((size_t)n + 1, sizeof(int));
            cover_ctx c = {hits, grain, 0};
            qm_pool_run(p, n, grain, cover_task, &c);
            int ok = !c.bad_start;
            for (int64_t i = 0; i < n; i++) ok &= hits[i] == 1;
            CHECK(ok, "coverage nt=%d n=%lld grain=%lld", nt, (long long)n, (long long)grain);
            free(hits);
        }
        qm_pool_destroy(p);
    }
    /* many tiny jobs: no lost wakeups / deadlocks */
    {
        qm_pool *p = qm_pool_create(4);
        int hits[64];
        int ok = 1;
        for (int it = 0; it < (q ? 500 : 20000); it++) {
            memset(hits, 0, sizeof hits);
            cover_ctx c = {hits, 1, 0};
            qm_pool_run(p, 64, 1, cover_task, &c);
            for (int i = 0; i < 64; i++) ok &= hits[i] == 1;
        }
        CHECK(ok, "stress");
        qm_pool_destroy(p);
    }

    /* thread-count invariance, for every available ISA */
    for (int isa = 0; isa < QM_ISA_COUNT; isa++) {
        if (qm_kernels_select((qm_isa)isa) != 0) continue;
        for (int it = 0; it < (q ? 2 : 12); it++) {
            int64_t rows = rnd_int(1, 300), cols = rnd_int(1, 400);
            float *W = rvec(rows * cols), *x = rvec(cols);
            void *w8 = malloc(rows * qm_row_bytes(QM_Q8_0, cols)), *w4 = malloc(rows * qm_row_bytes(QM_Q4_0, cols));
            for (int64_t r = 0; r < rows; r++) {
                qm_quantize_row_q8(W + r * cols, (qm_block_q8 *)((uint8_t *)w8 + r * qm_row_bytes(QM_Q8_0, cols)), cols);
                qm_quantize_row_q4(W + r * cols, (qm_block_q4 *)((uint8_t *)w4 + r * qm_row_bytes(QM_Q4_0, cols)), cols);
            }
            qm_tensor t[3] = {{"f", QM_F32, 2, {rows, cols}, W}, {"q8", QM_Q8_0, 2, {rows, cols}, w8}, {"q4", QM_Q4_0, 2, {rows, cols}, w4}};
            qm_block_q8 *xq = malloc(sizeof(qm_block_q8) * (size_t)qm_nblocks(cols));
            float *y1 = malloc(4 * (size_t)rows), *yn = malloc(4 * (size_t)rows);
            for (int ti = 0; ti < 3; ti++) {
                qm_matvec(NULL, &t[ti], x, y1, xq);
                for (int nt = 2; nt <= 7; nt++) {
                    qm_pool *p = qm_pool_create(nt);
                    qm_matvec(p, &t[ti], x, yn, xq);
                    CHECK(memcmp(y1, yn, 4 * (size_t)rows) == 0, "matvec %s %s nt=%d not bitwise equal", t[ti].name,
                          qm_isa_name((qm_isa)isa), nt);
                    qm_pool_destroy(p);
                }
            }
            /* matmul: M both small (split along N) and large (split along M) */
            int64_t M = it % 2 ? rnd_int(1, 7) : rnd_int(8, 90), N = rnd_int(1, 300), K = rnd_int(1, 300);
            float *A = rvec(M * K), *B = rvec(N * K), *C1 = malloc(4 * (size_t)(M * N)), *Cn = malloc(4 * (size_t)(M * N));
            qm_matmul_nt(NULL, A, B, C1, M, N, K);
            for (int nt = 2; nt <= 6; nt++) {
                qm_pool *p = qm_pool_create(nt);
                qm_matmul_nt(p, A, B, Cn, M, N, K);
                CHECK(memcmp(C1, Cn, 4 * (size_t)(M * N)) == 0, "matmul %lldx%lldx%lld nt=%d %s not bitwise equal", (long long)M,
                      (long long)N, (long long)K, nt, qm_isa_name((qm_isa)isa));
                qm_pool_destroy(p);
            }
            free(A), free(B), free(C1), free(Cn), free(W), free(x), free(w8), free(w4), free(xq), free(y1), free(yn);
        }
        printf("isa %-6s: thread-count invariance checked\n", qm_isa_name((qm_isa)isa));
    }
    TEST_DONE();
}
