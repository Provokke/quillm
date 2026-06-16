/* Every SIMD kernel vs the scalar reference and a float64 oracle, over random shapes
 * (including lengths that are not multiples of 8/16/32), plus elementwise ops. */
#include "ops.h"
#include "test.h"

static float *rvec(int64_t n) {
    float *v = malloc(sizeof(float) * (size_t)(n ? n : 1));
    for (int64_t i = 0; i < n; i++) v[i] = rnd_normal();
    return v;
}

/* float accumulation error bound: gamma_n * sum|terms| (with slack for reassociation) */
static double accum_tol(int64_t n, double abs_sum) { return 4.0 * (double)(n + 16) * 6e-8 * abs_sum + 1e-30; }

static void test_dot_f32(const qm_kernels *k, int iters) {
    int bad = 0;
    for (int it = 0; it < iters; it++) {
        int64_t n = it < 70 ? it : rnd_int(1, 3000);
        float *a = rvec(n), *b = rvec(n);
        double ref = 0, abs = 0;
        for (int64_t i = 0; i < n; i++) {
            ref += (double)a[i] * b[i];
            abs += fabs((double)a[i] * b[i]);
        }
        float got = k->dot_f32(a, b, n), sc = qm_kernels_scalar.dot_f32(a, b, n);
        if (fabs(got - ref) > accum_tol(n, abs) || fabs(sc - ref) > accum_tol(n, abs)) {
            bad++;
            fprintf(stderr, "  dot_f32[%s] n=%lld got=%g scalar=%g ref=%g\n", qm_isa_name(k->isa), (long long)n, got, sc, ref);
        }
        free(a);
        free(b);
    }
    CHECK(bad == 0, "dot_f32 %s: %d/%d out of tolerance", qm_isa_name(k->isa), bad, iters);
}

static void test_dot_quant(const qm_kernels *k, int iters) {
    int bad8 = 0, bad4 = 0, exact_mism = 0;
    for (int it = 0; it < iters; it++) {
        int64_t n = rnd_int(1, 2000), nb = qm_nblocks(n);
        float *w = rvec(n), *x = rvec(n);
        qm_block_q8 *w8 = malloc(sizeof(qm_block_q8) * (size_t)nb), *x8 = malloc(sizeof(qm_block_q8) * (size_t)nb);

        qm_block_q4 *w4 = malloc(sizeof(qm_block_q4) * (size_t)nb);
        qm_quantize_row_q8(w, w8, n);
        qm_quantize_row_q4(w, w4, n);
        qm_quantize_row_q8(x, x8, n);
        /* exact oracle: integer block dots scaled in double */
        double r8 = 0, a8 = 0, r4 = 0, a4 = 0;
        for (int64_t b = 0; b < nb; b++) {
            long s8 = 0, s4 = 0;
            for (int i = 0; i < 32; i++) {
                s8 += (long)w8[b].qs[i] * x8[b].qs[i];
                int q = ((w4[b].qs[i / 2] >> ((i & 1) * 4)) & 0xf) - 8;
                s4 += (long)q * x8[b].qs[i];
            }
            double dx = qm_fp16_to_fp32(x8[b].d);
            double t8 = (double)s8 * qm_fp16_to_fp32(w8[b].d) * dx, t4 = (double)s4 * qm_fp16_to_fp32(w4[b].d) * dx;
            r8 += t8;
            a8 += fabs(t8);
            r4 += t4;
            a4 += fabs(t4);
        }
        float g8 = k->dot_q8(w8, x8, nb), g4 = k->dot_q4(w4, x8, nb);
        if (fabs(g8 - r8) > accum_tol(nb * 8, a8)) bad8++;
        if (fabs(g4 - r4) > accum_tol(nb * 8, a4)) bad4++;
        /* SIMD vs scalar reference kernel (different but bounded rounding order) */
        if (fabs(g8 - qm_kernels_scalar.dot_q8(w8, x8, nb)) > 2 * accum_tol(nb * 8, a8) ||
            fabs(g4 - qm_kernels_scalar.dot_q4(w4, x8, nb)) > 2 * accum_tol(nb * 8, a4))
            exact_mism++;
        free(w);
        free(x);
        free(w8);
        free(x8);
        free(w4);
    }
    CHECK(bad8 == 0, "dot_q8 %s: %d/%d out of tolerance", qm_isa_name(k->isa), bad8, iters);
    CHECK(bad4 == 0, "dot_q4 %s: %d/%d out of tolerance", qm_isa_name(k->isa), bad4, iters);
    CHECK(exact_mism == 0, "quant dots vs scalar kernel out of tolerance for %s (%d)", qm_isa_name(k->isa), exact_mism);
}

static void test_gemm(const qm_kernels *k, int iters) {
    int bad = 0;
    for (int it = 0; it < iters; it++) {
        int64_t M = rnd_int(1, 40), N = rnd_int(1, 150), K = rnd_int(1, 600);
        if (it == 0) M = 4, N = 4, K = 8;
        float *A = rvec(M * K), *B = rvec(N * K), *C = malloc(sizeof(float) * (size_t)(M * N));
        k->gemm_nt(A, B, C, M, N, K, K, K, N);
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++) {
                double ref = 0, abs = 0;
                for (int64_t kk = 0; kk < K; kk++) {
                    ref += (double)A[i * K + kk] * B[j * K + kk];
                    abs += fabs((double)A[i * K + kk] * B[j * K + kk]);
                }
                if (fabs(C[i * N + j] - ref) > accum_tol(K, abs)) bad++;
            }
        free(A);
        free(B);
        free(C);
    }
    CHECK(bad == 0, "gemm %s: %d elements out of tolerance", qm_isa_name(k->isa), bad);
}

static void test_elementwise(void) {
    /* rmsnorm */
    int n = 77;
    float *x = rvec(n), *w = rvec(n), *y = malloc(sizeof(float) * (size_t)n);
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    qm_rmsnorm(y, x, w, n, 1e-5f);
    double inv = 1.0 / sqrt(ss / n + 1e-5);
    for (int i = 0; i < n; i++) CHECK(fabs(y[i] - x[i] * inv * w[i]) < 1e-5 * (1 + fabs(x[i] * inv * w[i])), "rmsnorm %d", i);
    /* softmax: sums to 1, order preserved, stable for huge inputs */
    float s[5] = {1000.0f, 1001.0f, 999.0f, -1e30f, 1001.0f};
    qm_softmax(s, 5);
    CHECK(fabsf(s[0] + s[1] + s[2] + s[3] + s[4] - 1.0f) < 1e-6f && s[1] == s[4] && s[1] > s[0] && s[3] == 0.0f, "softmax");
    /* rope: norm-preserving, identity at pos 0, matches closed form */
    float v[24], v0[24];
    for (int i = 0; i < 24; i++) v[i] = v0[i] = rnd_normal();
    qm_rope(v, 2, 12, 0, 10000.0f);
    CHECK(memcmp(v, v0, sizeof v) == 0, "rope pos0 identity");
    qm_rope(v, 2, 12, 37, 10000.0f);
    for (int h = 0; h < 2; h++)
        for (int i = 0; i < 6; i++) {
            float *p = v + h * 12 + 2 * i, *q = v0 + h * 12 + 2 * i;
            double ang = 37.0 * pow(10000.0, -2.0 * i / 12.0);
            CHECK(fabs(p[0] - (q[0] * cos(ang) - q[1] * sin(ang))) < 1e-5 && fabs(p[1] - (q[0] * sin(ang) + q[1] * cos(ang))) < 1e-5,
                  "rope h=%d i=%d", h, i);
        }
    /* swiglu */
    float g[3] = {0.0f, 2.0f, -3.0f}, u[3] = {5.0f, 0.5f, 2.0f};
    qm_swiglu(g, u, 3);
    CHECK(g[0] == 0.0f && fabsf(g[1] - 2.0f / (1.0f + expf(-2.0f)) * 0.5f) < 1e-6f && fabsf(g[2] - (-3.0f / (1.0f + expf(3.0f)) * 2.0f)) < 1e-6f,
          "swiglu");
    free(x);
    free(w);
    free(y);
}

int main(void) {
    int q = quick();
    int tested = 0;
    for (int isa = 0; isa < QM_ISA_COUNT; isa++) {
        const qm_kernels *k = qm_kernels_for((qm_isa)isa);
        if (!k) {
            printf("isa %-6s: not available on this CPU/build, skipped\n", qm_isa_name((qm_isa)isa));
            continue;
        }
        tested++;
        test_dot_f32(k, q ? 100 : 1500);
        test_dot_quant(k, q ? 60 : 1500);
        test_gemm(k, q ? 6 : 60);
        printf("isa %-6s: kernels checked\n", qm_isa_name((qm_isa)isa));
    }
    CHECK(tested >= 1, "no ISA tested");
    test_elementwise();
    TEST_DONE();
}
