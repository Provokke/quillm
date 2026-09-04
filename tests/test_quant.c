/* Q8_0 / Q4_0 quantize/dequantize: error bounds, padding, numpy byte-exactness. */
#include "qm_base.h"
#include "test.h"

static void check_bounds(int q4, int n, float scale) {
    float *x = malloc(sizeof(float) * (size_t)n), *y = malloc(sizeof(float) * (size_t)n);
    int64_t nb = qm_nblocks(n);
    void *blk = malloc(qm_row_bytes(q4 ? QM_Q4_0 : QM_Q8_0, n));
    for (int i = 0; i < n; i++) x[i] = rnd_normal() * scale;
    if (rnd_int(0, 3) == 0) x[rnd_int(0, n - 1)] *= 50.0f; /* outlier */
    float qmax = q4 ? 7.0f : 127.0f;
    if (q4) {
        qm_quantize_row_q4(x, blk, n);
        qm_dequantize_row_q4(blk, y, n);
    } else {
        qm_quantize_row_q8(x, blk, n);
        qm_dequantize_row_q8(blk, y, n);
    }
    double se = 0, sx = 0, se_model = 0;
    int bad = 0;
    for (int64_t b = 0; b < nb; b++) {
        uint16_t dh = q4 ? ((qm_block_q4 *)blk)[b].d : ((qm_block_q8 *)blk)[b].d;
        float dd = qm_fp16_to_fp32(dh), amax = 0;
        for (int i = (int)b * 32; i < n && i < (int)b * 32 + 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        /* rounding error <= d/2, plus clamp error when fp16 rounding shrank qmax*d below amax */
        float bound = 0.5f * dd + fmaxf(0.0f, amax - qmax * dd) + 1e-6f * amax;
        for (int i = (int)b * 32; i < n && i < (int)b * 32 + 32; i++) {
            se_model += (double)dd * dd / 12.0; /* uniform rounding-error model */
            if (fabsf(x[i] - y[i]) > bound) bad++;
            se += (double)(x[i] - y[i]) * (x[i] - y[i]);
            sx += (double)x[i] * x[i];
        }
    }
    CHECK(bad == 0, "%s n=%d: %d elements exceed the per-block bound", q4 ? "q4" : "q8", n, bad);
    /* padding of the final block must quantize to zero */
    if (n % 32) {
        int64_t b = nb - 1;
        for (int i = n % 32; i < 32; i++) {
            int q = q4 ? ((((qm_block_q4 *)blk)[b].qs[i / 2] >> ((i & 1) * 4)) & 0xf) - 8 : ((qm_block_q8 *)blk)[b].qs[i];
            CHECK(q == 0, "pad elem %d not zero", i);
        }
    }
    /* statistical check: observed error energy close to the d^2/12 rounding model, and
     * absolute relative rmse sane for gaussian data */
    double rel = sqrt(se / (sx + 1e-30)), rel_model = sqrt(se_model / (sx + 1e-30));  
    CHECK(rel <= 1.6 * rel_model + 1e-7, "%s n=%d rmse %.4f vs model %.4f", q4 ? "q4" : "q8", n, rel, rel_model);
    CHECK(rel < (q4 ? 0.35 : 0.03), "%s n=%d relative rmse %.4f", q4 ? "q4" : "q8", n, rel);
    free(x);
    free(y);
    free(blk);
}

int main(void) {
    int iters = quick() ? 40 : 600;
    for (int it = 0; it < iters; it++) {
        int n = rnd_int(1, 700);
        float scale = ldexpf(1.0f, rnd_int(-12, 12));
        check_bounds(0, n, scale);
        check_bounds(1, n, scale);
    }
    /* all-zero input -> zero scale and zero quants */
    float z[64] = {0}, out[64];
    qm_block_q8 b8[2];
    qm_block_q4 b4[2];
    qm_quantize_row_q8(z, b8, 64);
    qm_quantize_row_q4(z, b4, 64);
    CHECK(b8[0].d == 0 && b8[1].qs[5] == 0 && b4[1].d == 0 && b4[1].qs[3] == 0x88, "zero block");
    qm_dequantize_row_q4(b4, out, 64);
    CHECK(out[17] == 0.0f, "zero dequant");
    /* large magnitudes stay finite */
    float big[32];
    for (int i = 0; i < 32; i++) big[i] = (i - 16) * 3000.0f;
    qm_quantize_row_q8(big, b8, 32);
    qm_dequantize_row_q8(b8, out, 32);
    for (int i = 0; i < 32; i++) CHECK(isfinite(out[i]) && fabsf(out[i] - big[i]) < 300.0f, "big %d", i);

    /* byte-exact agreement with the numpy quantizer (tools/qlm.py) */
    char p[512];
    size_t sz_in, sz8, sz4;
    snprintf(p, sizeof p, "%s/qin.f32", fixtures_dir());
    float *x = read_file(p, &sz_in);
    snprintf(p, sizeof p, "%s/qin.q8", fixtures_dir());
    uint8_t *ref8 = read_file(p, &sz8);
    snprintf(p, sizeof p, "%s/qin.q4", fixtures_dir());
    uint8_t *ref4 = read_file(p, &sz4);
    CHECK(x && ref8 && ref4, "fixtures missing (run make fixtures)");
    if (x && ref8 && ref4) {
        int rows = 5, n = 100;
        CHECK(sz_in == (size_t)rows * n * 4 && sz8 == rows * qm_row_bytes(QM_Q8_0, n) && sz4 == rows * qm_row_bytes(QM_Q4_0, n),
              "fixture sizes");
        uint8_t buf[1024];
        for (int r = 0; r < rows; r++) {
            qm_quantize_row_q8(x + r * n, (qm_block_q8 *)buf, n);
            CHECK(memcmp(buf, ref8 + r * qm_row_bytes(QM_Q8_0, n), qm_row_bytes(QM_Q8_0, n)) == 0, "q8 row %d != numpy", r);
            qm_quantize_row_q4(x + r * n, (qm_block_q4 *)buf, n);
            CHECK(memcmp(buf, ref4 + r * qm_row_bytes(QM_Q4_0, n), qm_row_bytes(QM_Q4_0, n)) == 0, "q4 row %d != numpy", r);
        }
    }
    free(x);
    free(ref8);
    free(ref4);
    TEST_DONE();
}
