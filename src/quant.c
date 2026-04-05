/* Reference (scalar) block quantizers. Rows are zero-padded to a multiple of 32. */
#include <math.h>
#include <string.h>

#include "qm_base.h"

size_t qm_row_bytes(qm_dtype t, int64_t n) {
    switch (t) {
    case QM_F32: return (size_t)n * 4;
    case QM_Q8_0: return (size_t)qm_nblocks(n) * sizeof(qm_block_q8);
    case QM_Q4_0: return (size_t)qm_nblocks(n) * sizeof(qm_block_q4);
    }
    return 0;
}

const char *qm_dtype_name(qm_dtype t) {
    switch (t) {
    case QM_F32: return "f32";
    case QM_Q8_0: return "q8_0";
    case QM_Q4_0: return "q4_0";
    }
    return "?";
}

int qm_dtype_valid(uint32_t t) { return t == QM_F32 || t == QM_Q8_0 || t == QM_Q4_0; }

/* load one block of up to 32 values, zero padding the tail */
static void load_block(const float *x, int64_t n, int64_t b, float out[QM_BLOCK]) {
    int64_t base = b * QM_BLOCK;
    for (int i = 0; i < QM_BLOCK; i++) out[i] = (base + i < n) ? x[base + i] : 0.0f;
}

/* Common scheme (mirrored exactly by tools/qlm.py):
 *   d = amax / qmax (fp32) -> rounded to fp16 -> dd = fp32(fp16(d))
 *   q = clamp(rint(x / dd), qmin, qmax)  */
static float block_scale(const float v[QM_BLOCK], float qmax, uint16_t *dh) {
    float amax = 0.0f;
    for (int i = 0; i < QM_BLOCK; i++) {
        float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }
    float d = amax / qmax;
    *dh = qm_fp32_to_fp16(d);
    return qm_fp16_to_fp32(*dh);
}

static int quant_one(float v, float dd, int qmin, int qmax) {
    if (dd == 0.0f) return 0;
    float r = rintf(v / dd);
    if (r < (float)qmin) r = (float)qmin;
    if (r > (float)qmax) r = (float)qmax;
    return (int)r;
}

void qm_quantize_row_q8(const float *x, qm_block_q8 *y, int64_t n) {
    int64_t nb = qm_nblocks(n);
    for (int64_t b = 0; b < nb; b++) {
        float v[QM_BLOCK];
        load_block(x, n, b, v);
        float dd = block_scale(v, 127.0f, &y[b].d);
        for (int i = 0; i < QM_BLOCK; i++) y[b].qs[i] = (int8_t)quant_one(v[i], dd, -127, 127);
    }
}

void qm_dequantize_row_q8(const qm_block_q8 *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        const qm_block_q8 *b = &x[i / QM_BLOCK];
        y[i] = qm_fp16_to_fp32(b->d) * (float)b->qs[i % QM_BLOCK];
    }
}

void qm_quantize_row_q4(const float *x, qm_block_q4 *y, int64_t n) {
    int64_t nb = qm_nblocks(n);
    for (int64_t b = 0; b < nb; b++) {
        float v[QM_BLOCK];
        load_block(x, n, b, v);
        float dd = block_scale(v, 7.0f, &y[b].d);
        for (int j = 0; j < 16; j++) {
            int lo = quant_one(v[2 * j], dd, -8, 7) + 8;
            int hi = quant_one(v[2 * j + 1], dd, -8, 7) + 8;
            y[b].qs[j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

void qm_dequantize_row_q4(const qm_block_q4 *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        const qm_block_q4 *b = &x[i / QM_BLOCK];
        int k = (int)(i % QM_BLOCK);
        uint8_t byte = b->qs[k / 2];
        int q = (k & 1) ? (byte >> 4) : (byte & 0xf);
        y[i] = qm_fp16_to_fp32(b->d) * (float)(q - 8);
    }
}

void qm_row_to_f32(qm_dtype t, const void *row, float *y, int64_t n) {
    switch (t) {
    case QM_F32: memcpy(y, row, (size_t)n * 4); break;
    case QM_Q8_0: qm_dequantize_row_q8(row, y, n); break;
    case QM_Q4_0: qm_dequantize_row_q4(row, y, n); break;
    }
}
