/* quillm - core types: dtypes, fp16, quant blocks, arena, cpu features. */
#ifndef QM_BASE_H
#define QM_BASE_H

#include <stddef.h>
#include <stdint.h>

#define QM_ALIGN 64
#define QM_BLOCK 32 /* elements per quantization block */

typedef enum {
    QM_F32 = 0,
    QM_Q8_0 = 2, /* 32 x int8, fp16 scale            (34 bytes/block) */
    QM_Q4_0 = 3, /* 32 x 4-bit (offset 8), fp16 scale (18 bytes/block) */
} qm_dtype;

typedef struct {
    uint16_t d;     /* fp16 scale */
    int8_t qs[32];  /* value = d * qs[i] */
} qm_block_q8;

typedef struct {
    uint16_t d;     /* fp16 scale */

    uint8_t qs[16]; /* byte j: elem 2j in low nibble, elem 2j+1 in high nibble; value = d*(q-8) */
} qm_block_q4;

_Static_assert(sizeof(qm_block_q8) == 34, "q8 block size");
_Static_assert(sizeof(qm_block_q4) == 18, "q4 block size");

/* ---- fp16 (own implementation, round-to-nearest-even) ---- */
float qm_fp16_to_fp32(uint16_t h);
uint16_t qm_fp32_to_fp16(float f);

/* ---- quantization (reference, scalar) ---- */
static inline int64_t qm_nblocks(int64_t n) { return (n + QM_BLOCK - 1) / QM_BLOCK; }
/* bytes needed for one row of n elements of given dtype (quantized rows are zero-padded to a block multiple) */
size_t qm_row_bytes(qm_dtype t, int64_t n);
const char *qm_dtype_name(qm_dtype t);
int qm_dtype_valid(uint32_t t);
void qm_quantize_row_q8(const float *x, qm_block_q8 *y, int64_t n);
void qm_dequantize_row_q8(const qm_block_q8 *x, float *y, int64_t n);
void qm_quantize_row_q4(const float *x, qm_block_q4 *y, int64_t n);
void qm_dequantize_row_q4(const qm_block_q4 *x, float *y, int64_t n);
/* dequantize/copy a row of any dtype to fp32 */
void qm_row_to_f32(qm_dtype t, const void *row, float *y, int64_t n);

/* ---- arena: 64-byte aligned bump allocator ---- */
typedef struct {
    uint8_t *base;
    size_t cap, used, peak;
} qm_arena;

int qm_arena_init(qm_arena *a, size_t cap);
void *qm_arena_alloc(qm_arena *a, size_t n); /* returns NULL when exhausted; memory is zeroed */
void qm_arena_reset(qm_arena *a);
void qm_arena_free(qm_arena *a);

/* ---- CPU features / ISA levels ---- */
typedef enum { QM_ISA_SCALAR = 0, QM_ISA_AVX2 = 1, QM_ISA_AVX512 = 2, QM_ISA_NEON = 3, QM_ISA_COUNT } qm_isa;
int qm_cpu_supports(qm_isa isa); /* runtime check (cpuid + xgetbv on x86) */
qm_isa qm_cpu_best(void);
const char *qm_isa_name(qm_isa isa);
int qm_isa_from_name(const char *s); /* -1 if unknown */

#endif
