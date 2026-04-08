#ifndef QM_SAMPLER_H
#define QM_SAMPLER_H

#include <stdint.h>

#include "qm_base.h"

typedef struct {
    float temperature; /* 0 => greedy */
    int top_k;         /* 0 => disabled */
    float top_p;       /* 1 => disabled */
    uint64_t rng;
    int32_t n_vocab;
    float *probs;      /* scratch (arena) */
    int32_t *idx;      /* scratch (arena) */
} qm_sampler;

int qm_sampler_init(qm_sampler *s, qm_arena *a, int32_t n_vocab, float temperature, int top_k, float top_p,
                    uint64_t seed);
int32_t qm_sample(qm_sampler *s, const float *logits);
int32_t qm_argmax(const float *x, int32_t n);
float qm_rng_uniform(uint64_t *state); /* [0, 1) */

#endif
