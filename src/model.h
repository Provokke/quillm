/* Llama-style decoder-only transformer (RMSNorm, RoPE, GQA attention, SwiGLU FFN). */
#ifndef QM_MODEL_H
#define QM_MODEL_H

#include "qlm_file.h"
#include "tokenizer.h"

typedef struct {
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq;
    float rope_theta, norm_eps;
} qm_config;

typedef struct {
    const qm_tensor *attn_norm, *wq, *wk, *wv, *wo, *ffn_norm, *w1, *w2, *w3;
} qm_layer;

typedef struct {
    qm_config cfg;
    qlm_file file;
    qm_tokenizer tok;
    const qm_tensor *tok_embd, *norm, *output;
    qm_layer *layers;
    /* runtime state: all buffers live in one arena allocated at load time */
    qm_arena arena;
    float *x, *xb, *xb2, *q, *k, *v, *hb, *hb2, *att, *logits;
    float *key_cache, *val_cache; /* [layer][max_seq][kv_dim] */
    qm_block_q8 *xq;
    qm_pool *pool;
    int n_threads;
} qm_model;

/* max_seq <= 0 uses the model's value. Returns 0 on success. */
int qm_model_load(qm_model *m, const char *path, int n_threads, int max_seq, char *err, size_t errlen);
void qm_model_free(qm_model *m);
/* Run one token at position pos; returns logits[vocab] (owned by the model). No heap allocation. */
float *qm_model_forward(qm_model *m, int32_t token, int32_t pos);

#endif
