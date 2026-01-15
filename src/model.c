#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int get_u32(const qlm_file *f, const char *key, int32_t *out, char *err, size_t errlen) {
    const qlm_kv *kv = qlm_find_kv(f, key);
    if (!kv || kv->type != QLM_KV_U32 || kv->u32 == 0 || kv->u32 > (1u << 24)) {
        snprintf(err, errlen, "missing/invalid metadata '%s'", key);
        return -1;
    }
    *out = (int32_t)kv->u32;
    return 0;
}

static float get_f32(const qlm_file *f, const char *key, float def) {
    const qlm_kv *kv = qlm_find_kv(f, key);
    return (kv && kv->type == QLM_KV_F32) ? kv->f32 : def;
}

static const qm_tensor *need(const qlm_file *f, const char *name, int64_t d0, int64_t d1, char *err,
                             size_t errlen) {
    const qm_tensor *t = qlm_find_tensor(f, name);
    int ok = t && ((d1 == 0 && t->ndim == 1 && t->ne[0] == d0) || (d1 && t->ndim == 2 && t->ne[0] == d0 && t->ne[1] == d1));
    if (ok && d1 == 0 && t->type != QM_F32) ok = 0; /* norm weights must be f32 */
    if (!ok) {
        snprintf(err, errlen, "missing or mis-shaped tensor '%s'", name);
        return NULL;
    }
    return t;
}

int qm_model_load(qm_model *m, const char *path, int n_threads, int max_seq, char *err, size_t errlen) {
    memset(m, 0, sizeof *m);
    if (qlm_open(&m->file, path, err, errlen) != 0) return -1;
    const qlm_file *f = &m->file;
    qm_config *c = &m->cfg;
    if (get_u32(f, "llama.dim", &c->dim, err, errlen) || get_u32(f, "llama.hidden_dim", &c->hidden_dim, err, errlen) ||
        get_u32(f, "llama.n_layers", &c->n_layers, err, errlen) || get_u32(f, "llama.n_heads", &c->n_heads, err, errlen) ||
        get_u32(f, "llama.n_kv_heads", &c->n_kv_heads, err, errlen) ||
        get_u32(f, "llama.vocab_size", &c->vocab_size, err, errlen) ||
        get_u32(f, "llama.max_seq", &c->max_seq, err, errlen))
        goto fail;
    c->rope_theta = get_f32(f, "llama.rope_theta", 10000.0f);
    c->norm_eps = get_f32(f, "llama.norm_eps", 1e-5f);
    if (c->dim % c->n_heads || c->n_heads % c->n_kv_heads || (c->dim / c->n_heads) % 2 || c->n_layers > 1024) {
        snprintf(err, errlen, "inconsistent head configuration");
        goto fail;
    }
    if (max_seq > 0 && max_seq < c->max_seq) c->max_seq = max_seq;

    int64_t D = c->dim, H = c->hidden_dim, KV = (int64_t)c->dim / c->n_heads * c->n_kv_heads, V = c->vocab_size;
    if (!(m->tok_embd = need(f, "tok_embd", V, D, err, errlen)) || !(m->norm = need(f, "norm", D, 0, err, errlen)) ||
        !(m->output = need(f, "output", V, D, err, errlen)))
        goto fail;
    m->layers = calloc((size_t)c->n_layers, sizeof(qm_layer));
    if (!m->layers) goto fail;
    for (int l = 0; l < c->n_layers; l++) {
        char nm[64];
        qm_layer *L = &m->layers[l];
#define NEED(field, suffix, d0, d1)                                                                                    \
    snprintf(nm, sizeof nm, "layers.%d." suffix, l);                                                                   \
    if (!(L->field = need(f, nm, d0, d1, err, errlen))) goto fail;
        NEED(attn_norm, "attn_norm", D, 0)
        NEED(wq, "wq", D, D)
        NEED(wk, "wk", KV, D)
        NEED(wv, "wv", KV, D)
        NEED(wo, "wo", D, D)
        NEED(ffn_norm, "ffn_norm", D, 0)
        NEED(w1, "w1", H, D)
        NEED(w2, "w2", D, H)
        NEED(w3, "w3", H, D)
#undef NEED
    }

    /* tokenizer (optional merges) */
    const qlm_kv *mk = qlm_find_kv(f, "tokenizer.merges");
    uint32_t nm = (mk && mk->type == QLM_KV_ARR_U32) ? mk->len / 2 : 0;
    uint32_t *merges = malloc(sizeof(uint32_t) * 2 * (nm ? nm : 1));
    if (!merges) goto fail;
    for (uint32_t i = 0; i < 2 * nm; i++) merges[i] = qlm_arr_get(mk, i);
    int tok_rc = qm_tokenizer_init(&m->tok, merges, (int32_t)nm);
    free(merges);
    if (tok_rc != 0) {
        snprintf(err, errlen, "invalid tokenizer merges");
        goto fail;
    }

    /* runtime state from a single arena */
    int64_t S = c->max_seq, maxdim = D > H ? D : H;
    size_t need_bytes = 0;
    size_t sizes[] = {D * 4, D * 4, D * 4, D * 4, KV * 4, KV * 4, H * 4, H * 4, (size_t)c->n_heads * S * 4, V * 4,
                      (size_t)c->n_layers * S * KV * 4, (size_t)c->n_layers * S * KV * 4,
                      qm_nblocks(maxdim) * sizeof(qm_block_q8)};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) need_bytes += sizes[i] + QM_ALIGN;
    if (qm_arena_init(&m->arena, need_bytes) != 0) {
        snprintf(err, errlen, "out of memory for runtime state");
        goto fail;
    }
    float **bufs[] = {&m->x, &m->xb, &m->xb2, &m->q, &m->k, &m->v, &m->hb, &m->hb2, &m->att, &m->logits,
                      &m->key_cache, &m->val_cache};
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) *bufs[i] = qm_arena_alloc(&m->arena, sizes[i]);
    m->xq = qm_arena_alloc(&m->arena, sizes[12]);

    m->n_threads = n_threads < 1 ? 1 : n_threads;
    m->pool = qm_pool_create(m->n_threads);
    qm_kernels_active(); /* resolve dispatch on the main thread */
    return 0;
fail:
    qm_model_free(m);
    return -1;
}

void qm_model_free(qm_model *m) {
    qm_pool_destroy(m->pool);
    if (m->arena.base) qm_arena_free(&m->arena);
    qm_tokenizer_free(&m->tok);
    free(m->layers);
    qlm_close(&m->file);
    memset(m, 0, sizeof *m);
}

/* ---- attention, parallel over heads ---- */
typedef struct {
    qm_model *m;
    int32_t pos, layer;
} attn_ctx;

static void attn_task(void *vctx, int64_t h0, int64_t h1, int tid) {
    (void)tid;
    attn_ctx *a = vctx;
    qm_model *m = a->m;
    const qm_config *c = &m->cfg;
    const qm_kernels *k = qm_kernels_active();
    int64_t hd = c->dim / c->n_heads, kvd = hd * c->n_kv_heads, group = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    const float *kc = m->key_cache + (size_t)a->layer * c->max_seq * kvd;
    const float *vc = m->val_cache + (size_t)a->layer * c->max_seq * kvd;
    for (int64_t h = h0; h < h1; h++) {
        const float *q = m->q + h * hd;
        float *att = m->att + h * c->max_seq;
        int64_t kvh = h / group;
        for (int32_t t = 0; t <= a->pos; t++) att[t] = k->dot_f32(q, kc + (size_t)t * kvd + kvh * hd, hd) * scale;
        qm_softmax(att, a->pos + 1);
        float *out = m->xb + h * hd;
        for (int64_t i = 0; i < hd; i++) out[i] = 0.0f;
        for (int32_t t = 0; t <= a->pos; t++) {
            const float *v = vc + (size_t)t * kvd + kvh * hd;
            float w = att[t];
            for (int64_t i = 0; i < hd; i++) out[i] += w * v[i];
        }
    }
}

float *qm_model_forward(qm_model *m, int32_t token, int32_t pos) {
    const qm_config *c = &m->cfg;
    if (token < 0 || token >= c->vocab_size || pos < 0 || pos >= c->max_seq) return NULL;
    int64_t D = c->dim, H = c->hidden_dim, hd = D / c->n_heads, kvd = hd * c->n_kv_heads;
    qm_row_to_f32(m->tok_embd->type, qm_tensor_row(m->tok_embd, token), m->x, D);

    for (int l = 0; l < c->n_layers; l++) {
        const qm_layer *L = &m->layers[l];
        qm_rmsnorm(m->xb, m->x, L->attn_norm->data, D, c->norm_eps);
        float *kslot = m->key_cache + ((size_t)l * c->max_seq + pos) * kvd;
        float *vslot = m->val_cache + ((size_t)l * c->max_seq + pos) * kvd;
        qm_matvec(m->pool, L->wq, m->xb, m->q, m->xq);
        qm_matvec(m->pool, L->wk, m->xb, kslot, m->xq);
        qm_matvec(m->pool, L->wv, m->xb, vslot, m->xq);
        qm_rope(m->q, c->n_heads, hd, pos, c->rope_theta);
        qm_rope(kslot, c->n_kv_heads, hd, pos, c->rope_theta);

        attn_ctx ac = {m, pos, l};
        qm_pool_run(m->pool, c->n_heads, 1, attn_task, &ac);

        qm_matvec(m->pool, L->wo, m->xb, m->xb2, m->xq);
        for (int64_t i = 0; i < D; i++) m->x[i] += m->xb2[i];

        qm_rmsnorm(m->xb, m->x, L->ffn_norm->data, D, c->norm_eps);
        qm_matvec(m->pool, L->w1, m->xb, m->hb, m->xq);
        qm_matvec(m->pool, L->w3, m->xb, m->hb2, m->xq);
        qm_swiglu(m->hb, m->hb2, H);
        qm_matvec(m->pool, L->w2, m->hb, m->xb, m->xq);
        for (int64_t i = 0; i < D; i++) m->x[i] += m->xb[i];
    }
    qm_rmsnorm(m->x, m->x, m->norm->data, D, c->norm_eps);
    qm_matvec(m->pool, m->output, m->x, m->logits, m->xq);
    return m->logits;
}
