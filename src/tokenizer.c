#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

static uint64_t hash64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static int32_t lookup(const qm_tokenizer *t, int32_t a, int32_t b) {
    if (!t->ht_cap) return -1;
    uint64_t key = (((uint64_t)(uint32_t)a << 32) | (uint32_t)b) + 1;
    uint32_t i = (uint32_t)(hash64(key) & (t->ht_cap - 1));
    while (t->ht_keys[i]) {
        if (t->ht_keys[i] == key) return t->ht_vals[i];
        i = (i + 1) & (t->ht_cap - 1);
    }
    return -1;
}

int qm_tokenizer_init(qm_tokenizer *t, const uint32_t *merges, int32_t n_merges) {
    memset(t, 0, sizeof *t);
    if (n_merges < 0 || n_merges > (1 << 22)) return -1;
    t->n_merges = n_merges;
    t->n_vocab = QM_TOK_FIRST_MERGE + n_merges;
    t->merges = malloc(sizeof(int32_t) * 2 * (size_t)(n_merges ? n_merges : 1));
    t->tok_off = malloc(sizeof(uint32_t) * (size_t)t->n_vocab);
    t->tok_len = malloc(sizeof(uint32_t) * (size_t)t->n_vocab);
    uint32_t cap = 16;
    while (cap < 2u * (uint32_t)n_merges + 16) cap <<= 1;
    t->ht_cap = cap;
    t->ht_keys = calloc(cap, sizeof(uint64_t));
    t->ht_vals = calloc(cap, sizeof(int32_t));
    if (!t->merges || !t->tok_off || !t->tok_len || !t->ht_keys || !t->ht_vals) goto fail;

    /* token texts: first pass computes lengths */
    uint64_t total = 256;
    t->tok_len[0] = t->tok_len[1] = 0;
    for (int i = 0; i < 256; i++) t->tok_len[QM_TOK_BYTE0 + i] = 1;
    for (int32_t m = 0; m < n_merges; m++) {
        uint32_t a = merges[2 * m], b = merges[2 * m + 1];
        int32_t id = QM_TOK_FIRST_MERGE + m;
        /* a merge may only reference byte tokens or earlier merges */
        if (a < QM_TOK_BYTE0 || b < QM_TOK_BYTE0 || a >= (uint32_t)id || b >= (uint32_t)id) goto fail;
        t->merges[2 * m] = (int32_t)a;
        t->merges[2 * m + 1] = (int32_t)b;
        uint64_t len = (uint64_t)t->tok_len[a] + t->tok_len[b];
        if (len > (1u << 20)) goto fail;
        t->tok_len[id] = (uint32_t)len;
        total += len;
    }
    if (total > (1u << 30)) goto fail;
    t->tok_text = malloc((size_t)total);
    if (!t->tok_text) goto fail;
    uint32_t off = 0;
    t->tok_off[0] = t->tok_off[1] = 0;
    for (int i = 0; i < 256; i++) {
        t->tok_off[QM_TOK_BYTE0 + i] = off;
        t->tok_text[off++] = (uint8_t)i;
    }
    for (int32_t m = 0; m < n_merges; m++) {
        int32_t a = t->merges[2 * m], b = t->merges[2 * m + 1], id = QM_TOK_FIRST_MERGE + m;
        t->tok_off[id] = off;
        memcpy(t->tok_text + off, t->tok_text + t->tok_off[a], t->tok_len[a]);
        memcpy(t->tok_text + off + t->tok_len[a], t->tok_text + t->tok_off[b], t->tok_len[b]);
        off += t->tok_len[id];
        /* insert pair -> rank (first occurrence wins) */
        if (lookup(t, a, b) < 0) {
            uint64_t key = (((uint64_t)(uint32_t)a << 32) | (uint32_t)b) + 1;
            uint32_t i = (uint32_t)(hash64(key) & (cap - 1));
            while (t->ht_keys[i]) i = (i + 1) & (cap - 1);
            t->ht_keys[i] = key;
            t->ht_vals[i] = m;
        }
    }
    return 0;
fail:
    qm_tokenizer_free(t);
    return -1;
}

void qm_tokenizer_free(qm_tokenizer *t) {
    free(t->merges);
    free(t->tok_off);
    free(t->tok_len);
    free(t->tok_text);
    free(t->ht_keys);
    free(t->ht_vals);
    memset(t, 0, sizeof *t);
}

int qm_tokenizer_encode(const qm_tokenizer *t, const char *text, size_t len, int add_bos, int32_t *out,
                        int max_out) {
    int n = 0;
    if (add_bos) {
        if (max_out < 1) return -1;
        out[n++] = QM_TOK_BOS;
    }
    if (len > (size_t)(max_out - n)) return -1;
    int start = n;
    for (size_t i = 0; i < len; i++) out[n++] = QM_TOK_BYTE0 + (uint8_t)text[i];
    for (;;) {
        int32_t best = -1;
        for (int i = start; i + 1 < n; i++) {
            int32_t r = lookup(t, out[i], out[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) best = r;
        }
        if (best < 0) break;
        int32_t a = t->merges[2 * best], b = t->merges[2 * best + 1];
        int w = start;
        for (int i = start; i < n; i++) {
            if (i + 1 < n && out[i] == a && out[i + 1] == b) {
                out[w++] = QM_TOK_FIRST_MERGE + best;
                i++;
            } else {
                out[w++] = out[i];
            }
        }
        n = w;
    }
    return n;
}

const uint8_t *qm_tokenizer_piece(const qm_tokenizer *t, int32_t id, uint32_t *len) {
    if (id < QM_TOK_BYTE0 || id >= t->n_vocab) {
        *len = 0;
        return (const uint8_t *)"";
    }
    *len = t->tok_len[id];
    return t->tok_text + t->tok_off[id];
}

int qm_tokenizer_decode(const qm_tokenizer *t, const int32_t *ids, int n, char *buf, size_t cap) {
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        uint32_t len;
        const uint8_t *p = qm_tokenizer_piece(t, ids[i], &len);
        if (w + len + 1 > cap) return -1;
        memcpy(buf + w, p, len);
        w += len;
    }
    if (w + 1 > cap) return -1;
    buf[w] = 0;
    return (int)w;
}
