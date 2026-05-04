/* Byte-level BPE tokenizer.
 * ids: 0 = BOS, 1 = EOS, 2..257 = raw bytes 0..255, 258+i = merge i (pair of earlier ids).
 * Encoding repeatedly merges the adjacent pair with the lowest merge rank (all occurrences,
 * left to right) until no mergeable pair remains -- mirrored exactly by tools/qlm.py. */
#ifndef QM_TOKENIZER_H
#define QM_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define QM_TOK_BOS 0
#define QM_TOK_EOS 1
#define QM_TOK_BYTE0 2
#define QM_TOK_FIRST_MERGE 258

typedef struct {
    int32_t n_merges, n_vocab;

    int32_t *merges;      /* [n_merges][2] */
    uint32_t *tok_off;    /* byte offset of each token's text in tok_text */
    uint32_t *tok_len;
    uint8_t *tok_text;
    uint64_t *ht_keys;    /* open-addressing hash: (a<<32|b)+1 -> rank */
    int32_t *ht_vals;
    uint32_t ht_cap;
} qm_tokenizer;

/* merges: flat array of 2*n_merges ids */
int qm_tokenizer_init(qm_tokenizer *t, const uint32_t *merges, int32_t n_merges);
void qm_tokenizer_free(qm_tokenizer *t);
/* returns number of tokens written (<= max_out), or -1 if out is too small */
int qm_tokenizer_encode(const qm_tokenizer *t, const char *text, size_t len, int add_bos, int32_t *out,
                        int max_out);
/* bytes for a token (empty for BOS/EOS/unknown ids) */
const uint8_t *qm_tokenizer_piece(const qm_tokenizer *t, int32_t id, uint32_t *len);
/* decode a sequence into buf (NUL terminated); returns bytes written or -1 on overflow */
int qm_tokenizer_decode(const qm_tokenizer *t, const int32_t *ids, int n, char *buf, size_t cap);

#endif
