#include "sampler.h"

#include <math.h>
#include <stddef.h>

int qm_sampler_init(qm_sampler *s, qm_arena *a, int32_t n_vocab, float temperature, int top_k, float top_p,
                    uint64_t seed) {
    s->temperature = temperature < 0 ? 0 : temperature;
    s->top_k = top_k < 0 ? 0 : top_k;
    s->top_p = (top_p <= 0 || top_p > 1) ? 1.0f : top_p;
    s->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
    s->n_vocab = n_vocab;
    s->probs = qm_arena_alloc(a, sizeof(float) * (size_t)n_vocab);
    s->idx = qm_arena_alloc(a, sizeof(int32_t) * (size_t)n_vocab);
    return (s->probs && s->idx) ? 0 : -1;
}

float qm_rng_uniform(uint64_t *state) { /* xorshift64* */
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

int32_t qm_argmax(const float *x, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* heap sift-down on idx by descending prob (min-heap of the current top-k) */
static int less(const float *p, int32_t a, int32_t b) { return p[a] < p[b] || (p[a] == p[b] && a > b); }

static void sift_down(int32_t *h, int32_t n, int32_t i, const float *p) {
    for (;;) {
        int32_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && less(p, h[l], h[m])) m = l;
        if (r < n && less(p, h[r], h[m])) m = r;
        if (m == i) return;
        int32_t t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

int32_t qm_sample(qm_sampler *s, const float *logits) {
    int32_t n = s->n_vocab;
    if (s->temperature == 0.0f || s->top_k == 1) return qm_argmax(logits, n);
    float *p = s->probs;
    float mx = logits[0];
    for (int32_t i = 1; i < n; i++)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int32_t i = 0; i < n; i++) {
        p[i] = expf((logits[i] - mx) / s->temperature);
        sum += p[i];
    }
    for (int32_t i = 0; i < n; i++) p[i] = (float)(p[i] / sum);

    /* candidates: top-k via a size-k min-heap, then sorted descending (heap sort) */
    int32_t *h = s->idx, nc = 0;
    if (s->top_k == 0 && s->top_p < 1.0f && n > 1) {
        /* a token with p < (1-top_p)/(n-1) can never be inside the nucleus: prefilter */
        float cut = (1.0f - s->top_p) / (float)(n - 1);
        for (int32_t i = 0; i < n; i++)
            if (p[i] >= cut) h[nc++] = i;
    }
    if (nc == 0) { /* no prefilter, or it removed everything (near-uniform, tiny top_p) */
        for (int32_t i = 0; i < n; i++) h[nc++] = i;
    }
    int32_t k = (s->top_k > 0 && s->top_k < nc) ? s->top_k : nc;
    for (int32_t i = k / 2 - 1; i >= 0; i--) sift_down(h, k, i, p);
    for (int32_t j = k; j < nc; j++)
        if (less(p, h[0], h[j])) {
            h[0] = h[j];
            sift_down(h, k, 0, p);
        }
    for (int32_t end = k - 1; end > 0; end--) { /* pop min to the back -> descending order */
        int32_t t = h[0];
        h[0] = h[end];
        h[end] = t;
        sift_down(h, end, 0, p);
    }
    /* top-p: smallest prefix with cumulative mass >= top_p */
    double cum = 0;
    int32_t m = k;
    for (int32_t i = 0; i < k; i++) {
        cum += p[h[i]];
        if (cum >= s->top_p) {
            m = i + 1;
            break;
        }
    }
    double mass = 0;
    for (int32_t i = 0; i < m; i++) mass += p[h[i]];
    double u = (double)qm_rng_uniform(&s->rng) * mass, c = 0;
    for (int32_t i = 0; i < m; i++) {
        c += p[h[i]];
        if (u < c) return h[i];
    }
    return h[m - 1];
}
