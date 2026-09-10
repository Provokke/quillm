/* Sampling: greedy, temperature, top-k, top-p distributions and determinism. */
#include "sampler.h"
#include "test.h"

int main(void) {
    qm_arena a;
    qm_arena_init(&a, 1 << 20);
    float logits[6] = {1.0f, 3.0f, 0.5f, 2.5f, -1.0f, 3.0f};
    qm_sampler s;

    qm_sampler_init(&s, &a, 6, 0.0f, 0, 1.0f, 1);
    CHECK(qm_sample(&s, logits) == 1, "greedy picks first max");
    qm_sampler_init(&s, &a, 6, 1.0f, 1, 1.0f, 1);
    CHECK(qm_sample(&s, logits) == 1, "top_k=1 == greedy");
    qm_sampler_init(&s, &a, 6, 1.0f, 0, 1e-6f, 1);
    for (int i = 0; i < 100; i++) CHECK(qm_sample(&s, logits) == 1, "tiny top_p == greedy (tie broken by index)");

    /* plain temperature sampling matches softmax(logits / T) */
    int N = quick() ? 20000 : 400000;
    float T = 0.7f;
    double pr[6], z = 0;
    for (int i = 0; i < 6; i++) z += (pr[i] = exp(logits[i] / T));
    for (int i = 0; i < 6; i++) pr[i] /= z;
    int cnt[6] = {0};
    qm_sampler_init(&s, &a, 6, T, 0, 1.0f, 123);
    for (int i = 0; i < N; i++) cnt[qm_sample(&s, logits)]++;
    for (int i = 0; i < 6; i++)
        CHECK(fabs(cnt[i] / (double)N - pr[i]) < 4 * sqrt(pr[i] * (1 - pr[i]) / N) + 1e-3, "temp freq %d: %.4f vs %.4f", i,
              cnt[i] / (double)N, pr[i]);

    /* top-k=3: only {1,5,3} may appear, renormalised */
    memset(cnt, 0, sizeof cnt);
    qm_sampler_init(&s, &a, 6, 1.0f, 3, 1.0f, 7);
    for (int i = 0; i < N; i++) cnt[qm_sample(&s, logits)]++;
    CHECK(cnt[0] == 0 && cnt[2] == 0 && cnt[4] == 0, "top-k leaked");
    double e3 = exp(2.5), e1 = exp(3.0);
    CHECK(fabs(cnt[3] / (double)N - e3 / (2 * e1 + e3)) < 0.01, "top-k renormalisation %.4f", cnt[3] / (double)N);

    /* top-p=0.6: probs (T=1) sorted: 1,5 ~0.33 each, 3 ~0.20 -> nucleus {1,5} */
    memset(cnt, 0, sizeof cnt);
    qm_sampler_init(&s, &a, 6, 1.0f, 0, 0.6f, 9);
    for (int i = 0; i < N; i++) cnt[qm_sample(&s, logits)]++;
    CHECK(cnt[1] + cnt[5] == N && fabs(cnt[1] / (double)N - 0.5) < 0.01, "top-p nucleus");

    /* determinism: same seed -> same sequence; different seed -> different */
    int seq1[200], seq2[200], seq3[200];
    qm_sampler_init(&s, &a, 6, 1.5f, 0, 1.0f, 99);
    for (int i = 0; i < 200; i++) seq1[i] = qm_sample(&s, logits);
    qm_sampler_init(&s, &a, 6, 1.5f, 0, 1.0f, 99);
    for (int i = 0; i < 200; i++) seq2[i] = qm_sample(&s, logits);
    qm_sampler_init(&s, &a, 6, 1.5f, 0, 1.0f, 100);
    for (int i = 0; i < 200; i++) seq3[i] = qm_sample(&s, logits);
    CHECK(memcmp(seq1, seq2, sizeof seq1) == 0, "same seed differs");
    CHECK(memcmp(seq1, seq3, sizeof seq1) != 0, "different seeds identical");

    /* rng is in [0,1) */
    uint64_t st = 5;
    int ok = 1;
    for (int i = 0; i < 100000; i++) {
        float u = qm_rng_uniform(&st);
        ok &= u >= 0.0f && u < 1.0f;
    }
    CHECK(ok, "rng range");
    /* near-uniform logits with a tiny top_p: prefilter would drop everything -> fallback */
    float flat[1000];
    for (int i = 0; i < 1000; i++) flat[i] = 0.0f;
    qm_sampler_init(&s, &a, 1000, 1.0f, 0, 1e-6f, 4);
    for (int i = 0; i < 100; i++) {
        int id = qm_sample(&s, flat);
        CHECK(id == 0, "uniform + tiny top_p must pick the first-ranked token, got %d", id);
    }
    /* top-p prefilter path agrees in distribution with the unfiltered top-k=V path */
    {
        float lg[300];
        for (int i = 0; i < 300; i++) lg[i] = rnd_normal() * 2;
        int c1[300] = {0}, c2[300] = {0};
        qm_sampler s1, s2;
        qm_sampler_init(&s1, &a, 300, 1.0f, 0, 0.8f, 11);   /* prefiltered */
        qm_sampler_init(&s2, &a, 300, 1.0f, 300, 0.8f, 11); /* no prefilter (top_k set) */
        for (int i = 0; i < 20000; i++) {
            c1[qm_sample(&s1, lg)]++;
            c2[qm_sample(&s2, lg)]++;
        }
        CHECK(memcmp(c1, c2, sizeof c1) == 0, "prefiltered top-p differs from reference path");
    }
    /* large vocab with top-k + top-p together */
    int V = 5000;
    float *big = malloc(sizeof(float) * (size_t)V);
    for (int i = 0; i < V; i++) big[i] = rnd_normal() * 3;
    qm_sampler_init(&s, &a, V, 0.9f, 40, 0.9f, 3);
    int top = qm_argmax(big, V), seen_top = 0;
    for (int i = 0; i < 2000; i++) {
        int id = qm_sample(&s, big);
        CHECK(id >= 0 && id < V, "range");
        seen_top |= id == top;
        /* must be within the top 40 */
        int rank = 0;
        for (int j = 0; j < V; j++) rank += big[j] > big[id];
        if (rank >= 40) CHECK(0, "token outside top-k (rank %d)", rank);
    }
    CHECK(seen_top, "argmax never sampled");
    free(big);
    qm_arena_free(&a);
    TEST_DONE();
}
