/* End-to-end forward pass vs the numpy reference (tools/qlm.py:forward_ref), for f32,
 * Q8_0 and Q4_0 models, every ISA, multiple thread counts; plus determinism. */
#include "model.h"
#include "sampler.h"
#include "test.h"

typedef struct {
    const char *name;
    double tol_deq; /* max|err| / max|ref| vs reference using the same (dequantized) weights */
    double tol_fp;  /* ... vs reference using the original fp32 weights */
} model_case;

static int load_tokens(const char *name, int32_t *toks, int *T, int *V) {
    char p[512];
    snprintf(p, sizeof p, "%s/%s.tokens", fixtures_dir(), name);
    FILE *f = fopen(p, "r");
    if (!f) return -1;
    int ok = fscanf(f, "%d %d", T, V) == 2;
    for (int i = 0; ok && i < *T; i++) ok = fscanf(f, "%d", &toks[i]) == 1;
    fclose(f);
    return ok ? 0 : -1;
}

static float *run_all(const char *name, int threads, int T, int V, const int32_t *toks) {
    char p[512], err[256];
    snprintf(p, sizeof p, "%s/%s.qlm", fixtures_dir(), name);
    qm_model m;
    if (qm_model_load(&m, p, threads, 0, err, sizeof err) != 0) {
        fprintf(stderr, "load %s: %s\n", name, err);
        return NULL;
    }
    float *out = malloc(sizeof(float) * (size_t)T * V);
    for (int t = 0; t < T; t++) memcpy(out + (size_t)t * V, qm_model_forward(&m, toks[t], t), sizeof(float) * (size_t)V);
    qm_model_free(&m);
    return out;
}

static double rel_err(const float *a, const float *ref, size_t n, int *argmax_agree, int T, int V) {
    double me = 0, mr = 0;
    for (size_t i = 0; i < n; i++) {
        me = fmax(me, fabs((double)a[i] - ref[i]));
        mr = fmax(mr, fabs((double)ref[i]));
    }
    *argmax_agree = 0;
    for (int t = 0; t < T; t++) *argmax_agree += qm_argmax(a + (size_t)t * V, V) == qm_argmax(ref + (size_t)t * V, V);
    return me / mr;
}

int main(void) {
    model_case cases[] = {
        {"small_f32", 1e-5, 1e-5}, {"odd_f32", 1e-5, 1e-5},   {"l2c_f32", 1e-5, 1e-5},
        {"small_q8_0", 2e-2, 3e-2}, {"odd_q8_0", 2e-2, 3e-2}, {"small_q4_0", 2e-2, 0.5},
        {"odd_q4_0", 2e-2, 0.5},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const char *name = cases[c].name;
        int32_t toks[256];
        int T, V;
        CHECK(load_tokens(name, toks, &T, &V) == 0, "%s tokens fixture", name);
        char p[512];
        size_t sz;
        snprintf(p, sizeof p, "%s/%s.ref.f32", fixtures_dir(), name);
        float *ref = read_file(p, &sz);
        snprintf(p, sizeof p, "%s/%s.reffp.f32", fixtures_dir(), name);
        float *reffp = read_file(p, &sz);
        if (!ref || !reffp) {
            CHECK(0, "missing reference for %s", name);
            continue;
        }
        float *base = NULL;
        for (int isa = 0; isa < QM_ISA_COUNT; isa++) {
            if (qm_kernels_select((qm_isa)isa) != 0) continue;
            float *got = run_all(name, 1, T, V, toks);
            CHECK(got != NULL, "%s load", name);
            if (!got) continue;
            int agree, agree_fp;
            double e = rel_err(got, ref, (size_t)T * V, &agree, T, V);
            double efp = rel_err(got, reffp, (size_t)T * V, &agree_fp, T, V);
            printf("%-11s %-6s rel.err vs numpy(same weights) %.2e  vs numpy(fp32) %.2e  argmax agree %d/%d\n", name,
                   qm_isa_name((qm_isa)isa), e, efp, agree_fp, T);
            CHECK(e < cases[c].tol_deq, "%s/%s error %.3e >= %.1e", name, qm_isa_name((qm_isa)isa), e, cases[c].tol_deq);
            CHECK(efp < cases[c].tol_fp, "%s/%s fp32 error %.3e >= %.1e", name, qm_isa_name((qm_isa)isa), efp, cases[c].tol_fp);
            if (cases[c].tol_deq <= 1e-5) CHECK(agree == T, "%s argmax mismatch", name);
            /* thread-count invariance: bitwise identical logits */
            for (int nt = 2; nt <= (quick() ? 2 : 4); nt++) {
                float *gt = run_all(name, nt, T, V, toks);
                CHECK(gt && memcmp(gt, got, sizeof(float) * (size_t)T * V) == 0, "%s/%s threads=%d not bitwise equal", name,
                      qm_isa_name((qm_isa)isa), nt);
                free(gt);
            }
            /* run-to-run determinism */
            float *again = run_all(name, 1, T, V, toks);
            CHECK(again && memcmp(again, got, sizeof(float) * (size_t)T * V) == 0, "%s rerun differs", name);
            free(again);
            if (!base) base = got;
            else free(got);
        }
        free(base);
        free(ref);
        free(reffp);
        qm_kernels_select(qm_cpu_best());
    }

    /* generation determinism through the sampler, and error handling */
    char p[512], err[256];
    snprintf(p, sizeof p, "%s/small_q8_0.qlm", fixtures_dir());
    int seqs[2][40];
    for (int r = 0; r < 2; r++) {
        qm_model m;
        CHECK(qm_model_load(&m, p, 3, 0, err, sizeof err) == 0, "load: %s", err);
        qm_arena a;
        qm_arena_init(&a, 1 << 16);
        qm_sampler s;
        qm_sampler_init(&s, &a, m.cfg.vocab_size, 0.9f, 20, 0.9f, 1234);
        int32_t tok = QM_TOK_BOS;
        for (int i = 0; i < 40; i++) {
            float *lg = qm_model_forward(&m, tok, i);
            tok = qm_sample(&s, lg);
            seqs[r][i] = tok;
        }
        CHECK(qm_model_forward(&m, 0, m.cfg.max_seq) == NULL, "pos beyond context rejected");
        CHECK(qm_model_forward(&m, m.cfg.vocab_size, 0) == NULL, "bad token rejected");
        qm_arena_free(&a);
        qm_model_free(&m);
    }
    CHECK(memcmp(seqs[0], seqs[1], sizeof seqs[0]) == 0, "sampled generation not deterministic");
    snprintf(p, sizeof p, "%s/merges.u32", fixtures_dir());
    qm_model bad;
    CHECK(qm_model_load(&bad, p, 1, 0, err, sizeof err) != 0, "non-model file accepted");
    TEST_DONE();
}
