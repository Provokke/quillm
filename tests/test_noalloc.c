/* Proves the decode hot loop (forward + sampling) performs zero heap allocations:
 * malloc/calloc/realloc/aligned_alloc are wrapped at link time and counted. */
#include "model.h"
#include "sampler.h"
#include "test.h"

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void *__real_aligned_alloc(size_t, size_t);
static volatile long g_allocs;
void *__wrap_malloc(size_t n) { g_allocs++; return __real_malloc(n); }
void *__wrap_calloc(size_t a, size_t b) { g_allocs++; return __real_calloc(a, b); }
void *__wrap_realloc(void *p, size_t n) { g_allocs++; return __real_realloc(p, n); }
void *__wrap_aligned_alloc(size_t a, size_t n) { g_allocs++; return __real_aligned_alloc(a, n); }

int main(void) {
    const char *names[] = {"small_f32", "small_q8_0", "odd_q4_0"};
    for (int c = 0; c < 3; c++) {
        char p[512], err[256];
        snprintf(p, sizeof p, "%s/%s.qlm", fixtures_dir(), names[c]);
        qm_model m;
        long before = g_allocs;
        CHECK(qm_model_load(&m, p, 4, 0, err, sizeof err) == 0, "load %s", err);
        CHECK(g_allocs > before, "wrapping inactive?");
        qm_arena a;
        qm_arena_init(&a, 1 << 16);
        qm_sampler s;
        qm_sampler_init(&s, &a, m.cfg.vocab_size, 0.8f, 10, 0.9f, 1);
        long start = g_allocs;
        int32_t tok = QM_TOK_BOS;
        for (int pos = 0; pos < m.cfg.max_seq; pos++) tok = qm_sample(&s, qm_model_forward(&m, tok, pos));
        long hot = g_allocs - start;
        CHECK(hot == 0, "%s: %ld heap allocations during %d decode steps", names[c], hot, m.cfg.max_seq);
        printf("%s: %d decode steps, %ld heap allocations (arena peak %zu bytes)\n", names[c], m.cfg.max_seq, hot, m.arena.peak);
        qm_arena_free(&a);
        qm_model_free(&m);
    }
    TEST_DONE();
}
