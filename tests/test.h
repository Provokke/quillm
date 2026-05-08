/* Tiny test harness shared by all C tests. */
#ifndef QM_TEST_H
#define QM_TEST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_fails;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        g_checks++;                                                                                                    \
        if (!(cond)) {                                                                                                 \
            g_fails++;                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);                                            \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fprintf(stderr, "\n");                                                                                     \
        }                                                                                                              \
    } while (0)

#define TEST_DONE()                                                                                                    \
    do {                                                                                                               \
        printf("%s: %d checks, %d failures\n", __FILE__, g_checks, g_fails);                                           \
        return g_fails ? 1 : 0;                                                                                        \
    } while (0)

/* deterministic test RNG (splitmix64) */
static unsigned long long g_rng = 0x1234567ULL;
static inline unsigned long long rnd_u64(void) {
    unsigned long long z = (g_rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static inline float rnd_unit(void) { return (float)(rnd_u64() >> 40) / 16777216.0f; }
static inline float rnd_normal(void) {
    float u = rnd_unit() + 1e-7f, v = rnd_unit();
    return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}
static inline int rnd_int(int lo, int hi) { return lo + (int)(rnd_u64() % (unsigned long long)(hi - lo + 1)); }

/* QM_QUICK=1 shrinks iteration counts (used under valgrind) */
static inline int quick(void) { return getenv("QM_QUICK") != NULL; }

static inline const char *fixtures_dir(void) {
    const char *d = getenv("QM_FIXTURES");
    return d ? d : "build/fixtures";
}

static inline void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (size) *size = (size_t)n;
    return buf;
}

#endif
