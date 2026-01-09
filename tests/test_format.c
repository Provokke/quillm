/* .qlm loader: metadata/tensor table, plus truncation and mutation fuzzing (never crash,
 * never return a tensor outside the buffer). Run under ASan via `make asan`. */
#include "qlm_file.h"
#include "test.h"

static int tensors_in_bounds(const qlm_file *f, const uint8_t *buf, size_t size) {
    for (uint32_t i = 0; i < f->n_tensors; i++) {
        const uint8_t *d = f->tensors[i].data;
        size_t b = qm_tensor_bytes(&f->tensors[i]);
        if (d < buf || d + b > buf + size) return 0;
    }
    return 1;
}

int main(void) {
    char p[512], err[256];
    snprintf(p, sizeof p, "%s/small_q4_0.qlm", fixtures_dir());
    qlm_file f;
    CHECK(qlm_open(&f, p, err, sizeof err) == 0, "open: %s", err);
    const qlm_kv *kv = qlm_find_kv(&f, "llama.dim");
    CHECK(kv && kv->type == QLM_KV_U32 && kv->u32 == 64, "llama.dim");
    kv = qlm_find_kv(&f, "general.name");
    CHECK(kv && kv->type == QLM_KV_STR && kv->len == 13 && memcmp(kv->str, "quillm-random", 13) == 0, "name");
    const qm_tensor *t = qlm_find_tensor(&f, "layers.1.w2");
    CHECK(t && t->type == QM_Q4_0 && t->ndim == 2 && t->ne[0] == 64 && t->ne[1] == 172, "w2 shape");
    CHECK(t && ((uintptr_t)t->data % 64) == 0, "tensor data 64-byte aligned");
    t = qlm_find_tensor(&f, "norm");
    CHECK(t && t->type == QM_F32 && t->ndim == 1, "norm");
    CHECK(qlm_find_tensor(&f, "nope") == NULL, "missing tensor");
    qlm_close(&f);
    CHECK(qlm_open(&f, "/nonexistent/x.qlm", err, sizeof err) != 0, "nonexistent");

    size_t size;
    uint8_t *orig = read_file(p, &size);
    CHECK(orig != NULL, "read");
    if (!orig) TEST_DONE();
    uint8_t *buf = malloc(size);
    /* header size: parse once to find the data offset */
    uint64_t data_off;
    memcpy(&data_off, orig + 16, 8);

    /* every truncation of the header region must fail cleanly */
    int ok = 1;
    for (size_t n = 0; n < data_off + 64 && n < size; n++) {
        memcpy(buf, orig, n);
        if (qlm_parse(&f, buf, n, err, sizeof err) == 0) {
            ok &= tensors_in_bounds(&f, buf, n);
            qlm_close(&f);
        }
    }
    CHECK(ok, "truncation");
    /* truncating the tensor data must be detected */
    CHECK(qlm_parse(&f, orig, size - 1000, err, sizeof err) != 0, "truncated data accepted");

    /* random header mutations */
    int iters = quick() ? 300 : 30000, accepted = 0;
    ok = 1;
    for (int it = 0; it < iters; it++) {
        memcpy(buf, orig, size);
        int nmut = rnd_int(1, 6);
        for (int m = 0; m < nmut; m++) {
            size_t pos = (size_t)rnd_int(0, (int)data_off - 1);
            switch (rnd_int(0, 2)) {
            case 0: buf[pos] ^= (uint8_t)(1u << rnd_int(0, 7)); break;
            case 1: buf[pos] = (uint8_t)rnd_int(0, 255); break;
            default: buf[pos] = 0xff; break;
            }
        }
        if (qlm_parse(&f, buf, size, err, sizeof err) == 0) {
            accepted++;
            ok &= tensors_in_bounds(&f, buf, size);
            qlm_close(&f);
        }
    }
    CHECK(ok, "mutated file produced an out-of-bounds tensor");
    printf("fuzz: %d mutated headers, %d still parsed (all in bounds)\n", iters, accepted);

    /* specific corruptions */
    memcpy(buf, orig, size);
    buf[0] = 'X';
    CHECK(qlm_parse(&f, buf, size, err, sizeof err) != 0 && strstr(err, "magic"), "bad magic: %s", err);
    memcpy(buf, orig, size);
    buf[4] = 9;
    CHECK(qlm_parse(&f, buf, size, err, sizeof err) != 0 && strstr(err, "version"), "bad version: %s", err);
    free(buf);

    free(orig);
    TEST_DONE();
}
