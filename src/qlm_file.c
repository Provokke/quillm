/* Bounds-checked .qlm parser + mmap loader. Every read is validated so malformed files
 * (truncation, bogus counts, overlapping/out-of-range tensors) are rejected, never crash. */
#include "qlm_file.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const uint8_t *p;
    size_t size, pos;
    int bad;

} reader;

static int rd(reader *r, void *dst, size_t n) {
    if (r->bad || n > r->size - r->pos) {
        r->bad = 1;
        return -1;
    }
    memcpy(dst, r->p + r->pos, n);
    r->pos += n;
    return 0;
}

static uint32_t rd_u32(reader *r) {
    uint8_t b[4] = {0};
    rd(r, b, 4);
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

static uint64_t rd_u64(reader *r) {
    uint64_t lo = rd_u32(r), hi = rd_u32(r);
    return lo | hi << 32;
}

static const uint8_t *rd_span(reader *r, size_t n) {
    if (r->bad || n > r->size - r->pos) {
        r->bad = 1;
        return NULL;
    }
    const uint8_t *s = r->p + r->pos;
    r->pos += n;
    return s;
}

static int fail(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static int rd_name(reader *r, char *dst, size_t cap) {
    uint32_t len = rd_u32(r);
    if (r->bad || len == 0 || len >= cap) return -1;
    const uint8_t *s = rd_span(r, len);
    if (!s) return -1;
    memcpy(dst, s, len);
    dst[len] = 0;
    if (strlen(dst) != len) return -1; /* embedded NUL */
    return 0;
}

int qlm_parse(qlm_file *f, const uint8_t *buf, size_t size, char *err, size_t errlen) {
    memset(f, 0, sizeof *f);
    reader r = {buf, size, 0, 0};
    char magic[4];
    if (rd(&r, magic, 4) || memcmp(magic, QLM_MAGIC, 4) != 0) return fail(err, errlen, "bad magic");
    uint32_t ver = rd_u32(&r);
    if (ver != QLM_VERSION) return fail(err, errlen, "unsupported version %u", ver);
    uint32_t n_kv = rd_u32(&r), n_t = rd_u32(&r);
    uint64_t data_off = rd_u64(&r);
    if (r.bad) return fail(err, errlen, "truncated header");
    if (n_kv > QLM_MAX_KV || n_t > QLM_MAX_TENSORS) return fail(err, errlen, "too many entries");
    if (data_off % QM_ALIGN || data_off > size) return fail(err, errlen, "bad data offset");

    f->kv = calloc(n_kv ? n_kv : 1, sizeof(qlm_kv));
    f->tensors = calloc(n_t ? n_t : 1, sizeof(qm_tensor));
    if (!f->kv || !f->tensors) {
        qlm_close(f);
        return fail(err, errlen, "out of memory");
    }
    f->n_kv = n_kv;
    f->n_tensors = n_t;

    for (uint32_t i = 0; i < n_kv; i++) {
        qlm_kv *kv = &f->kv[i];
        if (rd_name(&r, kv->key, sizeof kv->key)) goto bad;
        kv->type = rd_u32(&r);
        switch (kv->type) {
        case QLM_KV_U32: kv->u32 = rd_u32(&r); break;
        case QLM_KV_F32: {
            uint32_t b = rd_u32(&r);
            memcpy(&kv->f32, &b, 4);
            break;
        }
        case QLM_KV_STR:
            kv->len = rd_u32(&r);
            kv->str = (const char *)rd_span(&r, kv->len);
            break;
        case QLM_KV_ARR_U32:
            kv->len = rd_u32(&r);
            if (kv->len > (size - r.pos) / 4) goto bad;
            kv->arr = rd_span(&r, (size_t)kv->len * 4);
            break;
        default: goto bad;
        }
        if (r.bad) goto bad;
    }

    size_t data_size = size - data_off;
    for (uint32_t i = 0; i < n_t; i++) {
        qm_tensor *t = &f->tensors[i];
        if (rd_name(&r, t->name, sizeof t->name)) goto bad;
        uint32_t dt = rd_u32(&r), nd = rd_u32(&r);
        if (r.bad || !qm_dtype_valid(dt) || nd < 1 || nd > 4) goto bad;
        t->type = (qm_dtype)dt;
        t->ndim = (int)nd;
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t ne = rd_u64(&r);
            if (ne == 0 || ne > ((uint64_t)1 << 40) || nelem > ((uint64_t)1 << 40) / ne) goto bad;
            nelem *= ne;
            t->ne[d] = (int64_t)ne;
        }
        uint64_t off = rd_u64(&r);
        if (r.bad || off % QM_ALIGN) goto bad;
        size_t bytes = qm_tensor_bytes(t);
        if (off > data_size || bytes > data_size - off) {
            fail(err, errlen, "tensor '%s' out of range", t->name); /* format before freeing t */
            qlm_close(f);
            return -1;
        }
        t->data = buf + data_off + off;
    }
    if (r.pos > data_off) goto bad; /* header overlaps data */
    return 0;
bad:
    qlm_close(f);
    return fail(err, errlen, "malformed header near byte %zu", r.pos);
}

int qlm_open(qlm_file *f, const char *path, char *err, size_t errlen) {
    memset(f, 0, sizeof *f);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(err, errlen, "cannot open %s", path);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return fail(err, errlen, "cannot stat %s", path);
    }
    size_t size = (size_t)st.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail(err, errlen, "mmap failed");
    if (qlm_parse(f, map, size, err, errlen) != 0) {
        munmap(map, size);
        return -1;
    }
    f->map = map;
    f->map_size = size;
    return 0;
}

void qlm_close(qlm_file *f) {
    if (f->map) munmap(f->map, f->map_size);
    free(f->kv);
    free(f->tensors);
    memset(f, 0, sizeof *f);
}

const qlm_kv *qlm_find_kv(const qlm_file *f, const char *key) {
    for (uint32_t i = 0; i < f->n_kv; i++)
        if (strcmp(f->kv[i].key, key) == 0) return &f->kv[i];
    return NULL;
}

const qm_tensor *qlm_find_tensor(const qlm_file *f, const char *name) {
    for (uint32_t i = 0; i < f->n_tensors; i++)
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    return NULL;
}

uint32_t qlm_arr_get(const qlm_kv *kv, uint32_t i) {
    const uint8_t *b = kv->arr + (size_t)i * 4;
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}
