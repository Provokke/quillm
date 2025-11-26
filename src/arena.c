#include <stdlib.h>
#include <string.h>

#include "qm_base.h"

int qm_arena_init(qm_arena *a, size_t cap) {
    cap = (cap + QM_ALIGN - 1) & ~(size_t)(QM_ALIGN - 1);
    if (cap == 0) cap = QM_ALIGN;
    a->base = aligned_alloc(QM_ALIGN, cap);
    if (!a->base) return -1;
    memset(a->base, 0, cap);
    a->cap = cap;
    a->used = 0;
    a->peak = 0;
    return 0;
}

void *qm_arena_alloc(qm_arena *a, size_t n) {
    size_t sz = (n + QM_ALIGN - 1) & ~(size_t)(QM_ALIGN - 1);
    if (sz < n || sz > a->cap - a->used) return NULL;
    void *p = a->base + a->used;
    a->used += sz;
    if (a->used > a->peak) a->peak = a->used;
    return p;
}

void qm_arena_reset(qm_arena *a) {
    memset(a->base, 0, a->used);
    a->used = 0;
}

void qm_arena_free(qm_arena *a) {
    free(a->base);
    a->base = NULL;

    a->cap = a->used = 0;
}
