/* Minimal persistent pthread pool with deterministic static partitioning. */
#include "threadpool.h"

#include <pthread.h>
#include <stdlib.h>

struct qm_pool {
    int n;
    pthread_t *threads;
    pthread_mutex_t mu;
    pthread_cond_t cv_start, cv_done;
    uint64_t generation;
    int pending, quit;
    /* current job */
    qm_task_fn fn;
    void *ctx;
    int64_t total, grain;
};

typedef struct {
    qm_pool *p;
    int tid;
} worker_arg;

static void run_chunk(qm_pool *p, int tid) {
    int64_t units = (p->total + p->grain - 1) / p->grain;
    int64_t per = units / p->n, extra = units % p->n;
    int64_t u0 = tid * per + (tid < extra ? tid : extra);
    int64_t u1 = u0 + per + (tid < extra ? 1 : 0);
    int64_t s = u0 * p->grain, e = u1 * p->grain;
    if (e > p->total) e = p->total;
    if (s < e) p->fn(p->ctx, s, e, tid);
}

static void *worker(void *arg) {
    worker_arg *wa = arg;
    qm_pool *p = wa->p;
    int tid = wa->tid;
    free(wa);
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->generation == seen && !p->quit) pthread_cond_wait(&p->cv_start, &p->mu);
        if (p->quit) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        seen = p->generation;
        pthread_mutex_unlock(&p->mu);

        run_chunk(p, tid);

        pthread_mutex_lock(&p->mu);
        if (--p->pending == 0) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

qm_pool *qm_pool_create(int n_threads) {
    if (n_threads < 1) n_threads = 1;
    qm_pool *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n = n_threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_start, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    p->threads = calloc((size_t)n_threads, sizeof(pthread_t));
    for (int i = 1; i < n_threads; i++) {
        worker_arg *wa = malloc(sizeof *wa);
        wa->p = p;
        wa->tid = i;
        pthread_create(&p->threads[i], NULL, worker, wa);
    }
    return p;
}

void qm_pool_destroy(qm_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->quit = 1;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_start);
    pthread_cond_destroy(&p->cv_done);
    free(p->threads);
    free(p);
}

int qm_pool_size(const qm_pool *p) { return p ? p->n : 1; }

void qm_pool_run(qm_pool *p, int64_t n, int64_t grain, qm_task_fn fn, void *ctx) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    if (!p || p->n == 1 || n <= grain) {
        fn(ctx, 0, n, 0);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->total = n;
    p->grain = grain;
    p->pending = p->n - 1;
    p->generation++;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);

    run_chunk(p, 0);

    pthread_mutex_lock(&p->mu);
    while (p->pending > 0) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}
