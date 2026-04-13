#ifndef QM_THREADPOOL_H
#define QM_THREADPOOL_H

#include <stdint.h>

typedef struct qm_pool qm_pool;

/* fn processes [start, end) of the range; tid in [0, n_threads) */
typedef void (*qm_task_fn)(void *ctx, int64_t start, int64_t end, int tid);

qm_pool *qm_pool_create(int n_threads); /* n_threads >= 1; the caller thread is worker 0 */
void qm_pool_destroy(qm_pool *p);
int qm_pool_size(const qm_pool *p);
/* Static partition of [0, n) into contiguous chunks whose starts are multiples of `grain`.
 * Blocks until all chunks are done. pool may be NULL (runs inline). */
void qm_pool_run(qm_pool *p, int64_t n, int64_t grain, qm_task_fn fn, void *ctx);

#endif
