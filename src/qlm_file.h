/* .qlm model file format (little-endian):
 *
 *   magic "QLMF" | u32 version (=1) | u32 n_kv | u32 n_tensors | u64 data_offset
 *   n_kv x      { u32 key_len, key bytes, u32 type, value }
 *                 type 0 = u32, 1 = f32, 2 = string (u32 len + bytes), 3 = u32 array (u32 count + u32[])
 *   n_tensors x { u32 name_len, name bytes, u32 dtype, u32 ndim, u64 ne[ndim], u64 offset }
 *   ... zero padding up to data_offset (multiple of 64)
 *   tensor data, each tensor at data_offset + offset (offset multiple of 64)
 */
#ifndef QM_QLM_FILE_H
#define QM_QLM_FILE_H

#include "ops.h"

#define QLM_MAGIC "QLMF"
#define QLM_VERSION 1
#define QLM_MAX_KV 256
#define QLM_MAX_TENSORS 4096

enum { QLM_KV_U32 = 0, QLM_KV_F32 = 1, QLM_KV_STR = 2, QLM_KV_ARR_U32 = 3 };

typedef struct {
    char key[64];
    uint32_t type;
    uint32_t u32;
    float f32;
    const char *str; /* not NUL-terminated; points into the mapping */
    uint32_t len;    /* string length or array count */
    const uint8_t *arr; /* u32 array (unaligned, little-endian) */
} qlm_kv;

typedef struct {
    void *map;
    size_t map_size;
    uint32_t n_kv, n_tensors;
    qlm_kv *kv;
    qm_tensor *tensors;
} qlm_file;

/* Map and validate a file. Returns 0 on success, otherwise fills err. */
int qlm_open(qlm_file *f, const char *path, char *err, size_t errlen);
/* Same, from an in-memory buffer (not owned; used by fuzz tests) */
int qlm_parse(qlm_file *f, const uint8_t *buf, size_t size, char *err, size_t errlen);
void qlm_close(qlm_file *f);

const qlm_kv *qlm_find_kv(const qlm_file *f, const char *key);
const qm_tensor *qlm_find_tensor(const qlm_file *f, const char *name);
uint32_t qlm_arr_get(const qlm_kv *kv, uint32_t i);

#endif
