/* quillm CLI:  quillm run model.qlm -p "prompt" [options]  |  quillm info model.qlm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "sampler.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(void) {
    fprintf(stderr,
            "usage:\n"
            "  quillm run  MODEL.qlm -p PROMPT [-n N] [--temp T] [--top-k K] [--top-p P] [--seed S]\n"
            "                        [-t THREADS] [--isa scalar|avx2|avx512] [--no-stats]\n"
            "  quillm info MODEL.qlm\n");
}

static int cmd_info(const char *path) {
    qlm_file f;
    char err[256];
    if (qlm_open(&f, path, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    printf("file: %s (%zu bytes)\nmetadata (%u):\n", path, f.map_size, f.n_kv);
    for (uint32_t i = 0; i < f.n_kv; i++) {
        const qlm_kv *kv = &f.kv[i];
        switch (kv->type) {
        case QLM_KV_U32: printf("  %-24s u32 %u\n", kv->key, kv->u32); break;
        case QLM_KV_F32: printf("  %-24s f32 %g\n", kv->key, (double)kv->f32); break;
        case QLM_KV_STR: printf("  %-24s str \"%.*s\"\n", kv->key, (int)kv->len, kv->str); break;
        case QLM_KV_ARR_U32: printf("  %-24s u32[%u]\n", kv->key, kv->len); break;
        }
    }
    printf("tensors (%u):\n", f.n_tensors);
    size_t total = 0;
    for (uint32_t i = 0; i < f.n_tensors; i++) {
        const qm_tensor *t = &f.tensors[i];
        total += qm_tensor_bytes(t);
        if (i < 12 || i + 3 >= f.n_tensors) {
            printf("  %-22s %-5s [", t->name, qm_dtype_name(t->type));
            for (int d = 0; d < t->ndim; d++) printf("%s%lld", d ? ", " : "", (long long)t->ne[d]);
            printf("]\n");
        } else if (i == 12) {
            printf("  ...\n");
        }
    }
    printf("tensor bytes: %zu\n", total);
    qlm_close(&f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    if (strcmp(argv[1], "info") == 0) return cmd_info(argv[2]);
    if (strcmp(argv[1], "run") != 0) {
        usage();
        return 2;
    }
    const char *path = argv[2], *prompt = "", *isa = NULL;
    int n_gen = 64, top_k = 0, threads = 4, stats = 1;
    float temp = 0.8f, top_p = 0.95f;
    uint64_t seed = 42;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if (!strcmp(a, "-p") && has) prompt = argv[++i];
        else if (!strcmp(a, "-n") && has) n_gen = atoi(argv[++i]);
        else if (!strcmp(a, "--temp") && has) temp = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--top-k") && has) top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--top-p") && has) top_p = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--seed") && has) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "-t") && has) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--isa") && has) isa = argv[++i];
        else if (!strcmp(a, "--no-stats")) stats = 0;
        else {
            usage();
            return 2;
        }
    }
    if (isa) {
        int lvl = qm_isa_from_name(isa);
        if (lvl < 0 || qm_kernels_select((qm_isa)lvl) != 0) {
            fprintf(stderr, "error: ISA '%s' not available on this CPU/build\n", isa);
            return 1;
        }
    }
    qm_model m;
    char err[256];
    if (qm_model_load(&m, path, threads, 0, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    qm_arena sarena;
    qm_sampler s;
    if (qm_arena_init(&sarena, (size_t)m.cfg.vocab_size * 8 + 256) != 0 ||
        qm_sampler_init(&s, &sarena, m.cfg.vocab_size, temp, top_k, top_p, seed) != 0) {
        fprintf(stderr, "error: sampler alloc\n");
        return 1;
    }
    size_t plen = strlen(prompt);
    int cap = (int)plen + 2;
    int32_t *toks = malloc(sizeof(int32_t) * (size_t)cap);
    int n_prompt = qm_tokenizer_encode(&m.tok, prompt, plen, 1, toks, cap);
    if (n_prompt + n_gen > m.cfg.max_seq) n_gen = m.cfg.max_seq - n_prompt;
    if (n_prompt < 1 || n_gen < 0) {
        fprintf(stderr, "error: prompt too long for context %d\n", m.cfg.max_seq);
        return 1;
    }
    for (int i = 0; i < n_prompt; i++)
        if (toks[i] >= m.cfg.vocab_size) {
            fprintf(stderr, "error: tokenizer produced id %d >= vocab %d\n", toks[i], m.cfg.vocab_size);
            return 1;
        }
    fwrite(prompt, 1, plen, stdout);
    fflush(stdout);

    double t0 = now_s(), t_prompt = 0;
    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) logits = qm_model_forward(&m, toks[i], i);
    t_prompt = now_s() - t0;
    int pos = n_prompt, generated = 0;
    double t1 = now_s();
    for (; generated < n_gen; generated++) {
        int32_t next = qm_sample(&s, logits);
        if (next == QM_TOK_EOS) break;
        uint32_t len;
        const uint8_t *piece = qm_tokenizer_piece(&m.tok, next, &len);
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
        if (generated + 1 == n_gen) {
            generated++;
            break;
        }
        logits = qm_model_forward(&m, next, pos++);
    }
    double t_gen = now_s() - t1;
    printf("\n");
    if (stats)
        fprintf(stderr, "[quillm] isa=%s threads=%d prompt=%d tok (%.1f tok/s) generated=%d tok (%.1f tok/s)\n",
                qm_isa_name(qm_kernels_active()->isa), m.n_threads, n_prompt, n_prompt / (t_prompt > 0 ? t_prompt : 1e-9),
                generated, generated / (t_gen > 0 ? t_gen : 1e-9));
    free(toks);
    qm_arena_free(&sarena);
    qm_model_free(&m);
    return 0;
}
