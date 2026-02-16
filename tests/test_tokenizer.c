/* BPE tokenizer: exact agreement with the Python encoder, round-trip property, error paths. */
#include "test.h"
#include "tokenizer.h"

static int hexval(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }

int main(void) {
    char p[512];
    size_t msz;
    snprintf(p, sizeof p, "%s/merges.u32", fixtures_dir());
    uint32_t *merges = read_file(p, &msz);
    CHECK(merges != NULL, "missing merges fixture");
    if (!merges) TEST_DONE();
    qm_tokenizer t;
    CHECK(qm_tokenizer_init(&t, merges, (int32_t)(msz / 8)) == 0, "init");

    /* vectors produced by tools/qlm.py */
    snprintf(p, sizeof p, "%s/tok.txt", fixtures_dir());
    FILE *f = fopen(p, "r");
    CHECK(f != NULL, "tok.txt");
    int ncases = 0;
    if (f && fscanf(f, "%d", &ncases) == 1) {
        for (int c = 0; c < ncases; c++) {
            char hex[1024];
            int nids;
            if (fscanf(f, "%1023s %d", hex, &nids) != 2) break;
            char text[512];
            size_t len = 0;
            if (strcmp(hex, "-") != 0)
                for (size_t i = 0; hex[i] && hex[i + 1]; i += 2) text[len++] = (char)(hexval(hex[i]) * 16 + hexval(hex[i + 1]));
            int32_t ref[600], got[600];
            for (int i = 0; i < nids; i++)
                if (fscanf(f, "%d", &ref[i]) != 1) break;
            int n = qm_tokenizer_encode(&t, text, len, 0, got, 600);
            CHECK(n == nids && memcmp(ref, got, sizeof(int32_t) * (size_t)n) == 0, "case %d: python %d tokens, C %d", c, nids, n);
        }
        fclose(f);
    }
    CHECK(ncases > 40, "too few tokenizer cases (%d)", ncases);

    /* property: decode(encode(s)) == s for random byte strings; merges shrink length */
    for (int it = 0; it < (quick() ? 50 : 3000); it++) {
        char s[300];
        int len = rnd_int(0, 299);
        const char *alpha = "the fox ran to river. ";

        for (int i = 0; i < len; i++) s[i] = rnd_int(0, 3) ? alpha[rnd_int(0, 21)] : (char)rnd_int(0, 255);
        int32_t ids[400];
        char back[400];
        int n = qm_tokenizer_encode(&t, s, (size_t)len, 1, ids, 400);
        CHECK(n >= 1 && n <= len + 1 && ids[0] == QM_TOK_BOS, "encode len=%d n=%d", len, n);
        int m = qm_tokenizer_decode(&t, ids, n, back, sizeof back);
        CHECK(m == len && memcmp(back, s, (size_t)len) == 0, "roundtrip len=%d", len);
    }
    /* common text actually compresses */
    int32_t ids[64];

    int n = qm_tokenizer_encode(&t, "the fox ran to the river", 24, 0, ids, 64);
    CHECK(n > 0 && n < 12, "expected merges to apply, got %d tokens", n);
    /* overflow & error paths */
    CHECK(qm_tokenizer_encode(&t, "abcdef", 6, 1, ids, 3) == -1, "small output buffer");
    char tiny[2];
    CHECK(qm_tokenizer_decode(&t, ids, n, tiny, sizeof tiny) == -1, "small decode buffer");
    uint32_t bad[4] = {2, 3, 258, 300}; /* second merge references a future id */
    qm_tokenizer tb;
    CHECK(qm_tokenizer_init(&tb, bad, 2) == -1, "forward reference rejected");
    uint32_t len;
    qm_tokenizer_piece(&t, QM_TOK_EOS, &len);
    CHECK(len == 0, "eos piece empty");
    qm_tokenizer_piece(&t, 1 << 30, &len);
    CHECK(len == 0, "unknown id empty");
    qm_tokenizer_free(&t);
    free(merges);
    TEST_DONE();
}
