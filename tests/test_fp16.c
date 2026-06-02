/* fp16 conversion: exhaustive over all 65536 halves + hardware (F16C) cross-check. */
#include <stdint.h>

#include "qm_base.h"
#include "test.h"

#if defined(__x86_64__)
#include <immintrin.h>
__attribute__((target("f16c"))) static float hw_to32(uint16_t h) { return _cvtsh_ss(h); }
__attribute__((target("f16c"))) static uint16_t hw_to16(float f) {
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
#endif

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static float bitsf(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static int h_is_nan(uint16_t h) { return (h & 0x7c00) == 0x7c00 && (h & 0x3ff); }

int main(void) {
    /* known values */
    CHECK(qm_fp32_to_fp16(1.0f) == 0x3c00, "1.0");
    CHECK(qm_fp32_to_fp16(-2.0f) == 0xc000, "-2.0");
    CHECK(qm_fp32_to_fp16(65504.0f) == 0x7bff, "max");
    CHECK(qm_fp32_to_fp16(65519.0f) == 0x7bff, "65519 rounds down");
    CHECK(qm_fp32_to_fp16(65520.0f) == 0x7c00, "65520 -> inf");
    CHECK(qm_fp32_to_fp16(5.9604644775390625e-08f) == 0x0001, "min subnormal");
    CHECK(qm_fp32_to_fp16(2.98023223876953125e-08f) == 0x0000, "2^-25 ties to even (0)");
    CHECK(qm_fp32_to_fp16(-0.0f) == 0x8000, "-0");
    CHECK(qm_fp16_to_fp32(0x3555) == 0.333251953125f, "1/3 approx");
    CHECK(h_is_nan(qm_fp32_to_fp16(NAN)), "nan");

    /* exhaustive: every half survives half->float->half; matches hardware decode */
    int hw = qm_cpu_supports(QM_ISA_AVX2);
    for (uint32_t h = 0; h < 65536; h++) {
        float f = qm_fp16_to_fp32((uint16_t)h);
        if (h_is_nan((uint16_t)h)) {
            CHECK(isnan(f), "h=%04x", h);
            continue;
        }
        CHECK(qm_fp32_to_fp16(f) == h, "roundtrip h=%04x", h);
#if defined(__x86_64__)
        if (hw) CHECK(fbits(f) == fbits(hw_to32((uint16_t)h)), "decode vs F16C h=%04x", h);
#endif
    }
    /* float->half vs hardware over a strided sweep of all 2^32 bit patterns + random */
#if defined(__x86_64__)
    if (hw) {

        long mism = 0, n = 0;
        uint64_t step = quick() ? 65537 * 31 : 4099;
        for (uint64_t u = 0; u < (1ULL << 32); u += step, n++) {
            float f = bitsf((uint32_t)u);
            uint16_t a = qm_fp32_to_fp16(f), b = hw_to16(f);
            if (isnan(f) ? !(h_is_nan(a) && h_is_nan(b)) : a != b) mism++;
        }
        for (int i = 0; i < (quick() ? 10000 : 2000000); i++, n++) {
            /* concentrate around the half range */
            float f = (rnd_unit() - 0.5f) * ldexpf(1.0f, rnd_int(-30, 17));
            if (qm_fp32_to_fp16(f) != hw_to16(f)) mism++;
        }
        CHECK(mism == 0, "%ld of %ld float->half conversions differ from F16C", mism, n);
        printf("float->half: %ld conversions compared against F16C, %ld mismatches\n", n, mism);
    } else {
        printf("F16C not available: hardware cross-check skipped\n");
    }

#endif
    TEST_DONE();
}
