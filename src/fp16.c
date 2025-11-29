/* IEEE-754 binary16 <-> binary32 conversion, written from the bit layouts. */
#include <string.h>

#include "qm_base.h"

float qm_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { /* subnormal: renormalise */
            int e = -1;
            do {
                e++;
                mant <<= 1;
            } while (!(mant & 0x400u));
            mant &= 0x3ffu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13); /* inf / nan (payload kept) */
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t qm_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    uint32_t ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) /* NaN: keep top payload bits, force quiet */
        return (uint16_t)(sign | 0x7e00u | ((ax >> 13) & 0x3ffu));
    if (ax >= 0x477ff000u) /* >= 65520 rounds to infinity (also maps inf) */
        return (uint16_t)(sign | 0x7c00u);
    if (ax >= 0x38800000u) { /* normal half range */
        uint32_t m = ax + 0xfffu + ((ax >> 13) & 1u); /* round to nearest even at bit 13 */
        return (uint16_t)(sign | ((m - 0x38000000u) >> 13));
    }
    /* subnormal half or zero */
    uint32_t e = ax >> 23;
    if (e < 102) return sign; /* < 2^-25 -> 0 */
    uint32_t mant = (ax & 0x7fffffu) | 0x800000u;
    uint32_t shift = 126 - e; /* 14..24 */
    uint32_t q = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (q & 1u))) q++;
    return (uint16_t)(sign | q);
}
