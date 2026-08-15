#include "random_abi.h"

#include <stdint.h>

static uint64_t sev_random_state = UINT64_C(0x9e3779b97f4a7c15);

static uint64_t sev_random_next(void) {
    uint64_t value = sev_random_state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    sev_random_state = value;
    return value * UINT64_C(2685821657736338717);
}

void sev_abi_v1_random_seed(int64_t value) {
    sev_random_state = value ? (uint64_t)value : UINT64_C(0x9e3779b97f4a7c15);
}

double sev_abi_v1_random_float(void) {
    return (double)(sev_random_next() >> 11) * (1.0 / 9007199254740992.0);
}

int64_t sev_abi_v1_random_int(int64_t start, int64_t stop) {
    uint64_t width = (uint64_t)(stop - start) + 1;
    return start + (int64_t)(sev_random_next() % width);
}
