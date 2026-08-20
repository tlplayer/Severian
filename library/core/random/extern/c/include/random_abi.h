#ifndef SEVERIAN_RANDOM_ABI_H
#define SEVERIAN_RANDOM_ABI_H

#include <stdint.h>

double sev_abi_v1_random_float(void);
int64_t sev_abi_v1_random_int(int64_t start, int64_t stop);
void sev_abi_v1_random_seed(int64_t value);

#endif
