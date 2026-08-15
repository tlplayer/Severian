#ifndef SEVERIAN_MATH_ABI_H
#define SEVERIAN_MATH_ABI_H

#include <stdbool.h>
#include <stdint.h>

double sev_abi_v1_math_pow(double value, double exponent);
double sev_abi_v1_math_exp(double value);
double sev_abi_v1_math_log(double value);
double sev_abi_v1_math_log2(double value);
double sev_abi_v1_math_log10(double value);
double sev_abi_v1_math_sin(double value);
double sev_abi_v1_math_cos(double value);
double sev_abi_v1_math_tan(double value);
int64_t sev_abi_v1_math_floor(double value);
int64_t sev_abi_v1_math_ceil(double value);
bool sev_abi_v1_math_isfinite(double value);
bool sev_abi_v1_math_isnan(double value);
double sev_abi_v1_math_round(double value, int64_t digits);

#endif
