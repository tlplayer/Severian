#include "math_abi.h"

#include <math.h>

double sev_abi_v1_math_pow(double value, double exponent) {
    return pow(value, exponent);
}

double sev_abi_v1_math_exp(double value) {
    return exp(value);
}

double sev_abi_v1_math_log(double value) {
    return log(value);
}

double sev_abi_v1_math_log2(double value) {
    return log2(value);
}

double sev_abi_v1_math_log10(double value) {
    return log10(value);
}

double sev_abi_v1_math_sin(double value) {
    return sin(value);
}

double sev_abi_v1_math_cos(double value) {
    return cos(value);
}

double sev_abi_v1_math_tan(double value) {
    return tan(value);
}

int64_t sev_abi_v1_math_floor(double value) {
    return (int64_t)floor(value);
}

int64_t sev_abi_v1_math_ceil(double value) {
    return (int64_t)ceil(value);
}

bool sev_abi_v1_math_isfinite(double value) {
    return isfinite(value);
}

bool sev_abi_v1_math_isnan(double value) {
    return isnan(value);
}

double sev_abi_v1_math_round(double value, int64_t digits) {
    double factor = pow(10.0, (double)digits);
    return round(value * factor) / factor;
}
