#ifndef SEVERIAN_STORAGE_STATISTICS_H
#define SEVERIAN_STORAGE_STATISTICS_H
#include <stdint.h>
uint64_t __sev_storage_measured_allocations(void);
uint64_t __sev_storage_measured_bytes(void);
#endif
