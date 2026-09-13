#ifndef SEV_XXI_BYTES_H
#define SEV_XXI_BYTES_H
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../../../../core/storage/native/storage.h"

typedef struct { uint8_t *data; uintptr_t length; } sev_xxi_bytes;
typedef struct { void *storage; } sev_xxi_list;
typedef struct {
    sev_xxi_bytes view;
    void *owner;
    uint8_t *scratch;
    uintptr_t capacity;
    int write_back;
} sev_xxi_bytes_loan;
extern uintptr_t __sev_list_len(void *);
extern uint8_t __sev_list_index_u8(void *, int64_t);
extern void __sev_list_set_u8(void *, int64_t, uint8_t);

static inline void sev_xxi_contract_failure(const char *message) {
    fprintf(stderr, "XXI contract violation: %s\n", message);
    abort();
}

/* Keep the source owner alive for the entire loan. Scratch contents have their
 * own core.storage owner; no foreign allocator or hidden ownership registry. */
static inline sev_xxi_bytes_loan sev_xxi_bytes_acquire(sev_xxi_list value, int write_back) {
    uintptr_t length = __sev_list_len(value.storage);
    if (length > INT64_MAX) sev_xxi_contract_failure("sequence length cannot fit the source index type");
    __sev_storage_retain(value.storage);
    uint8_t *scratch = __sev_storage_new(length ? length : 1, NULL);
    for (uintptr_t index = 0; index < length; ++index)
        scratch[index] = __sev_list_index_u8(value.storage, (int64_t)index);
    return (sev_xxi_bytes_loan){{scratch, length}, value.storage, scratch, length, write_back};
}

static inline void sev_xxi_bytes_release(sev_xxi_bytes_loan *loan) {
    if (loan->view.data != loan->scratch || loan->view.length != loan->capacity)
        sev_xxi_contract_failure("foreign code changed a borrowed buffer's address or initialized length");
    if (loan->write_back) {
        for (uintptr_t index = 0; index < loan->capacity; ++index)
            __sev_list_set_u8(loan->owner, (int64_t)index, loan->scratch[index]);
    }
    __sev_storage_release(loan->scratch);
    __sev_storage_release(loan->owner);
    loan->owner = NULL;
    loan->scratch = NULL;
}
#endif
