#include <stdint.h>
#include <stddef.h>
typedef struct { uint8_t *data; uintptr_t length; } bytes_view;
int64_t xxi_test_sum(bytes_view data) {
    int64_t total = 0;
    for (uintptr_t index = 0; index < data.length; ++index) total += data.data[index];
    return total;
}
int64_t xxi_test_fill(bytes_view *data, uint8_t value) {
    for (uintptr_t index = 0; index < data->length; ++index) data->data[index] = value;
    return (int64_t)data->length;
}
