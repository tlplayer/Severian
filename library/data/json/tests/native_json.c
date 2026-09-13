#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "../../../core/storage/native/storage.h"
#include "../native/json.h"
extern uintptr_t __sev_list_len(void *);
extern const char *__sev_list_index_ptr(void *, int64_t);
typedef struct { void *storage; } list_value;
typedef struct { int64_t tag, payload; } any_value;
extern list_value __sev_list_index_list(void *, int64_t);
extern any_value __sev_list_index_any(void *, int64_t);
static const char *source = "[{\"name\":\"Ada\",\"age\":36},{\"name\":\"Grace\",\"active\":true}]";
int main(void) {
    uint64_t live = __sev_storage_live_bytes();
    uint64_t calls = __sev_storage_thread_allocations();
    uint64_t bytes = __sev_storage_thread_allocated_bytes();
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        void *document = __sev_json_parse(source);
        assert(document && __sev_json_error() == 0);
        void *columns = __sev_json_document_columns(document);
        void *rows = __sev_json_document_rows(document);
        __sev_json_document_release(document);
        assert(__sev_list_len(columns) == 3 && __sev_list_len(rows) == 2);
        assert(!strcmp(__sev_list_index_ptr(columns, 0), "name"));
        list_value row = __sev_list_index_list(rows, 0);
        any_value name = __sev_list_index_any(row.storage, 0);
        assert(!strcmp((const char *)(intptr_t)name.payload, "Ada"));
        __sev_storage_release(columns);
        __sev_storage_release(rows);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("{\"iterations\":1000,\"nanoseconds\":%lld,\"allocations\":%llu,\"allocated_bytes\":%llu,\"retained_bytes\":%llu}\n",
        (long long)(end.tv_sec-start.tv_sec)*1000000000+end.tv_nsec-start.tv_nsec,
        (unsigned long long)(__sev_storage_thread_allocations()-calls),
        (unsigned long long)(__sev_storage_thread_allocated_bytes()-bytes),
        (unsigned long long)(__sev_storage_live_bytes()-live));
    assert(__sev_storage_live_bytes() == live);
    void *document = __sev_json_parse("[{\"text\":\"\\u03bb\\ud83d\\ude00\",\"nested\":{\"x\":[1,true,null]}}]");
    assert(document && __sev_json_error() == 0);
    void *rows = __sev_json_document_rows(document);
    list_value row = __sev_list_index_list(rows, 0);
    any_value text = __sev_list_index_any(row.storage, 0);
    assert(!strcmp((const char *)(intptr_t)text.payload,"λ😀"));
    __sev_storage_release(rows);
    __sev_json_document_release(document);
    const char *invalid[] = {"", "[", "[{\"x\":}]", "[{\"x\":1,}]", "[{\"x\":01}]", "[{\"x\":1e}]", "[{\"x\":\"\\uD800\"}]", "[{\"x\":\"\xc0\x80\"}]", "[{\"x\":\"\xed\xa0\x80\"}]", "[{}] trailing", "[{},]"};
    for (size_t i=0;i<sizeof(invalid)/sizeof(*invalid);++i) {
        assert(__sev_json_parse(invalid[i]) == NULL);
        assert(__sev_json_error() != 0);
        assert(__sev_storage_live_bytes() == live);
    }
}
