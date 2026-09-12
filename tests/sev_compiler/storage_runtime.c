#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "owned.h"

extern void *__sev_list_create(void);
extern void __sev_list_push_ptr(void *, const char *);
extern void __sev_list_clear(void *);
extern void *__sev_list_copy_ptr(void *);
extern void __sev_list_set_ptr(void *, int64_t, const char *);
extern void __sev_list_extend(void *, void *);
extern const char *__sev_list_pop_ptr(void *);
extern void *__sev_aggregate_box_owned(const void *, int64_t, sev_storage_destructor, sev_storage_destructor);
extern const char *__sev_string_from_int(int64_t);
extern void *__sev_string_characters(const char *);


static unsigned destroyed;
static void counted(void *value) { (void)value; ++destroyed; }
static void retain_field(void *box) { __sev_storage_retain(*(void **)box); }
static void drop_field(void *box) { __sev_storage_release(*(void **)box); }

int main(void) {
    __sev_storage_release("literal");
    __sev_storage_retain(NULL);
    for (int iteration = 0; iteration < 1000; ++iteration) {
        void *child = __sev_storage_new(17, counted);
        void *box = __sev_aggregate_box_owned(&child, sizeof(child), retain_field, drop_field);
        __sev_storage_release(child);
        void *list = __sev_list_create();
        __sev_list_push_ptr(list, box);
        __sev_storage_release(box);
        void *copy = __sev_list_copy_ptr(list);
        __sev_list_clear(list);
        assert(destroyed == (unsigned)iteration);
        __sev_storage_release(list);
        __sev_storage_release(copy);
        assert(destroyed == (unsigned)iteration + 1);
        const char *text = __sev_string_from_int(iteration);
        void *characters = __sev_string_characters(text);
        __sev_storage_release(text);
        __sev_storage_release(characters);
        assert(__sev_storage_live_bytes() == 0);
    }
    void *list = __sev_list_create();
    char *text = __sev_storage_new(4, NULL);
    memcpy(text, "abc", 4);
    __sev_list_push_ptr(list, text);
    __sev_storage_release(text);
    __sev_list_set_ptr(list, 0, text); /* alias replacement retains first */
    const char *taken = __sev_list_pop_ptr(list);
    __sev_storage_release(list);
    assert(strcmp(taken, "abc") == 0);
    __sev_storage_release(taken);
    assert(__sev_storage_live_bytes() == 0);
}
