/* Tabular JSON provider. The document owns every decoded string and row.
 * Parsing visits each record once; column discovery shares that traversal.
 * Nested values remain JSON text, matching the Data table cell contract. */
#include "../../../core/memory/native/memory.h"
#include "../../../core/storage/native/storage.h"
#include "json.h"
#include <errno.h>
#include <stdint.h>
#include <string.h>

typedef struct { int64_t tag, payload; } json_any;
typedef struct { void *storage; } json_list;
extern void *__sev_list_create(void);
extern uintptr_t __sev_list_len(void *);
extern const char *__sev_list_get_ptr(void *, int64_t);
extern void __sev_list_push_ptr(void *, const char *);
extern void __sev_list_push_any(void *, json_any);
extern void __sev_list_set_any(void *, int64_t, json_any);
extern void __sev_list_push_list(void *, json_list);

typedef struct {
    char **columns;
    size_t count, capacity;
    void *rows;
} json_document;
static _Thread_local int32_t json_error;
int32_t __sev_json_error(void) { return json_error; }
static const char *space(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
    return p;
}
static int hex(char c) {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'a' && c <= 'f') return c-'a'+10;
    if (c >= 'A' && c <= 'F') return c-'A'+10;
    return -1;
}
static uint32_t escape(const char **p) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        int digit = hex(**p);
        if (digit < 0) { json_error = EILSEQ; return 0; }
        value = value*16+(unsigned)digit;
        ++*p;
    }
    return value;
}
/* With output=NULL this validates/skips a string without allocating it. */
static const char *string_end(const char *p, char *output) {
    if (*p++ != '"') { json_error = EINVAL; return p-1; }
    size_t used = 0;
    while (*p && *p != '"' && !json_error) {
        uint32_t c = (unsigned char)*p++;
        if (c < 32) { json_error = EILSEQ; break; }
        if (c != '\\') {
            const char *start = p-1;
            unsigned width = c < 128 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 :
                c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
            if (!width) { json_error=EILSEQ; break; }
            uint32_t scalar = c & ((1u << (7-width))-1);
            for (unsigned i=1;i<width;++i) {
                unsigned char continuation = (unsigned char)*p;
                if (continuation < 0x80 || continuation > 0xbf) { json_error=EILSEQ; break; }
                scalar = (scalar<<6) | (continuation & 63);
                ++p;
            }
            if (json_error || (width == 3 && scalar < 0x800) ||
                (width == 4 && scalar < 0x10000) || scalar > 0x10ffff ||
                (scalar >= 0xd800 && scalar <= 0xdfff)) { json_error=EILSEQ; break; }
            if (output) memcpy(output+used,start,width);
            used += width;
            continue;
        }
        c = (unsigned char)*p;
        if (!c) { json_error = EINVAL; break; }
        ++p;
        switch (c) {
            case '"': case '\\': case '/': break;
            case 'b': c=8; break; case 'f': c=12; break;
            case 'n': c=10; break; case 'r': c=13; break; case 't': c=9; break;
            case 'u':
                c = escape(&p);
                if (!json_error && c >= 0xd800 && c <= 0xdbff) {
                    if (p[0] != '\\' || p[1] != 'u') { json_error=EILSEQ; break; }
                    p += 2;
                    uint32_t low = escape(&p);
                    if (low < 0xdc00 || low > 0xdfff) { json_error=EILSEQ; break; }
                    c = 0x10000 + ((c-0xd800)<<10) + low-0xdc00;
                } else if (c >= 0xdc00 && c <= 0xdfff) { json_error=EILSEQ; }
                /* The hosted string ABI cannot represent embedded NUL. */
                if (c == 0) json_error=EILSEQ;
                break;
            default: json_error=EILSEQ; break;
        }
        if (json_error) break;
        unsigned width = c < 128 ? 1 : c < 2048 ? 2 : c < 65536 ? 3 : 4;
        if (output) {
            if (width == 1) output[used] = (char)c;
            else {
                output[used] = (char)((width == 2 ? 0xc0 : width == 3 ? 0xe0 : 0xf0) | (c >> (6*(width-1))));
                for (unsigned i=1;i<width;++i) output[used+i]=(char)(0x80|((c>>(6*(width-i-1)))&63));
            }
        }
        used += width;
    }
    if (*p != '"') json_error = EINVAL;
    else ++p;
    if (output) output[used] = 0;
    return p;
}
static char *text(const char **p) {
    const char *end = string_end(*p, NULL);
    if (json_error) return NULL;
    char *result = __sev_storage_new((uint64_t)(end-*p), NULL);
    *p = string_end(*p, result);
    return result;
}
static const char *skip(const char *p, unsigned depth) {
    p = space(p);
    if (depth > 256) { json_error=EOVERFLOW; return p; }
    if (*p == '"') return string_end(p, NULL);
    if (*p == '{' || *p == '[') {
        char close = *p == '{' ? '}' : ']';
        p = space(p+1);
        if (*p == close) return p+1;
        for (;;) {
            if (close == '}') {
                if (*p != '"') { json_error=EINVAL; return p; }
                p = space(string_end(p, NULL));
                if (json_error || *p != ':') { json_error=EINVAL; return p; }
                ++p;
            }
            p = space(skip(p, depth+1));
            if (json_error) return p;
            if (*p == close) return p+1;
            if (*p != ',') { json_error=EINVAL; return p; }
            p = space(p+1);
        }
    }
    if (!strncmp(p,"true",4) || !strncmp(p,"null",4)) return p+4;
    if (!strncmp(p,"false",5)) return p+5;
    if (*p == '-') ++p;
    if (*p == '0') ++p;
    else if (*p >= '1' && *p <= '9') { do { ++p; } while (*p >= '0' && *p <= '9'); }
    else { json_error=EINVAL; return p; }
    if (*p == '.') {
        ++p;
        if (*p < '0' || *p > '9') { json_error=EINVAL; return p; }
        do { ++p; } while (*p >= '0' && *p <= '9');
    }
    if (*p == 'e' || *p == 'E') {
        ++p;
        if (*p == '+' || *p == '-') ++p;
        if (*p < '0' || *p > '9') { json_error=EINVAL; return p; }
        do { ++p; } while (*p >= '0' && *p <= '9');
    }
    return p;
}
static char *value(const char **p) {
    *p = space(*p);
    if (**p == '"') return text(p);
    const char *start = *p;
    *p = skip(start, 0);
    if (json_error) return NULL;
    size_t length = (size_t)(*p-start);
    if (length == 4 && !memcmp(start,"null",4)) length=0;
    char *result = __sev_storage_new(length+1, NULL);
    memcpy(result,start,length); result[length]=0;
    return result;
}
static size_t column(json_document *document, char *key) {
    for (size_t i=0;i<document->count;++i)
        if (!strcmp(document->columns[i],key)) return i;
    if (document->count == document->capacity) {
        size_t capacity = document->capacity ? document->capacity*2 : 16;
        if (capacity < document->capacity || capacity > SIZE_MAX/sizeof(char *)) abort();
        char **columns = sev_memory_resize(document->columns, capacity*sizeof(char *));
        if (!columns) abort();
        document->columns=columns; document->capacity=capacity;
    }
    __sev_storage_retain(key);
    document->columns[document->count]=key;
    return document->count++;
}
static void pad(void *row, size_t count) {
    while (__sev_list_len(row)<count) __sev_list_push_any(row,(json_any){0,(int64_t)(intptr_t)""});
}
static const char *object(const char *p, json_document *document) {
    if (*p != '{') { json_error=EINVAL; return p; }
    void *row=__sev_list_create();
    p=space(p+1);
    while (*p != '}' && !json_error) {
        char *key=text(&p);
        if (json_error) break;
        p=space(p);
        if (*p != ':') { __sev_storage_release(key); json_error=EINVAL; break; }
        ++p;
        char *contents=value(&p);
        if (!json_error) {
            size_t index=column(document,key);
            json_any item={0,(int64_t)(intptr_t)contents};
            pad(row,index);
            if (__sev_list_len(row)==index) __sev_list_push_any(row,item);
            else __sev_list_set_any(row,(int64_t)index,item);
        }
        __sev_storage_release(key); __sev_storage_release(contents);
        p=space(p);
        if (*p=='}' || json_error) break;
        if (*p!=',') { json_error=EINVAL; break; }
        p=space(p+1);
        if (*p=='}') { json_error=EINVAL; break; }
    }
    if (!json_error && *p=='}') { __sev_list_push_list(document->rows,(json_list){row}); ++p; }
    else json_error=EINVAL;
    __sev_storage_release(row);
    return p;
}
static void destroy(void *pointer) {
    json_document *document=pointer;
    for(size_t i=0;i<document->count;++i) __sev_storage_release(document->columns[i]);
    sev_memory_release(document->columns);
    __sev_storage_release(document->rows);
}
void *__sev_json_parse(const char *source) {
    json_error=0;
    json_document *document=__sev_storage_new(sizeof(*document),destroy);
    memset(document,0,sizeof(*document));
    document->rows=__sev_list_create();
    const char *p=space(source ? source : "");
    if (*p=='{') p=object(p,document);
    else if (*p=='[') {
        p=space(p+1);
        while (*p!=']' && !json_error) {
            p=space(object(p,document));
            if (json_error || *p==']') break;
            if (*p!=',') { json_error=EINVAL; break; }
            p=space(p+1);
            if (*p==']') { json_error=EINVAL; break; }
        }
        if (*p==']') ++p; else json_error=EINVAL;
    } else json_error=EINVAL;
    if (*space(p)) json_error=EINVAL;
    if (json_error) { __sev_storage_release(document); return NULL; }
    for(size_t i=0;i<__sev_list_len(document->rows);++i)
        pad((void *)__sev_list_get_ptr(document->rows,(int64_t)i),document->count);
    return document;
}
void *__sev_json_document_columns(void *pointer) {
    json_document *document=pointer;
    void *columns=__sev_list_create();
    if(document) for(size_t i=0;i<document->count;++i) __sev_list_push_ptr(columns,document->columns[i]);
    return columns;
}
void *__sev_json_document_rows(void *pointer) {
    json_document *document=pointer;
    if(!document) return __sev_list_create();
    __sev_storage_retain(document->rows);
    return document->rows;
}
void __sev_json_document_release(void *pointer) { __sev_storage_release(pointer); }
/* Pointer-sized scalar handles keep document ownership within this provider. */
uintptr_t __sev_json_open(const char *source) { return (uintptr_t)__sev_json_parse(source); }
void __sev_json_close(uintptr_t handle) { __sev_json_document_release((void *)handle); }
void *__sev_json_handle_columns(uintptr_t handle) { return __sev_json_document_columns((void *)handle); }
void *__sev_json_handle_rows(uintptr_t handle) { return __sev_json_document_rows((void *)handle); }
/* Compatibility entry points for bootstrap consumers built before document ABI. */
void *__sev_json_columns(const char *source) {
    void *document=__sev_json_parse(source);
    void *result=__sev_json_document_columns(document);
    __sev_json_document_release(document); return result;
}
void *__sev_json_rows(const char *source) {
    void *document=__sev_json_parse(source);
    void *result=__sev_json_document_rows(document);
    __sev_json_document_release(document); return result;
}
