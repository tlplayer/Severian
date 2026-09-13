#ifndef SEVERIAN_JSON_H
#define SEVERIAN_JSON_H
#include <stdint.h>
/* The document owns parsed storage; accessors return retained output owners. */
void *__sev_json_parse(const char *source);
int32_t __sev_json_error(void);
void *__sev_json_document_columns(void *document);
void *__sev_json_document_rows(void *document);
void __sev_json_document_release(void *document);
uintptr_t __sev_json_open(const char *source);
void __sev_json_close(uintptr_t handle);
void *__sev_json_handle_columns(uintptr_t handle);
void *__sev_json_handle_rows(uintptr_t handle);
void *__sev_json_columns(const char *source);
void *__sev_json_rows(const char *source);
#endif
