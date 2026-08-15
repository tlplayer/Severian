#include "file_abi.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *data;
    size_t length;
} sev_file_text;

static sev_string_view_v1 sev_file_view(const char *message) {
    sev_string_view_v1 view = {
        .data = (const uint8_t *)message,
        .length = message ? strlen(message) : 0,
    };
    return view;
}

static int32_t sev_file_error(sev_error_v1 *error, int code, const char *message) {
    if (error) {
        error->code = code;
        error->message = sev_file_view(message);
    }
    return -1;
}

static char *sev_file_copy_path(sev_string_view_v1 path) {
    if ((!path.data && path.length) ||
        (path.length && memchr(path.data, '\0', path.length))) {
        return NULL;
    }
    char *copy = malloc(path.length + 1);
    if (!copy) return NULL;
    if (path.length) memcpy(copy, path.data, path.length);
    copy[path.length] = '\0';
    return copy;
}

int32_t sev_abi_v1_file_read_text(
    sev_string_view_v1 path,
    sev_handle_v1 *content,
    sev_error_v1 *error
) {
    if (!content) return sev_file_error(error, EINVAL, "missing output handle");
    content->value = NULL;

    char *path_text = sev_file_copy_path(path);
    if (!path_text) return sev_file_error(error, EINVAL, "invalid file path");

    FILE *stream = fopen(path_text, "rb");
    free(path_text);
    if (!stream) return sev_file_error(error, errno, strerror(errno));

    size_t capacity = 4096;
    size_t length = 0;
    uint8_t *data = malloc(capacity + 1);
    if (!data) {
        fclose(stream);
        return sev_file_error(error, ENOMEM, "could not allocate file buffer");
    }

    while (!feof(stream)) {
        if (length == capacity) {
            if (capacity > SIZE_MAX / 2) {
                free(data);
                fclose(stream);
                return sev_file_error(error, EFBIG, "file is too large");
            }
            capacity *= 2;
            uint8_t *grown = realloc(data, capacity + 1);
            if (!grown) {
                free(data);
                fclose(stream);
                return sev_file_error(error, ENOMEM, "could not grow file buffer");
            }
            data = grown;
        }
        size_t count = fread(data + length, 1, capacity - length, stream);
        length += count;
        if (ferror(stream)) {
            int code = errno ? errno : EIO;
            free(data);
            fclose(stream);
            return sev_file_error(error, code, strerror(code));
        }
    }
    fclose(stream);
    data[length] = '\0';

    sev_file_text *text = malloc(sizeof(*text));
    if (!text) {
        free(data);
        return sev_file_error(error, ENOMEM, "could not allocate text handle");
    }
    text->data = data;
    text->length = length;
    content->value = text;
    return 0;
}

sev_string_view_v1 sev_abi_v1_file_text_value(sev_handle_v1 content) {
    const sev_file_text *text = content.value;
    sev_string_view_v1 view = {
        .data = text ? text->data : NULL,
        .length = text ? text->length : 0,
    };
    return view;
}

void sev_abi_v1_file_text_release(sev_handle_v1 content) {
    sev_file_text *text = content.value;
    if (!text) return;
    free(text->data);
    free(text);
}
