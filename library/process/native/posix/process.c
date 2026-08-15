#include "process_abi.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <crt_externs.h>
#endif

typedef struct {
    char **values;
    size_t length;
    size_t capacity;
} sev_process_arguments;

static char *sev_copy_view(sev_string_view_v1 view) {
    if (!view.data && view.length) return NULL;
    char *copy = malloc(view.length + 1);
    if (!copy) return NULL;
    if (view.length) memcpy(copy, view.data, view.length);
    copy[view.length] = '\0';
    return copy;
}

static sev_string_view_v1 sev_view(const char *value) {
    sev_string_view_v1 view = {
        .data = (const uint8_t *)value,
        .length = value ? strlen(value) : 0,
    };
    return view;
}

static bool sev_arguments_push(sev_process_arguments *arguments, const char *value, size_t length) {
    if (arguments->length == arguments->capacity) {
        size_t capacity = arguments->capacity ? arguments->capacity * 2 : 8;
        char **values = realloc(arguments->values, capacity * sizeof(*values));
        if (!values) return false;
        arguments->values = values;
        arguments->capacity = capacity;
    }
    char *copy = malloc(length + 1);
    if (!copy) return false;
    if (length) memcpy(copy, value, length);
    copy[length] = '\0';
    arguments->values[arguments->length++] = copy;
    return true;
}

static void sev_arguments_destroy(sev_process_arguments *arguments) {
    if (!arguments) return;
    for (size_t index = 0; index < arguments->length; ++index) {
        free(arguments->values[index]);
    }
    free(arguments->values);
    free(arguments);
}

int64_t sev_abi_v1_process_run(sev_string_view_v1 command) {
    char *text = sev_copy_view(command);
    if (!text) return -1;
    int status = system(text);
    free(text);
    return status < 0 ? -1 : WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

int64_t sev_abi_v1_process_spawn(sev_string_view_v1 command) {
    char *text = sev_copy_view(command);
    if (!text) return -1;
    pid_t child = fork();
    if (child == 0) {
        execl("/bin/sh", "sh", "-c", text, NULL);
        _exit(127);
    }
    free(text);
    return (int64_t)child;
}

int64_t sev_abi_v1_process_wait(int64_t process) {
    int status = 0;
    if (waitpid((pid_t)process, &status, 0) < 0) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

bool sev_abi_v1_process_kill(int64_t process) {
    return kill((pid_t)process, SIGTERM) == 0;
}

void sev_abi_v1_process_exit(int64_t status) {
    exit((int)status);
}

int32_t sev_abi_v1_process_arguments(sev_handle_v1 *output) {
    if (!output) return -1;
    output->value = NULL;
    sev_process_arguments *arguments = calloc(1, sizeof(*arguments));
    if (!arguments) return -1;
#ifdef __APPLE__
    int count = *_NSGetArgc();
    char **values = *_NSGetArgv();
    for (int index = 0; index < count; ++index) {
        if (!sev_arguments_push(arguments, values[index], strlen(values[index]))) {
            sev_arguments_destroy(arguments);
            return -1;
        }
    }
#else
    FILE *file = fopen("/proc/self/cmdline", "rb");
    if (!file) {
        sev_arguments_destroy(arguments);
        return -1;
    }
    char *buffer = NULL;
    size_t capacity = 0;
    ssize_t count;
    while ((count = getdelim(&buffer, &capacity, '\0', file)) >= 0) {
        size_t length = count > 0 && buffer[count - 1] == '\0'
            ? (size_t)count - 1
            : (size_t)count;
        if (!sev_arguments_push(arguments, buffer, length)) {
            free(buffer);
            fclose(file);
            sev_arguments_destroy(arguments);
            return -1;
        }
    }
    free(buffer);
    if (fclose(file) != 0) {
        sev_arguments_destroy(arguments);
        return -1;
    }
#endif
    output->value = arguments;
    return 0;
}

size_t sev_abi_v1_process_arguments_length(sev_handle_v1 handle) {
    sev_process_arguments *arguments = handle.value;
    return arguments ? arguments->length : 0;
}

sev_string_view_v1 sev_abi_v1_process_arguments_at(sev_handle_v1 handle, size_t index) {
    sev_process_arguments *arguments = handle.value;
    if (!arguments || index >= arguments->length) return sev_view("");
    return sev_view(arguments->values[index]);
}

void sev_abi_v1_process_arguments_release(sev_handle_v1 handle) {
    sev_arguments_destroy(handle.value);
}
