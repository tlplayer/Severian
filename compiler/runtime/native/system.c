#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char *message;
    char *call_stack;
} SevError;

static pthread_mutex_t __sev_task_mutex = PTHREAD_MUTEX_INITIALIZER;

void __sev_task_lock(void) {
    if (pthread_mutex_lock(&__sev_task_mutex) != 0) abort();
}

void __sev_task_unlock(void) {
    if (pthread_mutex_unlock(&__sev_task_mutex) != 0) abort();
}

static char *__sev_copy_text(const char *text) {
    const char *source = text == NULL ? "" : text;
    size_t size = strlen(source) + 1;
    char *copy = malloc(size);
    if (copy == NULL) abort();
    memcpy(copy, source, size);
    return copy;
}

const char *__sev_error_create(const char *message, const char *function) {
    SevError *error = malloc(sizeof(SevError));
    if (error == NULL) abort();
    error->message = __sev_copy_text(message);
    error->call_stack = __sev_copy_text(function);
    return (const char *)error;
}

const char *__sev_error_propagate(const char *opaque, const char *function) {
    SevError *error = (SevError *)opaque;
    size_t current = strlen(error->call_stack);
    size_t frame = strlen(function);
    char *stack = realloc(error->call_stack, current + frame + 5);
    if (stack == NULL) abort();
    memcpy(stack + current, " -> ", 4);
    memcpy(stack + current + 4, function, frame + 1);
    error->call_stack = stack;
    return opaque;
}

const char *__sev_error_message(const char *opaque) {
    return ((const SevError *)opaque)->message;
}

const char *__sev_error_call_stack(const char *opaque) {
    return ((const SevError *)opaque)->call_stack;
}

double __sev_time_monotonic(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}

_Bool __sev_approximate_f64(double actual, double expected, double atol, double rtol) {
    if (atol < 0.0 || rtol < 0.0) return 0;
    double error = fabs(actual - expected);
    double tolerance = atol + rtol * fabs(expected);
    if (error <= tolerance) return 1;
    fprintf(
        stderr,
        "approximate mismatch: actual=%.17g expected=%.17g error=%.17g tolerance=%.17g\n",
        actual,
        expected,
        error,
        tolerance
    );
    return 0;
}

void __sev_os_wait(double seconds) {
    if (!(seconds > 0.0)) return;
    struct timespec remaining = {
        .tv_sec = (time_t)seconds,
        .tv_nsec = (long)((seconds - floor(seconds)) * 1000000000.0),
    };
    while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {
    }
}

void __sev_throw(const char *error) {
    fflush(stdout);
    fprintf(stderr, "error: %s\n", __sev_error_message(error));
    exit(EXIT_FAILURE);
}

void __sev_panic(const char *message) {
    fflush(stdout);
    fprintf(stderr, "panic: %s\n", message == NULL ? "" : message);
    fflush(stderr);
    abort();
}

int64_t __sev_process_run(const char *command) {
    int status = system(command);
    if (status == -1) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128;
    return status;
}

int64_t __sev_process_spawn(const char *command) {
    pid_t process = fork();
    if (process < 0) return -1;
    if (process == 0) {
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
        _exit(127);
    }
    return (int64_t)process;
}

_Bool __sev_process_kill(int64_t process) {
    return kill((pid_t)process, SIGTERM) == 0;
}

int64_t __sev_process_wait(int64_t process) {
    int status = 0;
    if (waitpid((pid_t)process, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128;
    return status;
}

const char *__sev_environment_get(const char *name) {
    const char *value = getenv(name);
    return value == NULL ? "" : value;
}

const char *__sev_environment_get_default(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value == NULL ? fallback : value;
}

_Bool __sev_environment_set(const char *name, const char *value) {
    return setenv(name, value, 1) == 0;
}

_Bool __sev_environment_remove(const char *name) {
    return unsetenv(name) == 0;
}

const char *__sev_select_string(_Bool condition, const char *then_value, const char *else_value) {
    return condition ? then_value : else_value;
}

double __sev_select_f64(_Bool condition, double then_value, double else_value) {
    return condition ? then_value : else_value;
}

float __sev_select_f32(_Bool condition, float then_value, float else_value) {
    return condition ? then_value : else_value;
}

int64_t __sev_select_i64(_Bool condition, int64_t then_value, int64_t else_value) {
    return condition ? then_value : else_value;
}

_Bool __sev_select_bool(_Bool condition, _Bool then_value, _Bool else_value) {
    return condition ? then_value : else_value;
}

double __sev_pow_f64_i64(double base, int64_t exponent) {
    _Bool reciprocal = exponent < 0;
    uint64_t remaining = reciprocal ? (uint64_t)(-(exponent + 1)) + 1 : (uint64_t)exponent;
    double result = 1.0;
    while (remaining != 0) {
        if ((remaining & 1u) != 0) result *= base;
        base *= base;
        remaining >>= 1u;
    }
    return reciprocal ? 1.0 / result : result;
}

double __sev_pow_f64_f64(double base, double exponent) {
    return pow(base, exponent);
}

int64_t __sev_pow_i64_i64(int64_t base, int64_t exponent) {
    if (exponent < 0) {
        if (base == 1) return 1;
        if (base == -1) return (exponent & 1) == 0 ? 1 : -1;
        return 0;
    }
    uint64_t factor = (uint64_t)base;
    uint64_t result = 1;
    uint64_t remaining = (uint64_t)exponent;
    while (remaining != 0) {
        if ((remaining & 1u) != 0) result *= factor;
        factor *= factor;
        remaining >>= 1u;
    }
    return (int64_t)result;
}
