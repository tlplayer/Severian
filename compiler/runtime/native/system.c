#define _POSIX_C_SOURCE 200809L

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
