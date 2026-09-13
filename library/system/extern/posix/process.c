#define _GNU_SOURCE
#include "../../../core/memory/native/memory.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/syscall.h>

#include <signal.h>
#include <sys/wait.h>
#include <sys/resource.h>

double __sev_process_user_seconds(int64_t children) {
    struct rusage usage;
    if (getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &usage) != 0) return -1.0;
    return (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1000000.0;
}
double __sev_process_system_seconds(int64_t children) {
    struct rusage usage;
    if (getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &usage) != 0) return -1.0;
    return (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1000000.0;
}
int64_t __sev_process_peak_rss_kib(int64_t children) {
    struct rusage usage;
    if (getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &usage) != 0) return -1;
#ifdef __APPLE__
    return (int64_t)usage.ru_maxrss / 1024;
#else
    return (int64_t)usage.ru_maxrss;
#endif
}
void __sev_process_write_error(const char *text) {
    fputs(text, stderr);
    fflush(stderr);
}
const char *__sev_process_executable(void) {
    char *path = sev_memory_allocate(4096);
    if (!path) abort();
    ssize_t length = readlink("/proc/self/exe", path, 4095);
    if (length < 0 || length == 4095) { sev_memory_release(path); return ""; }
    path[length] = 0;
    return path;
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

/* A single-threaded package driver can share an immutable compiler snapshot
 * with isolated workers. Flush inherited streams to avoid duplicate output. */
int64_t __sev_process_fork_snapshot(void) {
    if (fflush(NULL) != 0) return -1;
    return (int64_t)fork();
}

_Bool __sev_process_kill(int64_t process) {
    return kill((pid_t)process, SIGTERM) == 0;
}

int64_t __sev_process_wait(int64_t process) {
    int status = 0;
    while (waitpid((pid_t)process, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
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


int64_t __sev_process_id(void) { return (int64_t)getpid(); }

typedef struct {
    char **arguments;
    size_t count;
    int64_t status;
    FILE *output;
    FILE *errors;
} SevCommand;

int64_t __sev_command_create(void) {
    SevCommand *command = sev_memory_zeroed(1, sizeof(*command));
    if (!command) abort();
    command->status = -1;
    return (int64_t)(intptr_t)command;
}
void __sev_command_argument(int64_t handle, const char *argument) {
    SevCommand *command = (SevCommand *)(intptr_t)handle;
    char **values = sev_memory_resize(command->arguments, (command->count + 2) * sizeof(char *));
    if (!values) abort();
    command->arguments = values;
    values[command->count++] = sev_memory_copy_text(argument);
    if (!values[command->count - 1]) abort();
    values[command->count] = NULL;
}
int64_t __sev_command_execute(int64_t handle, _Bool capture) {
    SevCommand *command = (SevCommand *)(intptr_t)handle;
    if (!command->count || command->output || command->errors) return -1;
    if (capture) {
        command->output = tmpfile();
        command->errors = tmpfile();
        if (!command->output || !command->errors) return -1;
    }
    pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0) {
        if (capture && (dup2(fileno(command->output), STDOUT_FILENO) < 0 ||
                        dup2(fileno(command->errors), STDERR_FILENO) < 0)) _exit(126);
        if (command->output) fclose(command->output);
        if (command->errors) fclose(command->errors);
        execvp(command->arguments[0], command->arguments);
        _exit(errno == ENOENT ? 127 : 126);
    }
    int status;
    while (waitpid(child, &status, 0) < 0) { if (errno != EINTR) return -1; }
    command->status = WIFEXITED(status) ? WEXITSTATUS(status) :
                      WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
    return command->status;
}
static const char *sev_command_text(FILE *file) {
    if (!file || fseek(file, 0, SEEK_END) != 0) return "";
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) return "";
    char *text = sev_memory_allocate((size_t)length + 1);
    if (!text) abort();
    size_t count = fread(text, 1, (size_t)length, file);
    text[count] = 0;
    return text;
}
const char *__sev_command_stdout(int64_t handle) {
    return sev_command_text(((SevCommand *)(intptr_t)handle)->output);
}
const char *__sev_command_stderr(int64_t handle) {
    return sev_command_text(((SevCommand *)(intptr_t)handle)->errors);
}
void __sev_command_drop(int64_t handle) {
    SevCommand *command = (SevCommand *)(intptr_t)handle;
    if (command->output) fclose(command->output);
    if (command->errors) fclose(command->errors);
    for (size_t i = 0; i < command->count; ++i) sev_memory_release(command->arguments[i]);
    sev_memory_release(command->arguments);
    sev_memory_release(command);
}

#include <time.h>
#include <math.h>
double __sev_time_monotonic(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
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


double __sev_time_wall(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
}
