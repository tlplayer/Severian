#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern _Bool __sev_os_make_directories(const char *);
extern int32_t __sev_file_write_text(const char *, const char *);
extern const char *__sev_file_read_text(const char *);
extern int32_t __sev_file_error(void);
extern const char *__sev_os_temporary(const char *, const char *);
extern int32_t __sev_os_remove_tree(const char *);
extern int32_t __sev_os_publish_directory(const char *, const char *);
extern int64_t __sev_os_directory_open(const char *);
extern const char *__sev_os_directory_next(int64_t);
extern int32_t __sev_os_directory_close(int64_t);
extern int64_t __sev_command_create(void);
extern void __sev_command_argument(int64_t, const char *);
extern int64_t __sev_command_execute(int64_t, _Bool);
extern const char *__sev_command_stdout(int64_t);
extern const char *__sev_command_stderr(int64_t);
extern void __sev_command_drop(int64_t);
extern double __sev_process_user_seconds(int64_t);
extern double __sev_process_system_seconds(int64_t);
extern int64_t __sev_process_peak_rss_kib(int64_t);
extern const char *__sev_process_executable(void);

static void joined(char *out, size_t size, const char *root, const char *name) {
    assert(snprintf(out, size, "%s/%s", root, name) < (int)size);
}
int main(void) {
    assert(__sev_process_user_seconds(0) >= 0.0);
    assert(__sev_process_system_seconds(0) >= 0.0);
    assert(__sev_process_peak_rss_kib(0) > 0);
    const char *self = __sev_process_executable();
    assert(*self == '/' && access(self, X_OK) == 0);
    free((void *)self);
    char root[] = "/tmp/sev-host-contract-XXXXXX";
    assert(mkdtemp(root));
    char path[4096], release[4096], left[4096], right[4096];
    joined(path, sizeof(path), root, "space ' quote\nλ.txt");
    assert(__sev_file_write_text(path, "") == 0);
    const char *text = __sev_file_read_text(path);
    assert(__sev_file_error() == 0 && !strcmp(text, ""));
    free((void *)text);
    assert(__sev_file_write_text(path, "λ😀\n") == 0);
    text = __sev_file_read_text(path);
    assert(__sev_file_error() == 0 && !strcmp(text, "λ😀\n"));
    free((void *)text);
    assert(!strcmp(__sev_file_read_text(root), ""));
    assert(__sev_file_error() != 0);
    FILE *binary = fopen(path, "wb");
    assert(binary && fwrite("a\0b", 1, 3, binary) == 3 && fclose(binary) == 0);
    assert(!strcmp(__sev_file_read_text(path), "") && __sev_file_error() == EILSEQ);
    assert(unlink(path) == 0);
    assert(!strcmp(__sev_file_read_text(path), "") && __sev_file_error() == ENOENT);

    const char *temporary = __sev_os_temporary(root, "stage ");
    assert(*temporary && access(temporary, F_OK) == 0);
    int64_t directory = __sev_os_directory_open(root);
    assert(directory);
    int entries = 0;
    while (*__sev_os_directory_next(directory)) ++entries;
    assert(entries >= 3 && __sev_os_directory_close(directory) == 0);
    free((void *)temporary);

    int64_t command = __sev_command_create();
    __sev_command_argument(command, "printf");
    __sev_command_argument(command, "%s");
    __sev_command_argument(command, "a ' $HOME $(false) ;\nλ");
    assert(__sev_command_execute(command, 1) == 0);
    text = __sev_command_stdout(command);
    assert(!strcmp(text, "a ' $HOME $(false) ;\nλ"));
    free((void *)text);
    __sev_command_drop(command);
    command = __sev_command_create();
    __sev_command_argument(command, "sh");
    __sev_command_argument(command, "-c");
    __sev_command_argument(command, "printf output; printf error >&2; exit 7");
    assert(__sev_command_execute(command, 1) == 7);
    text = __sev_command_stdout(command);
    assert(!strcmp(text, "output")); free((void *)text);
    text = __sev_command_stderr(command);
    assert(!strcmp(text, "error")); free((void *)text);
    __sev_command_drop(command);
    command = __sev_command_create();
    __sev_command_argument(command, "/does-not-exist/sev-test");
    assert(__sev_command_execute(command, 1) == 127);
    __sev_command_drop(command);

    joined(left, sizeof(left), root, "left");
    joined(right, sizeof(right), root, "right");
    joined(release, sizeof(release), root, "release");
    assert(__sev_os_make_directories(left) && __sev_os_make_directories(right));
    pid_t children[2];
    for (int i = 0; i < 2; ++i) {
        children[i] = fork(); assert(children[i] >= 0);
        if (!children[i]) _exit(__sev_os_publish_directory(i ? right : left, release) == 0 ? 0 : 1);
    }
    int winners = 0;
    for (int i = 0; i < 2; ++i) {
        int status; assert(waitpid(children[i], &status, 0) == children[i]);
        assert(WIFEXITED(status)); winners += WEXITSTATUS(status) == 0;
    }
    assert(winners == 1);
    assert(__sev_os_publish_directory(access(left, F_OK) == 0 ? left : right, release) == EEXIST);
    joined(path, sizeof(path), root, "dangling");
    assert(symlink("missing", path) == 0);
    assert(__sev_os_publish_directory(access(left, F_OK) == 0 ? left : right, path) == EEXIST);
    assert(__sev_os_remove_tree(root) == 0 && access(root, F_OK) != 0);
    puts("system host contracts passed");
}
