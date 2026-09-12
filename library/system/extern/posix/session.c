#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

/* One private, single-client channel per package invocation. No filesystem
 * socket, global daemon, inherited exec descriptor, or mutable shared heap. */
static int session_client = -1;
static int session_server = -1;
static int session_client_reply = -1;
static int session_server_reply = -1;

int64_t __sev_session_start(void) {
    if (session_client >= 0 || session_server >= 0) return -EALREADY;
    int channel[2], reply[2];
    if (pipe2(channel, O_CLOEXEC)) return -errno;
    if (pipe2(reply, O_CLOEXEC)) { int e = errno; close(channel[0]); close(channel[1]); return -e; }
    if (fflush(NULL)) { int e = errno; close(channel[0]); close(channel[1]); close(reply[0]); close(reply[1]); return -e; }
    pid_t child = fork();
    if (child < 0) { int e = errno; close(channel[0]); close(channel[1]); close(reply[0]); close(reply[1]); return -e; }
    if (child == 0) { close(channel[0]); close(reply[1]); session_client = channel[1]; session_client_reply = reply[0]; }
    else { close(channel[1]); close(reply[0]); session_server = channel[0]; session_server_reply = reply[1]; }
    return child;
}

_Bool __sev_session_available(void) { return session_client >= 0; }

static ssize_t send_packet(int fd, const void *data, size_t length) {
    unsigned char packet[4100];
    if (length > 4096) return -1;
    uint32_t size = (uint32_t)length;
    memcpy(packet, &size, sizeof(size));
    memcpy(packet + sizeof(size), data, length);
    sigset_t blocked, previous;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGPIPE);
    if (sigprocmask(SIG_BLOCK, &blocked, &previous)) return -1;
    size_t offset = 0;
    while (offset < length + sizeof(size)) {
        ssize_t count = write(fd, packet + offset, length + sizeof(size) - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += (size_t)count;
    }
    if (!sigismember(&previous, SIGPIPE)) {
        struct timespec zero = {0, 0};
        while (sigtimedwait(&blocked, NULL, &zero) >= 0) {}
    }
    sigprocmask(SIG_SETMASK, &previous, NULL);
    return offset == length + sizeof(size) ? (ssize_t)length : -1;
}

static _Bool read_all(int fd, void *data, size_t length) {
    size_t offset = 0;
    while (offset < length) {
        ssize_t count = read(fd, (char *)data + offset, length - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        offset += (size_t)count;
    }
    return 1;
}

static ssize_t receive_packet(int fd, void *data, size_t capacity) {
    uint32_t length;
    if (!read_all(fd, &length, sizeof(length))) return 0;
    if (length > capacity) return -1;
    return read_all(fd, data, length) ? (ssize_t)length : -1;
}

int64_t __sev_session_submit(const char *request) {
    if (session_client < 0) return -ENOTCONN;
    size_t length = strlen(request);
    if (!length || length >= 4096) return -EINVAL;
    if (send_packet(session_client, request, length) != (ssize_t)length) return -EPIPE;
    int64_t status;
    if (receive_packet(session_client_reply, &status, sizeof(status)) != sizeof(status)) return -EPIPE;
    return status;
}

const char *__sev_session_request(void) {
    static char request[4096];
    if (session_server < 0) return "";
    ssize_t length = receive_packet(session_server, request, sizeof(request));
    if (length <= 0 || length >= (ssize_t)sizeof(request)) return "";
    request[length] = 0;
    return request;
}

_Bool __sev_session_reply(int64_t status) {
    return session_server_reply >= 0 && send_packet(session_server_reply, &status, sizeof(status)) == sizeof(status);
}

void __sev_session_detach(void) {
    if (session_client >= 0) close(session_client);
    if (session_server >= 0) close(session_server);
    if (session_client_reply >= 0) close(session_client_reply);
    if (session_server_reply >= 0) close(session_server_reply);
    session_client = session_server = -1;
    session_client_reply = session_server_reply = -1;
}

/* Applied only after forking a unit; the warm parent keeps its own limits. */
_Bool __sev_session_limits(int64_t seconds, int64_t bytes) {
    if (seconds < 0 || seconds > UINT_MAX || bytes < 0) return 0;
    if (setpgid(0, 0)) return 0;
    if (bytes) {
        struct rlimit limit;
        if (getrlimit(RLIMIT_AS, &limit)) return 0;
        limit.rlim_cur = (uint64_t)bytes > limit.rlim_max ? limit.rlim_max : (rlim_t)bytes;
        if (setrlimit(RLIMIT_AS, &limit)) return 0;
    }
    alarm((unsigned)seconds);
    return 1;
}

int64_t __sev_session_wait(int64_t worker) {
    if (worker <= 0 || worker > INT_MAX) return -EINVAL;
    int status;
    pid_t result;
    do { result = waitpid((pid_t)worker, &status, 0); } while (result < 0 && errno == EINTR);
    if (result < 0) return -errno;
    /* A timed-out compiler may have tool children. Reap its entire group. */
    kill(-(pid_t)worker, SIGKILL);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return WTERMSIG(status) == SIGALRM ? 124 : 128 + WTERMSIG(status);
    return 1;
}
