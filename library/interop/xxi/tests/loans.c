#define _GNU_SOURCE
#include "../extern/c/bytes.h"
#include <assert.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
extern void *__sev_list_create(void);
extern void __sev_list_push_u8(void *, uint8_t);
int main(void) {
    uint64_t baseline = __sev_storage_live_bytes();
    void *owner = __sev_list_create();
    __sev_list_push_u8(owner, 0);
    __sev_list_push_u8(owner, 255);
    uint64_t live = __sev_storage_live_bytes();
    uint64_t allocations = __sev_storage_thread_allocations();
    uint64_t allocated = __sev_storage_thread_allocated_bytes();
    for (int iteration = 0; iteration < 1000; ++iteration) {
        sev_xxi_bytes_loan loan = sev_xxi_bytes_acquire((sev_xxi_list){owner}, 0);
        assert(loan.view.length == 2 && loan.view.data[1] == 255);
        loan.view.data[1] = 42;
        sev_xxi_bytes_release(&loan);
        assert(__sev_list_index_u8(owner, 1) == 255);
        assert(__sev_storage_live_bytes() == live);
    }
    uint64_t calls = __sev_storage_thread_allocations() - allocations;
    uint64_t bytes = __sev_storage_thread_allocated_bytes() - allocated;
    sev_xxi_bytes_loan loan = sev_xxi_bytes_acquire((sev_xxi_list){owner}, 1);
    loan.view.data[0] = 127;
    sev_xxi_bytes_release(&loan);
    assert(__sev_list_index_u8(owner, 0) == 127);
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        struct rlimit limit = {0, 0};
        assert(setrlimit(RLIMIT_CORE, &limit) == 0);
        freopen("/dev/null", "w", stderr);
        sev_xxi_bytes_loan invalid = sev_xxi_bytes_acquire((sev_xxi_list){owner}, 1);
        invalid.view.length += 1;
        sev_xxi_bytes_release(&invalid);
        _exit(99);
    }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    __sev_storage_release(owner);
    assert(__sev_storage_live_bytes() == baseline);
    printf("{\"iterations\":1000,\"allocations\":%llu,\"allocated_bytes\":%llu,\"retained_bytes\":%llu}\n", (unsigned long long)calls, (unsigned long long)bytes, (unsigned long long)(__sev_storage_live_bytes() - baseline));
}
