#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sched.h>

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t readable;
    pthread_cond_t writable;
    size_t capacity;
    size_t length;
    size_t head;
    bool unbounded;
    bool closed;
    uintptr_t *values;
} sev_channel;

// A successful select claim keeps the channel locked until its immediately
// following receive. This prevents another consumer from taking the value
// between the readiness check and the selected case's receive.
static _Thread_local sev_channel *sev_claimed_channel;

void *__sev_channel_create(size_t capacity) {
    sev_channel *channel = calloc(1, sizeof(sev_channel));
    if (channel == NULL) abort();
    channel->unbounded = capacity == 0;
    channel->capacity = channel->unbounded ? 16 : capacity;
    channel->values = calloc(channel->capacity, sizeof(uintptr_t));
    if (channel->values == NULL) abort();
    if (pthread_mutex_init(&channel->mutex, NULL) != 0
        || pthread_cond_init(&channel->readable, NULL) != 0
        || pthread_cond_init(&channel->writable, NULL) != 0) abort();
    return channel;
}

static void sev_channel_send(sev_channel *channel, uintptr_t value) {
    pthread_mutex_lock(&channel->mutex);
    while (channel->length == channel->capacity) {
        if (channel->unbounded) {
            size_t next_capacity = channel->capacity * 2;
            uintptr_t *next = calloc(next_capacity, sizeof(uintptr_t));
            if (next == NULL) abort();
            for (size_t index = 0; index < channel->length; index += 1) {
                next[index] = channel->values[(channel->head + index) % channel->capacity];
            }
            free(channel->values);
            channel->values = next;
            channel->capacity = next_capacity;
            channel->head = 0;
            break;
        }
        pthread_cond_wait(&channel->writable, &channel->mutex);
    }
    size_t tail = (channel->head + channel->length) % channel->capacity;
    channel->values[tail] = value;
    channel->length += 1;
    pthread_cond_signal(&channel->readable);
    pthread_mutex_unlock(&channel->mutex);
}

static uintptr_t sev_channel_recv(sev_channel *channel) {
    bool claimed = sev_claimed_channel == channel;
    if (!claimed) pthread_mutex_lock(&channel->mutex);
    while (channel->length == 0) {
        pthread_cond_wait(&channel->readable, &channel->mutex);
    }
    uintptr_t value = channel->values[channel->head];
    channel->head = (channel->head + 1) % channel->capacity;
    channel->length -= 1;
    pthread_cond_signal(&channel->writable);
    if (claimed) sev_claimed_channel = NULL;
    pthread_mutex_unlock(&channel->mutex);
    return value;
}

bool __sev_channel_claim(void *storage) {
    sev_channel *channel = storage;
    if (sev_claimed_channel != NULL) abort();
    pthread_mutex_lock(&channel->mutex);
    if (channel->length == 0) {
        pthread_mutex_unlock(&channel->mutex);
        return false;
    }
    sev_claimed_channel = channel;
    return true;
}

bool __sev_channel_is_closed(void *storage) {
    sev_channel *channel = storage;
    pthread_mutex_lock(&channel->mutex);
    bool closed = channel->closed && channel->length == 0;
    pthread_mutex_unlock(&channel->mutex);
    return closed;
}

void __sev_channel_yield(void) {
    sched_yield();
}

void __sev_channel_send_i64(void *storage, int64_t value) {
    sev_channel_send(storage, (uintptr_t)value);
}

void __sev_channel_send_ptr(void *storage, const char *value) {
    sev_channel_send(storage, (uintptr_t)value);
}

int64_t __sev_channel_recv_i64(void *storage) {
    return (int64_t)sev_channel_recv(storage);
}

const char *__sev_channel_recv_ptr(void *storage) {
    return (const char *)sev_channel_recv(storage);
}
