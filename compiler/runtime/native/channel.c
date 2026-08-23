#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t readable;
    pthread_cond_t writable;
    size_t capacity;
    size_t length;
    size_t head;
    uintptr_t *values;
} sev_channel;

void *__sev_channel_create(size_t capacity) {
    sev_channel *channel = calloc(1, sizeof(sev_channel));
    if (channel == NULL) abort();
    channel->capacity = capacity == 0 ? 16 : capacity;
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
        pthread_cond_wait(&channel->writable, &channel->mutex);
    }
    size_t tail = (channel->head + channel->length) % channel->capacity;
    channel->values[tail] = value;
    channel->length += 1;
    pthread_cond_signal(&channel->readable);
    pthread_mutex_unlock(&channel->mutex);
}

static uintptr_t sev_channel_recv(sev_channel *channel) {
    pthread_mutex_lock(&channel->mutex);
    while (channel->length == 0) {
        pthread_cond_wait(&channel->readable, &channel->mutex);
    }
    uintptr_t value = channel->values[channel->head];
    channel->head = (channel->head + 1) % channel->capacity;
    channel->length -= 1;
    pthread_cond_signal(&channel->writable);
    pthread_mutex_unlock(&channel->mutex);
    return value;
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
