#include "network_abi.h"

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef NI_MAXHOST
#define NI_MAXHOST 1025
#endif
#ifndef NI_MAXSERV
#define NI_MAXSERV 32
#endif

enum sev_network_kind {
    SEV_NETWORK_TCP = 1,
    SEV_NETWORK_LISTENER = 2,
    SEV_NETWORK_BYTES = 3,
    SEV_NETWORK_ADDRESS = 4,
    SEV_NETWORK_ADDRESS_LIST = 5,
    SEV_NETWORK_UDP = 6,
    SEV_NETWORK_PACKET = 7,
    SEV_NETWORK_TEXT = 8,
};

typedef struct { int kind; int descriptor; bool closed; } sev_network_socket;
typedef struct { int kind; uint8_t *data; size_t length; } sev_network_bytes;
typedef struct { int kind; char host[NI_MAXHOST]; uint16_t port; int32_t family; } sev_network_address;
typedef struct { int kind; char **values; size_t length; } sev_network_address_list;
typedef struct { int kind; uint8_t *data; size_t length; char host[NI_MAXHOST]; uint16_t port; } sev_network_packet;
typedef struct { int kind; char *value; size_t length; } sev_network_text;

static sev_string_view_v1 sev_view(const char *value) {
    sev_string_view_v1 view = { .data = (const uint8_t *)value, .length = value ? strlen(value) : 0 };
    return view;
}

static void sev_clear_error(sev_error_v1 *error) {
    if (!error) return;
    error->code = 0;
    error->message = sev_view("");
}

static int32_t sev_fail(sev_error_v1 *error, int32_t code, const char *message) {
    if (error) {
        error->code = code ? code : -1;
        error->message = sev_view(message);
    }
    return code ? code : -1;
}

static char *sev_copy_view(sev_string_view_v1 view) {
    if (!view.data && view.length) return NULL;
    char *copy = malloc(view.length + 1);
    if (!copy) return NULL;
    if (view.length) memcpy(copy, view.data, view.length);
    copy[view.length] = '\0';
    return copy;
}

static bool sev_socket_kind(sev_handle_v1 handle, int kind, sev_network_socket **output) {
    sev_network_socket *socket = handle.value;
    if (!socket || socket->kind != kind || socket->closed) return false;
    *output = socket;
    return true;
}

static int32_t sev_open_socket(
    sev_string_view_v1 host,
    uint16_t port,
    int type,
    bool passive,
    int kind,
    sev_handle_v1 *output,
    sev_error_v1 *error
) {
    if (!output) return sev_fail(error, EINVAL, "native provider received no output handle");
    output->value = NULL;
    char *host_text = sev_copy_view(host);
    if (!host_text) return sev_fail(error, ENOMEM, "could not copy network host");
    char service[8];
    snprintf(service, sizeof(service), "%u", (unsigned)port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = type;
    hints.ai_protocol = type == SOCK_STREAM ? IPPROTO_TCP : IPPROTO_UDP;
    hints.ai_flags = passive ? AI_PASSIVE : 0;
    const char *query_host = passive && host.length == 0 ? NULL : host_text;
    struct addrinfo *addresses = NULL;
    int resolved = getaddrinfo(query_host, service, &hints, &addresses);
    free(host_text);
    if (resolved != 0) return sev_fail(error, resolved, gai_strerror(resolved));

    int descriptor = -1;
    int saved_error = 0;
    for (struct addrinfo *address = addresses; address; address = address->ai_next) {
        descriptor = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (descriptor < 0) { saved_error = errno; continue; }
        if (passive) {
            int reuse = 1;
            setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        }
        int status = passive
            ? bind(descriptor, address->ai_addr, address->ai_addrlen)
            : connect(descriptor, address->ai_addr, address->ai_addrlen);
        if (status == 0 && (!passive || type != SOCK_STREAM || listen(descriptor, 128) == 0)) break;
        saved_error = errno;
        close(descriptor);
        descriptor = -1;
    }
    freeaddrinfo(addresses);
    if (descriptor < 0) return sev_fail(error, saved_error, passive ? "could not bind network address" : "could not connect to network host");

    sev_network_socket *socket = calloc(1, sizeof(*socket));
    if (!socket) { close(descriptor); return sev_fail(error, ENOMEM, "could not allocate network handle"); }
    socket->kind = kind;
    socket->descriptor = descriptor;
    output->value = socket;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_connect(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *connection, sev_error_v1 *error) {
    if (host.length == 0) return sev_fail(error, EINVAL, "network host cannot be empty");
    return sev_open_socket(host, port, SOCK_STREAM, false, SEV_NETWORK_TCP, connection, error);
}

int32_t sev_abi_v1_network_listen(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *listener, sev_error_v1 *error) {
    return sev_open_socket(host, port, SOCK_STREAM, true, SEV_NETWORK_LISTENER, listener, error);
}

int32_t sev_abi_v1_network_accept(sev_handle_v1 listener, sev_handle_v1 *connection, sev_error_v1 *error) {
    sev_network_socket *server = NULL;
    if (!connection || !sev_socket_kind(listener, SEV_NETWORK_LISTENER, &server))
        return sev_fail(error, EBADF, "network listener is closed");
    int descriptor;
    do descriptor = accept(server->descriptor, NULL, NULL); while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) return sev_fail(error, errno, "could not accept network connection");
    sev_network_socket *client = calloc(1, sizeof(*client));
    if (!client) { close(descriptor); return sev_fail(error, ENOMEM, "could not allocate connection handle"); }
    client->kind = SEV_NETWORK_TCP;
    client->descriptor = descriptor;
    connection->value = client;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_read(sev_handle_v1 connection, size_t count, sev_handle_v1 *bytes, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!bytes || !sev_socket_kind(connection, SEV_NETWORK_TCP, &socket))
        return sev_fail(error, EBADF, "network connection is closed");
    if (count > (size_t)SSIZE_MAX) return sev_fail(error, EINVAL, "network read is too large");
    sev_network_bytes *result = calloc(1, sizeof(*result));
    if (!result) return sev_fail(error, ENOMEM, "could not allocate read result");
    result->kind = SEV_NETWORK_BYTES;
    result->data = malloc(count ? count : 1);
    if (!result->data) { free(result); return sev_fail(error, ENOMEM, "could not allocate read buffer"); }
    ssize_t received;
    do received = recv(socket->descriptor, result->data, count, 0); while (received < 0 && errno == EINTR);
    if (received < 0) { int code = errno; free(result->data); free(result); return sev_fail(error, code, "network read failed"); }
    result->length = (size_t)received;
    bytes->value = result;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_write(sev_handle_v1 connection, sev_bytes_view_v1 data, size_t *written, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!written || !sev_socket_kind(connection, SEV_NETWORK_TCP, &socket))
        return sev_fail(error, EBADF, "network connection is closed");
    *written = 0;
    while (*written < data.length) {
        ssize_t count;
#ifdef MSG_NOSIGNAL
        do count = send(socket->descriptor, data.data + *written, data.length - *written, MSG_NOSIGNAL); while (count < 0 && errno == EINTR);
#else
        do count = send(socket->descriptor, data.data + *written, data.length - *written, 0); while (count < 0 && errno == EINTR);
#endif
        if (count <= 0) return sev_fail(error, errno, "network write failed");
        *written += (size_t)count;
    }
    sev_clear_error(error);
    return 0;
}

static int32_t sev_close_socket(sev_handle_v1 handle, int kind, sev_error_v1 *error) {
    sev_network_socket *socket = handle.value;
    if (!socket || socket->kind != kind || socket->closed)
        return sev_fail(error, EBADF, "network handle is closed");
    if (close(socket->descriptor) != 0) return sev_fail(error, errno, "could not close network handle");
    socket->closed = true;
    free(socket);
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_close(sev_handle_v1 connection, sev_error_v1 *error) {
    return sev_close_socket(connection, SEV_NETWORK_TCP, error);
}

int32_t sev_abi_v1_network_descriptor(sev_handle_v1 connection) {
    sev_network_socket *socket = NULL;
    return sev_socket_kind(connection, SEV_NETWORK_TCP, &socket) ? socket->descriptor : -1;
}

int32_t sev_abi_v1_network_listener_close(sev_handle_v1 listener, sev_error_v1 *error) {
    return sev_close_socket(listener, SEV_NETWORK_LISTENER, error);
}

static int32_t sev_inspect_address(sev_handle_v1 handle, int kind, bool peer, sev_handle_v1 *output, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!output || !sev_socket_kind(handle, kind, &socket)) return sev_fail(error, EBADF, "network handle is closed");
    struct sockaddr_storage raw;
    socklen_t raw_size = sizeof(raw);
    int status = peer
        ? getpeername(socket->descriptor, (struct sockaddr *)&raw, &raw_size)
        : getsockname(socket->descriptor, (struct sockaddr *)&raw, &raw_size);
    if (status != 0) return sev_fail(error, errno, "could not inspect network address");
    sev_network_address *address = calloc(1, sizeof(*address));
    if (!address) return sev_fail(error, ENOMEM, "could not allocate network address");
    address->kind = SEV_NETWORK_ADDRESS;
    if (getnameinfo((struct sockaddr *)&raw, raw_size, address->host, sizeof(address->host), NULL, 0, NI_NUMERICHOST) != 0) {
        free(address);
        return sev_fail(error, EINVAL, "could not format network address");
    }
    if (raw.ss_family == AF_INET) {
        address->family = 4;
        address->port = ntohs(((struct sockaddr_in *)&raw)->sin_port);
    } else if (raw.ss_family == AF_INET6) {
        address->family = 6;
        address->port = ntohs(((struct sockaddr_in6 *)&raw)->sin6_port);
    }
    output->value = address;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_local_address(sev_handle_v1 connection, sev_handle_v1 *address, sev_error_v1 *error) {
    return sev_inspect_address(connection, SEV_NETWORK_TCP, false, address, error);
}
int32_t sev_abi_v1_network_remote_address(sev_handle_v1 connection, sev_handle_v1 *address, sev_error_v1 *error) {
    return sev_inspect_address(connection, SEV_NETWORK_TCP, true, address, error);
}
int32_t sev_abi_v1_network_listener_address(sev_handle_v1 listener, sev_handle_v1 *address, sev_error_v1 *error) {
    return sev_inspect_address(listener, SEV_NETWORK_LISTENER, false, address, error);
}

sev_string_view_v1 sev_abi_v1_network_address_host(sev_handle_v1 handle) {
    sev_network_address *address = handle.value;
    return address && address->kind == SEV_NETWORK_ADDRESS ? sev_view(address->host) : sev_view("");
}
uint16_t sev_abi_v1_network_address_port(sev_handle_v1 handle) {
    sev_network_address *address = handle.value;
    return address && address->kind == SEV_NETWORK_ADDRESS ? address->port : 0;
}
int32_t sev_abi_v1_network_address_family(sev_handle_v1 handle) {
    sev_network_address *address = handle.value;
    return address && address->kind == SEV_NETWORK_ADDRESS ? address->family : 0;
}
void sev_abi_v1_network_address_release(sev_handle_v1 handle) {
    sev_network_address *address = handle.value;
    if (address && address->kind == SEV_NETWORK_ADDRESS) free(address);
}

size_t sev_abi_v1_network_bytes_length(sev_handle_v1 handle) {
    sev_network_bytes *bytes = handle.value;
    return bytes && bytes->kind == SEV_NETWORK_BYTES ? bytes->length : 0;
}
uint8_t sev_abi_v1_network_bytes_at(sev_handle_v1 handle, size_t index) {
    sev_network_bytes *bytes = handle.value;
    return bytes && bytes->kind == SEV_NETWORK_BYTES && index < bytes->length ? bytes->data[index] : 0;
}
void sev_abi_v1_network_bytes_release(sev_handle_v1 handle) {
    sev_network_bytes *bytes = handle.value;
    if (!bytes || bytes->kind != SEV_NETWORK_BYTES) return;
    free(bytes->data);
    free(bytes);
}

static int32_t sev_socket_timeout(sev_handle_v1 connection, int option, uint64_t milliseconds, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!sev_socket_kind(connection, SEV_NETWORK_TCP, &socket)) return sev_fail(error, EBADF, "network connection is closed");
    struct timeval timeout = {
        .tv_sec = (time_t)(milliseconds / 1000),
        .tv_usec = (suseconds_t)((milliseconds % 1000) * 1000),
    };
    if (setsockopt(socket->descriptor, SOL_SOCKET, option, &timeout, sizeof(timeout)) != 0)
        return sev_fail(error, errno, "could not set network timeout");
    sev_clear_error(error);
    return 0;
}
int32_t sev_abi_v1_network_set_read_timeout(sev_handle_v1 connection, uint64_t milliseconds, sev_error_v1 *error) {
    return sev_socket_timeout(connection, SO_RCVTIMEO, milliseconds, error);
}
int32_t sev_abi_v1_network_set_write_timeout(sev_handle_v1 connection, uint64_t milliseconds, sev_error_v1 *error) {
    return sev_socket_timeout(connection, SO_SNDTIMEO, milliseconds, error);
}
int32_t sev_abi_v1_network_set_keep_alive(sev_handle_v1 connection, bool enabled, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!sev_socket_kind(connection, SEV_NETWORK_TCP, &socket)) return sev_fail(error, EBADF, "network connection is closed");
    int value = enabled ? 1 : 0;
    if (setsockopt(socket->descriptor, SOL_SOCKET, SO_KEEPALIVE, &value, sizeof(value)) != 0)
        return sev_fail(error, errno, "could not set TCP keep-alive");
    sev_clear_error(error);
    return 0;
}
int32_t sev_abi_v1_network_shutdown(sev_handle_v1 connection, int32_t direction, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!sev_socket_kind(connection, SEV_NETWORK_TCP, &socket)) return sev_fail(error, EBADF, "network connection is closed");
    int how = direction == 0 ? SHUT_RD : direction == 1 ? SHUT_WR : SHUT_RDWR;
    if (shutdown(socket->descriptor, how) != 0) return sev_fail(error, errno, "could not shut down network connection");
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_resolve(sev_string_view_v1 host, sev_handle_v1 *output, sev_error_v1 *error) {
    if (!output || host.length == 0) return sev_fail(error, EINVAL, "network host cannot be empty");
    char *host_text = sev_copy_view(host);
    if (!host_text) return sev_fail(error, ENOMEM, "could not copy network host");
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *addresses = NULL;
    int status = getaddrinfo(host_text, NULL, &hints, &addresses);
    free(host_text);
    if (status != 0) return sev_fail(error, status, gai_strerror(status));
    sev_network_address_list *list = calloc(1, sizeof(*list));
    if (!list) { freeaddrinfo(addresses); return sev_fail(error, ENOMEM, "could not allocate address list"); }
    list->kind = SEV_NETWORK_ADDRESS_LIST;
    for (struct addrinfo *item = addresses; item; item = item->ai_next) {
        char numeric[NI_MAXHOST];
        if (getnameinfo(item->ai_addr, item->ai_addrlen, numeric, sizeof(numeric), NULL, 0, NI_NUMERICHOST) != 0) continue;
        bool duplicate = false;
        for (size_t index = 0; index < list->length; ++index)
            if (strcmp(list->values[index], numeric) == 0) duplicate = true;
        if (duplicate) continue;
        char **grown = realloc(list->values, (list->length + 1) * sizeof(*grown));
        if (!grown) { status = ENOMEM; break; }
        list->values = grown;
        list->values[list->length] = strdup(numeric);
        if (!list->values[list->length]) { status = ENOMEM; break; }
        list->length += 1;
    }
    freeaddrinfo(addresses);
    if (status == ENOMEM) {
        for (size_t index = 0; index < list->length; ++index) free(list->values[index]);
        free(list->values); free(list);
        return sev_fail(error, ENOMEM, "could not allocate resolved addresses");
    }
    output->value = list;
    sev_clear_error(error);
    return 0;
}

size_t sev_abi_v1_network_address_list_length(sev_handle_v1 handle) {
    sev_network_address_list *list = handle.value;
    return list && list->kind == SEV_NETWORK_ADDRESS_LIST ? list->length : 0;
}
sev_string_view_v1 sev_abi_v1_network_address_list_at(sev_handle_v1 handle, size_t index) {
    sev_network_address_list *list = handle.value;
    return list && list->kind == SEV_NETWORK_ADDRESS_LIST && index < list->length ? sev_view(list->values[index]) : sev_view("");
}
void sev_abi_v1_network_address_list_release(sev_handle_v1 handle) {
    sev_network_address_list *list = handle.value;
    if (!list || list->kind != SEV_NETWORK_ADDRESS_LIST) return;
    for (size_t index = 0; index < list->length; ++index) free(list->values[index]);
    free(list->values); free(list);
}

int32_t sev_abi_v1_network_parse_ip(sev_string_view_v1 value, sev_handle_v1 *output, sev_error_v1 *error) {
    if (!output) return sev_fail(error, EINVAL, "native provider received no output handle");
    char *text = sev_copy_view(value);
    if (!text) return sev_fail(error, ENOMEM, "could not copy IP address");
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_NUMERICHOST;
    struct addrinfo *address = NULL;
    int status = getaddrinfo(text, NULL, &hints, &address);
    free(text);
    if (status != 0) return sev_fail(error, status, "invalid IP address");
    sev_network_address *parsed = calloc(1, sizeof(*parsed));
    if (!parsed) { freeaddrinfo(address); return sev_fail(error, ENOMEM, "could not allocate IP address"); }
    parsed->kind = SEV_NETWORK_ADDRESS;
    parsed->family = address->ai_family == AF_INET6 ? 6 : 4;
    status = getnameinfo(address->ai_addr, address->ai_addrlen, parsed->host, sizeof(parsed->host), NULL, 0, NI_NUMERICHOST);
    freeaddrinfo(address);
    if (status != 0) { free(parsed); return sev_fail(error, status, "invalid IP address"); }
    output->value = parsed;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_udp_bind(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *socket, sev_error_v1 *error) {
    return sev_open_socket(host, port, SOCK_DGRAM, true, SEV_NETWORK_UDP, socket, error);
}

int32_t sev_abi_v1_network_udp_send_to(sev_handle_v1 handle, sev_bytes_view_v1 data, sev_string_view_v1 host, uint16_t port, size_t *sent, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!sent || !sev_socket_kind(handle, SEV_NETWORK_UDP, &socket)) return sev_fail(error, EBADF, "UDP socket is closed");
    char *host_text = sev_copy_view(host);
    if (!host_text) return sev_fail(error, ENOMEM, "could not copy UDP host");
    char service[8]; snprintf(service, sizeof(service), "%u", (unsigned)port);
    struct addrinfo hints; memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_DGRAM; hints.ai_protocol = IPPROTO_UDP;
    struct addrinfo *addresses = NULL;
    int status = getaddrinfo(host_text, service, &hints, &addresses);
    free(host_text);
    if (status != 0) return sev_fail(error, status, gai_strerror(status));
    ssize_t result = -1;
    int saved_error = EIO;
    for (struct addrinfo *address = addresses; address; address = address->ai_next) {
        do result = sendto(socket->descriptor, data.data, data.length, 0, address->ai_addr, address->ai_addrlen); while (result < 0 && errno == EINTR);
        if (result >= 0) break;
        saved_error = errno;
    }
    freeaddrinfo(addresses);
    if (result < 0) return sev_fail(error, saved_error, "UDP send failed");
    *sent = (size_t)result;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_udp_receive_from(sev_handle_v1 handle, size_t count, sev_handle_v1 *output, sev_error_v1 *error) {
    sev_network_socket *socket = NULL;
    if (!output || !sev_socket_kind(handle, SEV_NETWORK_UDP, &socket)) return sev_fail(error, EBADF, "UDP socket is closed");
    if (count > (size_t)SSIZE_MAX) return sev_fail(error, EINVAL, "UDP receive is too large");
    sev_network_packet *packet = calloc(1, sizeof(*packet));
    if (!packet) return sev_fail(error, ENOMEM, "could not allocate UDP packet");
    packet->kind = SEV_NETWORK_PACKET;
    packet->data = malloc(count ? count : 1);
    if (!packet->data) { free(packet); return sev_fail(error, ENOMEM, "could not allocate UDP buffer"); }
    struct sockaddr_storage sender; socklen_t sender_size = sizeof(sender);
    ssize_t received;
    do received = recvfrom(socket->descriptor, packet->data, count, 0, (struct sockaddr *)&sender, &sender_size); while (received < 0 && errno == EINTR);
    if (received < 0) { int code = errno; free(packet->data); free(packet); return sev_fail(error, code, "UDP receive failed"); }
    char service[NI_MAXSERV];
    int status = getnameinfo((struct sockaddr *)&sender, sender_size, packet->host, sizeof(packet->host), service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV);
    if (status != 0) { free(packet->data); free(packet); return sev_fail(error, status, "could not inspect UDP sender"); }
    packet->length = (size_t)received;
    packet->port = (uint16_t)strtoul(service, NULL, 10);
    output->value = packet;
    sev_clear_error(error);
    return 0;
}

int32_t sev_abi_v1_network_udp_close(sev_handle_v1 socket, sev_error_v1 *error) {
    return sev_close_socket(socket, SEV_NETWORK_UDP, error);
}
size_t sev_abi_v1_network_packet_length(sev_handle_v1 handle) {
    sev_network_packet *packet = handle.value;
    return packet && packet->kind == SEV_NETWORK_PACKET ? packet->length : 0;
}
uint8_t sev_abi_v1_network_packet_at(sev_handle_v1 handle, size_t index) {
    sev_network_packet *packet = handle.value;
    return packet && packet->kind == SEV_NETWORK_PACKET && index < packet->length ? packet->data[index] : 0;
}
sev_string_view_v1 sev_abi_v1_network_packet_host(sev_handle_v1 handle) {
    sev_network_packet *packet = handle.value;
    return packet && packet->kind == SEV_NETWORK_PACKET ? sev_view(packet->host) : sev_view("");
}
uint16_t sev_abi_v1_network_packet_port(sev_handle_v1 handle) {
    sev_network_packet *packet = handle.value;
    return packet && packet->kind == SEV_NETWORK_PACKET ? packet->port : 0;
}
void sev_abi_v1_network_packet_release(sev_handle_v1 handle) {
    sev_network_packet *packet = handle.value;
    if (!packet || packet->kind != SEV_NETWORK_PACKET) return;
    free(packet->data); free(packet);
}

int32_t sev_abi_v1_network_loopback_echo(sev_string_view_v1 message, sev_handle_v1 *output, sev_error_v1 *error) {
    if (!output) return sev_fail(error, EINVAL, "native provider received no output handle");
    int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return sev_fail(error, errno, "could not create loopback server");
    struct sockaddr_in endpoint; memset(&endpoint, 0, sizeof(endpoint));
    endpoint.sin_family = AF_INET; endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(server, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0 || listen(server, 1) != 0) {
        int code = errno; close(server); return sev_fail(error, code, "could not bind loopback server");
    }
    socklen_t endpoint_size = sizeof(endpoint);
    if (getsockname(server, (struct sockaddr *)&endpoint, &endpoint_size) != 0) {
        int code = errno; close(server); return sev_fail(error, code, "could not inspect loopback server");
    }
    int client = socket(AF_INET, SOCK_STREAM, 0);
    if (client < 0 || connect(client, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0) {
        int code = errno; if (client >= 0) close(client); close(server); return sev_fail(error, code, "could not connect loopback client");
    }
    int peer = accept(server, NULL, NULL);
    sev_network_text *text = calloc(1, sizeof(*text));
    if (peer < 0 || !text) {
        int code = peer < 0 ? errno : ENOMEM; if (peer >= 0) close(peer); close(client); close(server); free(text); return sev_fail(error, code, "could not accept loopback connection");
    }
    text->kind = SEV_NETWORK_TEXT; text->length = message.length; text->value = malloc(message.length + 1);
    bool success = text->value != NULL;
    if (success) {
        size_t offset = 0;
        while (offset < message.length) { ssize_t count = send(client, message.data + offset, message.length - offset, 0); if (count <= 0) { success = false; break; } offset += (size_t)count; }
        offset = 0;
        while (success && offset < message.length) { ssize_t count = recv(peer, text->value + offset, message.length - offset, 0); if (count <= 0) { success = false; break; } offset += (size_t)count; }
        offset = 0;
        while (success && offset < message.length) { ssize_t count = send(peer, text->value + offset, message.length - offset, 0); if (count <= 0) { success = false; break; } offset += (size_t)count; }
        offset = 0;
        while (success && offset < message.length) { ssize_t count = recv(client, text->value + offset, message.length - offset, 0); if (count <= 0) { success = false; break; } offset += (size_t)count; }
    }
    close(peer); close(client); close(server);
    if (!success) { free(text->value); free(text); return sev_fail(error, errno, "loopback transfer failed"); }
    text->value[text->length] = '\0';
    output->value = text;
    sev_clear_error(error);
    return 0;
}

sev_string_view_v1 sev_abi_v1_network_text_value(sev_handle_v1 handle) {
    sev_network_text *text = handle.value;
    sev_string_view_v1 empty = { .data = NULL, .length = 0 };
    if (!text || text->kind != SEV_NETWORK_TEXT) return empty;
    sev_string_view_v1 result = { .data = (const uint8_t *)text->value, .length = text->length };
    return result;
}
void sev_abi_v1_network_text_release(sev_handle_v1 handle) {
    sev_network_text *text = handle.value;
    if (!text || text->kind != SEV_NETWORK_TEXT) return;
    free(text->value); free(text);
}
