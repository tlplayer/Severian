pub fn source() -> &'static str {
    r#"
#include <netdb.h>
#include <netinet/tcp.h>

typedef struct { int socket; bool closed; } sev_tcp_handle;
typedef struct { int socket; bool closed; } sev_tcp_listener;
typedef struct { int socket; bool closed; } sev_udp_handle;

static void *sev_bytes_from_buffer(const unsigned char *buffer, size_t size) {
  sev_collection *result = __sev_collection_new(0);
  for (size_t index = 0; index < size; ++index)
    __sev_collection_push(result, __sev_box_i64(buffer[index]));
  return __sev_box_collection(result);
}

static bool sev_buffer_from_bytes(void *raw, unsigned char **buffer, size_t *size) {
  sev_collection *bytes = raw;
  if (!bytes || bytes->size < 0 || (uint64_t)bytes->size > SIZE_MAX) return false;
  *size = (size_t)bytes->size;
  *buffer = sev_allocate(*size ? *size : 1);
  for (int64_t index = 0; index < bytes->size; ++index) {
    int64_t byte = __sev_unbox_i64(bytes->items[index]);
    if (byte < 0 || byte > 255) { free(*buffer); return false; }
    (*buffer)[index] = (unsigned char)byte;
  }
  return true;
}

static sev_tcp_handle *sev_tcp_connect_native(const char *host, int64_t port) {
  if (!host || !*host || port < 0 || port > 65535) return NULL;
  char service[8]; snprintf(service, sizeof(service), "%lld", (long long)port);
  struct addrinfo hints = {0}, *addresses = NULL;
  hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_protocol = IPPROTO_TCP;
  if (getaddrinfo(host, service, &hints, &addresses) != 0) return NULL;
  int descriptor = -1;
  for (struct addrinfo *address = addresses; address; address = address->ai_next) {
    descriptor = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (descriptor >= 0 && connect(descriptor, address->ai_addr, address->ai_addrlen) == 0) break;
    if (descriptor >= 0) close(descriptor);
    descriptor = -1;
  }
  freeaddrinfo(addresses);
  if (descriptor < 0) return NULL;
  sev_tcp_handle *handle = sev_allocate(sizeof(*handle));
  handle->socket = descriptor;
  return handle;
}

void *__sev_network_connect(void *host_raw, int64_t port) {
  sev_tcp_handle *handle = sev_tcp_connect_native(host_raw, port);
  return handle ? __sev_variant_new("ok", handle) : sev_failure("could not resolve or connect to network host");
}

void *__sev_network_listen(void *host_raw, int64_t port) {
  const char *host = host_raw;
  if (port < 0 || port > 65535) return sev_failure("invalid network port");
  char service[8]; snprintf(service, sizeof(service), "%lld", (long long)port);
  struct addrinfo hints = {0}, *addresses = NULL;
  hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_PASSIVE;
  if (getaddrinfo(host && *host ? host : NULL, service, &hints, &addresses) != 0)
    return sev_failure("could not resolve listener address");
  int descriptor = -1;
  for (struct addrinfo *address = addresses; address; address = address->ai_next) {
    descriptor = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (descriptor < 0) continue;
    int reuse = 1; setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (bind(descriptor, address->ai_addr, address->ai_addrlen) == 0 && listen(descriptor, 128) == 0) break;
    close(descriptor); descriptor = -1;
  }
  freeaddrinfo(addresses);
  if (descriptor < 0) return sev_failure("could not bind network listener");
  sev_tcp_listener *listener = sev_allocate(sizeof(*listener));
  listener->socket = descriptor;
  return __sev_variant_new("ok", listener);
}

void *__sev_network_accept(void *listener_raw) {
  sev_tcp_listener *listener = listener_raw;
  if (!listener || listener->closed) return sev_failure("network listener is closed");
  int descriptor = accept(listener->socket, NULL, NULL);
  if (descriptor < 0) return sev_failure("could not accept network connection");
  sev_tcp_handle *handle = sev_allocate(sizeof(*handle)); handle->socket = descriptor;
  return __sev_variant_new("ok", handle);
}

void *__sev_network_read(void *handle_raw, int64_t count) {
  sev_tcp_handle *handle = handle_raw;
  if (!handle || handle->closed || count < 0 || (uint64_t)count > SIZE_MAX)
    return sev_failure("invalid network read");
  unsigned char *buffer = sev_allocate(count > 0 ? (size_t)count : 1);
  ssize_t received;
  do received = recv(handle->socket, buffer, (size_t)count, 0); while (received < 0 && errno == EINTR);
  if (received < 0) { free(buffer); return sev_failure("network read failed"); }
  void *result = sev_bytes_from_buffer(buffer, (size_t)received); free(buffer);
  return __sev_variant_new("ok", result);
}

void *__sev_network_write(void *handle_raw, void *bytes_raw) {
  sev_tcp_handle *handle = handle_raw; unsigned char *buffer = NULL; size_t size = 0;
  if (!handle || handle->closed || !sev_buffer_from_bytes(bytes_raw, &buffer, &size))
    return sev_failure("invalid network write");
  size_t offset = 0;
  while (offset < size) {
    ssize_t written = send(handle->socket, buffer + offset, size - offset, 0);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) { free(buffer); return sev_failure("network write failed"); }
    offset += (size_t)written;
  }
  free(buffer);
  return __sev_variant_new("ok", __sev_box_i64((int64_t)size));
}

static void *sev_close_descriptor(int descriptor, bool *closed) {
  if (!closed || *closed) return sev_failure("network handle is closed");
  if (close(descriptor) != 0) return sev_failure("could not close network handle");
  *closed = true; return __sev_variant_new("ok", NULL);
}

void *__sev_network_close(void *handle_raw) {
  sev_tcp_handle *handle = handle_raw;
  return handle ? sev_close_descriptor(handle->socket, &handle->closed) : sev_failure("invalid network connection");
}
void *__sev_network_listener_close(void *listener_raw) {
  sev_tcp_listener *listener = listener_raw;
  return listener ? sev_close_descriptor(listener->socket, &listener->closed) : sev_failure("invalid network listener");
}

static void *sev_socket_host(int descriptor, bool peer) {
  struct sockaddr_storage address; socklen_t size = sizeof(address); char host[NI_MAXHOST];
  int status = peer ? getpeername(descriptor, (struct sockaddr *)&address, &size) : getsockname(descriptor, (struct sockaddr *)&address, &size);
  if (status != 0 || getnameinfo((struct sockaddr *)&address, size, host, sizeof(host), NULL, 0, NI_NUMERICHOST) != 0)
    return sev_failure("could not inspect network address");
  return __sev_variant_new("ok", __sev_box_string(strdup(host)));
}
static void *sev_socket_port(int descriptor, bool peer) {
  struct sockaddr_storage address; socklen_t size = sizeof(address);
  int status = peer ? getpeername(descriptor, (struct sockaddr *)&address, &size) : getsockname(descriptor, (struct sockaddr *)&address, &size);
  if (status != 0) return sev_failure("could not inspect network port");
  int64_t port = address.ss_family == AF_INET ? ntohs(((struct sockaddr_in *)&address)->sin_port) : ntohs(((struct sockaddr_in6 *)&address)->sin6_port);
  return __sev_variant_new("ok", __sev_box_i64(port));
}
void *__sev_network_local_host(void *raw) { sev_tcp_handle *handle = raw; return handle && !handle->closed ? sev_socket_host(handle->socket, false) : sev_failure("network connection is closed"); }
void *__sev_network_local_port(void *raw) { sev_tcp_handle *handle = raw; return handle && !handle->closed ? sev_socket_port(handle->socket, false) : sev_failure("network connection is closed"); }
void *__sev_network_remote_host(void *raw) { sev_tcp_handle *handle = raw; return handle && !handle->closed ? sev_socket_host(handle->socket, true) : sev_failure("network connection is closed"); }
void *__sev_network_remote_port(void *raw) { sev_tcp_handle *handle = raw; return handle && !handle->closed ? sev_socket_port(handle->socket, true) : sev_failure("network connection is closed"); }
void *__sev_network_listener_host(void *raw) { sev_tcp_listener *listener = raw; return listener && !listener->closed ? sev_socket_host(listener->socket, false) : sev_failure("network listener is closed"); }
void *__sev_network_listener_port(void *raw) { sev_tcp_listener *listener = raw; return listener && !listener->closed ? sev_socket_port(listener->socket, false) : sev_failure("network listener is closed"); }

static void *sev_socket_timeout(sev_tcp_handle *handle, int option, int64_t milliseconds) {
  if (!handle || handle->closed || milliseconds < 0) return sev_failure("invalid network timeout");
  struct timeval value = { .tv_sec = milliseconds / 1000, .tv_usec = (milliseconds % 1000) * 1000 };
  return setsockopt(handle->socket, SOL_SOCKET, option, &value, sizeof(value)) == 0 ? __sev_variant_new("ok", NULL) : sev_failure("could not set network timeout");
}
void *__sev_network_set_read_timeout(void *raw, int64_t milliseconds) { return sev_socket_timeout(raw, SO_RCVTIMEO, milliseconds); }
void *__sev_network_set_write_timeout(void *raw, int64_t milliseconds) { return sev_socket_timeout(raw, SO_SNDTIMEO, milliseconds); }
void *__sev_network_set_keep_alive(void *raw, bool enabled) { sev_tcp_handle *handle = raw; int value = enabled; return handle && !handle->closed && setsockopt(handle->socket, SOL_SOCKET, SO_KEEPALIVE, &value, sizeof(value)) == 0 ? __sev_variant_new("ok", NULL) : sev_failure("could not set TCP keep-alive"); }
void *__sev_network_shutdown(void *raw, int64_t direction) { sev_tcp_handle *handle = raw; int how = direction == 0 ? SHUT_RD : direction == 1 ? SHUT_WR : SHUT_RDWR; return handle && !handle->closed && shutdown(handle->socket, how) == 0 ? __sev_variant_new("ok", NULL) : sev_failure("could not shut down network connection"); }

void *__sev_network_resolve(void *host_raw) {
  const char *host = host_raw; struct addrinfo hints = {0}, *addresses = NULL;
  hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
  if (!host || !*host || getaddrinfo(host, NULL, &hints, &addresses) != 0) return sev_failure("could not resolve network host");
  sev_collection *result = __sev_collection_new(0);
  for (struct addrinfo *address = addresses; address; address = address->ai_next) {
    char numeric[NI_MAXHOST];
    if (getnameinfo(address->ai_addr, address->ai_addrlen, numeric, sizeof(numeric), NULL, 0, NI_NUMERICHOST) != 0) continue;
    bool duplicate = false;
    for (int64_t index = 0; index < result->size; ++index)
      if (strcmp(result->items[index]->as.string, numeric) == 0) duplicate = true;
    if (!duplicate) __sev_collection_push(result, __sev_box_string(strdup(numeric)));
  }
  freeaddrinfo(addresses);
  return __sev_variant_new("ok", __sev_box_collection(result));
}

void *__sev_network_parse_ip(void *value_raw) {
  const char *value = value_raw; struct addrinfo hints = {0}, *address = NULL;
  hints.ai_family = AF_UNSPEC; hints.ai_flags = AI_NUMERICHOST;
  if (!value || getaddrinfo(value, NULL, &hints, &address) != 0) return sev_failure("invalid IP address");
  char numeric[NI_MAXHOST]; int status = getnameinfo(address->ai_addr, address->ai_addrlen, numeric, sizeof(numeric), NULL, 0, NI_NUMERICHOST);
  int family = address->ai_family; freeaddrinfo(address);
  if (status != 0) return sev_failure("invalid IP address");
  sev_collection *pair = __sev_collection_new(1);
  __sev_collection_push(pair, __sev_box_string(strdup(numeric)));
  __sev_collection_push(pair, __sev_box_i64(family == AF_INET6 ? 6 : 4));
  return __sev_variant_new("ok", __sev_box_collection(pair));
}

void *__sev_udp_bind(void *host_raw, int64_t port) {
  const char *host = host_raw; char service[8]; snprintf(service, sizeof(service), "%lld", (long long)port);
  struct addrinfo hints = {0}, *addresses = NULL; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_DGRAM; hints.ai_flags = AI_PASSIVE;
  if (port < 0 || port > 65535 || getaddrinfo(host && *host ? host : NULL, service, &hints, &addresses) != 0) return sev_failure("could not resolve UDP bind address");
  int descriptor = -1;
  for (struct addrinfo *address = addresses; address; address = address->ai_next) {
    descriptor = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (descriptor >= 0 && bind(descriptor, address->ai_addr, address->ai_addrlen) == 0) break;
    if (descriptor >= 0) close(descriptor); descriptor = -1;
  }
  freeaddrinfo(addresses); if (descriptor < 0) return sev_failure("could not bind UDP socket");
  sev_udp_handle *handle = sev_allocate(sizeof(*handle)); handle->socket = descriptor;
  return __sev_variant_new("ok", handle);
}

void *__sev_udp_send_to(void *handle_raw, void *bytes_raw, void *host_raw, int64_t port) {
  sev_udp_handle *handle = handle_raw; unsigned char *buffer = NULL; size_t size = 0; char service[8];
  if (!handle || handle->closed || port < 0 || port > 65535 || !sev_buffer_from_bytes(bytes_raw, &buffer, &size)) return sev_failure("invalid UDP send");
  snprintf(service, sizeof(service), "%lld", (long long)port); struct addrinfo hints = {0}, *addresses = NULL; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_DGRAM;
  if (getaddrinfo(host_raw, service, &hints, &addresses) != 0) { free(buffer); return sev_failure("could not resolve UDP destination"); }
  ssize_t sent = -1;
  for (struct addrinfo *address = addresses; address; address = address->ai_next) { sent = sendto(handle->socket, buffer, size, 0, address->ai_addr, address->ai_addrlen); if (sent >= 0) break; }
  freeaddrinfo(addresses); free(buffer);
  return sent < 0 ? sev_failure("UDP send failed") : __sev_variant_new("ok", __sev_box_i64(sent));
}

void *__sev_udp_receive_from(void *handle_raw, int64_t count) {
  sev_udp_handle *handle = handle_raw; if (!handle || handle->closed || count < 0 || (uint64_t)count > SIZE_MAX) return sev_failure("invalid UDP receive");
  unsigned char *buffer = sev_allocate(count > 0 ? (size_t)count : 1); struct sockaddr_storage address; socklen_t address_size = sizeof(address);
  ssize_t received = recvfrom(handle->socket, buffer, (size_t)count, 0, (struct sockaddr *)&address, &address_size);
  if (received < 0) { free(buffer); return sev_failure("UDP receive failed"); }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  if (getnameinfo((struct sockaddr *)&address, address_size, host, sizeof(host), service, sizeof(service), NI_NUMERICHOST | NI_NUMERICSERV) != 0) { free(buffer); return sev_failure("could not inspect UDP sender"); }
  sev_collection *triple = __sev_collection_new(1);
  __sev_collection_push(triple, sev_bytes_from_buffer(buffer, (size_t)received));
  __sev_collection_push(triple, __sev_box_string(strdup(host)));
  __sev_collection_push(triple, __sev_box_i64(strtoll(service, NULL, 10)));
  free(buffer); return __sev_variant_new("ok", __sev_box_collection(triple));
}
void *__sev_udp_close(void *raw) { sev_udp_handle *handle = raw; return handle ? sev_close_descriptor(handle->socket, &handle->closed) : sev_failure("invalid UDP socket"); }

void *__sev_network_loopback_echo(void *message_raw) {
  const char *message = message_raw; int server = socket(AF_INET, SOCK_STREAM, 0); if (server < 0) return sev_failure("could not create loopback server");
  struct sockaddr_in endpoint = {0}; endpoint.sin_family = AF_INET; endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(server, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0 || listen(server, 1) != 0) { close(server); return sev_failure("could not bind loopback server"); }
  socklen_t endpoint_size = sizeof(endpoint); getsockname(server, (struct sockaddr *)&endpoint, &endpoint_size);
  int client = socket(AF_INET, SOCK_STREAM, 0); if (client < 0 || connect(client, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0) { if (client >= 0) close(client); close(server); return sev_failure("could not connect loopback client"); }
  int peer = accept(server, NULL, NULL); size_t size = strlen(message); char *buffer = sev_allocate(size + 1); bool success = peer >= 0;
  if (success) success = send(client, message, size, 0) == (ssize_t)size && recv(peer, buffer, size, MSG_WAITALL) == (ssize_t)size && send(peer, buffer, size, 0) == (ssize_t)size && recv(client, buffer, size, MSG_WAITALL) == (ssize_t)size;
  if (peer >= 0) close(peer); close(client); close(server); if (!success) return sev_failure("loopback transfer failed"); buffer[size] = '\0';
  return __sev_variant_new("ok", __sev_box_string(buffer));
}
"#
}
