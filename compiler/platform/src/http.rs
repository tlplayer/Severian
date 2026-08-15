pub fn source() -> &'static str {
    r#"
#include <strings.h>

int32_t sev_abi_v1_network_connect(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *connection, sev_error_v1 *error);
int32_t sev_abi_v1_network_read(sev_handle_v1 connection, size_t count, sev_handle_v1 *bytes, sev_error_v1 *error);
int32_t sev_abi_v1_network_write(sev_handle_v1 connection, sev_bytes_view_v1 data, size_t *written, sev_error_v1 *error);
int32_t sev_abi_v1_network_close(sev_handle_v1 connection, sev_error_v1 *error);
int32_t sev_abi_v1_network_set_read_timeout(sev_handle_v1 connection, uint64_t milliseconds, sev_error_v1 *error);
int32_t sev_abi_v1_network_set_write_timeout(sev_handle_v1 connection, uint64_t milliseconds, sev_error_v1 *error);
size_t sev_abi_v1_network_bytes_length(sev_handle_v1 bytes);
uint8_t sev_abi_v1_network_bytes_at(sev_handle_v1 bytes, size_t index);
void sev_abi_v1_network_bytes_release(sev_handle_v1 bytes);

typedef struct { bool secure; sev_handle_v1 tcp; sev_tls_handle *tls; } sev_http_stream;
typedef struct { int status; int64_t content_length; bool chunked; char location[4096]; } sev_http_head;
typedef struct { bool secure; char host[256]; int64_t port; char path[4096]; } sev_http_url;

static sev_string_view_v1 sev_http_view(const char *value) {
  sev_string_view_v1 view = { .data = (const uint8_t *)value, .length = value ? strlen(value) : 0 }; return view;
}

static ssize_t sev_http_stream_read(sev_http_stream *stream, unsigned char *buffer, size_t size) {
  if (stream->secure) {
    int received = SSL_read(stream->tls->stream, buffer, size > INT_MAX ? INT_MAX : (int)size);
    if (received <= 0 && SSL_get_error(stream->tls->stream, received) == SSL_ERROR_ZERO_RETURN) return 0;
    return received;
  }
  sev_handle_v1 bytes = {0}; sev_error_v1 error = {0};
  if (sev_abi_v1_network_read(stream->tcp, size, &bytes, &error) != 0) return -1;
  size_t received = sev_abi_v1_network_bytes_length(bytes);
  for (size_t index = 0; index < received; ++index) buffer[index] = sev_abi_v1_network_bytes_at(bytes, index);
  sev_abi_v1_network_bytes_release(bytes); return (ssize_t)received;
}

static bool sev_http_stream_write_all(sev_http_stream *stream, const unsigned char *buffer, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    ssize_t written;
    if (stream->secure) written = SSL_write(stream->tls->stream, buffer + offset, size - offset > INT_MAX ? INT_MAX : (int)(size - offset));
    else { size_t count = 0; sev_error_v1 error = {0}; sev_bytes_view_v1 data = { .data = buffer + offset, .length = size - offset }; written = sev_abi_v1_network_write(stream->tcp, data, &count, &error) == 0 ? (ssize_t)count : -1; }
    if (written <= 0) return false;
    offset += (size_t)written;
  }
  return true;
}

static void sev_http_stream_close(sev_http_stream *stream) {
  if (stream->secure && stream->tls && !stream->tls->closed) {
    SSL_shutdown(stream->tls->stream); SSL_free(stream->tls->stream); SSL_CTX_free(stream->tls->context);
    stream->tls->closed = true;
  }
  if (stream->tcp.value) { sev_error_v1 error = {0}; sev_abi_v1_network_close(stream->tcp, &error); stream->tcp.value = NULL; }
}

static bool sev_http_parse_url(const char *url, sev_http_url *parsed) {
  memset(parsed, 0, sizeof(*parsed)); const char *authority;
  if (strncmp(url, "https://", 8) == 0) { parsed->secure = true; parsed->port = 443; authority = url + 8; }
  else if (strncmp(url, "http://", 7) == 0) { parsed->port = 80; authority = url + 7; }
  else return false;
  const char *path = strchr(authority, '/'); const char *end = path ? path : authority + strlen(authority);
  const char *port_start = NULL;
  if (authority < end && *authority == '[') {
    const char *closing = memchr(authority, ']', (size_t)(end - authority)); if (!closing) return false;
    size_t host_size = (size_t)(closing - authority - 1); if (!host_size || host_size >= sizeof(parsed->host)) return false;
    memcpy(parsed->host, authority + 1, host_size); parsed->host[host_size] = '\0';
    if (closing + 1 < end) { if (closing[1] != ':') return false; port_start = closing + 2; }
  } else {
    const char *colon = memchr(authority, ':', (size_t)(end - authority)); const char *host_end = colon ? colon : end;
    size_t host_size = (size_t)(host_end - authority); if (!host_size || host_size >= sizeof(parsed->host)) return false;
    memcpy(parsed->host, authority, host_size); parsed->host[host_size] = '\0'; if (colon) port_start = colon + 1;
  }
  if (port_start) { char port_text[8]; size_t port_size = (size_t)(end - port_start); if (!port_size || port_size >= sizeof(port_text)) return false; memcpy(port_text, port_start, port_size); port_text[port_size] = '\0'; char *tail = NULL; long port = strtol(port_text, &tail, 10); if (*tail || port < 1 || port > 65535) return false; parsed->port = port; }
  const char *request_path = path ? path : "/"; if (strlen(request_path) >= sizeof(parsed->path)) return false; strcpy(parsed->path, request_path);
  return true;
}

static bool sev_http_resolve_redirect(const sev_http_url *base, const char *location, char *output, size_t capacity) {
  if (strncmp(location, "https://", 8) == 0 || strncmp(location, "http://", 7) == 0) return snprintf(output, capacity, "%s", location) > 0;
  if (*location != '/') return false;
  bool default_port = (base->secure && base->port == 443) || (!base->secure && base->port == 80);
  int written = snprintf(output, capacity, "%s://%s%s%lld%s", base->secure ? "https" : "http", base->host, default_port ? "" : ":", default_port ? 0LL : (long long)base->port, location);
  if (default_port) written = snprintf(output, capacity, "%s://%s%s", base->secure ? "https" : "http", base->host, location);
  return written > 0 && (size_t)written < capacity;
}

static bool sev_http_read_exact(sev_http_stream *stream, unsigned char *buffer, size_t size) {
  size_t offset = 0; while (offset < size) { ssize_t received = sev_http_stream_read(stream, buffer + offset, size - offset); if (received <= 0) return false; offset += (size_t)received; } return true;
}

static bool sev_http_read_line(sev_http_stream *stream, char *line, size_t capacity) {
  size_t used = 0; bool carriage = false;
  while (used + 1 < capacity) {
    unsigned char byte; if (!sev_http_read_exact(stream, &byte, 1)) return false;
    if (carriage && byte == '\n') { line[used] = '\0'; return true; }
    if (carriage) line[used++] = '\r';
    carriage = byte == '\r'; if (!carriage) line[used++] = (char)byte;
  }
  return false;
}

static bool sev_http_open(const char *method, const char *url, const char *body, sev_http_stream *stream, sev_http_url *parsed, sev_http_head *head) {
  if (!sev_http_parse_url(url, parsed)) return false;
  memset(stream, 0, sizeof(*stream)); memset(head, 0, sizeof(*head)); head->content_length = -1;
  sev_error_v1 network_error = {0};
  if (sev_abi_v1_network_connect(sev_http_view(parsed->host), (uint16_t)parsed->port, &stream->tcp, &network_error) != 0) return false;
  sev_abi_v1_network_set_read_timeout(stream->tcp, 30000, &network_error);
  sev_abi_v1_network_set_write_timeout(stream->tcp, 30000, &network_error);
  stream->secure = parsed->secure;
  if (stream->secure) { stream->tls = sev_tls_wrap_native(stream->tcp, parsed->host); if (!stream->tls) { sev_http_stream_close(stream); return false; } }
  size_t body_size = body ? strlen(body) : 0; char request[8192];
  bool default_port = (parsed->secure && parsed->port == 443) || (!parsed->secure && parsed->port == 80);
  int request_size;
  if (default_port && body_size) request_size = snprintf(request, sizeof(request), "%s %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: Severian/0.1\r\nAccept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\n\r\n", method, parsed->path, parsed->host, body_size);
  else if (default_port) request_size = snprintf(request, sizeof(request), "%s %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: Severian/0.1\r\nAccept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n", method, parsed->path, parsed->host);
  else if (body_size) request_size = snprintf(request, sizeof(request), "%s %s HTTP/1.1\r\nHost: %s:%lld\r\nUser-Agent: Severian/0.1\r\nAccept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\n\r\n", method, parsed->path, parsed->host, (long long)parsed->port, body_size);
  else request_size = snprintf(request, sizeof(request), "%s %s HTTP/1.1\r\nHost: %s:%lld\r\nUser-Agent: Severian/0.1\r\nAccept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n", method, parsed->path, parsed->host, (long long)parsed->port);
  if (request_size <= 0 || (size_t)request_size >= sizeof(request) || !sev_http_stream_write_all(stream, (unsigned char *)request, (size_t)request_size) || (body_size && !sev_http_stream_write_all(stream, (unsigned char *)body, body_size))) { sev_http_stream_close(stream); return false; }
  char line[16384]; if (!sev_http_read_line(stream, line, sizeof(line)) || sscanf(line, "HTTP/%*u.%*u %d", &head->status) != 1) { sev_http_stream_close(stream); return false; }
  while (sev_http_read_line(stream, line, sizeof(line))) {
    if (!*line) return true;
    char *colon = strchr(line, ':'); if (!colon) continue; *colon = '\0'; char *value = colon + 1; while (*value && isspace((unsigned char)*value)) ++value;
    if (strcasecmp(line, "Content-Length") == 0) { char *tail = NULL; long long length = strtoll(value, &tail, 10); if (tail != value && length >= 0) head->content_length = length; }
    else if (strcasecmp(line, "Transfer-Encoding") == 0 && strcasestr(value, "chunked")) head->chunked = true;
    else if (strcasecmp(line, "Location") == 0) snprintf(head->location, sizeof(head->location), "%s", value);
  }
  sev_http_stream_close(stream); return false;
}

typedef bool (*sev_http_sink)(void *, const unsigned char *, size_t);
static bool sev_http_transfer_body(sev_http_stream *stream, const sev_http_head *head, sev_http_sink sink, void *context) {
  unsigned char buffer[65536];
  if (head->chunked) {
    char line[128];
    while (sev_http_read_line(stream, line, sizeof(line))) {
      char *extension = strchr(line, ';'); if (extension) *extension = '\0'; char *tail = NULL; unsigned long long chunk = strtoull(line, &tail, 16); if (tail == line || *tail) return false;
      if (chunk == 0) { while (sev_http_read_line(stream, line, sizeof(line)) && *line) {} return true; }
      uint64_t remaining = chunk;
      while (remaining) { size_t requested = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer); if (!sev_http_read_exact(stream, buffer, requested) || !sink(context, buffer, requested)) return false; remaining -= requested; }
      unsigned char ending[2]; if (!sev_http_read_exact(stream, ending, 2) || ending[0] != '\r' || ending[1] != '\n') return false;
    }
    return false;
  }
  if (head->content_length >= 0) {
    int64_t remaining = head->content_length;
    while (remaining) { size_t requested = remaining < (int64_t)sizeof(buffer) ? (size_t)remaining : sizeof(buffer); if (!sev_http_read_exact(stream, buffer, requested) || !sink(context, buffer, requested)) return false; remaining -= (int64_t)requested; }
    return true;
  }
  while (true) { ssize_t received = sev_http_stream_read(stream, buffer, sizeof(buffer)); if (received == 0) return true; if (received < 0 || !sink(context, buffer, (size_t)received)) return false; }
}

typedef struct { unsigned char *data; size_t size; size_t capacity; } sev_http_memory;
static bool sev_http_memory_sink(void *raw, const unsigned char *buffer, size_t size) {
  sev_http_memory *memory = raw; if (size > SIZE_MAX - memory->size - 1) return false; size_t required = memory->size + size + 1;
  if (required > memory->capacity) { size_t capacity = memory->capacity ? memory->capacity : 8192; while (capacity < required) { if (capacity > SIZE_MAX / 2) return false; capacity *= 2; } memory->data = realloc(memory->data, capacity); memory->capacity = capacity; }
  memcpy(memory->data + memory->size, buffer, size); memory->size += size; memory->data[memory->size] = '\0'; return true;
}
static bool sev_http_file_sink(void *raw, const unsigned char *buffer, size_t size) { return fwrite(buffer, 1, size, raw) == size; }

static bool sev_http_follow(const char *method, const char *initial_url, const char *body, sev_http_sink sink, void *context, int *final_status) {
  char url[8192]; if (snprintf(url, sizeof(url), "%s", initial_url) <= 0) return false;
  for (int redirects = 0; redirects <= 10; ++redirects) {
    sev_http_stream stream; sev_http_url parsed; sev_http_head head;
    if (!sev_http_open(method, url, body, &stream, &parsed, &head)) return false;
    if (head.status >= 300 && head.status < 400 && *head.location) {
      char next[8192]; bool resolved = sev_http_resolve_redirect(&parsed, head.location, next, sizeof(next)); sev_http_url redirected;
      if (resolved) resolved = sev_http_parse_url(next, &redirected) && (!parsed.secure || redirected.secure);
      sev_http_stream_close(&stream); if (!resolved) return false; strcpy(url, next); if (head.status == 303) { method = "GET"; body = ""; } continue;
    }
    *final_status = head.status;
    bool success = head.status >= 200 && head.status < 300 && sev_http_transfer_body(&stream, &head, sink, context); sev_http_stream_close(&stream); return success;
  }
  return false;
}

void *__sev_http_request(void *method_raw, void *url_raw, void *body_raw) {
  sev_http_memory memory = {0}; int status = 0;
  if (!sev_http_follow(method_raw, url_raw, body_raw, sev_http_memory_sink, &memory, &status)) { free(memory.data); return sev_failure("HTTP request failed, was rejected, or exceeded the redirect limit"); }
  if (!memory.data) { memory.data = sev_allocate(1); memory.data[0] = '\0'; }
  return __sev_variant_new("ok", __sev_box_string(memory.data));
}

void *__sev_http_download(void *url_raw, void *destination_raw) {
  const char *destination = destination_raw; if (!destination || !*destination) return sev_failure("HTTP download destination is empty");
  size_t destination_size = strlen(destination); if (destination_size > SIZE_MAX - 10) return sev_failure("HTTP download destination is too long");
  char *temporary = sev_allocate(destination_size + 10); snprintf(temporary, destination_size + 10, "%s.download", destination);
  FILE *file = fopen(temporary, "wb"); if (!file) { free(temporary); return sev_failure("could not open HTTP download destination"); }
  int status = 0; bool success = sev_http_follow("GET", url_raw, "", sev_http_file_sink, file, &status);
  if (fclose(file) != 0) success = false;
  if (success && rename(temporary, destination) != 0) success = false;
  if (!success) { unlink(temporary); free(temporary); return sev_failure("HTTP download failed, was rejected, or exceeded the redirect limit"); }
  free(temporary);
  return __sev_variant_new("ok", NULL);
}
"#
}

#[cfg(test)]
mod tests {
    use super::source;

    #[test]
    fn native_http_is_streaming_and_has_no_process_fallback() {
        let source = source();
        assert!(source.contains("unsigned char buffer[65536]"));
        assert!(source.contains("Transfer-Encoding"));
        assert!(source.contains("head.status >= 300"));
        assert!(source.contains("sev_tls_wrap_native"));
        assert!(!source.contains("curl"));
        assert!(!source.contains("fork("));
        assert!(!source.contains("execl"));
    }
}
