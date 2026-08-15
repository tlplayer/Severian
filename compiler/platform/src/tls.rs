pub fn source() -> &'static str {
    r#"
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509v3.h>

int32_t sev_abi_v1_network_descriptor(sev_handle_v1 connection);
int32_t sev_abi_v1_network_close(sev_handle_v1 connection, sev_error_v1 *error);

typedef struct { SSL_CTX *context; SSL *stream; sev_handle_v1 transport; bool closed; } sev_tls_handle;

static void *sev_tls_bytes_from_buffer(const unsigned char *buffer, size_t size) {
  sev_collection *result = __sev_collection_new(0);
  for (size_t index = 0; index < size; ++index) __sev_collection_push(result, __sev_box_i64(buffer[index]));
  return __sev_box_collection(result);
}

static bool sev_tls_buffer_from_bytes(void *raw, unsigned char **buffer, size_t *size) {
  sev_collection *bytes = raw;
  if (!bytes || bytes->size < 0 || (uint64_t)bytes->size > SIZE_MAX) return false;
  *size = (size_t)bytes->size; *buffer = sev_allocate(*size ? *size : 1);
  for (int64_t index = 0; index < bytes->size; ++index) {
    int64_t byte = __sev_unbox_i64(bytes->items[index]);
    if (byte < 0 || byte > 255) { free(*buffer); return false; }
    (*buffer)[index] = (unsigned char)byte;
  }
  return true;
}

static sev_tls_handle *sev_tls_wrap_native(sev_handle_v1 transport, const char *server_name) {
  int32_t descriptor = sev_abi_v1_network_descriptor(transport);
  if (descriptor < 0 || !server_name || !*server_name) return NULL;
  SSL_CTX *context = SSL_CTX_new(TLS_client_method()); if (!context) return NULL;
  SSL_CTX_set_min_proto_version(context, TLS1_2_VERSION);
  SSL_CTX_set_verify(context, SSL_VERIFY_PEER, NULL);
  if (SSL_CTX_set_default_verify_paths(context) != 1) { SSL_CTX_free(context); return NULL; }
  SSL *stream = SSL_new(context); if (!stream) { SSL_CTX_free(context); return NULL; }
  if (SSL_set_tlsext_host_name(stream, server_name) != 1 || X509_VERIFY_PARAM_set1_host(SSL_get0_param(stream), server_name, 0) != 1 || SSL_set_fd(stream, descriptor) != 1 || SSL_connect(stream) != 1 || SSL_get_verify_result(stream) != X509_V_OK) {
    SSL_free(stream); SSL_CTX_free(context); return NULL;
  }
  sev_tls_handle *handle = sev_allocate(sizeof(*handle)); handle->context = context; handle->stream = stream; handle->transport = transport;
  return handle;
}

void *__sev_tls_connect(void *transport_raw, void *server_name_raw) {
  sev_handle_v1 transport = { .value = transport_raw };
  sev_tls_handle *handle = sev_tls_wrap_native(transport, server_name_raw);
  return handle ? __sev_variant_new("ok", handle) : sev_failure("TLS handshake or certificate verification failed");
}

void *__sev_tls_read(void *handle_raw, int64_t count) {
  sev_tls_handle *handle = handle_raw;
  if (!handle || handle->closed || count < 0 || count > INT_MAX) return sev_failure("invalid TLS read");
  unsigned char *buffer = sev_allocate(count > 0 ? (size_t)count : 1); int received = SSL_read(handle->stream, buffer, (int)count);
  if (received <= 0) { int error = SSL_get_error(handle->stream, received); if (error == SSL_ERROR_ZERO_RETURN) received = 0; else { free(buffer); return sev_failure("TLS read failed"); } }
  void *result = sev_tls_bytes_from_buffer(buffer, (size_t)received); free(buffer); return __sev_variant_new("ok", result);
}

void *__sev_tls_write(void *handle_raw, void *bytes_raw) {
  sev_tls_handle *handle = handle_raw; unsigned char *buffer = NULL; size_t size = 0;
  if (!handle || handle->closed || !sev_tls_buffer_from_bytes(bytes_raw, &buffer, &size)) return sev_failure("invalid TLS write");
  if (size > INT_MAX) { free(buffer); return sev_failure("TLS write is too large"); }
  size_t offset = 0; while (offset < size) { int written = SSL_write(handle->stream, buffer + offset, (int)(size - offset)); if (written <= 0) { free(buffer); return sev_failure("TLS write failed"); } offset += (size_t)written; }
  free(buffer); return __sev_variant_new("ok", __sev_box_i64((int64_t)size));
}

void *__sev_tls_close(void *handle_raw) {
  sev_tls_handle *handle = handle_raw; if (!handle || handle->closed) return sev_failure("TLS connection is closed");
  SSL_shutdown(handle->stream); SSL_free(handle->stream); SSL_CTX_free(handle->context);
  sev_error_v1 error = {0}; sev_abi_v1_network_close(handle->transport, &error);
  handle->closed = true; return __sev_variant_new("ok", NULL);
}
"#
}

#[cfg(test)]
mod tests {
    use super::source;

    #[test]
    fn tls_provider_verifies_chain_and_hostname() {
        let source = source();
        assert!(source.contains("SSL_CTX_set_default_verify_paths"));
        assert!(source.contains("SSL_VERIFY_PEER"));
        assert!(source.contains("X509_VERIFY_PARAM_set1_host"));
        assert!(source.contains("SSL_get_verify_result"));
    }
}
