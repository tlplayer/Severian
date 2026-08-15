#ifndef SEVERIAN_NETWORK_ABI_H
#define SEVERIAN_NETWORK_ABI_H

#ifndef SEVERIAN_C_V1_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct { const uint8_t *data; size_t length; } sev_string_view_v1;
typedef struct { const uint8_t *data; size_t length; } sev_bytes_view_v1;
typedef struct { void *value; } sev_handle_v1;
typedef struct { int32_t code; sev_string_view_v1 message; } sev_error_v1;
#endif

int32_t sev_abi_v1_network_connect(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *connection, sev_error_v1 *error);
int32_t sev_abi_v1_network_listen(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *listener, sev_error_v1 *error);
int32_t sev_abi_v1_network_accept(sev_handle_v1 listener, sev_handle_v1 *connection, sev_error_v1 *error);
int32_t sev_abi_v1_network_read(sev_handle_v1 connection, size_t count, sev_handle_v1 *bytes, sev_error_v1 *error);
int32_t sev_abi_v1_network_write(sev_handle_v1 connection, sev_bytes_view_v1 data, size_t *written, sev_error_v1 *error);
int32_t sev_abi_v1_network_close(sev_handle_v1 connection, sev_error_v1 *error);
int32_t sev_abi_v1_network_descriptor(sev_handle_v1 connection);
int32_t sev_abi_v1_network_listener_close(sev_handle_v1 listener, sev_error_v1 *error);
int32_t sev_abi_v1_network_local_address(sev_handle_v1 connection, sev_handle_v1 *address, sev_error_v1 *error);
int32_t sev_abi_v1_network_remote_address(sev_handle_v1 connection, sev_handle_v1 *address, sev_error_v1 *error);
int32_t sev_abi_v1_network_listener_address(sev_handle_v1 listener, sev_handle_v1 *address, sev_error_v1 *error);
int32_t sev_abi_v1_network_set_read_timeout(sev_handle_v1 connection, uint64_t milliseconds, sev_error_v1 *error);
int32_t sev_abi_v1_network_set_write_timeout(sev_handle_v1 connection, uint64_t milliseconds, sev_error_v1 *error);
int32_t sev_abi_v1_network_set_keep_alive(sev_handle_v1 connection, bool enabled, sev_error_v1 *error);
int32_t sev_abi_v1_network_shutdown(sev_handle_v1 connection, int32_t direction, sev_error_v1 *error);

sev_string_view_v1 sev_abi_v1_network_address_host(sev_handle_v1 address);
uint16_t sev_abi_v1_network_address_port(sev_handle_v1 address);
int32_t sev_abi_v1_network_address_family(sev_handle_v1 address);
void sev_abi_v1_network_address_release(sev_handle_v1 address);
size_t sev_abi_v1_network_bytes_length(sev_handle_v1 bytes);
uint8_t sev_abi_v1_network_bytes_at(sev_handle_v1 bytes, size_t index);
void sev_abi_v1_network_bytes_release(sev_handle_v1 bytes);

int32_t sev_abi_v1_network_resolve(sev_string_view_v1 host, sev_handle_v1 *addresses, sev_error_v1 *error);
size_t sev_abi_v1_network_address_list_length(sev_handle_v1 addresses);
sev_string_view_v1 sev_abi_v1_network_address_list_at(sev_handle_v1 addresses, size_t index);
void sev_abi_v1_network_address_list_release(sev_handle_v1 addresses);
int32_t sev_abi_v1_network_parse_ip(sev_string_view_v1 value, sev_handle_v1 *address, sev_error_v1 *error);

int32_t sev_abi_v1_network_udp_bind(sev_string_view_v1 host, uint16_t port, sev_handle_v1 *socket, sev_error_v1 *error);
int32_t sev_abi_v1_network_udp_send_to(sev_handle_v1 socket, sev_bytes_view_v1 data, sev_string_view_v1 host, uint16_t port, size_t *sent, sev_error_v1 *error);
int32_t sev_abi_v1_network_udp_receive_from(sev_handle_v1 socket, size_t count, sev_handle_v1 *packet, sev_error_v1 *error);
int32_t sev_abi_v1_network_udp_close(sev_handle_v1 socket, sev_error_v1 *error);
size_t sev_abi_v1_network_packet_length(sev_handle_v1 packet);
uint8_t sev_abi_v1_network_packet_at(sev_handle_v1 packet, size_t index);
sev_string_view_v1 sev_abi_v1_network_packet_host(sev_handle_v1 packet);
uint16_t sev_abi_v1_network_packet_port(sev_handle_v1 packet);
void sev_abi_v1_network_packet_release(sev_handle_v1 packet);

int32_t sev_abi_v1_network_loopback_echo(sev_string_view_v1 message, sev_handle_v1 *text, sev_error_v1 *error);
sev_string_view_v1 sev_abi_v1_network_text_value(sev_handle_v1 text);
void sev_abi_v1_network_text_release(sev_handle_v1 text);

#endif
