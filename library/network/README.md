# Network

`network` exposes typed, byte-oriented TCP and UDP connections. Hostnames are
resolved with the host resolver across IPv4 and IPv6; operating-system socket
descriptors remain opaque behind package-owned C-v1 handles.

```sev
import network

connection = network.connect("example.com", 80)
_written = connection.write_all([71, 69, 84, 32, 47, 32, 72, 84, 84, 80, 47, 49, 46, 48, 13, 10, 13, 10])
response = connection.read(4096)
_closed = connection.close()
```

`TcpConnection.read`, `write`, and `close` match `io.Reader`, `io.Writer`, and
`io.Closer`; `read_exact` and `write_all` handle partial stream operations.
`resolve` and `parse_ip` expose normalized addresses. Listeners, timeouts,
keep-alive, half-close, UDP datagrams, and local/remote address inspection use
the same typed error boundary.

The unsafe declarations live in `src/ffi.sev`, while the POSIX socket and DNS
provider lives under `native/`. The compiler generates representation-conversion
shims and links the selected provider without embedding networking behavior.

Networking transports bytes. Text encoding and protocol framing belong to
higher layers such as `tls` and `http`.
