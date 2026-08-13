# network

The public networking package. The name is deliberately `network`, not `net`.

This package owns connection, listener, address, and socket APIs while the
explicit `platform` package owns OS handles and syscalls. Native binding and
loopback echo are compile-link-execute tested. Rich connection handles,
ownership, TLS, timeouts, and lifecycle APIs remain future work.

`loopback_echo(message)` is the canonical deterministic native transport probe.
The older `loopbackEcho` spelling remains a temporary compatibility alias.
