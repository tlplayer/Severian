# network

The public networking package. The name is deliberately `network`, not `net`.

This package will own typed connection, listener, address, and socket APIs. The
runtime will own OS handles and syscalls. Connection ownership, concurrent
access, closure, timeout, and `IOError` behavior must be represented in the
public declarations before this package becomes experimental.

