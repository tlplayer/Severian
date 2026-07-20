# network

The public networking package. The name is deliberately `network`, not `net`.

This package owns connection, listener, address, and socket APIs while the
runtime owns OS handles and syscalls. The initial typed `listen` declaration is
enough to check the server examples. Connection methods, ownership, concurrent
access, closure, timeout, and runtime symbols are still pending.
