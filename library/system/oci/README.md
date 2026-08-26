# OCI

This package models and writes OCI runtime bundles. It deliberately delegates
namespace, cgroup, and lifecycle mechanics to an installed OCI runtime such as
`crun` or `runc`.

The development bundle keeps the host network namespace and gives the service
a distinct internal port. `container.run` applies every
`PortForward(host, container)` with a user-space TCP proxy before starting the
application. This avoids privileged NAT while still testing a real forwarded
connection.

The root filesystem is a development bundle: the application, data, and host
runtime libraries are read-only bind mounts. Producing distributable image
layers is intentionally a separate image-builder concern.
