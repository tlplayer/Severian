# hypervisor

The hypervisor layer models virtual CPUs, guest-memory mappings, interrupts,
and devices. Its native KVM boundary performs read-only API discovery and, when
authorized by the host, a create-and-close VM descriptor probe. Tests remain
portable by accepting an unavailable backend rather than pretending KVM exists.

Guest execution, device emulation, and long-lived descriptor ownership are not
yet enabled; those require privileged integration tests on a KVM-capable host.
