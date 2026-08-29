# Network

API ID: `library.network`

The public module covers typed addresses, TCP/UDP resources, resolution, exchange helpers, and typed failures. Native FFI helpers are an implementation boundary; current language visibility still exposes some wrapper helpers.

```sev
def network_port_subject(port: int) -> bool:
    return port >= 0 and port <= 65535
```

Current weakness: per-declaration visibility is required to hide internal wrapper helpers cleanly.
