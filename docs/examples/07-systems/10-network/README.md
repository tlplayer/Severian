# Network

Networking builds owned socket resources over platform providers. Address
parsing is deterministic and local; DNS resolution, socket operations, and HTTP
requests are fallible system operations.

TCP and UDP expose byte-oriented readers and writers compatible with `io`.
Higher protocols such as TLS and HTTP depend on those contracts instead of
introducing new compiler boundaries.

Integration tests bind loopback addresses and provider-selected ephemeral
ports. They do not require public internet access. The HTTP example is a usage
example rather than a hermetic test unless a local test server is supplied.
