# TLS

`tls` wraps a `network.TCPConnection` in a certificate-verified TLS stream:

```sev
import network
import tls

transport = network.connect("huggingface.co", 443)
secure = tls.connect(transport, "huggingface.co")
```

The runtime requires TLS 1.2 or newer, loads the operating system trust store,
verifies the certificate chain, verifies the requested hostname, and sends SNI.
`TlsConnection` implements the same byte-oriented read, write, and close shape
as a TCP connection.
