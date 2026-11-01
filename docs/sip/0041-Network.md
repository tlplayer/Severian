# SIP-0041: Networking

Status: Draft
Type: Package | Runtime | Interop
Tags: network | tcp | udp | io | readiness | ownership | chat

## Appendix

- address: numeric IP address and port; IPv6 also carries a scope identifier.
- resolver: hostname/service lookup returning candidate addresses or a typed error.
- listener: owner of a bound TCP listening socket.
- connection: owner of a TCP byte stream; reads and writes may complete partially.
- datagram: one UDP packet, its source address, and explicit truncation metadata.
- readiness: a hint that attempting an operation may make progress, not completion.
- half-close: end one TCP direction while retaining the other.
- frame: application message reconstructed from bytes; not a TCP packet.
- backpressure: bounded queued work makes its producer wait or reject new work.
- pending: an operation would block; distinct from EOF and failure.
- deadline: a monotonic time point shared across retries.
- reactor: multiplexes readiness for many nonblocking sockets.

## Goal(s)

```text
network.listen(host, port) -> listener -> accept -> connection
network.connect(host, port) -> connection
network.udp_bind(host, port) -> socket -> send/receive datagrams
TCP -> shared IO Reader/Writer contracts from SIP-0037
UDP -> datagram contract; never pretend packet boundaries are a byte stream
library control flow -> MLIR; target-specific native boundary -> MLIR call/syscall
```

## Examples, Problems, & Tests

### Portable interface contract

```sev
class Address:
    host: string
    port: int

def resolve(host: string, port: int, deadline: Instant) -> list[Address] | NetworkError

def listen(host: string, port: int, backlog: int = 64) -> Listener | NetworkError

def connect(host: string, port: int, deadline: Instant) -> Connection | NetworkError

def udp_bind(host: string, port: int) -> UdpSocket | NetworkError

# Listener:
#   address() -> actual local Address, including ephemeral port chosen for port 0
#   accept(deadline?) -> Connection | NetworkError
#   close() -> unit | NetworkError
# Connection implements io.Reader and io.Writer:
#   address() -> local Address
#   peer() -> remote Address
#   shutdown_write() -> unit | NetworkError
#   close() -> unit | NetworkError
# UdpSocket:
#   address() -> local Address
#   send(data, host, port, deadline?) -> Unit[Data] | NetworkError
#   receive(capacity: Unit[Data], deadline?) -> Datagram | NetworkError
# Datagram -> data, address, truncated, wire_bytes: Unit[Data]
```

```text
"127.0.0.1", 0 -> IPv4 loopback + ephemeral port
"::1", 0 -> IPv6 loopback + ephemeral port, when provider supports IPv6
"0.0.0.0", 7777 -> explicit IPv4 wildcard; never the default for examples
hostname -> resolver -> bounded candidate attempts sharing one connect deadline
IPv4-mapped IPv6 -> explicit dual-stack policy, not platform-dependent guessing
invalid port / embedded NUL / malformed numeric address -> InvalidInput
unsupported address family -> Unsupported, not silently bind a different address
bind collision -> AddressInUse with operation and native code
connect refused -> ConnectionRefused; resolver failure -> NameResolution
```

### Resource lifetime and cancellation

```text
socket -> one owner; borrow for IO, move to a task, close/drop once
listen succeeds + later initialization fails -> listener released
connect timeout -> close pending socket; no background connection leaked
accept succeeds -> accepted socket is close-on-exec and nonblocking atomically
close -> invalidate owner before native close; never retry Linux close on EINTR
pending IO -> retain buffer loan until completion/cancellation is acknowledged
cancel scope -> wake accept/read/write -> join tasks -> release resources
native error -> capture before cleanup; preserve operation and transferred bytes
send on disconnected peer -> typed error, never process-wide SIGPIPE termination
```

### TCP is a byte stream

```text
sender writes "hel", then "lo\n" -> receiver may read "hello\n" in one call
sender writes "a\nb\n" -> receiver may read "a", "\nb", "\n"
read positive capacity -> 0 bytes means EOF
read zero capacity -> 0 bytes; this is not evidence of EOF
read pending -> readiness wait, not empty data
partial send n -> resume at n, not at zero
send returns zero with bytes remaining -> NoProgress
shutdown_write -> queued writes precede peer EOF; reads remain usable
peer half-close -> drain bytes already received before reporting EOF
EOF in a partial application frame -> protocol-specific error/discard policy
```

```sev
# Portable target example, with units at the IO boundary.
import network
import io
import clock

def greeting(listener: borrow network.Listener):
    connection = listener.accept()
    io.write_all(connection, "hello\n".bytes())
    connection.shutdown_write()

test with integ "greeting terminates at peer EOF":
    listener = network.listen("127.0.0.1", 0)
    serving = async greeting(borrow listener) with local, thread
    address = listener.address()
    client = network.connect(address.host, address.port, deadline=clock.monotonic() + 1s)
    assert(io.read_exact(client, 6B) == "hello\n".bytes())
    assert(client.read(1B) == [])
    await serving
```

### UDP keeps packet boundaries

```text
send 4 bytes -> one datagram; partial datagram sends are errors, not retry suffixes
receive capacity 2, packet size 4 -> 2 bytes + truncated=true + wire_bytes=4
receive empty datagram -> valid packet with source address, not EOF
receive capacity 0 -> consumes one packet, reports truncation if nonempty
lost / duplicated / reordered datagram -> transport behavior, not an API failure
send succeeds -> accepted locally; does not prove remote delivery
source address -> packet metadata, not authenticated identity
```

```sev
# The runnable Linux provider currently uses explicit integer byte counts.
test with integ "UDP reports truncation":
    receiver = network.udp_bind("127.0.0.1", 0)
    sender = network.udp_bind("127.0.0.1", 0)
    address = receiver.address()
    assert(sender.send(tuple[u8,4](1, 2, 3, 4), address.host, address.port) == 4)
    readiness = network.PollSet(1)
    readiness.watch(0, receiver.handle)
    assert(readiness.wait(1000) == 1)
    packet = receiver.receive(2)
    assert(packet.truncated and packet.wire_bytes == 4)
    assert(len(packet.data) == 2)
```

### Readiness and executors

```text
virtual task -> register readiness -> suspend -> retry operation after wake
thread task -> block only its own OS thread, or use the same readiness mechanism
process task -> own its sockets; transfer serialized messages over explicit IPC
readiness stale -> Pending is legal; no assumption that a following call must finish
error/hangup readiness -> attempt read to drain buffered bytes, then handle EOF/error
interrupted wait -> original deadline; never restart a full timeout indefinitely
server flood -> finite accepts/reads per turn; other connections still progress
slow receiver -> bounded output queue; wait or disconnect according to protocol
```

### Runnable Linux provider and chat lab

Implementation: `library/network/linux_x86_64/src/lib.sev`.
Application: `docs/labs/17-chat/src/main.sev`.

```bash
# From the repository root, Linux x86-64:
./docs/labs/17-chat/run.sh

# Two other terminals:
nc 127.0.0.1 7777
nc 127.0.0.1 7777

# In the first client:
/nick alice
hello everyone
/who
/quit

# Alternate port, explicit bind address, optional lifetime in seconds:
./docs/labs/17-chat/run.sh 9000 127.0.0.1 60

# Ephemeral port; read the actual address from startup output:
./docs/labs/17-chat/run.sh 0

# Adjacent tests, for the human to run:
./docs/labs/17-chat/check.sh
```

```text
client A: /nick alice -> both clients see the rename
client A: hello -> both clients see "[alice] hello"
client B: /who -> active nicknames
client B: /nick alice -> name-already-in-use error
client A: /quit -> bye, flush, close, one leave notification
new connection -> unique guest-N name; names may contain ASCII letters/digits/-/_
LF and CRLF -> one message; empty line -> ignored
UTF-8 split across receives -> assemble first, validate complete frame
invalid UTF-8 / terminal controls / >1024 payload bytes -> error and disconnect
valid line followed by malformed line in one read -> deliver valid line, then error
EOF with incomplete line -> discard partial line; no fabricated message
64 clients -> room full; new connection closed
queued output >64KiB -> disconnect slow client; other clients continue
no input for 300s -> idle close; no output progress for 10s -> stalled close
/quit flush grace -> at most 2s; finite server lifetime -> close every client
Ctrl+C -> process exits; kernel closes descriptors
```

The current provider deliberately exposes native nonblocking attempts:

```text
Connection.receive(buffer, offset, count) -> byte count or negative Linux errno
Connection.send(buffer, offset, count) -> byte count or negative Linux errno
-11 -> Pending; -4 -> Interrupted; other negative values -> failure
Listener.accept / connect / UDP receive -> NetworkError(operation, code)
PollSet.watch(slot, descriptor, readable, writable) -> borrowed registration
PollSet.wait(timeout_ms) -> ready count, 0 timeout, or negative errno
PollSet.readable/writable -> include error/hangup so callers make a final attempt
Connection/Listener/UdpSocket.close -> idempotent; drop closes remaining ownership
```

These integer byte-count APIs are the implemented provider surface, not a claim
that portable dimensioned IO, DNS, IPv6, TLS, or task cancellation is finished.
TCP tests exercise binary zero/high bytes, half-close, and close idempotence;
UDP tests exercise sender metadata, truncation, and zero-byte packets. HTTP/TLS
remain separate libraries; a plaintext lab does not claim authentication.

### Required provider acceptance cases

```text
listen port 0 -> actual nonzero port; two listeners never share a fixed test port
failed connect / failed bind / failed accept -> original error, no leaked socket
fragmented frame / coalesced frames -> same application messages
partial write / pending write -> exact unsent suffix queued
peer disconnect during broadcast -> other clients continue
queued traffic beyond client limit -> bounded memory
multiple readiness events -> deterministic ordering within one sender's stream
all tests -> local resources, finite deadlines, no public internet dependency
```
