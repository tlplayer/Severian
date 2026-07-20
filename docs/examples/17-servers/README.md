# Server Fixtures

These examples establish the intended shape of Severian's networking and
long-running concurrency APIs.

- `network.listen(address)` creates an owned `TCPListener`.
- `await listener.accept()` returns a move-only `TCPConnection`.
- `with` deterministically closes listeners and connections at scope exit.
- `connection.split()` creates independently owned reader and writer halves.
- Functions and methods use `lowerCamelCase`; variables, parameters, and fields
  use `snake_case`; class, trait, enum, and enum-variant names use
  `UpperCamelCase`.
- `Channel[type] with Buffer(capacity)` creates a bounded channel.
- Cloning a channel clones an endpoint, never the queued values.
- Dropping the final sending endpoint closes the channel.
- A channel `switch` commits exactly one ready receive and leaves the others
  untouched.
- `Type from channel:` receives and destructures an owned class or enum value;
  `name from channel:` binds an entire received value under that lowercase name.
- Parameter declarations contain only names and types. An async call lists each
  scoped captured binding after its owner, as in
  `async serve(connection) with self and connection`.

The examples are ordered by the amount of runtime support they require:

1. A simple request/response TCP server with one structured task per connection.
2. A chat server with a single-owner hub and channel-based client sessions.
3. A map/reduce server with a bounded worker queue and per-job reply channels.
