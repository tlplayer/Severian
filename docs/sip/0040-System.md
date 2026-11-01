# SIP-0040: System

Status: Draft
Tags: system | platform | providers | abi | xxi | ownership | memory | io | files | process | environment | network | concurrency | devices | testing

## Appendix

- system: package group for interaction with the execution environment.
  - platform: compilation target facts, layout, and available contracts.
  - abi: target representation and foreign calling conventions.
  - xxi: external declarations, conversions, checked statuses, and transfer budgets.
  - path: lexical path operations; no filesystem access.
  - os: directories, metadata, filesystem mutations, and host services.
  - file: owned file access, format dispatch, mappings, and locks.
  - io: readers, writers, seekers, stream algorithms, and standard streams.
  - cli: pure argument parsing, help rendering, and command-line diagnostics.
  - process: arguments, child processes, pipes, completion, and termination.
  - environment: process environment lookup, mutation, and snapshots.
  - network: proposed addresses, name resolution, and socket providers.
  - clock: proposed monotonic time, wall time, and deadlines.
  - device: runtime device discovery and availability.
  - driver: provider connecting a device to its supported operations.
- host: machine running the compiler.
- target: architecture, operating system, and ABI of the generated program.
- provider: implementation selected for a target and required capability.
- capability: supported operation; availability does not guarantee permission or success.
- FFI: foreign function interface; argument conversion, ownership, and lifetime boundary.
- ABI: application binary interface; widths, alignment, layout, and call classification.
- MLIR: compiler representation through which native operations and foreign calls lower.
- MIR: intermediate representation carrying checked ownership and cleanup operations.
- LLVM: native code generation infrastructure following MLIR lowering.
- OS: operating system providing processes, files, and other external resources.
- POSIX: system interface family used by the explicitly POSIX integration examples.
- unsafe: explicit scope permitting operations whose safety requires external guarantees.
- codec: conversion between bytes and a typed representation such as text or JSON.
- TCP: ordered byte-stream transport; individual write boundaries are not retained.
- UDP: datagram transport; message boundaries are retained.
- DNS: name resolution to network addresses.
- HTTP: application protocol layered over a transport.
- TLS: authenticated encrypted transport layer.
- SIMD: arithmetic across multiple lanes on the host CPU.
- GPU: device executing kernels with separately owned device storage.
- handle: typed reference to an external resource; raw integers remain provider details.
  - owner: responsible for releasing the resource once.
  - view: read-only access tied to a live owner.
  - borrow: exclusive mutable access tied to a live owner.
  - move: ownership transfer that invalidates the source binding.
  - close: explicit release that can report an error.
  - drop: deterministic fallback cleanup; cannot throw during unwinding.
- SystemError: proposed structured error carrying operation, category, and native code.
- Unit[Data]: data extent; `4B` and `4KiB` are byte sizes, not element counts.
- EOF: end of a stream; distinct from an empty request, failure, and temporary unavailability.
- partial transfer: successful read or write completing fewer bytes than requested.
- budget: bound on calls and transferred data; distinct from a time deadline.
- deadline: expiration measured by monotonic time.
- mapping: owned virtual-memory view of file storage.
- lock: owned synchronization resource; filesystem lock semantics depend on the provider.
- child: owned process identity plus lifecycle state and retained completion status.
- snapshot: owned copy unaffected by later changes to its source.
- compiler test: `accept`/`reject` checks of source contracts.
- integration test: `test with integ`; isolates external state and releases it on failure.
- proposed: desired API or test contract in this draft; not a claim of implementation.

## Goal(s)

- User call -> ordinary library declaration -> selected provider -> checked foreign boundary -> MLIR -> native call.
- Same portable contract + different target -> target provider; missing provider -> diagnostic.
- Open resource -> one owner -> move or borrow -> one release on return or error.
- External failure -> typed error; preserve operation, category, and native code.
- Byte extent -> `Unit[Data]`; element count -> integer; no implicit interchange.
- Partial progress -> bounded retry; no progress -> error or readiness wait.
- System tests -> deterministic local resources; no public network or shared fixed filenames.
- Examples and tests -> one adjacent block; proposed behavior stays Draft until implemented and verified.

## Examples, Problems, & Tests

### Core interface surface

The signatures below are the portable target contract, not a claim that every
provider already implements it. Byte-oriented provider APIs currently using
integer counts must expose their units explicitly at this boundary.

```sev
# io; Reader/Writer/Seeker contracts are defined in SIP-0037.
def stdin() -> Reader
def stdout() -> Writer
def stderr() -> Writer
def read_exact(source: borrow Reader, count: Unit[Data]) -> tensor[u8] | SystemError
def read_all(source: borrow Reader, limit: Unit[Data]) -> tensor[u8] | SystemError
def write_all(destination: borrow Writer, data: view tensor[u8]) -> Unit[Data] | SystemError
def copy(source: borrow Reader, destination: borrow Writer,
         buffer_size: Unit[Data] = 64KiB) -> Unit[Data] | SystemError
def pipe() -> Pipe | SystemError       # Owned reader and writer; close-on-exec.

# path; lexical only, with target path rules.
def join(directory: string, name: string) -> string
def parent(value: string) -> string
def basename(value: string) -> string
def extension(value: string) -> string
def normalize(value: string) -> string
def is_absolute(value: string) -> bool

# os; effectful filesystem operations.
def current_directory() -> string | SystemError
def temporary(parent: string, prefix: string) -> string | SystemError
def entries(directory: string) -> list[DirectoryEntry] | SystemError
def stat(path: string, follow_links: bool = true) -> Metadata | SystemError
def mkdir(path: string, parents: bool = false) -> unit | SystemError
def copy(source: string, destination: string, replace: bool = false) -> unit | SystemError
def rename(source: string, destination: string, replace: bool = false) -> unit | SystemError
def remove(path: string) -> unit | SystemError
def remove_tree(path: string) -> unit | SystemError

# file; descriptor ownership is separate from filename/format dispatch.
def open(path: string, mode: OpenMode) -> File | SystemError
def read_bytes(path: string, limit: Unit[Data] = 64MiB) -> tensor[u8] | SystemError
def write_bytes(path: string, data: view tensor[u8]) -> Unit[Data] | SystemError
def sync(file: borrow File) -> unit | SystemError

# environment; child configuration uses a private snapshot.
def snapshot() -> Snapshot | SystemError
def get(name: string) -> string | None | SystemError

# process; no implicit shell, including run("command text").
def arguments() -> list[string]         # Includes executable at index zero.
def executable() -> string | SystemError
def id() -> ProcessId
def spawn(argv: list[string], options: SpawnOptions = SpawnOptions()) -> Child | SystemError
def wait(child: borrow Child) -> Exit | SystemError
def try_wait(child: borrow Child) -> Exit | None | SystemError
def terminate(child: borrow Child) -> bool | SystemError
def kill(child: borrow Child) -> bool | SystemError
def execute(argv: list[string], options: CaptureOptions = CaptureOptions()) -> Output | SystemError

# clock; time points cannot be substituted for durations.
def monotonic() -> Instant | SystemError
def wall() -> Timestamp | SystemError
def sleep(duration: Unit[Time]) -> unit | SystemError
def sleep_until(deadline: Instant) -> unit | SystemError

# signal; ordinary task delivery, never user code inside a native handler.
def subscribe(kinds: list[Signal]) -> SignalStream | SystemError
```

```text
Metadata -> kind, size: Unit[Data], modified: Timestamp, permissions
SpawnOptions -> cwd?, environment?, stdin/stdout/stderr: Inherit | Null | Pipe
CaptureOptions -> SpawnOptions + byte_limit + deadline + termination_grace
Exit -> Exited(code) | Signalled(signal)       # Portable, no 128+signal guess.
Output -> exit: Exit, stdout: tensor[u8], stderr: tensor[u8]
Child -> owning identity, completion cache, stdin?, stdout?, stderr?
terminate -> graceful OS request; wait observes actual completion
kill -> force termination where supported; still requires reap
try_wait -> None while live; terminal Exit after completion, cached thereafter
File.close -> report error; File.drop -> best-effort nonthrowing release
File.sync -> storage durability request; Writer.flush -> buffering only
SignalStream.close -> unsubscribe; restore previous process signal disposition
SIGINT/SIGTERM -> request task-scope cancellation -> join -> close -> exit
SIGKILL -> cannot subscribe, catch, or promise cleanup
```

### CLI parsing is a pure operation

```sev
import cli

# Existing positional surface; parse() excludes argv[0], parse_process() skips it.
def chat_arguments() -> cli.Command:
    return cli.command("chat", "TCP chat lab", [
        cli.argument("port", required=false),
        cli.argument("host", required=false),
    ])

test "CLI parsing uses its supplied input and retains literal values":
    options = cli.parse(chat_arguments(), ["7777", "127.0.0.1"])
    assert(cli.values(options, "port") == ["7777"])
    assert(cli.values(options, "host") == ["127.0.0.1"])
    assert(cli.values(cli.parse(chat_arguments(), []), "port") == [])
    throws(cli.parse(chat_arguments(), ["1", "host", "extra"]) -> cli.CliError)

# Proposed named-option extension, alongside positional argument():
# option("port", short="p", default="7777")
# flag("verbose", short="v")
# parse(["--port=9000", "--", "-literal"]) -> port="9000", positional="-literal"
# parse(["--unknown"]) -> CliError with offending token and usage
# parse(["--port"]) -> CliError: missing value
# repeated scalar option -> error; multiple=true -> ordered values
# absent optional argument -> []; present empty argument -> [""]
# --help/--version -> renderable result; parsing never prints or exits
# integer conversion/range validation -> explicit application operation
# parser does not expand $HOME, quotes, wildcards, or shell substitutions
```

### Blocking, deadlines, and progress

```text
Portable blocking operation -> ready result | typed error
Nonblocking attempt -> Ready(value) | Pending | typed error
Pending -> register readiness -> suspend current virtual task; never busy-spin
Thread executor -> may block its own OS thread
Virtual executor + non-pollable blocking syscall -> bounded blocking-worker pool
Process executor -> IPC transports owned values; inherited descriptors are opt-in

read_exact failure -> error includes bytes transferred; consumed input stays consumed
write_all timeout -> error includes bytes transferred; do not replay the prefix
deadline D + interrupted syscall -> retry with D, not a fresh timeout
cancelled native operation -> acknowledge cancellation before freeing its buffer
shutdown requested -> stop accepting work, request cancellation, join, release resources
cleanup failure during an existing failure -> attach as secondary diagnostic
```

Networking has its own contract in [SIP-0041](0041-Network.md). Task ownership,
completion and executor selection are defined in [SIP-0035](0035-Concurrency.md).

### Package and lowering boundaries

```text
application
  -> path                         lexical operations, no provider needed
  -> file / os / process / environment / network / clock
       -> io                      shared stream contracts where applicable
       -> selected provider
            -> xxi                checked conversions, statuses, lifetimes
                 -> abi           compilation-target representation
                      -> MLIR call -> LLVM -> linked external symbol

prelude.print -> io.print -> Writer
core types and containers -> no dependency on OS providers
system packages -> no dependency on compiler AST packages

C needed for an OS/incompatible call -> fixed provider source in c/
Severian arithmetic/control flow -> MLIR operations
Severian expression -> generated C implementation -> REJECT
missing provider -> silently shell out or choose host ABI -> REJECT

# Related contracts:
# 0039-C-ABI-XXI.md: native boundaries and GPU completion
# 0037-io.md: Reader, Writer, Seeker, and print
# 0027-Ownership.md: resource lifetimes
# 0035-Concurrency.md: task owners and cancellation
# 0025-Testing.md: accept/reject, throws, memory, and integration tests
# ../examples/04-systems/: portable system examples
```

### Target facts and capability selection

```sev
import platform

class Header:
    tag: u8
    length: u32

def can_launch_tools(selected: platform.Target) -> bool:
    return selected.capabilities.contains(platform.Processes)

test "layout and widths describe the compilation target":
    selected = platform.current()
    layout = platform.layout[Header](selected)
    assert(selected.pointer_bits == 32b or selected.pointer_bits == 64b)
    assert(selected.byte_order == platform.LittleEndian or
           selected.byte_order == platform.BigEndian)
    assert(layout.size >= 5B)
    assert(layout.field("tag").offset == 0B)
    assert(layout.field("length").offset % layout.field("length").alignment == 0B)

# Capability absence -> Unsupported before allocating a resource.
# Capability presence + access denied -> PermissionDenied, not Unsupported.
# Processes does not imply ProcessPipes; check each required capability.
# Host Linux + target Windows -> Windows provider and Windows ABI.
# Target capability != a successful probe of hardware on the build host.
```

```sev
import abi

test "cross-target C layout does not consult the host":
    linux = abi.c_target(abi.X86_64, abi.Linux)
    windows = abi.c_target(abi.X86_64, abi.Windows)
    assert(abi.scalar_layout(abi.CLong, linux).size_bytes == 8)
    assert(abi.scalar_layout(abi.CLong, windows).size_bytes == 4)
    assert(abi.scalar_layout(abi.Pointer, windows).size_bytes == 8)

# ABI *_bytes fields explicitly count bytes as integers at this boundary.
# Convert to Unit[Data] before exposing an extent through portable IO.
# Cross-target layout tests inspect data/IR; never execute foreign objects.
```

### Structured errors preserve the failed operation

```sev
# Proposed shared categories; packages may expose more specific error types.
enum SystemErrorKind:
    NotFound
    PermissionDenied
    AlreadyExists
    InvalidInput
    Unsupported
    Interrupted
    WouldBlock
    TimedOut
    Cancelled
    UnexpectedEof
    NoProgress
    NativeFailure

class SystemError: Error
    operation: string
    kind: SystemErrorKind
    native_code: int

def denied_open() -> unit | SystemError:
    throw SystemError("file.open", SystemErrorKind.PermissionDenied, 13)

test "error capture retains structure":
    result ?= denied_open()
    assert(result is SystemError)
    assert(result.operation == "file.open")
    assert(result.kind == SystemErrorKind.PermissionDenied)
    assert(result.native_code == 13)
    throws(denied_open() -> SystemError)

# Native 13 is fixture data, not a portable PermissionDenied constant.
# Unknown native code -> NativeFailure with the original code.
# Failed metadata lookup -> error, not size 0B or is_file=false.
# Failed read -> error, not empty data pretending to be EOF.
# Error translation captures the native error before another foreign call.
```

### Byte sizes, offsets, and element counts

```sev
import io

test "stream offsets count bytes":
    stream = io.IOStream([10, 20, 30, 40])
    assert(stream.read(2B) == [10, 20])
    assert(stream.tell() == 2B)
    assert(stream.remaining() == 2B)
    stream.seek(-1B, io.End)
    assert(stream.read(1B) == [40])
    assert(stream.read(1B) == [])             # EOF

test "an empty request does not establish EOF":
    stream = io.IOStream([10])
    assert(stream.read(0B) == [])
    assert(stream.tell() == 0B)
    assert(stream.read(1B) == [10])

test with compiler "bytes and element counts are distinct":
    reject:
        stream = io.IOStream([10])
        stream.seek(1, io.Start)
    accept:
        stream = io.IOStream([10])
        stream.seek(0B, io.Start)

# Four u32 elements -> 16B of payload, not 4B.
# Negative read size -> InvalidInput; negative relative seek may be valid.
# count * element_size overflow -> error before allocation or foreign call.
```

### Ownership, closing, and unwinding

```sev
from io import capture_stdout

# Local resource model: cleanup emits an observable event.
class RecordedHandle:
    identity: int

    def drop():
        recorded ?= print(identity)          # Fixture probe; do not throw from drop.

def release_on_return():
    owner = RecordedHandle(41)
    destination = move owner

def release_on_error() -> unit | Error:
    owner = RecordedHandle(42)
    throw Error("operation failed")

test with integ "move and unwinding each release one owner":
    release_on_return()
    throws(release_on_error() -> Error)
    assert(capture_stdout() == "41\n42\n")

# The ownership rule below uses the real file mapping contract instead of
# treating a copyable integer descriptor as an owned file.
import file

test with compiler "a released mapping cannot be read":
    reject:
        mapping = file.map("owned.bin")
        drop(mapping)
        value = mapping[0]

test with compiler "a mapping view cannot escape its owner":
    reject:
        def invalid_mapping_view() -> view u8:
            mapping = file.map("owned.bin")
            return view mapping[0]

test with compiler "moving a mapping invalidates its source":
    reject:
        mapping = file.map("owned.bin")
        destination = move mapping
        value = mapping[0]

# Required provider lifecycle cases:
# acquire -> normal return                     -> release once
# acquire -> throw                            -> release once
# acquire -> move -> drop destination          -> release once
# acquire -> explicit close -> scope exit      -> no second native close
# acquire A -> acquire B fails                 -> release A once
# operation fails -> cleanup also fails        -> preserve operation failure
# explicit close fails                        -> report close failure
# borrowed stdin/stdout/stderr wrapper drops   -> process handle stays open
# POSIX close reports EINTR                    -> provider policy; never blind retry
# copy owned handle                           -> reject; explicit duplicate may fail
```

### Foreign calls stay behind unsafe provider boundaries

```sev
import c from xxi

@c(symbol="strlen")
def c_strlen(text: view string) -> usize

def c_string_length(text: string) -> usize | Error:
    if text.contains("\0"):
        throw Error("C string contains an embedded NUL")
    unsafe:
        return c_strlen(text)

test "the adapter rejects truncation before entering C":
    assert(c_string_length("") == 0)
    assert(c_string_length("severian") == 8)
    throws(c_string_length("sev\0erian") -> Error)

test with compiler "safe callers use the wrapper":
    reject:
        c_strlen("direct foreign call")
    accept:
        c_string_length("checked call")

# string -> temporary NUL-terminated storage -> MLIR call -> release temporary.
# C retaining that pointer after return -> invalid provider implementation.
# Binary data containing zero -> pointer + explicit extent; never strlen.
# Foreign pointer result -> explicit allocator/owner/release contract.
# Unsupported varargs, callbacks, or aggregates by value -> diagnostic.
# unsafe permits the foreign operation; it does not prove the foreign code safe.
```

### Partial transfers, EOF, and bounded retries

```sev
import io
import xxi

def duplicate(source: io.Reader, destination: io.Writer) -> Unit[Data] | io.StreamError:
    return io.copy(source, destination, 2B)

test "copy loops until EOF":
    source = io.IOStream([1, 2, 3, 4, 5])
    destination = io.IOStream([])
    assert(duplicate(source, destination) == 5B)
    assert(destination.snapshot() == [1, 2, 3, 4, 5])

test "partial writes account only for transferred bytes":
    budget = xxi.budget(2, 5)
    budget = xxi.step(budget, xxi.transfer(2, 5, true))
    assert(budget.calls_remaining == 1)
    assert(budget.bytes_remaining == 3)
    budget = xxi.step(budget, xxi.transfer(3, 3, true))
    assert(budget.calls_remaining == 0)
    assert(budget.bytes_remaining == 0)

test "invalid transfer reports cannot escape the boundary":
    throws(xxi.transfer(-1, 5) -> xxi.BoundaryError)
    throws(xxi.transfer(6, 5) -> xxi.BoundaryError)
    throws(xxi.transfer(0, 5, true) -> xxi.BoundaryError)
    throws(xxi.step(xxi.budget(0, 5)) -> xxi.BoundaryError)
    assert(xxi.transfer(0, 0, true) == 0)

# Reader returns 0B for a nonempty blocking request -> EOF.
# read_exact(4B), input has 3B -> UnexpectedEof with progress retained.
# Writer returns 0B while bytes remain -> NoProgress, never busy-loop.
# WouldBlock -> readiness wait bounded by deadline/cancellation, not EOF.
# Interrupted -> retry only when the operation's provider contract permits it.
# read_all -> finite configured byte/call limits; limit reached -> error.
# flush -> userspace buffers submitted; durable storage needs a separate sync.
```

### Printing is a library operation

```sev
import io
import print from io

def write_message(destination: io.Writer) -> Unit[Data] | io.StreamError:
    return io.write_all(destination, "hello".bytes())

test "formatting can target an ordinary memory writer":
    destination = io.IOStream([])
    assert(write_message(destination) == 5B)
    assert(destination.snapshot() == "hello".bytes())

# print(value) -> format -> stdout Writer -> propagate write failure.
# stderr -> separate Writer; no implicit process termination.
# print -> no compiler opcode, implicit flush, or generated C expression.
# Broken pipe -> stream error; provider must not unexpectedly terminate the process.
```

### Paths and filesystem mutations

```sev
import path
import os
import file

test "lexical path operations do not require an existing file":
    value = path.join("blueprint-directory-that-need-not-exist", "data.txt")
    assert(path.basename(value) == "data.txt")
    assert(path.extension(value) == ".txt")

test with integ "filesystem changes use a unique owned directory":
    # POSIX provider integration example; another provider supplies its temp root.
    directory = os.temporary("/tmp", "sev-system-")
    defer os.remove_tree(directory)
    original = path.join(directory, "original.txt")
    copied = path.join(directory, "copied.txt")
    renamed = path.join(directory, "renamed.txt")
    file.write(original, "severian")
    assert(os.stat(original).size == 8B)
    os.copy(original, copied)
    assert(file.read(copied) == "severian")
    os.rename(copied, renamed)
    assert(not os.exists(copied))
    assert(file.read(renamed) == "severian")
    os.remove(renamed)
    assert(not os.exists(renamed))

# Lexical normalization does not prove containment or resolve symlinks.
# exists(path) -> another process removes path -> open may still fail.
# Never use exists() as authorization for a later destructive operation.
# rename across filesystems -> explicit error; no hidden copy/delete fallback.
# publish without replacement + destination exists -> AlreadyExists.
# read directory -> entries may change; no implicit snapshot/order guarantee.
# remove_tree -> do not follow symlink entries into another directory tree.
```

### Binary files, text codecs, mappings, and locks

```sev
import file
import os
import path
from collections import equal

test with integ "binary data preserves zero and non-UTF-8 bytes":
    directory = os.temporary("/tmp", "sev-binary-")
    defer os.remove_tree(directory)
    filename = path.join(directory, "payload.bin")
    payload = tuple[u8,4](0, 127, 128, 255)
    file.write_bytes(filename, payload)
    assert(equal(file.read_bytes(filename), payload))
    with file.map(filename) as mapping:
        assert(bytes(mapping) == 4B)
        assert(mapping[0] == 0)
        assert(mapping[3] == 255)
    # Mapping has been released before directory cleanup.

# Proposed return contract: write_bytes(...) -> Unit[Data], here 4B.
# Existing examples returning 4 element counts must converge on the IO contract.
# read_bytes -> exact bytes; invalid UTF-8 is valid binary input.
# text decoding + invalid encoding -> typed codec error, never silent replacement.
# file.read("document.json") -> registered JSON reader -> parsed value or error.
# Unknown format + no explicit codec -> error; never guess arbitrary content.
# Empty file -> empty owned mapping or Unsupported; never dereference address 0.
# mapping[index == element_count] -> bounds failure.
# External file truncation can invalidate a mapping; document provider limitations.
# Lock contention -> wait with deadline or WouldBlock; no spin loop.
# Lock owner returns/throws -> unlock once; scope does not protect unrelated writers.
```

### Processes preserve arguments and completion

```sev
import process

# Current provider compatibility: wait(child) -> int, Output.status -> int.
# The portable target above replaces this encoding with Exit/Output.exit.

test with integ "arguments are passed literally to the executable":
    # POSIX fixture; shell use is explicit and its program is constant.
    result = process.execute([
        "/bin/sh", "-c", "printf '%s' \"$1\"", "sev-fixture",
        "two words; $HOME; $(echo expanded)",
    ])
    assert(result.status == 0)
    assert(result.stdout == "two words; $HOME; $(echo expanded)")
    assert(result.stderr == "")

test with integ "nonzero exit is a completion result":
    child = process.spawn(["/bin/sh", "-c", "exit 7"])
    assert(process.wait(child) == 7)
    assert(child.completed)
    assert(process.wait(child) == 7)
    assert(not process.kill(child))

# Proposed safe surface: spawn(argv) -> owned Child | SystemError.
# Empty argv / embedded NUL -> InvalidInput before spawning.
# Missing executable -> spawn error, not a fabricated child exit status.
# run/execute -> wait and reap on every completion path.
# execute -> drain stdout and stderr concurrently; bound captured output.
# Output limit reached -> terminate according to policy, drain/close pipes, reap, error.
# Dropped live Child -> terminate and reap under its declared lifecycle policy.
# Waiting twice -> cached status; never wait on a recycled numeric PID.
# POSIX signal 15 -> existing status convention 143; not a portable signal encoding.
# Descriptor inheritance -> explicit allowlist; close unrelated handles in child.
# Untrusted string concatenation -> shell execution -> rejected API design.
```

### Environment lookup distinguishes absence from empty text

```sev
# Proposed snapshot interface; mutations below affect owned data, not the host.
import environment

test "snapshot lookup preserves empty values":
    values = environment.Snapshot()
    assert(values.get("MODE") is None)
    values.set("MODE", "")
    assert(values.get("MODE") == "")
    values.set("MODE", "development")
    assert(values.get("MODE") == "development")
    values.remove("MODE")
    assert(values.get("MODE") is None)

test "an independent snapshot is stable":
    original = environment.Snapshot()
    original.set("MODE", "before")
    copied = copy original
    original.set("MODE", "after")
    assert(copied.get("MODE") == "before")

# snapshot() -> owned snapshot of current process environment.
# Spawn with explicit snapshot -> child receives those values; parent unchanged.
# Invalid name (empty, '=', NUL) / value containing NUL -> InvalidInput.
# Global environment mutation -> serialized integration test or isolated subprocess.
# Test changes host environment -> restore both original value and original absence.
```

### Deadlines use monotonic time

```sev
# Proposed clock contract: time points and durations are different types.
# FakeClock is a proposed provider fixture, advanced without sleeping.
import clock

test "wall-clock changes do not change a timeout":
    source = clock.FakeClock()
    deadline = source.monotonic() + 5s
    source.advance(4s)
    source.adjust_wall(-1h)
    assert(source.monotonic() < deadline)
    source.advance(1s)
    assert(source.monotonic() >= deadline)

# Timeout 0s -> immediate poll; negative timeout -> InvalidInput.
# Readiness and timeout together -> defined provider arbitration; complete once.
# Interrupted wait -> recompute remaining time from the original deadline.
# Restarting a full timeout on every interruption -> reject; can wait forever.
# Clock.sleep -> may wake later; never promise exact scheduler timing.
```

### Network sockets reuse IO contracts

```sev
# Proposed network API; bind an ephemeral loopback port, never a fixed port.
import network
import io
import clock

def serve_once(listener: borrow network.Listener) -> unit | Error:
    connection = listener.accept()
    defer connection.close()
    io.write_all(connection, "hello".bytes())
    connection.shutdown_write()

test with integ "TCP is a byte stream and peer shutdown produces EOF":
    listener = network.listen("127.0.0.1", 0)
    defer listener.close()
    server = async serve_once(listener) with local, thread
    address = listener.address()
    connection = network.connect(address.host, address.port, deadline=clock.monotonic() + 1s)
    defer connection.close()
    assert(io.read_exact(connection, 5B) == "hello".bytes())
    assert(connection.read(1B) == [])
    await server

# Test harness deadline cancels pending accept/read and joins the server on failure.
# TCP write boundaries -> not preserved as read boundaries.
# UDP -> separate datagram API; truncation must be reported, not hidden.
# DNS -> addresses or resolution error; no dependency on public DNS in unit tests.
# HTTP/TLS -> libraries over transport; no compiler syntax or automatic plaintext fallback.
# Cancellation -> release pending operation's buffer loan after native completion.
```

### Tasks and devices retain resources until completion

```sev
def inspect_payload(data: view tensor[u8]) -> int:
    return int(data[0])

test with compiler "a task's live loan prevents mutation":
    reject:
        data = tensor[u8](4)
        task = async inspect_payload(view data) with local, thread
        data[0] = 9
        await task
    accept:
        data = tensor[u8](4)
        task = async inspect_payload(view data) with local, thread
        await task
        data[0] = 9

# Return/throw from a task owner -> cancel or finish its tasks, then release captures.
# Cancellation requested != native IO completed; do not release borrowed buffers yet.
# Process executor -> serialized/copied values; never transfer a host pointer as data.
# Cancellation and completion race -> one terminal result, one resource release.
```

```sev
# GPU/SIMD arithmetic remains visible to MLIR; see SIP-0039.
def gpu_multiply(left: tensor[f64], right: tensor[f64]) -> tensor[f64]:
    with gpu:
        return left * right

def simd_multiply(left: tensor[f64], right: tensor[f64]) -> tensor[f64]:
    with simd:
        return left * right

test "SIMD returns owned output and preserves its inputs":
    left = tensor[f64](tuple[f64,2](2.0, 3.0))
    right = tensor[f64](tuple[f64,2](4.0, 5.0))
    output = simd_multiply(left, right)
    assert(output[0] == 8.0 and output[1] == 15.0)
    assert(left[0] == 2.0 and right[1] == 5.0)
    squared = simd_multiply(left, left)
    assert(squared[0] == 4.0 and squared[1] == 9.0)

# Provider test cases:
# [2.0, 3.0] * [4.0, 5.0] -> owned [8.0, 15.0], inputs unchanged.
# Same tensor supplied twice -> legal read-only inputs.
# Shape mismatch -> failure before output/device allocation.
# Empty input -> empty output, no kernel launch.
# GPU return -> kernel complete AND result copied to owned host storage.
# Driver unavailable -> explicit failure, no silent CPU fallback.
# Provider fails after allocations -> attempt every acquired-resource release.
# Device pointer -> host dereference -> reject.
# A future asynchronous API must return an owning completion token.
# Hardware tests require a matching device; IR/layout tests do not.
```

### Acceptance cases for every provider

```text
build/semantic checks:
  portable library call -> resolved declaration -> checked MIR -> MLIR
  foreign call outside unsafe -> reject
  moved/closed owner used again -> reject
  view outlives resource -> reject
  wrong byte/element dimension -> reject
  target has no required provider -> diagnostic

deterministic provider fixtures:
  success -> expected value and exact transferred extent
  partial success -> resume at first untransferred byte
  empty request -> success without external access
  zero progress / interrupted / would-block -> documented distinct outcomes
  unknown native failure -> original operation and code retained
  failure after each acquisition -> no resource lost or released twice
  cancellation at each suspension -> completion acknowledged before release
  quota/deadline exhausted -> bounded failure and cleanup

integration checks:
  unique temporary directory -> cleanup even when assertions fail
  child writes more than pipe capacity to both outputs -> no deadlock
  loopback port 0 -> actual bound address -> bounded local exchange
  unrelated parent descriptor -> absent from child
  borrowed standard stream wrapper drops -> stream remains usable
  explicit close error -> observable; destructor cleanup error -> no double throw

# Draft examples specify acceptance criteria; Markdown fences are not a test runner.
# When implemented, keep executable test blocks beside their .sev/.py implementation.
# No build, execution, hardware support, or passing regression is claimed by this SIP.
```
