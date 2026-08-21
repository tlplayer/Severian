# IO

`io` defines the portable stream layer used by files, sockets, process pipes,
codecs, archives, and the prelude.

Its core contracts are:

```sev
trait Reader:
    def read(buffer: borrow mut bytes) -> Unit[Data] | StreamError

trait Writer:
    def write(data: borrow bytes) -> Unit[Data] | StreamError
    def flush() -> unit | StreamError

trait Seeker:
    def seek(offset: Unit[Data], origin: SeekOrigin) -> Unit[Data] | StreamError
```

`read` and `write` may complete partially. `read_exact`, `write_all`, `copy`, and
`read_all` are library algorithms layered on those primitives. EOF is distinct
from an operating-system error. Byte counts and stream offsets retain the
`Unit[Data]` dimension, so calls use values such as `0B`, `4KiB`, and `1MiB`
instead of dimensionless integers.

## Standard streams and printing

The process provider supplies `stdin`, `stdout`, and `stderr` as ordinary IO
handles. Formatting and printing are library operations:

```sev
io.write_all(io.stdout(), "hello".bytes())?
io.print("hello")?
io.println("hello")?
```

`io.print` formats values and writes to stdout. It does not exit, implicitly
flush every write, or bypass stream errors. The core prelude re-exports it:

```sev
import print from io
```

The compiler must see the resulting expression as an ordinary function call in
every pipeline, including mixed CompileType programs.
