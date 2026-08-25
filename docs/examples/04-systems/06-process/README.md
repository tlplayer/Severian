# Processes

The process library owns executable arguments, child creation, pipes, exit
status, waiting, and termination. It is layered over a selected platform
provider rather than compiler or backend special cases.

`process.arguments()` is the sole owner-facing API for the immutable argument
vector. Executable entry functions remain ordinary `main()` declarations and
do not acquire a compiler-defined argument signature.

Declarative applications use `cli.parse_process(command)`. The bootstrap CLI
library obtains the raw vector explicitly through `process.arguments()`,
removes the executable name at the application boundary, and exposes the
remaining positional values. Options, subcommands, generated help, and
structured parse errors are layered on this boundary once aggregate
specification lists have a stable runtime representation.

Spawning returns an owned `Child`. Waiting consumes or transitions that handle
to a completed state; dropping a live child does not silently report success.
Shell parsing is opt-in. An argument-vector API is the default so values are not
reinterpreted by a shell.
