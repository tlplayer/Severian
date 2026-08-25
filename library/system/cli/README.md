# cli

`cli` is the bootstrap command-line boundary used by Severian package drivers.
The current executable contract is intentionally small:

- `command(...)` records a command name, description, version, and positional
  argument names.
- `argument(...)` declares a positional name.
- `parse_process(...)` obtains the real immutable vector from
  `process.arguments()` and removes the executable name.
- `values(...)` returns the application argument values.

This surface is enough for the self-hosted driver to receive its command-line
inputs without a compiler-defined `main(args)` convention.

Options, flags, subcommands, generated help, completion, and schema linting are
the next library layer. They require stable runtime storage for lists of
aggregate specification values. Until that ABI exists, those features should
not be simulated with compiler tables or pointer casts.
