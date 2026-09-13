# IO

`read`, `read_all`, and `write_all` transfer bytes through ordinary descriptors,
including files and pipes. EOF is an empty successful read. Writes continue
after a partial transfer; errors and nonprogress propagate to the caller.

XXI generates the borrowed/inout byte-view conversion from the declarations in
`src/lib.sev`. Providers in `extern/posix/descriptor.c` operate on pointers and
lengths; they never receive a Severian list. The descriptor provider retries
interrupted system calls. The source compiler's typed buffers use the same
provider through `src/buffer.sev` and shared XXI bounds checks.

`read_all` accepts byte and call budgets, and `write_all` accepts a call budget.
For a hard deadline around a provider that can hang within one call, use an
isolated `xxi.worker` invocation. File paths, permissions and metadata belong to
`system.file`; text encoding belongs to `system.file.text`.

The package also owns standard-stream printing. The broader Reader/Writer and
MemoryStream interfaces in the examples remain proposed APIs.
