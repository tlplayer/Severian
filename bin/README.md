# Compiled examples

Run `tools/check_docs_examples.sh --native` to populate `bin/examples` with
verified native executables. Each output mirrors its source path below
`docs/examples`:

```text
docs/examples/00-getting-started/01-hello.sev
bin/examples/00-getting-started/01-hello
```

Only examples containing `main()` can produce standalone executables. The
harness runs each temporary native build and compares it byte-for-byte with the
adjacent `.stdout` fixture. A binary is published here only after that check
passes. Unsupported lowering, crashes, timeouts, stderr, output differences, and
missing expectations are failures and do not produce a binary.
