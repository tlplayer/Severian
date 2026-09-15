# Generic consumer and incremental compilation

The producer defines `foo[T](value: T)` and prints the value.
The consumer calls it three times each with string, char, int, float, and bool,
and checks all 15 printed lines against a fixture.
The generic producer is consumed as source and specialized in the consumer;
this example does not claim a source-free generic binary interface.

After building the source compiler, run from the repository root:

```sh
python3 docs/examples/10-building/03-generic-consumer/e2e.py \
  --compiler package.pkg/bin/sev_compiler
```

The runner builds from a fresh invocation directory, checks
`package.pkg/bin/generic-consumer`, and verifies that the producer and consumer
source trees receive no generated package directories. It checks the unused-file
diagnostic for `consumer/src/unused.sev`, warm unit reuse, a `+1` behavior change,
unchanged native-provider reuse, and downstream IR reuse after a same-line
comment edit. JSON profiles and command evidence are retained in its work area.
