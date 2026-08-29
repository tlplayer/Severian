# `args`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.args`.

## Representation

`args` uses `PrimitiveRepresentation.Arguments`. It is runtime-provided process
entry data, has no literal kind, and is not a general-purpose collection type.
The driver materializes the public `process.arguments() -> list[string]` view.

## Source semantics

Programs obtain arguments through the process API rather than constructing an
`args` literal. The primitive records the entry-boundary representation so the
driver does not encode process arguments as an ad hoc function-name special case.

```sev
import process

def argument_count() -> usize:
    return size(process.arguments())
```

## ABI and lowering

`args` cannot cross an arbitrary external function boundary. The executable
wrapper receives platform `argc/argv`, installs the runtime view, and then calls
the Severian entry point.

## Tensor

`args` is not a tensor element. Tokenization produces explicitly typed integer
tensors after parsing argument strings.

## Current weakness

The exact public relation between primitive `args`, the driver entry record,
and `list[string]` is not yet expressed as a versioned API object.
