# MLIR builder

API ID: `compiler.mlir.builder`

Severian lowering constructs MLIR as typed operations, values, blocks, regions,
and modules. It does not concatenate a module string.

```sev
builder = MlirOpBuilder(0)
left = builder_value(builder, 0, integer_type(32))
right = builder_value(builder, 1, integer_type(32))
function = builder_create_function(
    builder,
    "add",
    public_visibility(),
    [left, right],
    [integer_type(32)],
)
builder_set_insertion_point(builder, function.entry)
result = builder_new_value(builder, integer_type(32))
builder_binary(builder, result, operation_name("arith", "addi"), left, right)
builder_return(builder, [result])
validate(builder.program)
```

The operation name is open data `(dialect, mnemonic)`. Types are structural
data. For example, BF16 and rank three belong to an `MlirType`; neither appears
in the operation identity or a source symbol suffix.

```sev
shape: list[int] = [known_dimension(1), dynamic_dimension(), known_dimension(128)]
input = tensor_type(bf16_scalar_type(), shape)
```

Here rank is statically three. Only the middle dimension's extent is dynamic.
Unknown rank uses `unranked_tensor_type`; it is not encoded as a list of dynamic
dimensions. Consequently `tensor<xbf16>` (rank zero), `tensor<?xbf16>` (rank
one with a dynamic extent), and `tensor<*xbf16>` (unknown rank) are three
different contracts.

Attributes are values too. Integer and floating values remain typed numeric
data; arrays recursively contain attributes; affine maps recursively contain
dimension, symbol, constant, and arithmetic expressions. Symbols and source
locations are structured fields rather than fragments spliced into MLIR text.

The native provider enforces a single-context ownership boundary. Regions and
detached blocks are consumed when attached, operations are consumed when
appended, and handles from another builder are rejected before entering MLIR.
MLIR diagnostics are copied into provider-owned storage before being returned.

After structural validation, the native provider walks the object graph and
creates a live `mlir::ModuleOp`. MLIR's native verifier is the second validation
layer. Text output is allowed only for `--emit=mlir`, diagnostics, golden tests,
and bootstrap tooling.

Direct native construction now covers the type, attribute, block, nested-region,
location, and diagnostic vocabulary. A parser-call counter proves these native
tests do not call `mlirModuleCreateParse`. The remaining weakness is integration:
the complete `.sev` graph walker, replacement of legacy print consumers, LLVM
translation, object emission, and embedded LLD are not yet wired into `sev
build`.
