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
Unknown rank requires a distinct future type representation and must not be
encoded as a list of dynamic dimensions.

After structural validation, the native provider walks the object graph and
creates a live `mlir::ModuleOp`. MLIR's native verifier is the second validation
layer. Text output is allowed only for `--emit=mlir`, diagnostics, golden tests,
and bootstrap tooling.

Current weakness: direct native construction is proven for scalar functions and
ranked tensor signatures, but the complete `.sev` graph walker, in-process pass
pipeline, LLVM translation, object emission, and embedded LLD link are not yet
wired into `sev build`.
