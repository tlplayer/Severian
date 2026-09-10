# Generics

Every `.sev` file in this directory is a standalone documentation example and validation source root.

## General generic examples

| File | Subject |
| --- | --- |
| `01-generic-numeric.sev` | Trait-bounded numeric function |
| `02-generic-map-sum.sev` | Multiple generic parameters and composed bounds |
| `03-custom-failure.sev` | Function constraints and custom failure |
| `04-boxed-generic.sev` | Generic class |
| `05-stack-generic.sev` | Generic collection |

## Compiler-term generic symbols

Each example performs a compiler operation. The generic term is transformed, executed, matched, folded, bound, or preserved; it is not reduced to a display string.

| Symbol | Category | Example operation | File |
| --- | --- | --- | --- |
| `T` | Type | Substitute a type parameter | `06-type-generic.sev` |
| `V` | Value | Fold compile-time values | `07-value-generic.sev` |
| `E` | Error | Wrap a concrete error with pass and source context | `08-error-generic.sev` |
| `Ex` | Expression | Constant-fold an expression | `09-expression-generic.sev` |
| `M` | Macro | Expand compiler input | `10-macro-generic.sev` |
| `O` | Operation (statement) | Execute a statement sequence | `11-statement-generic.sev` |
| `X` | Any compiler term (declaration) | Bind declarations to symbols | `12-declaration-generic.sev` |
| `X` | Any compiler term (pattern) | Select a matching pattern | `13-pattern-generic.sev` |
| `Lx` | Lexeme | Locate matched source text | `26-lexeme-generic.sev` |
| `To` | Token | Validate origin while preserving token type | `27-token-generic.sev` |
| `L` | Literal | Fold literal values | `14-literal-generic.sev` |
| `O` | Operation | Execute operation semantics | `15-operation-generic.sev` |
| `O` | Operation (instruction) | Remove no-op instructions | `16-instruction-generic.sev` |
| `B` | Block | Add a missing terminator | `17-block-generic.sev` |
| `V` | Value (argument) | Prepend a receiver without erasing argument type | `18-argument-generic.sev` |
| `R` | Result | Merge pass results | `19-result-generic.sev` |
| `F` | Callable | Invoke a callable generically | `20-callable-generic.sev` |
| `W` | With-clause operation (constraint) | Evaluate semantic constraints | `21-constraint-generic.sev` |
| `T` | Type (kind metadata) | Select a compatible wider kind | `22-kind-generic.sev` |
| `Y` | Symbol | Resolve symbol identity to operation semantics | `23-symbol-generic.sev` |
| `X` | Any compiler term (node) | Traverse nodes across IR levels | `24-node-generic.sev` |
| `X` | Any compiler term | Run a category-independent pass | `25-compiler-term-generic.sev` |

The [canonical vocabulary](../../../../sev_compiler/universal/README.md) also
reserves `S` for Shape, `N` for Number of elements, `C` for Container, and `G`
for Grammar. `W` describes With-clause operations, including constraints.

`Lx` lexemes preserve matched source text/spans; `To` tokens classify that source
for the parser. Both retain their specific contracts in these examples. `L` remains
Literal and `Y` remains Symbol. A lexical rule produces lexemes and tokens.

```text
characters → Lx → To → parser → Y / L / Ex / O / X → G → CFG
```

`E` is Error; `Ex` is Expression. `O` covers statements and instructions; `V`
covers arguments; `T` covers type-kind metadata. `X` covers declarations,
patterns and nodes without erasing their specific API constraints. These are
naming conventions, not a requirement to combine their concrete representations.

## Generative callables

An arrow-prefixed declaration is a compile-time callable. It uses the same generic parameters, parameters, return annotation, body syntax, and call syntax as an ordinary function:

```sev
-> expand_macro[M: MacroTerm](macro_term: M, input_tokens: list[i64]) -> list[i64]:
    return macro_term.expand(input_tokens)

expanded := expand_macro[RepeatMacro](RepeatMacro(3), [4, 2])
```

The arrow belongs before the callable name. Postfix spellings such as `expand_macro->` are not part of the syntax. The declaration is resolved and specialized through the existing semantic, HIR, MIR, and MLIR pipeline.

## Compiler type representation

- `PrimitiveType`
- `GenericType`
- `NominalType`
- `FunctionType`
- `UnionType`
- `TupleType`
- `ReferenceType`
- `CompileType`

```text
enum Type {
    Primitive(PrimitiveType),

    Named(TypeId),

    Applied {
        constructor: TypeId,
        args: Vec<TypeArg>,
    },

    Function {
        params: Vec<Type>,
        result: Box<Type>,
    },

    Union(Vec<Type>),

    Reference {
        kind: BorrowKind,
        inner: Box<Type>,
    },

    Compile(CompileType),
}
```
