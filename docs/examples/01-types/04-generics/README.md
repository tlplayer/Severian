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

| Symbol | Category | Example |
| --- | --- | --- |
| `T` | Type | `06-type-generic.sev` |
| `V` | Value | `07-value-generic.sev` |
| `E` | Error | `08-error-generic.sev` |
| `Ex` | Expression | `09-expression-generic.sev` |
| `M` | Macro | `10-macro-generic.sev` |
| `S` | Statement | `11-statement-generic.sev` |
| `D` | Declaration | `12-declaration-generic.sev` |
| `P` | Pattern | `13-pattern-generic.sev` |
| `L` | Literal | `14-literal-generic.sev` |
| `O` | Operation | `15-operation-generic.sev` |
| `I` | Instruction | `16-instruction-generic.sev` |
| `B` | Block | `17-block-generic.sev` |
| `A` | Argument | `18-argument-generic.sev` |
| `R` | Result | `19-result-generic.sev` |
| `F` | Callable | `20-callable-generic.sev` |
| `C` | Constraint | `21-constraint-generic.sev` |
| `K` | Kind | `22-kind-generic.sev` |
| `Y` | Symbol | `23-symbol-generic.sev` |
| `N` | Node | `24-node-generic.sev` |
| `X` | Any compiler term | `25-compiler-term-generic.sev` |

`E` is Error and `Ex` is Expression. `O` is a semantic operation; `Y` is the symbol or resolved identity that names it. `X` is the umbrella compiler-term parameter, conceptually `X: T | V | E | Ex | M | S | D | P | L | O | I | B | A | R | F | C | K | Y | N`.

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
