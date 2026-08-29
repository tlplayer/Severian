# Operators

Operators resolve through universal operation identities and visible operator
signatures. Their spelling does not encode operand type.

| Section | Spellings | API record |
| --- | --- | --- |
| [Unary](unary.md) | `+`, `-`, `not` | `operator.unary` |
| [Arithmetic](arithmetic.md) | `+ - * / % **` | `operator.binary` |
| [Bitwise](bitwise.md) | `| & ^` | `operator.binary` |
| [Comparison](comparison.md) | `== != < <= > >=` | `operator.binary` |
| [Logical and membership](logical-and-membership.md) | `and or in` | `operator.binary` |

The authoritative identities are `UnaryOperator` and `BinaryOperator` in
`compiler/universal/src/operator.rs`. Machine records are in
[`../language/operators/core.toml`](../language/operators/core.toml).

Current weakness: the identity catalogue is exhaustive, but the public
type/representation/target conformance matrix is not.
